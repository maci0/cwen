/* libFuzzer harness for the untrusted-input parsers in run.c.
   Build: make fuzz    Smoke run: make fuzz-run

   Each input is parsed five ways: once as a GGUF container
   (load_gguf header/kv/tensor-table walk), once as a CWENR v2/v3/v4
   sidecar (load_cwenr directory + offset binding), once as a server
   request-frame stream (server_read_frame header validation, payload
   drain, token range check), once as a DFlash drafter .spec
   container (load_dflash entry walk, offset/nbytes binding, geometry
   checks), and once as an NGC2 n-gram cache (ng_load into a fresh map,
   then ng_save + reload with the entry sets diffed, since the cache
   crosses a persistence boundary), all surfaces that take bytes from
   outside the process.
   exit()-rejections are recoverable; memory bugs surface via ASan/UBSan,
   broken post-load invariants abort here, any divergence between the
   frame parser and the independent decoder below aborts too (protocol
   desync becomes a liveness failure), and an ngram save/load round trip
   that loses entries or hit counts aborts as well.

   Run with -close_fd_mask=3 to silence parser chatter during long runs. */
#define _POSIX_C_SOURCE 200809L /* mkstemp under -std=c11 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int cw_fuzz_once(const char *gguf, const char *cwenr);
void cw_fuzz_reset(void);
void cw_fuzz_limits(unsigned *vocab, unsigned *max_seq);
int cw_fuzz_server(const uint8_t *data, size_t len,
                   uint32_t *dec, uint64_t *pos, int cap);
int cw_fuzz_dflash(const char *path);
int cw_fuzz_ngcache(const char *path, const char *path2);

/* Trace cap: an input needs >=8 bytes per frame, so 1 MiB of mutations can
   never overflow this; cw_fuzz_server returns -1 if it ever would. */
#define TRACE_CAP (1 << 16)
static uint32_t got_dec[TRACE_CAP], ref_dec[TRACE_CAP];
static uint64_t got_pos[TRACE_CAP], ref_pos[TRACE_CAP];

static char g_tmpdir[192], g_gguf[256], g_cwenr[256], g_spec[256];
static char g_ngc[256], g_ngc2[256];

/* Unique 0600 files via mkstemp: fixed cwen_fuzz_<pid> names let any local
   user pre-place a symlink and clobber whatever it points at when the harness
   opens the path for writing (CWE-377/59).

   Default is the repo's .scratch, not /tmp: a fuzz run writes a candidate
   GGUF per iteration, and /tmp is tmpfs here, so the corpus would be paid for
   in RAM and lost on reboot. TMPDIR still overrides for a run elsewhere. */
__attribute__((constructor)) static void fuzz_paths(void) {
  const char *tmp = getenv("TMPDIR");
  if (!tmp || !tmp[0]) tmp = ".scratch";
  int n = snprintf(g_tmpdir, sizeof g_tmpdir, "%s", tmp);
  if (n < 0 || n >= (int)sizeof g_tmpdir) abort();
}

/* Create leaf (trailing XXXXXX required by mkstemp) under g_tmpdir, fill it
   with data, and store the materialized path in out. */
static void write_unique(const char *leaf, char *out, size_t cap,
                         const uint8_t *data, size_t size) {
  int n = snprintf(out, cap, "%s/%s", g_tmpdir, leaf);
  if (n < 0 || n >= (int)cap) abort();
  int fd = mkstemp(out);
  if (fd < 0) abort();
  size_t off = 0;
  while (off < size) {
    ssize_t w = write(fd, data + off, size - off);
    if (w < 0) { close(fd); abort(); }
    off += (size_t)w;
  }
  close(fd);
}

static uint32_t rd_le32(const uint8_t *p) {
  return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
         (uint32_t)p[3] << 24;
}

/* Independent reference decoder for the request protocol documented in
   run.c: <u32 n_prompt><u32 n_gen><n_prompt * i32>, little-endian. Reject =
   consume the declared payload and reply 0; short header or payload = close;
   0xffffffff in a header word is the reserved EOF sentinel and also closes.
   Deliberately written from the spec text, not by copying the parser: any
   drift between the two is a finding. */
static int ref_decode(const uint8_t *d, size_t len) {
  unsigned vocab, max_seq;
  cw_fuzz_limits(&vocab, &max_seq);
  size_t o = 0, n = 0;
  while (o + 8 <= len) {
    uint32_t np = rd_le32(d + o), ng = rd_le32(d + o + 4);
    o += 8;
    if (np == 0xffffffffu || ng == 0xffffffffu) break; /* EOF sentinel */
    int hdr_ok = np > 0 && np <= max_seq && ng > 0 && ng <= max_seq;
    if (!hdr_ok) { /* rejected header: drain declared payload */
      uint64_t pay = (uint64_t)np * 4u;
      if (pay > len - o) { /* truncated drain consumes to EOF */
        ref_dec[n] = 0; ref_pos[n] = len; n++;
        break;
      }
      ref_dec[n] = 0; ref_pos[n] = o + pay; n++;
      o += (size_t)pay;
    } else if ((uint64_t)np * 4u > len - o) {
      break; /* truncated payload: no trace entry */
    } else {
      int bad = 0;
      for (uint32_t i = 0; i < np && !bad; i++) {
        uint32_t t = rd_le32(d + o + (size_t)i * 4u);
        bad = t >= vocab || (int32_t)t < 0;
      }
      ref_dec[n] = (uint32_t)!bad;
      ref_pos[n] = o + (size_t)np * 4u;
      n++;
      o += (size_t)np * 4u;
    }
  }
  return (int)n;
}

static void fuzz_frames(const uint8_t *data, size_t size) {
  int ngot = cw_fuzz_server(data, size, got_dec, got_pos, TRACE_CAP);
  if (ngot < 0) return; /* cap signalled: input larger than any real corpus */
  int nref = ref_decode(data, size);
  if (ngot != nref) abort(); /* framing decisions diverge from the spec */
  for (int i = 0; i < ngot; i++)
    if (got_dec[i] != ref_dec[i] || got_pos[i] != ref_pos[i]) abort();
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  fuzz_frames(data, size);
  write_unique("cwen_fuzz_gguf_XXXXXX", g_gguf, sizeof g_gguf, data, size);
  write_unique("cwen_fuzz_cwenr_XXXXXX", g_cwenr, sizeof g_cwenr, data, size);
  write_unique("cwen_fuzz_spec_XXXXXX", g_spec, sizeof g_spec, data, size);
  write_unique("cwen_fuzz_ngc_XXXXXX", g_ngc, sizeof g_ngc, data, size);
  write_unique("cwen_fuzz_ngc2_XXXXXX", g_ngc2, sizeof g_ngc2, NULL, 0);
  int r = cw_fuzz_once(g_gguf, g_cwenr);
  int rd = cw_fuzz_dflash(g_spec); /* independent of the container verdict */
  int rn = cw_fuzz_ngcache(g_ngc, g_ngc2);
  unlink(g_gguf);
  unlink(g_cwenr);
  unlink(g_spec);
  unlink(g_ngc);
  unlink(g_ngc2);
  cw_fuzz_reset();
  if (r == 2 || rd == 2 || rn == 2)
    abort(); /* bound tensor escaped its backing region / cache round-trip drifted */
  return 0;
}
