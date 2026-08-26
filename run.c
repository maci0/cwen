/* cwen: mmap Q4 GGUF Qwen3.8-27B, CPU decode. Zen3+ AVX2. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <assert.h>
#include <time.h>
#include <errno.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX2__) || defined(__F16C__)
#include <immintrin.h>
#endif
#include "cwen_tune.h"
/* Bench and fuzz builds compile this whole TU but call a fraction of it;
   they carry their own -Werror, so silence unused statics there only.
   Production builds keep full unused-function discipline. */
#if defined(CWEN_BENCH_Q4_GEMV) || defined(CWEN_BENCH_SPEC) || defined(CWEN_FUZZ_LOADER)
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#ifdef CWEN_FUZZ_LOADER
/* libFuzzer harness support (tools/fuzz_loader.c): the parsers reject
   malformed files via exit(); reroute those to a recoverable jump so a single
   process can keep fuzzing. Not defined for production builds. */
#include <setjmp.h>
static jmp_buf cw_fuzz_jmp;
static int cw_fuzz_armed;
static void cw_fuzz_exit(int rc){ (void)rc; if(cw_fuzz_armed) longjmp(cw_fuzz_jmp,1); abort(); }
#define exit(rc) cw_fuzz_exit(rc)
#endif
/* Idea A/B flags (tools/idea_bench.py): default 0 unless noted. */
#ifndef CWEN_IDEA_FAST_SILU
#define CWEN_IDEA_FAST_SILU 0
#endif
#ifndef CWEN_IDEA_MADVISE
#define CWEN_IDEA_MADVISE 0
#endif
#ifndef CWEN_IDEA_CCD
#define CWEN_IDEA_CCD 1 /* pin OMP first 16 cores */
#endif
#ifndef CWEN_IDEA_PF_T0
#define CWEN_IDEA_PF_T0 0
#endif
#ifndef CWEN_IDEA_NO_PF
#define CWEN_IDEA_NO_PF 0
#endif
#ifndef CWEN_IDEA_PIPE_PF
#define CWEN_IDEA_PIPE_PF 0
#endif
#ifndef CWEN_IDEA_COLLAPSE
#define CWEN_IDEA_COLLAPSE 0
#endif
#ifndef CWEN_IDEA_SERIAL_MLP
#define CWEN_IDEA_SERIAL_MLP 0
#endif
#ifndef CWEN_IDEA_GDN_OMP
#define CWEN_IDEA_GDN_OMP 1 /* parallel GDN over LVH heads (I06 keep) */
#endif
#ifndef CWEN_IDEA_PAIR_GEMV
#define CWEN_IDEA_PAIR_GEMV 1 /* one OMP team: qkv+gate_z (measured) */
#endif
#ifndef CWEN_IDEA_PF_ONE
#define CWEN_IDEA_PF_ONE 0 /* fused mlp: prefetch only gate stream */
#endif
#ifndef CWEN_IDEA_MADV_SEQ
#define CWEN_IDEA_MADV_SEQ 0 /* MADV_SEQUENTIAL on current layer mats */
#endif
#ifndef CWEN_IDEA_NO_SILU_OMP
#define CWEN_IDEA_NO_SILU_OMP 1 /* serial silu: drop GOMP fork (elementwise cheap) */
#endif
#ifndef CWEN_IDEA_CONV_OMP
#define CWEN_IDEA_CONV_OMP 0 /* parallel conv1d over channels */
#endif
/* ---- dims (Qwen3.8-27B / GGUF qwen35 hybrid; docs/DESIGN.md "Model geometry") ---- */
enum {
  H = 5120, I = 17408, V = 248320, L = 64, MAX_SEQ = 32768,
  NH = 24, NKV = 4, HD = 256, NROT = 64,
  LKH = 16, LVH = 48, LSD = 128, CONV_K = 4, QKV_DIM = 10240,
  FULL_INT = 4
};
static const float RMS_EPS = 1e-6f;
static const float ROPE_THETA = 1e7f;
static const int ROPE_SEC[3] = {11, 11, 10};

/* ---- ggml types ---- */
/* ggml type ids this build can size or compute (unknown quants: row_bytes 0,
   validators reject). F16/Q4_K never appear in the pinned model or sidecar. */
enum { T_F32=0, T_Q4_0=2, T_Q4_1=3, T_Q8_0=8, T_Q5_K=13, T_Q6_K=14,
       T_Q4_0R=100,  /* packed 20B {qs,d} from CWENR v2 */
       T_Q4_0RS=101, /* split: qs[16*nb*ne1] + f16 sc[nb*ne1] (v3) */
       T_Q4_0RSI=102, /* interleaved dual: (qsA|qsB)*nb, (scA|scB)*nb per row (v4) */
       T_Q8S=103,     /* drafter split Q8: int8 stream + f16 scale channel */
       T_Q8SI=104 };  /* drafter paired split Q8: rows alternate A,B */
#define QK4 32
#define QK_K 256
/* CWENR directory flags (v4) */
enum { CWENR_F_SOLO=0, CWENR_F_IL_A=1, CWENR_F_IL_B=2 };
/* reserved[24..31] stamp: {u32 tag "CWEN", u32 source GGUF size in 4KiB pages} */
#define CWENR_STAMP_TAG 0x4e455743u /* "CWEN" */
typedef struct { uint16_t d; int8_t qs[32]; } block_q8_0;
_Static_assert(sizeof(block_q8_0)==34, "q8_0 size");
typedef struct { uint16_t d; uint8_t qs[16]; } block_q4_0;
typedef struct { uint16_t d, m; uint8_t qs[16]; } block_q4_1;
typedef struct {
  uint16_t d, dmin; uint8_t scales[12]; uint8_t qh[QK_K/8]; uint8_t qs[QK_K/2];
} block_q5_K;
typedef struct {
  uint8_t ql[QK_K/2]; uint8_t qh[QK_K/4]; int8_t scales[QK_K/16]; uint16_t d;
} block_q6_K;

/* Q4_0R dense offline (CWENR v2): 20B {qs[16], f32 d}. v2 load splits to
   T_Q4_0RS; v3 binds qs+f16 scales; v4 interleaves dual-mat pairs. */
typedef struct { uint8_t qs[16]; float d; } __attribute__((packed)) block_q4_0r;
_Static_assert(sizeof(block_q4_0r)==20, "q4_0r size");
static inline size_t q4r_nb(int ne0){ return (size_t)(ne0/QK4); }
static inline size_t q4r_row_bytes(int ne0){ return q4r_nb(ne0)*sizeof(block_q4_0r); }
static inline size_t q4rs_row_qs(int ne0){ return q4r_nb(ne0)*16u; }
static inline size_t q4rsi_row_qs(int ne0){ return q4r_nb(ne0)*32u; } /* qsA|qsB */
/* GGUF Q4_0 blob size (for MADV_DONTNEED after CWENR rebind). */
static inline size_t q4_0_nbytes(int ne0, int ne1){
  return (size_t)(ne0/QK4)*sizeof(block_q4_0)*(size_t)ne1;
}

/* ---- globals: model ---- */
static void *Gmap; static size_t Gmap_len;
static void *Rmap; static size_t Rmap_len; /* offline .cwenr mmap (Q4_0R blobs) */
static void *Q4qs_arena; static void *Q4sc_arena; /* qs uint8, scales f16 (half BW) */
static size_t Q4qs_n, Q4sc_n; /* live byte counts (fuzzer invariant checks) */
/* pair_side: 0=A/solo, 1=B (second half of interleaved block) */
typedef struct { const void *data; const void *scales; int type; int ne0, ne1; int pair_side; } Tensor;
static Tensor Tens[1024]; static char Tnames[1024][96]; static int Ntens;

typedef struct {
  Tensor attn_norm, post_norm;
  /* linear (GDN) */
  Tensor qkv, gate_z, ssm_a, ssm_dt, ssm_alpha, ssm_beta, ssm_conv, ssm_norm, ssm_out;
  /* full attn */
  Tensor wq, wk, wv, wo, q_norm, k_norm;
  /* mlp */
  Tensor ffn_gate, ffn_up, ffn_down;
  int is_linear;
} LayerW;
static LayerW W[L];
static Tensor tok_embd, output_norm, output;

/* ---- globals: state (64B-aligned for cache-line / AVX loads) ---- */
static float x[H] __attribute__((aligned(64)));
static float xb[H] __attribute__((aligned(64)));
static float xb2[H] __attribute__((aligned(64)));
static float hb[I] __attribute__((aligned(64)));
static float hb2[I] __attribute__((aligned(64)));
static float qkvb[QKV_DIM] __attribute__((aligned(64)));
static float zb[LVH*LSD] __attribute__((aligned(64)));
static float ab[LVH], bb[LVH];
static float qh[NH*HD] __attribute__((aligned(64)));
static float kh[NKV*HD] __attribute__((aligned(64)));
static float vh[NKV*HD] __attribute__((aligned(64)));
static float gateq[NH*HD] __attribute__((aligned(64)));
static float qfull_g[NH*HD*2] __attribute__((aligned(64)));
static float yatt_g[NH*HD] __attribute__((aligned(64)));
static float *logits; /* V floats, heap */
/* GDN state: only linear layers need it; index by layer */
static float *Srec;   /* L * LVH * LSD * LSD */
static float *Cstate; /* L * QKV_DIM * (CONV_K-1) */
/* KV cache for full layers only; use full L slots, unused for linear */
static float *Kcache, *Vcache; /* L * MAX_SEQ * NKV * HD */
static int pos_n; /* current sequence length after embed */
/* Runtime context cap (CWEN_CTX). Compile-time MAX_SEQ is the hard ceiling;
   KV/drafter caches size to this, every sequence bound checks against it. */
static int g_ctx = 4096;
static int g_yarn_on;             /* CWEN_ROPE_YARN present */
static double Yarn_orig_max, Yarn_factor, Yarn_beta_fast=32.0, Yarn_beta_slow=1.0;
/* cache row stride: allocation and indexing both use the runtime cap */
#define CTX_STRIDE g_ctx

/* ---- block speculation (DFlash-style verify, n-gram drafter) ----
   A drafter proposes a block after the pending token; one batched forward
   scores the whole block (every weight matrix streams once, not once per
   position); a greedy walk keeps the longest verified prefix plus the
   target's own next pick (the bonus). Greedy output is bit-identical to
   serial decode. The trained DFlash2 drafter plugs in behind ngram_draft's
   contract (propose <= max_draft ids from history) without touching this
   machinery; see docs/DESIGN.md PR17. */
#define SPEC_BMAX 16
#define PF_CHUNK 8   /* prefill rows per batched pass (prefill_forward) */
static int spec_enabled;
static int Scfg_n_key = 16;     /* lookup key length (recent-match scan) */
static int Scfg_max_draft = 8;  /* block = 1 pending + up to N drafts */
static int Scfg_min_draft = 2;  /* below this a block pass is not worth it */
static int Scfg_cooldown = 8;   /* plain steps after repeated full rejects */
static int Scfg_debug;          /* per-cycle speculation trace (CWEN_SPEC_DEBUG=1) */
/* block scratch: per-position activations (heap, only when enabled) */
static float *BXres, *Bxbn, *Bqkvb, *Bzb, *Bhb, *Bhb2, *Bxb2, *Blogits;
static float *Bqh, *Bgateq, *Bkh, *Bvh, *Byatt, *Bqfull;
static float *Sab, *Sbb;        /* per-position GDN decay/beta */
static float *SnapS, *SnapC;    /* pre-block GDN snapshots (rollback) */
/* ---- n-gram counted map (llama.cpp ngram-cache style) ----
   Key of Scfg_n_key confirmed tokens -> {most recent continuation, hit
   count}. Updated from history as it grows; drafts by chained lookup (tail
   -> tok -> shifted key -> ...), falling back to the scan above when the map
   has nothing to say. A full table evicts its least-hit entry so long
   sessions keep learning. CWEN_NGRAM_CACHE=path persists entries across
   runs; the file header carries n_key+vocab so maps never merge across
   configs. */
typedef struct {
  int *keys;                  /* cap*NKEY inline keys */
  int *tok;                   /* cap */
  int *cnt;                   /* cap */
  unsigned char *used;
  size_t cap, used_n;
} NGMap;
static NGMap NGM;
#define NG_CAP ((size_t)1<<18)


/* ---- DFlash2 drafter (trained proposal model, PR17) ----
   5-layer sliding-window transformer conditioned on target hidden states.
   Weights ship in a .spec container (tools/pack_dflash.py); embeddings and
   the candidate lm_head are borrowed from the target model. Enable with
   CWEN_DFLASH=<path>; implies CWEN_SPEC=1. Reference semantics:
   github.com/z-lab/dflash dflash/model_mlx.py (GroupedDynamicCausalConv,
   CandidateSelector greedy walk, bidirectional noise window). */
enum { DL_LAYERS = 5, DL_HD = 128, DL_NH = 32, DL_NKV = 8, DL_KV = 1024,
       DL_Q = 4096, DL_I = 17408, DL_WIN = 2048, DL_RANK = 256,
       DL_TOPK = 16, DL_BLOCK = 8, DL_TAPN = 5, DL_CTXIN = 5 * H };
static const int DL_TAPS[DL_TAPN] = { 5, 19, 33, 47, 61 };
#define DL_MASKTOK 248070
#define DL_ROPE_THETA 1e7f
typedef struct {
  Tensor q, k, v, o, gate, up, down;
  Tensor gu, kv;                /* paired split-Q8: gate+up, k+v */
  Tensor ln1, ln2, qn, kn;
  Tensor ac_base, mc_base;      /* [2*2*H] f32 */
  Tensor ac_proj, mc_proj;      /* [H -> 2*k*groups] */
} DraftW;
static DraftW DW[DL_LAYERS];
static Tensor Dw_fc, Dw_hnorm, Dw_norm, Dw_pred, Dw_succ, Dw_hproj;
static int dflash_on;

/* ---- MTP nextn layer (PR18) ---- */
typedef struct {
  Tensor attn_norm, post_norm;
  Tensor wq, wk, wv, wo;
  Tensor q_norm, k_norm;
  Tensor ffn_gate, ffn_up, ffn_down;
  Tensor eh_proj;
  Tensor enorm, hnorm;
  Tensor shared_head_norm;
} NextnW;
static NextnW NW;
static int mtp_on;    /* blk.64 present and fully bound */
static int mtp_use;   /* ...and selected as this run's drafter */
static float *NKc, *NVc;   /* nextn K/V cache [g_ctx][NKV*HD] */
static float NHprev[H] __attribute__((aligned(64)));  /* target final normed hidden at mtp_hpos */
/* nextn scratch, file scope like the target path's x/xb/qh/... : every buffer
   that reaches gemv as the activation vector must be 64B aligned, because the
   AVX-512 Q4 kernels load it with _mm512_load_ps. */
static float Nemb[H]        __attribute__((aligned(64)));
static float Ncat[2*H]      __attribute__((aligned(64)));
static float Nxw[H]         __attribute__((aligned(64)));
static float Nresid[H]      __attribute__((aligned(64)));
static float Nln[H]         __attribute__((aligned(64)));
static float Nop[H]         __attribute__((aligned(64)));
static float Nqfull[NH*HD*2] __attribute__((aligned(64)));
static float Nqh[NH*HD]     __attribute__((aligned(64)));
static float Ngq[NH*HD]     __attribute__((aligned(64)));
static float Nao[NH*HD]     __attribute__((aligned(64)));
static float Nkh[NKV*HD]    __attribute__((aligned(64)));
static float Nvh[NKV*HD]    __attribute__((aligned(64)));
static float Nffg[I]        __attribute__((aligned(64)));
static float Nffu[I]        __attribute__((aligned(64)));
static int mtp_hpos=-1;
static float *DTapSer, *DTapBlk;   /* tap captures: serial / per-block-row */
static float *Dctx;                /* committed context vectors [MAX_SEQ*H] */
static float *Dkc, *Dvc;           /* drafter KV caches [L][MAX_SEQ][KV] */
static float *Dhw, *Dln_t, *Din_t, *DdynA, *DdynM, *Dq_t, *Dkw_t, *Dvw_t,
             *Dao_t, *Doh_t;
static float *Dg_t, *Dpred_r, *Dsucc_rows;
static int *Dcand;
static float DRcos[MAX_SEQ][64], DRsin[MAX_SEQ][64];
static uint8_t DRdone[MAX_SEQ];
static float Datt[MAX_SEQ] __attribute__((aligned(64)));

/* ---- f16 ---- */
static inline float f16_to_f32(uint16_t h) {
  uint32_t s = (h>>15)&1, e = (h>>10)&0x1f, f = h&0x3ff, u;
  if (e==0) { if (!f) return s ? -0.f : 0.f; /* denorm */ float x=ldexpf((float)f, -24); return s?-x:x; }
  if (e==31) return f ? NAN : (s ? -INFINITY : INFINITY);
  u = (s<<31) | ((e-15+127)<<23) | (f<<13);
  float r; memcpy(&r,&u,4); return r;
}

/* ---- math helpers ---- */
#if defined(__AVX2__)
static inline float hsum256(__m256 v) {
  __m128 lo=_mm256_castps256_ps128(v), hi=_mm256_extractf128_ps(v,1);
  lo=_mm_add_ps(lo,hi); lo=_mm_add_ps(lo,_mm_movehl_ps(lo,lo));
  __m128 sh=_mm_movehdup_ps(lo); return _mm_cvtss_f32(_mm_add_ss(lo,sh));
}
#endif
#if defined(CWEN_AVX512)
/* Cephes-style exp; clamp + degree-5 on reduced r. ~1 ulp, not FAST_SILU. */
static inline __m512 exp512(__m512 x) {
  x=_mm512_min_ps(x,_mm512_set1_ps(88.3762626647949f));
  x=_mm512_max_ps(x,_mm512_set1_ps(-88.3762626647949f));
  __m512 n=_mm512_roundscale_ps(_mm512_mul_ps(x,_mm512_set1_ps(1.4426950408889634f)),
                               _MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
  __m512 r=_mm512_fnmadd_ps(n,_mm512_set1_ps(0.693359375f),x);
  r=_mm512_fnmadd_ps(n,_mm512_set1_ps(-2.12194440e-4f),r);
  __m512 y=_mm512_set1_ps(1.9875691500e-4f);
  y=_mm512_fmadd_ps(y,r,_mm512_set1_ps(1.3981999507e-3f));
  y=_mm512_fmadd_ps(y,r,_mm512_set1_ps(8.3334519073e-3f));
  y=_mm512_fmadd_ps(y,r,_mm512_set1_ps(4.1665795894e-2f));
  y=_mm512_fmadd_ps(y,r,_mm512_set1_ps(1.6666665459e-1f));
  y=_mm512_fmadd_ps(y,r,_mm512_set1_ps(5.0000001201e-1f));
  y=_mm512_fmadd_ps(y,_mm512_mul_ps(r,r),r);
  y=_mm512_add_ps(y,_mm512_set1_ps(1.f));
  __m512i two=_mm512_slli_epi32(_mm512_add_epi32(_mm512_cvtps_epi32(n),_mm512_set1_epi32(127)),23);
  return _mm512_mul_ps(y,_mm512_castsi512_ps(two));
}
static inline __m512 silu512(__m512 x) {
  __m512 one=_mm512_set1_ps(1.f);
  __m512 sig=_mm512_div_ps(one,_mm512_add_ps(one,exp512(_mm512_sub_ps(_mm512_setzero_ps(),x))));
  return _mm512_mul_ps(x,sig);
}
/* Zero-pad last SIMD lane for linear reduce; mask-store so extra lanes never commit. */
static inline __m512 load_pad16(const float *p, int n, int i) {
  if(i+16<=n) return _mm512_loadu_ps(p+i);
  __mmask16 m=(__mmask16)((1u<<(n-i))-1u);
  return _mm512_maskz_loadu_ps(m, p+i);
}
static inline void store_pad16(float *p, int n, int i, __m512 v) {
  if(i+16<=n){ _mm512_storeu_ps(p+i,v); return; }
  __mmask16 m=(__mmask16)((1u<<(n-i))-1u);
  _mm512_mask_storeu_ps(p+i,m,v);
}
#endif
static void rmsnorm(float *o, const float *x, const float *w, int n) {
  float ss=0;
#if defined(CWEN_AVX512)
  __m512 vss=_mm512_setzero_ps(); int i=0;
  for(;i<n;i+=16){ __m512 v=load_pad16(x,n,i); vss=_mm512_fmadd_ps(v,v,vss); }
  ss=_mm512_reduce_add_ps(vss);
  float s=1.f/sqrtf(ss/(float)n+RMS_EPS); __m512 vs=_mm512_set1_ps(s);
  for(i=0;i<n;i+=16)
    store_pad16(o,n,i,_mm512_mul_ps(_mm512_mul_ps(load_pad16(w,n,i),load_pad16(x,n,i)),vs));
#elif defined(__AVX2__)
  __m256 vss=_mm256_setzero_ps(); int i=0;
  for(;i+8<=n;i+=8){ __m256 v=_mm256_loadu_ps(x+i); vss=_mm256_fmadd_ps(v,v,vss); }
  ss=hsum256(vss); for(;i<n;i++) ss+=x[i]*x[i];
  float s=1.f/sqrtf(ss/(float)n+RMS_EPS); __m256 vs=_mm256_set1_ps(s);
  for(i=0;i+8<=n;i+=8)
    _mm256_storeu_ps(o+i,_mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(w+i),_mm256_loadu_ps(x+i)),vs));
  for(;i<n;i++) o[i]=w[i]*x[i]*s;
#else
  for(int i=0;i<n;i++) ss+=x[i]*x[i];
  float s=1.f/sqrtf(ss/n + RMS_EPS);
  for(int i=0;i<n;i++) o[i]=w[i]*x[i]*s;
#endif
}
/* Fast sigmoid: clipped rational approx (idea FAST_SILU). Quality-rejected
   (broke the argmax chain); reachable only via the flag. */
static inline float sigmoid_fast(float x) {
  if(x>12.f) return 1.f;
  if(x<-12.f) return 0.f;
  /* 0.5 + 0.5*x*(1 - |x|/6 + x^2/24) clipped rational-ish */
  float a=fabsf(x), y=0.5f*x/(1.f+0.25f*a+0.0625f*a*a)+0.5f;
  return y<0.f?0.f:(y>1.f?1.f:y);
}
static inline float silu_one(float x) {
#if CWEN_IDEA_FAST_SILU
  return x*sigmoid_fast(x);
#else
  return x*(1.f/(1.f+expf(-x)));
#endif
}
static void silu_vec(float *restrict x, int n) {
#if defined(CWEN_AVX512)
  int i=0;
  for(;i<n;i+=16) store_pad16(x,n,i,silu512(load_pad16(x,n,i)));
#else
  for(int i=0;i<n;i++) x[i]=silu_one(x[i]);
#endif
}
static float softplus(float x) { return x>20.f ? x : logf(1.f+expf(x)); }
static float sigmoid(float x) {
#if CWEN_IDEA_FAST_SILU
  return sigmoid_fast(x);
#else
  return 1.f/(1.f+expf(-x));
#endif
}
static void l2norm_rows(float *restrict x, int rows, int dim) {
  for(int r=0;r<rows;r++) {
    float *p=x+r*dim; float ss=0;
#if defined(CWEN_AVX512)
    __m512 vss=_mm512_setzero_ps(); int i=0;
    for(;i<dim;i+=16){ __m512 v=load_pad16(p,dim,i); vss=_mm512_fmadd_ps(v,v,vss); }
    ss=_mm512_reduce_add_ps(vss);
    float s=1.f/sqrtf(ss+RMS_EPS); __m512 vs=_mm512_set1_ps(s);
    for(i=0;i<dim;i+=16) store_pad16(p,dim,i,_mm512_mul_ps(load_pad16(p,dim,i),vs));
#elif defined(__AVX2__)
    __m256 vss=_mm256_setzero_ps(); int i=0;
    for(;i+8<=dim;i+=8){ __m256 v=_mm256_loadu_ps(p+i); vss=_mm256_fmadd_ps(v,v,vss); }
    ss=hsum256(vss); for(;i<dim;i++) ss+=p[i]*p[i];
    float s=1.f/sqrtf(ss+RMS_EPS); __m256 vs=_mm256_set1_ps(s);
    for(i=0;i+8<=dim;i+=8) _mm256_storeu_ps(p+i,_mm256_mul_ps(_mm256_loadu_ps(p+i),vs));
    for(;i<dim;i++) p[i]*=s;
#else
    for(int i=0;i<dim;i++) ss+=p[i]*p[i];
    float s=1.f/sqrtf(ss+RMS_EPS);
    for(int i=0;i<dim;i++) p[i]*=s;
#endif
  }
}

/* ---- quant dequant one row → f32, then dot ---- */
static void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
  if (j < 4) { *d = q[j]&63; *m = q[j+4]&63; }
  else { *d = (q[j+4]&0xF)|((q[j-4]>>6)<<4); *m = (q[j+4]>>4)|((q[j-0]>>6)<<4); }
}
/* Scalar fallback shared by every Q4_0-family kernel's non-SIMD path:
   one 16-byte block, low nibble -> yy[j], high nibble -> yy[j+16], both -8. */
static inline float dot_nib16(const uint8_t *restrict q, float d, const float *restrict yy) {
  float s=0;
  for(int j=0;j<16;j++)
    s+=((int)(q[j]&0xF)-8)*d*yy[j]+((int)(q[j]>>4)-8)*d*yy[j+16];
  return s;
}

#if defined(__F16C__)
static inline float f16_fast(uint16_t h) {
  return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h)));
}
static inline uint16_t f32_to_f16(float x) {
  return (uint16_t)_mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(x), _MM_FROUND_TO_NEAREST_INT));
}
#else
static inline float f16_fast(uint16_t h) { return f16_to_f32(h); }
static inline uint16_t f32_to_f16(float x) {
  /* Software round-to-nearest-even, bit-matching _mm_cvtps_ph: a plain
     exponent-field shift wraps small exponents through unsigned arithmetic
     (1e-8f decoded as a ~65504-scale value) and maps Inf onto a finite
     number, silently corrupting CWENR v2 split scales on non-F16C builds. */
  union { float f; uint32_t u; } v; v.f=x;
  uint32_t sign=(v.u>>16)&0x8000u, rest=v.u&0x7fffffffu;
  if(rest>=0x7f800000u) /* Inf; NaN -> canonical quiet NaN */
    return (uint16_t)(sign|0x7c00u|(rest>0x7f800000u?0x0200u:0));
  if(rest<0x33000000u) /* |x| < 2^-25: rounds to zero, keeps signed zero */
    return (uint16_t)sign;
  uint32_t e=rest>>23, m=rest&0x7fffffu;
  if(e<113){ /* half denormal range: quantize to 2^-24 units, RNE */
    uint32_t t=0x800000u|m, sh=126u-e, half=1u<<(sh-1);
    uint32_t r=t>>sh, rem=t&((half<<1)-1u);
    if(rem>half||(rem==half&&(r&1u))) r++;
    return (uint16_t)(sign+r);
  }
  uint32_t he=e-112u, r=m>>13u, rem=m&0x1fffu;
  if(rem>0x1000u||(rem==0x1000u&&(r&1u))){ if(++r==0x400u){ r=0; he++; } }
  if(he>=31u) return (uint16_t)(sign|0x7c00u);
  return (uint16_t)(sign|(he<<10)|r);
}
#endif

#if defined(__AVX2__)
static inline void i8x16_to_f32x16(__m128i a, __m256 *o0, __m256 *o1) {
  *o0=_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(a));
  *o1=_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(a,8)));
}
/* unpack Q4_0 block → 4x __m256 weight floats (already *scale) */
static inline void q4_0_w4(const block_q4_0 *b, __m256 *w0, __m256 *w1, __m256 *w2, __m256 *w3) {
  float d=f16_fast(b->d);
  __m128i q=_mm_loadu_si128((const __m128i*)b->qs);
  const __m128i m4=_mm_set1_epi8(0x0f), m8=_mm_set1_epi8(8);
  __m128i lo=_mm_sub_epi8(_mm_and_si128(q,m4), m8);
  __m128i hi=_mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(q,4),m4), m8);
  __m256 vd=_mm256_set1_ps(d), a0,a1,b0,b1;
  i8x16_to_f32x16(lo,&a0,&a1); i8x16_to_f32x16(hi,&b0,&b1);
  *w0=_mm256_mul_ps(a0,vd); *w1=_mm256_mul_ps(a1,vd);
  *w2=_mm256_mul_ps(b0,vd); *w3=_mm256_mul_ps(b1,vd);
}
/* Dual-acc FMA chains hide mul latency on Zen4/5; NTA prefetch for streaming W. */
static inline float __attribute__((hot))
dot_q4_0_avx(const block_q4_0 *restrict x, const float *restrict y, int nb) {
  __m256 acc0=_mm256_setzero_ps(), acc1=_mm256_setzero_ps();
  int i=0;
  int pf = CWEN_PREFETCH > 0 ? CWEN_PREFETCH : 4;
#if CWEN_Q4_UNROLL >= 2
  for(; i+1<nb; i+=2) {
    if(i+pf<nb){
      __builtin_prefetch(x+i+pf,0,0); /* NTA: weights stream once */
      __builtin_prefetch(y+(i+pf)*QK4,0,3); /* y reused across rows */
    }
    __m256 w0,w1,w2,w3;
    q4_0_w4(x+i,&w0,&w1,&w2,&w3);
    acc0=_mm256_fmadd_ps(w0,_mm256_loadu_ps(y+i*QK4+0),acc0);
    acc1=_mm256_fmadd_ps(w1,_mm256_loadu_ps(y+i*QK4+8),acc1);
    acc0=_mm256_fmadd_ps(w2,_mm256_loadu_ps(y+i*QK4+16),acc0);
    acc1=_mm256_fmadd_ps(w3,_mm256_loadu_ps(y+i*QK4+24),acc1);
    q4_0_w4(x+i+1,&w0,&w1,&w2,&w3);
    acc0=_mm256_fmadd_ps(w0,_mm256_loadu_ps(y+(i+1)*QK4+0),acc0);
    acc1=_mm256_fmadd_ps(w1,_mm256_loadu_ps(y+(i+1)*QK4+8),acc1);
    acc0=_mm256_fmadd_ps(w2,_mm256_loadu_ps(y+(i+1)*QK4+16),acc0);
    acc1=_mm256_fmadd_ps(w3,_mm256_loadu_ps(y+(i+1)*QK4+24),acc1);
  }
#endif
  for(;i<nb;i++){
    if(i+pf<nb) __builtin_prefetch(x+i+pf,0,0);
    __m256 w0,w1,w2,w3; q4_0_w4(x+i,&w0,&w1,&w2,&w3);
    const float *yy=y+i*QK4;
    acc0=_mm256_fmadd_ps(w0,_mm256_loadu_ps(yy+0),acc0);
    acc1=_mm256_fmadd_ps(w1,_mm256_loadu_ps(yy+8),acc1);
    acc0=_mm256_fmadd_ps(w2,_mm256_loadu_ps(yy+16),acc0);
    acc1=_mm256_fmadd_ps(w3,_mm256_loadu_ps(yy+24),acc1);
  }
  return hsum256(_mm256_add_ps(acc0,acc1));
}
#if defined(CWEN_AVX512)
/* 4-way acc + 2-block unroll: keep Zen5 FMA pipes busy while DRAM feeds W. */
static inline float __attribute__((hot))
dot_q4_0_avx512(const block_q4_0 *restrict x, const float *restrict y, int nb) {
  __m512 a0=_mm512_setzero_ps(), a1=_mm512_setzero_ps();
  __m512 a2=_mm512_setzero_ps(), a3=_mm512_setzero_ps();
  const __m128i m4=_mm_set1_epi8(0x0f), m8=_mm_set1_epi8(8);
  int pf = CWEN_Q4_PF_BLOCKS > 0 ? CWEN_Q4_PF_BLOCKS : 8;
  int i=0;
  for(;i+1<nb;i+=2){
    if(i+pf<nb){
      __builtin_prefetch(x+i+pf,0,0);
      __builtin_prefetch(y+(size_t)(i+pf)*QK4,0,3);
    }
    float d0=f16_fast(x[i].d), d1=f16_fast(x[i+1].d);
    __m128i q0=_mm_loadu_si128((const __m128i*)x[i].qs);
    __m128i q1=_mm_loadu_si128((const __m128i*)x[i+1].qs);
    __m128i lo0=_mm_sub_epi8(_mm_and_si128(q0,m4),m8);
    __m128i hi0=_mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(q0,4),m4),m8);
    __m128i lo1=_mm_sub_epi8(_mm_and_si128(q1,m4),m8);
    __m128i hi1=_mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(q1,4),m4),m8);
    __m512 vd0=_mm512_set1_ps(d0), vd1=_mm512_set1_ps(d1);
    a0=_mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo0)),vd0),
                       _mm512_loadu_ps(y+i*QK4), a0);
    a1=_mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi0)),vd0),
                       _mm512_loadu_ps(y+i*QK4+16), a1);
    a2=_mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo1)),vd1),
                       _mm512_loadu_ps(y+(i+1)*QK4), a2);
    a3=_mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi1)),vd1),
                       _mm512_loadu_ps(y+(i+1)*QK4+16), a3);
  }
  for(;i<nb;i++){
    if(i+pf<nb) __builtin_prefetch(x+i+pf,0,0);
    float d=f16_fast(x[i].d);
    __m128i q=_mm_loadu_si128((const __m128i*)x[i].qs);
    __m128i lo=_mm_sub_epi8(_mm_and_si128(q,m4),m8);
    __m128i hi=_mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(q,4),m4),m8);
    __m512 vd=_mm512_set1_ps(d);
    a0=_mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo)),vd),
                       _mm512_loadu_ps(y+i*QK4), a0);
    a1=_mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi)),vd),
                       _mm512_loadu_ps(y+i*QK4+16), a1);
  }
  return _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0,a1),_mm512_add_ps(a2,a3)));
}
static inline float __attribute__((hot))
dot_q4_1_avx512(const block_q4_1 *restrict x, const float *restrict y, int nb) {
  __m512 a0=_mm512_setzero_ps(), a1=_mm512_setzero_ps();
  const __m128i m4=_mm_set1_epi8(0x0f);
  int pf = CWEN_Q4_PF_BLOCKS > 0 ? CWEN_Q4_PF_BLOCKS : 8;
  for(int i=0;i<nb;i++){
    if(i+pf<nb) __builtin_prefetch(x+i+pf,0,0);
    float d=f16_fast(x[i].d), m=f16_fast(x[i].m);
    __m128i q=_mm_loadu_si128((const __m128i*)x[i].qs);
    __m128i lo=_mm_and_si128(q,m4), hi=_mm_and_si128(_mm_srli_epi16(q,4),m4);
    __m512 vd=_mm512_set1_ps(d), vm=_mm512_set1_ps(m);
    __m512 wlo=_mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo)),vd,vm);
    __m512 whi=_mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi)),vd,vm);
    a0=_mm512_fmadd_ps(wlo,_mm512_loadu_ps(y+i*QK4),a0);
    a1=_mm512_fmadd_ps(whi,_mm512_loadu_ps(y+i*QK4+16),a1);
  }
  return _mm512_reduce_add_ps(_mm512_add_ps(a0,a1));
}
/* Q4_0R AVX-512: 4-acc, 8-block SW pipeline, dual-distance NTA PF (Zen5 dual FMA + DRAM).
   20B blocks stream once → NTA; activations T0 (reused across rows). */
static inline float __attribute__((hot))
dot_q4_0r_avx512(const block_q4_0r *restrict w, const float *restrict y, int nb) {
  __m512 a0=_mm512_setzero_ps(), a1=_mm512_setzero_ps();
  __m512 a2=_mm512_setzero_ps(), a3=_mm512_setzero_ps();
  const __m128i m4=_mm_set1_epi8(0x0f), m8=_mm_set1_epi8(8);
  int pf = CWEN_Q4_PF_BLOCKS > 0 ? CWEN_Q4_PF_BLOCKS : 16;
  int pf2 = pf + (pf>0?pf:8); /* second PF distance: hide longer DRAM latency */
  int i=0;
#define Q4R_FMA(q, d, yl, yh, al, ah) do{ \
    __m128i _lo=_mm_sub_epi8(_mm_and_si128((q),m4),m8); \
    __m128i _hi=_mm_sub_epi8(_mm_and_si128(_mm_srli_epi16((q),4),m4),m8); \
    (al)=_mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_lo)),(d)),(yl),(al)); \
    (ah)=_mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_hi)),(d)),(yh),(ah)); \
  }while(0)
  /* 8-block stride: issue enough independent FMAs to cover L2/DRAM latency */
  for(;i+7<nb;i+=8){
    if(i+pf<nb){
      __builtin_prefetch(w+i+pf,0,0);
      __builtin_prefetch(y+(size_t)(i+pf)*QK4,0,3);
    }
    if(i+pf2<nb){
      __builtin_prefetch(w+i+pf2,0,0);
      __builtin_prefetch(y+(size_t)(i+pf2)*QK4,0,3);
    }
    /* load qs + scales first (hide convert latency behind later FMAs) */
    __m128i q0=_mm_loadu_si128((const __m128i*)w[i+0].qs);
    __m128i q1=_mm_loadu_si128((const __m128i*)w[i+1].qs);
    __m128i q2=_mm_loadu_si128((const __m128i*)w[i+2].qs);
    __m128i q3=_mm_loadu_si128((const __m128i*)w[i+3].qs);
    __m128i q4=_mm_loadu_si128((const __m128i*)w[i+4].qs);
    __m128i q5=_mm_loadu_si128((const __m128i*)w[i+5].qs);
    __m128i q6=_mm_loadu_si128((const __m128i*)w[i+6].qs);
    __m128i q7=_mm_loadu_si128((const __m128i*)w[i+7].qs);
    __m512 d0=_mm512_set1_ps(w[i+0].d), d1=_mm512_set1_ps(w[i+1].d);
    __m512 d2=_mm512_set1_ps(w[i+2].d), d3=_mm512_set1_ps(w[i+3].d);
    __m512 d4=_mm512_set1_ps(w[i+4].d), d5=_mm512_set1_ps(w[i+5].d);
    __m512 d6=_mm512_set1_ps(w[i+6].d), d7=_mm512_set1_ps(w[i+7].d);
    __m512 y0=_mm512_loadu_ps(y+(i+0)*QK4), y0h=_mm512_loadu_ps(y+(i+0)*QK4+16);
    __m512 y1=_mm512_loadu_ps(y+(i+1)*QK4), y1h=_mm512_loadu_ps(y+(i+1)*QK4+16);
    __m512 y2=_mm512_loadu_ps(y+(i+2)*QK4), y2h=_mm512_loadu_ps(y+(i+2)*QK4+16);
    __m512 y3=_mm512_loadu_ps(y+(i+3)*QK4), y3h=_mm512_loadu_ps(y+(i+3)*QK4+16);
    Q4R_FMA(q0,d0,y0,y0h,a0,a1); Q4R_FMA(q1,d1,y1,y1h,a2,a3);
    Q4R_FMA(q2,d2,y2,y2h,a0,a1); Q4R_FMA(q3,d3,y3,y3h,a2,a3);
    y0=_mm512_loadu_ps(y+(i+4)*QK4); y0h=_mm512_loadu_ps(y+(i+4)*QK4+16);
    y1=_mm512_loadu_ps(y+(i+5)*QK4); y1h=_mm512_loadu_ps(y+(i+5)*QK4+16);
    y2=_mm512_loadu_ps(y+(i+6)*QK4); y2h=_mm512_loadu_ps(y+(i+6)*QK4+16);
    y3=_mm512_loadu_ps(y+(i+7)*QK4); y3h=_mm512_loadu_ps(y+(i+7)*QK4+16);
    Q4R_FMA(q4,d4,y0,y0h,a0,a1); Q4R_FMA(q5,d5,y1,y1h,a2,a3);
    Q4R_FMA(q6,d6,y2,y2h,a0,a1); Q4R_FMA(q7,d7,y3,y3h,a2,a3);
  }
  for(;i+3<nb;i+=4){
    if(i+pf<nb){ __builtin_prefetch(w+i+pf,0,0); __builtin_prefetch(y+(size_t)(i+pf)*QK4,0,3); }
    __m128i q0=_mm_loadu_si128((const __m128i*)w[i+0].qs);
    __m128i q1=_mm_loadu_si128((const __m128i*)w[i+1].qs);
    __m128i q2=_mm_loadu_si128((const __m128i*)w[i+2].qs);
    __m128i q3=_mm_loadu_si128((const __m128i*)w[i+3].qs);
    __m512 y0=_mm512_loadu_ps(y+(i+0)*QK4), y0h=_mm512_loadu_ps(y+(i+0)*QK4+16);
    __m512 y1=_mm512_loadu_ps(y+(i+1)*QK4), y1h=_mm512_loadu_ps(y+(i+1)*QK4+16);
    __m512 y2=_mm512_loadu_ps(y+(i+2)*QK4), y2h=_mm512_loadu_ps(y+(i+2)*QK4+16);
    __m512 y3=_mm512_loadu_ps(y+(i+3)*QK4), y3h=_mm512_loadu_ps(y+(i+3)*QK4+16);
    __m512 d0=_mm512_set1_ps(w[i+0].d), d1=_mm512_set1_ps(w[i+1].d);
    __m512 d2=_mm512_set1_ps(w[i+2].d), d3=_mm512_set1_ps(w[i+3].d);
    Q4R_FMA(q0,d0,y0,y0h,a0,a1); Q4R_FMA(q1,d1,y1,y1h,a2,a3);
    Q4R_FMA(q2,d2,y2,y2h,a0,a1); Q4R_FMA(q3,d3,y3,y3h,a2,a3);
  }
  for(;i+1<nb;i+=2){
    if(i+pf<nb) __builtin_prefetch(w+i+pf,0,0);
    __m128i q0=_mm_loadu_si128((const __m128i*)w[i].qs);
    __m128i q1=_mm_loadu_si128((const __m128i*)w[i+1].qs);
    __m512 vd0=_mm512_set1_ps(w[i].d), vd1=_mm512_set1_ps(w[i+1].d);
    __m512 yl0=_mm512_loadu_ps(y+i*QK4), yh0=_mm512_loadu_ps(y+i*QK4+16);
    __m512 yl1=_mm512_loadu_ps(y+(i+1)*QK4), yh1=_mm512_loadu_ps(y+(i+1)*QK4+16);
    Q4R_FMA(q0,vd0,yl0,yh0,a0,a1); Q4R_FMA(q1,vd1,yl1,yh1,a2,a3);
  }
  for(;i<nb;i++){
    __m128i q=_mm_loadu_si128((const __m128i*)w[i].qs);
    __m512 vd=_mm512_set1_ps(w[i].d);
    Q4R_FMA(q,vd,_mm512_loadu_ps(y+i*QK4),_mm512_loadu_ps(y+i*QK4+16),a0,a1);
  }
#undef Q4R_FMA
  return _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0,a1),_mm512_add_ps(a2,a3)));
}
/* Split Q4_0RS: dense qs[16*nb] + f16 scales[nb] (half scale BW vs f32). */
static inline float __attribute__((hot))
dot_q4_0rs_avx512(const uint8_t *restrict qs, const uint16_t *restrict sc,
                  const float *restrict y, int nb) {
  __m512 a0=_mm512_setzero_ps(), a1=_mm512_setzero_ps();
  __m512 a2=_mm512_setzero_ps(), a3=_mm512_setzero_ps();
  const __m128i m4=_mm_set1_epi8(0x0f), m8=_mm_set1_epi8(8);
  int pf = CWEN_Q4_PF_BLOCKS > 0 ? CWEN_Q4_PF_BLOCKS : 16;
  int pf2 = pf + (pf>0?pf:8);
  int i=0;
#define Q4RS_FMA(qi, di, yl, yh, al, ah) do{ \
    __m128i _lo=_mm_sub_epi8(_mm_and_si128((qi),m4),m8); \
    __m128i _hi=_mm_sub_epi8(_mm_and_si128(_mm_srli_epi16((qi),4),m4),m8); \
    (al)=_mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_lo)),(di)),(yl),(al)); \
    (ah)=_mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_hi)),(di)),(yh),(ah)); \
  }while(0)
  for(;i+7<nb;i+=8){
    if(i+pf<nb){
      __builtin_prefetch(qs+(size_t)(i+pf)*16,0,0);
      __builtin_prefetch(sc+i+pf,0,0);
      __builtin_prefetch(y+(size_t)(i+pf)*QK4,0,3);
    }
    if(i+pf2<nb){
      __builtin_prefetch(qs+(size_t)(i+pf2)*16,0,0);
      __builtin_prefetch(sc+i+pf2,0,0);
    }
    /* aligned 16B qs loads when arena is 64B-aligned */
    __m128i q0=_mm_load_si128((const __m128i*)(qs+(size_t)(i+0)*16));
    __m128i q1=_mm_load_si128((const __m128i*)(qs+(size_t)(i+1)*16));
    __m128i q2=_mm_load_si128((const __m128i*)(qs+(size_t)(i+2)*16));
    __m128i q3=_mm_load_si128((const __m128i*)(qs+(size_t)(i+3)*16));
    __m128i q4=_mm_load_si128((const __m128i*)(qs+(size_t)(i+4)*16));
    __m128i q5=_mm_load_si128((const __m128i*)(qs+(size_t)(i+5)*16));
    __m128i q6=_mm_load_si128((const __m128i*)(qs+(size_t)(i+6)*16));
    __m128i q7=_mm_load_si128((const __m128i*)(qs+(size_t)(i+7)*16));
    __m512 d0=_mm512_set1_ps(f16_fast(sc[i+0])), d1=_mm512_set1_ps(f16_fast(sc[i+1]));
    __m512 d2=_mm512_set1_ps(f16_fast(sc[i+2])), d3=_mm512_set1_ps(f16_fast(sc[i+3]));
    __m512 d4=_mm512_set1_ps(f16_fast(sc[i+4])), d5=_mm512_set1_ps(f16_fast(sc[i+5]));
    __m512 d6=_mm512_set1_ps(f16_fast(sc[i+6])), d7=_mm512_set1_ps(f16_fast(sc[i+7]));
    __m512 y0=_mm512_load_ps(y+(i+0)*QK4), y0h=_mm512_load_ps(y+(i+0)*QK4+16);
    __m512 y1=_mm512_load_ps(y+(i+1)*QK4), y1h=_mm512_load_ps(y+(i+1)*QK4+16);
    __m512 y2=_mm512_load_ps(y+(i+2)*QK4), y2h=_mm512_load_ps(y+(i+2)*QK4+16);
    __m512 y3=_mm512_load_ps(y+(i+3)*QK4), y3h=_mm512_load_ps(y+(i+3)*QK4+16);
    Q4RS_FMA(q0,d0,y0,y0h,a0,a1); Q4RS_FMA(q1,d1,y1,y1h,a2,a3);
    Q4RS_FMA(q2,d2,y2,y2h,a0,a1); Q4RS_FMA(q3,d3,y3,y3h,a2,a3);
    y0=_mm512_load_ps(y+(i+4)*QK4); y0h=_mm512_load_ps(y+(i+4)*QK4+16);
    y1=_mm512_load_ps(y+(i+5)*QK4); y1h=_mm512_load_ps(y+(i+5)*QK4+16);
    y2=_mm512_load_ps(y+(i+6)*QK4); y2h=_mm512_load_ps(y+(i+6)*QK4+16);
    y3=_mm512_load_ps(y+(i+7)*QK4); y3h=_mm512_load_ps(y+(i+7)*QK4+16);
    Q4RS_FMA(q4,d4,y0,y0h,a0,a1); Q4RS_FMA(q5,d5,y1,y1h,a2,a3);
    Q4RS_FMA(q6,d6,y2,y2h,a0,a1); Q4RS_FMA(q7,d7,y3,y3h,a2,a3);
  }
  for(;i+3<nb;i+=4){
    if(i+pf<nb){
      __builtin_prefetch(qs+(size_t)(i+pf)*16,0,0);
      __builtin_prefetch(sc+i+pf,0,0);
    }
    __m128i q0=_mm_load_si128((const __m128i*)(qs+(size_t)(i+0)*16));
    __m128i q1=_mm_load_si128((const __m128i*)(qs+(size_t)(i+1)*16));
    __m128i q2=_mm_load_si128((const __m128i*)(qs+(size_t)(i+2)*16));
    __m128i q3=_mm_load_si128((const __m128i*)(qs+(size_t)(i+3)*16));
    __m512 y0=_mm512_load_ps(y+(i+0)*QK4), y0h=_mm512_load_ps(y+(i+0)*QK4+16);
    __m512 y1=_mm512_load_ps(y+(i+1)*QK4), y1h=_mm512_load_ps(y+(i+1)*QK4+16);
    __m512 y2=_mm512_load_ps(y+(i+2)*QK4), y2h=_mm512_load_ps(y+(i+2)*QK4+16);
    __m512 y3=_mm512_load_ps(y+(i+3)*QK4), y3h=_mm512_load_ps(y+(i+3)*QK4+16);
    Q4RS_FMA(q0,_mm512_set1_ps(f16_fast(sc[i+0])),y0,y0h,a0,a1);
    Q4RS_FMA(q1,_mm512_set1_ps(f16_fast(sc[i+1])),y1,y1h,a2,a3);
    Q4RS_FMA(q2,_mm512_set1_ps(f16_fast(sc[i+2])),y2,y2h,a0,a1);
    Q4RS_FMA(q3,_mm512_set1_ps(f16_fast(sc[i+3])),y3,y3h,a2,a3);
  }
  for(;i<nb;i++){
    __m128i q=_mm_load_si128((const __m128i*)(qs+(size_t)i*16));
    Q4RS_FMA(q,_mm512_set1_ps(f16_fast(sc[i])),_mm512_load_ps(y+i*QK4),_mm512_load_ps(y+i*QK4+16),a0,a1);
  }
#undef Q4RS_FMA
  return _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0,a1),_mm512_add_ps(a2,a3)));
}
#endif /* CWEN_AVX512 */
#endif /* __AVX2__ */

static float __attribute__((hot))
dot_q4_0rs(const uint8_t *restrict qs, const uint16_t *restrict sc, const float *restrict y, int n) {
  assert(n%QK4==0);
  int nb=n/QK4;
#if defined(CWEN_AVX512)
  return dot_q4_0rs_avx512(qs,sc,y,nb);
#else
  float sum=0;
  for(int i=0;i<nb;i++){
    float d=f16_fast(sc[i]); const float *yy=y+i*QK4; const uint8_t *q=qs+(size_t)i*16;
    sum += dot_nib16(q,d,yy);
  }
  return sum;
#endif
}

/* Two rows, one pass over activation y (halves act DRAM traffic). */
static void __attribute__((hot))
dot_q4_0rs_2row(const uint8_t *restrict qs0, const uint16_t *restrict sc0,
                const uint8_t *restrict qs1, const uint16_t *restrict sc1,
                const float *restrict y, int n, float *restrict o0, float *restrict o1) {
  assert(n%QK4==0);
  int nb=n/QK4;
#if defined(CWEN_AVX512)
  __m512 a0=_mm512_setzero_ps(), a1=_mm512_setzero_ps();
  __m512 a2=_mm512_setzero_ps(), a3=_mm512_setzero_ps();
  __m512 b0=_mm512_setzero_ps(), b1=_mm512_setzero_ps();
  __m512 b2=_mm512_setzero_ps(), b3=_mm512_setzero_ps();
  const __m128i m4=_mm_set1_epi8(0x0f), m8=_mm_set1_epi8(8);
  int pf=CWEN_Q4_PF_BLOCKS>0?CWEN_Q4_PF_BLOCKS:16;
  int i=0;
#define Q4RS_ROW(ii, al, ah, bl, bh) do{ \
    __m128i q0=_mm_load_si128((const __m128i*)(qs0+(size_t)(ii)*16)); \
    __m128i q1=_mm_load_si128((const __m128i*)(qs1+(size_t)(ii)*16)); \
    __m512 yl=_mm512_load_ps(y+(ii)*QK4), yh=_mm512_load_ps(y+(ii)*QK4+16); \
    __m512 d0=_mm512_set1_ps(f16_fast(sc0[ii])), d1=_mm512_set1_ps(f16_fast(sc1[ii])); \
    __m128i lo0=_mm_sub_epi8(_mm_and_si128(q0,m4),m8); \
    __m128i hi0=_mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(q0,4),m4),m8); \
    __m128i lo1=_mm_sub_epi8(_mm_and_si128(q1,m4),m8); \
    __m128i hi1=_mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(q1,4),m4),m8); \
    (al)=_mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo0)),d0),yl,(al)); \
    (ah)=_mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi0)),d0),yh,(ah)); \
    (bl)=_mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo1)),d1),yl,(bl)); \
    (bh)=_mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi1)),d1),yh,(bh)); \
  }while(0)
  for(;i+1<nb;i+=2){
    if(i+pf<nb){
      __builtin_prefetch(qs0+(size_t)(i+pf)*16,0,0);
      __builtin_prefetch(qs1+(size_t)(i+pf)*16,0,0);
      __builtin_prefetch(sc0+i+pf,0,0);
      __builtin_prefetch(y+(size_t)(i+pf)*QK4,0,3);
    }
    Q4RS_ROW(i,   a0,a1,b0,b1);
    Q4RS_ROW(i+1, a2,a3,b2,b3);
  }
  for(;i<nb;i++) Q4RS_ROW(i, a0,a1,b0,b1);
#undef Q4RS_ROW
  *o0=_mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0,a1),_mm512_add_ps(a2,a3)));
  *o1=_mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(b0,b1),_mm512_add_ps(b2,b3)));
#else
  float s0=0,s1=0;
  for(int i=0;i<nb;i++){
    float d0=f16_fast(sc0[i]), d1=f16_fast(sc1[i]); const float *yy=y+i*QK4;
    const uint8_t *q0=qs0+(size_t)i*16, *q1=qs1+(size_t)i*16;
    s0+=dot_nib16(q0,d0,yy); s1+=dot_nib16(q1,d1,yy);
  }
  *o0=s0; *o1=s1;
#endif
}

/* Two mats, same row, one pass over x (ffn_gate+ffn_up / wk+wv).
   Dual-block unroll keeps 4 FMA chains busy while DRAM feeds qs streams. */
static void __attribute__((hot))
dot_q4_0rs_2mat(const uint8_t *restrict qsa, const uint16_t *restrict sca,
                const uint8_t *restrict qsb, const uint16_t *restrict scb,
                const float *restrict y, int n, float *restrict oa, float *restrict ob) {
  assert(n%QK4==0);
  int nb=n/QK4;
#if defined(CWEN_AVX512)
  __m512 a0=_mm512_setzero_ps(), a1=_mm512_setzero_ps();
  __m512 a2=_mm512_setzero_ps(), a3=_mm512_setzero_ps();
  __m512 b0=_mm512_setzero_ps(), b1=_mm512_setzero_ps();
  __m512 b2=_mm512_setzero_ps(), b3=_mm512_setzero_ps();
  const __m128i m4=_mm_set1_epi8(0x0f), m8=_mm_set1_epi8(8);
  int pf=CWEN_Q4_PF_BLOCKS>0?CWEN_Q4_PF_BLOCKS:16;
  int pf2=pf+(pf>0?pf:8);
  int i=0;
#define Q4RS2_BODY(ii, al, ah, bl, bh) do{ \
    __m128i qa=_mm_load_si128((const __m128i*)(qsa+(size_t)(ii)*16)); \
    __m128i qb=_mm_load_si128((const __m128i*)(qsb+(size_t)(ii)*16)); \
    __m512 yl=_mm512_load_ps(y+(ii)*QK4), yh=_mm512_load_ps(y+(ii)*QK4+16); \
    __m512 da=_mm512_set1_ps(f16_fast(sca[ii])), db=_mm512_set1_ps(f16_fast(scb[ii])); \
    __m128i loa=_mm_sub_epi8(_mm_and_si128(qa,m4),m8); \
    __m128i hia=_mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(qa,4),m4),m8); \
    __m128i lob=_mm_sub_epi8(_mm_and_si128(qb,m4),m8); \
    __m128i hib=_mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(qb,4),m4),m8); \
    (al)=_mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(loa)),da),yl,(al)); \
    (ah)=_mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hia)),da),yh,(ah)); \
    (bl)=_mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lob)),db),yl,(bl)); \
    (bh)=_mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hib)),db),yh,(bh)); \
  }while(0)
  for(;i+1<nb;i+=2){
    if(i+pf<nb){
      __builtin_prefetch(qsa+(size_t)(i+pf)*16,0,0);
      __builtin_prefetch(qsb+(size_t)(i+pf)*16,0,0);
      __builtin_prefetch(sca+i+pf,0,0);
      __builtin_prefetch(scb+i+pf,0,0);
      __builtin_prefetch(y+(size_t)(i+pf)*QK4,0,3);
    }
    if(i+pf2<nb){
      __builtin_prefetch(qsa+(size_t)(i+pf2)*16,0,0);
      __builtin_prefetch(qsb+(size_t)(i+pf2)*16,0,0);
    }
    Q4RS2_BODY(i,   a0,a1,b0,b1);
    Q4RS2_BODY(i+1, a2,a3,b2,b3);
  }
  for(;i<nb;i++) Q4RS2_BODY(i, a0,a1,b0,b1);
#undef Q4RS2_BODY
  *oa=_mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0,a1),_mm512_add_ps(a2,a3)));
  *ob=_mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(b0,b1),_mm512_add_ps(b2,b3)));
#else
  float sa=0,sb=0;
  for(int i=0;i<nb;i++){
    float da=f16_fast(sca[i]), db=f16_fast(scb[i]); const float *yy=y+i*QK4;
    const uint8_t *qa=qsa+(size_t)i*16, *qb=qsb+(size_t)i*16;
    sa+=dot_nib16(qa,da,yy); sb+=dot_nib16(qb,db,yy);
  }
  *oa=sa; *ob=sb;
#endif
}

/* Interleaved dual-mat (CWENR v4): one qs stream (qa|qb)*nb, scales (sca|scb)*nb.
   Single sequential DRAM stream for gate+up / wk+wv. */
static void __attribute__((hot))
dot_q4_0rsi_2mat(const uint8_t *restrict qsab, const uint16_t *restrict scab,
                 const float *restrict y, int n, float *restrict oa, float *restrict ob) {
  assert(n%QK4==0);
  int nb=n/QK4;
#if defined(CWEN_AVX512)
  __m512 a0=_mm512_setzero_ps(), a1=_mm512_setzero_ps();
  __m512 a2=_mm512_setzero_ps(), a3=_mm512_setzero_ps();
  __m512 b0=_mm512_setzero_ps(), b1=_mm512_setzero_ps();
  __m512 b2=_mm512_setzero_ps(), b3=_mm512_setzero_ps();
  const __m128i m4=_mm_set1_epi8(0x0f), m8=_mm_set1_epi8(8);
  int pf=CWEN_Q4_PF_BLOCKS>0?CWEN_Q4_PF_BLOCKS:16;
  int pf2=pf+(pf>0?pf:8);
  int i=0;
#define Q4RSI_BODY(ii, al, ah, bl, bh) do{ \
    const uint8_t *_q=qsab+(size_t)(ii)*32; \
    __m128i qa=_mm_load_si128((const __m128i*)_q); \
    __m128i qb=_mm_load_si128((const __m128i*)(_q+16)); \
    __m512 yl=_mm512_load_ps(y+(ii)*QK4), yh=_mm512_load_ps(y+(ii)*QK4+16); \
    __m512 da=_mm512_set1_ps(f16_fast(scab[2*(ii)])), db=_mm512_set1_ps(f16_fast(scab[2*(ii)+1])); \
    __m128i loa=_mm_sub_epi8(_mm_and_si128(qa,m4),m8); \
    __m128i hia=_mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(qa,4),m4),m8); \
    __m128i lob=_mm_sub_epi8(_mm_and_si128(qb,m4),m8); \
    __m128i hib=_mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(qb,4),m4),m8); \
    (al)=_mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(loa)),da),yl,(al)); \
    (ah)=_mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hia)),da),yh,(ah)); \
    (bl)=_mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lob)),db),yl,(bl)); \
    (bh)=_mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hib)),db),yh,(bh)); \
  }while(0)
  for(;i+1<nb;i+=2){
    if(i+pf<nb){
      __builtin_prefetch(qsab+(size_t)(i+pf)*32,0,0);
      __builtin_prefetch(scab+2*(i+pf),0,0);
      __builtin_prefetch(y+(size_t)(i+pf)*QK4,0,3);
    }
    if(i+pf2<nb) __builtin_prefetch(qsab+(size_t)(i+pf2)*32,0,0);
    Q4RSI_BODY(i,   a0,a1,b0,b1);
    Q4RSI_BODY(i+1, a2,a3,b2,b3);
  }
  for(;i<nb;i++) Q4RSI_BODY(i, a0,a1,b0,b1);
#undef Q4RSI_BODY
  *oa=_mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0,a1),_mm512_add_ps(a2,a3)));
  *ob=_mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(b0,b1),_mm512_add_ps(b2,b3)));
#else
  float sa=0,sb=0;
  for(int i=0;i<nb;i++){
    float da=f16_fast(scab[2*i]), db=f16_fast(scab[2*i+1]);
    const float *yy=y+i*QK4;
    const uint8_t *qa=qsab+(size_t)i*32, *qb=qa+16;
    sa+=dot_nib16(qa,da,yy); sb+=dot_nib16(qb,db,yy);
  }
  *oa=sa; *ob=sb;
#endif
}

/* Single side of interleaved dual (pair_side 0=A, 1=B). */
static float __attribute__((hot))
dot_q4_0rsi(const uint8_t *restrict qsab, const uint16_t *restrict scab,
            int side, const float *restrict y, int n) {
  assert(n%QK4==0);
  assert(side==0||side==1);
  int nb=n/QK4;
#if defined(CWEN_AVX512)
  __m512 a0=_mm512_setzero_ps(), a1=_mm512_setzero_ps();
  const __m128i m4=_mm_set1_epi8(0x0f), m8=_mm_set1_epi8(8);
  int pf=CWEN_Q4_PF_BLOCKS>0?CWEN_Q4_PF_BLOCKS:16;
  for(int i=0;i<nb;i++){
    if(i+pf<nb){
      __builtin_prefetch(qsab+(size_t)(i+pf)*32,0,0);
      __builtin_prefetch(y+(size_t)(i+pf)*QK4,0,3);
    }
    __m128i q=_mm_load_si128((const __m128i*)(qsab+(size_t)i*32+side*16));
    __m512 d=_mm512_set1_ps(f16_fast(scab[2*i+side]));
    __m512 yl=_mm512_load_ps(y+i*QK4), yh=_mm512_load_ps(y+i*QK4+16);
    __m128i lo=_mm_sub_epi8(_mm_and_si128(q,m4),m8);
    __m128i hi=_mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(q,4),m4),m8);
    a0=_mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo)),d),yl,a0);
    a1=_mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi)),d),yh,a1);
  }
  return _mm512_reduce_add_ps(_mm512_add_ps(a0,a1));
#else
  float sum=0;
  for(int i=0;i<nb;i++){
    float d=f16_fast(scab[2*i+side]); const float *yy=y+i*QK4;
    const uint8_t *q=qsab+(size_t)i*32+side*16;
    sum+=dot_nib16(q,d,yy);
  }
  return sum;
#endif
}

static float __attribute__((hot))
dot_q4_0r(const block_q4_0r *restrict w, const float *restrict y, int n) {
  assert(n%QK4==0);
  int nb=n/QK4;
#if defined(CWEN_AVX512)
  return dot_q4_0r_avx512(w,y,nb);
#elif defined(__AVX2__)
  __m256 acc0=_mm256_setzero_ps(), acc1=_mm256_setzero_ps();
  const __m128i m4=_mm_set1_epi8(0x0f), m8=_mm_set1_epi8(8);
  for(int i=0;i<nb;i++){
    __m128i q=_mm_loadu_si128((const __m128i*)w[i].qs);
    __m128i lo=_mm_sub_epi8(_mm_and_si128(q,m4),m8);
    __m128i hi=_mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(q,4),m4),m8);
    __m256 vd=_mm256_set1_ps(w[i].d), a0,a1,b0,b1;
    i8x16_to_f32x16(lo,&a0,&a1); i8x16_to_f32x16(hi,&b0,&b1);
    const float *yy=y+i*QK4;
    acc0=_mm256_fmadd_ps(_mm256_mul_ps(a0,vd),_mm256_loadu_ps(yy+0),acc0);
    acc1=_mm256_fmadd_ps(_mm256_mul_ps(a1,vd),_mm256_loadu_ps(yy+8),acc1);
    acc0=_mm256_fmadd_ps(_mm256_mul_ps(b0,vd),_mm256_loadu_ps(yy+16),acc0);
    acc1=_mm256_fmadd_ps(_mm256_mul_ps(b1,vd),_mm256_loadu_ps(yy+24),acc1);
  }
  return hsum256(_mm256_add_ps(acc0,acc1));
#else
  float sum=0;
  for(int i=0;i<nb;i++){
    float sc=w[i].d; const float *yy=y+i*QK4; const uint8_t *q=w[i].qs;
    sum += dot_nib16(q,sc,yy);
  }
  return sum;
#endif
}

static float __attribute__((hot))
dot_q4_0(const block_q4_0 *restrict x, const float *restrict y, int n) {
  assert(n%QK4==0);
  int nb=n/QK4;
#if defined(CWEN_AVX512)
  return dot_q4_0_avx512(x,y,nb);
#elif defined(__AVX2__)
  return dot_q4_0_avx(x,y,nb);
#else
  float sum=0;
  for(int i=0;i<nb;i++) {
    float d=f16_to_f32(x[i].d); const float *yy=y+i*QK4;
    sum += dot_nib16(x[i].qs,d,yy);
  }
  return sum;
#endif
}
static float __attribute__((hot))
dot_q4_1(const block_q4_1 *restrict x, const float *restrict y, int n) {
  assert(n%QK4==0); int nb=n/QK4;
#if defined(CWEN_AVX512)
  return dot_q4_1_avx512(x,y,nb);
#elif defined(__AVX2__)
  __m256 acc0=_mm256_setzero_ps(), acc1=_mm256_setzero_ps();
  int pf = CWEN_PREFETCH > 0 ? CWEN_PREFETCH : 4;
  for(int i=0;i<nb;i++) {
    if(i+pf<nb) __builtin_prefetch(x+i+pf,0,0);
    float d=f16_fast(x[i].d), m=f16_fast(x[i].m);
    const float *yy=y+i*QK4;
    __m128i q=_mm_loadu_si128((const __m128i*)x[i].qs);
    const __m128i m4=_mm_set1_epi8(0x0f);
    __m128i lo=_mm_and_si128(q,m4), hi=_mm_and_si128(_mm_srli_epi16(q,4),m4);
    __m256 vd=_mm256_set1_ps(d), vm=_mm256_set1_ps(m), a0,a1,b0,b1;
    i8x16_to_f32x16(lo,&a0,&a1); i8x16_to_f32x16(hi,&b0,&b1);
    acc0=_mm256_fmadd_ps(_mm256_fmadd_ps(a0,vd,vm), _mm256_loadu_ps(yy+0), acc0);
    acc1=_mm256_fmadd_ps(_mm256_fmadd_ps(a1,vd,vm), _mm256_loadu_ps(yy+8), acc1);
    acc0=_mm256_fmadd_ps(_mm256_fmadd_ps(b0,vd,vm), _mm256_loadu_ps(yy+16), acc0);
    acc1=_mm256_fmadd_ps(_mm256_fmadd_ps(b1,vd,vm), _mm256_loadu_ps(yy+24), acc1);
  }
  return hsum256(_mm256_add_ps(acc0,acc1));
#else
  float sum=0;
  for(int i=0;i<nb;i++) {
    float d=f16_to_f32(x[i].d), m=f16_to_f32(x[i].m); const float *yy=y+i*QK4;
    for(int j=0;j<QK4/2;j++) {
      int x0=x[i].qs[j]&0xF, x1=x[i].qs[j]>>4;
      sum += (x0*d+m)*yy[j] + (x1*d+m)*yy[j+QK4/2];
    }
  }
  return sum;
#endif
}
/* Two Q4_1 rows, one pass over y (ffn_down). */
static void __attribute__((hot))
dot_q4_1_2row(const block_q4_1 *restrict w0, const block_q4_1 *restrict w1,
              const float *restrict y, int n, float *restrict o0, float *restrict o1) {
  assert(n%QK4==0);
#if defined(CWEN_AVX512)
  int nb=n/QK4;
  __m512 a0=_mm512_setzero_ps(), a1=_mm512_setzero_ps();
  __m512 b0=_mm512_setzero_ps(), b1=_mm512_setzero_ps();
  const __m128i m4=_mm_set1_epi8(0x0f);
  int pf=CWEN_Q4_PF_BLOCKS>0?CWEN_Q4_PF_BLOCKS:12;
  for(int i=0;i<nb;i++){
    if(i+pf<nb){
      __builtin_prefetch(w0+i+pf,0,0);
      __builtin_prefetch(w1+i+pf,0,0);
      __builtin_prefetch(y+(size_t)(i+pf)*QK4,0,3);
    }
    __m512 yl=_mm512_load_ps(y+i*QK4), yh=_mm512_load_ps(y+i*QK4+16);
    float d0=f16_fast(w0[i].d), m0=f16_fast(w0[i].m);
    float d1=f16_fast(w1[i].d), m1=f16_fast(w1[i].m);
    __m128i q0=_mm_loadu_si128((const __m128i*)w0[i].qs);
    __m128i q1=_mm_loadu_si128((const __m128i*)w1[i].qs);
    __m128i lo0=_mm_and_si128(q0,m4), hi0=_mm_and_si128(_mm_srli_epi16(q0,4),m4);
    __m128i lo1=_mm_and_si128(q1,m4), hi1=_mm_and_si128(_mm_srli_epi16(q1,4),m4);
    __m512 vd0=_mm512_set1_ps(d0), vm0=_mm512_set1_ps(m0);
    __m512 vd1=_mm512_set1_ps(d1), vm1=_mm512_set1_ps(m1);
    __m512 wl0=_mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo0)),vd0,vm0);
    __m512 wh0=_mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi0)),vd0,vm0);
    __m512 wl1=_mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo1)),vd1,vm1);
    __m512 wh1=_mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi1)),vd1,vm1);
    a0=_mm512_fmadd_ps(wl0,yl,a0); a1=_mm512_fmadd_ps(wh0,yh,a1);
    b0=_mm512_fmadd_ps(wl1,yl,b0); b1=_mm512_fmadd_ps(wh1,yh,b1);
  }
  *o0=_mm512_reduce_add_ps(_mm512_add_ps(a0,a1));
  *o1=_mm512_reduce_add_ps(_mm512_add_ps(b0,b1));
#else
  *o0=dot_q4_1(w0,y,n); *o1=dot_q4_1(w1,y,n);
#endif
}
/* Q5_K: dual-acc FMA; ssm_out ~6% cycles. */
static float __attribute__((hot))
dot_q5_K(const block_q5_K *restrict x, const float *restrict y, int n) {
  assert(n%QK_K==0);
#if defined(__AVX2__)
  __m256 acc0=_mm256_setzero_ps(), acc1=_mm256_setzero_ps();
  const float *yy=y;
  for(int i=0,nb=n/QK_K;i<nb;i++) {
    const uint8_t *ql=x[i].qs, *qh=x[i].qh;
    float d=f16_fast(x[i].d), min=f16_fast(x[i].dmin);
    int is=0; uint8_t u1=1,u2=2;
    __builtin_prefetch(x+i+1,0,0);
    __builtin_prefetch(yy+256,0,3);
    for(int j=0;j<QK_K;j+=64) {
      uint8_t sc,m;
      get_scale_min_k4(is+0,x[i].scales,&sc,&m); float d1=d*sc,m1=min*m;
      get_scale_min_k4(is+1,x[i].scales,&sc,&m); float d2=d*sc,m2=min*m;
      __m256 vd1=_mm256_set1_ps(d1), vm1=_mm256_set1_ps(m1);
      __m256 vd2=_mm256_set1_ps(d2), vm2=_mm256_set1_ps(m2);
      for(int l=0;l<32;l+=8){
        __m256i qv=_mm256_setr_epi32(
          (ql[l+0]&0xF)|((qh[l+0]&u1)?16:0),(ql[l+1]&0xF)|((qh[l+1]&u1)?16:0),
          (ql[l+2]&0xF)|((qh[l+2]&u1)?16:0),(ql[l+3]&0xF)|((qh[l+3]&u1)?16:0),
          (ql[l+4]&0xF)|((qh[l+4]&u1)?16:0),(ql[l+5]&0xF)|((qh[l+5]&u1)?16:0),
          (ql[l+6]&0xF)|((qh[l+6]&u1)?16:0),(ql[l+7]&0xF)|((qh[l+7]&u1)?16:0));
        acc0=_mm256_fmadd_ps(_mm256_sub_ps(_mm256_mul_ps(vd1,_mm256_cvtepi32_ps(qv)),vm1),
                             _mm256_loadu_ps(yy+l), acc0);
      }
      for(int l=0;l<32;l+=8){
        __m256i qv=_mm256_setr_epi32(
          (ql[l+0]>>4)|((qh[l+0]&u2)?16:0),(ql[l+1]>>4)|((qh[l+1]&u2)?16:0),
          (ql[l+2]>>4)|((qh[l+2]&u2)?16:0),(ql[l+3]>>4)|((qh[l+3]&u2)?16:0),
          (ql[l+4]>>4)|((qh[l+4]&u2)?16:0),(ql[l+5]>>4)|((qh[l+5]&u2)?16:0),
          (ql[l+6]>>4)|((qh[l+6]&u2)?16:0),(ql[l+7]>>4)|((qh[l+7]&u2)?16:0));
        acc1=_mm256_fmadd_ps(_mm256_sub_ps(_mm256_mul_ps(vd2,_mm256_cvtepi32_ps(qv)),vm2),
                             _mm256_loadu_ps(yy+32+l), acc1);
      }
      yy+=64; ql+=32; is+=2; u1<<=2; u2<<=2;
    }
  }
  return hsum256(_mm256_add_ps(acc0,acc1));
#else
  float sum=0; const float *yy=y;
  for(int i=0,nb=n/QK_K;i<nb;i++) {
    const uint8_t *ql=x[i].qs, *qh=x[i].qh;
    float d=f16_fast(x[i].d), min=f16_fast(x[i].dmin);
    int is=0; uint8_t u1=1,u2=2;
    for(int j=0;j<QK_K;j+=64) {
      uint8_t sc,m;
      get_scale_min_k4(is+0,x[i].scales,&sc,&m); float d1=d*sc,m1=min*m;
      get_scale_min_k4(is+1,x[i].scales,&sc,&m); float d2=d*sc,m2=min*m;
      for(int l=0;l<32;l++) sum += (d1*((ql[l]&0xF)+(qh[l]&u1?16:0))-m1)*yy[l];
      for(int l=0;l<32;l++) sum += (d2*((ql[l]>>4)+(qh[l]&u2?16:0))-m2)*yy[32+l];
      yy+=64; ql+=32; is+=2; u1<<=2; u2<<=2;
    }
  }
  return sum;
#endif
}
/* Q6_K: 4-acc FMA, no stack int temps (lm_head ~11% cycles). */
static float __attribute__((hot))
dot_q6_K(const block_q6_K *restrict x, const float *restrict y, int n) {
  assert(n%QK_K==0);
#if defined(__AVX2__)
  __m256 acc0=_mm256_setzero_ps(), acc1=_mm256_setzero_ps();
  __m256 acc2=_mm256_setzero_ps(), acc3=_mm256_setzero_ps();
  const float *yy=y;
  for(int i=0,nb=n/QK_K;i<nb;i++) {
    float d=f16_fast(x[i].d);
    const uint8_t *ql=x[i].ql, *qh=x[i].qh; const int8_t *sc=x[i].scales;
    __builtin_prefetch(x+i+1,0,0);
    __builtin_prefetch(yy+256,0,3);
    for(int n0=0;n0<QK_K;n0+=128) {
      for(int grp=0;grp<2;grp++){
        int l0=grp*16;
        __m256 s0=_mm256_set1_ps(d*(float)sc[grp+0]);
        __m256 s2=_mm256_set1_ps(d*(float)sc[grp+2]);
        __m256 s4=_mm256_set1_ps(d*(float)sc[grp+4]);
        __m256 s6=_mm256_set1_ps(d*(float)sc[grp+6]);
        for(int l=l0;l<l0+16;l+=8){
          /* setr: no int[8] stack spill (prior Q6 flame) */
          __m256i q1=_mm256_setr_epi32(
            (int)((int8_t)((ql[l+0]&0xF)|(((qh[l+0]>>0)&3)<<4))-32),
            (int)((int8_t)((ql[l+1]&0xF)|(((qh[l+1]>>0)&3)<<4))-32),
            (int)((int8_t)((ql[l+2]&0xF)|(((qh[l+2]>>0)&3)<<4))-32),
            (int)((int8_t)((ql[l+3]&0xF)|(((qh[l+3]>>0)&3)<<4))-32),
            (int)((int8_t)((ql[l+4]&0xF)|(((qh[l+4]>>0)&3)<<4))-32),
            (int)((int8_t)((ql[l+5]&0xF)|(((qh[l+5]>>0)&3)<<4))-32),
            (int)((int8_t)((ql[l+6]&0xF)|(((qh[l+6]>>0)&3)<<4))-32),
            (int)((int8_t)((ql[l+7]&0xF)|(((qh[l+7]>>0)&3)<<4))-32));
          __m256i q2=_mm256_setr_epi32(
            (int)((int8_t)((ql[l+32]&0xF)|(((qh[l+0]>>2)&3)<<4))-32),
            (int)((int8_t)((ql[l+33]&0xF)|(((qh[l+1]>>2)&3)<<4))-32),
            (int)((int8_t)((ql[l+34]&0xF)|(((qh[l+2]>>2)&3)<<4))-32),
            (int)((int8_t)((ql[l+35]&0xF)|(((qh[l+3]>>2)&3)<<4))-32),
            (int)((int8_t)((ql[l+36]&0xF)|(((qh[l+4]>>2)&3)<<4))-32),
            (int)((int8_t)((ql[l+37]&0xF)|(((qh[l+5]>>2)&3)<<4))-32),
            (int)((int8_t)((ql[l+38]&0xF)|(((qh[l+6]>>2)&3)<<4))-32),
            (int)((int8_t)((ql[l+39]&0xF)|(((qh[l+7]>>2)&3)<<4))-32));
          __m256i q3=_mm256_setr_epi32(
            (int)((int8_t)((ql[l+0]>>4)|(((qh[l+0]>>4)&3)<<4))-32),
            (int)((int8_t)((ql[l+1]>>4)|(((qh[l+1]>>4)&3)<<4))-32),
            (int)((int8_t)((ql[l+2]>>4)|(((qh[l+2]>>4)&3)<<4))-32),
            (int)((int8_t)((ql[l+3]>>4)|(((qh[l+3]>>4)&3)<<4))-32),
            (int)((int8_t)((ql[l+4]>>4)|(((qh[l+4]>>4)&3)<<4))-32),
            (int)((int8_t)((ql[l+5]>>4)|(((qh[l+5]>>4)&3)<<4))-32),
            (int)((int8_t)((ql[l+6]>>4)|(((qh[l+6]>>4)&3)<<4))-32),
            (int)((int8_t)((ql[l+7]>>4)|(((qh[l+7]>>4)&3)<<4))-32));
          __m256i q4=_mm256_setr_epi32(
            (int)((int8_t)((ql[l+32]>>4)|(((qh[l+0]>>6)&3)<<4))-32),
            (int)((int8_t)((ql[l+33]>>4)|(((qh[l+1]>>6)&3)<<4))-32),
            (int)((int8_t)((ql[l+34]>>4)|(((qh[l+2]>>6)&3)<<4))-32),
            (int)((int8_t)((ql[l+35]>>4)|(((qh[l+3]>>6)&3)<<4))-32),
            (int)((int8_t)((ql[l+36]>>4)|(((qh[l+4]>>6)&3)<<4))-32),
            (int)((int8_t)((ql[l+37]>>4)|(((qh[l+5]>>6)&3)<<4))-32),
            (int)((int8_t)((ql[l+38]>>4)|(((qh[l+6]>>6)&3)<<4))-32),
            (int)((int8_t)((ql[l+39]>>4)|(((qh[l+7]>>6)&3)<<4))-32));
          acc0=_mm256_fmadd_ps(_mm256_mul_ps(s0,_mm256_cvtepi32_ps(q1)),_mm256_loadu_ps(yy+l),acc0);
          acc1=_mm256_fmadd_ps(_mm256_mul_ps(s2,_mm256_cvtepi32_ps(q2)),_mm256_loadu_ps(yy+l+32),acc1);
          acc2=_mm256_fmadd_ps(_mm256_mul_ps(s4,_mm256_cvtepi32_ps(q3)),_mm256_loadu_ps(yy+l+64),acc2);
          acc3=_mm256_fmadd_ps(_mm256_mul_ps(s6,_mm256_cvtepi32_ps(q4)),_mm256_loadu_ps(yy+l+96),acc3);
        }
      }
      yy+=128; ql+=64; qh+=32; sc+=8;
    }
  }
  return hsum256(_mm256_add_ps(_mm256_add_ps(acc0,acc1),_mm256_add_ps(acc2,acc3)));
#else
  float sum=0; const float *yy=y;
  for(int i=0,nb=n/QK_K;i<nb;i++) {
    float d=f16_fast(x[i].d);
    const uint8_t *ql=x[i].ql, *qh=x[i].qh; const int8_t *sc=x[i].scales;
    for(int n0=0;n0<QK_K;n0+=128) {
      for(int l=0;l<32;l++) {
        int is=l/16;
        int8_t q1=(int8_t)((ql[l]&0xF)|(((qh[l]>>0)&3)<<4))-32;
        int8_t q2=(int8_t)((ql[l+32]&0xF)|(((qh[l]>>2)&3)<<4))-32;
        int8_t q3=(int8_t)((ql[l]>>4)|(((qh[l]>>4)&3)<<4))-32;
        int8_t q4=(int8_t)((ql[l+32]>>4)|(((qh[l]>>6)&3)<<4))-32;
        sum += d*sc[is+0]*q1*yy[l] + d*sc[is+2]*q2*yy[l+32]
             + d*sc[is+4]*q3*yy[l+64] + d*sc[is+6]*q4*yy[l+96];
      }
      yy+=128; ql+=64; qh+=32; sc+=8;
    }
  }
  return sum;
#endif
}
static float __attribute__((hot))
dot_q8_0(const block_q8_0 *restrict w, const float *restrict y, int n) {
  assert(n%QK4==0);
  int nb=n/QK4;
#if defined(CWEN_AVX512)
  __m512 acc=_mm512_setzero_ps();
  for(int i=0;i<nb;i++){
    float d=f16_fast(w[i].d);
    __m256i q=_mm256_loadu_si256((const __m256i*)w[i].qs);
    __m512 vd=_mm512_set1_ps(d);
    __m512 yl=_mm512_loadu_ps(y+i*QK4);
    __m512 yh=_mm512_loadu_ps(y+i*QK4+16);
    __m512 ql=_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm256_castsi256_si128(q)));
    __m512 qh=_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm256_extracti128_si256(q,1)));
    acc=_mm512_fmadd_ps(_mm512_mul_ps(ql,vd),yl,acc);
    acc=_mm512_fmadd_ps(_mm512_mul_ps(qh,vd),yh,acc);
  }
  return _mm512_reduce_add_ps(acc);
#else
  float sum=0;
  for(int i=0;i<nb;i++){
    float d=f16_to_f32(w[i].d); const float *yy=y+i*QK4;
    for(int j=0;j<32;j++) sum += (float)w[i].qs[j]*d*yy[j];
  }
  return sum;
#endif
}
static float dot_f32(const float *restrict w, const float *restrict x, int n) {
#if defined(CWEN_AVX512)
  __m512 acc=_mm512_setzero_ps(); int i=0;
  for(;i<n;i+=16)
    acc=_mm512_fmadd_ps(load_pad16(w,n,i),load_pad16(x,n,i),acc);
  return _mm512_reduce_add_ps(acc);
#elif defined(__AVX2__)
  __m256 acc=_mm256_setzero_ps(); int i=0;
  for(;i+8<=n;i+=8) acc=_mm256_fmadd_ps(_mm256_loadu_ps(w+i),_mm256_loadu_ps(x+i),acc);
  float s=hsum256(acc);
  for(;i<n;i++) s+=w[i]*x[i];
  return s;
#else
  float s=0; for(int i=0;i<n;i++) s+=w[i]*x[i]; return s;
#endif
}

/* W is [ne0=K, ne1=M] row-major quant rows; y[M] = W x */
static size_t row_bytes(int type, int ne0) {
  switch(type) {
    case T_F32: return (size_t)ne0*4;
    case T_Q4_0: return (size_t)(ne0/QK4)*sizeof(block_q4_0);
    case T_Q4_0R: return q4r_row_bytes(ne0);
    case T_Q4_0RS: return q4rs_row_qs(ne0); /* qs only; scales side channel */
    case T_Q4_0RSI: return q4rsi_row_qs(ne0); /* interleaved qsA|qsB per block */
    case T_Q4_1: return (size_t)(ne0/QK4)*sizeof(block_q4_1);
    case T_Q8_0: return (size_t)(ne0/QK4)*sizeof(block_q8_0);
    case T_Q8S:  return (size_t)ne0 + (size_t)(ne0/QK4)*2;
    case T_Q5_K: return (size_t)(ne0/QK_K)*sizeof(block_q5_K);
    case T_Q6_K: return (size_t)(ne0/QK_K)*sizeof(block_q6_K);
    default: return 0; /* unused tensors (e.g. unknown quants) */
  }
}
/* Types the gemv dispatch actually computes. Kept next to row_bytes so the
   three lists (this, row_bytes, gemv/gemv_row switches) cannot drift apart:
   load-time validators accept exactly these. */
static int matmul_type_ok(int ty) {
  switch(ty){
    case T_F32: case T_Q4_0: case T_Q4_1: case T_Q8_0: case T_Q5_K: case T_Q6_K:
    case T_Q4_0R: case T_Q4_0RS: case T_Q4_0RSI:
    case T_Q8S: return 1; /* split-Q8 (drafter); Q8SI consumed via df_dual_gemvb */
    default: return 0;
  }
}
/* prefetch control (runtime knobs: CWEN_RESIDENCY, CWEN_PF_T0, CWEN_NO_PF) */
static int g_no_pf;      /* disable weight prefetch entirely */
static int g_pf_t0;      /* use T0 (all levels) instead of NTA */
static int g_pipe_pf;    /* prefetch next layer's first weights */
static __attribute__((unused)) int g_residency;  /* THP+mlock+prefault memory hygiene */

static inline void cwen_pf_w(const void *p) {
  if(g_no_pf) return;
  if(g_pf_t0) __builtin_prefetch(p,0,3);
  else __builtin_prefetch(p,0,0); /* NTA */
}

/* Row kernel: static inline so fused call sites can be one OMP loop + inlined bodies.
   Globals hold activations; pass only W / x / row index. */
/* split-Q8 dot (drafter): int8 stream contiguous, f16 scale channel beside it.
   Scales pointer comes from the Tensor; row index selects both rows. */
static inline float dot_q8s(const Tensor *restrict W, const float *restrict x,
                            int row) {
  const int K=W->ne0, nb=K/QK4;
  const int8_t *q=(const int8_t*)W->data+(size_t)row*K;
  const uint16_t *sc=(const uint16_t*)W->scales+(size_t)row*nb;
  /* AVX2 only: same reason as dot_q8si below. The 512-bit branch this once
     carried fed __m256 scales into _mm512 FMAs and never compiled. */
  __m256 acc=_mm256_setzero_ps();
  for(int b=0;b<nb;b++){
    const __m128i q0=_mm_loadl_epi64((const __m128i*)(q+b*32));
    const __m128i q1=_mm_loadl_epi64((const __m128i*)(q+b*32+8));
    const __m128i q2=_mm_loadl_epi64((const __m128i*)(q+b*32+16));
    const __m128i q3=_mm_loadl_epi64((const __m128i*)(q+b*32+24));
    const __m256 vd=_mm256_set1_ps(f16_to_f32(sc[b]));
    acc=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q0)),vd),
                        _mm256_loadu_ps(x+b*32),acc);
    acc=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q1)),vd),
                        _mm256_loadu_ps(x+b*32+8),acc);
    acc=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q2)),vd),
                        _mm256_loadu_ps(x+b*32+16),acc);
    acc=_mm256_fmadd_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q3)),vd),
                        _mm256_loadu_ps(x+b*32+24),acc);
  }
  return hsum256(acc);
}
static inline float gemv_row(const Tensor *W, const float *x, int i) {
  /* an unknown type here would return 0.f and poison downstream activations
     silently; load-time validators accept exactly matmul_type_ok's set */
  assert(matmul_type_ok(W->type));
  size_t rb=row_bytes(W->type,W->ne0);
  const void *row=(const char*)W->data+(size_t)i*rb;
  int K=W->ne0;
  switch(W->type){
    case T_Q4_0RS: {
      int nb=(int)q4r_nb(K);
      return dot_q4_0rs((const uint8_t*)row, (const uint16_t*)W->scales+(size_t)i*nb, x, K);
    }
    case T_Q4_0RSI: {
      int nb=(int)q4r_nb(K);
      return dot_q4_0rsi((const uint8_t*)row, (const uint16_t*)W->scales+(size_t)i*(nb*2),
                         W->pair_side, x, K);
    }
    case T_Q4_0R: return dot_q4_0r((const block_q4_0r*)row,x,K);
    case T_Q4_0:  return dot_q4_0((const block_q4_0*)row,x,K);
    case T_Q4_1:  return dot_q4_1((const block_q4_1*)row,x,K);
    case T_Q5_K:  return dot_q5_K((const block_q5_K*)row,x,K);
    case T_Q6_K:  return dot_q6_K((const block_q6_K*)row,x,K);
    case T_Q8_0:  return dot_q8_0((const block_q8_0*)row,x,K);
    case T_Q8S:   return dot_q8s(W,x,i); /* split layout: needs row index */
    case T_F32:   return dot_f32((const float*)row,x,K);
    default: return 0.f;
  }
}
static inline void gemv_pf_row(const Tensor *W, int i, int pfd, int M) {
  if(i+pfd>=M) return;
  size_t rb=row_bytes(W->type,W->ne0);
  cwen_pf_w((const char*)W->data+(size_t)(i+pfd)*rb);
}

/* Parallel if row-work amortizes GOMP fork (~300 regions/token was still a tax). */
static inline int gemv_use_omp(int M, int K) {
  /* long long, not int: the GA's evolved expression multiplies M and K, and a
     32-bit intermediate would overflow (undefined) on the wide mats while the
     GA scored the genome on Python's exact value. tools/ga_expr_check.py pins
     the two evaluators together. */
  long long thr = CWEN_OMP_THRESH_EXPR(M, K);
  if(M<=thr) return 0;
  long blocks = (long)M * ((long)K > 0 ? ((long)K/32) : 1);
  return blocks >= 4096; /* was 2048; cut medium-mat forks */
}

/* Monomorphic gemv: hoist type once (per-row switch blocked inlining of hot dots). */
#define GEMV_TYPED(dotfn, T) do{ \
  const T *_base=(const T*)W->data; \
  size_t _rb=row_bytes(W->type,K); \
  int _pfd=CWEN_PREFETCH>0?CWEN_PREFETCH:4; \
  if(!gemv_use_omp(M,K)){ for(int i=0;i<M;i++) y[i]=dotfn((const T*)((const char*)_base+(size_t)i*_rb),x,K); return; } \
  _Pragma("omp parallel for schedule(static)") \
  for(int i=0;i<M;i++){ \
    if(i+_pfd<M) cwen_pf_w((const char*)_base+(size_t)(i+_pfd)*_rb); \
    y[i]=dotfn((const T*)((const char*)_base+(size_t)i*_rb),x,K); \
  } \
}while(0)

static void __attribute__((hot))
gemv(const Tensor *W, const float *restrict x, float *restrict y) {
  /* gemv writes nothing for a type outside its dispatch table: an unchecked
     tensor would serve stale output with no error (see rebind_layers_from_tens) */
  assert(matmul_type_ok(W->type));
  int M=W->ne1, K=W->ne0;
  if(W->type==T_Q4_0RS){
    /* split-stream + multi-row (shared activation loads) */
    int nb=(int)q4r_nb(K);
    const uint8_t *qs=(const uint8_t*)W->data;
    const uint16_t *sc=(const uint16_t*)W->scales;
    int pfd=CWEN_PREFETCH>0?CWEN_PREFETCH:4;
    size_t rqs=(size_t)nb*16, rsc=(size_t)nb;
    if(!gemv_use_omp(M,K)){
      int i=0;
      for(;i+1<M;i+=2)
        dot_q4_0rs_2row(qs+(size_t)i*rqs, sc+(size_t)i*rsc,
                        qs+(size_t)(i+1)*rqs, sc+(size_t)(i+1)*rsc, x, K, y+i, y+i+1);
      for(;i<M;i++) y[i]=dot_q4_0rs(qs+(size_t)i*rqs, sc+(size_t)i*rsc, x, K);
      return;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<M;i+=2){
      if(i+1<M){
        if(i+pfd+1<M){
          cwen_pf_w(qs+(size_t)(i+pfd)*rqs);
          cwen_pf_w(qs+(size_t)(i+pfd+1)*rqs);
        }
        dot_q4_0rs_2row(qs+(size_t)i*rqs, sc+(size_t)i*rsc,
                        qs+(size_t)(i+1)*rqs, sc+(size_t)(i+1)*rsc, x, K, y+i, y+i+1);
      } else {
        y[i]=dot_q4_0rs(qs+(size_t)i*rqs, sc+(size_t)i*rsc, x, K);
      }
    }
    return;
  }
  if(W->type==T_Q4_0RSI){
    /* single-side of interleaved pair (prefer dual-mat at call sites) */
    int nb=(int)q4r_nb(K);
    const uint8_t *qs=(const uint8_t*)W->data;
    const uint16_t *sc=(const uint16_t*)W->scales;
    int side=W->pair_side;
    int pfd=CWEN_PREFETCH>0?CWEN_PREFETCH:4;
    size_t rqs=q4rsi_row_qs(K), rsc=(size_t)nb*2;
    if(!gemv_use_omp(M,K)){
      for(int i=0;i<M;i++)
        y[i]=dot_q4_0rsi(qs+(size_t)i*rqs, sc+(size_t)i*rsc, side, x, K);
      return;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<M;i++){
      if(i+pfd<M) cwen_pf_w(qs+(size_t)(i+pfd)*rqs);
      y[i]=dot_q4_0rsi(qs+(size_t)i*rqs, sc+(size_t)i*rsc, side, x, K);
    }
    return;
  }
  if(W->type==T_Q4_1){
    /* multi-row Q4_1: share activation loads (ffn_down) */
    const block_q4_1 *base=(const block_q4_1*)W->data;
    size_t rb=row_bytes(T_Q4_1,K);
    int pfd=CWEN_PREFETCH>0?CWEN_PREFETCH:4;
    int nblk=K/QK4;
    if(!gemv_use_omp(M,K)){
      int i=0;
      for(;i+1<M;i+=2)
        dot_q4_1_2row(base+(size_t)i*nblk, base+(size_t)(i+1)*nblk, x, K, y+i, y+i+1);
      for(;i<M;i++) y[i]=dot_q4_1(base+(size_t)i*nblk, x, K);
      return;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<M;i+=2){
      if(i+1<M){
        if(i+pfd+1<M){
          cwen_pf_w((const char*)base+(size_t)(i+pfd)*rb);
          cwen_pf_w((const char*)base+(size_t)(i+pfd+1)*rb);
        }
        dot_q4_1_2row(base+(size_t)i*nblk, base+(size_t)(i+1)*nblk, x, K, y+i, y+i+1);
      } else y[i]=dot_q4_1(base+(size_t)i*nblk, x, K);
    }
    return;
  }
  switch(W->type){
    case T_Q4_0R: GEMV_TYPED(dot_q4_0r, block_q4_0r); return;
    case T_Q4_0:  GEMV_TYPED(dot_q4_0,  block_q4_0);  return;
    case T_Q5_K:  GEMV_TYPED(dot_q5_K,  block_q5_K);  return;
    case T_Q6_K:  GEMV_TYPED(dot_q6_K,  block_q6_K);  return;
    case T_Q8_0:  GEMV_TYPED(dot_q8_0,  block_q8_0);  return;
    case T_F32: {
      const float *base=(const float*)W->data;
      int pfd=CWEN_PREFETCH>0?CWEN_PREFETCH:4;
      if(!gemv_use_omp(M,K)){ for(int i=0;i<M;i++) y[i]=dot_f32(base+(size_t)i*K,x,K); return; }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for(int i=0;i<M;i++){
        if(i+pfd<M) cwen_pf_w(base+(size_t)(i+pfd)*K);
        y[i]=dot_f32(base+(size_t)i*K,x,K);
      }
      return;
    }
    default: break;
  }
}

#if defined(CWEN_AVX512) && !defined(CWEN_NO_BCOL)
/* Unpack-once Q4 row against B activation columns.
   The per-column dot calls gemvb used to make redid the nibble split, the
   int8->f32 widen and the scale multiply once per column; only the weight
   *load* was shared. Streaming the row is already amortized by the block, so
   what is left to cut is that ALU work: dequantize each 32-weight block into
   two vectors once, then FMA them against every column. One accumulator per
   column keeps the register file inside 32 zmm for B<=8 (8 accumulators,
   2 weight vectors, 2 activation vectors); wider blocks stay on the
   per-column path rather than spilling.

   qstr/qoff and scstr/scoff select the layout, and both call sites pass
   constants: split rows are (16,0)/(1,0), interleaved rows pack the pair into
   one block as (32,side*16)/(2,side). */
#define Q4RS_BCOL_MAX 8
static inline void dot_q4_bcol(const uint8_t *restrict qs, size_t qstr, size_t qoff,
                               const uint16_t *restrict sc, size_t scstr, size_t scoff,
                               const float *restrict X, int xs, int nb,
                               float *restrict Y, int ys, int B) {
  const __m128i m4=_mm_set1_epi8(0x0f), m8=_mm_set1_epi8(8);
  __m512 acc[Q4RS_BCOL_MAX];
  for(int b=0;b<B;b++) acc[b]=_mm512_setzero_ps();
  int pf = CWEN_Q4_PF_BLOCKS > 0 ? CWEN_Q4_PF_BLOCKS : 16;
  for(int i=0;i<nb;i++){
    if(i+pf<nb){
      __builtin_prefetch(qs+(size_t)(i+pf)*qstr,0,0);
      __builtin_prefetch(sc+(size_t)(i+pf)*scstr,0,0);
    }
    const __m128i q=_mm_load_si128((const __m128i*)(qs+(size_t)i*qstr+qoff));
    const __m512 d=_mm512_set1_ps(f16_fast(sc[(size_t)i*scstr+scoff]));
    const __m128i lo=_mm_sub_epi8(_mm_and_si128(q,m4),m8);
    const __m128i hi=_mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(q,4),m4),m8);
    const __m512 wl=_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo)),d);
    const __m512 wh=_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi)),d);
    const float *y=X+(size_t)i*QK4;
    for(int b=0;b<B;b++,y+=xs){
      acc[b]=_mm512_fmadd_ps(wl,_mm512_loadu_ps(y),acc[b]);
      acc[b]=_mm512_fmadd_ps(wh,_mm512_loadu_ps(y+16),acc[b]);
    }
  }
  for(int b=0;b<B;b++) Y[(size_t)b*ys]=_mm512_reduce_add_ps(acc[b]);
}
#endif

/* Batched gemv: Y[b*ys+j] = row j of W dotted against activation column b.
   The weight row stays hot across all B dots, so a verify block costs one
   weight sweep instead of B; activations (KBs) live in cache. Row-pair fast
   paths mirror gemv (shared activation loads); same OMP shape. */
static void __attribute__((hot))
gemvb(const Tensor *Wt, const float *X, int xs, float *Y, int ys, int B) {
  int M=Wt->ne1,K=Wt->ne0;
  if(Wt->type==T_Q4_0RS){
    int nb=(int)q4r_nb(K);
    const uint8_t *qs=(const uint8_t*)Wt->data;
    const uint16_t *sc=(const uint16_t*)Wt->scales;
    size_t rqs=(size_t)nb*16,rsc=(size_t)nb;
    int pfd=CWEN_PREFETCH>0?CWEN_PREFETCH:4;
#if defined(CWEN_AVX512) && !defined(CWEN_NO_BCOL)
    if(B>=2&&B<=Q4RS_BCOL_MAX){
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for(int i=0;i<M;i++){
        if(i+pfd<M) cwen_pf_w(qs+(size_t)(i+pfd)*rqs);
        dot_q4_bcol(qs+(size_t)i*rqs,16,0,sc+(size_t)i*rsc,1,0,X,xs,nb,Y+i,ys,B);
      }
      return;
    }
#endif
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<M;i+=2){
      if(i+1<M){
        if(i+pfd+1<M){
          cwen_pf_w(qs+(size_t)(i+pfd)*rqs);
          cwen_pf_w(qs+(size_t)(i+pfd+1)*rqs);
        }
        const uint8_t *qa=qs+(size_t)i*rqs,*qb=qs+(size_t)(i+1)*rqs;
        const uint16_t *sa=sc+(size_t)i*rsc,*sb=sc+(size_t)(i+1)*rsc;
        const float *xb=X;
        for(int b=0;b<B;b++,xb+=xs)
          dot_q4_0rs_2row(qa,sa,qb,sb,xb,K,
                          Y+(size_t)b*ys+i,Y+(size_t)b*ys+i+1);
      }else{
        const float *xb=X;
        for(int b=0;b<B;b++,xb+=xs)
          Y[(size_t)b*ys+i]=dot_q4_0rs(qs+(size_t)i*rqs,sc+(size_t)i*rsc,xb,K);
      }
    }
    return;
  }
  if(Wt->type==T_Q4_1){
    const block_q4_1 *base=(const block_q4_1*)Wt->data;
    int nblk=K/QK4;
    size_t rb=row_bytes(T_Q4_1,K);
    int pfd=CWEN_PREFETCH>0?CWEN_PREFETCH:4;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<M;i+=2){
      if(i+1<M){
        if(i+pfd+1<M){
          cwen_pf_w((const char*)base+(size_t)(i+pfd)*rb);
          cwen_pf_w((const char*)base+(size_t)(i+pfd+1)*rb);
        }
        const block_q4_1 *ba=base+(size_t)i*nblk,*bb2=base+(size_t)(i+1)*nblk;
        const float *xb=X;
        for(int b=0;b<B;b++,xb+=xs)
          dot_q4_1_2row(ba,bb2,xb,K,Y+(size_t)b*ys+i,Y+(size_t)b*ys+i+1);
      }else{
        const float *xb=X;
        for(int b=0;b<B;b++,xb+=xs)
          Y[(size_t)b*ys+i]=dot_q4_1(base+(size_t)i*nblk,xb,K);
      }
    }
    return;
  }
#if defined(CWEN_AVX512) && !defined(CWEN_NO_BCOL)
  if(Wt->type==T_Q4_0RSI&&B>=2&&B<=Q4RS_BCOL_MAX){
    int nb=(int)q4r_nb(K);
    const uint8_t *qs=(const uint8_t*)Wt->data;
    const uint16_t *sc=(const uint16_t*)Wt->scales;
    size_t rqs=q4rsi_row_qs(K), rsc=(size_t)nb*2;
    int side=Wt->pair_side;
    int pf2=CWEN_PREFETCH>0?CWEN_PREFETCH:4;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<M;i++){
      if(i+pf2<M) cwen_pf_w(qs+(size_t)(i+pf2)*rqs);
      dot_q4_bcol(qs+(size_t)i*rqs,32,(size_t)side*16,
                  sc+(size_t)i*rsc,2,(size_t)side,X,xs,nb,Y+i,ys,B);
    }
    return;
  }
#endif
  if(!gemv_use_omp(M,K)){
    for(int i=0;i<M;i++)
      for(int b=0;b<B;b++) Y[(size_t)b*ys+i]=gemv_row(Wt,X+(size_t)b*xs,i);
    return;
  }
  int pfd=CWEN_PREFETCH>0?CWEN_PREFETCH:4;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(int i=0;i<M;i++){
    gemv_pf_row(Wt,i,pfd,M);
    const float *xb=X;
    for(int b=0;b<B;b++,xb+=xs) Y[(size_t)b*ys+i]=gemv_row(Wt,xb,i);
  }
}

/* residual: x += delta  (static inline, globals or pointers) */
static inline void residual_add(float *dst, const float *delta, int n) {
#if defined(CWEN_AVX512)
  for(int i=0;i<n;i+=16)
    store_pad16(dst,n,i,_mm512_add_ps(load_pad16(dst,n,i),load_pad16(delta,n,i)));
#elif defined(__AVX2__)
  int i=0;
  for(;i+8<=n;i+=8)
    _mm256_storeu_ps(dst+i,_mm256_add_ps(_mm256_loadu_ps(dst+i),_mm256_loadu_ps(delta+i)));
  for(;i<n;i++) dst[i]+=delta[i];
#else
  for(int i=0;i<n;i++) dst[i]+=delta[i];
#endif
}
/* silu(a)*b into a, serial by default (NO_SILU_OMP=1): GOMP fork > elementwise work. */
static inline void silu_mul(float *a, const float *b, int n) {
#if !CWEN_IDEA_NO_SILU_OMP
  if(n>=1024){
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<n;i++) a[i]=silu_one(a[i])*b[i];
    return;
  }
#endif
#if defined(CWEN_AVX512)
  for(int i=0;i<n;i+=16)
    store_pad16(a,n,i,_mm512_mul_ps(silu512(load_pad16(a,n,i)),load_pad16(b,n,i)));
#else
  for(int i=0;i<n;i++) a[i]=silu_one(a[i])*b[i];
#endif
}

/* Two gemvs, one OMP team (nowait between) - fewer GOMP_parallel than two gemv().
   When both Q4_0RS and same K: dual-mat on overlapping rows (shared x, one pass). */
static void gemv_pair(const Tensor *Wa, const Tensor *Wb,
                      const float *restrict x, float *restrict ya, float *restrict yb) {
  int Ma=Wa->ne1, Mb=Wb->ne1, Ka=Wa->ne0, Kb=Wb->ne0;
  int pfd=CWEN_PREFETCH>0?CWEN_PREFETCH:4;
  /* Q4RS same-K: dual-mat on overlapping rows (2 W streams, shared x). */
  if(Wa->type==T_Q4_0RS && Wb->type==T_Q4_0RS && Ka==Kb){
    int K=Ka, Mmin=Ma<Mb?Ma:Mb;
    int nb=(int)q4r_nb(K); size_t rqs=(size_t)nb*16, rsc=(size_t)nb;
    const uint8_t *qa=(const uint8_t*)Wa->data, *qb=(const uint8_t*)Wb->data;
    const uint16_t *sa=(const uint16_t*)Wa->scales, *sb=(const uint16_t*)Wb->scales;
    int use=gemv_use_omp(Ma,K)||gemv_use_omp(Mb,K);
    if(!use){
      for(int i=0;i<Mmin;i++)
        dot_q4_0rs_2mat(qa+(size_t)i*rqs,sa+(size_t)i*rsc,
                        qb+(size_t)i*rqs,sb+(size_t)i*rsc,x,K,ya+i,yb+i);
      for(int j=Mmin;j<Ma;j++) ya[j]=dot_q4_0rs(qa+(size_t)j*rqs,sa+(size_t)j*rsc,x,K);
      for(int j=Mmin;j<Mb;j++) yb[j]=dot_q4_0rs(qb+(size_t)j*rqs,sb+(size_t)j*rsc,x,K);
      return;
    }
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
#pragma omp for schedule(static) nowait
#endif
      for(int i=0;i<Mmin;i++){
        if(i+pfd<Mmin){
          cwen_pf_w(qa+(size_t)(i+pfd)*rqs);
          cwen_pf_w(qb+(size_t)(i+pfd)*rqs);
        }
        dot_q4_0rs_2mat(qa+(size_t)i*rqs,sa+(size_t)i*rsc,
                        qb+(size_t)i*rqs,sb+(size_t)i*rsc,x,K,ya+i,yb+i);
      }
#ifdef _OPENMP
#pragma omp for schedule(static) nowait
#endif
      for(int j=Mmin;j<Ma;j++){
        if(j+pfd<Ma) cwen_pf_w(qa+(size_t)(j+pfd)*rqs);
        ya[j]=dot_q4_0rs(qa+(size_t)j*rqs,sa+(size_t)j*rsc,x,K);
      }
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
      for(int j=Mmin;j<Mb;j++){
        if(j+pfd<Mb) cwen_pf_w(qb+(size_t)(j+pfd)*rqs);
        yb[j]=dot_q4_0rs(qb+(size_t)j*rqs,sb+(size_t)j*rsc,x,K);
      }
    }
    return;
  }
  int use=gemv_use_omp(Ma,Ka)||gemv_use_omp(Mb,Kb);
  if(!use){
    for(int i=0;i<Ma;i++) ya[i]=gemv_row(Wa,x,i);
    for(int i=0;i<Mb;i++) yb[i]=gemv_row(Wb,x,i);
    return;
  }
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifdef _OPENMP
#pragma omp for schedule(static) nowait
#endif
    for(int i=0;i<Ma;i++){
      gemv_pf_row(Wa,i,pfd,Ma);
      ya[i]=gemv_row(Wa,x,i);
    }
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for(int i=0;i<Mb;i++){
      gemv_pf_row(Wb,i,pfd,Mb);
      yb[i]=gemv_row(Wb,x,i);
    }
  }
}

/* Two same-shape mats over one activation, row-locked: an interleaved CWENR v4
   pair (single data blob, A|B halves) or two split Q4_0RS streams. */
static inline int dual_mat_ok(const Tensor *Wa, const Tensor *Wb) {
  return (Wa->type==T_Q4_0RSI && Wb->type==T_Q4_0RSI && Wa->data==Wb->data) ||
         (Wa->type==T_Q4_0RS && Wb->type==T_Q4_0RS);
}
static void __attribute__((hot))
gemv_dual_same_shape(const Tensor *Wa, const Tensor *Wb,
                     const float *restrict x, float *restrict ya, float *restrict yb) {
  int M=Wa->ne1, K=Wa->ne0;
  int il=(Wa->type==T_Q4_0RSI);
  int nb=(int)q4r_nb(K);
  size_t rqs=il?q4rsi_row_qs(K):((size_t)nb*16);
  size_t rsc=il?((size_t)nb*2):(size_t)nb;
  const uint8_t *qa=(const uint8_t*)Wa->data, *qb=il?NULL:(const uint8_t*)Wb->data;
  const uint16_t *sa=(const uint16_t*)Wa->scales, *sb=il?NULL:(const uint16_t*)Wb->scales;
  if(!gemv_use_omp(M,K)){
    for(int i=0;i<M;i++){
      if(il) dot_q4_0rsi_2mat(qa+(size_t)i*rqs,sa+(size_t)i*rsc,x,K,ya+i,yb+i);
      else   dot_q4_0rs_2mat(qa+(size_t)i*rqs,sa+(size_t)i*rsc,
                             qb+(size_t)i*rqs,sb+(size_t)i*rsc,x,K,ya+i,yb+i);
    }
    return;
  }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(int i=0;i<M;i++){
    if(il) dot_q4_0rsi_2mat(qa+(size_t)i*rqs,sa+(size_t)i*rsc,x,K,ya+i,yb+i);
    else   dot_q4_0rs_2mat(qa+(size_t)i*rqs,sa+(size_t)i*rsc,
                           qb+(size_t)i*rqs,sb+(size_t)i*rsc,x,K,ya+i,yb+i);
  }
}

/* embed: token_embd [H, V] Q4_0 / Q4_0R / Q4_0RS / Q6_K; row = token id */
static void embed_q6_k(const block_q6_K *b, float *out) {
  int nb = H / QK_K;
  for (int i = 0; i < nb; i++) {
    float d = f16_fast(b[i].d);
    const uint8_t *ql = b[i].ql, *qh = b[i].qh;
    const int8_t *sc = b[i].scales;
    float *o = out + (size_t)i * QK_K;
    for (int n0 = 0; n0 < QK_K; n0 += 128) {
      for (int l = 0; l < 32; l++) {
        int is = l / 16;
        o[n0 + l]      = d * sc[is + 0] * ((int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
        o[n0 + l + 32] = d * sc[is + 2] * ((int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
        o[n0 + l + 64] = d * sc[is + 4] * ((int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
        o[n0 + l + 96] = d * sc[is + 6] * ((int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
      }
      ql += 64; qh += 32; sc += 8;
    }
  }
}
static void embed_token_to(int token, float *dst) {
  /* token ids are untrusted input (prompt file / server frame): hard check,
     not assert, so no -DNDEBUG build turns this into an OOB read */
  if(token<0||token>=V){ fprintf(stderr,"token id %d out of range [0,%d)\n",token,V); exit(1); }
  size_t rb=row_bytes(tok_embd.type, tok_embd.ne0);
  const void *row=(const char*)tok_embd.data+(size_t)token*rb;
  int nb=H/QK4;
  if(tok_embd.type==T_Q4_0RS){
    const uint8_t *q=(const uint8_t*)row;
    const uint16_t *sc=(const uint16_t*)tok_embd.scales+(size_t)token*nb;
    for(int i=0;i<nb;i++){
      float d=f16_fast(sc[i]); const uint8_t *qi=q+(size_t)i*16;
      for(int j=0;j<16;j++){
        dst[i*QK4+j]=(float)((qi[j]&0xF)-8)*d;
        dst[i*QK4+j+16]=(float)((qi[j]>>4)-8)*d;
      }
    }
  } else if(tok_embd.type==T_Q4_0R){
    const block_q4_0r *b=row;
    for(int i=0;i<nb;i++){
      float d=b[i].d; const uint8_t *qi=b[i].qs;
      for(int j=0;j<16;j++){
        dst[i*QK4+j]=(float)((qi[j]&0xF)-8)*d;
        dst[i*QK4+j+16]=(float)((qi[j]>>4)-8)*d;
      }
    }
  } else if (tok_embd.type==T_Q4_0) {
    const block_q4_0 *b=row;
    for(int i=0;i<nb;i++) {
      float d=f16_to_f32(b[i].d);
      for(int j=0;j<QK4/2;j++) {
        dst[i*QK4+j]=(float)((b[i].qs[j]&0xF)-8)*d;
        dst[i*QK4+j+QK4/2]=(float)((b[i].qs[j]>>4)-8)*d;
      }
    }
  } else if (tok_embd.type==T_Q6_K) {
    embed_q6_k((const block_q6_K*)row, dst);
  } else { fprintf(stderr,"embed type %d\n",tok_embd.type); exit(1); }
}

/* ---- RoPE (partial, interleaved mRoPE text: same pos all sections) ----
   Angles depend only on (pos, idx), yet every head of every full-attn layer
   recomputed powf+sinf+cosf per token. Tables are filled once per position
   (identical inputs -> bit-identical results) and reused by all heads/layers. */
static float Rope_cos[MAX_SEQ][NROT] __attribute__((aligned(64)));
static float Rope_sin[MAX_SEQ][NROT] __attribute__((aligned(64)));
static uint8_t Rope_done[MAX_SEQ];
/* CWEN_ROPE_YARN="orig_max,factor[,beta_fast,beta_slow]" applies HF-style
   YaRN on top of this file's inv(p)=theta^(-p/NROT) parameterization
   (transformers modeling_rope_utils._compute_yarn_parameters): low pair
   indices extrapolate, high indices interpolate by 1/factor, smoothed ramp
   between the correction dimensions, cos/sin scaled by
   get_mscale(factor)=0.1*ln(factor)+1. Off by default; the checkpoint
   declares plain rope at 262144. */
static void rope_tables(int pos) {
  if(Rope_done[pos]) return;
  int idx=0;
  for(int sec=0;sec<3;sec++) {
    int n=ROPE_SEC[sec];
    for(int p=0;p<n;p++) {
      float freq = 1.f/powf(ROPE_THETA, (float)(idx)/(float)NROT);
      float mscale=1.f;
      if(g_yarn_on){
        double inv_e=1.0/pow(ROPE_THETA,(double)(idx)/(double)NROT);
        double inv_i=inv_e/Yarn_factor;
        /* correction dims in HF units (their pair index i_h = idx/2) */
        double cd_fast=0.5*(double)NROT*log(Yarn_orig_max/(Yarn_beta_fast*2.0*M_PI))/log(ROPE_THETA);
        double cd_slow=0.5*(double)NROT*log(Yarn_orig_max/(Yarn_beta_slow*2.0*M_PI))/log(ROPE_THETA);
        double lo_floor=floor(cd_fast), hi_ceil=ceil(cd_slow);
        if(lo_floor<0) lo_floor=0;
        if(hi_ceil>(double)NROT-1) hi_ceil=NROT-1;
        double ih=(double)idx*0.5;
        double ramp=(ih-lo_floor)/(hi_ceil-lo_floor);
        if(ramp<0) ramp=0;
        if(ramp>1) ramp=1;
        freq=(float)(inv_i*ramp+inv_e*(1.0-ramp));
        mscale=Yarn_factor>1.0?(float)(0.1*log(Yarn_factor)+1.0):1.f;
      }
      float ang=(float)pos*freq;
      Rope_cos[pos][idx]=cosf(ang)*mscale;
      Rope_sin[pos][idx]=sinf(ang)*mscale;
      idx++;
    }
  }
  Rope_done[pos]=1;
}
static void rope_apply(float *q, int n_heads, int pos) {
  rope_tables(pos);
  const float *C=Rope_cos[pos], *S=Rope_sin[pos];
  int npairs=ROPE_SEC[0]+ROPE_SEC[1]+ROPE_SEC[2]; /* 32; i1<NROT always holds */
  for(int h=0;h<n_heads;h++) {
    float *v=q+h*HD;
    for(int idx=0;idx<npairs;idx++) {
      float c=C[idx], s=S[idx];
      int i0=idx*2, i1=i0+1;
      float a=v[i0], b=v[i1];
      v[i0]=a*c - b*s; v[i1]=a*s + b*c;
    }
  }
}

static void touch_span(const void *p, size_t n, size_t stride) {
  if(!p||!n||!stride) return;
  const volatile char *c=(const volatile char*)p; volatile char s=0;
  for(size_t i=0;i<n;i+=stride) s^=c[i];
  (void)s;
}
/* File-backed weight map. warm=1 faults whole range; warm=0 only mmaps (used when
   CWENR will replace Q4_0 and we touch only live tensors afterward). */
static void *mmap_resident_ex(int fd, size_t len, const char *tag, int warm) {
  int fl=MAP_PRIVATE;
#ifdef MAP_POPULATE
  if(warm) fl|=MAP_POPULATE;
#endif
  void *p=mmap(NULL,len,PROT_READ,fl,fd,0);
  if(p==MAP_FAILED) return MAP_FAILED;
#ifdef MADV_HUGEPAGE
  madvise(p,len,MADV_HUGEPAGE);
#endif
  if(warm){
#ifdef MADV_WILLNEED
    madvise(p,len,MADV_WILLNEED);
#endif
#if defined(MADV_POPULATE_READ)
    /* faults + reads every page: the sweep below would only re-read what is
       already resident (a second multi-GiB pass at startup) */
    madvise(p,len,MADV_POPULATE_READ);
#else
    touch_span(p,len,4096);
#endif
    fprintf(stderr,"%s: resident %zu MiB in RAM\n",tag,len>>20);
  } else {
    fprintf(stderr,"%s: mmap %zu MiB (deferred warm; CWENR path)\n",tag,len>>20);
  }
  return p;
}
static void *mmap_resident(int fd, size_t len, const char *tag) {
  return mmap_resident_ex(fd,len,tag,1);
}
/* Page-in tensors still backed by Gmap (norms, Q5/Q6, F32, etc.). */
static void warm_live_gguf_tensors(void) {
  size_t n=0;
  for(int i=0;i<Ntens;i++){
    if(!Tens[i].data) continue;
    const char *p=(const char*)Tens[i].data;
    if(p<(const char*)Gmap || p>=(const char*)Gmap+Gmap_len) continue;
    size_t nb=row_bytes(Tens[i].type,Tens[i].ne0)*(size_t)Tens[i].ne1;
    if((Tens[i].type==T_Q4_0RS || Tens[i].type==T_Q4_0RSI) && Tens[i].scales){
      /* qs is in cwenr; already rebound off Gmap */
      continue;
    }
    touch_span(p, nb, 4096);
    n+=nb;
  }
  fprintf(stderr,"gguf: warm live tensors %zu MiB\n",n>>20);
}
/* Only called under CWEN_IDEA_MADV_SEQ / CWEN_IDEA_MADVISE experiments. */
static void __attribute__((unused)) madvise_span(const Tensor *t, int advice) {
  if(!t||!t->data) return;
  size_t n=row_bytes(t->type,t->ne0)*(size_t)t->ne1;
  uintptr_t p=(uintptr_t)t->data, page=p&~(uintptr_t)4095;
  madvise((void*)page, n+(p-page), advice);
}
/* ---- GDN recurrent step (one token) ---- */
/* S[h][i][j] : head h, dim i (k), dim j (v); matches llama AR [Sv,Sv,H] with row=k */
static void gdn_step(int layer, float *qkv_mixed, float *z_row,
                     float *decay, float *beta, float *out) {
  float *S = Srec + (size_t)layer*LVH*LSD*LSD;
  float scale = 1.f/sqrtf((float)LSD);
  float *q0=qkv_mixed, *k0=qkv_mixed+LKH*LSD, *v0=qkv_mixed+2*LKH*LSD;
  l2norm_rows(q0, LKH, LSD);
  l2norm_rows(k0, LKH, LSD);
#if CWEN_IDEA_GDN_OMP
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
#endif
  for(int hv=0;hv<LVH;hv++) {
    int hk = hv % LKH;
    float *q=q0+hk*LSD, *k=k0+hk*LSD, *v=v0+hv*LSD;
    float *Sh = S + (size_t)hv*LSD*LSD;
    float g = expf(decay[hv]);
    float beta_h = beta[hv];
#if defined(CWEN_AVX512)
    {
      __m512 vg=_mm512_set1_ps(g);
      for(int i=0;i<LSD*LSD;i+=16)
        _mm512_storeu_ps(Sh+i,_mm512_mul_ps(_mm512_loadu_ps(Sh+i),vg));
    }
#elif defined(__AVX2__)
    __m256 vg=_mm256_set1_ps(g);
    for(int i=0;i<LSD*LSD;i+=8) _mm256_storeu_ps(Sh+i,_mm256_mul_ps(_mm256_loadu_ps(Sh+i),vg));
#else
    for(int i=0;i<LSD*LSD;i++) Sh[i]*=g;
#endif
    float sk[LSD] __attribute__((aligned(64)));
    /* sk[j] = sum_i S[i,j]*k[i], a column matvec */
#if defined(CWEN_AVX512)
    for(int j=0;j<LSD;j+=16){
      __m512 s=_mm512_setzero_ps();
      int i=0;
      for(;i+1<LSD;i+=2){
        s=_mm512_fmadd_ps(_mm512_loadu_ps(Sh+i*LSD+j),_mm512_set1_ps(k[i]),s);
        s=_mm512_fmadd_ps(_mm512_loadu_ps(Sh+(i+1)*LSD+j),_mm512_set1_ps(k[i+1]),s);
      }
      for(;i<LSD;i++)
        s=_mm512_fmadd_ps(_mm512_loadu_ps(Sh+i*LSD+j),_mm512_set1_ps(k[i]),s);
      _mm512_store_ps(sk+j,s);
    }
#elif defined(__AVX2__)
    for(int j=0;j<LSD;j+=8){
      __m256 s=_mm256_setzero_ps();
      for(int i=0;i<LSD;i++)
        s=_mm256_fmadd_ps(_mm256_loadu_ps(Sh+i*LSD+j),_mm256_set1_ps(k[i]),s);
      _mm256_store_ps(sk+j,s);
    }
#else
    for(int j=0;j<LSD;j++) {
      float s=0; for(int i=0;i<LSD;i++) s+=Sh[i*LSD+j]*k[i];
      sk[j]=s;
    }
#endif
    float delta[LSD] __attribute__((aligned(64)));
#if defined(CWEN_AVX512)
    {
      __m512 vb=_mm512_set1_ps(beta_h);
      for(int j=0;j<LSD;j+=16)
        _mm512_store_ps(delta+j,_mm512_mul_ps(vb,_mm512_sub_ps(_mm512_loadu_ps(v+j),_mm512_load_ps(sk+j))));
    }
#elif defined(__AVX2__)
    __m256 vb=_mm256_set1_ps(beta_h);
    for(int j=0;j<LSD;j+=8)
      _mm256_store_ps(delta+j,_mm256_mul_ps(vb,_mm256_sub_ps(_mm256_loadu_ps(v+j),_mm256_load_ps(sk+j))));
#else
    for(int j=0;j<LSD;j++) delta[j]=beta_h*(v[j]-sk[j]);
#endif
#if defined(CWEN_AVX512)
    {
      int i=0;
      for(;i+1<LSD;i+=2){
        __m512 vki0=_mm512_set1_ps(k[i]), vki1=_mm512_set1_ps(k[i+1]);
        for(int j=0;j<LSD;j+=16){
          __m512 dj=_mm512_load_ps(delta+j);
          _mm512_storeu_ps(Sh+i*LSD+j,
            _mm512_fmadd_ps(vki0,dj,_mm512_loadu_ps(Sh+i*LSD+j)));
          _mm512_storeu_ps(Sh+(i+1)*LSD+j,
            _mm512_fmadd_ps(vki1,dj,_mm512_loadu_ps(Sh+(i+1)*LSD+j)));
        }
      }
      for(;i<LSD;i++){
        __m512 vki=_mm512_set1_ps(k[i]);
        for(int j=0;j<LSD;j+=16)
          _mm512_storeu_ps(Sh+i*LSD+j,
            _mm512_fmadd_ps(vki,_mm512_load_ps(delta+j),_mm512_loadu_ps(Sh+i*LSD+j)));
      }
    }
#else
    for(int i=0;i<LSD;i++){
      float ki=k[i];
#if defined(__AVX2__)
      __m256 vki=_mm256_set1_ps(ki);
      for(int j=0;j<LSD;j+=8)
        _mm256_storeu_ps(Sh+i*LSD+j,
          _mm256_fmadd_ps(vki,_mm256_load_ps(delta+j),_mm256_loadu_ps(Sh+i*LSD+j)));
#else
      for(int j=0;j<LSD;j++) Sh[i*LSD+j]+=ki*delta[j];
#endif
    }
#endif
    float *o=out+hv*LSD;
#if defined(CWEN_AVX512)
    {
      __m512 vsc=_mm512_set1_ps(scale);
      for(int j=0;j<LSD;j+=16){
        __m512 s=_mm512_setzero_ps();
        int i=0;
        for(;i+1<LSD;i+=2){
          s=_mm512_fmadd_ps(_mm512_loadu_ps(Sh+i*LSD+j),_mm512_set1_ps(q[i]),s);
          s=_mm512_fmadd_ps(_mm512_loadu_ps(Sh+(i+1)*LSD+j),_mm512_set1_ps(q[i+1]),s);
        }
        for(;i<LSD;i++)
          s=_mm512_fmadd_ps(_mm512_loadu_ps(Sh+i*LSD+j),_mm512_set1_ps(q[i]),s);
        _mm512_storeu_ps(o+j,_mm512_mul_ps(s,vsc));
      }
    }
#elif defined(__AVX2__)
    __m256 vsc=_mm256_set1_ps(scale);
    for(int j=0;j<LSD;j+=8){
      __m256 s=_mm256_setzero_ps();
      for(int i=0;i<LSD;i++)
        s=_mm256_fmadd_ps(_mm256_loadu_ps(Sh+i*LSD+j),_mm256_set1_ps(q[i]),s);
      _mm256_storeu_ps(o+j,_mm256_mul_ps(s,vsc));
    }
#else
    for(int j=0;j<LSD;j++) {
      float s=0; for(int i=0;i<LSD;i++) s+=Sh[i*LSD+j]*q[i];
      o[j]=s*scale;
    }
#endif
  }
  const float *wn=(const float*)W[layer].ssm_norm.data;
  for(int hv=0;hv<LVH;hv++) {
    float *o=out+hv*LSD; float *z=z_row+hv*LSD;
#if defined(CWEN_AVX512)
    {
      __m512 vss=_mm512_setzero_ps();
      for(int i=0;i<LSD;i+=16){
        __m512 vo=_mm512_loadu_ps(o+i);
        vss=_mm512_fmadd_ps(vo,vo,vss);
      }
      float ss=_mm512_reduce_add_ps(vss);
      float s=1.f/sqrtf(ss/LSD+RMS_EPS);
      __m512 vs=_mm512_set1_ps(s);
      for(int i=0;i<LSD;i+=16){
        __m512 vo=_mm512_loadu_ps(o+i);
        __m512 vz=_mm512_loadu_ps(z+i);
        __m512 vw=_mm512_loadu_ps(wn+i);
        __m512 gz=silu512(vz);
        _mm512_storeu_ps(o+i,_mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(vw,vo),vs),gz));
      }
    }
#else
    float ss=0; for(int i=0;i<LSD;i++) ss+=o[i]*o[i];
    float s=1.f/sqrtf(ss/LSD+RMS_EPS);
    for(int i=0;i<LSD;i++) {
      float gz=z[i]*(1.f/(1.f+expf(-z[i])));
      o[i]=wn[i]*o[i]*s*gz;
    }
#endif
  }
  /* ssm_out gemv is outside: must not call gemv from inside omp single */
}

/* depthwise conv update: mixed QKV row in qkv_mixed, conv state, kernel [CONV_K, QKV_DIM] F32 */
static void conv1d_update(int layer, float *qkv_mixed) {
  float *cs = Cstate + (size_t)layer*QKV_DIM*(CONV_K-1);
  const float *kw = (const float*)W[layer].ssm_conv.data;
#if CWEN_IDEA_CONV_OMP
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
#endif
  /* in-place is safe: channel c reads only st[c] and qkv_mixed[c], never a neighbor */
  for(int c=0;c<QKV_DIM;c++) {
    const float *w=kw+c*CONV_K;
    float *st=cs+c*(CONV_K-1);
    float cur=qkv_mixed[c];
    qkv_mixed[c]=st[0]*w[0]+st[1]*w[1]+st[2]*w[2]+cur*w[3];
    st[0]=st[1]; st[1]=st[2]; st[2]=cur;
  }
  silu_vec(qkv_mixed, QKV_DIM);
}

/* ---- layers ----
   Fuse shared-x gemvs by one OMP loop calling static inline gemv_row twice.
   Keep separate logical steps; no gemv2 API (see AGENTS.md). */
static void mlp(int layer) {
  const Tensor *Wg=&W[layer].ffn_gate, *Wu=&W[layer].ffn_up, *Wd=&W[layer].ffn_down;
#if CWEN_IDEA_SERIAL_MLP
  gemv(Wg, xb, hb); gemv(Wu, xb, hb2);
  silu_mul(hb, hb2, I); gemv(Wd, hb, xb2); residual_add(x, xb2, H); return;
#else
  int M=Wg->ne1, K=Wg->ne0;
  int pfd=CWEN_PREFETCH>0?CWEN_PREFETCH:4;
  /* One OMP team: gate+up dual-mat → silu → down (one GOMP fork, not three). */
  int gate_up_ok = (Wg->ne0==Wu->ne0 && Wg->ne1==Wu->ne1) && dual_mat_ok(Wg,Wu);
  if(gate_up_ok){
    int nb=(int)q4r_nb(K);
    int Md=Wd->ne1, Kd=Wd->ne0;
    int use=gemv_use_omp(M,K)||gemv_use_omp(Md,Kd);
    int il=(Wg->type==T_Q4_0RSI);
    size_t rqs=il?q4rsi_row_qs(K):((size_t)nb*16);
    size_t rsc=il?((size_t)nb*2):(size_t)nb;
    const uint8_t *qg=(const uint8_t*)Wg->data, *qu=il?NULL:(const uint8_t*)Wu->data;
    const uint16_t *sg=(const uint16_t*)Wg->scales, *su=il?NULL:(const uint16_t*)Wu->scales;
    if(!use){
      gemv_dual_same_shape(Wg,Wu,xb,hb,hb2);
      silu_mul(hb, hb2, I);
      gemv(Wd, hb, xb2); residual_add(x, xb2, H); return;
    }
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
      for(int i=0;i<M;i++){
        if(i+pfd<M){
          cwen_pf_w(qg+(size_t)(i+pfd)*rqs);
          if(!il) cwen_pf_w(qu+(size_t)(i+pfd)*rqs);
        }
        if(il)
          dot_q4_0rsi_2mat(qg+(size_t)i*rqs, sg+(size_t)i*rsc, xb, K, hb+i, hb2+i);
        else
          dot_q4_0rs_2mat(qg+(size_t)i*rqs, sg+(size_t)i*rsc,
                          qu+(size_t)i*rqs, su+(size_t)i*rsc, xb, K, hb+i, hb2+i);
      }
#ifdef _OPENMP
#pragma omp single
#endif
      { for(int i=0;i<I;i++) hb[i]=silu_one(hb[i])*hb2[i]; }
      if(Wd->type==T_Q4_0RS){
        int nbd=(int)q4r_nb(Kd); size_t rqsd=(size_t)nbd*16, rscd=(size_t)nbd;
        const uint8_t *qd=(const uint8_t*)Wd->data;
        const uint16_t *sd=(const uint16_t*)Wd->scales;
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for(int i=0;i<Md;i+=2){
          if(i+1<Md)
            dot_q4_0rs_2row(qd+(size_t)i*rqsd,sd+(size_t)i*rscd,
                            qd+(size_t)(i+1)*rqsd,sd+(size_t)(i+1)*rscd,
                            hb,Kd,xb2+i,xb2+i+1);
          else
            xb2[i]=dot_q4_0rs(qd+(size_t)i*rqsd,sd+(size_t)i*rscd,hb,Kd);
        }
      } else if(Wd->type==T_Q4_1){
        const block_q4_1 *bd=(const block_q4_1*)Wd->data;
        int nblk=Kd/QK4;
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for(int i=0;i<Md;i+=2){
          if(i+1<Md)
            dot_q4_1_2row(bd+(size_t)i*nblk, bd+(size_t)(i+1)*nblk, hb, Kd, xb2+i, xb2+i+1);
          else
            xb2[i]=dot_q4_1(bd+(size_t)i*nblk, hb, Kd);
        }
      } else {
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for(int i=0;i<Md;i++){
          gemv_pf_row(Wd,i,pfd,Md);
          xb2[i]=gemv_row(Wd,hb,i);
        }
      }
    }
    residual_add(x, xb2, H); return;
  }
  if(!gemv_use_omp(M,K)){
    for(int i=0;i<M;i++){ hb[i]=gemv_row(Wg,xb,i); hb2[i]=gemv_row(Wu,xb,i); }
    silu_mul(hb, hb2, I); gemv(Wd, hb, xb2); residual_add(x, xb2, H); return;
  }
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for(int i=0;i<M;i++){
      gemv_pf_row(Wg,i,pfd,M);
#if !CWEN_IDEA_PF_ONE
      gemv_pf_row(Wu,i,pfd,M);
#endif
      hb[i]=gemv_row(Wg,xb,i); hb2[i]=gemv_row(Wu,xb,i);
    }
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for(int i=0;i<I;i++) hb[i]=silu_one(hb[i])*hb2[i];
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for(int i=0;i<Wd->ne1;i++){
      gemv_pf_row(Wd,i,pfd,Wd->ne1); xb2[i]=gemv_row(Wd,hb,i);
    }
  }
  residual_add(x, xb2, H); return;
#endif
}

static void layer_linear(int layer) {
  rmsnorm(xb, x, (const float*)W[layer].attn_norm.data, H);
#if CWEN_IDEA_MADV_SEQ
  madvise_span(&W[layer].qkv, MADV_SEQUENTIAL);
  madvise_span(&W[layer].ffn_gate, MADV_SEQUENTIAL);
#endif
#if CWEN_IDEA_PAIR_GEMV
  /* shared xb: one team, two for-loops (fewer GOMP_parallel) */
  gemv_pair(&W[layer].qkv, &W[layer].gate_z, xb, qkvb, zb);
#else
  gemv(&W[layer].qkv, xb, qkvb);
  gemv(&W[layer].gate_z, xb, zb);
#endif
  /* alpha/beta tiny (M=LVH); keep separate serial-friendly gemvs */
  gemv(&W[layer].ssm_alpha, xb, ab);
  gemv(&W[layer].ssm_beta, xb, bb);
  const float *dt=(const float*)W[layer].ssm_dt.data;
  const float *A=(const float*)W[layer].ssm_a.data;
  for(int i=0;i<LVH;i++) {
    ab[i]=A[i]*softplus(ab[i]+dt[i]);
    bb[i]=sigmoid(bb[i]);
  }
  conv1d_update(layer, qkvb);
  gdn_step(layer, qkvb, zb, ab, bb, hb);
  gemv(&W[layer].ssm_out, hb, xb2);
  residual_add(x, xb2, H);
  rmsnorm(xb, x, (const float*)W[layer].post_norm.data, H);
  mlp(layer);
}

/* causal GQA heads at sequence position pos: softmax over KV[0..pos],
   sigmoid-gated, output NH*HD into yout. Shared by serial and block paths.
   Heads are visited per KV group so every K/V row streams once per GROUP
   instead of once per head: kv_mul redundant passes over the KV cache were
   DRAM re-reads (the weight sweep between layers flushes L3). Per-head math
   is unchanged and bit-identical: same dot kernels, same softmax sequence,
   same t-ascending output accumulation. */
/* Gated GQA over an explicit K/V stream. t0 lets a cache whose slot 0 is
   never written (the nextn stream starts at slot 1) skip that row. */
static void attention_heads_kv(const float *Kb,const float *Vb,int t0,int pos,
                               const float *qin,const float *gatein,float *yout) {
  float scale=1.f/sqrtf((float)HD);
  const int kv_mul=NH/NKV;
  /* one softmax row per group member; rows stay L1-hot across t */
  static float ag[NH/NKV][MAX_SEQ] __attribute__((aligned(64)));
  assert(t0>=0&&t0<=pos); /* an empty window would softmax uninitialized scores */
  memset(yout,0,(size_t)NH*HD*sizeof(float));
  for(int hkv=0;hkv<NKV;hkv++) {
    /* score the whole group against one shared K stream */
    for(int t=t0;t<=pos;t++) {
      const float *k=Kb+((size_t)t*NKV+hkv)*HD;
      for(int m=0;m<kv_mul;m++)
        ag[m][t]=dot_f32(qin+(size_t)(hkv*kv_mul+m)*HD,k,HD)*scale;
    }
    for(int m=0;m<kv_mul;m++) {
      int h=hkv*kv_mul+m;
      float *at=ag[m];
      float mx=at[t0]; for(int t=t0+1;t<=pos;t++) if(at[t]>mx) mx=at[t];
      float sum=0; for(int t=t0;t<=pos;t++){ at[t]=expf(at[t]-mx); sum+=at[t]; }
      float inv=1.f/sum; for(int t=t0;t<=pos;t++) at[t]*=inv;
      float *o=yout+(size_t)h*HD;
      for(int t=t0;t<=pos;t++) {
        const float *vv=Vb+((size_t)t*NKV+hkv)*HD;
        float a=at[t];
#if defined(CWEN_AVX512)
        __m512 va=_mm512_set1_ps(a);
        for(int i=0;i<HD;i+=16)
          _mm512_storeu_ps(o+i,_mm512_fmadd_ps(va,_mm512_loadu_ps(vv+i),_mm512_loadu_ps(o+i)));
#else
        for(int i=0;i<HD;i++) o[i]+=a*vv[i];
#endif
      }
#if defined(CWEN_AVX512)
      {
        __m512 one=_mm512_set1_ps(1.f);
        for(int i=0;i<HD;i+=16){
          __m512 g=_mm512_loadu_ps(gatein+(size_t)h*HD+i);
          __m512 sig=_mm512_div_ps(one,_mm512_add_ps(one,exp512(_mm512_sub_ps(_mm512_setzero_ps(),g))));
          _mm512_storeu_ps(o+i,_mm512_mul_ps(_mm512_loadu_ps(o+i),sig));
        }
      }
#else
      for(int i=0;i<HD;i++) o[i]*=sigmoid(gatein[(size_t)h*HD+i]);
#endif
    }
  }
}

static void attention_heads(int layer,int pos,const float *qin,const float *gatein,float *yout) {
  attention_heads_kv(Kcache+(size_t)layer*CTX_STRIDE*NKV*HD,
                     Vcache+(size_t)layer*CTX_STRIDE*NKV*HD,
                     0,pos,qin,gatein,yout);
}

static void layer_full(int layer) {
  rmsnorm(xb, x, (const float*)W[layer].attn_norm.data, H);
  gemv(&W[layer].wq, xb, qfull_g);
  for(int h=0;h<NH;h++) {
    memcpy(qh+h*HD, qfull_g+h*HD*2, HD*sizeof(float));
    memcpy(gateq+h*HD, qfull_g+h*HD*2+HD, HD*sizeof(float));
  }
  /* wk + wv share xb → dual-mat when Q4_0RS */
  {
    const Tensor *Wk=&W[layer].wk, *Wv=&W[layer].wv;
    int M=Wk->ne1;
    if(dual_mat_ok(Wk,Wv)){
      gemv_dual_same_shape(Wk,Wv,xb,kh,vh);
    } else if(!gemv_use_omp(M,Wk->ne0)){
      for(int i=0;i<M;i++){ kh[i]=gemv_row(Wk,xb,i); vh[i]=gemv_row(Wv,xb,i); }
    } else {
      int pfd=CWEN_PREFETCH>0?CWEN_PREFETCH:4;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for(int i=0;i<M;i++){
        gemv_pf_row(Wk,i,pfd,M); gemv_pf_row(Wv,i,pfd,M);
        kh[i]=gemv_row(Wk,xb,i); vh[i]=gemv_row(Wv,xb,i);
      }
    }
  }
  const float *qn=(const float*)W[layer].q_norm.data;
  const float *kn=(const float*)W[layer].k_norm.data;
  for(int h=0;h<NH;h++) rmsnorm(qh+h*HD, qh+h*HD, qn, HD);
  for(int h=0;h<NKV;h++) rmsnorm(kh+h*HD, kh+h*HD, kn, HD);
  rope_apply(qh, NH, pos_n);
  rope_apply(kh, NKV, pos_n);
  float *kc=Kcache+((size_t)layer*CTX_STRIDE+pos_n)*NKV*HD;
  float *vc=Vcache+((size_t)layer*CTX_STRIDE+pos_n)*NKV*HD;
  memcpy(kc, kh, NKV*HD*sizeof(float));
  memcpy(vc, vh, NKV*HD*sizeof(float));
  attention_heads(layer, pos_n, qh, gateq, yatt_g);
  gemv(&W[layer].wo, yatt_g, xb2);
  residual_add(x, xb2, H);
  rmsnorm(xb, x, (const float*)W[layer].post_norm.data, H);
  mlp(layer);
}

/* dump float vector for e2e goldens (CWEN_DUMP=dir, CWEN_DUMP_LAYERS=n).
   Failures are fatal: these files are the artifacts downstream compares
   judge correctness by, so continuing would either fail far from the cause
   or match a stale dump left by an earlier run. */
static void dump_f32(const char *dir, const char *name, const float *v, int n) {
  if(!dir||!dir[0]) return;
  char path[512]; snprintf(path,sizeof path,"%s/%s",dir,name);
  FILE *f=fopen(path,"wb");
  if(!f){ fprintf(stderr,"cwen: CWEN_DUMP open failed\n"); perror(path); exit(1); }
  size_t wr=fwrite(v,4,(size_t)n,f);
  int bad=wr!=(size_t)n;
  bad|=fclose(f)!=0; /* close even after a short write: never strand the fd */
  if(bad){
    fprintf(stderr,"cwen: CWEN_DUMP: short write %s (%zu/%d floats;"
            " disk full?)\n",path,wr,n);
    exit(1);
  }
}
/* Strict integer env parse: unset or empty -> return 0 and leave *out alone.
   Set but not a whole integer -> named error and exit; silent atoi fallbacks
   here have meant wrong dump limits and 1-thread runs. */
static int env_int(const char *name, int *out) {
  const char *e=getenv(name);
  if(!e||!e[0]) return 0;
  char *end; long v=strtol(e,&end,10);
  if(end==e||*end||v<INT_MIN||v>INT_MAX){
    fprintf(stderr,"cwen: %s=%s is not an integer\n",name,e);
    exit(1);
  }
  *out=(int)v;
  return 1;
}
/* Strict [lo,hi] CLI integer parse; exits 2 naming the argument, like every
   bad CLI argument. Shared by the bench and production mains so bounds and
   wording cannot drift apart. */
static int arg_int_range(const char *arg,long lo,long hi,
                         const char *prefix,const char *name){
  char *end;
  long v=strtol(arg,&end,10);
  if(end==arg||*end||v<lo||v>hi){
    fprintf(stderr,"%s%s must be an integer in [%ld,%ld], got '%s'\n",
            prefix,name,lo,hi,arg);
    exit(2);
  }
  return (int)v;
}
/* Strict boolean env parse: unset or empty -> return 0 and leave *out alone.
   Only "0"/"1" are accepted, matching CWEN_SPEC; presence-style truthiness
   (any value = on) made CWEN_SERVER=0 start the stdin server and
   CWEN_DUMP_LOGITS=0 emit logits dumps. */
static int env_bool(const char *name, int *out) {
  const char *e=getenv(name);
  if(!e||!e[0]) return 0;
  if(strcmp(e,"0")&&strcmp(e,"1")){
    fprintf(stderr,"cwen: %s must be 0 or 1\n",name);
    exit(1);
  }
  *out=(e[0]=='1');
  return 1;
}
/* Env is fixed for the process lifetime; these run once, not per token. */
static const char *dump_dir(void) {
  static const char *d; static int done;
  if(!done){ d=getenv("CWEN_DUMP"); done=1; }
  return d;
}
/* Startup check for CWEN_DUMP, same policy as every other env knob: a bad
   value must fail before the model load. Left to the first dump write it
   would cost a full load + prefill before dying on a typo'd directory. */
static void dump_dir_preflight(void) {
  const char *d=dump_dir();
  if(!d||!d[0]) return;
  struct stat st;
  if(stat(d,&st)||!S_ISDIR(st.st_mode)){
    fprintf(stderr,"cwen: CWEN_DUMP=%s is not a directory\n",d);
    exit(1);
  }
  if(access(d,W_OK|X_OK)){
    fprintf(stderr,"cwen: CWEN_DUMP=%s is not writable\n",d);
    exit(1);
  }
}
static int dump_layers_lim(void) {
  static int lim; static int done;
  if(!done){
    lim=L;
    if(env_int("CWEN_DUMP_LAYERS",&lim)){
      if(lim<0){
        fprintf(stderr,"cwen: CWEN_DUMP_LAYERS must be >=0\n");
        exit(1);
      }
      if(lim==0) lim=L; /* 0 = all layers, same convention as e2e_ref.py --layers */
    }
    done=1;
  }
  return lim;
}
static int dump_logits_flag(void) {
  static int on; static int done;
  if(!done){ env_bool("CWEN_DUMP_LOGITS",&on); done=1; }
  return on;
}

#if CWEN_IDEA_MADVISE
static void madvise_layer(int l) {
  #define MV(t) madvise_span(&W[l].t, MADV_WILLNEED)
  MV(attn_norm); MV(post_norm);
  MV(ffn_gate); MV(ffn_up); MV(ffn_down);
  if(W[l].is_linear){
    MV(qkv); MV(gate_z);
    MV(ssm_out); MV(ssm_alpha);
    MV(ssm_beta); MV(ssm_conv);
  } else {
    MV(wq); MV(wk);
    MV(wv); MV(wo);
  }
  #undef MV
}
#endif

/* Logits dumps for the goldens: positional copy plus the stable name. */
static void dump_logits(const char *ddir) {
  if(!ddir||!ddir[0]) return;
  char nm[64];
  snprintf(nm,sizeof nm,"logits_pos%02d.bin",pos_n);
  dump_f32(ddir,nm,logits,V);
  dump_f32(ddir,"logits.bin",logits,V);
}

/* need_logits=0 skips the final rmsnorm + lm_head sweep (+logits dump): a
   prefill position whose logits nothing consumes would still pay the full
   Q6_K [V,H] gemv (~136 ms measured via bench_q4_gemv output.weight). Only
   the last prefill token's logits feed argmax. Dump mode keeps the head at
   every position so CWEN dumps stay byte-identical. */
static void forward_ex(int token, int need_logits) {
  /* rope tables and KV slots index by pos_n; both decode drivers stop before
     MAX_SEQ, but the bound belongs to this function, not its callers */
  assert(pos_n>=0 && pos_n<g_ctx);
  const char *ddir=dump_dir();
  int lim=dump_layers_lim();
  embed_token_to(token,x);
  if(ddir){
    char nm[64];
    snprintf(nm,sizeof nm,"embed_pos%02d.bin", pos_n);
    dump_f32(ddir,nm,x,H);
    if(pos_n==0) dump_f32(ddir,"embed.bin",x,H);
  }
  for(int l=0;l<L;l++) {
#if CWEN_IDEA_MADVISE
    if(l+1<L) madvise_layer(l+1);
#endif
    if(g_pipe_pf&&l+1<L){
      const Tensor *t=W[l+1].is_linear?&W[l+1].qkv:&W[l+1].wq;
      size_t n=row_bytes(t->type,t->ne0)*(size_t)t->ne1; if(n>(4u<<20)) n=4u<<20;
      const char *pp2=(const char*)t->data;
      for(size_t o=0;o<n;o+=4096) __builtin_prefetch(pp2+o,0,0);
    }
    if (W[l].is_linear) layer_linear(l);
    else layer_full(l);
    /* DFlash2 tap capture: residual stream after the tap layers */
    if(dflash_on){
      for(int i=0;i<DL_TAPN;i++)
        if(DL_TAPS[i]==l) memcpy(DTapSer+(size_t)i*H,x,H*sizeof(float));
    }
    if(ddir && l<lim){
      char nm[64];
      snprintf(nm,sizeof nm,"layer%02d_pos%02d.bin", l, pos_n);
      dump_f32(ddir,nm,x,H);
      if(pos_n==0){ snprintf(nm,sizeof nm,"layer%02d.bin", l); dump_f32(ddir,nm,x,H); }
    }
    if(ddir && lim<L && l+1>=lim){
      rmsnorm(xb, x, (const float*)output_norm.data, H);
      /* Head always runs at the boundary: the decode drivers argmax the
         shared logits buffer right after prefill, and skipping the sweep
         here would leave them scoring stale heap bytes as token ids. Only
         the dump is gated by CWEN_DUMP_LOGITS. */
      gemv(&output, xb, logits);
      if(dump_logits_flag()) dump_logits(ddir);
      return;
    }
  }
  if(!need_logits) return;
  rmsnorm(xb, x, (const float*)output_norm.data, H);
  gemv(&output, xb, logits);
  dump_logits(ddir);
}

static int argmax_of(const float *lg) {
  /* AVX2 8-wide max scan over V=248320 logits (~1MB) */
  int bi=0; float bv=lg[0];
  int i=0;
#if defined(__AVX2__)
  __m256 vmax=_mm256_set1_ps(bv);
  for(;i+8<=V;i+=8){
    __m256 v=_mm256_loadu_ps(lg+i);
    __m256 cmp=_mm256_cmp_ps(v,vmax,_CMP_GT_OQ);
    if(_mm256_movemask_ps(cmp)){
      float t[8] __attribute__((aligned(32))); _mm256_store_ps(t,v);
      for(int k=0;k<8;k++) if(t[k]>bv){bv=t[k];bi=i+k;}
      vmax=_mm256_set1_ps(bv);
    }
  }
#endif
  /* pointer walk: gcc13+flto misproves the indexed form as OOB once this
     inlines into the speculative driver (heap rows of Blogits) */
  { const float *p=lg+i,*pe=lg+V;
    for(;p<pe;p++,i++){ if(*p>bv){bv=*p;bi=i;} } }
  return bi;
}
static int argmax_logits(void){ return argmax_of(logits); }

/* ---- block speculation: batched verify pass ----
   Same math as calling forward() serially per position (GDN recurrence and
   causal attention run in order), but every weight matrix is streamed once
   for the whole block via gemvb. No CWEN_DUMP support: dumps stay a property
   of the serial path so golden bytes never depend on speculation. */

static void mlp_blk(int layer,int B) {
  gemvb(&W[layer].ffn_gate,Bxbn,H,Bhb,I,B);
  gemvb(&W[layer].ffn_up,Bxbn,H,Bhb2,I,B);
  for(int b=0;b<B;b++) silu_mul(Bhb+(size_t)b*I,Bhb2+(size_t)b*I,I);
  gemvb(&W[layer].ffn_down,Bhb,I,Bxb2,H,B);
  for(int b=0;b<B;b++) residual_add(BXres+(size_t)b*H,Bxb2+(size_t)b*H,H);
}

static void layer_linear_blk(int layer,int B) {
  size_t Zh=(size_t)LVH*LSD;
  for(int b=0;b<B;b++)
    rmsnorm(Bxbn+(size_t)b*H,BXres+(size_t)b*H,(const float*)W[layer].attn_norm.data,H);
  gemvb(&W[layer].qkv,Bxbn,H,Bqkvb,QKV_DIM,B);
  gemvb(&W[layer].gate_z,Bxbn,H,Bzb,Zh,B);
  gemvb(&W[layer].ssm_alpha,Bxbn,H,Sab,LVH,B);
  gemvb(&W[layer].ssm_beta,Bxbn,H,Sbb,LVH,B);
  const float *dt=(const float*)W[layer].ssm_dt.data;
  const float *A=(const float*)W[layer].ssm_a.data;
  for(int b=0;b<B;b++)
    for(int i=0;i<LVH;i++){
      Sab[(size_t)b*LVH+i]=A[i]*softplus(Sab[(size_t)b*LVH+i]+dt[i]);
      Sbb[(size_t)b*LVH+i]=sigmoid(Sbb[(size_t)b*LVH+i]);
    }
  /* stateful core stays sequential over the block, exactly like serial decode */
  for(int b=0;b<B;b++){
    conv1d_update(layer,Bqkvb+(size_t)b*QKV_DIM);
    gdn_step(layer,Bqkvb+(size_t)b*QKV_DIM,Bzb+(size_t)b*Zh,
             Sab+(size_t)b*LVH,Sbb+(size_t)b*LVH,Bhb+(size_t)b*I);
  }
  gemvb(&W[layer].ssm_out,Bhb,I,Bxb2,H,B);
  for(int b=0;b<B;b++){
    residual_add(BXres+(size_t)b*H,Bxb2+(size_t)b*H,H);
    rmsnorm(Bxbn+(size_t)b*H,BXres+(size_t)b*H,(const float*)W[layer].post_norm.data,H);
  }
  mlp_blk(layer,B);
}

static void layer_full_blk(int layer,int B,int pbase) {
  size_t Qo=(size_t)NH*HD, Ko=(size_t)NKV*HD;
  for(int b=0;b<B;b++)
    rmsnorm(Bxbn+(size_t)b*H,BXres+(size_t)b*H,(const float*)W[layer].attn_norm.data,H);
  gemvb(&W[layer].wq,Bxbn,H,Bqfull,Qo*2,B);
  for(int b=0;b<B;b++)
    for(int h=0;h<NH;h++){
      memcpy(Bqh+(size_t)b*Qo+h*HD,       Bqfull+(size_t)b*Qo*2+h*HD*2,     HD*sizeof(float));
      memcpy(Bgateq+(size_t)b*Qo+h*HD,    Bqfull+(size_t)b*Qo*2+h*HD*2+HD, HD*sizeof(float));
    }
  gemvb(&W[layer].wk,Bxbn,H,Bkh,Ko,B);
  gemvb(&W[layer].wv,Bxbn,H,Bvh,Ko,B);
  const float *qn=(const float*)W[layer].q_norm.data;
  const float *kn=(const float*)W[layer].k_norm.data;
  for(int b=0;b<B;b++){
    float *qs=Bqh+(size_t)b*Qo,*ks=Bkh+(size_t)b*Ko;
    int pos=pbase+b;
    for(int h=0;h<NH;h++) rmsnorm(qs+h*HD,qs+h*HD,qn,HD);
    for(int h=0;h<NKV;h++) rmsnorm(ks+h*HD,ks+h*HD,kn,HD);
    rope_apply(qs,NH,pos);
    rope_apply(ks,NKV,pos);
    memcpy(Kcache+((size_t)layer*CTX_STRIDE+pos)*Ko, ks, Ko*sizeof(float));
    memcpy(Vcache+((size_t)layer*CTX_STRIDE+pos)*Ko, Bvh+(size_t)b*Ko, Ko*sizeof(float));
    attention_heads(layer,pos,qs,Bgateq+(size_t)b*Qo,Byatt+(size_t)b*Qo);
  }
  gemvb(&W[layer].wo,Byatt,Qo,Bxb2,H,B);
  for(int b=0;b<B;b++){
    residual_add(BXres+(size_t)b*H,Bxb2+(size_t)b*H,H);
    rmsnorm(Bxbn+(size_t)b*H,BXres+(size_t)b*H,(const float*)W[layer].post_norm.data,H);
  }
  mlp_blk(layer,B);
}

/* Score toks[0..B-1] occupying slots pos_n+1..pos_n+B in order; on return
   pos_n has advanced by B and Blogits[b] predicts the token after toks[b]
   for b < nlogits (0 skips the lm_head sweep entirely: prefill chunks pay
   it once, serially, after the last chunk instead of once per chunk).
   Caller guarantees pos_n+B < MAX_SEQ. */
static void forward_block(const int *toks,int B,int nlogits){
  assert(B>=1&&B<=SPEC_BMAX);
  assert(nlogits>=0&&nlogits<=B);
  assert(pos_n>=0&&pos_n+B<g_ctx);
  int pbase=pos_n+1;
  for(int b=0;b<B;b++){
    if(toks[b]<0||toks[b]>=V){ fprintf(stderr,"token id %d out of range [0,%d)\n",toks[b],V); exit(1); }
    embed_token_to(toks[b],BXres+(size_t)b*H);
  }
  for(int l=0;l<L;l++){
    if(W[l].is_linear) layer_linear_blk(l,B);
    else layer_full_blk(l,B,pbase);
    /* DFlash2 tap capture: residual rows after the tap layers */
    if(dflash_on){
      for(int i=0;i<DL_TAPN;i++)
        if(DL_TAPS[i]==l)
          for(int b=0;b<B;b++)
            memcpy(DTapBlk+(size_t)b*DL_TAPN*H+(size_t)i*H,
                   BXres+(size_t)b*H,H*sizeof(float));
    }
  }
  if(nlogits>0){
    for(int b=0;b<nlogits;b++)
      rmsnorm(Bxbn+(size_t)b*H,BXres+(size_t)b*H,(const float*)output_norm.data,H);
    gemvb(&output,Bxbn,H,Blogits,V,nlogits);
  }
  pos_n=pbase+B-1;
}

/* GDN state is a dense accumulated recurrence: a rejected draft tail would
   poison every later step. Snapshot before a drafted block, restore when the
   walk stops short; full-attn KV needs nothing (slots past the accepted
   prefix are rewritten before any read, same invariant as reset_state).
   Copies are chunk-parallel: ~200 MiB serially is tens of ms, which would
   tax every drafted cycle. */
static void snap_copy(void *dst,const void *src,size_t bytes){
  long long nt=1;
#ifdef _OPENMP
  nt=omp_get_max_threads();
#endif
  size_t chunk=(bytes+(size_t)nt-1)/(size_t)nt;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(long long off=0;off<(long long)bytes;off+=(long long)chunk){
    size_t o=(size_t)off;
    size_t n=o+chunk<bytes?chunk:bytes-o;
    memcpy((char*)dst+o,(const char*)src+o,n);
  }
}
static void snap_save(void){
  snap_copy(SnapS,Srec,(size_t)L*LVH*LSD*LSD*sizeof(float));
  snap_copy(SnapC,Cstate,(size_t)L*QKV_DIM*(CONV_K-1)*sizeof(float));
}
static void snap_load(void){
  snap_copy(Srec,SnapS,(size_t)L*LVH*LSD*LSD*sizeof(float));
  snap_copy(Cstate,SnapC,(size_t)L*QKV_DIM*(CONV_K-1)*sizeof(float));
}

/* ---- DFlash2 drafter runtime ---- */

/* rotate-half RoPE on one DL_HD head at absolute position pos; the drafter
   uses plain theta=1e7 rope, not the target's interleaved mRoPE */
static void rope_std_apply(float *v,int pos) {
  if(!DRdone[pos]){
    for(int i=0;i<DL_HD/2;i++){
      float freq=powf(DL_ROPE_THETA,-(float)(2*i)/(float)DL_HD);
      float ang=(float)pos*freq;
      DRcos[pos][i]=cosf(ang); DRsin[pos][i]=sinf(ang);
    }
    DRdone[pos]=1;
  }
  const float *C=DRcos[pos],*S=DRsin[pos];
  for(int i=0;i<DL_HD/2;i++){
    int i2=i+DL_HD/2;
    float a=v[i],b=v[i2];
    v[i]=a*C[i]-b*S[i];
    v[i2]=a*S[i]+b*C[i];
  }
}
/* per-head rmsnorm (q_norm/k_norm are [128] weights) */
static inline void df_head_norm(float *x,const float *w,int nh){
  for(int h=0;h<nh;h++)
    rmsnorm(x+(size_t)h*DL_HD,x+(size_t)h*DL_HD,w,DL_HD);
}
/* grouped dynamic depthwise conv, one window row. base = this half's
   [tap][chan] rows; dyn = this row's half slice ([tap][group]). tap1 reads
   the previous row of the same stream; the first row zero-pads, matching
   the reference _grouped_dynamic_convolve. */
static inline void df_conv_row(const float *x,const float *xprev,
                               const float *dyn,const float *base,float *out) {
  for(int c=0;c<H;c++){
    float acc=(base[c]+dyn[c>>4])*x[c];
    if(xprev) acc+=(base[H+c]+dyn[320+(c>>4)])*xprev[c];
    out[c]=acc;
  }
}
static void df_q8_row(const Tensor *t,int row,float *out,int cap) {
  int nb=t->ne0/32;
  if(nb*32>cap){fprintf(stderr,"dfq8 ovf: ne0=%d cap=%d\n",t->ne0,cap);exit(1);}
  const uint8_t *p=(const uint8_t*)t->data+(size_t)row*(size_t)nb*34;
  for(int b=0;b<nb;b++,p+=34){
    float d=f16_to_f32((uint16_t)(p[0]|(p[1]<<8)));
    float *o=out+(size_t)b*32;
    for(int j=0;j<32;j++) o[j]=(float)(int8_t)p[2+j]*d;
  }
}
/* top-DL_TOPK of one logits row into idx[] (unordered set is fine: the walk
   scores every candidate) */
static void df_top16(const float *lg,int *idx) {
  float mn[DL_TOPK];
  for(int k=0;k<DL_TOPK;k++){mn[k]=-INFINITY;idx[k]=0;}
  int mpos=0;
  for(int v=0;v<V;v++){
    float x=lg[v];
    if(x>mn[mpos]){
      mn[mpos]=x; idx[mpos]=v;
      mpos=0;
      for(int k=1;k<DL_TOPK;k++) if(mn[k]<mn[mpos]) mpos=k;
    }
  }
}
/* project committed context position pos through fc + hidden_norm once and
   append its per-layer K/V to the drafter caches (keys post-norm + rope at
   their absolute position; values raw). Called exactly once per token that
   becomes part of the verified context. taps points at [5][H] captured
   target residual states in layer order. */
/* paired split-Q8 dual dot: physical rows [2r]=A, [2r+1]=B share x loads */
static inline void dot_q8si(const Tensor *restrict W, int r,
                            const float *restrict x,
                            float *restrict ya, float *restrict yb) {
  const int K=W->ne0, nb=K/QK4;
  const size_t rb=(size_t)K+(size_t)nb*2;
  const int8_t *qa=(const int8_t*)W->data+(size_t)(2*r)*rb;
  const int8_t *qb=qa+rb;
  const uint16_t *sa=(const uint16_t*)(qa+K);
  const uint16_t *sb=(const uint16_t*)(qb+K);
  /* AVX2 only: the 512-bit variant this once carried mixed __m256 accumulators
     into _mm512 FMAs and never compiled. Q8SI is the experimental split
     container that measured slower than blocks, so it gets the portable
     kernel, not a second set of intrinsics. */
  __m256 aa=_mm256_setzero_ps(), ab=_mm256_setzero_ps();
  for(int b=0;b<nb;b++){
    __m256 da=_mm256_set1_ps(f16_to_f32(sa[b]));
    __m256 db=_mm256_set1_ps(f16_to_f32(sb[b]));
    const __m128i a0=_mm_loadl_epi64((const __m128i*)(qa+b*32));
    const __m128i a1=_mm_loadl_epi64((const __m128i*)(qa+b*32+8));
    const __m128i a2=_mm_loadl_epi64((const __m128i*)(qa+b*32+16));
    const __m128i a3=_mm_loadl_epi64((const __m128i*)(qa+b*32+24));
    const __m128i b0=_mm_loadl_epi64((const __m128i*)(qb+b*32));
    const __m128i b1=_mm_loadl_epi64((const __m128i*)(qb+b*32+8));
    const __m128i b2=_mm_loadl_epi64((const __m128i*)(qb+b*32+16));
    const __m128i b3=_mm_loadl_epi64((const __m128i*)(qb+b*32+24));
    const __m256 xl0=_mm256_loadu_ps(x+b*32), xl1=_mm256_loadu_ps(x+b*32+8);
    const __m256 xh0=_mm256_loadu_ps(x+b*32+16), xh1=_mm256_loadu_ps(x+b*32+24);
    __m256 wa0=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(a0)),da);
    __m256 wa1=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(a1)),da);
    __m256 wa2=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(a2)),da);
    __m256 wa3=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(a3)),da);
    __m256 wb0=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b0)),db);
    __m256 wb1=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b1)),db);
    __m256 wb2=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b2)),db);
    __m256 wb3=_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b3)),db);
    aa=_mm256_fmadd_ps(wa0,xl0,aa); aa=_mm256_fmadd_ps(wa1,xl1,aa);
    aa=_mm256_fmadd_ps(wa2,xh0,aa); aa=_mm256_fmadd_ps(wa3,xh1,aa);
    ab=_mm256_fmadd_ps(wb0,xl0,ab); ab=_mm256_fmadd_ps(wb1,xl1,ab);
    ab=_mm256_fmadd_ps(wb2,xh0,ab); ab=_mm256_fmadd_ps(wb3,xh1,ab);
  }
  *ya=hsum256(aa); *yb=hsum256(ab);
}
/* batched dual pass over all B window columns for a Q8SI pair tensor */
static void df_dual_gemvb(const Tensor *Wt,const float *X,int xs,
                          float *YA,int yas,float *YB,int ybs,int B){
  const int M=Wt->ne1/2;   /* manifest ne1 counts both matrices' rows */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for(int r=0;r<M;r++)
    for(int b=0;b<B;b++)
      dot_q8si(Wt,r,X+(size_t)b*xs,YA+(size_t)b*yas+r,YB+(size_t)b*ybs+r);
}
static void dflash_commit(int pos,const float *taps) {
  assert(pos>=0&&pos<g_ctx);
  float *ctxrow=Dctx+(size_t)pos*H;
  gemv(&Dw_fc,taps,ctxrow);
  rmsnorm(ctxrow,ctxrow,(const float*)Dw_hnorm.data,H);
  for(int l=0;l<DL_LAYERS;l++){
    DraftW *w=&DW[l];
    float *kd=Dkc+((size_t)l*CTX_STRIDE+pos)*DL_KV;
    float *vd=Dvc+((size_t)l*CTX_STRIDE+pos)*DL_KV;
    if(w->kv.data)
      df_dual_gemvb(&w->kv,ctxrow,H,kd,DL_KV,vd,DL_KV,1);
    else{
      gemv(&w->k,ctxrow,kd);
      gemv(&w->v,ctxrow,vd);
    }
    df_head_norm(kd,(const float*)DW[l].kn.data,DL_NKV);
    for(int h=0;h<DL_NKV;h++) rope_std_apply(kd+(size_t)h*DL_HD,pos);
  }
}
/* attention over visible keys for all B window rows of layer l. Visible set
   per query slot p=P+b: cached context keys j in [max(0,p-(WIN-1)), P-1]
   plus every noise-window key P..P+B-1 (bidirectional inside the block;
   the checkpoint's is_causal=false). */
static void df_attn_window(int l,int P,int B) {
  float scale=1.f/sqrtf((float)DL_HD);
  for(int b=0;b<B;b++){
    int qpos=P+b;
    int tstart=qpos-(DL_WIN-1); if(tstart<0)tstart=0;
    for(int h=0;h<DL_NH;h++){
      int hk=h/(DL_NH/DL_NKV);
      const float *qh=Dq_t+(size_t)b*DL_Q+(size_t)h*DL_HD;
      float mx=-INFINITY,sum=0.f;
      for(int j=tstart;j<P+B;j++){
        const float *kj=(j<P)
          ? Dkc+((size_t)l*CTX_STRIDE+j)*DL_KV+(size_t)hk*DL_HD
          : Dkw_t+(size_t)(j-P)*DL_KV+(size_t)hk*DL_HD;
        float s=dot_f32(qh,kj,DL_HD)*scale;
        Datt[j-tstart]=s; if(s>mx)mx=s;
      }
      int nvis=P+B-tstart;
      for(int n=0;n<nvis;n++){Datt[n]=expf(Datt[n]-mx);sum+=Datt[n];}
      float inv=1.f/sum;
      float *o=Dao_t+(size_t)b*DL_Q+(size_t)h*DL_HD;
      memset(o,0,DL_HD*sizeof(float));
      for(int j=tstart;j<P+B;j++){
        const float *vj=(j<P)
          ? Dvc+((size_t)l*CTX_STRIDE+j)*DL_KV+(size_t)hk*DL_HD
          : Dvw_t+(size_t)(j-P)*DL_KV+(size_t)hk*DL_HD;
        float a=Datt[j-tstart]*inv;
#if defined(__AVX2__)
        __m256 va=_mm256_set1_ps(a);
        for(int i=0;i<DL_HD;i+=8)
          _mm256_storeu_ps(o+i,_mm256_fmadd_ps(va,_mm256_loadu_ps(vj+i),_mm256_loadu_ps(o+i)));
#else
        for(int i=0;i<DL_HD;i++) o[i]+=a*vj[i];
#endif
      }
    }
  }
}
/* one drafter layer over the B-row window; P = first window slot (absolute) */
static void df_layer(int l,int P,int B) {
  DraftW *w=&DW[l];
  for(int b=0;b<B;b++)
    rmsnorm(Dln_t+(size_t)b*H,Dhw+(size_t)b*H,(const float*)w->ln1.data,H);
  gemvb(&w->ac_proj,Dln_t,H,DdynA,1280,B);
  /* attention sublayer: conv in -> attn -> conv out, residual around */
  for(int b=0;b<B;b++)
    df_conv_row(Dln_t+(size_t)b*H,b?Dln_t+(size_t)(b-1)*H:NULL,
                DdynA+(size_t)b*1280,(const float*)w->ac_base.data,
                Din_t+(size_t)b*H);
  gemvb(&w->q,Din_t,H,Dq_t,DL_Q,B);
  if(w->kv.data) df_dual_gemvb(&w->kv,Din_t,H,Dkw_t,DL_KV,Dvw_t,DL_KV,B);
  else{ gemvb(&w->k,Din_t,H,Dkw_t,DL_KV,B);
        gemvb(&w->v,Din_t,H,Dvw_t,DL_KV,B); }
  for(int b=0;b<B;b++){
    int pos=P+b;
    df_head_norm(Dq_t+(size_t)b*DL_Q,(const float*)w->qn.data,DL_NH);
    df_head_norm(Dkw_t+(size_t)b*DL_KV,(const float*)w->kn.data,DL_NKV);
    for(int h=0;h<DL_NH;h++) rope_std_apply(Dq_t+(size_t)b*DL_Q+(size_t)h*DL_HD,pos);
    for(int h=0;h<DL_NKV;h++) rope_std_apply(Dkw_t+(size_t)b*DL_KV+(size_t)h*DL_HD,pos);
  }
  df_attn_window(l,P,B);
  gemvb(&w->o,Dao_t,DL_Q,Doh_t,H,B);
  for(int b=0;b<B;b++)
    df_conv_row(Doh_t+(size_t)b*H,b?Doh_t+(size_t)(b-1)*H:NULL,
                DdynA+(size_t)b*1280+640,
                (const float*)w->ac_base.data+2*H,
                Dln_t+(size_t)b*H);          /* scratch: ln rows are dead now */
  for(int b=0;b<B;b++)
    residual_add(Dhw+(size_t)b*H,Dln_t+(size_t)b*H,H);
  /* mlp sublayer, same wrap */
  for(int b=0;b<B;b++)
    rmsnorm(Dln_t+(size_t)b*H,Dhw+(size_t)b*H,(const float*)w->ln2.data,H);
  gemvb(&w->mc_proj,Dln_t,H,DdynM,1280,B);
  for(int b=0;b<B;b++)
    df_conv_row(Dln_t+(size_t)b*H,b?Dln_t+(size_t)(b-1)*H:NULL,
                DdynM+(size_t)b*1280,(const float*)w->mc_base.data,
                Din_t+(size_t)b*H);
  if(w->gu.data) df_dual_gemvb(&w->gu,Din_t,H,Bhb,I,Bhb2,I,B);
  else{ gemvb(&w->gate,Din_t,H,Bhb,I,B);
        gemvb(&w->up,Din_t,H,Bhb2,I,B); }
  for(int b=0;b<B;b++) silu_mul(Bhb+(size_t)b*I,Bhb2+(size_t)b*I,I);
  gemvb(&w->down,Bhb,I,Doh_t,H,B);
  for(int b=0;b<B;b++)
    df_conv_row(Doh_t+(size_t)b*H,b?Doh_t+(size_t)(b-1)*H:NULL,
                DdynM+(size_t)b*1280+640,
                (const float*)w->mc_base.data+2*H,
                Dln_t+(size_t)b*H);
  for(int b=0;b<B;b++)
    residual_add(Dhw+(size_t)b*H,Dln_t+(size_t)b*H,H);
}
/* propose up to max_d draft tokens after hist[hn-1] (the anchor). Window =
   [anchor, mask...]; slots 1..bs-1 each yield one candidate via the
   selector's greedy walk. Returns proposal count (may be 0 when no room). */
static int dflash_draft(const int *hist,int hn,int *out,int max_d){
  if(hn<1||max_d<=0) return 0;
  int room=MAX_SEQ-1-(pos_n+1);
  if(room<=0) return 0;
  int bs=max_d+1;                       /* anchor + drafts */
  if(bs>DL_BLOCK) bs=DL_BLOCK;
  if(bs>room+1) bs=room+1;
  if(bs<2) return 0;
  int E=bs-1,P=pos_n+1;
  int prev=hist[hn-1];
  if(prev<0||prev>=V) return 0;
  embed_token_to(prev,Dhw);
  for(int b=1;b<bs;b++) embed_token_to(DL_MASKTOK,Dhw+(size_t)b*H);
  for(int l=0;l<DL_LAYERS;l++) df_layer(l,P,bs);
  /* final norm; candidate logits for prediction slots 1..bs-1 */
  for(int s=1;s<bs;s++)
    rmsnorm(Doh_t+(size_t)(s-1)*H,Dhw+(size_t)s*H,(const float*)Dw_norm.data,H);
  gemvb(&output,Doh_t,H,Blogits,V,E);
  /* selector: greedy path through per-slot top-K */
  for(int s=1;s<bs;s++){
    float pg[DL_RANK],best=-INFINITY; int bestc=0;
    gemv(&Dw_hproj,Doh_t+(size_t)(s-1)*H,Dg_t);
    df_q8_row(&Dw_pred,prev,Dpred_r,DL_RANK);
    for(int d=0;d<DL_RANK;d++) pg[d]=Dpred_r[d]*Dg_t[d];
    const float *lg=Blogits+(size_t)(s-1)*V;
    df_top16(lg,Dcand+(size_t)(s-1)*DL_TOPK);
    int *cand=Dcand+(size_t)(s-1)*DL_TOPK;
    for(int c=0;c<DL_TOPK;c++){
      df_q8_row(&Dw_succ,cand[c],Dsucc_rows+(size_t)c*DL_RANK,DL_RANK);
      float e=lg[cand[c]];                 /* unary = U_t(b) */
      for(int d=0;d<DL_RANK;d++) e+=pg[d]*Dsucc_rows[(size_t)c*DL_RANK+d];
      if(e>best){best=e;bestc=c;}
    }
    out[s-1]=cand[bestc];
    prev=cand[bestc];
  }
  return E;
}

/* ---- MTP forward (PR18) ----
   blk.64 is a full-attention decoder block with the same geometry as the
   target's full layers, fronted by the nextn projection. Slot j of the nextn
   stream pairs token t_j with the target's final normed hidden h_{j-1} and
   predicts t_{j+1}; slot 0 has no predecessor hidden, so the stream starts at
   slot 1 and attention runs over [1..pos]. RoPE is relative, so indexing the
   stream by the input token's own position (rather than the trained-time
   shift-by-one) leaves scores unchanged. */

/* one nextn step at stream slot `pos`: input token `token_id`, predecessor
   target hidden `h_target`. Writes K/V into the nextn cache at `pos` and
   returns the shared-head-normed hidden in out_h[H]. */
static void mtp_step(int token_id,int pos,const float *h_target,float *out_h){
  embed_token_to(token_id,Nemb);
  rmsnorm(Ncat,  Nemb,    (const float*)NW.enorm.data,H);
  rmsnorm(Ncat+H,h_target,(const float*)NW.hnorm.data,H);
  gemv(&NW.eh_proj,Ncat,Nxw);

  memcpy(Nresid,Nxw,H*sizeof(float));
  rmsnorm(Nln,Nxw,(const float*)NW.attn_norm.data,H);
  gemv(&NW.wq,Nln,Nqfull);
  for(int h=0;h<NH;h++){                       /* q and gate interleave per head */
    memcpy(Nqh+(size_t)h*HD,Nqfull+(size_t)h*HD*2,   HD*sizeof(float));
    memcpy(Ngq+(size_t)h*HD,Nqfull+(size_t)h*HD*2+HD,HD*sizeof(float));
  }
  gemv(&NW.wk,Nln,Nkh);
  gemv(&NW.wv,Nln,Nvh);
  for(int h=0;h<NH;h++)  rmsnorm(Nqh+(size_t)h*HD,Nqh+(size_t)h*HD,(const float*)NW.q_norm.data,HD);
  for(int h=0;h<NKV;h++) rmsnorm(Nkh+(size_t)h*HD,Nkh+(size_t)h*HD,(const float*)NW.k_norm.data,HD);
  rope_apply(Nqh,NH,pos);
  rope_apply(Nkh,NKV,pos);
  memcpy(NKc+(size_t)pos*NKV*HD,Nkh,NKV*HD*sizeof(float));
  memcpy(NVc+(size_t)pos*NKV*HD,Nvh,NKV*HD*sizeof(float));
  attention_heads_kv(NKc,NVc,1,pos,Nqh,Ngq,Nao);

  gemv(&NW.wo,Nao,Nop);
  residual_add(Nresid,Nop,H);
  rmsnorm(Nln,Nresid,(const float*)NW.post_norm.data,H);
  gemv(&NW.ffn_gate,Nln,Nffg);
  gemv(&NW.ffn_up,Nln,Nffu);
  silu_mul(Nffg,Nffu,I);
  gemv(&NW.ffn_down,Nffg,Nop);
  residual_add(Nresid,Nop,H);
  rmsnorm(out_h,Nresid,(const float*)NW.shared_head_norm.data,H);
}

/* Where a drafted cycle's time actually goes, under CWEN_SPEC_DEBUG=1: the
   nextn layer is ~2% of a target sweep but the shared lm_head it needs per
   proposal is ~7%, so the head, not the layer, sets the draft budget. */
static double mtp_t_step,mtp_t_head,mtp_t_commit; static long mtp_n_step,mtp_n_head;
static double mtp_now(void){
  if(!Scfg_debug) return 0.0;
  struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
  return (double)t.tv_sec+1e-9*(double)t.tv_nsec;
}

/* Fold the token the target just scored at `pos` into the nextn stream, then
   park that position's final normed hidden for the next slot. Called once per
   committed position, in order; x_pre is the target's residual stream before
   output_norm. */
static void mtp_commit(int pos,int token,const float *x_pre){
  if(!mtp_use) return;
  /* pos 0 has no predecessor hidden and no attendable slot (the stream starts
     at 1): stepping it would softmax an empty window. It only parks h_0. */
  if(pos>=1&&mtp_hpos==pos-1){
    float oh[H] __attribute__((aligned(64)));
    double t0=mtp_now();
    mtp_step(token,pos,NHprev,oh);
    mtp_t_commit+=mtp_now()-t0;
  }
  rmsnorm(NHprev,x_pre,(const float*)output_norm.data,H);
  mtp_hpos=pos;
}

/* Autoregressive drafting: slot pos_n+1 takes the pending token against the
   parked hidden, and every further slot recurses on the nextn layer's own
   output (the single-block MTP head has no deeper hidden to feed it). Slots
   past pos_n are speculative and get rewritten by mtp_commit as tokens land,
   the same invariant the target's full-attn KV relies on. */
static int mtp_draft(int pend,int *out,int max_d){
  static float lg[V] __attribute__((aligned(64)));
  float hprev[H] __attribute__((aligned(64))), oh[H] __attribute__((aligned(64)));
  if(!mtp_use||mtp_hpos!=pos_n||max_d<=0) return 0;
  int n=max_d;
  if(pos_n+n>=g_ctx) n=g_ctx-1-pos_n;
  if(n<=0) return 0;
  memcpy(hprev,NHprev,sizeof hprev);
  int tok=pend;
  for(int t=0;t<n;t++){
    double t0=mtp_now();
    mtp_step(tok,pos_n+1+t,hprev,oh);
    double t1=mtp_now();
    gemv(&output,oh,lg);
    out[t]=argmax_of(lg);
    double t2=mtp_now();
    mtp_t_step+=t1-t0; mtp_t_head+=t2-t1; mtp_n_step++; mtp_n_head++;
    tok=out[t];
    memcpy(hprev,oh,sizeof hprev);
  }
  return n;
}
static const uint8_t *Gend; /* mapped-GGUF end; all cursor reads are bounds-checked */
static uint64_t *G_offs;    /* tensor offsets (global so the fuzzer can free mid-parse) */
static void gguf_bad(const char *why){ fprintf(stderr,"gguf: %s\n",why); exit(1); }
static const uint8_t *adv(const uint8_t **p, size_t k){
  if((size_t)(Gend-*p)<k) gguf_bad("truncated");
  const uint8_t *q=*p; *p+=k; return q;
}
static uint64_t rd_u64(const uint8_t **p){ uint64_t v; memcpy(&v,adv(p,8),8); return v; }
static uint32_t rd_u32(const uint8_t **p){ uint32_t v; memcpy(&v,adv(p,4),4); return v; }
static void rd_str(const uint8_t **p, char *buf, int cap){
  uint64_t n=rd_u64(p);
  if(n>=(uint64_t)cap) gguf_bad("str long"); /* u64 compare: no int truncation */
  memcpy(buf,adv(p,(size_t)n),n); buf[n]=0;
}
static void skip_val(const uint8_t **p, uint32_t t, int depth){
  if(depth>64) gguf_bad("kv nest too deep"); /* hostile deep nesting would smash the stack */
  switch(t){
    case 0: case 1: case 7: adv(p,1); break;
    case 2: case 3: adv(p,2); break;
    case 4: case 5: case 6: adv(p,4); break;
    case 8: { uint64_t n=rd_u64(p); adv(p,(size_t)n); } break;
    case 9: {
      uint32_t at=rd_u32(p); uint64_t n=rd_u64(p);
      for(uint64_t i=0;i<n;i++) skip_val(p,at,depth+1);
    } break;
    case 10: case 11: case 12: adv(p,8); break;
    default: { char m[32]; snprintf(m,sizeof m,"kv type %u",t); gguf_bad(m); }
  }
}

static Tensor *find_tensor(const char *name) {
  for(int i=0;i<Ntens;i++) if(!strcmp(Tnames[i],name)) return &Tens[i];
  return NULL;
}
static const char *tens_name(const Tensor *t) { return Tnames[t-Tens]; }
static Tensor must(const char *name) {
  Tensor *t=find_tensor(name); if(!t){fprintf(stderr,"gguf: missing %s\n",name);exit(1);} return *t;
}

static int g_gguf_deferred_warm; /* set when CWENR will own Q4_0 */
static void rebind_layers_from_tens(void); /* binds W[] from Tens[], validating kernel contracts */
static void dflash_bad(const char *why);
static void load_gguf(const char *path) {
  int fd=open(path,O_RDONLY); if(fd<0){perror(path);exit(1);}
  struct stat st;
  if(fstat(fd,&st)){perror(path);exit(1);}
  if(st.st_size<20){fprintf(stderr,"gguf: %s: %lld bytes, not a gguf\n",path,(long long)st.st_size);exit(1);}
  Gmap_len=(size_t)st.st_size;
  Gmap=mmap_resident_ex(fd,Gmap_len,"gguf", !g_gguf_deferred_warm); close(fd);
  if(Gmap==MAP_FAILED){perror("mmap");exit(1);}
#if CWEN_IDEA_COLLAPSE && defined(MADV_COLLAPSE)
  madvise(Gmap,Gmap_len,MADV_COLLAPSE);
#endif
  const uint8_t *p=(const uint8_t*)Gmap, *begin=p;
  Gend=(const uint8_t*)Gmap+Gmap_len;
  /* size was checked against st_size before mmap; here only the magic */
  if(memcmp(p,"GGUF",4)){fprintf(stderr,"not gguf\n");exit(1);} p+=4;
  uint32_t ver=rd_u32(&p);
  /* v2/v3 share this parser's layout (u64 counts, 32B-aligned data). v1 counted
     in u32 and any later version may move fields: reject rather than misparse. */
  if(ver<2u||ver>3u){ char m[32]; snprintf(m,sizeof m,"gguf version %u",ver); gguf_bad(m); }
  uint64_t n_tensors=rd_u64(&p), n_kv=rd_u64(&p);
  for(uint64_t i=0;i<n_kv;i++){ char key[256]; rd_str(&p,key,sizeof key); uint32_t t=rd_u32(&p); skip_val(&p,t,0); }
  if(n_tensors>sizeof Tens/sizeof Tens[0]) gguf_bad("too many tensors");
  Ntens=(int)n_tensors;
  G_offs=calloc(n_tensors,8);
  if(!G_offs){fprintf(stderr,"oom\n");exit(1);}
  for(uint64_t i=0;i<n_tensors;i++){
    rd_str(&p,Tnames[i],96);
    uint32_t n_dims=rd_u32(&p);
    uint64_t dims[4]={1,1,1,1};
    if(n_dims>sizeof dims/sizeof dims[0]) gguf_bad("n_dims > 4");
    for(uint32_t d=0;d<n_dims;d++){
      dims[d]=rd_u64(&p);
      /* ne0/ne1 are ints downstream: reject zero and int-truncating values */
      if(dims[d]==0||dims[d]>(uint64_t)INT_MAX){
        /* %.200s: bounded precision so -Wformat-truncation is provable at O3/IPA */
        char m[256]; snprintf(m,sizeof m,"%.200s: dim %u = %llu",Tnames[i],d,(unsigned long long)dims[d]);
        gguf_bad(m);
      }
    }
    uint32_t typ=rd_u32(&p);
    /* T_Q4_0R/RS/RSI (100+) are internal tags only load_cwenr may set after
       binding; scales stay NULL until then, so a GGUF claiming one would pass
       the extent check and fault the first gemv. */
    if(typ>=(uint32_t)T_Q4_0R){
      char m[32]; snprintf(m,sizeof m,"reserved type %u",typ); gguf_bad(m);
    }
    G_offs[i]=rd_u64(&p);
    Tens[i].type=(int)typ;
    Tens[i].ne0=(int)dims[0];
    Tens[i].ne1=(int)(n_dims>1?dims[1]:1);
    Tens[i].pair_side=0;
    Tens[i].scales=NULL;
  }
  /* align data section */
  uint64_t data_off=(uint64_t)(p-begin);
  uint64_t align=32;
  data_off=(data_off+align-1)&~(align-1);
  for(int i=0;i<Ntens;i++){
    /* offsets are file-relative: keep every data pointer inside the mapping */
    if(data_off>=Gmap_len || G_offs[i]>=Gmap_len-data_off) gguf_bad("tensor offset oob");
    Tens[i].data=(const char*)Gmap+data_off+G_offs[i];
    /* kernels index data by row_bytes*ne1: keep the whole extent mapped too.
       row_bytes==0 marks types this build cannot size; they are never read.
       Product cannot overflow: both factors were INT_MAX-checked at parse. */
    uint64_t rb=(uint64_t)row_bytes(Tens[i].type,Tens[i].ne0);
    if(rb && rb*(uint64_t)Tens[i].ne1>Gmap_len-data_off-G_offs[i]){
      char m[128]; snprintf(m,sizeof m,"%s: extent beyond map",Tnames[i]); gguf_bad(m);
    }
  }
  free(G_offs); G_offs=NULL;

  rebind_layers_from_tens();
  fprintf(stderr,"loaded %d tensors from %s\n",Ntens,path);
}

/* Offline CWENR sidecar (tools/repack_q4.py). Rebinds GGUF Q4_0 tensors to
   mmap views in the sidecar: v3/v4 bind split/interleaved T_Q4_0RS/RSI, v2
   binds packed T_Q4_0R then splits at runtime.
   Returns -1 when the path does not fit (same as the .gguf branch): a silently
   truncated path would just miss an existing sidecar. */
static int cwenr_path_for(const char *gguf, char *out, size_t cap) {
  const char *env=getenv("CWEN_REPACK");
  if(env && env[0]){
    int k=snprintf(out,cap,"%s",env);
    return (k<0||(size_t)k>=cap)?-1:0;
  }
  /* foo.gguf → foo.cwenr */
  size_t n=strlen(gguf);
  if(n>5 && !strcmp(gguf+n-5,".gguf")){
    if(n-5+6>=cap) return -1;
    memcpy(out,gguf,n-5); memcpy(out+n-5,".cwenr",7);
    return 0;
  }
  int k=snprintf(out,cap,"%s.cwenr",gguf);
  return (k<0||(size_t)k>=cap)?-1:0;
}
/* Kernel contracts: rmsnorm/gdn/conv read these as raw float rows; a quantized
   type here would silently produce garbage activations. Matmul weights dispatch
   through gemv_row/embed_token_to, which return 0 or exit for unknown types. */
static Tensor must_f32(const char *name) {
  Tensor t=must(name);
  if(t.type!=T_F32){ fprintf(stderr,"gguf: %s: type %d, need F32\n",name,t.type); exit(1); }
  return t;
}
/* embed_token_to dispatches exactly these; the wider matmul accept-set would
   load fine and die mid-run ("embed type %d") on the first forward. */
static int embed_type_ok(int ty) {
  switch(ty){
    case T_Q4_0: case T_Q4_0R: case T_Q4_0RS: case T_Q6_K: return 1;
    default: return 0;
  }
}
static Tensor must_mat(const char *name) {
  Tensor t=must(name);
  if(!matmul_type_ok(t.type)){ fprintf(stderr,"gguf: %s: unsupported weight type %d\n",name,t.type); exit(1); }
  return t;
}
/* Kernels index activations and outputs by the compile-time dims in the enum
   above, so a declared weight shape that disagrees is an OOB read/write on a
   hostile model file. Mats are pinned exactly (ne1 drives writes into fixed
   buffers, ne0 drives reads of them). F32 vectors only need to *contain* the
   n floats kernels read, so a floor cannot reject any model that runs today. */
static void shape_bad(const char *name,int gne0,int gne1,int wne0,long long wneed){
  fprintf(stderr,"gguf: %s: shape (%d,%d), need (%d,%lld)\n",name,gne0,gne1,wne0,wneed);
  exit(1);
}
static Tensor must_mat_sh(const char *name,int ne0,int ne1){
  Tensor t=must_mat(name);
  if(t.ne0!=ne0||t.ne1!=ne1) shape_bad(name,t.ne0,t.ne1,ne0,(long long)ne1);
  return t;
}
static Tensor must_f32_n(const char *name,int n){
  Tensor t=must_f32(name);
  if((long long)t.ne0*t.ne1<n) shape_bad(name,t.ne0,t.ne1,n,-1);
  return t;
}
static void rebind_layers_from_tens(void) {
  tok_embd=must_mat("token_embd.weight");
  if(!embed_type_ok(tok_embd.type)){
    fprintf(stderr,"gguf: token_embd.weight: unsupported embed type %d\n",tok_embd.type);
    exit(1);
  }
  if(tok_embd.ne0!=H||tok_embd.ne1!=V) shape_bad("token_embd.weight",tok_embd.ne0,tok_embd.ne1,H,(long long)V);
  output_norm=must_f32_n("output_norm.weight",H);
  { Tensor *o=find_tensor("output.weight"); output = o ? *o : tok_embd; } /* tied embeddings */
  if(output.ne0!=H||output.ne1!=V) shape_bad("output.weight",output.ne0,output.ne1,H,(long long)V);
  /* gemv() writes nothing for a type outside its dispatch table: an unchecked
     lm_head would serve stale logits and argmax garbage with no error. */
  if(!matmul_type_ok(output.type)){
    fprintf(stderr,"gguf: output.weight: unsupported weight type %d\n",output.type);
    exit(1);
  }
  for(int l=0;l<L;l++){
    char n[96];
    W[l].is_linear=((l+1)%FULL_INT)!=0;
    snprintf(n,sizeof n,"blk.%d.attn_norm.weight",l); W[l].attn_norm=must_f32_n(n,H);
    snprintf(n,sizeof n,"blk.%d.post_attention_norm.weight",l); W[l].post_norm=must_f32_n(n,H);
    snprintf(n,sizeof n,"blk.%d.ffn_gate.weight",l); W[l].ffn_gate=must_mat_sh(n,H,I);
    snprintf(n,sizeof n,"blk.%d.ffn_up.weight",l); W[l].ffn_up=must_mat_sh(n,H,I);
    snprintf(n,sizeof n,"blk.%d.ffn_down.weight",l); W[l].ffn_down=must_mat_sh(n,I,H);
    if(W[l].is_linear){
      snprintf(n,sizeof n,"blk.%d.attn_qkv.weight",l); W[l].qkv=must_mat_sh(n,H,QKV_DIM);
      snprintf(n,sizeof n,"blk.%d.attn_gate.weight",l); W[l].gate_z=must_mat_sh(n,H,LVH*LSD);
      snprintf(n,sizeof n,"blk.%d.ssm_a",l); W[l].ssm_a=must_f32_n(n,LVH);
      snprintf(n,sizeof n,"blk.%d.ssm_dt.bias",l); W[l].ssm_dt=must_f32_n(n,LVH);
      snprintf(n,sizeof n,"blk.%d.ssm_alpha.weight",l); W[l].ssm_alpha=must_mat_sh(n,H,LVH);
      snprintf(n,sizeof n,"blk.%d.ssm_beta.weight",l); W[l].ssm_beta=must_mat_sh(n,H,LVH);
      snprintf(n,sizeof n,"blk.%d.ssm_conv1d.weight",l); W[l].ssm_conv=must_f32_n(n,CONV_K*QKV_DIM);
      snprintf(n,sizeof n,"blk.%d.ssm_norm.weight",l); W[l].ssm_norm=must_f32_n(n,LSD); /* one shared head_dim vector, broadcast over heads */
      snprintf(n,sizeof n,"blk.%d.ssm_out.weight",l); W[l].ssm_out=must_mat_sh(n,LVH*LSD,H);
    } else {
      snprintf(n,sizeof n,"blk.%d.attn_q.weight",l); W[l].wq=must_mat_sh(n,H,NH*HD*2);
      snprintf(n,sizeof n,"blk.%d.attn_k.weight",l); W[l].wk=must_mat_sh(n,H,NKV*HD);
      snprintf(n,sizeof n,"blk.%d.attn_v.weight",l); W[l].wv=must_mat_sh(n,H,NKV*HD);
      snprintf(n,sizeof n,"blk.%d.attn_output.weight",l); W[l].wo=must_mat_sh(n,NH*HD,H);
      snprintf(n,sizeof n,"blk.%d.attn_q_norm.weight",l); W[l].q_norm=must_f32_n(n,HD);
      snprintf(n,sizeof n,"blk.%d.attn_k_norm.weight",l); W[l].k_norm=must_f32_n(n,HD);
    }
  }
  /* ---- MTP nextn layer (blk.64) binding ---- */
  {
    memset(&NW,0,sizeof NW);
    Tensor *mtp_tp; char mtp_nn[96];
    #define MTP_BIND(f,suf) do{ \
      snprintf(mtp_nn,sizeof mtp_nn,"blk.64.%s",suf); \
      mtp_tp=find_tensor(mtp_nn); if(mtp_tp) NW.f=*mtp_tp; }while(0)
    MTP_BIND(attn_norm,"attn_norm.weight");
    MTP_BIND(post_norm,"post_attention_norm.weight");
    MTP_BIND(wq,"attn_q.weight");
    MTP_BIND(wk,"attn_k.weight");
    MTP_BIND(wv,"attn_v.weight");
    MTP_BIND(wo,"attn_output.weight");
    MTP_BIND(q_norm,"attn_q_norm.weight");
    MTP_BIND(k_norm,"attn_k_norm.weight");
    MTP_BIND(ffn_gate,"ffn_gate.weight");
    MTP_BIND(ffn_up,"ffn_up.weight");
    MTP_BIND(ffn_down,"ffn_down.weight");
    { snprintf(mtp_nn,sizeof mtp_nn,"blk.64.nextn.eh_proj.weight");
      mtp_tp=find_tensor(mtp_nn); if(mtp_tp){ NW.eh_proj=*mtp_tp; mtp_on=1; } }
    { snprintf(mtp_nn,sizeof mtp_nn,"blk.64.nextn.enorm.weight");
      mtp_tp=find_tensor(mtp_nn); if(mtp_tp) NW.enorm=*mtp_tp; }
    { snprintf(mtp_nn,sizeof mtp_nn,"blk.64.nextn.hnorm.weight");
      mtp_tp=find_tensor(mtp_nn); if(mtp_tp) NW.hnorm=*mtp_tp; }
    { snprintf(mtp_nn,sizeof mtp_nn,"blk.64.nextn.shared_head_norm.weight");
      mtp_tp=find_tensor(mtp_nn); if(mtp_tp) NW.shared_head_norm=*mtp_tp; }
    #undef MTP_BIND
    if(mtp_on){
      /* the drafter runs gemv straight off these; a partial bind would read
         a zeroed Tensor (null data, type 0) instead of failing here */
      const struct { const Tensor *t; const char *nm; int ne0,ne1; } req[] = {
        {&NW.wq,"blk.64.attn_q",H,NH*HD*2},   {&NW.wk,"blk.64.attn_k",H,NKV*HD},
        {&NW.wv,"blk.64.attn_v",H,NKV*HD},    {&NW.wo,"blk.64.attn_output",NH*HD,H},
        {&NW.ffn_gate,"blk.64.ffn_gate",H,I}, {&NW.ffn_up,"blk.64.ffn_up",H,I},
        {&NW.ffn_down,"blk.64.ffn_down",I,H}, {&NW.eh_proj,"blk.64.nextn.eh_proj",2*H,H},
      };
      for(size_t i=0;i<sizeof req/sizeof*req;i++){
        if(req[i].t->ne0!=req[i].ne0||req[i].t->ne1!=req[i].ne1)
          shape_bad(req[i].nm,req[i].t->ne0,req[i].t->ne1,req[i].ne0,req[i].ne1);
        if(!matmul_type_ok(req[i].t->type)){
          fprintf(stderr,"gguf: blk.64.%s: unsupported weight type %d\n",
                  req[i].nm,req[i].t->type);
          exit(1);
        }
      }
      const struct { const Tensor *t; const char *nm; int n; } vreq[] = {
        {&NW.attn_norm,"blk.64.attn_norm",H}, {&NW.post_norm,"blk.64.post_attention_norm",H},
        {&NW.q_norm,"blk.64.attn_q_norm",HD}, {&NW.k_norm,"blk.64.attn_k_norm",HD},
        {&NW.enorm,"blk.64.nextn.enorm",H},   {&NW.hnorm,"blk.64.nextn.hnorm",H},
        {&NW.shared_head_norm,"blk.64.nextn.shared_head_norm",H},
      };
      for(size_t i=0;i<sizeof vreq/sizeof*vreq;i++)
        if((long long)vreq[i].t->ne0*vreq[i].t->ne1<vreq[i].n)
          shape_bad(vreq[i].nm,vreq[i].t->ne0,vreq[i].t->ne1,vreq[i].n,-1);
      fprintf(stderr,"mtp: nextn layer bound\n");
    }
  }
}
/* Parse one CWENR directory entry and match it to a loaded GGUF Q4_0 tensor.
   Returns NULL (reason already printed) when the entry cannot apply; else the
   tensor plus its declared shape, both u64 payload fields, and (when flags is
   non-NULL, v4) the entry flag word. One copy of the guards so v2 and v3/v4
   cannot drift on what makes an entry bindable. */
static Tensor *cwenr_entry(const uint8_t *dir, uint32_t i,
                           int32_t *ne0, int32_t *ne1,
                           uint64_t *off_a, uint64_t *off_b, uint32_t *flags){
  const uint8_t *e=dir+32+(size_t)i*128;
  char name[96]; memcpy(name,e,96); name[95]=0;
  memcpy(ne0,e+96,4); memcpy(ne1,e+100,4);
  memcpy(off_a,e+104,8); memcpy(off_b,e+112,8);
  if(flags) memcpy(flags,e+120,4);
  Tensor *t=find_tensor(name);
  if(!t){ fprintf(stderr,"cwenr: unknown %s\n",name); return NULL; }
  if(t->type!=T_Q4_0){ fprintf(stderr,"cwenr: skip non-Q4_0 %s\n",name); return NULL; }
  if(t->ne0!=*ne0||t->ne1!=*ne1){ fprintf(stderr,"cwenr: shape %s\n",name); return NULL; }
  return t;
}
static void load_cwenr(const char *path, const char *src_gguf) {
  int fd=open(path,O_RDONLY); if(fd<0) return; /* optional */
  struct stat st;
  if(fstat(fd,&st)||st.st_size<32){
    fprintf(stderr,"cwenr: %s: unusable (%s), ignoring sidecar\n",path,st.st_size<32?"too small":"stat failed");
    close(fd); return;
  }
  Rmap_len=(size_t)st.st_size;
  Rmap=mmap_resident(fd,Rmap_len,"cwenr"); close(fd);
  if(Rmap==MAP_FAILED){ Rmap=NULL; perror("cwenr mmap"); return; }
#if CWEN_IDEA_COLLAPSE && defined(MADV_COLLAPSE)
  madvise(Rmap,Rmap_len,MADV_COLLAPSE);
#endif
  const uint8_t *p=(const uint8_t*)Rmap;
  if(Rmap_len<32 || memcmp(p,"CWENR001",8)){
    fprintf(stderr,"cwenr: bad magic %s\n",path); munmap(Rmap,Rmap_len); Rmap=NULL; return;
  }
#ifndef CWEN_FUZZ_LOADER
  /* Staleness gate: a sidecar stamped by tools/repack_q4.py carries the source
     GGUF's size in pages. If the GGUF on disk no longer matches, the sidecar
     would silently serve old weights over the new model: drop it and fall back
     to the full-GGUF path. Untagged (zeros) = legacy sidecar, trusted as
     before. The fuzz harness feeds one blob as both GGUF and sidecar and is
     not testing coherence policy, so the gate is bypassed there. */
  {
    uint32_t tag,src_pages;
    memcpy(&tag,p+24,4); memcpy(&src_pages,p+28,4);
    if(tag==CWENR_STAMP_TAG){
      struct stat gs;
      uint64_t want=(src_gguf && stat(src_gguf,&gs)==0)
                    ? (((uint64_t)gs.st_size+4095)/4096) : 0;
      if((uint64_t)src_pages!=want){
        fprintf(stderr,
                "cwenr: stale sidecar %s (source %s changed since repack;"
                " regenerate with tools/repack_q4.py)\n",
                path,src_gguf?src_gguf:"(unknown)");
        munmap(Rmap,Rmap_len); Rmap=NULL; return;
      }
    }
  }
#endif
  uint32_t ver, n;
  uint64_t data_base;
  memcpy(&ver,p+8,4); memcpy(&n,p+12,4); memcpy(&data_base,p+16,8);
  /* v4 = interleaved dual pairs + solo; v3 = split solo; v2 = packed 20B */
  if((ver!=2 && ver!=3 && ver!=4) || n>1024 || data_base>=Rmap_len
     || 32u+(size_t)n*128u>Rmap_len){ /* entry directory must fit in file */
    fprintf(stderr,"cwenr: bad header ver=%u n=%u\n",ver,n);
    munmap(Rmap,Rmap_len); Rmap=NULL; return;
  }
  int bound=0;
  if(ver==3 || ver==4){
    size_t freed=0;
    int n_il=0;
    for(uint32_t i=0;i<n;i++){
      int32_t ne0,ne1; uint64_t qs_off,sc_off; uint32_t flags=0;
      Tensor *t=cwenr_entry(p,i,&ne0,&ne1,&qs_off,&sc_off,&flags);
      if(!t) continue;
      size_t nblk=q4r_nb(ne0)*(size_t)ne1;
      size_t qsb, scb;
      if(flags==CWENR_F_IL_A || flags==CWENR_F_IL_B){
        qsb=nblk*32u; scb=nblk*4u; /* interleaved */
      } else {
        qsb=nblk*16u; scb=nblk*2u;
      }
      { /* remainder form: cannot wrap like data_base+off+size can */
        size_t rem=Rmap_len-(size_t)data_base;
        if(qs_off>rem || sc_off>rem || qsb>rem-qs_off || scb>rem-sc_off){
          fprintf(stderr,"cwenr: OOB %s\n",tens_name(t)); continue;
        }
      }
      /* Drop GGUF Q4_0 pages: weights live in .cwenr now (halves resident DRAM). */
      if(t->data && Gmap){
        const char *old=(const char*)t->data;
        if(old>=(const char*)Gmap && old<(const char*)Gmap+Gmap_len){
          size_t nbytes=q4_0_nbytes(ne0,ne1);
          uintptr_t pg=(uintptr_t)old&~(uintptr_t)4095;
          size_t span=nbytes+((uintptr_t)old-pg);
          if(pg+(uintptr_t)span<=(uintptr_t)Gmap+Gmap_len){
            madvise((void*)pg,span,MADV_DONTNEED);
            freed+=nbytes;
          }
        }
      }
      t->data=(const char*)Rmap+data_base+qs_off;
      t->scales=(const void*)((const char*)Rmap+data_base+sc_off);
      if(flags==CWENR_F_IL_A || flags==CWENR_F_IL_B){
        t->type=T_Q4_0RSI;
        t->pair_side=(flags==CWENR_F_IL_B)?1:0;
        n_il++;
      } else {
        t->type=T_Q4_0RS;
        t->pair_side=0;
      }
      bound++;
    }
    fprintf(stderr,"cwenr: v%u bound %d (il=%d) mmap; DONTNEED %zu MiB GGUF Q4\n",
            ver, bound, n_il, freed>>20);
  } else {
    for(uint32_t i=0;i<n;i++){
      int32_t ne0,ne1; uint64_t off,nbytes;
      Tensor *t=cwenr_entry(p,i,&ne0,&ne1,&off,&nbytes,NULL);
      if(!t) continue;
      { /* remainder form: cannot wrap like data_base+off+nbytes can */
        size_t rem=Rmap_len-(size_t)data_base;
        if(off>rem || nbytes>rem-off){ fprintf(stderr,"cwenr: OOB %s\n",tens_name(t)); continue; }
      }
      size_t expect=q4r_row_bytes(ne0)*(size_t)ne1;
      if(nbytes!=expect){ fprintf(stderr,"cwenr: size %s\n",tens_name(t)); continue; }
      t->data=(const char*)Rmap+data_base+off;
      t->type=T_Q4_0R;
      t->scales=NULL;
      bound++;
    }
    /* v2 fallback: runtime split packed 20B → qs + f16 scales */
    if(bound>0){
      size_t nblk=0;
      for(int i=0;i<Ntens;i++)
        if(Tens[i].type==T_Q4_0R)
          nblk += q4r_nb(Tens[i].ne0)*(size_t)Tens[i].ne1;
      Q4qs_arena=aligned_alloc(64, nblk*16u);
      Q4sc_arena=aligned_alloc(64, nblk*sizeof(uint16_t));
      if(!Q4qs_arena||!Q4sc_arena){fprintf(stderr,"cwenr: split oom\n");exit(1);}
      Q4qs_n=nblk*16u; Q4sc_n=nblk*sizeof(uint16_t);
      size_t oq=0, os=0;
      for(int i=0;i<Ntens;i++){
        if(Tens[i].type!=T_Q4_0R) continue;
        int nb=(int)q4r_nb(Tens[i].ne0), rows=Tens[i].ne1;
        const block_q4_0r *src=(const block_q4_0r*)Tens[i].data;
        uint8_t *qdst=(uint8_t*)Q4qs_arena+oq;
        uint16_t *sdst=(uint16_t*)Q4sc_arena+os;
        size_t nn=(size_t)nb*rows;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(nn>4096)
#endif
        for(size_t b=0;b<nn;b++){
          memcpy(qdst+b*16, src[b].qs, 16);
          sdst[b]=f32_to_f16(src[b].d);
        }
        Tens[i].data=qdst; Tens[i].scales=sdst; Tens[i].type=T_Q4_0RS;
        oq+=nn*16; os+=nn;
      }
      fprintf(stderr,"cwenr: v2 runtime split %zu blocks\n",nblk);
      munmap(Rmap,Rmap_len); Rmap=NULL; Rmap_len=0;
    }
  }
  /* Nothing bound means the sidecar is unusable (stale names, all entries
     rejected): say so and drop the mapping instead of pinning it for the
     process lifetime behind a success-worded "bound 0" line. Tens[] is
     untouched in this case, so the load_gguf bindings stay authoritative. */
  if(!bound){
    fprintf(stderr,"cwenr: %s: no usable entries; ignoring sidecar\n",path);
    if(Rmap){munmap(Rmap,Rmap_len);Rmap=NULL;Rmap_len=0;}
    return;
  }
  rebind_layers_from_tens();
  fprintf(stderr,"cwenr: bound %d Q4 tensors from %s\n",bound,path);
}

/* ---- DFlash2 .spec container ----
   Header: "DFSP", u32 version, u32 count, then per tensor:
   char name[64]; u32 ne0,ne1,type; u64 offset,u64 nbytes (payload offsets
   are absolute from file start). Types: 0=F32, 1=Q4_0R, 2=Q8_0. */
static void *Dmap; static size_t Dmap_len;
static void dflash_bad(const char *why){
  fprintf(stderr,"dflash: %s\n",why); exit(1);
}
/* exact payload bytes a .spec entry must carry for its declared geometry
   (types: 0=F32, 1=Q4_0R, 2=Q8_0); shared by the loader and the fuzzer's
   post-load invariant walk */
static uint64_t dflash_tensor_bytes(uint32_t typ,uint32_t ne0,uint32_t ne1){
  switch(typ){
    case 0: return (uint64_t)ne0*ne1*4u;
    case 1: return (uint64_t)(ne0/QK4)*sizeof(block_q4_0r)*ne1;
    case 3: /* Q8S: int8 stream + f16 scale channel */
      return (uint64_t)ne1*((uint64_t)ne0+(uint64_t)(ne0/QK4)*2);
    case 4: /* Q8SI pair: manifest ne1 counts BOTH matrices' rows */
      return (uint64_t)ne1*((uint64_t)ne0+(uint64_t)(ne0/QK4)*2);
    default: return (uint64_t)(ne0/QK4)*sizeof(block_q8_0)*ne1;
  }
}
static void load_dflash(const char *path) {
  int fd=open(path,O_RDONLY);
  if(fd<0){perror(path);exit(1);}
  struct stat st;
  if(fstat(fd,&st)){perror(path);exit(1);}
  Dmap_len=(size_t)st.st_size;
  Dmap=mmap(NULL,Dmap_len,PROT_READ,MAP_PRIVATE,fd,0);
  close(fd);
  if(Dmap==MAP_FAILED){perror("mmap");exit(1);}
  const uint8_t *p=Dmap;
  if(Dmap_len<12||memcmp(p,"DFSP",4)) dflash_bad("bad magic");
  uint32_t ver,n;
  memcpy(&ver,p+4,4); memcpy(&n,p+8,4);
  if(ver!=1) dflash_bad("unsupported version");
  p+=12;                                   /* entries start after the header */
  /* duplicate-detection flags for the global drafter tensors; per-layer
     entries detect dups via dst[hit]->data instead */
  int seen_fc=0,seen_hn=0,seen_norm=0,seen_pred=0,seen_succ=0,seen_hp=0;
  for(uint32_t i=0;i<n;i++){
    if((size_t)(p-(const uint8_t*)Dmap)+96>(size_t)Dmap_len) dflash_bad("truncated header");
    char name[65]; memcpy(name,p,64); name[64]=0; p+=64;
    uint32_t ne0,ne1,typ; uint64_t off,nb;
    memcpy(&ne0,p,4);memcpy(&ne1,p+4,4);
    uint64_t typ8;
    memcpy(&typ8,p+8,8);                    /* type stored as u64 */
    memcpy(&off,p+16,8);memcpy(&nb,p+24,8); p+=32;
    typ=(uint32_t)typ8;
    /* dims must cast to a positive int and quant rows need whole QK4 blocks;
       nb is validated against this geometry below, never via Dmap_len-nb,
       which wraps when nb>Dmap_len and lets an oversized blob through */
    if(!ne0||!ne1||ne0>0x7fffffffu||ne1>0x7fffffffu||ne0%QK4)
      dflash_bad("bad tensor dims");
    if((uint64_t)Dmap_len<off||nb>Dmap_len-off){
      fprintf(stderr,"dflash: entry %u '%s' off=%llu nb=%llu len=%zu\n",
              i,name,(unsigned long long)off,(unsigned long long)nb,Dmap_len);
      dflash_bad("offset out of range");
    }
    if(nb!=dflash_tensor_bytes(typ,ne0,ne1)){
      fprintf(stderr,"dflash: entry %u '%s' nb=%llu expect=%llu\n",
              i,name,(unsigned long long)nb,
              (unsigned long long)dflash_tensor_bytes(typ,ne0,ne1));
      dflash_bad("nbytes does not match dims");
    }
    int ctype = typ==0?T_F32 : typ==1?T_Q4_0R : typ==2?T_Q8_0 :
                typ==3?T_Q8S  : T_Q8SI;
    Tensor t={ (const char*)Dmap+off,NULL,ctype,(int)ne0,(int)ne1,0 };
    if(ctype==T_Q8S)  t.scales=(const char*)Dmap+off+(size_t)ne0*(size_t)ne1;
    const char *suffix=NULL; int li=-1;
    if(!strncmp(name,"layers.",7)){
      char *endp; long idx=strtol(name+7,&endp,10);
      if(idx<0||idx>=DL_LAYERS||*endp!='.') dflash_bad("bad layer name");
      li=(int)idx; suffix=endp+1;
    }
    if(li>=0){
      DraftW *w=&DW[li];
      const char *m[]={ "q_proj","k_proj","v_proj","o_proj","gate","up","down",
                        "ln1","ln2","qn","kn","attn_conv_base","mlp_conv_base",
                        "attn_conv_proj","mlp_conv_proj","mlp_gu","attn_kv" };
      int hit=-1;
      for(unsigned k2=0;k2<sizeof m/sizeof*m;k2++)
        if(!strcmp(suffix,m[k2])){hit=(int)k2;break;}
      Tensor *dst[]={ &w->q,&w->k,&w->v,&w->o,&w->gate,&w->up,&w->down,
                      &w->ln1,&w->ln2,&w->qn,&w->kn,
                      &w->ac_base,&w->mc_base,&w->ac_proj,&w->mc_proj,
                      &w->gu,&w->kv };
      if(hit<0) dflash_bad("unknown drafter tensor");
      if(dst[hit]->data) { fprintf(stderr,"dflash: dup %s\n",name); dflash_bad("duplicate drafter tensor"); }
      *dst[hit]=t;
    }else{
      struct {const char *n; Tensor *t; int *f;} g[]={
        {"fc",&Dw_fc,&seen_fc},{"hidden_norm",&Dw_hnorm,&seen_hn},
        {"norm",&Dw_norm,&seen_norm},{"sel.pred",&Dw_pred,&seen_pred},
        {"sel.succ",&Dw_succ,&seen_succ},{"sel.hproj",&Dw_hproj,&seen_hp},
      };
      int hit=-1;
      for(unsigned k2=0;k2<sizeof g/sizeof*g;k2++)
        if(!strcmp(name,g[k2].n)){hit=(int)k2;break;}
      if(hit<0) dflash_bad("unknown drafter tensor");
      if(*g[hit].f) { fprintf(stderr,"dflash: dup %s\n",name); dflash_bad("duplicate drafter tensor"); }
      *g[hit].f=1; *g[hit].t=t;
    }
  }
  /* geometry checks against the trained checkpoint */
  if(Dw_fc.ne0!=DL_CTXIN||Dw_fc.ne1!=H) dflash_bad("fc shape mismatch");
  for(int l=0;l<DL_LAYERS;l++){
    DraftW *w=&DW[l];
    if(w->q.ne0!=H||w->q.ne1!=DL_Q) dflash_bad("q shape");
    if(w->kv.data){
      if(w->kv.ne0!=H||w->kv.ne1!=2*DL_KV) dflash_bad("attn_kv shape");
    }else{
      if(w->k.ne0!=H||w->k.ne1!=DL_KV) dflash_bad("k shape");
      if(w->v.ne0!=H||w->v.ne1!=DL_KV) dflash_bad("v shape");
    }
    if(w->o.ne0!=DL_Q||w->o.ne1!=H) dflash_bad("o shape");
    if(w->gu.data){
      if(w->gu.ne0!=H||w->gu.ne1!=2*I) dflash_bad("mlp_gu shape");
    }else{
      if(w->gate.ne0!=H||w->gate.ne1!=I) dflash_bad("gate shape");
      if(w->up.ne0!=H||w->up.ne1!=I) dflash_bad("up shape");
    }
    if(w->down.ne0!=I||w->down.ne1!=H) dflash_bad("down shape");
    if(w->ac_proj.ne0!=H||w->ac_proj.ne1!=1280) dflash_bad("conv proj shape");
    if(w->mc_proj.ne0!=H||w->mc_proj.ne1!=1280) dflash_bad("conv proj shape");
    if(w->ac_base.ne0!=20480||w->mc_base.ne0!=20480) dflash_bad("conv base shape");
    /* rmsnorm/df_head_norm read these as raw F32 rows of fixed length: a
       quantized or short entry would read past the .spec mapping */
    if(w->ln1.type!=T_F32||(long long)w->ln1.ne0*w->ln1.ne1<H) dflash_bad("ln1 shape");
    if(w->ln2.type!=T_F32||(long long)w->ln2.ne0*w->ln2.ne1<H) dflash_bad("ln2 shape");
    if(w->qn.type!=T_F32||(long long)w->qn.ne0*w->qn.ne1<DL_HD) dflash_bad("qn shape");
    if(w->kn.type!=T_F32||(long long)w->kn.ne0*w->kn.ne1<DL_HD) dflash_bad("kn shape");
  }
  if(Dw_hnorm.type!=T_F32||(long long)Dw_hnorm.ne0*Dw_hnorm.ne1<H)
    dflash_bad("hidden_norm shape");
  if(Dw_norm.type!=T_F32||(long long)Dw_norm.ne0*Dw_norm.ne1<H)
    dflash_bad("norm shape");
  /* df_q8_row walks fixed 34-byte Q8_0 blocks and indexes rows by token id:
     another type, fewer ranks (leaves the output tail uninitialized), or
     fewer rows reads past the .spec mapping */
  if(Dw_pred.type!=T_Q8_0||Dw_pred.ne0!=DL_RANK||Dw_pred.ne1!=V)
    dflash_bad("selector pred shape");
  if(Dw_succ.type!=T_Q8_0||Dw_succ.ne0!=DL_RANK||Dw_succ.ne1!=V)
    dflash_bad("selector succ shape");
  /* gemv writes ne1 outputs into Dg_t[DL_RANK] from an H-float row */
  if(Dw_hproj.ne0!=H||Dw_hproj.ne1!=DL_RANK) dflash_bad("sel hproj shape");
}

static void load_model(const char *gguf_path) {
  char rpath[512];
  g_gguf_deferred_warm=0;
  int have_path=cwenr_path_for(gguf_path,rpath,sizeof rpath)==0;
  if(have_path && access(rpath,R_OK)==0){
    g_gguf_deferred_warm=1; /* skip full GGUF page-in; CWENR holds Q4 */
    load_gguf(gguf_path);
    load_cwenr(rpath,gguf_path);
    warm_live_gguf_tensors();
    return;
  }
  load_gguf(gguf_path);
  if(have_path)
    fprintf(stderr,"cwenr: no sidecar %s (Q4_0 GGUF path)\n",rpath);
  else
    fprintf(stderr,"cwenr: sidecar path for %s does not fit; skipping sidecar\n",gguf_path);
}

#ifdef CWEN_FUZZ_LOADER
/* Harness hooks (tools/fuzz_loader.c): reset all loader state between
   iterations, then run the parse path with exit() converted to a return. */
static int cw_in_region(const void *p,const void *base,size_t len){
  return base && (const char*)p>=(const char*)base
              && (const char*)p<(const char*)base+len;
}
void cw_fuzz_reset(void){
  if(Gmap){munmap(Gmap,Gmap_len);Gmap=NULL;} Gmap_len=0; Gend=NULL;
  free(G_offs); G_offs=NULL;
  if(Rmap){munmap(Rmap,Rmap_len);Rmap=NULL;} Rmap_len=0;
  free(Q4qs_arena); Q4qs_arena=NULL; Q4qs_n=0;
  free(Q4sc_arena); Q4sc_arena=NULL; Q4sc_n=0;
  Ntens=0; memset(Tens,0,sizeof Tens); memset(Tnames,0,sizeof Tnames);
  memset(W,0,sizeof W);
  memset(&tok_embd,0,sizeof tok_embd);
  memset(&output_norm,0,sizeof output_norm);
  memset(&output,0,sizeof output);
  /* DFlash .spec loader state */
  if(Dmap){munmap(Dmap,Dmap_len);Dmap=NULL;} Dmap_len=0;
  memset(DW,0,sizeof DW);
  Tensor *dwg[]={&Dw_fc,&Dw_hnorm,&Dw_norm,&Dw_pred,&Dw_succ,&Dw_hproj};
  for(unsigned i=0;i<sizeof dwg/sizeof dwg[0];i++)
    memset(dwg[i],0,sizeof(Tensor));
}
/* Returns 0 = parsed clean (invariants held), 1 = rejected via exit(), and
   aborts from the caller on 2 = invariant break. */
int cw_fuzz_once(const char *gguf,const char *cwenr){
  cw_fuzz_armed=0;
  if(setjmp(cw_fuzz_jmp)){ cw_fuzz_armed=0; return 1; }
  cw_fuzz_armed=1;
  setenv("CWEN_REPACK",cwenr,1);
  load_model(gguf);
  cw_fuzz_armed=0;
  /* invariant: every bound pointer stays inside a live backing region */
  for(int i=0;i<Ntens;i++){
    const char *d=Tens[i].data,*s=Tens[i].scales;
    if(d && !cw_in_region(d,Gmap,Gmap_len) && !cw_in_region(d,Rmap,Rmap_len)
        && !cw_in_region(d,Q4qs_arena,Q4qs_n)) return 2;
    if(s && !cw_in_region(s,Gmap,Gmap_len) && !cw_in_region(s,Rmap,Rmap_len)
        && !cw_in_region(s,Q4sc_arena,Q4sc_n)) return 2;
  }
  return 0;
}

/* ---- frame-protocol harness hooks (tools/fuzz_loader.c) ----
   Drive the request-frame parser over an arbitrary byte stream with the
   executor stubbed out, tracing each framing decision so the harness can
   diff it against an independent decoder of the documented protocol. */
static int server_read_frame(FILE *in,int *tokens,uint32_t *np_p,uint32_t *ng_p);

/* Runtime limits, so the harness mirrors validation without hardcoding them.
   server_read_frame bounds both header words by the context window (CWEN_CTX,
   default 4096), not by the MAX_SEQ enum ceiling. */
void cw_fuzz_limits(unsigned *vocab,unsigned *max_seq){
  *vocab=(unsigned)V; *max_seq=(unsigned)g_ctx;
}

/* Replay len bytes through server_read_frame. dec[i] is 0 for a rejected
   frame and 1 for an accepted one; pos[i] the input offset after the frame
   was fully consumed. EOF and truncated payloads terminate without a trace
   entry. Returns frames traced, or -1 if the caller's trace cap was hit. */
int cw_fuzz_server(const uint8_t *data,size_t len,
                   uint32_t *dec,uint64_t *pos,int cap){
  FILE *in=fmemopen((void*)data,len,"rb");
  if(!in) return -1;
  static int tokens[MAX_SEQ]; /* single-threaded fuzzer */
  int n=0;
  for(;;){
    uint32_t np,ng;
    int r=server_read_frame(in,tokens,&np,&ng);
    if(r<0) break;
    if(n>=cap){ n=-1; break; }
    long off=ftell(in); /* glibc: cursor sits after consumed bytes */
    dec[n]=(uint32_t)(r==1);
    pos[n]=(off>0)?(uint64_t)off:(uint64_t)len;
    n++;
  }
  fclose(in);
  return n;
}

/* ---- DFlash .spec harness hook ----
   Same pattern as cw_fuzz_once: exit-rejections recover via the armed jump,
   then every bound drafter tensor is asserted to lie fully inside the
   mapping it was bound from (start pointer and start+geometry), so an
   offset or nbytes regression reopens the OOB class as a fuzzer-visible
   abort instead of a silent out-of-map read. */
static int cw_fuzz_df_inside(const Tensor *t){
  if(!t->data) return 1;
  const char *d=t->data;
  if(d<(const char*)Dmap||d>=(const char*)Dmap+Dmap_len) return 0;
  uint32_t typ=t->type==T_F32?0u:t->type==T_Q4_0R?1u:2u;
  uint64_t nb=dflash_tensor_bytes(typ,(uint32_t)t->ne0,(uint32_t)t->ne1);
  return nb<=(uint64_t)Dmap_len-(uint64_t)(d-(const char*)Dmap);
}
int cw_fuzz_dflash(const char *path){
  cw_fuzz_armed=0;
  if(setjmp(cw_fuzz_jmp)){ cw_fuzz_armed=0; return 1; }
  cw_fuzz_armed=1;
  load_dflash(path);
  cw_fuzz_armed=0;
  for(int l=0;l<DL_LAYERS;l++){
    const DraftW *w=&DW[l];
    const Tensor *ts[]={ &w->q,&w->k,&w->v,&w->o,&w->gate,&w->up,&w->down,
                         &w->ln1,&w->ln2,&w->qn,&w->kn,
                         &w->ac_base,&w->mc_base,&w->ac_proj,&w->mc_proj };
    for(unsigned k=0;k<sizeof ts/sizeof ts[0];k++)
      if(!cw_fuzz_df_inside(ts[k])) return 2;
  }
  const Tensor *gs[]={ &Dw_fc,&Dw_hnorm,&Dw_norm,&Dw_pred,&Dw_succ,&Dw_hproj };
  for(unsigned k=0;k<sizeof gs/sizeof gs[0];k++)
    if(!cw_fuzz_df_inside(gs[k])) return 2;
  return 0;
}
#endif

static void alloc_state(void) {
  logits=aligned_alloc(64,(size_t)V*sizeof(float));
  /* 64B align: GDN S is 128x128; aligned AVX-512 loads */
  Srec=aligned_alloc(64,(size_t)L*LVH*LSD*LSD*sizeof(float));
  Cstate=aligned_alloc(64,(size_t)L*QKV_DIM*(CONV_K-1)*sizeof(float));
  Kcache=aligned_alloc(64,(size_t)L*CTX_STRIDE*NKV*HD*sizeof(float));
  Vcache=aligned_alloc(64,(size_t)L*CTX_STRIDE*NKV*HD*sizeof(float));
  if(!logits||!Srec||!Cstate||!Kcache||!Vcache){fprintf(stderr,"oom\n");exit(1);}
  memset(Srec,0,(size_t)L*LVH*LSD*LSD*sizeof(float));
  memset(Cstate,0,(size_t)L*QKV_DIM*(CONV_K-1)*sizeof(float));
  if(mtp_use){
    NKc=aligned_alloc(64,(size_t)g_ctx*NKV*HD*sizeof(float));
    NVc=aligned_alloc(64,(size_t)g_ctx*NKV*HD*sizeof(float));
    if(!NKc||!NVc){fprintf(stderr,"oom (nextn kv)\n");exit(1);}
  }
  /* No KV memset: layer_full writes slot pos before any read at t<=pos.
      Scrubbing 2 GiB here would be pure startup latency. */
  /* Block scratch: shared by speculative verify AND batched prefill
     (prefill_forward), so it maps regardless of spec_enabled. */
  size_t Zh=(size_t)LVH*LSD;
  BXres=aligned_alloc(64,(size_t)SPEC_BMAX*H*sizeof(float));
  Bxbn =aligned_alloc(64,(size_t)SPEC_BMAX*H*sizeof(float));
  Bxb2 =aligned_alloc(64,(size_t)SPEC_BMAX*H*sizeof(float));
  Bqkvb=aligned_alloc(64,(size_t)SPEC_BMAX*QKV_DIM*sizeof(float));
  Bzb  =aligned_alloc(64,(size_t)SPEC_BMAX*Zh*sizeof(float));
  Bhb  =aligned_alloc(64,(size_t)SPEC_BMAX*I*sizeof(float));
  Bhb2 =aligned_alloc(64,(size_t)SPEC_BMAX*I*sizeof(float));
  Blogits=aligned_alloc(64,(size_t)SPEC_BMAX*V*sizeof(float));
  Bqh   =aligned_alloc(64,(size_t)SPEC_BMAX*NH*HD*sizeof(float));
  Bgateq=aligned_alloc(64,(size_t)SPEC_BMAX*NH*HD*sizeof(float));
  Byatt =aligned_alloc(64,(size_t)SPEC_BMAX*NH*HD*sizeof(float));
  Bqfull=aligned_alloc(64,(size_t)SPEC_BMAX*NH*HD*2*sizeof(float));
  Bkh=aligned_alloc(64,(size_t)SPEC_BMAX*NKV*HD*sizeof(float));
  Bvh=aligned_alloc(64,(size_t)SPEC_BMAX*NKV*HD*sizeof(float));
  Sab=aligned_alloc(64,(size_t)SPEC_BMAX*LVH*sizeof(float));
  Sbb=aligned_alloc(64,(size_t)SPEC_BMAX*LVH*sizeof(float));
  if(!BXres||!Bxbn||!Bxb2||!Bqkvb||!Bzb||!Bhb||!Bhb2||!Blogits||!Bqh||!Bgateq||
     !Byatt||!Bqfull||!Bkh||!Bvh||!Sab||!Sbb){fprintf(stderr,"oom\n");exit(1);}
  if(!spec_enabled) return;
  /* Rollback snapshots only exist to undo a rejected draft tail. */
  SnapS=aligned_alloc(64,(size_t)L*LVH*LSD*LSD*sizeof(float));
  SnapC=aligned_alloc(64,(size_t)L*QKV_DIM*(CONV_K-1)*sizeof(float));
  if(!SnapS||!SnapC){fprintf(stderr,"oom\n");exit(1);}
  /* n-gram counted map: drafting aid when no trained drafter is loaded */
  if(spec_enabled&&!dflash_on){
    NGM.cap=NG_CAP;
    NGM.keys=malloc(sizeof(int)*NG_CAP*(size_t)Scfg_n_key);
    NGM.tok=malloc(sizeof(int)*NG_CAP);
    NGM.cnt=calloc(NG_CAP,sizeof(int));
    NGM.used=calloc(NG_CAP,1);
    if(!NGM.keys||!NGM.tok||!NGM.cnt||!NGM.used){fprintf(stderr,"oom\n");exit(1);}
  }
  if(dflash_on){
    size_t kv=(size_t)DL_KV;
    DTapSer=aligned_alloc(64,(size_t)DL_TAPN*H*4);
    DTapBlk=aligned_alloc(64,(size_t)SPEC_BMAX*DL_TAPN*H*4);
    Dctx  =aligned_alloc(64,(size_t)g_ctx*H*4);
    Dkc   =aligned_alloc(64,(size_t)DL_LAYERS*g_ctx*kv*4);
    Dvc   =aligned_alloc(64,(size_t)DL_LAYERS*g_ctx*kv*4);
    Dhw   =aligned_alloc(64,(size_t)SPEC_BMAX*H*4);
    Dln_t =aligned_alloc(64,(size_t)SPEC_BMAX*H*4);
    Din_t =aligned_alloc(64,(size_t)SPEC_BMAX*H*4);
    Doh_t =aligned_alloc(64,(size_t)SPEC_BMAX*H*4);
    DdynA =aligned_alloc(64,(size_t)SPEC_BMAX*1280*4);
    DdynM =aligned_alloc(64,(size_t)SPEC_BMAX*1280*4);
    Dq_t  =aligned_alloc(64,(size_t)SPEC_BMAX*DL_Q*4);
    Dao_t =aligned_alloc(64,(size_t)SPEC_BMAX*DL_Q*4);
    Dkw_t =aligned_alloc(64,(size_t)SPEC_BMAX*kv*4);
    Dvw_t =aligned_alloc(64,(size_t)SPEC_BMAX*kv*4);
    Dg_t  =aligned_alloc(64,DL_RANK*4);
    Dpred_r=aligned_alloc(64,DL_RANK*4);
    Dsucc_rows=aligned_alloc(64,(size_t)DL_TOPK*DL_RANK*4);
    Dcand =malloc(sizeof(int)*(size_t)SPEC_BMAX*DL_TOPK);
    if(!DTapSer||!DTapBlk||!Dctx||!Dkc||!Dvc||!Dhw||!Dln_t||!Din_t||!Doh_t||
       !DdynA||!DdynM||!Dq_t||!Dao_t||!Dkw_t||!Dvw_t||!Dg_t||!Dpred_r||
       !Dsucc_rows||!Dcand){fprintf(stderr,"oom\n");exit(1);}
  }
}

/* Fresh sequence state: scrub the recurrent GDN state and rewind the KV
   cursor. K/V need no scrub: every slot read at t<=pos_n was written this
   sequence (layer_full stores kv at pos before attending). Saves ~2 GiB per
   request. Shared by the decode server/CLI and the spec microbench. */
static void reset_state(void) {
  memset(Srec, 0, (size_t)L * LVH * LSD * LSD * sizeof(float));
  memset(Cstate, 0, (size_t)L * QKV_DIM * (CONV_K - 1) * sizeof(float));
  pos_n = 0;
  mtp_hpos = -1;   /* nextn stream restarts with the new prompt */
}


/* Strict range-checked parse of the speculation knobs. Unset = default;
   set-but-invalid exits with a named error like every other env knob. */
static void spec_config_init(void) {
  int v;
  if(env_int("CWEN_SPEC",&v)){
    if(v!=0&&v!=1){fprintf(stderr,"cwen: CWEN_SPEC must be 0 or 1\n");exit(1);}
    spec_enabled=v;
  }
  if(env_int("CWEN_SPEC_NGRAM_N",&v)){
    if(v<2||v>256){fprintf(stderr,"cwen: CWEN_SPEC_NGRAM_N must be in [2,256]\n");exit(1);}
    Scfg_n_key=v;
  }
  if(env_int("CWEN_SPEC_MAX_DRAFT",&v)){
    if(v<1||v>SPEC_BMAX-1){
      fprintf(stderr,"cwen: CWEN_SPEC_MAX_DRAFT must be in [1,%d]\n",SPEC_BMAX-1);exit(1);}
    Scfg_max_draft=v;
  }
  if(env_int("CWEN_SPEC_MIN_DRAFT",&v)){
    if(v<1||v>Scfg_max_draft){
      fprintf(stderr,"cwen: CWEN_SPEC_MIN_DRAFT must be in [1,%d]\n",Scfg_max_draft);exit(1);}
    Scfg_min_draft=v;
  }
  if(env_int("CWEN_SPEC_COOLDOWN",&v)){
    if(v<0||v>100000){fprintf(stderr,"cwen: CWEN_SPEC_COOLDOWN must be in [0,100000]\n");exit(1);}
    Scfg_cooldown=v;
  }
  /* validated here, not per request: a bad value must fail before the model
     load, and the server loop must not re-read env on every frame */
  env_bool("CWEN_SPEC_DEBUG",&Scfg_debug);
  if(Scfg_min_draft>Scfg_max_draft){
    fprintf(stderr,"cwen: CWEN_SPEC_MIN_DRAFT (%d) exceeds CWEN_SPEC_MAX_DRAFT (%d)\n",
            Scfg_min_draft,Scfg_max_draft);
    exit(1);
  }
}

static void prefault(void *p,size_t len){
  volatile char *c=(volatile char*)p;
  for(size_t i=0;i<len;i+=4096) (void)c[i];
}
/* ---- memory residency (cachelm learnings applied to a large model) ----
   Full 12.7 GiB never fits L3 (32 MiB CCD). "Keeping attention layers in
   cache" means keeping KV + activations hot while weights stream. What
   transfers from small-model L3-residency work: THP on large arenas,
   mlock weight mmap, next-layer software prefetch. */
static void thp_hint(void *p,size_t len){
#ifdef MADV_HUGEPAGE
  madvise(p,len,MADV_HUGEPAGE);
#else
  (void)p;(void)len;
#endif
}
static void residency_init(void){
#ifdef MADV_HUGEPAGE
  if(Kcache){size_t n=(size_t)L*g_ctx*NKV*HD*4;thp_hint(Kcache,n);prefault(Kcache,n);}
  if(Vcache){size_t n=(size_t)L*g_ctx*NKV*HD*4;thp_hint(Vcache,n);prefault(Vcache,n);}
  if(Srec){thp_hint(Srec,(size_t)L*LVH*LSD*LSD*4);}
  if(Cstate){thp_hint(Cstate,(size_t)L*QKV_DIM*(CONV_K-1)*4);}
#endif
  if(Gmap&&Gmap_len){
    if(mlock(Gmap,Gmap_len))
      fprintf(stderr,"residency: mlock gguf failed (%s)\n",strerror(errno));
    else fprintf(stderr,"residency: locked gguf %zu MiB\n",Gmap_len>>20);
  }
  if(Rmap&&Rmap_len){
    if(mlock(Rmap,Rmap_len))
      fprintf(stderr,"residency: mlock cwenr failed (%s)\n",strerror(errno));
  }
}

/* Apply GA-evolved OpenMP defaults unless the environment already sets them.
   libgomp parses its ICVs in a shared-library constructor that runs before
   main(), so setenv() from inside the process never reaches them (probe:
   OMP_PROC_BIND set in-main still reads back omp_proc_bind_false). When any
   knob is missing, export the tuned defaults plus the CWEN_OMPREEXEC marker
   and re-exec once; the second pass starts with a complete environment.
   Runs before load_model, so nothing expensive is redone. */
/* CWEN_CTX runtime cap + CWEN_ROPE_YARN opt-in; both must fail fast here so
   allocation sizes and every sequence bound agree from the start. */
static void rope_env_init(void) {
  int v;
  if(env_int("CWEN_CTX",&v)){
    if(v<64||v>MAX_SEQ){
      fprintf(stderr,"cwen: CWEN_CTX must be in [64,%d]\n",MAX_SEQ);exit(1);}
    g_ctx=v;
  }
  const char *y=getenv("CWEN_ROPE_YARN");
  if(y&&y[0]){
    double vals[4]={0,0,32,1};
    int slot=0;
    const char *c=y;
    while(*c&&slot<4){
      char *end; double d=strtod(c,&end);
      if(end==c){fprintf(stderr,"cwen: CWEN_ROPE_YARN parse error at '%s'\n",c);exit(1);}
      vals[slot++]=d; c=end;
      while(*c==','||*c==' ') c++;
    }
    if(slot<2){fprintf(stderr,"cwen: CWEN_ROPE_YARN needs orig_max,factor\n");exit(1);}
    Yarn_orig_max=vals[0]; Yarn_factor=vals[1];
    if(slot>=3) Yarn_beta_fast=vals[2];
    if(slot>=4) Yarn_beta_slow=vals[3];
    if(Yarn_orig_max<1024||Yarn_factor<=1.0||Yarn_beta_fast<=0||Yarn_beta_slow<=0){
      fprintf(stderr,"cwen: CWEN_ROPE_YARN values out of range\n");exit(1);}
    g_yarn_on=1;
  }
  fprintf(stderr,"ctx: %d%s\n",g_ctx,g_yarn_on?" (yaRN)":"");
}

static void cwen_omp_init(int argc,char **argv){
  (void)argc; /* argv is used by the re-exec below */
#ifdef _OPENMP
#ifndef CWEN_FUZZ_LOADER
  if(!getenv("CWEN_OMPREEXEC")
     &&(!getenv("OMP_WAIT_POLICY")||!getenv("GOMP_SPINCOUNT")
        ||!getenv("OMP_PROC_BIND")||!getenv("OMP_PLACES"))){
    char exe[4096];
    ssize_t n=readlink("/proc/self/exe",exe,sizeof exe-1);
    if(n>0){
      exe[n]=0;
      if(!getenv("OMP_WAIT_POLICY")) setenv("OMP_WAIT_POLICY","passive",0);
      if(!getenv("GOMP_SPINCOUNT")) setenv("GOMP_SPINCOUNT","100",0);
      /* Zen5 9950X: stay on one CCD (16c). Cross-CCD OMP thrash L3 and lose BW. */
      if(!getenv("OMP_PROC_BIND")) setenv("OMP_PROC_BIND","close",0);
#if CWEN_IDEA_CCD
      if(!getenv("OMP_PLACES")) setenv("OMP_PLACES","{0}:16:1",0);
#else
      if(!getenv("OMP_PLACES")) setenv("OMP_PLACES","cores",0);
#endif
      setenv("CWEN_OMPREEXEC","1",1);
      fprintf(stderr,"omp: re-exec with tuned OMP env\n");
      execv(exe,argv);
      perror("omp: execv"); /* keep going with the in-process defaults */
    }
  }
#endif
  /* BEFORE any other OMP call: passive wait stops worker spin during serial
     GDN/rmsnorm. Flamecharts showed ~8-17% in libgomp wait while one core
     runs serial; the env knobs above only take effect via the re-exec. */
  omp_set_dynamic(0);
  /* Thread count precedence: OMP_NUM_THREADS (any libgomp form: int, AUTO,
     "n,m" lists; applied by libgomp before main) > CWEN_OMP_THREADS env
     > compiled CWEN_OMP_THREADS (cwen_tune.h) > generic half-core cap. */
  int nproc=omp_get_num_procs();
  const char *omp_nt=getenv("OMP_NUM_THREADS");
  if(!(omp_nt&&omp_nt[0])){
    int nt;
    if(env_int("CWEN_OMP_THREADS",&nt)){
      if(nt<1){fprintf(stderr,"cwen: CWEN_OMP_THREADS must be >=1\n");exit(1);}
    }else{
      nt=CWEN_OMP_THREADS>0?CWEN_OMP_THREADS:nproc>16?16:(nproc>1?nproc/2:1);
      if(nt>nproc) nt=nproc;
    }
    omp_set_num_threads(nt);
  }
  fprintf(stderr,"omp: %d threads\n",omp_get_max_threads());
#endif
}

/* ---- request-frame parser (shared: server frontend + loader fuzzer) ----
   Request frames: <u32 n_prompt><u32 n_gen><n_prompt * i32>, little-endian.
   0xffffffff in a header word is the reserved EOF sentinel and closes the
   stream. Kept out of the frontend #if chain so the fuzzer can drive it with
   a stubbed executor; bench builds silence -Wunused-function already. */

/* strict u32 from f; 0xffffffff marks EOF (also an invalid frame size) */
static uint32_t rd_u32f(FILE *f) {
  uint32_t v = 0;
  if (fread(&v,4,1,f)!=1) return 0xffffffffu;
  return v;
}

/* Consume a rejected frame's declared payload so the stream stays aligned for
   the next header; skipping it would parse prompt bytes as a frame header.
   glibc consumes fully-read items even when fread then falls short, so a
   truncated drain leaves the cursor at EOF either way: the next header read
   fails and the caller closes. */
static void drain_u32f(FILE *f, uint64_t n) {
  uint32_t buf[1024];
  while (n) {
    uint32_t take = n > 1024 ? 1024 : (uint32_t)n;
    if (fread(buf,4,take,f) != take) return; /* EOF mid-frame: caller sees it */
    n -= take;
  }
}

/* index of the first out-of-range token id, or -1 */
static int first_bad_token(const int *t, uint32_t n) {
  for (uint32_t i = 0; i < n; i++)
    if (t[i] < 0 || t[i] >= V) return (int)i;
  return -1;
}

/* Parse one request frame from in. Returns 1 with counts stored and tokens
   filled for a valid frame, 0 for a rejected frame (declared payload drained
   so the next header stays aligned), -1 on end-of-stream (truncated header or
   payload). Never executes: generation stays with the caller so the fuzzer
   can stub it and observe framing decisions in isolation. */
static int server_read_frame(FILE *in, int *tokens,
                             uint32_t *np_p, uint32_t *ng_p) {
  uint32_t n_prompt=rd_u32f(in);
  uint32_t n_gen=rd_u32f(in); /* past EOF both fail; one close path suffices */
  if(n_prompt==0xffffffffu||n_gen==0xffffffffu) return -1; /* EOF */
  if(!n_prompt||n_prompt>(unsigned)g_ctx||!n_gen||n_gen>(unsigned)g_ctx){
    fprintf(stderr,"cwen server: bad frame %u %u\n",n_prompt,n_gen);
    drain_u32f(in,n_prompt);
    return 0;
  }
  if(fread(tokens,4,n_prompt,in)!=n_prompt) return -1; /* EOF mid-payload */
  int bad=first_bad_token(tokens,n_prompt);
  if(bad>=0){
    fprintf(stderr,"cwen server: token %d out of range [0,%d)\n",tokens[bad],V);
    return 0;
  }
  *np_p=n_prompt; *ng_p=n_gen;
  return 1;
}

#ifdef CWEN_BENCH_Q4_GEMV
/* usage: bench_q4_gemv [MODEL] [GOLDEN_DIR] [ITERS]
   golden/dir has meta.json {name,ne0,ne1}, x.bin, y_ref.bin; writes y_c.bin
   The PASS/FAIL summary line format is parsed by tools/idea_bench.py,
   tools/ga_evolve.py, tools/median_bench.py, tools/test_*.sh: keep it. */
static void bench_usage(FILE *out) {
  fprintf(out,
    "usage: bench_q4_gemv [MODEL] [GOLDEN_DIR] [ITERS]\n"
    "Run one gemv kernel over a golden vector dir and compare against y_ref.\n"
    "\n"
    "Arguments:\n"
    "  MODEL       GGUF model path (default: model/Qwen3.8-27B-Q4_0.gguf)\n"
    "  GOLDEN_DIR  golden/<tensor>/ with meta.json, x.bin, y_ref.bin\n"
    "              (default: golden/blk_0_ffn_down_weight)\n"
    "  ITERS       timed iterations, >=1 (default: 3)\n"
    "\n"
    "Writes y_c.bin into GOLDEN_DIR; prints 'PASS|FAIL ... ms/iter' on stdout.\n"
    "Exits 0 on PASS, 1 on FAIL or error.\n");
}

static int read_meta_name(const char *dir, char *name, int *ne0, int *ne1) {
  char path[512]; snprintf(path,sizeof path,"%s/meta.json",dir);
  FILE *f=fopen(path,"r"); if(!f) return -1;
  char buf[4096]; size_t n=fread(buf,1,sizeof buf-1,f); fclose(f); buf[n]=0;
  const char *p=strstr(buf,"\"name\""); if(!p) return -1;
  p=strchr(p,':'); if(!p) return -1; p=strchr(p,'"'); if(!p) return -1; p++;
  const char *e=strchr(p,'"'); if(!e) return -1;
  int len=(int)(e-p); if(len>95) len=95; memcpy(name,p,len); name[len]=0;
  p=strstr(buf,"\"ne0\""); if(!p) return -1; p=strchr(p,':'); if(!p) return -1; *ne0=atoi(p+1);
  p=strstr(buf,"\"ne1\""); if(!p) return -1; p=strchr(p,':'); if(!p) return -1; *ne1=atoi(p+1);
  return 0;
}

/* Fixture preflight for bench_q4_gemv: page-in of the model is tens of
   seconds, so a missing or truncated golden file must fail here, in ms,
   with the path named, not after load_model. */
static void golden_preflight(const char *dir,int ne0,int ne1){
  static const char *fn[]={"x.bin","y_ref.bin"};
  const size_t want[2]={(size_t)ne0*4,(size_t)ne1*4};
  for(int i=0;i<2;i++){
    char path[512]; snprintf(path,sizeof path,"%s/%s",dir,fn[i]);
    struct stat st;
    if(stat(path,&st)||!S_ISREG(st.st_mode)){
      fprintf(stderr,"bench_q4_gemv: missing %s (generate goldens with 'make golden')\n",path);
      exit(1);
    }
    if((size_t)st.st_size<want[i]){
      fprintf(stderr,"bench_q4_gemv: %s: %lld bytes, need %zu (stale dump? re-run 'make golden')\n",
              path,(long long)st.st_size,want[i]);
      exit(1);
    }
  }
}
int main(int argc, char **argv) {
  if(argc>4){fprintf(stderr,"bench_q4_gemv: too many arguments (got %d, max 3)\n\n",argc-1);bench_usage(stderr);return 2;}
  /* same contract as ./run: -h/--help wins wherever it appears */
  for(int i=1;i<argc;i++)
    if(argv[i][0]=='-'&&argv[i][1]){
      if(!strcmp(argv[i],"-h")||!strcmp(argv[i],"--help")){bench_usage(stdout);return 0;}
      fprintf(stderr,"bench_q4_gemv: unknown option '%s'\n\n",argv[i]);bench_usage(stderr);return 2;
    }
  const char *model=argc>1?argv[1]:"model/Qwen3.8-27B-Q4_0.gguf";
  const char *gdir=argc>2?argv[2]:"golden/blk_0_ffn_down_weight";
  int iters=argc>3?arg_int_range(argv[3],1,INT_MAX,"bench_q4_gemv: ","ITERS"):3;
  cwen_omp_init(argc,argv);
  /* Cheap checks first: never pay the model load to learn the golden path
     is bad (fresh clones hit this before 'make golden'). */
  char name[96]; int ne0,ne1;
  if(read_meta_name(gdir,name,&ne0,&ne1)){
    fprintf(stderr,"bench_q4_gemv: cannot read %s/meta.json (generate goldens with 'make golden')\n",gdir);
    return 1;
  }
  golden_preflight(gdir,ne0,ne1);
  load_model(model);
  FILE *f;
  Tensor *t=find_tensor(name);
  if(!t){fprintf(stderr,"bench_q4_gemv: tensor '%s' not found in %s\n",name,model);return 1;}
  if(t->ne0!=ne0||t->ne1!=ne1){
    fprintf(stderr,"bench_q4_gemv: shape mismatch for %s: model (%d,%d), meta (%d,%d)\n",
            name,t->ne0,t->ne1,ne0,ne1);
    return 1;
  }
  float *xin=aligned_alloc(64,(size_t)ne0*4);
  float *yout=aligned_alloc(64,(size_t)ne1*4);
  float *yref=aligned_alloc(64,(size_t)ne1*4);
  if(!xin||!yout||!yref){fprintf(stderr,"oom\n");return 1;}
  char path[512];
  snprintf(path,sizeof path,"%s/x.bin",gdir);
  f=fopen(path,"rb");
  if(!f){perror(path);return 1;}
  if(fread(xin,4,ne0,f)!=(size_t)ne0){fprintf(stderr,"%s: short read\n",path);fclose(f);return 1;}
  fclose(f);
  snprintf(path,sizeof path,"%s/y_ref.bin",gdir);
  f=fopen(path,"rb");
  if(!f){perror(path);return 1;}
  if(fread(yref,4,ne1,f)!=(size_t)ne1){fprintf(stderr,"%s: short read\n",path);fclose(f);return 1;}
  fclose(f);
  /* warmup */
  gemv(t,xin,yout);
  struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
  for(int i=0;i<iters;i++) gemv(t,xin,yout);
  clock_gettime(CLOCK_MONOTONIC,&t1);
  double ms=((t1.tv_sec-t0.tv_sec)*1e3)+((t1.tv_nsec-t0.tv_nsec)*1e-6);
  float ma=0,mr=0; int bi=0;
  for(int i=0;i<ne1;i++){
    float d=fabsf(yout[i]-yref[i]);
    if(d>ma){ma=d;bi=i;}
    float r=d/fmaxf(fmaxf(fabsf(yout[i]),fabsf(yref[i])),1e-8f);
    if(r>mr) mr=r;
  }
  snprintf(path,sizeof path,"%s/y_c.bin",gdir);
  f=fopen(path,"wb");
  if(!f){perror(path);return 1;}
  int wbad=fwrite(yout,4,ne1,f)!=(size_t)ne1;
  wbad|=fclose(f)!=0; /* close even after a short write: never strand the fd */
  if(wbad){fprintf(stderr,"%s: write failed\n",path);return 1;}
  int ok=ma<=1e-4f || (ma<=5e-3f && mr<=2e-2f); /* abs primary; rel soft near 0 */
  printf("%s %s  max_abs=%.6g@%d max_rel=%.6g  %.3f ms/iter (n=%d)\n",
         ok?"PASS":"FAIL", name, (double)ma, bi, (double)mr, ms/iters, iters);
  return ok?0:1;
}
#elif defined(CWEN_BENCH_SPEC)
/* usage: bench_spec [MODEL] [ITERS]
   Microbenchmarks for the block-speculation machinery. One PASS line per
   section; exit 0 iff all pass.
     GEMVB <tensor> B=<b>  batched gemv ms/iter + effective weight GB/s,
                           output checked against B separate gemv calls
     SNAP                  GDN snapshot save+load pair (rollback cost)
     BLOCK B=<b>           reset+prefill-subtracted cost of one verify block
                           vs B serial forwards (the speculation break-even) */
static void bench_spec_usage(FILE *out) {
  fprintf(out,
    "usage: bench_spec [MODEL] [ITERS]\n"
    "Microbench gemvb scaling, GDN snapshot rollback, and verify-block cost.\n"
    "\n"
    "Arguments:\n"
    "  MODEL       GGUF model path (default: model/Qwen3.8-27B-Q4_0.gguf)\n"
    "  ITERS       timed iterations per point, >=1 (default: 2)\n");
}
static uint32_t bench_lcg(uint32_t *s){ *s=*s*1664525u+1013904223u; return *s>>8; }
static void bench_fill(float *v,int n,uint32_t *s){
  for(int i=0;i<n;i++) v[i]=((int)(bench_lcg(s)%4096)-2048)*(1.f/512.f);
}
static double bench_now(void){
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
  return ts.tv_sec+ts.tv_nsec*1e-9;
}
static void bench_prefill(const int *pre,int P){
  pos_n=0;
  for(int i=0;i<P;i++){
    forward_ex(pre[i], i+1==P);
    if(i+1<P) pos_n++;
  }
}
int main(int argc,char **argv) {
  if(argc>3){fprintf(stderr,"bench_spec: too many arguments\n\n");bench_spec_usage(stderr);return 2;}
  /* same contract as ./run: -h/--help wins wherever it appears; any other
     option-shaped argument is a usage error, not a model path */
  for(int i=1;i<argc;i++)
    if(argv[i][0]=='-'&&argv[i][1]){
      if(!strcmp(argv[i],"-h")||!strcmp(argv[i],"--help")){bench_spec_usage(stdout);return 0;}
      fprintf(stderr,"bench_spec: unknown option '%s'\n\n",argv[i]);
      bench_spec_usage(stderr);return 2;
    }
  const char *model=argc>1?argv[1]:"model/Qwen3.8-27B-Q4_0.gguf";
  int iters=argc>2?arg_int_range(argv[2],1,INT_MAX,"bench_spec: ","ITERS"):2;
  cwen_omp_init(argc,argv);
  spec_enabled=1; /* alloc_state must map the block scratch + snapshots */
  load_model(model);
  if(g_residency) residency_init();
  alloc_state();
  static const char *names[]={"blk.0.attn_qkv.weight","blk.0.ffn_gate.weight",
                              "blk.0.ffn_down.weight","blk.0.ssm_out.weight",
                              "output.weight"};
  const int Bs[]={1,2,4,8};
  int fails=0;
  /* --- GEMVB scaling + correctness vs B separate gemv calls --- */
  for(size_t ni=0;ni<sizeof names/sizeof*names;ni++){
    Tensor *t=find_tensor(names[ni]);
    if(!t){fprintf(stderr,"bench_spec: tensor '%s' not found\n",names[ni]);return 1;}
    int K=t->ne0,M=t->ne1,Bmax=8;
    float *x=aligned_alloc(64,(size_t)K*Bmax*4); /* Bmax activation columns */
    float *Y =aligned_alloc(64,(size_t)M*Bmax*4);
    float *Yr=aligned_alloc(64,(size_t)M*Bmax*4);
    if(!x||!Y||!Yr){fprintf(stderr,"oom\n");return 1;}
    uint32_t seed=0xc0ffeeu+(uint32_t)ni;
    bench_fill(x,K*Bmax,&seed);
    size_t wbytes=row_bytes(t->type,K)*(size_t)M;
    for(int bi=0;bi<4;bi++){
      int b=Bs[bi];
      memset(Y,0,(size_t)M*b*4); memset(Yr,0,(size_t)M*b*4);
      for(int bb=0;bb<b;bb++) gemv(t,x+(size_t)bb*K,Yr+(size_t)bb*M);
      gemvb(t,x,K,Y,M,b);
      float ma=0;
      for(size_t i=0;i<(size_t)M*b;i++){float d=fabsf(Y[i]-Yr[i]);if(d>ma)ma=d;}
      if(ma>1e-4f){
        printf("FAIL GEMVB %s B=%d max_abs=%.6g\n",names[ni],b,(double)ma);
        fails++; continue;
      }
      double t0=bench_now();
      for(int it=0;it<iters;it++) gemvb(t,x,K,Y,M,b);
      double el=bench_now()-t0; /* one sample feeds both ms and GB/s */
      double ms=el*1e3/(double)iters;
      double gbps=wbytes*(double)iters/el*1e-9;
      printf("PASS GEMVB %-16s B=%d %9.3f ms/iter %8.1f GB/s(w)\n",
             names[ni],b,ms,gbps);
      fflush(stdout);
    }
    free(x);free(Y);free(Yr);
  }
  /* --- snapshot pair --- */
  {
    int R=iters*8;
    double t0=bench_now();
    for(int i=0;i<R;i++){ snap_save(); snap_load(); }
    double ms=(bench_now()-t0)*1e3/(double)R;
    printf("%s SNAP pair %.3f ms (Srec+Cstate rollback)\n","PASS",ms);
  }
  /* --- verify-block sweep vs serial forwards ---
     Anchor state once (reset+prefill), then time R repeats of the block or
     the serial equivalent in place; pos_n grows by b per repeat, bounded
     well below MAX_SEQ, so both arms see the same growing-context regime.
     No baseline subtraction: under machine load it produced noise larger
     than the signal. */
  {
    uint32_t s=0x5eed1234u;
    enum { P=32 };
    int pre[P+SPEC_BMAX];
    for(int i=0;i<P+SPEC_BMAX;i++) pre[i]=(int)(bench_lcg(&s)%V);
    int fails_b=0;
    int R=iters*2; if(R<4) R=4;
    /* B=1 is not a block regime: the driver takes the serial step when no
       drafts were proposed, so sweep starts at 2. Arms alternate rep-by-rep
       (sequential phases tracked machine drift, not code) and keep min-of-R:
       shared-box preemption inflates means. */
    for(int bi=1;bi<4;bi++){
      int b=Bs[bi];
      reset_state(); bench_prefill(pre,P);
      double blkms=1e30,serms=1e30;
      for(int r=0;r<R;r++){
        /* both arms advance pos_n by b per rep: re-anchor before the block
           would run past MAX_SEQ (forward_block asserts the bound), so any
           ITERS stays inside the documented [1,INT_MAX] domain */
        if(pos_n+2*b+2>=g_ctx){ reset_state(); bench_prefill(pre,P); }
        double t0=bench_now();
        forward_block(pre+P,b,b);
        double dt=(bench_now()-t0)*1e3;
        if(dt<blkms) blkms=dt;
        t0=bench_now();
        for(int i=0;i<b;i++){ pos_n++; forward_ex(pre[P+i],1); }
        dt=(bench_now()-t0)*1e3;
        if(dt<serms) serms=dt;
      }
      int ok=blkms<=serms*1.05;   /* batching must not lose to serial */
      printf("%s BLOCK B=%d serial %8.1f ms  block %8.1f ms  x%.2f\n",
             ok?"PASS":"FAIL",b,serms,blkms,serms/(blkms>0?blkms:1e-9));
      fflush(stdout);
      if(!ok) fails_b++;
    }
    fails+=fails_b;
  }
  return fails?1:0;
}
#else
/* Production frontends (CLI decode path, persistent frame server) share the
   n-gram cache below; the fuzz build compiles that block too so
   tools/fuzz_loader.c can drive ng_load/ng_save across the persistence
   boundary, and stubs everything else out behind its own main. */
#ifndef CWEN_FUZZ_LOADER
/* ---- persistent decode server ----
    Binary frames on stdin: <u32 n_prompt><u32 n_gen><n_prompt * i32>.
    Binary reply on stdout: <u32 n_out><n_out * i32>. Little-endian, host = x86. */


/* Generated-token sinks: the server appends to its reply buffer, the CLI
   prints streaming style. One prefill+decode loop serves both frontends. */
typedef void (*tok_emit)(int token,int idx,void *ud);
typedef struct { int *out; int n; } tok_sink;
static void sink_store(int token,int idx,void *ud){
  (void)idx;
  tok_sink *s=ud;
  s->out[s->n++]=token;
}
static void sink_print(int token,int idx,void *ud){
  (void)ud; /* separator goes before each token; main closes the line, so an
               early context-cap stop cannot strand a trailing space */
  printf(idx?" %d":"%d",token);
  fflush(stdout);
}
#endif /* !CWEN_FUZZ_LOADER */

/* ---- n-gram drafter (prompt-lookup) ----
   Find the most recent earlier occurrence of the trailing key and propose the
   tokens that followed it. O(history) scan; history is <= 2*MAX_SEQ ints.
   Contract: propose up to max_draft confirmed-plausible ids from history. A
   trained DFlash2 drafter replaces this function without touching the
   verify machinery. */
static int ngram_draft(const int *h,int hn,int *out,int n_key,int max_d){
  if(hn<=n_key||max_d<=0) return 0;
  const int *tail=h+hn-n_key;
  /* most recent match wins ties, but an older match with a longer run of
     following tokens drafts more: keep scanning for the best continuation */
  int best=0;
  for(int j=hn-n_key-1;j>=0&&best<max_d;j--){
    if(memcmp(h+j,tail,(size_t)n_key*sizeof(int))) continue;
    int avail=hn-(j+n_key);
    if(avail>best){
      int n=avail<max_d?avail:max_d;
      memcpy(out,h+j+n_key,(size_t)n*sizeof(int));
      best=n;
    }
  }
  return best;
}


static uint64_t ng_hash(const int *k,int n){
  uint64_t h=1469598103934665603ull;
  for(int i=0;i<n;i++){
    h^=(uint64_t)(uint32_t)k[i];
    h*=1099511628211ull;
    h^=h>>29;
  }
  return h;
}
/* returns entry index or -1; *slot_out receives insertion point when absent */
static size_t ng_find(const int *key,int *found){
  uint64_t h=ng_hash(key,Scfg_n_key);
  size_t i=(size_t)h&(NGM.cap-1);
  for(size_t p=0;p<NGM.cap;p++,i=(i+1)&(NGM.cap-1)){
    if(!NGM.used[i]){ if(found)*found=0; return i; }
    if(!memcmp(NGM.keys+(size_t)i*Scfg_n_key,key,(size_t)Scfg_n_key*sizeof(int))){
      if(found)*found=1;
      return i;
    }
  }
  if(found)*found=0;
  return NGM.cap; /* full */
}
static void ng_evict_one(void); /* LFU eviction, defined below ng_update_key */
/* hits>1 is the cache-restore path (ng_load): persisted counts re-enter the
   map so LFU eviction keeps cross-run frequency information. Merges saturate
   at INT_MAX instead of wrapping. */
static void ng_update_key_n(const int *key,int tok,int hits){
  int found;
  size_t i=ng_find(key,&found);
  if(found){
    NGM.cnt[i]=hits<INT_MAX-NGM.cnt[i]?NGM.cnt[i]+hits:INT_MAX;
    NGM.tok[i]=tok;
    return;
  }
  if(i==NGM.cap){                 /* full: evict least-hit, keep learning */
    ng_evict_one();
    i=ng_find(key,&found);
  }
  memcpy(NGM.keys+(size_t)i*Scfg_n_key,key,(size_t)Scfg_n_key*sizeof(int));
  NGM.tok[i]=tok; NGM.cnt[i]=hits; NGM.used[i]=1; NGM.used_n++;
}
static void ng_update_key(const int *key,int tok){ ng_update_key_n(key,tok,1); }
/* Table at cap and a fresh key wants in: drop one least-hit entry (llama.cpp
   ngram-cache eviction) instead of freezing the key set forever. Equal-count
   ties rotate through the table (NG_evict_hand) so victims spread instead of
   hammering one corner. Backward shift after the removal keeps linear-probe
   chains intact: a successor moves into the hole iff its home is not
   cyclically inside (hole, slot]. */
static size_t NG_evict_hand;
static void ng_evict_one(void){
  int mn=INT_MAX;
  for(size_t s=0;s<NGM.cap;s++)
    if(NGM.used[s]&&NGM.cnt[s]<mn) mn=NGM.cnt[s];
  size_t victim=NGM.cap;
  for(size_t k=0;k<NGM.cap&&victim==NGM.cap;k++){
    size_t s=(NG_evict_hand+k)&(NGM.cap-1);
    if(NGM.used[s]&&NGM.cnt[s]==mn) victim=s;
  }
  NG_evict_hand=(victim+1)&(NGM.cap-1);
  size_t hole=victim;
  for(size_t step=1;step<NGM.cap;step++){
    size_t j=(victim+step)&(NGM.cap-1);
    if(!NGM.used[j]) break;
    size_t home=(size_t)ng_hash(NGM.keys+(size_t)j*Scfg_n_key,Scfg_n_key)
               &(NGM.cap-1);
    size_t dh=(home-hole)&(NGM.cap-1), dj=(j-hole)&(NGM.cap-1);
    if(dh==0||dh>dj){
      memcpy(NGM.keys+(size_t)hole*Scfg_n_key,
             NGM.keys+(size_t)j*Scfg_n_key,(size_t)Scfg_n_key*sizeof(int));
      NGM.tok[hole]=NGM.tok[j]; NGM.cnt[hole]=NGM.cnt[j];
      NGM.used[hole]=1; NGM.used[j]=0;
      hole=j;
    }
  }
  NGM.used_n--;
}
/* fold hist[a..b) positions: every position p with p>=NKEY contributes
   key=hist[p-NKEY..p), tok=hist[p] */
static void ng_update_range(const int *h,int a,int b){
  if(!NGM.keys||Scfg_n_key<1) return;
  for(int p=a;p<b;p++){
    if(p-Scfg_n_key<0) continue;
    ng_update_key(h+p-Scfg_n_key,h[p]);
  }
}
/* chained proposals: tail lookup gives one token, shift, repeat */
static int ng_draft_map(const int *h,int hn,int *out,int max_d){
  static int key[256]; /* Scfg_n_key<=256 (validated) */
  if(hn<Scfg_n_key||max_d<=0) return 0;
  memcpy(key,h+hn-Scfg_n_key,(size_t)Scfg_n_key*sizeof(int));
  int n=0;
  while(n<max_d){
    int found;
    size_t i=ng_find(key,&found);
    if(!found) break;
    out[n]=NGM.tok[i];
    memmove(key,key+1,(size_t)(Scfg_n_key-1)*sizeof(int));
    key[Scfg_n_key-1]=out[n];
    n++;
  }
  return n;
}
/* Format NGC2: {magic,n_key,vocab,used} then records of {key[nk],tok,cnt}.
   n_key+vocab in the header refuse maps from another config/model instead of
   silently merging foreign continuations. */
#define NG_MAGIC 0x3243474eu /* "NGC2" LE */
static void ng_save(const char *path){
  char tmp[512];
  snprintf(tmp,sizeof tmp,"%s.tmp",path);
  FILE *f=fopen(tmp,"wb");
  if(!f){perror(tmp);return;}
  uint32_t nk=(uint32_t)Scfg_n_key, nwv=(uint32_t)V, used=(uint32_t)NGM.used_n;
  int bad=fwrite(&(uint32_t){NG_MAGIC},4,1,f)!=1;
  bad|=fwrite(&nk,4,1,f)!=1 || fwrite(&nwv,4,1,f)!=1 || fwrite(&used,4,1,f)!=1;
  for(size_t i=0;!bad&&i<NGM.cap;i++){
    if(!NGM.used[i]) continue;
    bad|=fwrite(NGM.keys+(size_t)i*Scfg_n_key,4,(size_t)Scfg_n_key,f)
         !=(size_t)Scfg_n_key;
    bad|=fwrite(&NGM.tok[i],4,1,f)!=1 || fwrite(&NGM.cnt[i],4,1,f)!=1;
  }
  bad|=fclose(f)!=0; /* close even after a short write: never strand the fd */
  if(bad){fprintf(stderr,"ngram cache: %s: write failed\n",tmp);remove(tmp);return;}
  if(rename(tmp,path)){perror(path);remove(tmp);}
}
static const char *NG_save_path;
static void ng_save_atexit(void){
  if(NG_save_path&&NG_save_path[0]&&NGM.used_n) ng_save(NG_save_path);
}
static void ng_load(const char *path){
  FILE *f=fopen(path,"rb");
  if(!f){
    /* missing file = cold start, not an error */
    fprintf(stderr,"ngram cache: %s: new map\n",path);
    return;
  }
  /* The map is an optional accelerator whose drafts are verified anyway: any
     load problem (stale format, other config, truncation) degrades to a cold
     start instead of failing the run. */
  uint32_t magic,nk,nwv,used;
  if(fread(&magic,4,1,f)!=1||fread(&nk,4,1,f)!=1||
     fread(&nwv,4,1,f)!=1||fread(&used,4,1,f)!=1){
    fprintf(stderr,"cwen: ngram cache %s: short header; ignoring\n",path);
    fclose(f); return;
  }
  if(magic!=NG_MAGIC){
    fprintf(stderr,"cwen: ngram cache %s: unknown format; ignoring\n",path);
    fclose(f); return;
  }
  if((int)nk!=Scfg_n_key||(int)nwv!=V){
    fprintf(stderr,"cwen: ngram cache %s: built with N=%u V=%u (running N=%d V=%d); ignoring\n",
            path,nk,nwv,Scfg_n_key,V);
    fclose(f); return;
  }
  if(fseek(f,16,SEEK_SET)){
    fprintf(stderr,"cwen: ngram cache %s: seek failed; ignoring\n",path);
    fclose(f); return;
  }
  static int key[256];
  uint32_t read_n=0;
  for(uint32_t i=0;i<used;i++){
    if(NGM.used_n>=NGM.cap) break; /* map full: nothing further can be added */
    if(fread(key,4,nk,f)!=(size_t)nk) break;
    int tok,cnt;
    if(fread(&tok,4,1,f)!=1||fread(&cnt,4,1,f)!=1) break; /* truncated tail */
    if(tok<0||tok>=V||cnt<1) continue;
    ng_update_key_n(key,tok,cnt); /* counts merge into the live map */
    read_n++;
  }
  fclose(f);
  fprintf(stderr,"ngram cache: loaded %u entries from %s\n",read_n,path);
}

#ifdef CWEN_FUZZ_LOADER
/* qsort comparator for packed {key[n_key],tok,cnt} rows; row width lives in
   a file-scope int because qsort comparators take no context. */
static int cw_fuzz_row;
static int cw_fuzz_rowcmp(const void *x,const void *y){
  return memcmp(x,y,(size_t)cw_fuzz_row*sizeof(int));
}
/* ---- ngram-cache harness hook (tools/fuzz_loader.c) ----
   Drive ng_load over an arbitrary NGC2 file against a fresh map, assert the
   post-load structural invariants (occupancy matches used_n; every entry
   carries an in-range continuation and a positive count), then cross back
   over the persistence boundary: ng_save to a second file, reload into a
   second fresh map, and diff the entry sets including hit counts. Any
   save/load drift (lost entries, lost counts, altered continuations)
   returns 2 and the harness aborts, so a deserialization regression becomes
   a fuzzer-visible liveness failure instead of silent cache decay.
   Returns 0 clean, 1 exit-rejected, 2 invariant or round-trip break. */
int cw_fuzz_ngcache(const char *path,const char *path2){
  static NGMap nm2;               /* reload target for the round trip */
  /* Both maps keep process lifetime: ~40 MiB resident beats per-iteration
     calloc churn under ASan quarantine. Occupancy resets each call. */
  if(!NGM.keys){
    NGM.cap=NG_CAP;
    NGM.keys=calloc(NGM.cap*(size_t)Scfg_n_key,sizeof(int));
    NGM.tok=calloc(NGM.cap,sizeof(int));
    NGM.cnt=calloc(NGM.cap,sizeof(int));
    NGM.used=calloc(NGM.cap,1);
    nm2.cap=NG_CAP;
    nm2.keys=calloc(nm2.cap*(size_t)Scfg_n_key,sizeof(int));
    nm2.tok=calloc(nm2.cap,sizeof(int));
    nm2.cnt=calloc(nm2.cap,sizeof(int));
    nm2.used=calloc(nm2.cap,1);
    if(!NGM.keys||!NGM.tok||!NGM.cnt||!NGM.used||
       !nm2.keys||!nm2.tok||!nm2.cnt||!nm2.used) abort();
  }
  memset(NGM.used,0,NGM.cap); NGM.used_n=0;
  memset(nm2.used,0,nm2.cap); nm2.used_n=0;

  int rc=0;
  cw_fuzz_armed=0;
  if(setjmp(cw_fuzz_jmp)){ cw_fuzz_armed=0; rc=1; }
  if(!rc){
    cw_fuzz_armed=1;
    ng_load(path);
    cw_fuzz_armed=0;
    size_t live=0;
    for(size_t i=0;i<NGM.cap;i++){
      if(!NGM.used[i]) continue;
      live++;
      if(NGM.tok[i]<0||NGM.tok[i]>=V||NGM.cnt[i]<1){ rc=2; break; }
    }
    if(rc==2||(size_t)live!=NGM.used_n) rc=2;
    else{
      ng_save(path2);
      /* ng_load fills the global map: swing NGM onto the secondary buffers
         for the reload, then swing the primary back for the diff */
      NGMap tmp=NGM; NGM=nm2;
      cw_fuzz_armed=1;
      ng_load(path2);
      cw_fuzz_armed=0;
      nm2=NGM; NGM=tmp;
      /* exact entry-set diff: sort packed {key,tok,cnt} rows from both maps
         and compare, so insertion-order differences cannot mask drift */
      size_t row=(size_t)Scfg_n_key+2;
      int *a=malloc(live*row*sizeof(int)),*b=malloc(live*row*sizeof(int));
      if(!a||!b) abort();
      size_t na=0,nb=0;
      const NGMap *maps[2]={&NGM,&nm2};
      int *dst[2]={a,b}; size_t *cnt_out[2]={&na,&nb};
      for(unsigned m=0;m<2;m++)
        for(size_t i=0;i<maps[m]->cap;i++){
          if(!maps[m]->used[i]) continue;
          int *r=dst[m]+(*cnt_out[m])*row;
          memcpy(r,maps[m]->keys+(size_t)i*Scfg_n_key,
                 (size_t)Scfg_n_key*sizeof(int));
          r[Scfg_n_key]=maps[m]->tok[i]; r[Scfg_n_key+1]=maps[m]->cnt[i];
          (*cnt_out[m])++;
        }
      cw_fuzz_row=(int)row;
      qsort(a,na,row,cw_fuzz_rowcmp); qsort(b,nb,row,cw_fuzz_rowcmp);
      if(na!=nb||memcmp(a,b,na*row*sizeof(int))) rc=2;
      free(a); free(b);
    }
  }
  return rc;
}
#endif /* CWEN_FUZZ_LOADER */

#ifndef CWEN_FUZZ_LOADER
/* ---- prefill ----
   Prompt tokens score through forward_block in PF_CHUNK-sized batches: every
   weight matrix streams once per chunk instead of once per token (the same
   batching the verify pass uses). Per-row math is bit-identical to the serial
   walk (same dot kernels, GDN/conv stay sequential, attention sees the same
   absolute slots), so decode chains and drafter taps are unchanged. Dump mode
   stays fully serial: CWEN_DUMP bytes are a property of the serial path.
   On return pos_n covers tokens[0..n_tok-1] and `logits` predicts after
   tokens[n_tok-1] (dump mode recomputes it per position inside forward_ex). */
static void prefill_forward(const int *tokens,int n_tok){
  pos_n=0;
  const char *ddir=dump_dir();
  if((ddir&&ddir[0])||n_tok<2){
    for(int i=0;i<n_tok;i++){
      forward_ex(tokens[i], ddir&&ddir[0] ? 1 : i+1==n_tok);
      if(dflash_on) dflash_commit(pos_n,DTapSer);
      mtp_commit(pos_n,tokens[i],x);
      if(i+1<n_tok) pos_n++;
    }
    return;
  }
  forward_ex(tokens[0],0);            /* seeds slot 0; forward_block appends */
  if(dflash_on) dflash_commit(0,DTapSer);
  mtp_commit(0,tokens[0],x);
  int i=1;
  while(i<n_tok){
    int B=n_tok-i;
    if(B>PF_CHUNK) B=PF_CHUNK;
    if(B>g_ctx-1-pos_n) B=g_ctx-1-pos_n;
    if(B<=0) break;                   /* context cap reached */
    forward_block(tokens+i,B,0);
    if(dflash_on)
      for(int b=0;b<B;b++)
        dflash_commit(pos_n-B+1+b,DTapBlk+(size_t)b*DL_TAPN*H);
    for(int b=0;b<B;b++)
      mtp_commit(pos_n-B+1+b,tokens[i+b],BXres+(size_t)b*H);
    if(i+B==n_tok){
      /* lm_head once, into the global the argmax reads (skipped in-block) */
      rmsnorm(xb,BXres+(size_t)(B-1)*H,(const float*)output_norm.data,H);
      gemv(&output,xb,logits);
    }
    i+=B;
  }
}

/* Greedy block speculation. Bit-identical output to generate_tokens' serial
   argmax loop: every emitted token is either an argmax of true target logits
   or a draft verified against them.

   Cycle invariant at the top: state covers hist[0..pos_n], hist[hn-1] is the
   pending token `pend` (confirmed but not yet in state), and `logits` from
   the previous cycle already scored it. The block [pend, drafts...] is scored
   by one forward_block sweep; Blogits[b] predicts after block[b]. The walk
   accepts draft k iff it equals the argmax of the logits at the preceding
   position; the first mismatch's argmax (or the last block position's) is
   the bonus token, again a plain target argmax. Full accepts keep the block
   state; short walks restore the GDN snapshot and replay only the kept
   prefix through the serial path. */
/* Greedy argmax has no EOS, so a count short of n_gen has exactly one cause:
   the context window filled first. Say so instead of leaving the caller to
   mistake truncation for a complete answer. */
static void warn_ctx_short(int n,int n_gen){
  if(n<n_gen)
    fprintf(stderr,
            "cwen: context window full (CWEN_CTX=%d); emitted %d of %d requested tokens\n",
            g_ctx,n,n_gen);
}

static int generate_tokens_spec(const int *tokens,int n_tok,int n_gen,
                                tok_emit emit,void *emit_ud){
  static int hist[2*MAX_SEQ];
  static int blk[SPEC_BMAX];
  if(dump_dir()&&dump_dir()[0])
    fprintf(stderr,"cwen: CWEN_DUMP covers prefill only under CWEN_SPEC=1\n");
  prefill_forward(tokens,n_tok);
  for(int i=0;i<n_tok;i++) hist[i]=tokens[i];
  long long cyc=0,draft_cyc=0,full=0,rej=0,acc_sum=0;  int hn=n_tok,upd=0;
  /* first pick: argmax over prefill logits becomes the pending token.
     n_gen==0 means prefill/dump only: emit nothing, like the serial path. */
  int g=0,n=0,streak=0,cool=0;
  /* Same room gate as the serial loop's first iteration: a prompt that
     filled the window emits nothing, so both drivers stay stream-identical
     at the cap (n_gen==0 means prefill/dump only: emit nothing either way). */
  if(n_gen>0&&pos_n+1<g_ctx){
    int pend=argmax_logits();
    hist[hn++]=pend;
    emit(pend,0,emit_ud);
    g=1; n=1;
  }
  if(NGM.keys){ ng_update_range(hist,upd,hn); upd=hn; }
  int spec_debug=Scfg_debug; /* parsed+validated once in spec_config_init */
  /* DL_BLOCK bounds the trained drafter's own walk; the verify block itself
     only has to fit SPEC_BMAX, which Scfg_max_draft is already checked against */
  const int cap_hard=(dflash_on&&Scfg_max_draft>DL_BLOCK)?DL_BLOCK:Scfg_max_draft;
  int E_cap=cap_hard;
  while(g<n_gen){
    if(pos_n+1>=g_ctx) break;            /* same context cap as serial path */
    int E=0;
    if(cool>0){ cool--; }
    else{
      int room=g_ctx-1-(pos_n+1);        /* block slots must stay in range */
      if(dflash_on){
        /* the trained drafter proposes every cycle; min_draft still gates
           tiny proposals, cooldown still protects against rejection streaks */
        int cap=E_cap<DL_BLOCK?E_cap:DL_BLOCK;
        E=dflash_draft(hist,hn,blk+1,cap<room?cap:room);
      }else if(mtp_use){
        E=mtp_draft(hist[hn-1],blk+1,E_cap<room?E_cap:room);
      }else{
        int cap=E_cap<room?E_cap:room;
        static int scan_tmp[SPEC_BMAX];
        if(NGM.keys){
          /* chained map lookups first; the history scan still wins when it
             can propose a longer verified-pattern run */
          E=ng_draft_map(hist,hn,blk+1,cap);
          int esc=ngram_draft(hist,hn,scan_tmp,Scfg_n_key,cap);
          if(esc>E){ E=esc; memcpy(blk+1,scan_tmp,(size_t)E*sizeof(int)); }
        }else{
          E=ngram_draft(hist,hn,blk+1,Scfg_n_key,cap);
        }
      }
      if(E&&E<Scfg_min_draft) E=0;
    }
    blk[0]=hist[hn-1];                   /* pend */
    if(spec_debug)
      fprintf(stderr,"spec dbg: cyc=%lld hn=%d pos_n=%d E=%d tail0=%d\n",
              cyc,hn,pos_n,E,hist[hn-1]);
    int k=0,ustar;
    cyc++;
    if(E==0){
      /* no proposal: plain step, exactly the serial decode cost */
      pos_n++;
      forward_ex(blk[0],1);
      if(dflash_on) dflash_commit(pos_n,DTapSer);
      mtp_commit(pos_n,blk[0],x);
      ustar=argmax_logits();
    }else{
      int B=E+1;
      snap_save(); draft_cyc++;
      forward_block(blk,B,B);
      const float *cur=Blogits;          /* predicts after pend */
      while(k<E&&argmax_of(cur)==blk[k+1]){ k++; cur=Blogits+(size_t)k*V; }
      ustar=argmax_of(cur);
      acc_sum+=k;                        /* drafts kept this cycle */
      if(k==E) full++;
      else rej++;
      /* Adaptive draft sizing (AIMD on full accepts). A short walk is what
         costs: the block sweep is wasted past row k and the kept prefix has
         to be re-scored. Acceptance *rate* is the wrong signal for that --
         drafting 6 and keeping 5 rates well and short-walks every time --
         so drop straight to what the target actually took, and probe upward
         only after a run of clean cycles. */
      {
        static int fa_streak=0;
        if(k==E){
          if(++fa_streak>=4 && E_cap<cap_hard){ E_cap++; fa_streak=0; }
        }else{
          fa_streak=0;
          E_cap = k>Scfg_min_draft ? k : Scfg_min_draft;
        }
      }
      if(k<E){
        /* short walk: undo the whole block (drafts and pend were committed
           by forward_block), then re-score pend + the accepted drafts. One
           more block sweep, not k+1 serial ones: the weights stream once
           either way and ustar already came from Blogits, so the head is
           skipped too. Full accepts keep the block state as-is. */
        snap_load();
        pos_n-=B;
        forward_block(blk,k+1,0);
        for(int b=0;b<=k;b++){
          if(dflash_on)
            dflash_commit(pos_n-k+b,DTapBlk+(size_t)b*DL_TAPN*H);
          mtp_commit(pos_n-k+b,blk[b],BXres+(size_t)b*H);
        }
      }else{
        /* every block row became context: commit row by row */
        for(int b=0;b<B;b++){
          if(dflash_on)
            dflash_commit(pos_n-B+1+b,DTapBlk+(size_t)b*DL_TAPN*H);
          mtp_commit(pos_n-B+1+b,blk[b],BXres+(size_t)b*H);
        }
      }
    }
    /* confirmed: accepted drafts are real context too; record them so hist
       stays 1:1 with the positions state covers (cycle invariant above) */
    for(int i=0;i<k;i++) hist[hn++]=blk[i+1];
    hist[hn++]=ustar;
    if(NGM.keys){ ng_update_range(hist,upd,hn); upd=hn; }
    /* pend was already emitted when first confirmed; now the accepted
       drafts and the fresh bonus */
    for(int i=0;i<k&&g<n_gen;i++){ emit(blk[i+1],g,emit_ud); g++; n++; }
    if(g>=n_gen) break;
    emit(ustar,g,emit_ud); g++; n++;
    if(k==E) streak=0; else if(++streak>=3) { streak=0; cool=Scfg_cooldown; }
  }
  warn_ctx_short(n,n_gen);
  fprintf(stderr,"spec: %lld cycles (%lld drafted, %lld full accept, %lld short; "
                 "avg kept %.2f)\n",cyc,draft_cyc,full,rej,
          draft_cyc?(double)acc_sum/(double)draft_cyc:0.0);
  if(mtp_use&&Scfg_debug)
    fprintf(stderr,"mtp: step %.2fs/%ld  head %.2fs/%ld  commit %.2fs\n",
            mtp_t_step,mtp_n_step,mtp_t_head,mtp_n_head,mtp_t_commit);
  return n;
}


/* Prefill, then argmax-decode. Stops at the MAX_SEQ context cap (a token is
   only emitted while there is room left to append it; a short count warns on
   stderr via warn_ctx_short) and skips the final forward unless CWEN_DUMP
   needs its logits. Returns tokens emitted.
   CWEN_SPEC=1 routes to the block-speculative driver instead; greedy output
   is identical, only faster when history repeats. */
static int generate_tokens(const int *tokens, int n_tok, int n_gen,
                            tok_emit emit, void *emit_ud) {
  if(spec_enabled) return generate_tokens_spec(tokens,n_tok,n_gen,emit,emit_ud);
  prefill_forward(tokens,n_tok);
  int n = 0;
  for (int g = 0; g < n_gen; g++) {
    if (pos_n + 1 >= g_ctx) break;
    int next = argmax_logits();
    emit(next, g, emit_ud);
    n++;
    /* forward after the last token is only needed for CWEN_DUMP logits */
    const char *d = dump_dir();
    if (g + 1 == n_gen && !(d && d[0])) break;
    pos_n++;
    forward_ex(next, 1);
  }
  warn_ctx_short(n,n_gen);
  return n;
}

/* Reply frames are the server's only output. SIGPIPE covers a closed pipe;
   reaching a failed fwrite means real I/O trouble (redirected stdout on a
   full disk, closed fd), and continuing would drop tokens while reporting
   success to nobody. */
static void wr_reply_or_die(const void *p, size_t n4) {
  if(!n4) return;
  if(fwrite(p,4,n4,stdout)!=n4){
    fprintf(stderr,"cwen server: reply write failed\n");
    exit(1);
  }
}

static void wr_u32_stdout(uint32_t v) {
  wr_reply_or_die(&v,1);
}

static int server_loop(void) {
  static int tokens[MAX_SEQ];
  for (;;) {
    uint32_t n_prompt,n_gen;
    int r=server_read_frame(stdin,tokens,&n_prompt,&n_gen);
    if(r<0) return 0; /* EOF: clean close between frames or mid-frame */
    if(r==0){ wr_u32_stdout(0); fflush(stdout); continue; }
    reset_state();
    static int out[MAX_SEQ];
    tok_sink sink = { .out = out, .n = 0 };
    int n_out = generate_tokens(tokens, (int)n_prompt, (int)n_gen, sink_store, &sink);
    wr_u32_stdout((uint32_t)n_out);
    wr_reply_or_die(out,(size_t)n_out);
    if(fflush(stdout)){ fprintf(stderr,"cwen server: reply flush failed\n"); exit(1); }
  }
}

/* ---- main ---- */
static void run_usage(FILE *out) {
  fprintf(out,
    "usage: run [MODEL] [IDS_FILE] [N_PREDICT] [OPTIONS]\n"
    "Argmax-generate token ids from raw little-endian int32 prompt ids.\n"
    "Generated ids go to stdout; logs go to stderr.\n"
    "\n"
    "Arguments:\n"
    "  MODEL       GGUF model path (default: model/Qwen3.8-27B-Q4_0.gguf)\n"
    "  IDS_FILE    file of int32 token ids; omitted = single default token\n"
    "  N_PREDICT   tokens to generate, >=0 (default: 16; 0 = prefill/dump only)\n"
    "\n"
    "Options:\n"
    "  -h, --help              this help\n"
    "  -d, --draft-tokens N    speculate up to N drafted tokens per verify\n"
    "                          block (1..15); implies CWEN_SPEC=1 and\n"
    "                          overrides CWEN_SPEC_MAX_DRAFT. Alias:\n"
    "                          --spec-draft-n-max\n"
    "\n"
    "Environment:\n"
    "  CWEN_DUMP=DIR          dump per-layer activations for verification\n"
    "  CWEN_DUMP_LAYERS=N     limit dumped layers\n"
    "  CWEN_DUMP_LOGITS=1     also dump logits under a layer subset\n"
    "  CWEN_REPACK=FILE       CWENR sidecar (auto: MODEL with .cwenr)\n"
    "  CWEN_OMP_THREADS=N     OpenMP threads (OMP_NUM_THREADS wins)\n"
    "  CWEN_SERVER=1          persistent binary-frame server on stdin/stdout\n"
    "  CWEN_CTX=N             context window, 64..32768 (default 4096)\n"
    "  CWEN_ROPE_YARN=o,f     YaRN long-context scaling: original max,\n"
    "                         factor [,beta_fast,beta_slow]; e.g. 8192,4\n"
    "  CWEN_RESIDENCY=1       THP+mlock+prefault+next-layer PF\n"
    "    CWEN_PF_T0=1             prefetch weights to L1 (T0) instead of NTA\n"
    "    CWEN_NO_PF=1             disable weight prefetch entirely\n"
    "    CWEN_PIPE_PF=1           prefetch the next layer's first weights\n"
    "  CWEN_NGRAM_CACHE=FILE  persist n-gram map across runs (needs CWEN_SPEC)\n"
    "  CWEN_SPEC=1            block speculation (greedy-lossless). Drafter:\n"
    "                         CWEN_DFLASH, else the model's MTP nextn head\n"
    "                         when present, else the n-gram map\n"
    "  CWEN_MTP=0             do not use the MTP nextn head as the drafter\n"
    "  CWEN_DFLASH=FILE       trained DFlash2 drafter (.spec, see\n"
    "                         tools/pack_dflash.py); implies CWEN_SPEC=1\n"
    "    CWEN_SPEC_NGRAM_N=N      lookup key length (default 16)\n"
    "    CWEN_SPEC_MAX_DRAFT=N    max drafts per block, 1..15 (default 8)\n"
    "    CWEN_SPEC_MIN_DRAFT=N    skip blocks shorter than N drafts (default 2)\n"
    "    CWEN_SPEC_COOLDOWN=N     plain steps after repeated full rejects (8)\n"
    "    CWEN_SPEC_DEBUG=1        per-cycle speculation trace on stderr\n");
}

int main(int argc, char **argv) {
  const char *model="model/Qwen3.8-27B-Q4_0.gguf";
  const char *ids_path=NULL;
  int n_predict=16;
  const char *pos[3]; int npos=0;
  int draft_tokens=-1; /* -1 = flag absent; env knobs stay in charge */
  for(int i=1;i<argc;i++){
    const char *a=argv[i];
    if(!strcmp(a,"-h")||!strcmp(a,"--help")){run_usage(stdout);return 0;}
    if(a[0]=='-'&&a[1]){
      int is_draft=!strcmp(a,"-d")||!strcmp(a,"--draft-tokens")
                   ||!strcmp(a,"--spec-draft-n-max");
      if(!is_draft){
        fprintf(stderr,"cwen: unknown option '%s'\n\n",a);
        run_usage(stderr); return 2;
      }
      if(i+1>=argc){
        fprintf(stderr,"cwen: %s requires a value\n\n",a);
        run_usage(stderr); return 2;
      }
      draft_tokens=arg_int_range(argv[++i],1,SPEC_BMAX-1,"cwen: ",a);
      continue;
    }
    if(npos>=3){
      fprintf(stderr,"cwen: too many arguments (got %d, max 3)\n\n",npos+1);
      run_usage(stderr); return 2;
    }
    pos[npos++]=a;
  }
  if(npos>=1) model=pos[0];
  if(npos>=2) ids_path=pos[1];
  if(npos>=3) n_predict=arg_int_range(pos[2],0,INT_MAX,"cwen: ","N_PREDICT");

  cwen_omp_init(argc,argv); /* before load_model: CWENR v2 split forks an OMP team */
  spec_config_init();
  { int v;
    if(env_int("CWEN_RESIDENCY",&v)&&v){
      g_residency=1; g_pipe_pf=1; g_pf_t0=1;
    }
    if(env_int("CWEN_PF_T0",&v)) g_pf_t0=v;
    if(env_int("CWEN_NO_PF",&v)) g_no_pf=v;
    if(env_int("CWEN_PIPE_PF",&v)) g_pipe_pf=v;
  }
  rope_env_init();
  int server_mode=0;
  env_bool("CWEN_SERVER",&server_mode); /* fail fast before the model load */
  dump_dir_preflight();                 /* same: bad CWEN_DUMP dies here */
  const char *dfpath=getenv("CWEN_DFLASH");
  if(dfpath&&dfpath[0]){ dflash_on=1; spec_enabled=1; }
  /* Inert-knob note, also before the load: CWEN_NGRAM_CACHE only feeds the
     n-gram drafter; with spec off or the trained drafter active it would
     otherwise do nothing without saying anything. */
  { const char *ngc=getenv("CWEN_NGRAM_CACHE");
    if(ngc&&ngc[0]&&!(spec_enabled&&!dflash_on))
      fprintf(stderr,"ngram cache: %s: ignored, needs CWEN_SPEC=1 and no "
              "CWEN_DFLASH\n",ngc); }
  if(draft_tokens>0){
    /* CLI wins over env, like other engines' speculative flags */
    Scfg_max_draft=draft_tokens;
    spec_enabled=1;
    if(Scfg_min_draft>Scfg_max_draft) Scfg_min_draft=Scfg_max_draft;
  }
  load_model(model);
  if(dflash_on) load_dflash(dfpath);
  /* nextn only drafts when speculation is on and no trained drafter outranks
     it; CWEN_MTP=0 forces it off so the n-gram path can be A/B'd on a model
     that carries blk.64. */
  { int want=1; env_bool("CWEN_MTP",&want);
    mtp_use = mtp_on && want && spec_enabled && !dflash_on; }
  alloc_state();
  if(g_residency) residency_init();
  { const char *ngc=getenv("CWEN_NGRAM_CACHE");
    static char ngc_path[512];
    if(ngc&&ngc[0]&&spec_enabled&&!dflash_on){
      snprintf(ngc_path,sizeof ngc_path,"%s",ngc);
      NG_save_path=ngc_path;
      ng_load(ngc_path);
      atexit(ng_save_atexit);
    } }
  (void)dump_layers_lim(); /* fail fast on bad CWEN_DUMP_LAYERS, not mid-run */
  (void)dump_logits_flag(); /* same: reject a bad CWEN_DUMP_LOGITS up front */
  if (server_mode) {
    fprintf(stderr, "cwen server: model loaded, waiting on stdin\n");
    return server_loop();
  }
  int tokens[MAX_SEQ]; int n_tok=0;
  if(ids_path){
    FILE *f=fopen(ids_path,"rb"); if(!f){perror(ids_path);return 1;}
    while(n_tok<g_ctx && fread(&tokens[n_tok],4,1,f)==1){
      if(tokens[n_tok]<0||tokens[n_tok]>=V){
        fprintf(stderr,"%s: token %d out of range [0,%d)\n",ids_path,tokens[n_tok],V);
        fclose(f); return 1;
      }
      n_tok++;
    }
    int rderr=ferror(f);
    struct stat st;
    int stok=!fstat(fileno(f),&st);
    fclose(f);
    if(rderr){fprintf(stderr,"%s: read error\n",ids_path);return 1;}
    if(stok && st.st_size>0){
      if((unsigned long long)st.st_size%4u)
        fprintf(stderr,"%s: warning: size %lld is not a multiple of 4; ignoring trailing bytes\n",
                ids_path,(long long)st.st_size);
      if((long long)(st.st_size/4)>n_tok)
        fprintf(stderr,"%s: prompt exceeds CWEN_CTX=%d; generating from the first %d tokens\n",
                ids_path,g_ctx,n_tok);
    }
  } else {
    tokens[0]=248044; n_tok=1;
  }
  if(n_tok<1){fprintf(stderr,"cwen: %s: no tokens in prompt\n",ids_path);return 1;}

  fprintf(stderr,"prefill %d tokens...\n",n_tok);
  int emitted=generate_tokens(tokens,n_tok,n_predict,sink_print,NULL);
  if(emitted>0) putchar('\n');
  /* Same policy as wr_reply_or_die: tokens are the product. SIGPIPE covers a
     closed pipe, but a latched stdout error (redirected stdout on a full
     disk) must not exit 0 after dropping output. */
  if(fflush(stdout)||ferror(stdout)){
    fprintf(stderr,"cwen: stdout write failed\n");
    return 1;
  }
  return 0;
}
#endif /* !CWEN_FUZZ_LOADER */
#endif
