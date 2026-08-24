# cwen agent rules

Correctness and security first, then speed, then style. Prefer a lean `run.c`; no hard LOC cap.

## Fusion style

Fuse for speed without mega-functions (`gemv2`, `gemv3`, `layer_fused_all`). Keep separate, named steps; fusion is inlining + global buffers.

### Rules

1. **Globals hold activations and scratch.** Use file-scope buffers (`x`, `xb`, `hb`, `hb2`, `qkvb`, `qh`, `kh`, `vh`) instead of packing pointers. New temporaries use globals, not heap args.

2. **Kernels are `static inline` and small.** Row/elementwise ops (`gemv_row`, `residual_add`, `silu_mul`) must be `static inline`.

3. **Fuse at the call site with one parallel loop.** When gemvs share `x` and row counts match, use one `#pragma omp parallel for` that calls inlined row kernels back-to-back:

   ```c
   /* good: one team, two gemvs */
   #pragma omp parallel for schedule(static)
   for (int i = 0; i < M; i++) {
     ya[i] = gemv_row(Wa, x, i);
     yb[i] = gemv_row(Wb, x, i);
   }
   ```

   ```c
   /* bad: frozen multi-output API */
   gemv2(Wa, Wb, x, ya, yb);
   gemv3(Wa, Wb, Wc, x, ya, yb, yc);
   ```

4. **Keep single-mat `gemv`.** Fuse only at shared-activation sites.

5. **Two OMP teams won't fuse.** Use one loop (or a non-OMP sequential region) that inlines both kernels. Elementwise steps (`silu_mul`, `residual_add`) stay separate inline functions.

6. **Match shapes.** Fuse only when `ne1` (and usually `ne0`/type) match. Different shapes stay separate `gemv` calls.

### Existing fusion

| Site | Shared input | Outputs |
|------|--------------|---------|
| `mlp` | `xb` | `hb` ← gate, `hb2` ← up |
| `layer_full` | `xb` | `kh` ← wk, `vh` ← wv |

### Adding fusion

- Add or reuse a `static inline` row/element kernel.
- Write the fused OMP loop at the use site, not a new gemv API.
- Keep gemv goldens green (`make verify AVX512=1`); regenerate dumps with `make golden` after weight changes. The pinned decode chain lives in `tools/test_speed_gates.sh` (`EXPECT`: Qwen3.8-27B, BOS 248044); re-pin only via `make e2e-full-c`.

## Norms

- Pure C, single-file `run.c`; tools in `tools/`.
- Offline Q4_0 → CWENR via `make repack` / `tools/repack_q4.py`; no at-load heap repack.
- Zen: default OpenMP on one CCD (`CWEN_OMP_THREADS` 16); `make AVX512=1` for peak.
- No AI attribution in commits or comments. No em dashes in prose.
