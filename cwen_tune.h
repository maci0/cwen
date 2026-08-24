/* Auto-tuned knobs (GA / symbolic search). Safe defaults if missing. */
#ifndef CWEN_TUNE_H
#define CWEN_TUNE_H

/* One CCD on 9950X (16c). GA often prefers 32 for gemv micros; e2e DRAM loses. */
#ifndef CWEN_OMP_THREADS
#define CWEN_OMP_THREADS 16
#endif
/* Symbolic OMP gate: parallel when M > EXPR(M,K). Evolved by GA. */
#ifndef CWEN_OMP_THRESH_EXPR
#define CWEN_OMP_THRESH_EXPR(M, K) (32)
#endif
#ifndef CWEN_PREFETCH
#define CWEN_PREFETCH 16
#endif
#ifndef CWEN_Q4_UNROLL
#define CWEN_Q4_UNROLL 1
#endif
#ifndef CWEN_Q4_PF_BLOCKS
#define CWEN_Q4_PF_BLOCKS 12
#endif

#endif
