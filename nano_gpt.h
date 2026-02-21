/**
 * nano_gpt.h — Tiny LLM inference for Nintendo 64
 *
 * Sophia-GPT: Q4 quantized character-level language model
 * trained on Sophia Elya's utterances.  Runs entirely on the
 * VR4300 MIPS CPU at 93 MHz — no DSP, no RCP co-processor.
 *
 * Compile with -msoft-float to avoid trunc.w.s / unimplemented
 * FPU instructions on the R4300 hardware FPU.
 *
 * Weight binary layout (sophia_weights.bin):
 *   [4]  magic  = 0x534F5048 ("SOPH")
 *   [4]  n_embd   — embedding dimension
 *   [4]  n_head   — number of attention heads
 *   [4]  n_layer  — number of transformer layers
 *   [4]  vocab_sz — vocabulary size (byte-level, typically 256)
 *   [4]  ctx_len  — maximum context window length
 *   [...]  packed Q4 weight blocks (32 values per block)
 *
 * API:
 *   sgai_init()     — load weights from a buffer in DMEM/ROM
 *   sgai_reset()    — wipe KV-cache, ready for a new conversation
 *   sgai_generate() — autoregressively produce up to max_tokens bytes
 */

#ifndef NANO_GPT_H
#define NANO_GPT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Compile-time limits ─────────────────────────────────────────────── */

#define SGAI_MAX_VOCAB   256   /* byte-level vocabulary                    */
#define SGAI_MAX_CTX     128   /* maximum sequence length (KV cache slots) */
#define SGAI_MAX_EMBD    64    /* maximum embedding dimension              */
#define SGAI_MAX_LAYERS  4     /* maximum transformer layers               */
#define SGAI_MAX_HEADS   4     /* maximum attention heads                  */

/* ─── Q4 quantisation block ──────────────────────────────────────────── */

#define SGAI_Q4_BLOCK    32    /* values per quantisation block            */

typedef struct {
    float   scale;             /* dequantisation scale factor              */
    uint8_t qs[SGAI_Q4_BLOCK / 2]; /* packed nibbles: lo=even, hi=odd     */
} SGAIBlockQ4;

/* ─── Key / Value cache ───────────────────────────────────────────────── */

typedef struct {
    /* Stored as Q8 fixed-point (scale = 1/128) to save RAM.
     * Layout: [layer][head][seq_pos][head_dim]                            */
    int8_t  k[SGAI_MAX_LAYERS][SGAI_MAX_HEADS][SGAI_MAX_CTX][SGAI_MAX_EMBD / SGAI_MAX_HEADS];
    int8_t  v[SGAI_MAX_LAYERS][SGAI_MAX_HEADS][SGAI_MAX_CTX][SGAI_MAX_EMBD / SGAI_MAX_HEADS];
    int     len;               /* number of filled positions               */
} SGAIKVCache;

/* ─── Model state ─────────────────────────────────────────────────────── */

typedef struct {
    /* ---- Architecture (read from weight header) ---- */
    int n_embd;
    int n_head;
    int n_layer;
    int vocab_sz;
    int ctx_len;
    int head_dim;              /* = n_embd / n_head                        */

    /* ---- Weight pointers into the weight buffer ---- */
    /* Token embedding table  [vocab_sz × n_embd] Q4                      */
    const SGAIBlockQ4 *wte;

    /* Per-layer weights; indexed by layer                                 */
    const SGAIBlockQ4 *wq   [SGAI_MAX_LAYERS]; /* query  [n_embd × n_embd] */
    const SGAIBlockQ4 *wk   [SGAI_MAX_LAYERS]; /* key    [n_embd × n_embd] */
    const SGAIBlockQ4 *wv   [SGAI_MAX_LAYERS]; /* value  [n_embd × n_embd] */
    const SGAIBlockQ4 *wo   [SGAI_MAX_LAYERS]; /* output [n_embd × n_embd] */
    const SGAIBlockQ4 *wff1 [SGAI_MAX_LAYERS]; /* FF up   [n_embd × 4*n_embd] */
    const SGAIBlockQ4 *wff2 [SGAI_MAX_LAYERS]; /* FF down [4*n_embd × n_embd] */

    /* Layer-norm scales (fp32, small, kept full-precision)               */
    const float *ln1_w[SGAI_MAX_LAYERS];
    const float *ln1_b[SGAI_MAX_LAYERS];
    const float *ln2_w[SGAI_MAX_LAYERS];
    const float *ln2_b[SGAI_MAX_LAYERS];

    /* Final layer-norm + LM head                                          */
    const float         *lnf_w;
    const float         *lnf_b;
    const SGAIBlockQ4   *lm_head; /* [vocab_sz × n_embd] Q4, tied to wte  */

    /* ---- Runtime state ---- */
    SGAIKVCache *kv;           /* pointer to caller-allocated KV cache    */
} SGAIState;

/* ─── Public API ─────────────────────────────────────────────────────── */

/**
 * sgai_init — parse weight buffer and fill SGAIState.
 *
 * @param s      uninitialized model state; kv must already point to a
 *               zero-initialized SGAIKVCache before calling.
 * @param weights pointer to the raw sophia_weights.bin data in RAM.
 *               Must remain valid for the lifetime of @s.
 */
void sgai_init(SGAIState *s, uint8_t *weights);

/**
 * sgai_reset — clear the KV cache to start a new conversation.
 */
void sgai_reset(SGAIState *s);

/**
 * sgai_generate — run autoregressive text generation.
 *
 * @param s           model state (must have been sgai_init'd)
 * @param prompt      input bytes (not NUL-terminated; byte-level tokens)
 * @param prompt_len  number of prompt bytes
 * @param out         output buffer — filled with generated bytes
 * @param max_tokens  maximum tokens to generate (≤ out buffer capacity)
 * @param temperature_q8  softmax temperature in Q8.0 fixed-point
 *                    (128 = 1.0, 64 = 0.5, 192 = 1.5)
 *                    Use 80–120 for coherent dialogue.
 */
void sgai_generate(SGAIState *s,
                   const uint8_t *prompt,  int prompt_len,
                   uint8_t       *out,     int max_tokens,
                   int            temperature_q8);

/* ─── Internal helpers (exposed for unit tests) ─────────────────────── */

/**
 * sgai_rsp_matmul_q4 — Q4 × F32 matrix–vector multiply.
 * Operates in-place: y[M] += W[M][K](Q4) · x[K]
 */
void sgai_rsp_matmul_q4(float *y, const SGAIBlockQ4 *W,
                         const float *x, int M, int K);

/** attention_layer — single transformer block forward pass. */
void attention_layer(SGAIState *s, float *x, int layer, int pos);

/** sgai_next_token — sample the next token from the logits buffer. */
int  sgai_next_token(const float *logits, int vocab_sz, int temperature_q8);

#ifdef __cplusplus
}
#endif

#endif /* NANO_GPT_H */
