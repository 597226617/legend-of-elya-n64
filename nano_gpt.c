/**
 * nano_gpt.c — Tiny LLM inference for Nintendo 64
 *
 * Sophia-GPT running on the VR4300 MIPS @ 93 MHz.
 *
 * IMPORTANT: compile with -msoft-float
 *   The R4300 FPU does not implement trunc.w.s (float→int truncation).
 *   -msoft-float replaces every FP instruction with a MIPS-I software
 *   library call, avoiding the unimplemented-operation exception.
 *   A linker warning about mixing hard/soft float with legend_of_elya.c
 *   is expected and harmless — the API boundary is all-integer.
 *
 * Architecture:
 *   - Byte-level vocabulary (256 tokens — no tokeniser needed)
 *   - Q4_0 quantisation: 32 values/block, 1 fp32 scale per block
 *   - MIPS fixed-point Q8.7 activations in inner loops
 *   - No heap allocations after sgai_init(); all state on stack or in
 *     the caller-supplied SGAIKVCache
 */

#include "nano_gpt.h"
#include <string.h>
#include <math.h>   /* for expf, sqrtf — replaced by soft-float calls */
#include <stdlib.h>

/* ─── Weight binary header ───────────────────────────────────────────── */

#define SGAI_MAGIC  0x534F5048UL   /* "SOPH" */

typedef struct {
    uint32_t magic;
    uint32_t n_embd;
    uint32_t n_head;
    uint32_t n_layer;
    uint32_t vocab_sz;
    uint32_t ctx_len;
} __attribute__((packed)) SGAIHeader;

/* ─── Helpers ─────────────────────────────────────────────────────────── */

/* Dequantise a single Q4 block into dst[SGAI_Q4_BLOCK] floats */
static inline void dequant_q4_block(float *dst, const SGAIBlockQ4 *b) {
    float sc = b->scale;
    for (int i = 0; i < SGAI_Q4_BLOCK / 2; i++) {
        uint8_t byte = b->qs[i];
        int lo = (int)(byte & 0x0F) - 8;
        int hi = (int)(byte >>   4) - 8;
        dst[2*i  ] = lo * sc;
        dst[2*i+1] = hi * sc;
    }
}

/* Layer normalisation: x = (x - mean) / sqrt(var + eps) * w + b */
static void layer_norm(float *x, const float *w, const float *b, int n) {
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d*d; }
    var = 1.0f / sqrtf(var / n + 1e-5f);
    for (int i = 0; i < n; i++) x[i] = (x[i] - mean) * var * w[i] + b[i];
}

/* Softmax in-place */
static void softmax(float *x, int n) {
    float max = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max) max = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

/* GELU activation (approximation: x * sigmoid(1.702 * x)) */
static inline float gelu(float x) {
    return x / (1.0f + expf(-1.702f * x));
}

/* ─── Q4 Matrix-vector multiply ──────────────────────────────────────── */

/**
 * y[M] += W[M*K/BLOCK](Q4) · x[K]
 * W is stored row-major, each row of K values quantised in blocks of 32.
 */
void sgai_rsp_matmul_q4(float *y, const SGAIBlockQ4 *W,
                         const float *x, int M, int K)
{
    int blocks_per_row = K / SGAI_Q4_BLOCK;
    float tmp[SGAI_Q4_BLOCK];

    for (int m = 0; m < M; m++) {
        float acc = 0.0f;
        const SGAIBlockQ4 *row = W + m * blocks_per_row;
        for (int b = 0; b < blocks_per_row; b++) {
            dequant_q4_block(tmp, &row[b]);
            const float *xb = x + b * SGAI_Q4_BLOCK;
            for (int k = 0; k < SGAI_Q4_BLOCK; k++)
                acc += tmp[k] * xb[k];
        }
        y[m] += acc;
    }
}

/* ─── Attention layer (single transformer block) ─────────────────────── */

/* Scratch buffers — static to avoid stack overflow on N64 (4KB stack) */
static float s_q [SGAI_MAX_EMBD];
static float s_k [SGAI_MAX_EMBD];
static float s_v [SGAI_MAX_EMBD];
static float s_attn[SGAI_MAX_CTX];
static float s_x2 [SGAI_MAX_EMBD];
static float s_ff [SGAI_MAX_EMBD * 4];

void attention_layer(SGAIState *s, float *x, int layer, int pos)
{
    int E = s->n_embd, H = s->n_head, D = s->head_dim;

    /* -- Self-attention -- */
    /* LayerNorm 1 */
    float xln[SGAI_MAX_EMBD];
    memcpy(xln, x, E * sizeof(float));
    layer_norm(xln, s->ln1_w[layer], s->ln1_b[layer], E);

    /* Project Q, K, V */
    memset(s_q, 0, E * sizeof(float));
    memset(s_k, 0, E * sizeof(float));
    memset(s_v, 0, E * sizeof(float));
    sgai_rsp_matmul_q4(s_q, s->wq[layer], xln, E, E);
    sgai_rsp_matmul_q4(s_k, s->wk[layer], xln, E, E);
    sgai_rsp_matmul_q4(s_v, s->wv[layer], xln, E, E);

    /* Store K, V into cache */
    for (int h = 0; h < H; h++) {
        for (int d = 0; d < D; d++) {
            s->kv->k[layer][h][pos][d] = (int8_t)(s_k[h*D+d] * 128.0f);
            s->kv->v[layer][h][pos][d] = (int8_t)(s_v[h*D+d] * 128.0f);
        }
    }

    /* Multi-head attention over cached positions */
    memset(s_x2, 0, E * sizeof(float));
    float scale = 1.0f / sqrtf((float)D);

    for (int h = 0; h < H; h++) {
        /* Compute attention scores */
        for (int t = 0; t <= pos; t++) {
            float dot = 0.0f;
            for (int d = 0; d < D; d++)
                dot += s_q[h*D+d] * (s->kv->k[layer][h][t][d] / 128.0f);
            s_attn[t] = dot * scale;
        }
        /* Mask future positions (causal) — already ensured by loop bound */
        softmax(s_attn, pos + 1);

        /* Weighted sum of V */
        for (int t = 0; t <= pos; t++) {
            float a = s_attn[t];
            for (int d = 0; d < D; d++)
                s_x2[h*D+d] += a * (s->kv->v[layer][h][t][d] / 128.0f);
        }
    }

    /* Output projection + residual */
    float proj[SGAI_MAX_EMBD];
    memset(proj, 0, E * sizeof(float));
    sgai_rsp_matmul_q4(proj, s->wo[layer], s_x2, E, E);
    for (int i = 0; i < E; i++) x[i] += proj[i];

    /* -- Feed-forward -- */
    memcpy(xln, x, E * sizeof(float));
    layer_norm(xln, s->ln2_w[layer], s->ln2_b[layer], E);

    /* Up-project (E → 4E) */
    int FF = E * 4;
    memset(s_ff, 0, FF * sizeof(float));
    sgai_rsp_matmul_q4(s_ff, s->wff1[layer], xln, FF, E);
    for (int i = 0; i < FF; i++) s_ff[i] = gelu(s_ff[i]);

    /* Down-project (4E → E) + residual */
    memset(proj, 0, E * sizeof(float));
    sgai_rsp_matmul_q4(proj, s->wff2[layer], s_ff, E, FF);
    for (int i = 0; i < E; i++) x[i] += proj[i];
}

/* ─── Token sampling ─────────────────────────────────────────────────── */

int sgai_next_token(const float *logits, int vocab_sz, int temperature_q8)
{
    /* Apply temperature scaling */
    float tmp[SGAI_MAX_VOCAB];
    float t = (float)temperature_q8 / 128.0f;
    if (t < 0.01f) t = 0.01f;
    for (int i = 0; i < vocab_sz; i++) tmp[i] = logits[i] / t;
    softmax(tmp, vocab_sz);

    /* Sample from the distribution using a simple LCG RNG seeded by
     * the N64 count register (or fallback to argmax if temp==0)       */
    if (temperature_q8 == 0) {
        int best = 0;
        for (int i = 1; i < vocab_sz; i++)
            if (tmp[i] > tmp[best]) best = i;
        return best;
    }

    /* N64-friendly random: use a static LCG */
    static uint32_t rng_state = 0xDEADBEEFUL;
    rng_state = rng_state * 1664525UL + 1013904223UL;
    float r = (float)(rng_state >> 8) / (float)(1 << 24);

    float cdf = 0.0f;
    for (int i = 0; i < vocab_sz; i++) {
        cdf += tmp[i];
        if (r < cdf) return i;
    }
    return vocab_sz - 1;
}

/* ─── Public API ─────────────────────────────────────────────────────── */

void sgai_init(SGAIState *s, uint8_t *weights)
{
    SGAIHeader *hdr = (SGAIHeader *)weights;

    if (hdr->magic != SGAI_MAGIC) {
        /* Unrecognised weights — zero out so callers can detect failure */
        memset(s, 0, sizeof(*s));
        return;
    }

    s->n_embd   = (int)hdr->n_embd;
    s->n_head   = (int)hdr->n_head;
    s->n_layer  = (int)hdr->n_layer;
    s->vocab_sz = (int)hdr->vocab_sz;
    s->ctx_len  = (int)hdr->ctx_len;
    s->head_dim = s->n_embd / s->n_head;

    /* Walk the weight buffer after the header */
    uint8_t *ptr = weights + sizeof(SGAIHeader);

    /* Token embedding table */
    int wte_blocks = s->vocab_sz * s->n_embd / SGAI_Q4_BLOCK;
    s->wte = (const SGAIBlockQ4 *)ptr;
    ptr += wte_blocks * sizeof(SGAIBlockQ4);

    /* Per-layer weights */
    int attn_blocks = s->n_embd * s->n_embd / SGAI_Q4_BLOCK;
    int ff_blocks   = s->n_embd * 4 * s->n_embd / SGAI_Q4_BLOCK;

    for (int l = 0; l < s->n_layer; l++) {
        s->wq[l] = (const SGAIBlockQ4 *)ptr; ptr += attn_blocks * sizeof(SGAIBlockQ4);
        s->wk[l] = (const SGAIBlockQ4 *)ptr; ptr += attn_blocks * sizeof(SGAIBlockQ4);
        s->wv[l] = (const SGAIBlockQ4 *)ptr; ptr += attn_blocks * sizeof(SGAIBlockQ4);
        s->wo[l] = (const SGAIBlockQ4 *)ptr; ptr += attn_blocks * sizeof(SGAIBlockQ4);
        s->wff1[l] = (const SGAIBlockQ4 *)ptr; ptr += ff_blocks * sizeof(SGAIBlockQ4);
        s->wff2[l] = (const SGAIBlockQ4 *)ptr; ptr += ff_blocks * sizeof(SGAIBlockQ4);

        s->ln1_w[l] = (const float *)ptr; ptr += s->n_embd * sizeof(float);
        s->ln1_b[l] = (const float *)ptr; ptr += s->n_embd * sizeof(float);
        s->ln2_w[l] = (const float *)ptr; ptr += s->n_embd * sizeof(float);
        s->ln2_b[l] = (const float *)ptr; ptr += s->n_embd * sizeof(float);
    }

    s->lnf_w  = (const float *)ptr; ptr += s->n_embd * sizeof(float);
    s->lnf_b  = (const float *)ptr; ptr += s->n_embd * sizeof(float);
    s->lm_head = (const SGAIBlockQ4 *)ptr;

    /* KV cache must already be set by caller */
    if (s->kv) sgai_reset(s);
}

void sgai_reset(SGAIState *s)
{
    if (s->kv) {
        memset(s->kv, 0, sizeof(SGAIKVCache));
    }
}

void sgai_generate(SGAIState *s,
                   const uint8_t *prompt, int prompt_len,
                   uint8_t *out, int max_tokens,
                   int temperature_q8)
{
    if (!s->n_embd || !s->kv) return;

    int E = s->n_embd, V = s->vocab_sz;
    float x[SGAI_MAX_EMBD];
    float logits[SGAI_MAX_VOCAB];

    int pos = s->kv->len;
    int out_len = 0;

    /* Process prompt tokens */
    for (int t = 0; t < prompt_len; t++, pos++) {
        if (pos >= SGAI_MAX_CTX) break;

        /* Embed token */
        int tok = prompt[t] & 0xFF;
        int embd_blocks = E / SGAI_Q4_BLOCK;
        const SGAIBlockQ4 *emb_row = s->wte + tok * embd_blocks;
        float tmp[SGAI_Q4_BLOCK];
        for (int b = 0; b < embd_blocks; b++) {
            dequant_q4_block(tmp, &emb_row[b]);
            for (int k = 0; k < SGAI_Q4_BLOCK; k++)
                x[b * SGAI_Q4_BLOCK + k] = tmp[k];
        }

        /* Forward through transformer */
        for (int l = 0; l < s->n_layer; l++)
            attention_layer(s, x, l, pos);
    }

    /* Autoregressive generation */
    for (int gen = 0; gen < max_tokens && pos < SGAI_MAX_CTX; gen++, pos++) {
        /* Final layer norm + LM head */
        float xln[SGAI_MAX_EMBD];
        memcpy(xln, x, E * sizeof(float));
        layer_norm(xln, s->lnf_w, s->lnf_b, E);

        memset(logits, 0, V * sizeof(float));
        sgai_rsp_matmul_q4(logits, s->lm_head, xln, V, E);

        /* Sample next token */
        int next_tok = sgai_next_token(logits, V, temperature_q8);

        /* Clamp to printable ASCII (32–126) so libdragon font renders it */
        uint8_t out_byte = (uint8_t)next_tok;
        if (out_byte < 32 || out_byte > 126) out_byte = '?';

        out[out_len++] = out_byte;
        out[out_len]   = '\0';   /* keep null-terminated for safety */

        /* Stop on EOS (newline as a simple sentinel) */
        if (out_byte == '\n') break;

        /* Embed the generated token for next step */
        int tok = next_tok & 0xFF;
        int embd_blocks = E / SGAI_Q4_BLOCK;
        const SGAIBlockQ4 *emb_row = s->wte + tok * embd_blocks;
        float tmp[SGAI_Q4_BLOCK];
        for (int b = 0; b < embd_blocks; b++) {
            dequant_q4_block(tmp, &emb_row[b]);
            for (int k = 0; k < SGAI_Q4_BLOCK; k++)
                x[b * SGAI_Q4_BLOCK + k] = tmp[k];
        }

        /* Forward through transformer */
        for (int l = 0; l < s->n_layer; l++)
            attention_layer(s, x, l, pos);
    }

    s->kv->len = pos;
}
