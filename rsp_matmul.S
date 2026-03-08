## RSP Vector Matmul Microcode for Sophia LLM
##
## Performs int16 dot product using RSP SIMD (8-wide vmulf/vmacf).
## CPU driver converts Q8 weights to int16 and handles scales separately.
##
## Protocol (DMEM layout):
##   0x000: Command word [0=idle, 1=matmul, 0xFF=halt]
##   0x004: M (output rows, max 8)
##   0x008: N (input dim, max 128)
##   0x00C: unused
##   0x010: Result ready flag (RSP sets to 1 when done)
##   0x020: Input vector (N int16 values)
##   0x200: Weight block (M rows x N int16 values, sign-extended from int8)
##   0x800: Output results (M int32 accumulator results as int16 >> shift)

#include "rsp.inc"

.set noreorder
.set at

## ── Data Segment ─────────────────────────────────────────────────

.data
.align 3

cmd_word:      .word 0          ## 0x000
out_dim:       .word 0          ## 0x004
in_dim:        .word 0          ## 0x008
unused1:       .word 0          ## 0x00C
result_ready:  .word 0          ## 0x010

.align 4
## 0x020: Input vector (128 int16 = 256 bytes max)
input_vec:     .space 256

## padding to 0x200
.align 4
.space 0x200 - 0x120
## 0x200: Weight rows (8 x 128 int16 = 2048 bytes max)
weight_block:  .space 2048

.align 4
## 0x800: Output (8 x int16 = 16 bytes)
output_block:  .space 16


## ── Text Segment ─────────────────────────────────────────────────

.text 1

.globl _start
_start:
    ## Initialize zero vector
    vxor $v00, $v00, $v00

poll_cmd:
    lw t0, 0(zero)              ## cmd_word
    beqz t0, poll_cmd
    nop

    ## Halt?
    li t1, 0xFF
    beq t0, t1, halt
    nop

    ## Matmul?
    li t1, 1
    bne t0, t1, clear_cmd
    nop

    ## ── Load parameters ──
    lw s0, 4(zero)              ## s0 = M (output rows, max 8)
    lw s1, 8(zero)              ## s1 = N (input dim)

    ## Clamp
    slti t0, s0, 9
    bnez t0, m_ok
    nop
    li s0, 8
m_ok:
    slti t0, s1, 129
    bnez t0, n_ok
    nop
    li s1, 128
n_ok:

    ## Compute N/8 (inner loop count)
    srl s2, s1, 3               ## s2 = N/8

    ## Row counter
    li s3, 0

row_loop:
    beq s3, s0, matmul_done
    nop

    ## Accumulator = 0
    vxor $v30, $v30, $v30

    ## Input base = 0x020
    li t0, 0x020
    ## Weight row base = 0x200 + row * N * 2
    ## RSP has no mult, use repeated addition
    li t1, 0x200                ## weight base
    li t2, 0                    ## counter for row offset
    beqz s3, skip_row_offset
    nop
    move t3, s3                 ## rows to skip
add_row:
    add t1, t1, s1              ## add N (in bytes, but N is count)
    add t1, t1, s1              ## add N again (N*2 bytes per row)
    addi t3, t3, -1
    bnez t3, add_row
    nop
skip_row_offset:

    ## Inner loop: process N elements, 8 at a time
    move t3, s2                 ## t3 = N/8 iterations
    move t4, t0                 ## t4 = input pointer
    move t5, t1                 ## t5 = weight pointer

inner_loop:
    beqz t3, inner_done
    nop

    ## Load 8 input values (int16) from input_vec
    lqv $v01,0, t4              ## v01 = input[j:j+8]

    ## Load 8 weight values (int16) from weight row
    lqv $v02,0, t5              ## v02 = weight[row][j:j+8]

    ## Multiply-accumulate: acc += input * weight (fractional multiply)
    vmulf $v03, $v01, $v02      ## v03 = v01 * v02
    vadd $v30, $v30, $v03       ## accumulate

    addi t4, t4, 16             ## next 8 int16 inputs (16 bytes)
    addi t5, t5, 16             ## next 8 int16 weights (16 bytes)
    addi t3, t3, -1
    j inner_loop
    nop

inner_done:
    ## Horizontal reduce v30: sum all 8 lanes → v30.e0
    ## Pairwise add using element broadcast
    vaddc $v28, $v30, $v30.e1
    vaddc $v28, $v28, $v28.e2
    vaddc $v28, $v28, $v28.e4

    ## Store result: output[row] = v28.e0
    li t6, 0x800
    sll t7, s3, 1               ## row * 2
    add t6, t6, t7
    ssv $v28.e0, 0,t6           ## Store int16 result

    addi s3, s3, 1
    j row_loop
    nop

matmul_done:
    li t0, 1
    sw t0, 0x010(zero)          ## result_ready = 1

clear_cmd:
    sw zero, 0(zero)            ## cmd = idle
    j poll_cmd
    nop

halt:
    break
    nop
