/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cube_helper.h
 * \brief AIC-side datapath of the dav_3510 LSTM kernels: GM -> L1 -> L0A/L0B/BT -> MMAD -> L0C -> UB.
 *        Vector-side helpers are in vec_helper.h, split by who issues the instruction.
 *
 * No TPipe, no Matmul<>, no REGIST_MATMUL_OBJ: KFC and a hand-driven MatmulImpl cannot coexist -- a
 * kernel carrying both deadlocks (RC=124) -- and the fused L0C->UB drain these kernels rest on needs
 * the hand-driven form. Shapes are runtime uint32_t parameters so the same code serves the
 * dynamic-shape op; bisheng constant-folds when callers pass constants.
 *
 * Measured facts this header encodes:
 *   - one native layout per operand: L0A = NZ, L0B = ZN, L0C = NZ (16 combinations measured, one
 *     winner). So SplitA is a straight copy and SplitB must transpose; that asymmetry is not
 *     arbitrary and must not be cleaned up.
 *   - C0 is type dependent: 32 bytes fixed, so 32/sizeof(T) elements. The 16x16 fractal is fp16-only;
 *     fp32 is 16x8.
 *
 * The tail contract, also measured:
 *   M (batch)  arbitrary, including odd and 1, via two separate corrections -- DrainStripe encodes
 *              the hardware's CeilDiv(m,2) split, and MmadM floors mp.m at 2 because mp.m == 1
 *              silently returns garbage.
 *   K          arbitrary: mp.k is honoured exactly and pad lanes do not contribute.
 *   I, H       each a multiple of C0<T>(). Two hardware reasons: two writers sharing an NZ tile by
 *              column collide inside a column block if the boundary is off a block edge, and every
 *              strided UB copy in the epilogue expresses its row pitch in 32-byte blocks. op_host
 *              refuses a non-aligned I or H rather than constraining batch.
 */
#ifndef OPS_RNN_SINGLE_LAYER_LSTM_CUBE_HELPER_H
#define OPS_RNN_SINGLE_LAYER_LSTM_CUBE_HELPER_H

#include "kernel_operator.h"
#include "onchip_budget.h"

/* 命名空间用算子全名。它们原先叫 ch / vh / sh (cube / vector / sync helper)、fwd、proj --
 * 按范式取的短名, 读起来顺, 但摆在顶层是抢地盘: 同名的 fwd::Layout 在另一份 RNN 实现里也
 * 存在过, 两者一旦被同一次编译收进来就是 struct 重定义, 而报错只会说 "redefinition of
 * struct fwd::Layout", 不会告诉你是哪两个算子撞了。仓内的同类都用全名 (LstmGradRegbase、
 * ThnnFusedLstmCellNS), 这里照办。 */
namespace SingleLayerLstmCube {

/* BtElems() is in onchip_budget.h so that op_host's budget counts the same L1 bytes LoadBiasToBT
 * consumes; it is in scope through that include.
 *
 * NzLayout -- the fractal layout as (Shape, Stride). Element (m, k) maps to
 * (k / c0) * rowsAligned * c0 + m * c0 + (k % c0). `rowsAligned` is a separate field from `rows` on
 * purpose: it is the parent tile's aligned row count, which is what distinguishes a compact tile
 * from a column-block slice of a bigger one -- two things with the same shape and different outer
 * strides. Nd2NzParams cannot express that distinction, deriving the column stride from nValue and
 * never reading dstNzC0Stride. */
struct NzLayout {
    uint32_t rows;        // logical rows written by THIS access
    uint32_t cols;        // logical columns
    uint32_t c0;          // elements per fractal column-block = 32 / sizeof(T)
    uint32_t rowsAligned; // the PARENT tile's aligned rows -- the outer stride

    CH_BOTH NzLayout(uint32_t r, uint32_t c, uint32_t c0In, uint32_t parentRows)
        : rows(r), cols(c), c0(c0In), rowsAligned(CeilAlign(parentRows, CUBE_BLOCK))
    {}

    // element offset, in elements
    CH_BOTH uint32_t At(uint32_t m, uint32_t k) const { return (k / c0) * rowsAligned * c0 + m * c0 + (k % c0); }

    // start of column-block kb, in elements
    CH_BOTH uint32_t ColBlock(uint32_t kb) const { return kb * rowsAligned * c0; }

    CH_BOTH uint32_t NumColBlocks() const { return CeilDiv(cols, c0); }
};

/* ---------------------------------------------------------------------------------------------
 * GM -> L1
 * ------------------------------------------------------------------------------------------- */

/* ND [rows, cols] in GM -> NZ tile in L1. `parentRows` is normally just `rows`; pass a larger
 * value when this access fills part of a taller tile. */
template <typename T>
__aicore__ inline void CopyInNd2Nz(const AscendC::LocalTensor<T>& l1, const AscendC::GlobalTensor<T>& gm, uint32_t rows,
                                   uint32_t cols, uint32_t srcRowStride, uint32_t parentRows)
{
    AscendC::Nd2NzParams p;
    p.ndNum = 1;
    p.nValue = rows;
    p.dValue = cols;
    p.srcNdMatrixStride = 0;
    p.srcDValue = srcRowStride;
    p.dstNzC0Stride = CeilAlign(parentRows, CUBE_BLOCK);
    p.dstNzNStride = 1;
    p.dstNzMatrixStride = 0;
    AscendC::DataCopy(l1, gm, p);
}

/* Flat byte copy GM -> L1, no layout reinterpretation. Used for the bias row. */
template <typename T>
__aicore__ inline void CopyInFlat(const AscendC::LocalTensor<T>& l1, const AscendC::GlobalTensor<T>& gm, uint32_t elems)
{
    AscendC::DataCopyParams cp;
    cp.blockCount = 1;
    cp.blockLen = static_cast<uint16_t>(elems * sizeof(T) / AscendC::ONE_BLK_SIZE);
    cp.srcStride = 0;
    cp.dstStride = 0;
    AscendC::DataCopy(l1, gm, cp);
}

/* ---------------------------------------------------------------------------------------------
 * L1 -> L0
 * ------------------------------------------------------------------------------------------- */

/* L1 (NZ) -> L0A (NZ). A STRAIGHT COPY -- no transpose -- because both sides are NZ.
 * Do not add a transpose here to "match" SplitB. */
template <typename T>
__aicore__ inline void SplitA(const AscendC::LocalTensor<T>& l0a, const AscendC::LocalTensor<T>& l1, uint32_t m,
                              uint32_t k)
{
    AscendC::LoadData2DParamsV2 p;
    p.mStep = CeilDiv(m, CUBE_BLOCK);
    p.kStep = CeilDiv(k, C0<T>());
    p.srcStride = CeilDiv(m, CUBE_BLOCK);
    p.dstStride = CeilDiv(m, CUBE_BLOCK);
    AscendC::LoadData(l0a, l1, p);
}

/* L1 (NZ) -> L0B (ZN). Must transpose: L0B's native layout is ZN, not NZ.
 *
 * The two widths need different instructions, and the 4-byte one is not a generalisation. A fractal
 * is 512 bytes, so 16x16 at fp16 and 16x8 at fp32, but the hardware's transpose granule is 16x16
 * elements either way -- at fp32 one repeat consumes two source fractals, and the pair it needs is
 * the same k-block in the next n-block, `kFrac` fractals apart. LoadData2DParams cannot say that:
 * the V1 forwarding constructor hardcodes srcFracGap to 0, so it grabs the adjacent pair. Measured:
 * C[0][8..15] came back as B[16..23][0..7]. Do not unify the two branches -- the fp32 form run at
 * fp16 scores bad=512/1024, because it asserts two fractals per repeat.
 *
 * The two gaps are formulas, and both were first pinned as constants that happened to be right at
 * one shape: srcFracGap == kFrac-1 reads as 1 when kFrac == 2, dstFracGap == nRep-1 reads as 1 when
 * nRep == 2. A wrong gap writes one destination half outside the k-block's region, so the Mmad reads
 * L0B bytes this launch never wrote -- which is why wrong settings gave three different scores for
 * one command depending on what ran before. Verified bad=0 on silicon and camodel at
 * N = 16/32/48/64/128 and kFrac = 2/4/8.
 *
 * `parentK` is the source tile's full row count, separate because the K axis is tiled: every source
 * stride here counts fractals along K, and the n-blocks of an NZ tile sit CeilAlign(parentK, 16)/16
 * fractals apart whatever slice of K one call reads. Pass 0 when the load covers the whole K extent.
 * Deriving the stride from `k` for a sub-K load reads the right rows of the wrong columns. */
template <typename T>
__aicore__ inline void SplitB(const AscendC::LocalTensor<T>& l0b, const AscendC::LocalTensor<T>& l1, uint32_t k,
                              uint32_t n, uint32_t parentK = 0)
{
    const uint32_t kFrac = CeilDiv(k, CUBE_BLOCK);                               // k-blocks THIS load covers
    const uint32_t kFracSrc = CeilDiv((parentK == 0) ? k : parentK, CUBE_BLOCK); // ... of the SOURCE tile
    const uint32_t nRep = CeilDiv(n, CUBE_BLOCK);
    const uint32_t dstOffset = nRep * CUBE_BLOCK * CUBE_BLOCK;
    const uint32_t srcOffset = CUBE_BLOCK * C0<T>(); // one fractal, 512B for every T
    if constexpr (sizeof(T) == 4) {
        AscendC::LoadData2dTransposeParamsV2 p;
        p.startIndex = 0;
        p.repeatTimes = static_cast<uint8_t>(nRep);
        p.srcStride = 2 * kFracSrc; // a repeat eats two 512B units, so step by two n-blocks
        p.dstGap = 0;
        p.dstFracGap = nRep - 1;     // its two destination halves are nRep-1 units apart
        p.srcFracGap = kFracSrc - 1; // ... and its two SOURCE fractals kFracSrc-1, not adjacent
        p.addrMode = 0;
        for (uint32_t i = 0; i < kFrac; ++i) {
            AscendC::LoadDataWithTranspose(l0b[i * dstOffset], l1[i * srcOffset], p);
        }
    } else {
        AscendC::LoadData2DParams p;
        p.startIndex = 0;
        p.repeatTimes = static_cast<uint8_t>(nRep);
        p.srcStride = kFracSrc;
        p.dstGap = 0;
        p.ifTranspose = true;
        for (uint32_t i = 0; i < kFrac; ++i) {
            AscendC::LoadData(l0b[i * dstOffset], l1[i * srcOffset], p);
        }
    }
}

/* A third corner: an operand already transposed in GM. Only one of the two directions exists on this
 * part. L0A is NZ and L0B is ZN, each other's transpose, so an L1 tile that is the NZ image of a
 * [p, q] matrix looks like it should serve four roles. It serves three:
 *
 *   direct    (LoadData2DParamsV2)    -> L0A : A of [M=p, K=q]   SplitA         works
 *   transpose (LoadDataWithTranspose) -> L0B : B of [K=p, N=q]   SplitB         works
 *   direct                            -> L0B : B of [K=q, N=p]   SplitBFromNK   works
 *   transpose                         -> L0A : A of [M=q, K=p]   -- does not work
 *
 * There is no SplitAFromKM, and one was measured before it was believed: writing the same
 * LoadDataWithTranspose to an L0A destination compiles, runs, raises nothing and leaves L0A
 * unchanged, so the Mmad multiplies whatever the previous launch left there and the product is
 * independent of the operands. The first probe "proved" it worked because its four arms ran back to
 * back over one set of operands -- L0A survives between launches, so the arm whose loader is a no-op
 * inherited the tile from the arm before it. Regenerating the operands per arm turned two of the
 * four red.
 *
 * So an A operand stored [K, M] has to be re-laid-out in workspace. It cannot be done in GM -> L1
 * either: Nd2NzParams reads rows contiguously, and a column of the source is not a row of anything.
 * That is why lstm_grad materialises xh^T. */

/* L1 holds the NZ image of an [n, k] matrix -> L0B, as the B operand of A[m,k] x B[k,n].
 * NO transpose: the parameters are SplitA's, with the first extent named `n`. Picking SplitB here
 * instead compiles, runs, and returns wrong numbers. */
template <typename T>
__aicore__ inline void SplitBFromNK(const AscendC::LocalTensor<T>& l0b, const AscendC::LocalTensor<T>& l1, uint32_t n,
                                    uint32_t k)
{
    AscendC::LoadData2DParamsV2 p;
    p.mStep = CeilDiv(n, CUBE_BLOCK);
    p.kStep = CeilDiv(k, C0<T>());
    p.srcStride = CeilDiv(n, CUBE_BLOCK);
    p.dstStride = CeilDiv(n, CUBE_BLOCK);
    AscendC::LoadData(l0b, l1, p);
}

/* L1 -> BT (bias table). The dav_3510 replacement for a unified-core kernel's UB -> L0C bias seed:
 * the table broadcasts one bias row over all M rows in hardware, so the "replicate bias across
 * baseM rows in UB" block of an older kernel gets DELETED, not translated. BT is a 4KB buffer
 * beside L0C, so the bias occupies neither L0C nor a UB plane.
 * `copy_cbuf_to_bt` asserts blockLen is EVEN for 4-byte types. */
__aicore__ inline void LoadBiasToBT(const AscendC::LocalTensor<float>& bt, const AscendC::LocalTensor<float>& l1Bias,
                                    uint32_t n)
{
    AscendC::DataCopyParams cp;
    cp.blockCount = 1;
    cp.blockLen = CeilAlign(n * sizeof(float), BT_ALIGN) / AscendC::ONE_BLK_SIZE;
    cp.srcStride = 0;
    cp.dstStride = 0;
    AscendC::DataCopy(bt, l1Bias, cp);
}

/* MMAD. The K tail is free: mp.k is honoured exactly, and lanes of A's last column block and rows of
 * B's last row block beyond k do not contribute. Measured with K=100 (6 full fp16 blocks + 4 real
 * lanes): filling both pads with 64.0 leaves the result bit-identical to a zero-filled pad, against
 * an expected contamination of 12*64*64 = 49152 per element. So callers may leave pad lanes dirty.
 * That measurement only means anything because both pads were poisoned at once -- poisoning one side
 * while the other is zero makes every pad term 0*x and the verdict vacuous. */

/* mp.m == 1 returns garbage: the M axis has a floor of 2.
 *
 * Measured at a production shape K=96 N=256 against a CPU oracle: at m=1 the plain `mp.m = m` gives
 * 251 of 256 elements wrong, deviations of O(2) on outputs of O(0.5), while `mp.m = 2` over the same
 * single valid row is exact to 2.4e-07. Every other m measured -- 2, 3, 16, 64, 65 -- is correct
 * untouched, so this is a degenerate case at exactly 1, not a round-M-up rule. The extra output row
 * is computed from whatever L0A row 1 holds and lands in L0C row 1, which the drain never reads;
 * zeroing it is not required, pad-only and pad-plus-zero arms being bit-identical.
 *
 * Batch 1 is a common inference shape and the failure is silent. It hid from an earlier probe at
 * K=16 N=32: one fp16 column block was not enough to expose it. A tail probe has to run the real
 * shape. */
CH_BOTH uint32_t MmadM(uint32_t m) { return (m < 2) ? 2 : m; }

/* With a bias table. Uses the 4-operand overload, which DERIVES cmatrixSource from the bias
 * tensor's position (TPosition::C2 -> read the table) and takes the BT ADDRESS FROM THE TENSOR, so
 * the table is addressed per-call rather than implicitly at offset 0.
 * cmatrixInitVal must be false: MmadCal computes `cmatrixInitVal && !isBias`, and a true value
 * discards the seed. */
template <typename TC, typename TA, typename TB>
__aicore__ inline void MmadBias(const AscendC::LocalTensor<TC>& l0c, const AscendC::LocalTensor<TA>& l0a,
                                const AscendC::LocalTensor<TB>& l0b, const AscendC::LocalTensor<float>& bt, uint32_t m,
                                uint32_t n, uint32_t k)
{
    AscendC::MmadParams mp;
    mp.m = MmadM(m); // floor of 2 -- see the note above; m=1 is silently wrong
    /* CUBE_BLOCK (16 ELEMENTS), not C0_BYTES (32 BYTES). Aligning an ELEMENT count to 32 gets away
     * with it only while every N is already a multiple of 32; the first N that is not (H=16)
     * scored far outside tolerance, because Mmad wrote 32 columns into an L0C tile laid out for 16
     * and the drain then read the wrong addresses. mp.n must agree with the L0C footprint, which
     * aligns N to CUBE_BLOCK. */
    mp.n = CeilAlign(n, CUBE_BLOCK);
    mp.k = k;
    mp.cmatrixInitVal = false;
    AscendC::Mmad(l0c, l0a, l0b, bt, mp);
}

/* Without bias: L0C is initialised from the product alone. */
template <typename TC, typename TA, typename TB>
__aicore__ inline void MmadPlain(const AscendC::LocalTensor<TC>& l0c, const AscendC::LocalTensor<TA>& l0a,
                                 const AscendC::LocalTensor<TB>& l0b, uint32_t m, uint32_t n, uint32_t k)
{
    AscendC::MmadParams mp;
    mp.m = MmadM(m);
    mp.n = CeilAlign(n, CUBE_BLOCK);
    mp.k = k;
    mp.cmatrixSource = false;
    mp.cmatrixInitVal = true;
    AscendC::Mmad(l0c, l0a, l0b, mp);
}

/* K-accumulating: cmatrixInitVal=false WITH cmatrixSource=false means "add into whatever L0C
 * already holds". MmadPlain is the k-block-0 form (seed from the product alone) and MmadBias is the
 * k-block-0 form when a bias seeds it instead.
 * The three differ only in two boolean fields, and picking the wrong one is not a build error -- it
 * silently keeps only the last k block, or adds the bias once per block instead of once. */
template <typename TC, typename TA, typename TB>
__aicore__ inline void MmadAccum(const AscendC::LocalTensor<TC>& l0c, const AscendC::LocalTensor<TA>& l0a,
                                 const AscendC::LocalTensor<TB>& l0b, uint32_t m, uint32_t n, uint32_t k)
{
    AscendC::MmadParams mp;
    mp.m = MmadM(m);
    mp.n = CeilAlign(n, CUBE_BLOCK);
    mp.k = k;
    mp.cmatrixSource = false;
    mp.cmatrixInitVal = false;
    AscendC::Mmad(l0c, l0a, l0b, mp);
}

/* ---------------------------------------------------------------------------------------------
 * L0C -> UB
 * ------------------------------------------------------------------------------------------- */

/* isToUB=true tells the dispatcher the destination is a UB address; the predefined CFG_NZ /
 * CFG_ROW_MAJOR / CFG_COLUMN_MAJOR all carry isToUB=false and must not be used here. L0C is natively
 * NZ and ROW_MAJOR is a transform on the way out; ROW_MAJOR is the variant verified here.
 * splitM=true sets dualDstCtl=0b01, splitting the tile by M so each AIV receives its own half.
 *
 * Measured with a tagged L0C so every element names its logical row: each AIV gets CeilDiv(m, 2)
 * rows, AIV k starting at k*CeilDiv(m,2). At odd m AIV0 gets one more real row than AIV1, and AIV1's
 * last delivered row is L0C padding. DrainStripe encodes exactly this; the two must be read
 * together. */
constexpr AscendC::FixpipeConfig CFG_ROW_MAJOR_UB = {AscendC::CO2Layout::ROW_MAJOR, true};

/* `dstPitch` is the row pitch of the DESTINATION plane, in elements; 0 means "the same as n",
 * which is the compact case -- one drain fills the whole plane.
 *
 * IT IS A SEPARATE PARAMETER BECAUSE THE N AXIS IS TILED. One gate's [m, H] result is produced in
 * column chunks, and chunk c has to land at column offset c*nChunk of a plane that is H wide. With
 * the pitch derived from `n` the second chunk would overwrite the first from column 0 -- every row
 * right, every column wrong, and nothing reports it. */
__aicore__ inline void DrainToUB(const AscendC::LocalTensor<float>& ub, const AscendC::LocalTensor<float>& l0c,
                                 uint32_t m, uint32_t n, bool splitM, uint32_t dstPitch = 0)
{
    AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> fp;
    fp.nSize = n;
    fp.srcStride = CeilAlign(m, CUBE_BLOCK);
    fp.dstStride = (dstPitch == 0) ? n : dstPitch;
    if (splitM) {
        fp.mSize = CeilAlign(m, 2);
        fp.dualDstCtl = 0b01;
    } else {
        fp.mSize = m;
        fp.dualDstCtl = 0b00;
        fp.subBlockId = false;
    }
    AscendC::Fixpipe<float, float, CFG_ROW_MAJOR_UB>(ub, l0c, fp);
}

/* RowStripe -- the rows of a shared tile that one AIV exclusively owns.
 *
 * L1 is a single buffer shared by the cluster and both AIVs can write it, with no hardware
 * arbitration: two AIVs writing the same address is last-writer-wins, silently. Safety comes only
 * from the addresses being disjoint by construction, and the invariant is sharper than disjoint --
 * the write-back partition must be the partition the drain used. DrainToUB(splitM=true) hands AIV k
 * rows [k*m/2, (k+1)*m/2), and the feedback must write back exactly those; passing one RowStripe to
 * both the epilogue and FeedbackToL1 makes that a shared value rather than two places that must
 * agree. A mis-computed offset once put AIV0's column blocks 1..3 on AIV1's 0..2: 1008 duplicate
 * tags, 1536 of AIV0's rows lost, no error anywhere.
 *
 * Same cause: when two writers share an NZ tile by column (the cube writing x into blocks [0, I/c0)
 * while the AIVs write h into [I/c0, K/c0)), I must be a multiple of c0 or the two halves collide
 * inside a column block. */
struct RowStripe {
    uint32_t base;  // first row of the shared tile that this core owns
    uint32_t count; // how many rows

    CH_BOTH uint32_t End() const { return base + count; }
};

/* Rows the drain hands to ONE AIV -- the ALLOCATION bound for that core's UB planes. Distinct from
 * RowStripe::count, which is how many of them carry real data: at odd m under splitM the two
 * differ by one for AIV1, and sizing a buffer by `count` would then under-allocate AIV0. */
CH_BOTH uint32_t DrainRowsMax(uint32_t m, bool splitM) { return splitM ? CeilDiv(m, 2) : m; }

/* The stripe DrainToUB hands to this AIV. `splitM` must match the value passed to DrainToUB.
 *
 * The odd-m split is CeilDiv, not m/2. Measured with a tagged L0C, `count = m/2; base = sub*count`
 * disagrees on 7 of 12 measured (m, aiv) pairs and the failures are silent: at m=65 AIV1 holds
 * logical rows 33..64 but claims base 32, so every row is off by one and row 32 is processed by
 * nobody; at m=63 one row of the batch is dropped; at m=1 both subcores get 0 and the kernel
 * computes nothing. Even m is correct throughout, which is why a corpus of divisible shapes never
 * sees it.
 *
 * Under !splitM the drain has a single destination, so subcore 0 owns every row and subcore 1 owns
 * none. Callers must test stripe.count, and a kernel using the vector->cube barrier must still let
 * the count==0 core participate in every round -- that flag needs both subcores to set or the cube
 * hangs. The same is true of AIV1 at m == 1. */
CH_BOTH RowStripe DrainStripe(uint32_t m, bool splitM, uint32_t subBlockIdx)
{
    RowStripe s;
    if (splitM) {
        const uint32_t half = CeilDiv(m, 2);
        s.base = subBlockIdx * half;
        // The tail AIV may own fewer rows than it was handed, or none at all when m == 1.
        s.count = (s.base >= m) ? 0 : ((m - s.base < half) ? (m - s.base) : half);
    } else {
        s.base = 0;
        s.count = (subBlockIdx == 0) ? m : 0;
    }
    return s;
}

} // namespace SingleLayerLstmCube

#endif // OPS_RNN_SINGLE_LAYER_LSTM_CUBE_HELPER_H
