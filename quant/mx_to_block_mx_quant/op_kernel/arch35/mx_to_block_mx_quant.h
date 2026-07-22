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
 * \file mx_to_block_mx_quant.h
 * \brief MxToBlockMxQuant kernel class.
 * \tparam T  Input dtype: fp4x2_e2m1_t, fp4x2_e1m2_t
 * \tparam U  Output dtype: fp8_e5m2_t, fp8_e4m3fn_t
 * \tparam rowMode  0=aligned (M%64==0), 1=not aligned
 */

#ifndef MX_TO_BLOCK_MX_QUANT_H
#define MX_TO_BLOCK_MX_QUANT_H
#define FLOAT_OVERFLOW_MODE_CTRL 60

#include "mx_to_block_mx_quant_common.h"

namespace MxToBlockMxQuantNs {
using namespace AscendC;

template <typename T, typename U, uint64_t rowMode>
class MxToBlockMxQuant {
public:
    __aicore__ inline MxToBlockMxQuant() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR mxScale, GM_ADDR y, GM_ADDR scale1, GM_ADDR scale2,
                                const MxToBlockMxQuantTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const MxToBlockMxQuantTilingData* tilingData);

    __aicore__ inline void CopyIn(int64_t batch, int64_t rowBlock, int64_t colBlock, int64_t curRowNum,
                                  int64_t curColNum);
    __aicore__ inline void CopyOut(int64_t batch, int64_t rowBlock, int64_t colBlock, int64_t curRowNum,
                                   int64_t curColNum);

    // ===== rowMode=0 (ALIGNED) 计算路径：单层 ComputeAligned，行恒为 64 =====
    __aicore__ inline void ComputeAligned(int64_t curRowNum, int64_t curColNum);

    // ===== rowMode=1 (NOT_ALIGNED) 计算路径：尾行任意行数兜底 =====
    __aicore__ inline void SubCompute(uint16_t loop, uint16_t scaleLoopStep, uint16_t scale1OutLoopNum,
                                      __ubuf__ uint8_t* xLocalAddr, __ubuf__ uint8_t* mxScaleLocalAddr,
                                      __ubuf__ uint8_t* yLocalAddr, __ubuf__ uint8_t* scale1LocalAddr,
                                      __ubuf__ uint32_t* scaleTmpBufAddr, __ubuf__ uint32_t* tmpBufAddr,
                                      __ubuf__ uint8_t* scale2InterleaveBufAddr, __ubuf__ uint32_t* scaleTmpBackBufAddr,
                                      Reg::RegTensor<uint8_t>& vdMaxOffset, Reg::RegTensor<int16_t>& zeroRegTensor);
    template <const int64_t padMode>
    __aicore__ inline void Compute(uint16_t loop0, uint16_t loop1, __ubuf__ uint8_t* xLocalAddr,
                                   __ubuf__ uint8_t* mxScaleLocalAddr, __ubuf__ uint8_t* yLocalAddr,
                                   __ubuf__ uint8_t* scale1LocalAddr, __ubuf__ uint8_t* scale2LocalAddr,
                                   __ubuf__ uint32_t* scaleTmpBufAddr, __ubuf__ uint32_t* tmpBufAddr,
                                   __ubuf__ uint8_t* scale2InterleaveBufAddr);
    __aicore__ inline void ComputeNotAligned(int64_t ubFactorRowNum, int64_t ubFactorColNum);

private:
    TPipe pipe_;
    // Input queues
    TQue<QuePosition::VECIN, DB_BUFFER> inQueueX_;
    TQue<QuePosition::VECIN, DB_BUFFER> inQueueMxScale_;
    // Output queues
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueueY_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueueScale1_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueueScale2_;
    // Temporary compute buffer
    TBuf<QuePosition::VECCALC> scaleTmpBuffer_;
    TBuf<QuePosition::VECCALC> tmpBuffer_;
    TBuf<QuePosition::VECCALC> scale2InterleaveBuffer_;

    // 绑定整个 GM 基址，偏移在 CopyIn/CopyOut 内按 (batch, rowBlock, colBlock) 动态计算。
    GlobalTensor<uint8_t> xGm_;
    GlobalTensor<uint8_t> mxScaleGm_;
    GlobalTensor<uint8_t> yGm_;
    GlobalTensor<uint8_t> scale1Gm_;
    GlobalTensor<uint8_t> scale2Gm_;

    int64_t batchNum_{0};    // batch 数 B
    int64_t rowNum_{0};      // 单 batch 行数 M
    int64_t colNum_{0};      // 列数 K
    int64_t colScaleNum_{0}; // mxscale 列方向合并最后两维的数量
    int64_t usedCoreNum_{0};
    int64_t dstType_{0};

    int64_t rowBlockNumPerBatch_{0}; // 单 batch 行方向基本块数 CeilDiv(M, 64)
    int64_t colBlockNumPerBatch_{0}; // 单 batch 列方向基本块数 CeilDiv(K, 512)
    int64_t rowTailLenPerBatch_{0};  // 单 batch 行尾块行数
    int64_t colTailLenPerBatch_{0};  // 单 batch 列尾块列数
    int64_t totalBlockNum_{0};       // 总块数
    int64_t headCoreBlockNum_{0};    // 头核处理块数
    int64_t tailCoreBlockNum_{0};    // 尾核处理块数
    int64_t headCoreNum_{0};         // 头核数量
    int64_t tailCoreNum_{0};         // 尾核数量

    // const
    uint8_t maxOffset{6}; // T fp4_e2m1 U fp8_e4m3_fn
};

// ============================================================================
// Init
// ============================================================================
template <typename T, typename U, uint64_t rowMode>
__aicore__ inline void MxToBlockMxQuant<T, U, rowMode>::Init(GM_ADDR x, GM_ADDR mxScale, GM_ADDR y, GM_ADDR scale1,
                                                             GM_ADDR scale2,
                                                             const MxToBlockMxQuantTilingData* tilingData)
{
#if (__NPU_ARCH__ == 3510)
    SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    ParseTilingData(tilingData);

    if constexpr (IsSame<T, fp4x2_e1m2_t>::value && IsSame<U, fp8_e4m3fn_t>::value) {
        maxOffset = 8;
    } else if constexpr (IsSame<T, fp4x2_e1m2_t>::value && IsSame<U, fp8_e5m2_t>::value) {
        maxOffset = 15;
    } else if constexpr (IsSame<T, fp4x2_e2m1_t>::value && IsSame<U, fp8_e5m2_t>::value) {
        maxOffset = 13;
    }

    int64_t xBufSize = DIGIT_ONE * SPLIT_M * SPLIT_N / DIGIT_TWO; // FP4 打包 0.5 byte
    int64_t mxScaleBufSize = DIGIT_ONE * SPLIT_M * MXSCALE_NUM_ALIGN32;
    int64_t yBufSize = DIGIT_ONE * SPLIT_M * SPLIT_N * sizeof(uint8_t);
    int64_t scale1BufSize = DIGIT_ONE * SPLIT_M * MXSCALE_NUM_ALIGN32;
    int64_t scale2BufSize = DIGIT_ONE * (SPLIT_M / BLOCK_SIZE) * SPLIT_N;
    int64_t scaleTmpBufSize = DIGIT_ONE * SPLIT_M * BLOCK_SIZE * DIGIT_TWO * sizeof(uint8_t);
    int64_t tmpBufSize = DIGIT_ONE * SCALE_TMP_SIZE * sizeof(uint8_t);
    int64_t scale2InterleaveBufSize = DIGIT_ONE * (SPLIT_M / BLOCK_SIZE) * SPLIT_N * sizeof(uint8_t);

    pipe_.InitBuffer(inQueueX_, DB_BUFFER, xBufSize);
    pipe_.InitBuffer(inQueueMxScale_, DB_BUFFER, mxScaleBufSize);
    pipe_.InitBuffer(outQueueY_, DB_BUFFER, yBufSize);
    pipe_.InitBuffer(outQueueScale1_, DB_BUFFER, scale1BufSize);
    pipe_.InitBuffer(outQueueScale2_, DB_BUFFER, scale2BufSize);
    pipe_.InitBuffer(scaleTmpBuffer_, scaleTmpBufSize);
    pipe_.InitBuffer(tmpBuffer_, tmpBufSize);
    pipe_.InitBuffer(scale2InterleaveBuffer_, scale2InterleaveBufSize);

    // 绑定整个 GM 基址，偏移在 CopyIn/CopyOut 内动态计算。
    xGm_.SetGlobalBuffer((__gm__ uint8_t*)x);
    mxScaleGm_.SetGlobalBuffer((__gm__ uint8_t*)mxScale);
    yGm_.SetGlobalBuffer((__gm__ uint8_t*)y);
    scale1Gm_.SetGlobalBuffer((__gm__ uint8_t*)scale1);
    scale2Gm_.SetGlobalBuffer((__gm__ uint8_t*)scale2);
}

// ============================================================================
// ParseTilingData
// ============================================================================
template <typename T, typename U, uint64_t rowMode>
__aicore__ inline void MxToBlockMxQuant<T, U, rowMode>::ParseTilingData(const MxToBlockMxQuantTilingData* tilingData)
{
    batchNum_ = tilingData->batchNum;
    rowNum_ = tilingData->rowNum;
    colNum_ = tilingData->colNum;
    colScaleNum_ = tilingData->colScaleNum;
    usedCoreNum_ = tilingData->usedCoreNum;
    dstType_ = tilingData->dstType;
    rowBlockNumPerBatch_ = tilingData->rowBlockNumPerBatch;
    colBlockNumPerBatch_ = tilingData->colBlockNumPerBatch;
    rowTailLenPerBatch_ = tilingData->rowTailLenPerBatch;
    colTailLenPerBatch_ = tilingData->colTailLenPerBatch;
    totalBlockNum_ = tilingData->totalBlockNum;
    headCoreBlockNum_ = tilingData->headCoreBlockNum;
    tailCoreBlockNum_ = tilingData->tailCoreBlockNum;
    headCoreNum_ = tilingData->headCoreNum;
    tailCoreNum_ = tilingData->tailCoreNum;
}

// ============================================================================
// Process — former/tail 分核 + (batch, rowBlock, colBlock) 块遍历。
//           Compute 路径按 rowMode 编译期路由，保证 rowMode=0 二进制不含兜底代码。
// ============================================================================
template <typename T, typename U, uint64_t rowMode>
__aicore__ inline void MxToBlockMxQuant<T, U, rowMode>::Process()
{
    int64_t coreId = GetBlockIdx();
    if (coreId >= usedCoreNum_) {
        return;
    }

    // 前 headCoreNum_ 个为头核（各 headCoreBlockNum_ 块），后 tailCoreNum_ 个为尾核（各 tailCoreBlockNum_ 块）。
    int64_t myBlockNum;
    int64_t myStartBlock;
    if (coreId < headCoreNum_) {
        myBlockNum = headCoreBlockNum_;
        myStartBlock = coreId * headCoreBlockNum_;
    } else {
        myBlockNum = tailCoreBlockNum_;
        myStartBlock = headCoreNum_ * headCoreBlockNum_ + (coreId - headCoreNum_) * tailCoreBlockNum_;
    }

    int64_t blocksPerBatch = rowBlockNumPerBatch_ * colBlockNumPerBatch_;
    for (int64_t b = 0; b < myBlockNum; b++) {
        int64_t gid = myStartBlock + b;
        int64_t batch = gid / blocksPerBatch;
        int64_t inBatch = gid % blocksPerBatch;
        int64_t rowBlock = inBatch % rowBlockNumPerBatch_;
        int64_t colBlock = inBatch / rowBlockNumPerBatch_;

        int64_t curRowNum = (rowBlock == rowBlockNumPerBatch_ - DIGIT_ONE) ? rowTailLenPerBatch_ : SPLIT_M;
        int64_t curColNum = (colBlock == colBlockNumPerBatch_ - DIGIT_ONE) ? colTailLenPerBatch_ : SPLIT_N;

        CopyIn(batch, rowBlock, colBlock, curRowNum, curColNum);
        if constexpr (rowMode == TPL_ROW_ALIGNED) {
            ComputeAligned(curRowNum, curColNum);
        } else {
            ComputeNotAligned(curRowNum, curColNum);
        }
        CopyOut(batch, rowBlock, colBlock, curRowNum, curColNum);
    }
}

// ============================================================================
// CopyIn — 按全局 (batch, rowBlock, colBlock) 偏移搬入，列方向 pad 到 512。
// ============================================================================
template <typename T, typename U, uint64_t rowMode>
__aicore__ inline void MxToBlockMxQuant<T, U, rowMode>::CopyIn(int64_t batch, int64_t rowBlock, int64_t colBlock,
                                                               int64_t curRowNum, int64_t curColNum)
{
    int64_t rowStart = batch * rowNum_ + rowBlock * SPLIT_M; // 全局行起点
    int64_t colStart = colBlock * SPLIT_N;                   // 全局列起点

    // ----- Copy x（FP4 打包 0.5 byte），列方向 pad 到 512 -----
    LocalTensor<uint8_t> xLocal = inQueueX_.AllocTensor<uint8_t>();
    int64_t xOffset = (rowStart * colNum_ + colStart) / DIGIT_TWO;
    int64_t anotherStride = (SPLIT_N - curColNum) / DIGIT_TWO / UBBlockSize_;
    int64_t xPad = curColNum % BLOCK_SIZE == 0 ? 0 : (BLOCK_SIZE - curColNum % BLOCK_SIZE) / DIGIT_TWO;

    DataCopyExtParams copyInParamX = {static_cast<uint16_t>(curRowNum),
                                      static_cast<uint32_t>(curColNum / DIGIT_TWO * sizeof(uint8_t)),
                                      static_cast<uint32_t>((colNum_ - curColNum) / DIGIT_TWO * sizeof(uint8_t)),
                                      static_cast<uint32_t>(anotherStride), 0};
    DataCopyPadExtParams<uint8_t> xPadParams{true, 0, static_cast<uint8_t>(xPad), 0};
    DataCopyPad(xLocal, xGm_[xOffset], copyInParamX, xPadParams);
    inQueueX_.EnQue(xLocal);

    // ----- Copy mxscale（FP8_E8M0）-----
    LocalTensor<uint8_t> scaleLocal = inQueueMxScale_.AllocTensor<uint8_t>();
    int64_t ubFactorColBlockNum = ops::CeilDiv(curColNum, BLOCK_SIZE);
    int64_t scaleNum = ops::CeilDiv(ubFactorColBlockNum, DIGIT_TWO) * DIGIT_TWO;
    int64_t mxScaleOffset = rowStart * colScaleNum_ + colBlock * DIGIT_SIXTEEN; // 16 = 512/32
    int64_t scalePad = UBBlockSize_ - scaleNum;

    DataCopyExtParams copyInParamScale = {static_cast<uint16_t>(curRowNum), static_cast<uint32_t>(scaleNum),
                                          static_cast<uint32_t>((colScaleNum_ - scaleNum)), 0, 0};
    DataCopyPadExtParams<uint8_t> scalePadParams{true, 0, static_cast<uint8_t>(scalePad), 0};
    DataCopyPad<uint8_t>(scaleLocal, mxScaleGm_[mxScaleOffset], copyInParamScale, scalePadParams);
    inQueueMxScale_.EnQue(scaleLocal);
}

// ============================================================================
// Compute (rowMode=0) — 对一个 64×512 块完成反量化到 FP8 + block 级 scale 计算。
//                       M 对齐 64，恒处理完整行；列方向不足 512 由 CopyIn pad 补齐。
// ============================================================================
template <typename T, typename U, uint64_t rowMode>
__aicore__ inline void MxToBlockMxQuant<T, U, rowMode>::ComputeAligned(int64_t curRowNum, int64_t curColNum)
{
    LocalTensor<uint8_t> xLocal = inQueueX_.DeQue<uint8_t>();
    LocalTensor<uint8_t> scaleLocal = inQueueMxScale_.DeQue<uint8_t>();
    LocalTensor<uint8_t> yLocal = outQueueY_.AllocTensor<uint8_t>();
    LocalTensor<uint8_t> scale1Local = outQueueScale1_.AllocTensor<uint8_t>();
    LocalTensor<uint8_t> scale2Local = outQueueScale2_.AllocTensor<uint8_t>();
    LocalTensor<uint8_t> scale2InterLeaveBuf = scale2InterleaveBuffer_.Get<uint8_t>();

    auto xLocalAddr = reinterpret_cast<__ubuf__ uint8_t*>(xLocal.GetPhyAddr());
    auto mxScaleLocalAddr = reinterpret_cast<__ubuf__ uint8_t*>(scaleLocal.GetPhyAddr());
    auto yLocalAddr = reinterpret_cast<__ubuf__ uint8_t*>(yLocal.GetPhyAddr());
    auto scale1LocalAddr = reinterpret_cast<__ubuf__ uint8_t*>(scale1Local.GetPhyAddr());
    auto scale2LocalAddr = reinterpret_cast<__ubuf__ uint8_t*>(scale2Local.GetPhyAddr());

    auto scaleTmpBufAddr = reinterpret_cast<__ubuf__ uint32_t*>(scaleTmpBuffer_.Get<uint32_t>().GetPhyAddr());
    auto tmpBufAddr = reinterpret_cast<__ubuf__ uint32_t*>(tmpBuffer_.Get<uint32_t>().GetPhyAddr());
    auto scale2InterleaveBufAddr = reinterpret_cast<__ubuf__ uint8_t*>(scale2InterLeaveBuf.GetPhyAddr());

    auto scaleTmpBackBufAddr = scaleTmpBufAddr;
    auto scale2InterleaveBackBufAddr = scale2InterleaveBufAddr;

    uint16_t loopNum = ops::CeilDiv(curRowNum, BLOCK_SIZE); // 2
    uint16_t loopNumBlock = static_cast<uint16_t>(BLOCK_SIZE);
    // scaleLoopStep / scale1OutLoopNum / DIGIT_EIGHT 复用 common.h 命名空间常量（值 32 / 4 / 8）。

    __VEC_SCOPE__
    {
        Reg::MaskReg scaleMask = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg scaleStoreMask = Reg::CreateMask<uint32_t, Reg::MaskPattern::VL16>();
        Reg::MaskReg mulMask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();

        Reg::RegTensor<uint8_t> maxScale;
        Reg::RegTensor<uint8_t> vdMaxOffset;
        Reg::RegTensor<int16_t> zeroRegTensor;
        Reg::RegTensor<uint8_t> preBroadCastScale, preBroadCastScale2x, preBroadCastScale4x;
        Reg::RegTensor<uint8_t> vdScale, vdScale2x, vdScale4x, vdScaleTmp;
        Reg::RegTensor<uint8_t> vdScale2One, vdScale2Two;
        Reg::RegTensor<uint8_t> vdScaleOne, vdScaleTwo;
        Reg::RegTensor<bfloat16_t> vdScaleOneBf16, vdScaleTwoBf16;
        Reg::RegTensor<bfloat16_t> vdScaleBf16_0_0, vdScaleBf16_0_1, vdScaleBf16_0_2, vdScaleBf16_0_3, vdScaleBf16_1_0,
            vdScaleBf16_1_1, vdScaleBf16_1_2, vdScaleBf16_1_3;
        Reg::RegTensor<bfloat16_t> vdFp4U8_0, vdFp4U8_1, vdFp4U8_2, vdFp4U8_3;
        Reg::MaskReg invalidMask;
        Reg::MaskReg zeroMask_0;
        Reg::MaskReg zeroMask_1;

        Reg::Duplicate(vdMaxOffset, maxOffset);
        Reg::Duplicate(zeroRegTensor, 0);

        for (uint16_t i = 0; i < loopNum; i++) {
            Reg::Duplicate(maxScale, 0);
            for (uint16_t j = 0; j < loopNumBlock; j++) {
                Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_BLK>(
                    vdScale, mxScaleLocalAddr, scaleLoopStep);
                Reg::Max(maxScale, maxScale, vdScale, scaleMask);
                Reg::Interleave(vdScale2x, vdScaleTmp, vdScale, vdScale);
                Reg::Interleave(vdScale4x, vdScaleTmp, vdScale2x, vdScale2x);
                Reg::StoreAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                    scaleTmpBufAddr, (Reg::RegTensor<uint32_t>&)vdScale4x, DIGIT_SIXTEEN, scaleStoreMask);
            }

            Reg::Compare<uint8_t, CMPMODE::LE>(invalidMask, maxScale, vdMaxOffset, scaleMask);
            Reg::Select(maxScale, vdMaxOffset, maxScale, invalidMask);
            Reg::Sub(preBroadCastScale, maxScale, vdMaxOffset, scaleMask);
            for (uint16_t k = 0; k < scale1OutLoopNum; k++) {
                Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(scale1LocalAddr, preBroadCastScale, vfLen8,
                                                                             scaleMask);
            }
            Reg::Interleave(preBroadCastScale2x, vdScaleTmp, preBroadCastScale, preBroadCastScale);
            Reg::Interleave(preBroadCastScale4x, vdScaleTmp, preBroadCastScale2x, preBroadCastScale2x);
            Reg::StoreAlign<uint32_t>(tmpBufAddr, (Reg::RegTensor<uint32_t>&)preBroadCastScale4x, scaleStoreMask);
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_E2B_B32>((Reg::RegTensor<uint32_t>&)vdScale2One, tmpBufAddr);
            Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_E2B_B32>((Reg::RegTensor<uint32_t>&)vdScale2Two,
                                                                  tmpBufAddr + DIGIT_EIGHT); // + 8
            Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(scale2InterleaveBufAddr, vdScale2One, vfLen8,
                                                                         scaleMask);
            Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(scale2InterleaveBufAddr, vdScale2Two, vfLen8,
                                                                         scaleMask);
            Reg::Cast<uint16_t, uint8_t, castTraitUint8ToUint16>(
                (Reg::RegTensor<uint16_t>&)vdScale2One, (Reg::RegTensor<uint8_t>&)vdScale2One,
                scaleMask); // uint8 无法直接转 int16，先转uint16再用int16进行解析
            Reg::Cast<uint16_t, uint8_t, castTraitUint8ToUint16>((Reg::RegTensor<uint16_t>&)vdScale2Two,
                                                                 (Reg::RegTensor<uint8_t>&)vdScale2Two, scaleMask);

            for (uint16_t j = 0; j < loopNumBlock; j++) {
                // Load 256 bytes of FP4 raw data
                Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK4_B8>(
                    (Reg::RegTensor<uint8_t>&)vdFp4U8_0, xLocalAddr,
                    vfLen32); // 01 00 00 00 23 00 00 00 45 00 00 00 ......
                Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK4_B8>(
                    (Reg::RegTensor<uint8_t>&)vdFp4U8_1, xLocalAddr, vfLen32); // 128129 00 00 00 130131 00 00 00 ......
                Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK4_B8>(
                    (Reg::RegTensor<uint8_t>&)vdFp4U8_2, xLocalAddr, vfLen32); // 256257 00 00 00 258259 00 00 00 ......
                Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK4_B8>(
                    (Reg::RegTensor<uint8_t>&)vdFp4U8_3, xLocalAddr, vfLen32); // 384385 00 00 00 386387 00 00 00 ......

                Reg::LoadAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B32>(
                    (Reg::RegTensor<uint32_t>&)vdScaleOne, scaleTmpBackBufAddr, DIGIT_EIGHT); // + 8
                Reg::LoadAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B32>(
                    (Reg::RegTensor<uint32_t>&)vdScaleTwo, scaleTmpBackBufAddr, DIGIT_EIGHT); // + 8

                // scale = 130(2**3) max_scale = 150(2**123)     130-(150-6)=-14   ===> 2**-14  ====> 130
                // 144->254-144=110  130+110-127=113 scale = -10 117 max_scale = 128 255 117 - (255 - 6) = -132   ==>
                // 249->254-249=5   117+5-127 = -5 -> 0   =====>scale 2**-127  * 1.5 = 0 scale = x/2^(z-y-6)=x*2^(y-z+6)
                // x = 1.5 (32*64)  y = 2**3  130  (32*2)   z = 2**10   137   (1*2)
                // 1.5*2**3   1.5*2**3/2**10=1.5*2**(3-10)  -->1.5*2**(3-10+6)  2**(10-6)

                Reg::Cast<uint16_t, uint8_t, castTraitUint8ToUint16>(
                    (Reg::RegTensor<uint16_t>&)vdScaleOne, (Reg::RegTensor<uint8_t>&)vdScaleOne,
                    scaleMask); // uint8 无法直接转 int16，先转uint16再用int16进行解析
                Reg::Cast<uint16_t, uint8_t, castTraitUint8ToUint16>((Reg::RegTensor<uint16_t>&)vdScaleTwo,
                                                                     (Reg::RegTensor<uint8_t>&)vdScaleTwo, scaleMask);
                Reg::Sub((Reg::RegTensor<int16_t>&)vdScaleOne, (Reg::RegTensor<int16_t>&)vdScaleOne,
                         (Reg::RegTensor<int16_t>&)vdScale2One, scaleMask);
                Reg::Sub((Reg::RegTensor<int16_t>&)vdScaleTwo, (Reg::RegTensor<int16_t>&)vdScaleTwo,
                         (Reg::RegTensor<int16_t>&)vdScale2Two, scaleMask);
                Reg::Adds((Reg::RegTensor<int16_t>&)vdScaleOne, (Reg::RegTensor<int16_t>&)vdScaleOne, 127, scaleMask);
                Reg::Adds((Reg::RegTensor<int16_t>&)vdScaleTwo, (Reg::RegTensor<int16_t>&)vdScaleTwo, 127, scaleMask);
                Reg::Compare<int16_t, CMPMODE::LE>(zeroMask_0, (Reg::RegTensor<int16_t>&)vdScaleOne, zeroRegTensor,
                                                   scaleMask);
                Reg::Compare<int16_t, CMPMODE::LE>(zeroMask_1, (Reg::RegTensor<int16_t>&)vdScaleTwo, zeroRegTensor,
                                                   scaleMask);
                Reg::Select((Reg::RegTensor<int16_t>&)vdScaleOne, zeroRegTensor, (Reg::RegTensor<int16_t>&)vdScaleOne,
                            zeroMask_0);
                Reg::Select((Reg::RegTensor<int16_t>&)vdScaleTwo, zeroRegTensor, (Reg::RegTensor<int16_t>&)vdScaleTwo,
                            zeroMask_1);

                Reg::Cast<bfloat16_t, T, castTraitFp4ToBf16>(vdFp4U8_0, (Reg::RegTensor<T>&)vdFp4U8_0,
                                                             scaleMask); // 0 - 127
                Reg::Cast<bfloat16_t, T, castTraitFp4ToBf16>(vdFp4U8_1, (Reg::RegTensor<T>&)vdFp4U8_1,
                                                             scaleMask); // 128 - 255
                Reg::Cast<bfloat16_t, T, castTraitFp4ToBf16>(vdFp4U8_2, (Reg::RegTensor<T>&)vdFp4U8_2,
                                                             scaleMask); // 256 - 383
                Reg::Cast<bfloat16_t, T, castTraitFp4ToBf16>(vdFp4U8_3, (Reg::RegTensor<T>&)vdFp4U8_3,
                                                             scaleMask); // 384 - 511

                Reg::ShiftLefts((Reg::RegTensor<uint16_t>&)vdScaleOneBf16, (Reg::RegTensor<uint16_t>&)vdScaleOne,
                                SHR_NUM_FOR_BF16, mulMask);
                Reg::ShiftLefts((Reg::RegTensor<uint16_t>&)vdScaleTwoBf16, (Reg::RegTensor<uint16_t>&)vdScaleTwo,
                                SHR_NUM_FOR_BF16, mulMask);
                Reg::Interleave((Reg::RegTensor<uint16_t>&)vdScaleBf16_0_0, (Reg::RegTensor<uint16_t>&)vdScaleBf16_0_1,
                                (Reg::RegTensor<uint16_t>&)vdScaleOneBf16, (Reg::RegTensor<uint16_t>&)vdScaleOneBf16);
                Reg::Interleave((Reg::RegTensor<uint16_t>&)vdScaleBf16_1_0, (Reg::RegTensor<uint16_t>&)vdScaleBf16_1_1,
                                (Reg::RegTensor<uint16_t>&)vdScaleTwoBf16, (Reg::RegTensor<uint16_t>&)vdScaleTwoBf16);

                Reg::Mul(vdFp4U8_0, vdFp4U8_0, vdScaleBf16_0_0, mulMask); // 0-127
                Reg::Mul(vdFp4U8_1, vdFp4U8_1, vdScaleBf16_0_1, mulMask); // 128-255
                Reg::Mul(vdFp4U8_2, vdFp4U8_2, vdScaleBf16_1_0, mulMask); // 256-383
                Reg::Mul(vdFp4U8_3, vdFp4U8_3, vdScaleBf16_1_1, mulMask); // 384-511

                Reg::Cast<float, bfloat16_t, castTraitBf16ToFp32_0>((Reg::RegTensor<float>&)vdScaleBf16_0_0, vdFp4U8_0,
                                                                    scaleMask);
                Reg::Cast<float, bfloat16_t, castTraitBf16ToFp32_1>((Reg::RegTensor<float>&)vdScaleBf16_0_1, vdFp4U8_0,
                                                                    scaleMask);
                Reg::Cast<float, bfloat16_t, castTraitBf16ToFp32_0>((Reg::RegTensor<float>&)vdScaleBf16_0_2, vdFp4U8_1,
                                                                    scaleMask);
                Reg::Cast<float, bfloat16_t, castTraitBf16ToFp32_1>((Reg::RegTensor<float>&)vdScaleBf16_0_3, vdFp4U8_1,
                                                                    scaleMask);
                Reg::Cast<float, bfloat16_t, castTraitBf16ToFp32_0>((Reg::RegTensor<float>&)vdScaleBf16_1_0, vdFp4U8_2,
                                                                    scaleMask);
                Reg::Cast<float, bfloat16_t, castTraitBf16ToFp32_1>((Reg::RegTensor<float>&)vdScaleBf16_1_1, vdFp4U8_2,
                                                                    scaleMask);
                Reg::Cast<float, bfloat16_t, castTraitBf16ToFp32_0>((Reg::RegTensor<float>&)vdScaleBf16_1_2, vdFp4U8_3,
                                                                    scaleMask);
                Reg::Cast<float, bfloat16_t, castTraitBf16ToFp32_1>((Reg::RegTensor<float>&)vdScaleBf16_1_3, vdFp4U8_3,
                                                                    scaleMask);

                Reg::Cast<U, float, castTrait32to80>((Reg::RegTensor<U>&)vdScaleBf16_0_0,
                                                     (Reg::RegTensor<float>&)vdScaleBf16_0_0, scaleMask);
                Reg::Cast<U, float, castTrait32to81>((Reg::RegTensor<U>&)vdScaleBf16_0_1,
                                                     (Reg::RegTensor<float>&)vdScaleBf16_0_1, scaleMask);
                Reg::Cast<U, float, castTrait32to80>((Reg::RegTensor<U>&)vdScaleBf16_0_2,
                                                     (Reg::RegTensor<float>&)vdScaleBf16_0_2, scaleMask);
                Reg::Cast<U, float, castTrait32to81>((Reg::RegTensor<U>&)vdScaleBf16_0_3,
                                                     (Reg::RegTensor<float>&)vdScaleBf16_0_3, scaleMask);
                Reg::Cast<U, float, castTrait32to82>((Reg::RegTensor<U>&)vdScaleBf16_1_0,
                                                     (Reg::RegTensor<float>&)vdScaleBf16_1_0, scaleMask);
                Reg::Cast<U, float, castTrait32to83>((Reg::RegTensor<U>&)vdScaleBf16_1_1,
                                                     (Reg::RegTensor<float>&)vdScaleBf16_1_1, scaleMask);
                Reg::Cast<U, float, castTrait32to82>((Reg::RegTensor<U>&)vdScaleBf16_1_2,
                                                     (Reg::RegTensor<float>&)vdScaleBf16_1_2, scaleMask);
                Reg::Cast<U, float, castTrait32to83>((Reg::RegTensor<U>&)vdScaleBf16_1_3,
                                                     (Reg::RegTensor<float>&)vdScaleBf16_1_3, scaleMask);

                Reg::Add((Reg::RegTensor<uint8_t>&)vdScaleBf16_0_0, (Reg::RegTensor<uint8_t>&)vdScaleBf16_0_0,
                         (Reg::RegTensor<uint8_t>&)vdScaleBf16_0_1, scaleMask);
                Reg::Add((Reg::RegTensor<uint8_t>&)vdScaleBf16_0_2, (Reg::RegTensor<uint8_t>&)vdScaleBf16_0_2,
                         (Reg::RegTensor<uint8_t>&)vdScaleBf16_0_3, scaleMask);
                Reg::Add((Reg::RegTensor<uint8_t>&)vdScaleBf16_1_0, (Reg::RegTensor<uint8_t>&)vdScaleBf16_1_0,
                         (Reg::RegTensor<uint8_t>&)vdScaleBf16_1_1, scaleMask);
                Reg::Add((Reg::RegTensor<uint8_t>&)vdScaleBf16_1_2, (Reg::RegTensor<uint8_t>&)vdScaleBf16_1_2,
                         (Reg::RegTensor<uint8_t>&)vdScaleBf16_1_3, scaleMask);
                Reg::Add((Reg::RegTensor<uint8_t>&)vdScaleBf16_0_0, (Reg::RegTensor<uint8_t>&)vdScaleBf16_0_0,
                         (Reg::RegTensor<uint8_t>&)vdScaleBf16_1_0, scaleMask);
                Reg::Add((Reg::RegTensor<uint8_t>&)vdScaleBf16_0_2, (Reg::RegTensor<uint8_t>&)vdScaleBf16_0_2,
                         (Reg::RegTensor<uint8_t>&)vdScaleBf16_1_2, scaleMask);

                Reg::DeInterleave(
                    (Reg::RegTensor<uint16_t>&)vdScaleBf16_0_0, (Reg::RegTensor<uint16_t>&)vdScaleBf16_0_2,
                    (Reg::RegTensor<uint16_t>&)vdScaleBf16_0_0, (Reg::RegTensor<uint16_t>&)vdScaleBf16_0_2);

                Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                    yLocalAddr, (Reg::RegTensor<uint8_t>&)vdScaleBf16_0_0, vfLen8, scaleMask);
                Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                    yLocalAddr, (Reg::RegTensor<uint8_t>&)vdScaleBf16_0_2, vfLen8, scaleMask);
            }
        }

        Reg::LocalMemBar<Reg::MemType::VEC_STORE,
                         Reg::MemType::VEC_LOAD>(); // 不加这个MemBar，scale2 会出现偶现的精度问题
        for (uint16_t i = 0; i < loopNum; i++) {
            Reg::LoadAlign<uint8_t>(vdScale2One, scale2InterleaveBackBufAddr + 0 * vfLen8 * 2 + i * vfLen8);
            Reg::LoadAlign<uint8_t>(vdScale2Two, scale2InterleaveBackBufAddr + 1 * vfLen8 * 2 + i * vfLen8);
            Reg::Interleave(vdScale2One, vdScale2Two, vdScale2One, vdScale2Two);
            Reg::StoreAlign<uint8_t>(scale2LocalAddr + i * vfLen8 * 2 + 0 * vfLen8, vdScale2One, scaleMask);
            Reg::StoreAlign<uint8_t>(scale2LocalAddr + i * vfLen8 * 2 + 1 * vfLen8, vdScale2Two, scaleMask);
        }
    }

    inQueueX_.FreeTensor(xLocal);
    inQueueMxScale_.FreeTensor(scaleLocal);
    outQueueY_.EnQue(yLocal);
    outQueueScale1_.EnQue(scale1Local);
    outQueueScale2_.EnQue(scale2Local);
}

// ============================================================================
// CopyOut — 按全局 (batch, rowBlock, colBlock) 偏移搬出，scale2 为 2 行交织。
// ============================================================================
template <typename T, typename U, uint64_t rowMode>
__aicore__ inline void MxToBlockMxQuant<T, U, rowMode>::CopyOut(int64_t batch, int64_t rowBlock, int64_t colBlock,
                                                                int64_t curRowNum, int64_t curColNum)
{
    int64_t rowStart = batch * rowNum_ + rowBlock * SPLIT_M;
    int64_t colStart = colBlock * SPLIT_N;

    // ----- Copy y（FP8 1 byte）-----
    LocalTensor<uint8_t> yLocal = outQueueY_.DeQue<uint8_t>();
    int64_t yOffset = rowStart * colNum_ + colStart;
    int64_t anotherStrideY = (SPLIT_N - curColNum) * sizeof(uint8_t) / UBBlockSize_;
    DataCopyExtParams copyOutParamY = {
        static_cast<uint16_t>(curRowNum), static_cast<uint32_t>(curColNum * sizeof(uint8_t)),
        static_cast<uint32_t>(anotherStrideY), static_cast<uint32_t>((colNum_ - curColNum) * sizeof(uint8_t)), 0};
    DataCopyPad(yGm_[yOffset], yLocal, copyOutParamY);
    outQueueY_.FreeTensor(yLocal);

    // ----- Copy scale1（FP8_E8M0）-----
    int64_t ubFactorColBlockNum = ops::CeilDiv(curColNum, BLOCK_SIZE);
    int64_t ubFactorColScaleNum = ops::CeilDiv(ubFactorColBlockNum, DIGIT_TWO) * DIGIT_TWO;
    LocalTensor<uint8_t> scale1Local = outQueueScale1_.DeQue<uint8_t>();
    int64_t scale1Offset = rowStart * colScaleNum_ + colBlock * DIGIT_SIXTEEN;
    DataCopyExtParams copyOutParamScale1 = {
        static_cast<uint16_t>(curRowNum), static_cast<uint32_t>(ubFactorColScaleNum * sizeof(uint8_t)),
        static_cast<uint32_t>(0), static_cast<uint32_t>((colScaleNum_ - ubFactorColScaleNum) * sizeof(uint8_t)), 0};
    DataCopyPad(scale1Gm_[scale1Offset], scale1Local, copyOutParamScale1);
    outQueueScale1_.FreeTensor(scale1Local);

    // ----- Copy scale2（FP8_E8M0，2 行交织）-----
    LocalTensor<uint8_t> scale2Local = outQueueScale2_.DeQue<uint8_t>();
    int64_t scale2Offset = batch * rowBlockNumPerBatch_ * DIGIT_TWO * colNum_ + rowBlock * DIGIT_TWO * colNum_ +
                           colBlock * DIGIT_TWO * SPLIT_N;
    DataCopyExtParams copyOutParamScale2 = {static_cast<uint16_t>(DIGIT_TWO),
                                            static_cast<uint32_t>(curColNum * sizeof(uint8_t)),
                                            static_cast<uint32_t>(0), static_cast<uint32_t>(0), 0};
    DataCopyPad<uint8_t, PaddingMode::Compact>(scale2Gm_[scale2Offset], scale2Local, copyOutParamScale2);
    outQueueScale2_.FreeTensor(scale2Local);
}

// ----------------------------------------------------------------------------
// SubCompute — 一个 32 行组的 VF 反量化计算（__VEC_SCOPE__ 内）。
//              因 CopyIn 已 pad 到完整 curRowNum×512，内部按 loop 常量处理。
// ----------------------------------------------------------------------------
template <typename T, typename U, uint64_t rowMode>
__aicore__ inline void MxToBlockMxQuant<T, U, rowMode>::SubCompute(
    uint16_t loop, uint16_t scaleLoopStep, uint16_t scale1OutLoopNum, __ubuf__ uint8_t* xLocalAddr,
    __ubuf__ uint8_t* mxScaleLocalAddr, __ubuf__ uint8_t* yLocalAddr, __ubuf__ uint8_t* scale1LocalAddr,
    __ubuf__ uint32_t* scaleTmpBufAddr, __ubuf__ uint32_t* tmpBufAddr, __ubuf__ uint8_t* scale2InterleaveBufAddr,
    __ubuf__ uint32_t* scaleTmpBackBufAddr, Reg::RegTensor<uint8_t>& vdMaxOffset,
    Reg::RegTensor<int16_t>& zeroRegTensor)
{
    Reg::MaskReg scaleMask = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg scaleStoreMask = Reg::CreateMask<uint32_t, Reg::MaskPattern::VL16>();
    Reg::MaskReg mulMask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();

    Reg::RegTensor<uint8_t> maxScale;
    Reg::RegTensor<uint8_t> preBroadCastScale, preBroadCastScale2x, preBroadCastScale4x;
    Reg::RegTensor<uint8_t> vdScale, vdScale2x, vdScale4x, vdScaleTmp;
    Reg::RegTensor<uint8_t> vdScale2One, vdScale2Two;
    Reg::RegTensor<uint8_t> vdScaleOne, vdScaleTwo;
    Reg::RegTensor<bfloat16_t> vdScaleOneBf16, vdScaleTwoBf16;
    Reg::RegTensor<bfloat16_t> vdScaleBf16_0_0, vdScaleBf16_0_1, vdScaleBf16_0_2, vdScaleBf16_0_3, vdScaleBf16_1_0,
        vdScaleBf16_1_1, vdScaleBf16_1_2, vdScaleBf16_1_3;
    Reg::RegTensor<bfloat16_t> vdFp4U8_0, vdFp4U8_1, vdFp4U8_2, vdFp4U8_3;
    Reg::MaskReg invalidMask;
    Reg::MaskReg zeroMask_0;
    Reg::MaskReg zeroMask_1;

    Reg::Duplicate(maxScale, 0);
    for (uint16_t j = 0; j < loop; j++) {
        Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_BLK>(vdScale, mxScaleLocalAddr,
                                                                                             scaleLoopStep);
        Reg::Max(maxScale, maxScale, vdScale, scaleMask);
        Reg::Interleave(vdScale2x, vdScaleTmp, vdScale, vdScale);
        Reg::Interleave(vdScale4x, vdScaleTmp, vdScale2x, vdScale2x);
        Reg::StoreAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(
            scaleTmpBufAddr, (Reg::RegTensor<uint32_t>&)vdScale4x, DIGIT_SIXTEEN, scaleStoreMask);
    }

    Reg::Compare<uint8_t, CMPMODE::LE>(invalidMask, maxScale, vdMaxOffset, scaleMask);
    Reg::Select(maxScale, vdMaxOffset, maxScale, invalidMask);
    Reg::Sub(preBroadCastScale, maxScale, vdMaxOffset, scaleMask);
    for (uint16_t k = 0; k < scale1OutLoopNum; k++) {
        Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(scale1LocalAddr, preBroadCastScale, vfLen8,
                                                                     scaleMask);
    }
    Reg::Interleave(preBroadCastScale2x, vdScaleTmp, preBroadCastScale, preBroadCastScale);
    Reg::Interleave(preBroadCastScale4x, vdScaleTmp, preBroadCastScale2x, preBroadCastScale2x);
    Reg::StoreAlign<uint32_t>(tmpBufAddr, (Reg::RegTensor<uint32_t>&)preBroadCastScale4x, scaleStoreMask);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_E2B_B32>((Reg::RegTensor<uint32_t>&)vdScale2One, tmpBufAddr);
    Reg::LoadAlign<uint32_t, Reg::LoadDist::DIST_E2B_B32>((Reg::RegTensor<uint32_t>&)vdScale2Two,
                                                          tmpBufAddr + DIGIT_EIGHT);
    Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(scale2InterleaveBufAddr, vdScale2One, vfLen8,
                                                                 scaleMask);
    Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(scale2InterleaveBufAddr, vdScale2Two, vfLen8,
                                                                 scaleMask);
    Reg::Cast<uint16_t, uint8_t, castTraitUint8ToUint16>((Reg::RegTensor<uint16_t>&)vdScale2One,
                                                         (Reg::RegTensor<uint8_t>&)vdScale2One, scaleMask);
    Reg::Cast<uint16_t, uint8_t, castTraitUint8ToUint16>((Reg::RegTensor<uint16_t>&)vdScale2Two,
                                                         (Reg::RegTensor<uint8_t>&)vdScale2Two, scaleMask);

    for (uint16_t j = 0; j < loop; j++) {
        Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK4_B8>(
            (Reg::RegTensor<uint8_t>&)vdFp4U8_0, xLocalAddr, vfLen32);
        Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK4_B8>(
            (Reg::RegTensor<uint8_t>&)vdFp4U8_1, xLocalAddr, vfLen32);
        Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK4_B8>(
            (Reg::RegTensor<uint8_t>&)vdFp4U8_2, xLocalAddr, vfLen32);
        Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK4_B8>(
            (Reg::RegTensor<uint8_t>&)vdFp4U8_3, xLocalAddr, vfLen32);

        Reg::LoadAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B32>(
            (Reg::RegTensor<uint32_t>&)vdScaleOne, scaleTmpBackBufAddr, DIGIT_EIGHT);
        Reg::LoadAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B32>(
            (Reg::RegTensor<uint32_t>&)vdScaleTwo, scaleTmpBackBufAddr, DIGIT_EIGHT);

        Reg::Cast<uint16_t, uint8_t, castTraitUint8ToUint16>((Reg::RegTensor<uint16_t>&)vdScaleOne,
                                                             (Reg::RegTensor<uint8_t>&)vdScaleOne, scaleMask);
        Reg::Cast<uint16_t, uint8_t, castTraitUint8ToUint16>((Reg::RegTensor<uint16_t>&)vdScaleTwo,
                                                             (Reg::RegTensor<uint8_t>&)vdScaleTwo, scaleMask);
        Reg::Sub((Reg::RegTensor<int16_t>&)vdScaleOne, (Reg::RegTensor<int16_t>&)vdScaleOne,
                 (Reg::RegTensor<int16_t>&)vdScale2One, scaleMask);
        Reg::Sub((Reg::RegTensor<int16_t>&)vdScaleTwo, (Reg::RegTensor<int16_t>&)vdScaleTwo,
                 (Reg::RegTensor<int16_t>&)vdScale2Two, scaleMask);
        Reg::Adds((Reg::RegTensor<int16_t>&)vdScaleOne, (Reg::RegTensor<int16_t>&)vdScaleOne, 127, scaleMask);
        Reg::Adds((Reg::RegTensor<int16_t>&)vdScaleTwo, (Reg::RegTensor<int16_t>&)vdScaleTwo, 127, scaleMask);
        Reg::Compare<int16_t, CMPMODE::LE>(zeroMask_0, (Reg::RegTensor<int16_t>&)vdScaleOne, zeroRegTensor, scaleMask);
        Reg::Compare<int16_t, CMPMODE::LE>(zeroMask_1, (Reg::RegTensor<int16_t>&)vdScaleTwo, zeroRegTensor, scaleMask);
        Reg::Select((Reg::RegTensor<int16_t>&)vdScaleOne, zeroRegTensor, (Reg::RegTensor<int16_t>&)vdScaleOne,
                    zeroMask_0);
        Reg::Select((Reg::RegTensor<int16_t>&)vdScaleTwo, zeroRegTensor, (Reg::RegTensor<int16_t>&)vdScaleTwo,
                    zeroMask_1);

        Reg::Cast<bfloat16_t, T, castTraitFp4ToBf16>(vdFp4U8_0, (Reg::RegTensor<T>&)vdFp4U8_0, scaleMask);
        Reg::Cast<bfloat16_t, T, castTraitFp4ToBf16>(vdFp4U8_1, (Reg::RegTensor<T>&)vdFp4U8_1, scaleMask);
        Reg::Cast<bfloat16_t, T, castTraitFp4ToBf16>(vdFp4U8_2, (Reg::RegTensor<T>&)vdFp4U8_2, scaleMask);
        Reg::Cast<bfloat16_t, T, castTraitFp4ToBf16>(vdFp4U8_3, (Reg::RegTensor<T>&)vdFp4U8_3, scaleMask);

        Reg::ShiftLefts((Reg::RegTensor<uint16_t>&)vdScaleOneBf16, (Reg::RegTensor<uint16_t>&)vdScaleOne,
                        SHR_NUM_FOR_BF16, mulMask);
        Reg::ShiftLefts((Reg::RegTensor<uint16_t>&)vdScaleTwoBf16, (Reg::RegTensor<uint16_t>&)vdScaleTwo,
                        SHR_NUM_FOR_BF16, mulMask);
        Reg::Interleave((Reg::RegTensor<uint16_t>&)vdScaleBf16_0_0, (Reg::RegTensor<uint16_t>&)vdScaleBf16_0_1,
                        (Reg::RegTensor<uint16_t>&)vdScaleOneBf16, (Reg::RegTensor<uint16_t>&)vdScaleOneBf16);
        Reg::Interleave((Reg::RegTensor<uint16_t>&)vdScaleBf16_1_0, (Reg::RegTensor<uint16_t>&)vdScaleBf16_1_1,
                        (Reg::RegTensor<uint16_t>&)vdScaleTwoBf16, (Reg::RegTensor<uint16_t>&)vdScaleTwoBf16);

        Reg::Mul(vdFp4U8_0, vdFp4U8_0, vdScaleBf16_0_0, mulMask);
        Reg::Mul(vdFp4U8_1, vdFp4U8_1, vdScaleBf16_0_1, mulMask);
        Reg::Mul(vdFp4U8_2, vdFp4U8_2, vdScaleBf16_1_0, mulMask);
        Reg::Mul(vdFp4U8_3, vdFp4U8_3, vdScaleBf16_1_1, mulMask);

        Reg::Cast<float, bfloat16_t, castTraitBf16ToFp32_0>((Reg::RegTensor<float>&)vdScaleBf16_0_0, vdFp4U8_0,
                                                            scaleMask);
        Reg::Cast<float, bfloat16_t, castTraitBf16ToFp32_1>((Reg::RegTensor<float>&)vdScaleBf16_0_1, vdFp4U8_0,
                                                            scaleMask);
        Reg::Cast<float, bfloat16_t, castTraitBf16ToFp32_0>((Reg::RegTensor<float>&)vdScaleBf16_0_2, vdFp4U8_1,
                                                            scaleMask);
        Reg::Cast<float, bfloat16_t, castTraitBf16ToFp32_1>((Reg::RegTensor<float>&)vdScaleBf16_0_3, vdFp4U8_1,
                                                            scaleMask);
        Reg::Cast<float, bfloat16_t, castTraitBf16ToFp32_0>((Reg::RegTensor<float>&)vdScaleBf16_1_0, vdFp4U8_2,
                                                            scaleMask);
        Reg::Cast<float, bfloat16_t, castTraitBf16ToFp32_1>((Reg::RegTensor<float>&)vdScaleBf16_1_1, vdFp4U8_2,
                                                            scaleMask);
        Reg::Cast<float, bfloat16_t, castTraitBf16ToFp32_0>((Reg::RegTensor<float>&)vdScaleBf16_1_2, vdFp4U8_3,
                                                            scaleMask);
        Reg::Cast<float, bfloat16_t, castTraitBf16ToFp32_1>((Reg::RegTensor<float>&)vdScaleBf16_1_3, vdFp4U8_3,
                                                            scaleMask);

        Reg::Cast<U, float, castTrait32to80>((Reg::RegTensor<U>&)vdScaleBf16_0_0,
                                             (Reg::RegTensor<float>&)vdScaleBf16_0_0, scaleMask);
        Reg::Cast<U, float, castTrait32to81>((Reg::RegTensor<U>&)vdScaleBf16_0_1,
                                             (Reg::RegTensor<float>&)vdScaleBf16_0_1, scaleMask);
        Reg::Cast<U, float, castTrait32to80>((Reg::RegTensor<U>&)vdScaleBf16_0_2,
                                             (Reg::RegTensor<float>&)vdScaleBf16_0_2, scaleMask);
        Reg::Cast<U, float, castTrait32to81>((Reg::RegTensor<U>&)vdScaleBf16_0_3,
                                             (Reg::RegTensor<float>&)vdScaleBf16_0_3, scaleMask);
        Reg::Cast<U, float, castTrait32to82>((Reg::RegTensor<U>&)vdScaleBf16_1_0,
                                             (Reg::RegTensor<float>&)vdScaleBf16_1_0, scaleMask);
        Reg::Cast<U, float, castTrait32to83>((Reg::RegTensor<U>&)vdScaleBf16_1_1,
                                             (Reg::RegTensor<float>&)vdScaleBf16_1_1, scaleMask);
        Reg::Cast<U, float, castTrait32to82>((Reg::RegTensor<U>&)vdScaleBf16_1_2,
                                             (Reg::RegTensor<float>&)vdScaleBf16_1_2, scaleMask);
        Reg::Cast<U, float, castTrait32to83>((Reg::RegTensor<U>&)vdScaleBf16_1_3,
                                             (Reg::RegTensor<float>&)vdScaleBf16_1_3, scaleMask);

        Reg::Add((Reg::RegTensor<uint8_t>&)vdScaleBf16_0_0, (Reg::RegTensor<uint8_t>&)vdScaleBf16_0_0,
                 (Reg::RegTensor<uint8_t>&)vdScaleBf16_0_1, scaleMask);
        Reg::Add((Reg::RegTensor<uint8_t>&)vdScaleBf16_0_2, (Reg::RegTensor<uint8_t>&)vdScaleBf16_0_2,
                 (Reg::RegTensor<uint8_t>&)vdScaleBf16_0_3, scaleMask);
        Reg::Add((Reg::RegTensor<uint8_t>&)vdScaleBf16_1_0, (Reg::RegTensor<uint8_t>&)vdScaleBf16_1_0,
                 (Reg::RegTensor<uint8_t>&)vdScaleBf16_1_1, scaleMask);
        Reg::Add((Reg::RegTensor<uint8_t>&)vdScaleBf16_1_2, (Reg::RegTensor<uint8_t>&)vdScaleBf16_1_2,
                 (Reg::RegTensor<uint8_t>&)vdScaleBf16_1_3, scaleMask);
        Reg::Add((Reg::RegTensor<uint8_t>&)vdScaleBf16_0_0, (Reg::RegTensor<uint8_t>&)vdScaleBf16_0_0,
                 (Reg::RegTensor<uint8_t>&)vdScaleBf16_1_0, scaleMask);
        Reg::Add((Reg::RegTensor<uint8_t>&)vdScaleBf16_0_2, (Reg::RegTensor<uint8_t>&)vdScaleBf16_0_2,
                 (Reg::RegTensor<uint8_t>&)vdScaleBf16_1_2, scaleMask);

        Reg::DeInterleave((Reg::RegTensor<uint16_t>&)vdScaleBf16_0_0, (Reg::RegTensor<uint16_t>&)vdScaleBf16_0_2,
                          (Reg::RegTensor<uint16_t>&)vdScaleBf16_0_0, (Reg::RegTensor<uint16_t>&)vdScaleBf16_0_2);

        Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(
            yLocalAddr, (Reg::RegTensor<uint8_t>&)vdScaleBf16_0_0, vfLen8, scaleMask);
        Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(
            yLocalAddr, (Reg::RegTensor<uint8_t>&)vdScaleBf16_0_2, vfLen8, scaleMask);
    }
}

// ----------------------------------------------------------------------------
// Compute<padMode> — 把一个 64 行块拆成「前 32 行组 + 可选后 32 行组」分别处理。
//                    padMode=0：只有前 32 行组（curRowNum<=32），scale2 第二行补 0。
//                    padMode=1：前 32 行组 + 后 32 行组（curRowNum>32）。
// ----------------------------------------------------------------------------
template <typename T, typename U, uint64_t rowMode>
template <const int64_t padMode>
__aicore__ inline void MxToBlockMxQuant<T, U, rowMode>::Compute(
    uint16_t loop0, uint16_t loop1, __ubuf__ uint8_t* xLocalAddr, __ubuf__ uint8_t* mxScaleLocalAddr,
    __ubuf__ uint8_t* yLocalAddr, __ubuf__ uint8_t* scale1LocalAddr, __ubuf__ uint8_t* scale2LocalAddr,
    __ubuf__ uint32_t* scaleTmpBufAddr, __ubuf__ uint32_t* tmpBufAddr, __ubuf__ uint8_t* scale2InterleaveBufAddr)
{
    auto scaleTmpBackBufAddr = scaleTmpBufAddr;
    auto scale2InterleaveBackBufAddr = scale2InterleaveBufAddr;
    uint16_t loopNum = 2;
    uint32_t xOffset = 256 * BLOCK_SIZE;
    uint32_t mxScaleOffset = 32 * BLOCK_SIZE;
    uint32_t yOffset = 512 * BLOCK_SIZE;
    uint32_t scaleTmpOffset = 16 * BLOCK_SIZE;
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint8_t> vdMaxOffset;
        Reg::RegTensor<int16_t> zeroRegTensor;
        Reg::RegTensor<uint8_t> vdScale2One, vdScale2Two;
        Reg::MaskReg scaleMask = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

        Reg::Duplicate(vdMaxOffset, maxOffset);
        Reg::Duplicate(zeroRegTensor, 0);

        SubCompute(loop0, scaleLoopStep, scale1OutLoopNum, xLocalAddr, mxScaleLocalAddr, yLocalAddr, scale1LocalAddr,
                   scaleTmpBufAddr, tmpBufAddr, scale2InterleaveBufAddr, scaleTmpBackBufAddr, vdMaxOffset,
                   zeroRegTensor);
        if constexpr (padMode == 1) {
            SubCompute(loop1, scaleLoopStep, scale1OutLoopNum, xLocalAddr + xOffset, mxScaleLocalAddr + mxScaleOffset,
                       yLocalAddr + yOffset, scale1LocalAddr + mxScaleOffset, scaleTmpBufAddr + scaleTmpOffset,
                       tmpBufAddr, scale2InterleaveBufAddr + SPLIT_N, scaleTmpBackBufAddr + scaleTmpOffset, vdMaxOffset,
                       zeroRegTensor);
        }
        Reg::LocalMemBar<Reg::MemType::VEC_STORE,
                         Reg::MemType::VEC_LOAD>(); // 不加这个 MemBar，scale2 会出现偶现的精度问题
        for (uint16_t i = 0; i < loopNum; i++) {
            Reg::LoadAlign<uint8_t>(vdScale2One, scale2InterleaveBackBufAddr + 0 * vfLen8 * 2 + i * vfLen8);
            if constexpr (padMode == 0) {
                Reg::Duplicate(vdScale2Two, 0);
            } else {
                Reg::LoadAlign<uint8_t>(vdScale2Two, scale2InterleaveBackBufAddr + 1 * vfLen8 * 2 + i * vfLen8);
            }
            Reg::Interleave(vdScale2One, vdScale2Two, vdScale2One, vdScale2Two);
            Reg::StoreAlign<uint8_t>(scale2LocalAddr + i * vfLen8 * 2 + 0 * vfLen8, vdScale2One, scaleMask);
            Reg::StoreAlign<uint8_t>(scale2LocalAddr + i * vfLen8 * 2 + 1 * vfLen8, vdScale2Two, scaleMask);
        }
    }
}

// ----------------------------------------------------------------------------
// ComputeNotAligned — 兜底 Compute 入口：按 curRowNum 选 padMode，分配 tensor 并取地址。
// ----------------------------------------------------------------------------
template <typename T, typename U, uint64_t rowMode>
__aicore__ inline void MxToBlockMxQuant<T, U, rowMode>::ComputeNotAligned(int64_t ubFactorRowNum,
                                                                          int64_t ubFactorColNum)
{
    LocalTensor<uint8_t> xLocal = inQueueX_.DeQue<uint8_t>();
    LocalTensor<uint8_t> scaleLocal = inQueueMxScale_.DeQue<uint8_t>();
    LocalTensor<uint8_t> yLocal = outQueueY_.AllocTensor<uint8_t>();
    LocalTensor<uint8_t> scale1Local = outQueueScale1_.AllocTensor<uint8_t>();
    LocalTensor<uint8_t> scale2Local = outQueueScale2_.AllocTensor<uint8_t>();
    LocalTensor<uint8_t> scale2InterLeaveBuf = scale2InterleaveBuffer_.Get<uint8_t>();

    auto xLocalAddr = reinterpret_cast<__ubuf__ uint8_t*>(xLocal.GetPhyAddr());
    auto mxScaleLocalAddr = reinterpret_cast<__ubuf__ uint8_t*>(scaleLocal.GetPhyAddr());
    auto yLocalAddr = reinterpret_cast<__ubuf__ uint8_t*>(yLocal.GetPhyAddr());
    auto scale1LocalAddr = reinterpret_cast<__ubuf__ uint8_t*>(scale1Local.GetPhyAddr());
    auto scale2LocalAddr = reinterpret_cast<__ubuf__ uint8_t*>(scale2Local.GetPhyAddr());

    auto scaleTmpBufAddr = reinterpret_cast<__ubuf__ uint32_t*>(scaleTmpBuffer_.Get<uint32_t>().GetPhyAddr());
    auto tmpBufAddr = reinterpret_cast<__ubuf__ uint32_t*>(tmpBuffer_.Get<uint32_t>().GetPhyAddr());
    auto scale2InterleaveBufAddr = reinterpret_cast<__ubuf__ uint8_t*>(scale2InterLeaveBuf.GetPhyAddr());

    uint16_t loop0 = 0;
    uint16_t loop1 = 0;
    if (ubFactorRowNum <= BLOCK_SIZE) {
        loop0 = static_cast<uint16_t>(ubFactorRowNum);
        loop1 = 0;
        Compute<0>(loop0, loop1, xLocalAddr, mxScaleLocalAddr, yLocalAddr, scale1LocalAddr, scale2LocalAddr,
                   scaleTmpBufAddr, tmpBufAddr, scale2InterleaveBufAddr);
    } else {
        loop0 = static_cast<uint16_t>(BLOCK_SIZE);
        loop1 = static_cast<uint16_t>(ubFactorRowNum - BLOCK_SIZE);
        Compute<1>(loop0, loop1, xLocalAddr, mxScaleLocalAddr, yLocalAddr, scale1LocalAddr, scale2LocalAddr,
                   scaleTmpBufAddr, tmpBufAddr, scale2InterleaveBufAddr);
    }

    inQueueX_.FreeTensor(xLocal);
    inQueueMxScale_.FreeTensor(scaleLocal);
    outQueueY_.EnQue(yLocal);
    outQueueScale1_.EnQue(scale1Local);
    outQueueScale2_.EnQue(scale2Local);
}

} // namespace MxToBlockMxQuantNs

// Expose kernel class at global scope for the entry file
template <typename T, typename U, uint64_t rowMode>
using MxToBlockMxQuant = MxToBlockMxQuantNs::MxToBlockMxQuant<T, U, rowMode>;

#endif // MX_TO_BLOCK_MX_QUANT_H
