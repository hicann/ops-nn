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
 * \file anti_mx_quant_tail_axis.h
 * \brief Tail-axis dequantization kernel for AntiMxQuant operator.
 * \tparam T  Input dtype: fp8_e4m3fn_t, fp8_e5m2_t, fp4x2_e2m1_t, fp4x2_e1m2_t
 * \tparam U  Output dtype: half, bfloat16_t, float
 */

#ifndef ANTI_MX_QUANT_TAIL_AXIS_H
#define ANTI_MX_QUANT_TAIL_AXIS_H
#define FLOAT_OVERFLOW_MODE_CTRL 60
#include "anti_mx_quant_common.h"

namespace AntiMxQuant {
using namespace AscendC;

/**
 * @brief Tail-axis AntiMxQuant kernel class.
 * @tparam T Input data type (FP4/FP8)
 * @tparam U Output data type (FP16/BF16/FP32)
 */
template <typename T, typename U>
class AntiMxQuantTailAxis {
public:
    __aicore__ inline AntiMxQuantTailAxis() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR mxScale, GM_ADDR y,
                                const AntiMxQuantTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const AntiMxQuantTilingData* tilingData);
    __aicore__ inline void GetGmParams();
    __aicore__ inline void GetUbParams();

    __aicore__ inline void CopyIn(int64_t rowLoopIdx, int64_t colLoopIdx,
                                  int64_t ubFactorRowNum, int64_t ubFactorColNum, int64_t ubFactorColBlockNum);
    __aicore__ inline void CopyOut(int64_t rowLoopIdx, int64_t colLoopIdx,
                                   int64_t ubFactorRowNum, int64_t ubFactorColNum, int64_t ubFactorColBlockNum);

    // Main compute: calls ComputeScale then ComputeData
    __aicore__ inline void Compute(int64_t rowBlockNum, int64_t colBlockNum);

    // Step 1: Preprocess mxscale (FP8_E8M0 -> compute type)
    __aicore__ inline void ComputeScale(
        __ubuf__ uint8_t* scaleLocalAddr, __ubuf__ float* scaleBufAddr, int64_t scaleNum);
    __aicore__ inline void ComputeScale(
        __ubuf__ uint8_t* scaleLocalAddr, __ubuf__ bfloat16_t* scaleBufAddr, int64_t scaleNum);

    // Step 2: Dequantization data compute
    __aicore__ inline void ComputeData(
        __ubuf__ uint8_t* xLocalAddr, __ubuf__ float* scaleBufAddr, __ubuf__ U* yLocalAddr, uint16_t loopNum2VF);
    __aicore__ inline void ComputeData(
        __ubuf__ uint8_t* xLocalAddr, __ubuf__ bfloat16_t* scaleBufAddr, __ubuf__ U* yLocalAddr, uint16_t loopNum2VF);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, DB_BUFFER> inQueueX_;
    TQue<QuePosition::VECIN, DB_BUFFER> inQueueScale_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueueY_;

    TBuf<QuePosition::VECCALC> scaleBuffer_;

    GlobalTensor<uint8_t> xGm_;
    GlobalTensor<uint8_t> mxScaleGm_;
    GlobalTensor<U> yGm_;

    int64_t totalCoreNum_{0};
    int64_t usedCoreNum_{0};
    int64_t rowTileNum_{0};
    int64_t colTileNum_{0};
    int64_t rowNum_{1};
    int64_t colNum_{1};
    int64_t colNormalBlockNum_{0};
    int64_t colTailLen_{0};
    int64_t rowNormalBlockNum_{0};
    int64_t rowTailLen_{0};
    int64_t maxUbBlockNum_{0};
    int64_t dstType_{0};

    int64_t coreIdx_{0};
    int64_t coreColIdx_{0};
    int64_t coreRowIdx_{0};
    int64_t xGmOffset_{0};
    int64_t scaleGmOffset_{0};
    int64_t scaleColNum_{0};

    int64_t ubFactorColBlockNum_{0};
    int64_t ubFactorColNum_{0};
    int64_t ubFactorRowNum_{0};
    int64_t ubFactorColLoopNum_{0};
    int64_t ubFactorRowLoopNum_{0};
    int64_t ubFactorColNormalBlockNum_{0};
    int64_t ubFactorColTailBlockNum_{0};
    int64_t ubFactorColNormalLen_{0};
    int64_t ubFactorColTailLen_{0};
    int64_t ubFactorRowNormalNum_{0};
    int64_t ubFactorRowTailNum_{0};
};

// ========================================================================
// Init
// ========================================================================
template <typename T, typename U>
__aicore__ inline void AntiMxQuantTailAxis<T, U>::Init(
    GM_ADDR x, GM_ADDR mxScale, GM_ADDR y, const AntiMxQuantTilingData* tilingData)
{
#if (__NPU_ARCH__ == 3510)
    SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    ParseTilingData(tilingData);
    GetGmParams();
    GetUbParams();

    int64_t xBufSize = 0;
    int64_t scaleBufSize = 0;
    int64_t yBufSize = 0;
    int64_t scaleComputeBufSize = 0;

    // maxUbBlockNum_ is in 1×32 block units (already multiplied by SPLIT_N/BLOCK_SIZE in tiling host).
    // Per 1×32 block: BLOCK_SIZE elements; one FP8_E8M0 scale byte per block.
    if constexpr (IsFp8Type<T>()) {
        xBufSize = maxUbBlockNum_ * BLOCK_SIZE * sizeof(uint8_t);
        // Scale buffer: 1 byte per 1×32 block, 32-byte aligned
        scaleBufSize = ((maxUbBlockNum_ + UBBlockSize_ - 1) / UBBlockSize_) * UBBlockSize_ * sizeof(uint8_t);
        yBufSize = maxUbBlockNum_ * BLOCK_SIZE * sizeof(U);
        // ComputeScale: each loop stores vfLen8 floats (256 * 4 = 1024 bytes)
        int64_t maxScaleNum = maxUbBlockNum_;
        int64_t scaleLoopNum = (maxScaleNum + vfLen8 - 1) / vfLen8;
        scaleComputeBufSize = scaleLoopNum * vfLen8 * sizeof(float);
    } else {
        xBufSize = maxUbBlockNum_ * BLOCK_SIZE / DIGIT_TWO;
        // Scale buffer: 1 byte per 1×32 block, 32-byte aligned
        scaleBufSize = ((maxUbBlockNum_ + UBBlockSize_ - 1) / UBBlockSize_) * UBBlockSize_ * sizeof(uint8_t);
        yBufSize = maxUbBlockNum_ * BLOCK_SIZE * sizeof(U);
        // ComputeScale: each loop stores vfLen8 bfloat16s (256 * 2 = 512 bytes)
        int64_t maxScaleNum = maxUbBlockNum_;
        int64_t scaleLoopNum = (maxScaleNum + vfLen8 - 1) / vfLen8;
        scaleComputeBufSize = scaleLoopNum * vfLen8 * sizeof(bfloat16_t);
    }

    pipe_.InitBuffer(inQueueX_, DB_BUFFER, xBufSize);
    pipe_.InitBuffer(inQueueScale_, DB_BUFFER, scaleBufSize);
    pipe_.InitBuffer(outQueueY_, DB_BUFFER, yBufSize);
    pipe_.InitBuffer(scaleBuffer_, scaleComputeBufSize);

    if constexpr (IsFp8Type<T>()) {
        xGm_.SetGlobalBuffer((__gm__ uint8_t*)x + xGmOffset_);
        yGm_.SetGlobalBuffer((__gm__ U*)y + xGmOffset_);
    } else {
        xGm_.SetGlobalBuffer((__gm__ uint8_t*)x + xGmOffset_ / DIGIT_TWO);
        yGm_.SetGlobalBuffer((__gm__ U*)y + xGmOffset_);
    }
    mxScaleGm_.SetGlobalBuffer((__gm__ uint8_t*)mxScale + scaleGmOffset_);
}

// ========================================================================
// ParseTilingData / GetGmParams / GetUbParams
// ========================================================================
template <typename T, typename U>
__aicore__ inline void AntiMxQuantTailAxis<T, U>::ParseTilingData(
    const AntiMxQuantTilingData* tilingData)
{
    totalCoreNum_ = tilingData->totalCoreNum;
    usedCoreNum_ = tilingData->usedCoreNum;
    rowTileNum_ = tilingData->rowTileNum;
    colTileNum_ = tilingData->colTileNum;
    rowNum_ = tilingData->rowNum;
    colNum_ = tilingData->colNum;
    colNormalBlockNum_ = tilingData->colNormalBlockNum;
    colTailLen_ = tilingData->colTailLen;
    rowNormalBlockNum_ = tilingData->rowNormalBlockNum;
    rowTailLen_ = tilingData->rowTailLen;
    maxUbBlockNum_ = tilingData->maxUbBlockNum;
    dstType_ = tilingData->dstType;
}

template <typename T, typename U>
__aicore__ inline void AntiMxQuantTailAxis<T, U>::GetGmParams()
{
    coreIdx_ = GetBlockIdx();
    coreColIdx_ = coreIdx_ % colTileNum_;
    coreRowIdx_ = coreIdx_ / colTileNum_;
    xGmOffset_ = coreRowIdx_ * rowNormalBlockNum_ * colNum_ +
                 coreColIdx_ * colNormalBlockNum_ * SPLIT_N;
    scaleColNum_ = (((((colNum_ + BLOCK_SIZE - 1) / BLOCK_SIZE) + DIGIT_TWO - 1) / DIGIT_TWO) * DIGIT_TWO);
    scaleGmOffset_ = coreRowIdx_ * rowNormalBlockNum_ * scaleColNum_ +
                     coreColIdx_ * colNormalBlockNum_ * DIGIT_SIXTEEN;
}

template <typename T, typename U>
__aicore__ inline void AntiMxQuantTailAxis<T, U>::GetUbParams()
{
    // Column factor is in 1×32 block units (intra-core unit).
    // Inter-core split uses 1×512 blocks (SPLIT_N), so each 1×512 contains
    // SPLIT_N / BLOCK_SIZE = DIGIT_SIXTEEN 1×32 blocks.
    if (coreColIdx_ == colTileNum_ - 1) {
        ubFactorColBlockNum_ = (colTailLen_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
        ubFactorColNum_ = colTailLen_;
    } else {
        ubFactorColBlockNum_ = colNormalBlockNum_ * DIGIT_SIXTEEN;
        ubFactorColNum_ = ubFactorColBlockNum_ * BLOCK_SIZE;
    }

    if (coreRowIdx_ == rowTileNum_ - 1) {
        ubFactorRowNum_ = rowTailLen_;
    } else {
        ubFactorRowNum_ = rowNormalBlockNum_;
    }

    // maxUbBlockNum_ is in 1×32 block units (already multiplied by 16 in tiling host)
    ubFactorColLoopNum_ = (ubFactorColBlockNum_ + maxUbBlockNum_ - 1) / maxUbBlockNum_;
    ubFactorColNormalBlockNum_ = (ubFactorColBlockNum_ + ubFactorColLoopNum_ - 1) / ubFactorColLoopNum_;
    // Even-align so the normal column chunk fits the FP8/scale 16-bit-pair processing layout.
    ubFactorColNormalBlockNum_ = ((ubFactorColNormalBlockNum_ + DIGIT_TWO - 1) / DIGIT_TWO) * DIGIT_TWO;
    ubFactorColTailBlockNum_ = ubFactorColBlockNum_ -
                                (ubFactorColLoopNum_ - DIGIT_ONE) * ubFactorColNormalBlockNum_;
    ubFactorColNormalLen_ = ubFactorColNormalBlockNum_ * BLOCK_SIZE;
    ubFactorColTailLen_ = ubFactorColNum_ -
                          (ubFactorColLoopNum_ - DIGIT_ONE) * ubFactorColNormalLen_;

    ubFactorRowNormalNum_ = maxUbBlockNum_ / ubFactorColNormalBlockNum_;
    ubFactorRowLoopNum_ = (ubFactorRowNum_ + ubFactorRowNormalNum_ - 1) / ubFactorRowNormalNum_;
    ubFactorRowNormalNum_ = (ubFactorRowNum_ + ubFactorRowLoopNum_ - 1) / ubFactorRowLoopNum_;
    ubFactorRowTailNum_ = ubFactorRowNum_ -
                          (ubFactorRowLoopNum_ - DIGIT_ONE) * ubFactorRowNormalNum_;
}

// ========================================================================
// Process
// ========================================================================
template <typename T, typename U>
__aicore__ inline void AntiMxQuantTailAxis<T, U>::Process()
{
    if (coreIdx_ >= usedCoreNum_) {
        return;
    }

    int64_t rowLoopIdx = 0;
    int64_t colLoopIdx = 0;

    for (rowLoopIdx = 0; rowLoopIdx < ubFactorRowLoopNum_ - 1; rowLoopIdx++) {
        for (colLoopIdx = 0; colLoopIdx < ubFactorColLoopNum_ - 1; colLoopIdx++) {
            CopyIn(rowLoopIdx, colLoopIdx, ubFactorRowNormalNum_, ubFactorColNormalLen_, ubFactorColNormalBlockNum_);
            Compute(ubFactorRowNormalNum_, ubFactorColNormalBlockNum_);
            CopyOut(rowLoopIdx, colLoopIdx, ubFactorRowNormalNum_, ubFactorColNormalLen_, ubFactorColNormalBlockNum_);
        }
        CopyIn(rowLoopIdx, colLoopIdx, ubFactorRowNormalNum_, ubFactorColTailLen_, ubFactorColTailBlockNum_);
        Compute(ubFactorRowNormalNum_, ubFactorColTailBlockNum_);
        CopyOut(rowLoopIdx, colLoopIdx, ubFactorRowNormalNum_, ubFactorColTailLen_, ubFactorColTailBlockNum_);
    }

    for (colLoopIdx = 0; colLoopIdx < ubFactorColLoopNum_ - 1; colLoopIdx++) {
        CopyIn(rowLoopIdx, colLoopIdx, ubFactorRowTailNum_, ubFactorColNormalLen_, ubFactorColNormalBlockNum_);
        Compute(ubFactorRowTailNum_, ubFactorColNormalBlockNum_);
        CopyOut(rowLoopIdx, colLoopIdx, ubFactorRowTailNum_, ubFactorColNormalLen_, ubFactorColNormalBlockNum_);
    }
    CopyIn(rowLoopIdx, colLoopIdx, ubFactorRowTailNum_, ubFactorColTailLen_, ubFactorColTailBlockNum_);
    Compute(ubFactorRowTailNum_, ubFactorColTailBlockNum_);
    CopyOut(rowLoopIdx, colLoopIdx, ubFactorRowTailNum_, ubFactorColTailLen_, ubFactorColTailBlockNum_);
}

// ========================================================================
// CopyIn
// ========================================================================
template <typename T, typename U>
__aicore__ inline void AntiMxQuantTailAxis<T, U>::CopyIn(
    int64_t rowLoopIdx, int64_t colLoopIdx, int64_t ubFactorRowNum, int64_t ubFactorColNum, int64_t ubFactorColBlockNum)
{
    LocalTensor<uint8_t> xLocal = inQueueX_.AllocTensor<uint8_t>();
    int64_t xOffset = rowLoopIdx * ubFactorRowNormalNum_ * colNum_ +
                      colLoopIdx * ubFactorColNormalLen_;
    if constexpr (IsFp4Type<T>()) {
        xOffset /= DIGIT_TWO;
    }

    if constexpr (IsFp8Type<T>()) {
        int64_t scaleIsNotOdd = ubFactorColBlockNum % DIGIT_TWO;
        int64_t xPad = ((ubFactorColNum + UBBlockSize_ - 1) / UBBlockSize_) * UBBlockSize_ - ubFactorColNum;
        DataCopyExtParams copyInParamX = {
            static_cast<uint16_t>(ubFactorRowNum),
            static_cast<uint32_t>(ubFactorColNum * sizeof(uint8_t)),
            static_cast<uint32_t>((colNum_ - ubFactorColNum) * sizeof(uint8_t)),
            static_cast<uint32_t>(scaleIsNotOdd), 0};
        DataCopyPadExtParams<uint8_t> xPadParams{true, 0, static_cast<uint8_t>(xPad), 0};
        DataCopyPad(xLocal, xGm_[xOffset], copyInParamX, xPadParams);
    } else {
        int64_t xBytes = ubFactorColNum / 2;
        int64_t xPadBytes = ((xBytes + UBBlockSize_ - 1) / UBBlockSize_) * UBBlockSize_ - xBytes;
        DataCopyExtParams copyInParamX = {
            static_cast<uint16_t>(ubFactorRowNum),
            static_cast<uint32_t>(xBytes),
            static_cast<uint32_t>((colNum_ - ubFactorColNum) / 2),
            0, 0};
        DataCopyPadExtParams<uint8_t> xPadParams{true, 0, static_cast<uint8_t>(xPadBytes), 0};
        DataCopyPad(xLocal, xGm_[xOffset], copyInParamX, xPadParams);
    }
    inQueueX_.EnQue(xLocal);

    LocalTensor<uint8_t> scaleLocal = inQueueScale_.AllocTensor<uint8_t>();
    // 1 FP8_E8M0 scale byte per 1×32 block; ubFactorColBlockNum is in 1×32 units.
    int64_t scaleNum = (ubFactorColBlockNum + DIGIT_TWO - 1) / DIGIT_TWO * DIGIT_TWO;
    int64_t scaleOffset = rowLoopIdx * ubFactorRowNormalNum_ * scaleColNum_ +
                          colLoopIdx * ubFactorColNormalBlockNum_;

    DataCopyExtParams copyInParamScale = {
        static_cast<uint16_t>(ubFactorRowNum),
        static_cast<uint32_t>(scaleNum),
        static_cast<uint32_t>((scaleColNum_ - scaleNum)),
        0, 0};
    DataCopyPadExtParams<uint8_t> scalePadParams{false, 0, 0, 0};
    DataCopyPad<uint8_t, PaddingMode::Compact>(scaleLocal, mxScaleGm_[scaleOffset], copyInParamScale, scalePadParams);
    inQueueScale_.EnQue(scaleLocal);
}

// ========================================================================
// CopyOut
// ========================================================================
template <typename T, typename U>
__aicore__ inline void AntiMxQuantTailAxis<T, U>::CopyOut(
    int64_t rowLoopIdx, int64_t colLoopIdx, int64_t ubFactorRowNum, int64_t ubFactorColNum, int64_t ubFactorColBlockNum)
{
    int64_t scaleIsNotOdd = ubFactorColBlockNum % DIGIT_TWO;

    int64_t anotherStride = 
        ((ubFactorColNum + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE - ubFactorColNum) * sizeof(U) / UBBlockSize_;
    int64_t srcStride = scaleIsNotOdd * sizeof(U) + anotherStride;          // scaleIsNotOdd为1时，表示输入 X 需要补一个BlockSize 32，搬出时不需要搬出这个补的block，对应的UBBlock数量就是scaleIsNotOdd * sizeof(U)；
                                                                            // anotherStride 指的是尾block块内的元素，多补到BlockSize的元素不应该搬出，这一部分需要跳过

    LocalTensor<U> yLocal = outQueueY_.DeQue<U>();
    int64_t yOffset = rowLoopIdx * ubFactorRowNormalNum_ * colNum_ +
                      colLoopIdx * ubFactorColNormalLen_;

    DataCopyExtParams copyOutParamY = {
        static_cast<uint16_t>(ubFactorRowNum),
        static_cast<uint32_t>(ubFactorColNum * sizeof(U)),
        static_cast<uint32_t>(srcStride),   // FP8 非偶数Block在数据搬入时会跳一个Block(32个数)，反量化为B16或者B32分别对应2*32字节或者4*32字节
                                            // FP4 在搬入时一个UBBlock是64个数对齐，天然偶数对齐，所以搬入无需跳Block，但是搬出时，非对齐场景，由于量化
                                            // 会有字节膨胀，ub存在多余的 0 不能搬出，这时候需要跳跃这些 0；
        static_cast<uint32_t>((colNum_ - ubFactorColNum) * sizeof(U)), 0};
    DataCopyPad(yGm_[yOffset], yLocal, copyOutParamY);
    outQueueY_.FreeTensor(yLocal);
}

// ========================================================================
// Compute (main framework)
// ========================================================================
template <typename T, typename U>
__aicore__ inline void AntiMxQuantTailAxis<T, U>::Compute(
    int64_t rowBlockNum, int64_t colBlockNum)
{
    colBlockNum += colBlockNum % DIGIT_TWO;
    // rowBlockNum / colBlockNum are in 1×32 block units (intra-core unit).
    // One FP8_E8M0 scale byte per 1×32 block.
    int64_t totalScaleNum = rowBlockNum * colBlockNum;
    uint32_t totalBlockNum = static_cast<uint32_t>(rowBlockNum * colBlockNum);
    // ComputeData (FP8 & FP4) processes DIGIT_SIXTEEN 1×32 blocks per VF loop,
    // i.e. one 1×512 outer block (512 elements).
    uint16_t loopNum2VF = static_cast<uint16_t>(
        (static_cast<int64_t>(totalBlockNum) + static_cast<int64_t>(DIGIT_SIXTEEN) - 1) /
        static_cast<int64_t>(DIGIT_SIXTEEN));

    LocalTensor<uint8_t> xLocal = inQueueX_.DeQue<uint8_t>();
    LocalTensor<uint8_t> scaleLocal = inQueueScale_.DeQue<uint8_t>();
    LocalTensor<U> yLocal = outQueueY_.AllocTensor<U>();

    auto xLocalAddr = reinterpret_cast<__ubuf__ uint8_t*>(xLocal.GetPhyAddr());
    auto yLocalAddr = reinterpret_cast<__ubuf__ U*>(yLocal.GetPhyAddr());
    auto scaleLocalAddr = reinterpret_cast<__ubuf__ uint8_t*>(scaleLocal.GetPhyAddr());

    if constexpr (IsFp8Type<T>()) {
        auto scaleBufAddr = reinterpret_cast<__ubuf__ float*>(scaleBuffer_.Get<float>().GetPhyAddr());
        ComputeScale(scaleLocalAddr, scaleBufAddr, totalScaleNum);
        ComputeData(xLocalAddr, scaleBufAddr, yLocalAddr, loopNum2VF);
    } else {
        auto scaleBufAddr = reinterpret_cast<__ubuf__ bfloat16_t*>(scaleBuffer_.Get<bfloat16_t>().GetPhyAddr());
        ComputeScale(scaleLocalAddr, scaleBufAddr, totalScaleNum);
        ComputeData(xLocalAddr, scaleBufAddr, yLocalAddr, loopNum2VF);
    }

    inQueueX_.FreeTensor(xLocal);
    inQueueScale_.FreeTensor(scaleLocal);
    outQueueY_.EnQue(yLocal);
}

// ========================================================================
// ComputeScale (FP8 input: FP8_E8M0 -> BF16 -> FP32)
// ========================================================================
template <typename T, typename U>
__aicore__ inline void AntiMxQuantTailAxis<T, U>::ComputeScale(
    __ubuf__ uint8_t* scaleLocalAddr, __ubuf__ float* scaleBufAddr, int64_t scaleNum)
{
    uint16_t loopNum = static_cast<uint16_t>((static_cast<int64_t>(scaleNum) + static_cast<int64_t>(vfLen8) - 1) / static_cast<int64_t>(vfLen8));
    __VEC_SCOPE__
    {
        Reg::MaskReg scaleMask = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg storeMask = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::RegTensor<uint8_t> vdScale0, vdScale1, vdu8_zero;
        Reg::RegTensor<bfloat16_t> vdScaleBf16_0, vdScaleBf16_1;
        Reg::RegTensor<float> vdScaleFp32_0_0, vdScaleFp32_0_1;
        Reg::RegTensor<float> vdScaleFp32_1_0, vdScaleFp32_1_1;

        Reg::Duplicate(vdu8_zero, 0);

        for (uint16_t i = 0; i < loopNum; i++) {
            Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                vdScale0, scaleLocalAddr, vfLen8);
            Reg::Interleave(vdScale0, vdScale1, vdScale0, vdu8_zero);

            // Step 1: 256 FP8_E8M0 -> 2 BF16 registers (ZERO/ONE layout)
            Reg::Cast<bfloat16_t, fp8_e8m0_t, castTraitFp8E8M0ToBf16_0>(
                vdScaleBf16_0, (Reg::RegTensor<fp8_e8m0_t>&)vdScale0, scaleMask);
            Reg::Cast<bfloat16_t, fp8_e8m0_t, castTraitFp8E8M0ToBf16_0>(
                vdScaleBf16_1, (Reg::RegTensor<fp8_e8m0_t>&)vdScale1, scaleMask);

            // Step 2: BF16 -> FP32 (each 128-bf16 produces 2x64-fp32)
            Reg::Cast<float, bfloat16_t, castTraitBf16ToFp32_0>(
                vdScaleFp32_0_0, vdScaleBf16_0, scaleMask);
            Reg::Cast<float, bfloat16_t, castTraitBf16ToFp32_1>(
                vdScaleFp32_0_1, vdScaleBf16_0, scaleMask);
            Reg::Cast<float, bfloat16_t, castTraitBf16ToFp32_0>(
                vdScaleFp32_1_0, vdScaleBf16_1, scaleMask);
            Reg::Cast<float, bfloat16_t, castTraitBf16ToFp32_1>(
                vdScaleFp32_1_1, vdScaleBf16_1, scaleMask);

            // Step 3: Interleave to restore FP32 order within each pair
            Reg::Interleave(vdScaleFp32_0_0, vdScaleFp32_0_1, vdScaleFp32_0_0, vdScaleFp32_0_1);
            Reg::Interleave(vdScaleFp32_1_0, vdScaleFp32_1_1, vdScaleFp32_1_0, vdScaleFp32_1_1);

            // Step 4: Store 4 FP32 registers to UB (order restored)
            Reg::StoreAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(
                scaleBufAddr, vdScaleFp32_0_0, vfLen32, storeMask);
            Reg::StoreAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(
                scaleBufAddr, vdScaleFp32_0_1, vfLen32, storeMask);
            Reg::StoreAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(
                scaleBufAddr, vdScaleFp32_1_0, vfLen32, storeMask);
            Reg::StoreAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(
                scaleBufAddr, vdScaleFp32_1_1, vfLen32, storeMask);
        }
    }
}

// ========================================================================
// ComputeScale (FP4 input: FP8_E8M0 -> BF16)
// ========================================================================
template <typename T, typename U>
__aicore__ inline void AntiMxQuantTailAxis<T, U>::ComputeScale(
    __ubuf__ uint8_t* scaleLocalAddr, __ubuf__ bfloat16_t* scaleBufAddr, int64_t scaleNum)
{
    uint16_t loopNum = static_cast<uint16_t>((static_cast<int64_t>(scaleNum) + static_cast<int64_t>(vfLen8) - 1) / static_cast<int64_t>(vfLen8));
    __VEC_SCOPE__
    {
        Reg::MaskReg scaleMask = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg storeMask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::RegTensor<uint8_t> vdScale0, vdScale1, vdu8_zero;
        Reg::RegTensor<bfloat16_t> vdScaleBf16_0, vdScaleBf16_1;

        Reg::Duplicate(vdu8_zero, 0);

        for (uint16_t i = 0; i < loopNum; i++) {
            Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                vdScale0, scaleLocalAddr, vfLen8);
            Reg::Interleave(vdScale0, vdScale1, vdScale0, vdu8_zero);

            // Step 1: 256 FP8_E8M0 -> 2 BF16 registers (ZERO/ONE layout)
            Reg::Cast<bfloat16_t, fp8_e8m0_t, castTraitFp8E8M0ToBf16_0>(
                vdScaleBf16_0, (Reg::RegTensor<fp8_e8m0_t>&)vdScale0, scaleMask);
            Reg::Cast<bfloat16_t, fp8_e8m0_t, castTraitFp8E8M0ToBf16_0>(
                vdScaleBf16_1, (Reg::RegTensor<fp8_e8m0_t>&)vdScale1, scaleMask);

            // Step 2: Store 2 BF16 registers to UB (order restored)
            Reg::StoreAlign<bfloat16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                scaleBufAddr, vdScaleBf16_0, vfLen16, storeMask);
            Reg::StoreAlign<bfloat16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                scaleBufAddr, vdScaleBf16_1, vfLen16, storeMask);
        }
    }
}

// ========================================================================
// ComputeData (FP8 input)
// ========================================================================
template <typename T, typename U>
__aicore__ inline void AntiMxQuantTailAxis<T, U>::ComputeData(
    __ubuf__ uint8_t* xLocalAddr, __ubuf__ float* scaleBufAddr, __ubuf__ U* yLocalAddr, uint16_t loopNum2VF)
{
    __VEC_SCOPE__
    {
        Reg::MaskReg maskAll = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskFp32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskFp8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

        Reg::RegTensor<uint8_t> vdFp8_0, vdFp8_1;
        Reg::RegTensor<float> vdFp32_0_0, vdFp32_0_1, vdFp32_0_2, vdFp32_0_3;
        Reg::RegTensor<float> vdFp32_1_0, vdFp32_1_1, vdFp32_1_2, vdFp32_1_3;
        Reg::RegTensor<float> vdScale_0, vdScale_1;

        for (uint16_t i = 0; i < loopNum2VF; i++) {
            Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE,
                           Reg::LoadDist::DIST_DINTLV_B8>(
                vdFp8_0, vdFp8_1, xLocalAddr, vfLen8Double);

            Reg::Interleave(vdFp8_0, vdFp8_1, vdFp8_0, vdFp8_1);
            Reg::Cast<float, T, castTraitFp8ToFp32_0>(
                vdFp32_0_0, (Reg::RegTensor<T>&)vdFp8_0, maskFp8);
            Reg::Cast<float, T, castTraitFp8ToFp32_1>(
                vdFp32_0_1, (Reg::RegTensor<T>&)vdFp8_0, maskFp8);
            Reg::Cast<float, T, castTraitFp8ToFp32_2>(
                vdFp32_0_2, (Reg::RegTensor<T>&)vdFp8_0, maskFp8);
            Reg::Cast<float, T, castTraitFp8ToFp32_3>(
                vdFp32_0_3, (Reg::RegTensor<T>&)vdFp8_0, maskFp8);
            Reg::Cast<float, T, castTraitFp8ToFp32_0>(
                vdFp32_1_0, (Reg::RegTensor<T>&)vdFp8_1, maskFp8);
            Reg::Cast<float, T, castTraitFp8ToFp32_1>(
                vdFp32_1_1, (Reg::RegTensor<T>&)vdFp8_1, maskFp8);
            Reg::Cast<float, T, castTraitFp8ToFp32_2>(
                vdFp32_1_2, (Reg::RegTensor<T>&)vdFp8_1, maskFp8);
            Reg::Cast<float, T, castTraitFp8ToFp32_3>(
                vdFp32_1_3, (Reg::RegTensor<T>&)vdFp8_1, maskFp8);

            Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE,
                           Reg::LoadDist::DIST_E2B_B32>(
                vdScale_0, scaleBufAddr, elementAfterReduce_);
            Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE,
                           Reg::LoadDist::DIST_E2B_B32>(
                vdScale_1, scaleBufAddr, elementAfterReduce_);

            Reg::Mul(vdFp32_0_0, vdFp32_0_0, vdScale_0, maskFp32);
            Reg::Mul(vdFp32_0_1, vdFp32_0_1, vdScale_0, maskFp32);
            Reg::Mul(vdFp32_0_2, vdFp32_0_2, vdScale_0, maskFp32);
            Reg::Mul(vdFp32_0_3, vdFp32_0_3, vdScale_0, maskFp32);
            Reg::Mul(vdFp32_1_0, vdFp32_1_0, vdScale_1, maskFp32);
            Reg::Mul(vdFp32_1_1, vdFp32_1_1, vdScale_1, maskFp32);
            Reg::Mul(vdFp32_1_2, vdFp32_1_2, vdScale_1, maskFp32);
            Reg::Mul(vdFp32_1_3, vdFp32_1_3, vdScale_1, maskFp32);

            // First-level Interleave: cross-pair (0+2, 1+3) to interleave
            // even/odd layouts. After this:
            //   _0_0: [val0,val2,val4,...,val126]   _0_1: [val1,val3,...,val127]
            //   _0_2: [val128,val130,...,val254]    _0_3: [val129,...,val255]
            //   _1_0: [val256,val258,...,val382]    _1_1: [val257,...,val383]
            //   _1_2: [val384,val386,...,val510]    _1_3: [val385,...,val511]
            Reg::Interleave(vdFp32_0_0, vdFp32_0_2, vdFp32_0_0, vdFp32_0_2);
            Reg::Interleave(vdFp32_0_1, vdFp32_0_3, vdFp32_0_1, vdFp32_0_3);
            Reg::Interleave(vdFp32_1_0, vdFp32_1_2, vdFp32_1_0, vdFp32_1_2);
            Reg::Interleave(vdFp32_1_1, vdFp32_1_3, vdFp32_1_1, vdFp32_1_3);

            if constexpr (IsFp32Type<U>()) {
                // Second-level Interleave (float granularity):
                // Merge _0_0 with _0_1 to get full sequential order.
                // After this: _0_0=[0..63], _0_1=[64..127], _0_2=[128..191], etc.
                Reg::Interleave(vdFp32_0_0, vdFp32_0_1, vdFp32_0_0, vdFp32_0_1);
                Reg::Interleave(vdFp32_0_2, vdFp32_0_3, vdFp32_0_2, vdFp32_0_3);
                Reg::Interleave(vdFp32_1_0, vdFp32_1_1, vdFp32_1_0, vdFp32_1_1);
                Reg::Interleave(vdFp32_1_2, vdFp32_1_3, vdFp32_1_2, vdFp32_1_3);

                Reg::StoreAlign<float, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::StoreDist::DIST_NORM_B32>(
                    yLocalAddr, vdFp32_0_0, vfLen32, maskFp32);
                Reg::StoreAlign<float, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::StoreDist::DIST_NORM_B32>(
                    yLocalAddr, vdFp32_0_1, vfLen32, maskFp32);
                Reg::StoreAlign<float, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::StoreDist::DIST_NORM_B32>(
                    yLocalAddr, vdFp32_0_2, vfLen32, maskFp32);
                Reg::StoreAlign<float, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::StoreDist::DIST_NORM_B32>(
                    yLocalAddr, vdFp32_0_3, vfLen32, maskFp32);
                Reg::StoreAlign<float, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::StoreDist::DIST_NORM_B32>(
                    yLocalAddr, vdFp32_1_0, vfLen32, maskFp32);
                Reg::StoreAlign<float, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::StoreDist::DIST_NORM_B32>(
                    yLocalAddr, vdFp32_1_1, vfLen32, maskFp32);
                Reg::StoreAlign<float, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::StoreDist::DIST_NORM_B32>(
                    yLocalAddr, vdFp32_1_2, vfLen32, maskFp32);
                Reg::StoreAlign<float, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::StoreDist::DIST_NORM_B32>(
                    yLocalAddr, vdFp32_1_3, vfLen32, maskFp32);
            } else if constexpr (IsBf16Type<U>()) {
                // After cross-pair Interleave:
                // _0_0 holds even-indexed values (0,2,4...), _0_1 holds odd-indexed (1,3,5...).
                // Use independent register pairs for each Cast+Add group to maximize ILP.
                Reg::RegTensor<bfloat16_t> vdBf16_0_z, vdBf16_0_o;
                Reg::RegTensor<bfloat16_t> vdBf16_1_z, vdBf16_1_o;
                Reg::RegTensor<bfloat16_t> vdBf16_2_z, vdBf16_2_o;
                Reg::RegTensor<bfloat16_t> vdBf16_3_z, vdBf16_3_o;

                // _0_0(ZERO) + _0_1(ONE) -> sequential bf16[0..127]
                Reg::Cast<bfloat16_t, float, castTraitFp32ToBf16_0>(vdBf16_0_z, vdFp32_0_0, maskAll);
                Reg::Cast<bfloat16_t, float, castTraitFp32ToBf16_1>(vdBf16_0_o, vdFp32_0_1, maskAll);
                Reg::Add(vdBf16_0_z, vdBf16_0_z, vdBf16_0_o, maskAll);
                Reg::StoreAlign<bfloat16_t, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::StoreDist::DIST_NORM_B16>(
                    yLocalAddr, vdBf16_0_z, vfLen16, maskAll);

                // _0_2(ZERO) + _0_3(ONE) -> sequential bf16[128..255]
                Reg::Cast<bfloat16_t, float, castTraitFp32ToBf16_0>(vdBf16_1_z, vdFp32_0_2, maskAll);
                Reg::Cast<bfloat16_t, float, castTraitFp32ToBf16_1>(vdBf16_1_o, vdFp32_0_3, maskAll);
                Reg::Add(vdBf16_1_z, vdBf16_1_z, vdBf16_1_o, maskAll);
                Reg::StoreAlign<bfloat16_t, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::StoreDist::DIST_NORM_B16>(
                    yLocalAddr, vdBf16_1_z, vfLen16, maskAll);

                // _1_0(ZERO) + _1_1(ONE) -> sequential bf16[256..383]
                Reg::Cast<bfloat16_t, float, castTraitFp32ToBf16_0>(vdBf16_2_z, vdFp32_1_0, maskAll);
                Reg::Cast<bfloat16_t, float, castTraitFp32ToBf16_1>(vdBf16_2_o, vdFp32_1_1, maskAll);
                Reg::Add(vdBf16_2_z, vdBf16_2_z, vdBf16_2_o, maskAll);
                Reg::StoreAlign<bfloat16_t, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::StoreDist::DIST_NORM_B16>(
                    yLocalAddr, vdBf16_2_z, vfLen16, maskAll);

                // _1_2(ZERO) + _1_3(ONE) -> sequential bf16[384..511]
                Reg::Cast<bfloat16_t, float, castTraitFp32ToBf16_0>(vdBf16_3_z, vdFp32_1_2, maskAll);
                Reg::Cast<bfloat16_t, float, castTraitFp32ToBf16_1>(vdBf16_3_o, vdFp32_1_3, maskAll);
                Reg::Add(vdBf16_3_z, vdBf16_3_z, vdBf16_3_o, maskAll);
                Reg::StoreAlign<bfloat16_t, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::StoreDist::DIST_NORM_B16>(
                    yLocalAddr, vdBf16_3_z, vfLen16, maskAll);
            } else {
                // FP32 -> FP16: same pattern as BF16
                Reg::RegTensor<half> vdFp16_0_z, vdFp16_0_o;
                Reg::RegTensor<half> vdFp16_1_z, vdFp16_1_o;
                Reg::RegTensor<half> vdFp16_2_z, vdFp16_2_o;
                Reg::RegTensor<half> vdFp16_3_z, vdFp16_3_o;

                Reg::Cast<half, float, castTraitFp32ToFp16_0>(vdFp16_0_z, vdFp32_0_0, maskAll);
                Reg::Cast<half, float, castTraitFp32ToFp16_1>(vdFp16_0_o, vdFp32_0_1, maskAll);
                Reg::Add(vdFp16_0_z, vdFp16_0_z, vdFp16_0_o, maskAll);
                Reg::StoreAlign<half, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::StoreDist::DIST_NORM_B16>(
                    yLocalAddr, vdFp16_0_z, vfLen16, maskAll);

                Reg::Cast<half, float, castTraitFp32ToFp16_0>(vdFp16_1_z, vdFp32_0_2, maskAll);
                Reg::Cast<half, float, castTraitFp32ToFp16_1>(vdFp16_1_o, vdFp32_0_3, maskAll);
                Reg::Add(vdFp16_1_z, vdFp16_1_z, vdFp16_1_o, maskAll);
                Reg::StoreAlign<half, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::StoreDist::DIST_NORM_B16>(
                    yLocalAddr, vdFp16_1_z, vfLen16, maskAll);

                Reg::Cast<half, float, castTraitFp32ToFp16_0>(vdFp16_2_z, vdFp32_1_0, maskAll);
                Reg::Cast<half, float, castTraitFp32ToFp16_1>(vdFp16_2_o, vdFp32_1_1, maskAll);
                Reg::Add(vdFp16_2_z, vdFp16_2_z, vdFp16_2_o, maskAll);
                Reg::StoreAlign<half, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::StoreDist::DIST_NORM_B16>(
                    yLocalAddr, vdFp16_2_z, vfLen16, maskAll);

                Reg::Cast<half, float, castTraitFp32ToFp16_0>(vdFp16_3_z, vdFp32_1_2, maskAll);
                Reg::Cast<half, float, castTraitFp32ToFp16_1>(vdFp16_3_o, vdFp32_1_3, maskAll);
                Reg::Add(vdFp16_3_z, vdFp16_3_z, vdFp16_3_o, maskAll);
                Reg::StoreAlign<half, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::StoreDist::DIST_NORM_B16>(
                    yLocalAddr, vdFp16_3_z, vfLen16, maskAll);
            }
        }
    }
}

// ========================================================================
// ComputeData (FP4 input)
// ========================================================================
template <typename T, typename U>
__aicore__ inline void AntiMxQuantTailAxis<T, U>::ComputeData(
    __ubuf__ uint8_t* xLocalAddr, __ubuf__ bfloat16_t* scaleBufAddr, __ubuf__ U* yLocalAddr, uint16_t loopNum2VF)
{
    __VEC_SCOPE__
    {
        Reg::MaskReg maskAll16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskU8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

        // uint8 registers for FP4 raw data load and interleave
        Reg::RegTensor<uint8_t> vdFp4U8_0, vdFp4U8_1, vdFp4U8_2, vdFp4U8_3;

        // BF16 registers: low 4bit cast, high 4bit cast, and merged results
        Reg::RegTensor<bfloat16_t> vdMerged0, vdMerged1, vdMerged2, vdMerged3;
        Reg::RegTensor<bfloat16_t> vdScale0, vdScale1, vdScale2, vdScale3;
        Reg::RegTensor<bfloat16_t> vdScaleTmp0, vdScaleTmp1;

        for (uint16_t i = 0; i < loopNum2VF; i++) {
            // Load 256 bytes of FP4 raw data
            Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK4_B8>(
                vdFp4U8_0, xLocalAddr, vfLen32);        // 01 00 00 00 23 00 00 00 45 00 00 00 ......
            Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK4_B8>(
                vdFp4U8_1, xLocalAddr, vfLen32);        // 128129 00 00 00 130131 00 00 00 ......
            Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK4_B8>(
                vdFp4U8_2, xLocalAddr, vfLen32);        // 256257 00 00 00 258259 00 00 00 ......
            Reg::LoadAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK4_B8>(
                vdFp4U8_3, xLocalAddr, vfLen32);        // 384385 00 00 00 386387 00 00 00 ......

            Reg::Cast<bfloat16_t, T, castTraitFp4ToBf16>(
                vdMerged0, (Reg::RegTensor<T>&)vdFp4U8_0, maskU8);  // 0 - 127
            Reg::Cast<bfloat16_t, T, castTraitFp4ToBf16>(
                vdMerged1, (Reg::RegTensor<T>&)vdFp4U8_1, maskU8);  // 128 - 255
            Reg::Cast<bfloat16_t, T, castTraitFp4ToBf16>(
                vdMerged2, (Reg::RegTensor<T>&)vdFp4U8_2, maskU8);  // 256 - 383
            Reg::Cast<bfloat16_t, T, castTraitFp4ToBf16>(
                vdMerged3, (Reg::RegTensor<T>&)vdFp4U8_3, maskU8);  // 384 - 511

            // Step 6: Load scale (BF16) and multiply
            Reg::LoadAlign<bfloat16_t, Reg::PostLiteral::POST_MODE_UPDATE,
                           Reg::LoadDist::DIST_E2B_B16>(
                vdScaleTmp0, scaleBufAddr, elementAfterReduce_);
            Reg::LoadAlign<bfloat16_t, Reg::PostLiteral::POST_MODE_UPDATE,
                           Reg::LoadDist::DIST_E2B_B16>(
                vdScaleTmp1, scaleBufAddr, elementAfterReduce_);

            Reg::Interleave(vdScale0, vdScale1, vdScaleTmp0, vdScaleTmp0);  // [0*16 1*16 ... 6*16 7*16] +  [0*16 1*16 ... 6*16 7*16] ->  [0*32 1*32 2*32 3*32] + [4*32 5*32 6*32 7*32]
            Reg::Interleave(vdScale2, vdScale3, vdScaleTmp1, vdScaleTmp1);  // 同上，但是处理的是第8-15个block的scale

            Reg::Mul(vdMerged0, vdMerged0, vdScale0, maskAll16); // 0-127
            Reg::Mul(vdMerged1, vdMerged1, vdScale1, maskAll16); // 128-255
            Reg::Mul(vdMerged2, vdMerged2, vdScale2, maskAll16); // 256-383
            Reg::Mul(vdMerged3, vdMerged3, vdScale3, maskAll16); // 384-511

            // Step 7: Output based on destination type
            if constexpr (IsFp32Type<U>()) {
                // 128 BF16 -> 2 FP32 registers per Cast (data expansion).
                // Use independent register pairs for each group to avoid
                // WAW dependencies and maximize ILP.
                Reg::RegTensor<float> vdFp32_0_0, vdFp32_0_1;
                Reg::RegTensor<float> vdFp32_1_0, vdFp32_1_1;
                Reg::RegTensor<float> vdFp32_2_0, vdFp32_2_1;
                Reg::RegTensor<float> vdFp32_3_0, vdFp32_3_1;

                // group 0: vdMerged0
                Reg::Cast<float, bfloat16_t, castTraitBf16ToFp32_0>(
                    vdFp32_0_0, vdMerged0, maskAll16);      
                Reg::Cast<float, bfloat16_t, castTraitBf16ToFp32_1>(
                    vdFp32_0_1, vdMerged0, maskAll16); 
                Reg::Interleave(vdFp32_0_0, vdFp32_0_1, vdFp32_0_0, vdFp32_0_1);
                Reg::StoreAlign<float, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::StoreDist::DIST_NORM_B32>(
                    yLocalAddr, vdFp32_0_0, vfLen32, maskAll16);
                Reg::StoreAlign<float, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::StoreDist::DIST_NORM_B32>(
                    yLocalAddr, vdFp32_0_1, vfLen32, maskAll16);

                // group 1: vdMerged1
                Reg::Cast<float, bfloat16_t, castTraitBf16ToFp32_0>(
                    vdFp32_1_0, vdMerged1, maskAll16);
                Reg::Cast<float, bfloat16_t, castTraitBf16ToFp32_1>(
                    vdFp32_1_1, vdMerged1, maskAll16);
                Reg::Interleave(vdFp32_1_0, vdFp32_1_1, vdFp32_1_0, vdFp32_1_1);
                Reg::StoreAlign<float, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::StoreDist::DIST_NORM_B32>(
                    yLocalAddr, vdFp32_1_0, vfLen32, maskAll16);
                Reg::StoreAlign<float, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::StoreDist::DIST_NORM_B32>(
                    yLocalAddr, vdFp32_1_1, vfLen32, maskAll16);

                // group 2: vdMerged2
                Reg::Cast<float, bfloat16_t, castTraitBf16ToFp32_0>(
                    vdFp32_2_0, vdMerged2, maskAll16);
                Reg::Cast<float, bfloat16_t, castTraitBf16ToFp32_1>(
                    vdFp32_2_1, vdMerged2, maskAll16);
                Reg::Interleave(vdFp32_2_0, vdFp32_2_1, vdFp32_2_0, vdFp32_2_1);
                Reg::StoreAlign<float, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::StoreDist::DIST_NORM_B32>(
                    yLocalAddr, vdFp32_2_0, vfLen32, maskAll16);
                Reg::StoreAlign<float, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::StoreDist::DIST_NORM_B32>(
                    yLocalAddr, vdFp32_2_1, vfLen32, maskAll16);

                // group 3: vdMerged3
                Reg::Cast<float, bfloat16_t, castTraitBf16ToFp32_0>(
                    vdFp32_3_0, vdMerged3, maskAll16);
                Reg::Cast<float, bfloat16_t, castTraitBf16ToFp32_1>(
                    vdFp32_3_1, vdMerged3, maskAll16);
                Reg::Interleave(vdFp32_3_0, vdFp32_3_1, vdFp32_3_0, vdFp32_3_1);
                Reg::StoreAlign<float, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::StoreDist::DIST_NORM_B32>(
                    yLocalAddr, vdFp32_3_0, vfLen32, maskAll16);
                Reg::StoreAlign<float, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::StoreDist::DIST_NORM_B32>(
                    yLocalAddr, vdFp32_3_1, vfLen32, maskAll16);
            } else if constexpr (IsBf16Type<U>()) {
                Reg::StoreAlign<bfloat16_t, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::StoreDist::DIST_NORM_B16>(
                    yLocalAddr, vdMerged0, vfLen16, maskAll16);
                Reg::StoreAlign<bfloat16_t, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::StoreDist::DIST_NORM_B16>(
                    yLocalAddr, vdMerged1, vfLen16, maskAll16);
                Reg::StoreAlign<bfloat16_t, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::StoreDist::DIST_NORM_B16>(
                    yLocalAddr, vdMerged2, vfLen16, maskAll16);
                Reg::StoreAlign<bfloat16_t, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::StoreDist::DIST_NORM_B16>(
                    yLocalAddr, vdMerged3, vfLen16, maskAll16);
            } else {
                // BF16 -> FP16: use independent registers for each group
                // to avoid WAW dependencies and maximize ILP.
                Reg::RegTensor<half> vdFp16_0, vdFp16_1;
                Reg::RegTensor<half> vdFp16_2, vdFp16_3;

                Reg::Cast<half, bfloat16_t, castTraitBf16ToFp16>(
                    vdFp16_0, vdMerged0, maskAll16);
                Reg::Cast<half, bfloat16_t, castTraitBf16ToFp16>(
                    vdFp16_1, vdMerged1, maskAll16);
                Reg::StoreAlign<half, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::StoreDist::DIST_NORM_B16>(
                    yLocalAddr, vdFp16_0, vfLen16, maskAll16);
                Reg::StoreAlign<half, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::StoreDist::DIST_NORM_B16>(
                    yLocalAddr, vdFp16_1, vfLen16, maskAll16);

                Reg::Cast<half, bfloat16_t, castTraitBf16ToFp16>(
                    vdFp16_2, vdMerged2, maskAll16);
                Reg::Cast<half, bfloat16_t, castTraitBf16ToFp16>(
                    vdFp16_3, vdMerged3, maskAll16);
                Reg::StoreAlign<half, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::StoreDist::DIST_NORM_B16>(
                    yLocalAddr, vdFp16_2, vfLen16, maskAll16);
                Reg::StoreAlign<half, Reg::PostLiteral::POST_MODE_UPDATE,
                                Reg::StoreDist::DIST_NORM_B16>(
                    yLocalAddr, vdFp16_3, vfLen16, maskAll16);
            }
        }
    }
}

} // namespace AntiMxQuant

#endif // ANTI_MX_QUANT_TAIL_AXIS_H
