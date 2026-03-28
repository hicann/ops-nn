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
 * \file dynamic_mx_quant_tail_axis.h
 * \brief
 */

#ifndef DYNAMIC_MX_QUANT_TAIL_AXIS_H
#define DYNAMIC_MX_QUANT_TAIL_AXIS_H
#define FLOAT_OVERFLOW_MODE_CTRL 60
#include "kernel_operator.h"
#include "dynamic_mx_quant_common.h"
namespace DynamicMxQuant {
using namespace AscendC;

template <typename T, typename U, int64_t SCALE_ALG>
class DynamicMxQuantTailAxis {
public:
    __aicore__ inline DynamicMxQuantTailAxis()
    {}
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, const DynamicMxQuantTailAxisTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const DynamicMxQuantTailAxisTilingData* tilingData);
    __aicore__ inline void GetGmParams();
    __aicore__ inline void GetUbParams();

    __aicore__ inline void CopyIn(int64_t dim0LoopIdx, int64_t dim1LoopIdx, int64_t ubFactorRowNum, int64_t ubFactorColNum, int64_t ubFactorColBlockNum);
    __aicore__ inline void Compute(int64_t ubFactorRowBlockNum, int64_t ubFactorColBlockNum);
    __aicore__ inline void CopyOut(int64_t dim0LoopIdx, int64_t dim1LoopIdx, int64_t ubFactorRowNum, int64_t ubFactorColNum, int64_t ubFactorColBlockNum);

    __aicore__ inline void ComputeMaxExp(__ubuf__ T* xLocalAddr, __ubuf__ uint16_t* maxExpAddr, uint16_t LoopNum2VF);
    __aicore__ inline void ComputeMaxExpDynamicDtypeRange(__ubuf__ T* xLocalAddr, __ubuf__ uint16_t* maxExpAddr, uint16_t LoopNum2VF);
    __aicore__ inline void ComputeMaxExpcuBLAS(__ubuf__ T* xLocalAddr, __ubuf__ uint16_t* maxExpAddr, uint16_t LoopNum2VF);
    __aicore__ inline void ComputeScale(__ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr, uint16_t LoopNum1VF, uint32_t totalScaleInUB);
    __aicore__ inline void ComputeScaleDynamicDtypeRange(__ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr, uint16_t LoopNum1VF, uint32_t totalScaleInUB);
    __aicore__ inline void ComputeScalecuBLASFP4(__ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr, uint16_t LoopNumHalfVF, uint32_t totalScaleInUB);
    template <RoundMode toBf16RoundMode, RoundMode roundMode>
    __aicore__ inline void ComputeData(__ubuf__ T* xLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr, __ubuf__ int8_t* yLocalAddr, uint16_t LoopNum2VF);
    template <RoundMode toBf16RoundMode, RoundMode roundMode>
    __aicore__ inline void ComputeDataOptimize(__ubuf__ T* xLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr, __ubuf__ int8_t* yLocalAddr, uint16_t LoopNum2VF);
    __aicore__ inline void FP16Convert(Reg::RegTensor<half>& output, Reg::RegTensor<half>& input, Reg::MaskReg& mask);
    template <RoundMode toBf16RoundMode, RoundMode roundMode>
    __aicore__ inline void ComputeFP4FromHalf(Reg::RegTensor<float>& Reg);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, DB_BUFFER> inQueue_;
    GlobalTensor<T> xGm_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueue_;
    GlobalTensor<uint8_t> yGm_;
    TQue<QuePosition::VECOUT, DB_BUFFER> mxScaleQueue_;
    GlobalTensor<uint8_t> scaleGm_;

    TBuf<QuePosition::VECCALC> maxExpBuffer_;
    TBuf<QuePosition::VECCALC> recipScaleBuffer_;

    int64_t roundMode_ = 0;
    int64_t blockSize_ = 0;
    int64_t totalCoreNum_ = 0;
    int64_t usedCoreNum_ = 0;

    int64_t rowTileNum_ = 0;                // row 方向上的切核数
    int64_t colTileNum_ = 0;                // col 方向上的切核数
    int64_t rowNum_ = 1;                    // 合轴之后 -2 轴大小
    int64_t colNum_ = 1;                    // 合轴之后 -1 轴大小
    int64_t colNormalBlockNum_ = 0;         // 列方向头核处理的块数 (1 x 256)
    int64_t colTailLen_ = 0;                // 列方向尾块长度 
    int64_t rowNormalBlockNum_ = 0;         // 行方向头核处理的块数 (1 行)
    int64_t rowTailLen_ = 0;                // 行方向尾块长度
    int64_t maxUbBlockNum_ = 0;             // UB最大能放下的处理块数 (1 x 32) (8 的倍数)
    float dstTypeMax_ = 0.0;
    float invDstTypeMax_ = 0.0;

    // GM Params
    int64_t coreIdx_ = 0;                       // Core ID
    int64_t coreColIdx_ = 0;                    // Core 处理的 GM 数据块列方向的核ID数
    int64_t coreRowIdx_ = 0;                    // Core 处理的 GM 数据块行方向的核ID数
    int64_t xGmOffset_ = 0;                     // Core 处理的数据块在 GM 的起始地址偏移
    int64_t scaleGmOffet_ = 0;                  // Core 处理的数据块得到的Scale在 GM 的地址偏移
    int64_t scaleColNum_ = 0;                   // 合轴后数据块在尾轴方向上的 scale 数量大小（偶数对齐）

    // UB Params
    int64_t ubFactorColNum_ = 0;                // Core 处理的 GM 数据块列方向元素个数
    int64_t ubFactorCol32BlockNum_ = 0;         // Core 处理的 GM 数据块列方向 32 长度块个数
    int64_t ubFactorRow1BlockNum_ = 0;          // Core 处理的 GM 数据块行方向 1 长度块个数
    int64_t ubFactorColLoopNum_ = 0;            // Core 处理的 GM 数据块列方向循环次数
    int64_t ubFactorRowLoopNum_ = 0;            // Core 处理的 GM 数据块行方向循环次数
    int64_t ubFactorColNormalBlockNum_ = 0;     // Core 处理的 GM 数据块搬入 UB 列方向 Normal 32 长度数据块
    int64_t ubFactorColTailBlockNum_ = 0;       // Core 处理的 GM 数据块搬入 UB 列方向 Tail 32 长度数据块
    int64_t ubFactorColNormalBlockLen_ = 0;     // Core 处理的 GM 数据块搬入 UB 列方向 Normal 数据块元素个数
    int64_t ubFactorColTailBlockLen_ = 0;       // Core 处理的 GM 数据块搬入 UB 列方向 Tail 数据块元素个数
    int64_t ubFactorRowNormalBlockNum_ = 0;     // Core 处理的 GM 数据块搬入 UB 行方向 Normal 1 长度数据块
    int64_t ubFactorRowTailBlockNum_ = 0;       // Core 处理的 GM 数据块搬入 UB 行方向 Tail 1 长度数据块

    uint16_t f4Emax_ = 0;
    uint16_t addValueBit = 0;
};

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, const DynamicMxQuantTailAxisTilingData* tilingData)
{
#if (__NPU_ARCH__ == 3510)
    SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    ParseTilingData(tilingData);                // 获取TilingData数据

    GetGmParams();                              // 计算核间GM地址偏移

    GetUbParams();                              // 计算核内切分参数

    pipe_.InitBuffer(inQueue_, DB_BUFFER, maxUbBlockNum_ * blockSize_ * sizeof(T));
    pipe_.InitBuffer(outQueue_, DB_BUFFER, maxUbBlockNum_ * blockSize_ * sizeof(uint8_t) / DIGIT_TWO);
    pipe_.InitBuffer(mxScaleQueue_, DB_BUFFER, maxUbBlockNum_ * sizeof(uint8_t));
    pipe_.InitBuffer(maxExpBuffer_, maxUbBlockNum_ * sizeof(uint16_t));
    pipe_.InitBuffer(recipScaleBuffer_, maxUbBlockNum_ * sizeof(uint16_t));

    xGm_.SetGlobalBuffer((__gm__ T*)x + xGmOffset_);
    yGm_.SetGlobalBuffer((__gm__ uint8_t*)y + xGmOffset_ / DIGIT_TWO);
    scaleGm_.SetGlobalBuffer((__gm__ uint8_t*)mxScale + scaleGmOffet_);

    if constexpr (IsSame<U, fp4x2_e2m1_t>::value) {
        f4Emax_ = FP4_E2M1_BF16_MAX_EXP;
    } else {
        f4Emax_ = FP4_E1M2_MAX_EXP;
    }

    if (dstTypeMax_ == DIGIT_ZERO_FLOAT || dstTypeMax_ == DIGIT_SIX_FLOAT) {
        addValueBit = ADD_VALUE_FOR_BF16_MAN1;
    } else if (dstTypeMax_ == DIGIT_SEVEN_FLOAT) {
        addValueBit = ADD_VALUE_FOR_BF16_MAN2;
    }
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::GetGmParams(){
    coreIdx_ = GetBlockIdx();
    coreColIdx_ = coreIdx_ % colTileNum_;
    coreRowIdx_ = coreIdx_ / colTileNum_;
    xGmOffset_ = coreRowIdx_ * rowNormalBlockNum_ * DIGIT_ONE * colNum_ + coreColIdx_ * colNormalBlockNum_ * DIGIT_EIGHT * blockSize_;
    scaleColNum_ = ops::CeilDiv(ops::CeilDiv(colNum_, blockSize_), DIGIT_TWO) * DIGIT_TWO;
    scaleGmOffet_ = coreRowIdx_ * rowNormalBlockNum_ * DIGIT_ONE * scaleColNum_ + coreColIdx_ * colNormalBlockNum_ * DIGIT_EIGHT;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::GetUbParams(){
    if (coreColIdx_ == colTileNum_ - 1) {
        ubFactorCol32BlockNum_ =  ops::CeilDiv(colTailLen_, blockSize_);
        ubFactorColNum_ = colTailLen_;
    } else {
        ubFactorCol32BlockNum_ =  colNormalBlockNum_ * DIGIT_EIGHT;
        ubFactorColNum_ = ubFactorCol32BlockNum_ * blockSize_;
    }

    if (coreRowIdx_ == rowTileNum_ - 1) {
        ubFactorRow1BlockNum_ = ops::CeilDiv(rowTailLen_, DIGIT_ONE);
    } else {
        ubFactorRow1BlockNum_ = rowNormalBlockNum_ * DIGIT_ONE;
    }

    ubFactorColLoopNum_ = ops::CeilDiv(ubFactorCol32BlockNum_, maxUbBlockNum_);                                             // 列方向上UB循环次数（列方向上的切 UB 块数）
    ubFactorColNormalBlockNum_ = ops::CeilDiv(ubFactorCol32BlockNum_, ubFactorColLoopNum_);                                 // 列方向上头块的Block数量（均衡处理）          用于VF内处理
    ubFactorColNormalBlockNum_ = ops::CeilDiv(ubFactorColNormalBlockNum_, DIGIT_TWO) * DIGIT_TWO;                           // 列方向上头块的Block数量补成偶数，只有尾块有可能需要补Pad
    ubFactorColTailBlockNum_ = ubFactorCol32BlockNum_ - (ubFactorColLoopNum_ - DIGIT_ONE) * ubFactorColNormalBlockNum_;     // 列方向上尾块的Block数量（只有一个尾块）      用于VF内处理
    ubFactorColNormalBlockLen_ = ubFactorColNormalBlockNum_ * blockSize_;                                                   // 列方向上头块的元素数量（均衡处理）           用于数据搬运
    ubFactorColTailBlockLen_ = ubFactorColNum_ - (ubFactorColLoopNum_ - DIGIT_ONE) * ubFactorColNormalBlockLen_;            // 列方向上尾块的元素数量（只有一个尾块）       用于数据搬运

    ubFactorRowNormalBlockNum_ = ops::FloorDiv(maxUbBlockNum_, ubFactorColNormalBlockNum_);                                 // UB 一次至多载入的行数
    ubFactorRowLoopNum_ = ops::CeilDiv(ubFactorRow1BlockNum_, ubFactorRowNormalBlockNum_);                                  // 行方向上 UB 循环次数（行方向上的切 UB 块数）
    ubFactorRowNormalBlockNum_ = ops::CeilDiv(ubFactorRow1BlockNum_, ubFactorRowLoopNum_);                                  // 行方向上的均衡处理                           用于VF内处理
    ubFactorRowTailBlockNum_ = ubFactorRow1BlockNum_ - (ubFactorRowLoopNum_ - DIGIT_ONE) * ubFactorRowNormalBlockNum_;      // 行方向上尾块的行数（只有一个尾块）            用于VF内处理
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ParseTilingData(const DynamicMxQuantTailAxisTilingData* tilingData)
{
    roundMode_ = tilingData->roundMode;
    blockSize_ = tilingData->blockSize;
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
    dstTypeMax_ = tilingData->dstTypeMax;
    invDstTypeMax_ = tilingData->invDstTypeMax;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::Process()
{
    if (coreIdx_ >= usedCoreNum_) {
        return;
    }
    int64_t dim0LoopIdx = 0;
    int64_t dim1LoopIdx = 0;
    for (dim0LoopIdx = 0; dim0LoopIdx < ubFactorRowLoopNum_ - 1; dim0LoopIdx++) {
        for (dim1LoopIdx = 0; dim1LoopIdx < ubFactorColLoopNum_ - 1; dim1LoopIdx++) {
            CopyIn(dim0LoopIdx, dim1LoopIdx, ubFactorRowNormalBlockNum_, ubFactorColNormalBlockLen_, ubFactorColNormalBlockNum_);
            Compute(ubFactorRowNormalBlockNum_, ubFactorColNormalBlockNum_);
            CopyOut(dim0LoopIdx, dim1LoopIdx, ubFactorRowNormalBlockNum_, ubFactorColNormalBlockLen_, ubFactorColNormalBlockNum_);
        }
        CopyIn(dim0LoopIdx, dim1LoopIdx, ubFactorRowNormalBlockNum_, ubFactorColTailBlockLen_, ubFactorColTailBlockNum_);
        Compute(ubFactorRowNormalBlockNum_, ubFactorColTailBlockNum_);
        CopyOut(dim0LoopIdx, dim1LoopIdx, ubFactorRowNormalBlockNum_, ubFactorColTailBlockLen_, ubFactorColTailBlockNum_);
    }
    for (dim1LoopIdx = 0; dim1LoopIdx < ubFactorColLoopNum_ - 1; dim1LoopIdx++) {
        CopyIn(dim0LoopIdx, dim1LoopIdx, ubFactorRowTailBlockNum_, ubFactorColNormalBlockLen_, ubFactorColNormalBlockNum_);
        Compute(ubFactorRowTailBlockNum_, ubFactorColNormalBlockNum_);
        CopyOut(dim0LoopIdx, dim1LoopIdx, ubFactorRowTailBlockNum_, ubFactorColNormalBlockLen_, ubFactorColNormalBlockNum_);
    }
    CopyIn(dim0LoopIdx, dim1LoopIdx, ubFactorRowTailBlockNum_, ubFactorColTailBlockLen_, ubFactorColTailBlockNum_);
    Compute(ubFactorRowTailBlockNum_, ubFactorColTailBlockNum_);
    CopyOut(dim0LoopIdx, dim1LoopIdx, ubFactorRowTailBlockNum_, ubFactorColTailBlockLen_, ubFactorColTailBlockNum_);
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::CopyIn(int64_t dim0LoopIdx, int64_t dim1LoopIdx, int64_t ubFactorRowNum, int64_t ubFactorColNum, int64_t ubFactorColBlockNum)
{
    int64_t scaleIsNotOdd = ubFactorColBlockNum % DIGIT_TWO;

    LocalTensor<T> xLocal = inQueue_.AllocTensor<T>();
    if (scaleIsNotOdd != 0) {                               // 当前 UB 块一行的scale数不是偶数时，需要将X的Buffer空间预先填充为0
        // 非偶数scale时，提前填充0
        Duplicate<T>(xLocal, static_cast<T>(0), maxUbBlockNum_ * blockSize_);
        event_t eventIDVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
    }

    DataCopyExtParams copyInParamX = {
        static_cast<uint16_t>(ubFactorRowNum), static_cast<uint32_t>(ubFactorColNum * sizeof(T)),
        static_cast<uint32_t>((colNum_ - ubFactorColNum) * sizeof(T)), static_cast<uint32_t>(scaleIsNotOdd * DIGIT_TWO), static_cast<uint32_t>(0)}; // 一个BlockSize=32，输入x为2Bytes，搬入UB需要跳64Bytes，即2个数据块

    // 搬运X，补Pad示例
    // Eg1: dataLen = 1  => leftPad = [0 * sizeof(T)] Bytes , rightPad = [16 * sizeof(T)] Bytes  , dummy = [15 * sizeof(T) + 2 * 16 * sizeof(T)] Bytes
    // Eg2: dataLen = 17 => leftPad = [0 * sizeof(T)] Bytes , rightPad = [15 * sizeof(T)] Bytes  , dummy = [ 0 * sizeof(T) + 2 * 16 * sizeof(T)] Bytes
    // Eg3: dataLen = 32 => leftPad = [0 * sizeof(T)] Bytes , rightPad = [ 0 * sizeof(T)] Bytes  , dummy = [ 0 * sizeof(T) + 2 * 16 * sizeof(T)] Bytes
    // Eg4: dataLen = 35 => leftPad = [0 * sizeof(T)] Bytes , rightPad = [16 * sizeof(T)] Bytes  , dummy = [13 * sizeof(T) + 0 * 16 * sizeof(T)] Bytes
    // 当补的数(rightPad)是16时，一定能补到UB的下一个DataBlock，实现补到32个数对齐的功能
    int64_t padRightNumTmp = (ubFactorColNum % 32 > 16) ? (32 - ubFactorColNum % 32) : 16;
    int64_t padRightNum = (ubFactorColNum % 32 == 0) ? 0 : padRightNumTmp;
    DataCopyPadExtParams<T> padParams_ = {true, static_cast<uint8_t>(0), static_cast<uint8_t>(padRightNum), static_cast<T>(0)};
    int64_t offset = dim0LoopIdx * ubFactorRowNormalBlockNum_ * colNum_ + dim1LoopIdx * ubFactorColNormalBlockLen_;

    DataCopyPad(xLocal, xGm_[offset], copyInParamX, padParams_);
    inQueue_.EnQue(xLocal);
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::CopyOut(int64_t dim0LoopIdx, int64_t dim1LoopIdx, int64_t ubFactorRowNum, int64_t ubFactorColNum, int64_t ubFactorColBlockNum)
{
    LocalTensor<uint8_t> scaleLocal = mxScaleQueue_.DeQue<uint8_t>();
    LocalTensor<uint8_t> yLocal = outQueue_.DeQue<uint8_t>();
    int64_t scaleIsNotOdd = ubFactorColBlockNum % DIGIT_TWO;

    DataCopyExtParams copyOutParamY = {
        static_cast<uint16_t>(ubFactorRowNum), static_cast<uint32_t>(ubFactorColNum / DIGIT_TWO), static_cast<uint32_t>(0), // 一个 BlockSize = 32，64个数对齐，输出 y 为 0.5 Bytes，占用 32 Bytes，srcStride = 0
        static_cast<uint32_t>((colNum_ - ubFactorColNum) / DIGIT_TWO), static_cast<uint32_t>(0)};

    int64_t offset = (dim0LoopIdx * ubFactorRowNormalBlockNum_ * colNum_ + dim1LoopIdx * ubFactorColNormalBlockLen_) / DIGIT_TWO;
    DataCopyPad(yGm_[offset], yLocal, copyOutParamY);

    DataCopyExtParams copyOutParamScale = {0, 0, 0, 0, 0};
    copyOutParamScale.blockCount = ubFactorRowNum;
    copyOutParamScale.blockLen = ubFactorColBlockNum + scaleIsNotOdd;
    copyOutParamScale.srcStride = 0;
    copyOutParamScale.dstStride = scaleColNum_ - copyOutParamScale.blockLen;

    int64_t scaleOffset = dim0LoopIdx * ubFactorRowNormalBlockNum_ * scaleColNum_  + dim1LoopIdx * ubFactorColNormalBlockNum_;
    DataCopyPad<uint8_t, PaddingMode::Compact>(scaleGm_[scaleOffset], scaleLocal, copyOutParamScale);

    outQueue_.FreeTensor(yLocal);
    mxScaleQueue_.FreeTensor(scaleLocal);
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ComputeMaxExp(__ubuf__ T* xLocalAddr, __ubuf__ uint16_t* maxExpAddr, uint16_t LoopNum2VF)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vdExp0;
        Reg::RegTensor<T> vdExp1;
        Reg::RegTensor<uint16_t> vdMaxExp;
        Reg::RegTensor<bfloat16_t> vdExp0BF16;
        Reg::RegTensor<bfloat16_t> vdExp1BF16;
        Reg::RegTensor<uint16_t> vdExpExtract0;
        Reg::RegTensor<uint16_t> vdExpExtract1;
        Reg::RegTensor<uint16_t> vdExpSelect0;
        Reg::RegTensor<uint16_t> vdExpSelect1;

        Reg::RegTensor<uint16_t> expMaskBF16;
        Reg::Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);
        Reg::RegTensor<uint16_t> invalidmaskfp16;
        Reg::Duplicate(invalidmaskfp16, INVALID_FLOAT16);
        
        Reg::MaskReg Mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg invalidDataMask0;
        Reg::MaskReg invalidDataMask1;
        Reg::UnalignReg ureg;

        static constexpr Reg::CastTrait castTraitHalf2Bf16 = {
            Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
            Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};

        for (uint16_t i = 0; i < LoopNum2VF; i++) {
            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, xLocalAddr, vfLen16Double);

            if constexpr (IsSame<T, half>::value) {
                Reg::And(
                    vdExpSelect0, (Reg::RegTensor<uint16_t>&)vdExp0, invalidmaskfp16, Mask);
                Reg::And(
                    vdExpSelect1, (Reg::RegTensor<uint16_t>&)vdExp1, invalidmaskfp16, Mask);
                Reg::Compare<uint16_t, CMPMODE::NE>(
                    invalidDataMask0, vdExpSelect0, invalidmaskfp16, Mask);
                Reg::Compare<uint16_t, CMPMODE::NE>(
                    invalidDataMask1, vdExpSelect1, invalidmaskfp16, Mask);
                Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp0BF16, vdExp0, Mask);
                Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp1BF16, vdExp1, Mask);
                Reg::And(
                    vdExpExtract0, (Reg::RegTensor<uint16_t>&)vdExp0BF16, expMaskBF16, Mask);
                Reg::And(
                    vdExpExtract1, (Reg::RegTensor<uint16_t>&)vdExp1BF16, expMaskBF16, Mask);
                Reg::Select<uint16_t>(vdExpExtract0, vdExpExtract0, expMaskBF16, invalidDataMask0);
                Reg::Select<uint16_t>(vdExpExtract1, vdExpExtract1, expMaskBF16, invalidDataMask1);
            } else {
                Reg::And(
                    vdExpExtract0, (Reg::RegTensor<uint16_t>&)vdExp0, expMaskBF16, Mask);
                Reg::And(
                    vdExpExtract1, (Reg::RegTensor<uint16_t>&)vdExp1, expMaskBF16, Mask);
            }

            Reg::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, Mask);
            Reg::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, Mask);

            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, vdMaxExp, ureg, elementAfterReduce_);
        }
        Reg::StoreUnAlignPost(maxExpAddr, ureg, 0);
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ComputeMaxExpDynamicDtypeRange(__ubuf__ T* xLocalAddr, __ubuf__ uint16_t* maxExpAddr, uint16_t LoopNum2VF){
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vdExp0;
        Reg::RegTensor<T> vdExp1;
        Reg::RegTensor<uint16_t> vdMaxExp;
        Reg::RegTensor<bfloat16_t> vdExp0BF16;
        Reg::RegTensor<bfloat16_t> vdExp1BF16;
        Reg::RegTensor<uint16_t> vdExpSelect0;
        Reg::RegTensor<uint16_t> vdExpSelect1;
        Reg::RegTensor<uint16_t> vdExpExtract0;
        Reg::RegTensor<uint16_t> vdExpExtract1;

        Reg::RegTensor<uint16_t> expMaskBF16;
        Reg::Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);
        Reg::RegTensor<uint16_t> invalidMaskFP16;
        Reg::Duplicate(invalidMaskFP16, INVALID_FLOAT16);
        Reg::RegTensor<uint16_t> absMask16Bit;
        Reg::Duplicate(absMask16Bit, ABS_MASK_FOR_16BIT);

        Reg::MaskReg Mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg invalidDataMask0;
        Reg::MaskReg invalidDataMask1;
        Reg::UnalignReg ureg;

        static constexpr Reg::CastTrait castTraitHalf2Bf16 = {
            Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
            Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

        for (uint16_t i = 0; i < LoopNum2VF; i++) {
            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE,
                Reg::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, xLocalAddr, vfLen16Double);
            if constexpr (ops::IsSame<T, half>::value) {
                Reg::And(vdExpSelect0, (Reg::RegTensor<uint16_t>&)vdExp0, invalidMaskFP16, Mask);
                Reg::And(vdExpSelect1, (Reg::RegTensor<uint16_t>&)vdExp1, invalidMaskFP16, Mask);
                Reg::Compare<uint16_t, CMPMODE::NE>(invalidDataMask0, vdExpSelect0, invalidMaskFP16, Mask);
                Reg::Compare<uint16_t, CMPMODE::NE>(invalidDataMask1, vdExpSelect1, invalidMaskFP16, Mask);
                Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp0BF16, vdExp0, Mask);
                Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp1BF16, vdExp1, Mask);
                Reg::And(vdExpExtract0, (Reg::RegTensor<uint16_t>&)vdExp0BF16, absMask16Bit, Mask);
                Reg::And(vdExpExtract1, (Reg::RegTensor<uint16_t>&)vdExp1BF16, absMask16Bit, Mask);
                Reg::Select<uint16_t>(vdExpExtract0, vdExpExtract0, expMaskBF16, invalidDataMask0);
                Reg::Select<uint16_t>(vdExpExtract1, vdExpExtract1, expMaskBF16, invalidDataMask1);
            } else {
                Reg::And(vdExpExtract0, (Reg::RegTensor<uint16_t>&)vdExp0, absMask16Bit, Mask);
                Reg::And(vdExpExtract1, (Reg::RegTensor<uint16_t>&)vdExp1, absMask16Bit, Mask);
            }

            Reg::Max(vdMaxExp, vdExpExtract0, vdExpExtract1, Mask);
            Reg::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, Mask);

            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, vdMaxExp, ureg, elementAfterReduce_);
        }
        Reg::StoreUnAlignPost(maxExpAddr, ureg, 0);
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ComputeMaxExpcuBLAS(__ubuf__ T* xLocalAddr, __ubuf__ uint16_t* maxExpAddr, uint16_t LoopNum2VF){
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vdExp0;
        Reg::RegTensor<T> vdExp1;
        Reg::RegTensor<uint16_t> vdMaxExp;

        Reg::RegTensor<uint16_t> absMask16Bit;
        Reg::Duplicate(absMask16Bit, ABS_MASK_FOR_16BIT);

        Reg::MaskReg Mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::UnalignReg ureg;

        for (uint16_t i = 0; i < LoopNum2VF; i++) {
            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE,
                Reg::LoadDist::DIST_DINTLV_B16>(vdExp0, vdExp1, xLocalAddr, vfLen16Double);
            Reg::And(
                (Reg::RegTensor<uint16_t>&)vdExp0, (Reg::RegTensor<uint16_t>&)vdExp0,
                absMask16Bit, Mask);
            Reg::And(
                (Reg::RegTensor<uint16_t>&)vdExp1, (Reg::RegTensor<uint16_t>&)vdExp1,
                absMask16Bit, Mask);
            Reg::Max(
                vdMaxExp, (Reg::RegTensor<uint16_t>&)vdExp0,
                (Reg::RegTensor<uint16_t>&)vdExp1, Mask);
            Reg::ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, Mask);
            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                maxExpAddr, vdMaxExp, ureg, elementAfterReduce_);
        }
        Reg::StoreUnAlignPost(maxExpAddr, ureg, 0);
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ComputeScale(__ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr, uint16_t LoopNum1VF, uint32_t totalScaleInUB)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vdExp0;
        Reg::RegTensor<T> vdExp1;
        Reg::RegTensor<uint16_t> vdMaxExp;
        Reg::RegTensor<uint16_t> sharedExp;
        Reg::RegTensor<uint16_t> scaleValue;
        Reg::RegTensor<uint16_t> halfScale;

        Reg::RegTensor<uint16_t> expMask;
        Reg::Duplicate(expMask, MAX_EXP_FOR_BF16);
        Reg::RegTensor<uint16_t> maxExpValue;
        Reg::Duplicate(maxExpValue, f4Emax_);
        Reg::RegTensor<uint16_t> scaleBias;
        Reg::Duplicate(scaleBias, BF16_EXP_BIAS);
        Reg::RegTensor<uint16_t> fp8NanRegTensor;
        Reg::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
        Reg::RegTensor<uint16_t> zeroRegTensor;
        Reg::Duplicate(zeroRegTensor, 0);
        Reg::RegTensor<uint16_t> nanRegTensor;
        Reg::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
        Reg::RegTensor<uint16_t> specialExpRegTensor;
        Reg::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);

        Reg::MaskReg cmpResult;
        Reg::MaskReg zeroMask;
        Reg::MaskReg preMaskScale;
        Reg::MaskReg invalidDataMask;
        Reg::MaskReg specialDataMask;

        for (uint16_t i = 0; i < LoopNum1VF; i++) {
            preMaskScale = Reg::UpdateMask<uint16_t>(totalScaleInUB);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                vdMaxExp, maxExpAddr, vfLen16);
            Reg::Compare<uint16_t, CMPMODE::NE>(cmpResult, vdMaxExp, expMask, preMaskScale); // INF/NAN
            Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, vdMaxExp, zeroRegTensor, preMaskScale);
            Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, vdMaxExp, maxExpValue, preMaskScale);

            Reg::Select<uint16_t>(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);

            Reg::Sub(sharedExp, vdMaxExp, maxExpValue, preMaskScale);
            Reg::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);

            Reg::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
            Reg::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);

            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr, scaleValue, vfLen32, preMaskScale);           // 128 个scale，占用 128 * 1 Btyes = vfLen32 * sizeof(uint16_t)

            Reg::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
            Reg::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            Reg::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            Reg::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            Reg::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                recipScaleLocalAddr, halfScale, vfLen16, preMaskScale);
        }
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ComputeScaleDynamicDtypeRange(__ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr, uint16_t LoopNum1VF, uint32_t totalScaleInUB){
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> vdMaxExp;
        Reg::RegTensor<uint16_t> sharedExp;
        Reg::RegTensor<uint16_t> scaleValue;
        Reg::RegTensor<uint16_t> halfScale;
        Reg::RegTensor<uint16_t> vdMaxExpAdd;
        Reg::RegTensor<uint16_t> vdMaxExpOnly;

        Reg::RegTensor<uint16_t> expMask;
        Reg::Duplicate(expMask, MAX_EXP_FOR_BF16);
        Reg::RegTensor<uint16_t> addValue;
        Reg::Duplicate(addValue, addValueBit);
        Reg::RegTensor<uint16_t> maxExpValue;
        Reg::Duplicate(maxExpValue, FP4_E2M1_BF16_MAX_EXP);
        Reg::RegTensor<uint16_t> scaleBias;
        Reg::Duplicate(scaleBias, BF16_EXP_BIAS);
        Reg::RegTensor<uint16_t> fp8NanRegTensor;
        Reg::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8);
        Reg::RegTensor<uint16_t> zeroRegTensor;
        Reg::Duplicate(zeroRegTensor, 0);
        Reg::RegTensor<uint16_t> nanRegTensor;
        Reg::Duplicate(nanRegTensor, NAN_CUSTOMIZATION);
        Reg::RegTensor<uint16_t> specialExpRegTensor;
        Reg::Duplicate(specialExpRegTensor, SPECIAL_EXP_THRESHOLD);

        Reg::MaskReg cmpResult;
        Reg::MaskReg zeroMask;
        Reg::MaskReg invalidDataMask;
        Reg::MaskReg specialDataMask;
        Reg::MaskReg preMaskScale;

        for (uint16_t i = 0; i < LoopNum1VF; i++) {
            preMaskScale = Reg::UpdateMask<uint16_t>(totalScaleInUB);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                vdMaxExp, maxExpAddr, vfLen16);
            Reg::And(vdMaxExpOnly, vdMaxExp, expMask, preMaskScale);  // 提取指数位
            Reg::Compare<uint16_t, CMPMODE::NE>(cmpResult, vdMaxExpOnly, expMask, preMaskScale); // INF/NAN
            Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, vdMaxExpOnly, zeroRegTensor, preMaskScale);
            Reg::Compare<uint16_t, CMPMODE::LT>(invalidDataMask, vdMaxExpOnly, maxExpValue, preMaskScale);
            
            Reg::Add(vdMaxExpAdd, vdMaxExp, addValue, preMaskScale);     // 进位后的结果
            Reg::And(vdMaxExpAdd, vdMaxExpAdd, expMask, preMaskScale);      // 提取进位结果的指数位
            Reg::Select<uint16_t>(vdMaxExpAdd, maxExpValue, vdMaxExpAdd, invalidDataMask);
            Reg::Sub(sharedExp, vdMaxExpAdd, maxExpValue, preMaskScale);

            Reg::ShiftRights(scaleValue, sharedExp, SHR_NUM_FOR_BF16, preMaskScale);
            Reg::Select<uint16_t>(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);
            Reg::Select<uint16_t>(scaleValue, scaleValue, zeroRegTensor, zeroMask);

            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr, scaleValue, vfLen32, preMaskScale);

            Reg::Compare<uint16_t, CMPMODE::EQ>(specialDataMask, sharedExp, scaleBias, preMaskScale);
            Reg::Sub(halfScale, scaleBias, sharedExp, preMaskScale);
            Reg::Select<uint16_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            Reg::Select<uint16_t>(halfScale, halfScale, zeroRegTensor, zeroMask);
            Reg::Select<uint16_t>(halfScale, specialExpRegTensor, halfScale, specialDataMask);

            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE,
                Reg::StoreDist::DIST_NORM>(recipScaleLocalAddr, halfScale, vfLen16, preMaskScale);
        }
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ComputeScalecuBLASFP4(__ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* scaleLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr, uint16_t LoopNumHalfVF, uint32_t totalScaleInUB)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> max16;
        Reg::RegTensor<uint32_t> max32;
        Reg::RegTensor<uint32_t> exp32;
        Reg::RegTensor<uint32_t> man32;
        Reg::RegTensor<uint32_t> normalExp32;
        Reg::RegTensor<uint32_t> expAddOne32;
        Reg::RegTensor<uint32_t> extractExp;
        Reg::RegTensor<uint16_t> expOut;
        Reg::RegTensor<uint32_t> halfScale;
        Reg::RegTensor<uint16_t> recExpOut;

        Reg::RegTensor<uint32_t> manMaskFP32;
        Reg::Duplicate(manMaskFP32, MAN_MASK_FLOAT);
        Reg::RegTensor<uint32_t> expMask;
        Reg::Duplicate(expMask, MAX_EXP_FOR_FP32);
        Reg::RegTensor<uint32_t> zeroRegTensor32;
        Reg::Duplicate(zeroRegTensor32, 0);
        Reg::RegTensor<uint32_t> scaleBias;
        Reg::Duplicate(scaleBias, FP32_EXP_BIAS_CUBLAS);
        Reg::RegTensor<uint32_t> nanRegTensor;
        Reg::Duplicate(nanRegTensor, NAN_CUSTOMIZATION_PACK);
        Reg::RegTensor<uint32_t> fp4NanRegTensor;
        Reg::Duplicate(fp4NanRegTensor, MAX_EXP_FOR_FP8_IN_FP32);
        Reg::RegTensor<float> invMax;
        Reg::Duplicate(invMax, invDstTypeMax_);

        Reg::MaskReg cmpResult;
        Reg::MaskReg zeroMask;
        Reg::MaskReg p0;
        Reg::MaskReg p1;
        Reg::MaskReg p2;
        Reg::MaskReg preMaskScale;
        uint32_t SixtyFour = 64;
        Reg::MaskReg dataMaskB16Half = Reg::UpdateMask<uint16_t>(SixtyFour);

        static constexpr Reg::CastTrait castTraitHalf2Float = {
            Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
            Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

        for (uint16_t i = 0; i < LoopNumHalfVF; i++) {
            preMaskScale = Reg::UpdateMask<uint32_t>(totalScaleInUB);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE,
                Reg::LoadDist::DIST_UNPACK_B16>(max16, maxExpAddr, vfLen32);

            Reg::Cast<float, T, castTraitHalf2Float>(
                (Reg::RegTensor<float>&)max32, (Reg::RegTensor<T>&)max16, preMaskScale);
            Reg::Compare<uint32_t, CMPMODE::LT>(cmpResult, max32, expMask, preMaskScale);
            Reg::Compare<uint32_t, CMPMODE::NE>(zeroMask, max32, zeroRegTensor32, preMaskScale);
         
            Reg::Mul((Reg::RegTensor<float>&)max32, (Reg::RegTensor<float>&)max32,
                invMax, preMaskScale);
            Reg::ShiftRights(exp32, max32, SHR_NUM_FOR_FP32, preMaskScale);
            Reg::And(man32, max32, manMaskFP32, preMaskScale);

            Reg::CompareScalar<uint32_t, CMPMODE::GT>(p0, exp32, NUMBER_ZERO, preMaskScale);
            Reg::CompareScalar<uint32_t, CMPMODE::LT>(p1, exp32, NUMBER_TWO_FIVE_FOUR, preMaskScale);
            Reg::CompareScalar<uint32_t, CMPMODE::GT>(p2, man32, NUMBER_ZERO, preMaskScale);
            Reg::MaskAnd(p0, p0, p1, preMaskScale);
            Reg::MaskAnd(p0, p0, p2, preMaskScale);

            Reg::Adds(expAddOne32, exp32, 1, preMaskScale);
            Reg::Select(extractExp, expAddOne32, exp32, p0);
            Reg::Select<uint32_t>(extractExp, extractExp, fp4NanRegTensor, cmpResult);
            Reg::Select<uint32_t>(extractExp, extractExp, zeroRegTensor32, zeroMask);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(expOut, extractExp);

            Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_PACK_B16>(
                scaleLocalAddr + i * 32, expOut, dataMaskB16Half);

            Reg::ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, preMaskScale);
            Reg::Sub(halfScale, scaleBias, extractExp, preMaskScale);
            Reg::Select<uint32_t>(halfScale, halfScale, nanRegTensor, cmpResult);
            Reg::Select<uint32_t>(halfScale, halfScale, zeroRegTensor32, zeroMask);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(recExpOut, halfScale);

            Reg::StoreAlign<uint16_t>(
                recipScaleLocalAddr + i * vfLen32, recExpOut, dataMaskB16Half);
        }
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
template <RoundMode toBf16RoundMode, RoundMode roundMode>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ComputeData(__ubuf__ T* xLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr, __ubuf__ int8_t* yLocalAddr, uint16_t LoopNum2VF)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vdExp0;
        Reg::RegTensor<T> vdExp1;
        Reg::RegTensor<T> vdExp0Convert;
        Reg::RegTensor<T> vdExp1Convert;
        Reg::RegTensor<uint16_t> halfScaleForMul;
        Reg::RegTensor<bfloat16_t> vdExp0BF16;
        Reg::RegTensor<bfloat16_t> vdExp1BF16;
        Reg::RegTensor<bfloat16_t> vdBF16Exp0FP4;
        Reg::RegTensor<bfloat16_t> vdBF16Exp1FP4;
        Reg::RegTensor<U> vdExp0FP4;
        Reg::RegTensor<U> vdExp1FP4;

        Reg::MaskReg Mask = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();

        static constexpr Reg::CastTrait castTrait = {
            Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
            Reg::MaskMergeMode::ZEROING, roundMode};
        static constexpr Reg::CastTrait castTraitHalf2Bf16 = {
            Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
            Reg::MaskMergeMode::ZEROING, toBf16RoundMode};

        for (uint16_t i = 0; i < LoopNum2VF; i++) {
            Reg::LoadAlign<
                T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, xLocalAddr, vfLen16Double);
            Reg::LoadAlign<
                uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                halfScaleForMul, recipScaleLocalAddr, elementAfterReduce_);

            if constexpr (IsSame<T, half>::value) {
                if constexpr (roundMode == RoundMode::CAST_RINT) {
                    FP16Convert(vdExp0, vdExp0, Mask);
                    FP16Convert(vdExp1, vdExp1, Mask);
                }
                Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp0BF16, vdExp0, Mask);
                Reg::Cast<bfloat16_t, T, castTraitHalf2Bf16>(vdExp1BF16, vdExp1, Mask);
                Reg::Mul(
                    vdExp0BF16, vdExp0BF16, (Reg::RegTensor<bfloat16_t>&)halfScaleForMul, Mask);
                Reg::Mul(
                    vdExp1BF16, vdExp1BF16, (Reg::RegTensor<bfloat16_t>&)halfScaleForMul, Mask);
                Reg::Interleave(vdExp0BF16, vdExp1BF16, vdExp0BF16, vdExp1BF16);
                Reg::Cast<U, bfloat16_t, castTrait>(vdExp0FP4, vdExp0BF16, Mask);
                Reg::Cast<U, bfloat16_t, castTrait>(vdExp1FP4, vdExp1BF16, Mask);
            } else {
                Reg::Mul(vdExp0, vdExp0, (Reg::RegTensor<T>&)halfScaleForMul, Mask);
                Reg::Mul(vdExp1, vdExp1, (Reg::RegTensor<T>&)halfScaleForMul, Mask);
                Reg::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
                Reg::Cast<U, T, castTrait>(vdExp0FP4, vdExp0, Mask);
                Reg::Cast<U, T, castTrait>(vdExp1FP4, vdExp1, Mask);
            }

            Reg::StoreAlign<
                int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                yLocalAddr, (Reg::RegTensor<int8_t>&)vdExp0FP4, OUT_ELE_NUM_ONE_BLK, Mask);
            Reg::StoreAlign<
                int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                yLocalAddr, (Reg::RegTensor<int8_t>&)vdExp1FP4, OUT_ELE_NUM_ONE_BLK, Mask);
        }
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
template <RoundMode toBf16RoundMode, RoundMode roundMode>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ComputeDataOptimize(__ubuf__ T* xLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr, __ubuf__ int8_t* yLocalAddr, uint16_t LoopNum2VF)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> halfScaleForMul;
        Reg::RegTensor<T> vdExp0;
        Reg::RegTensor<T> vdExp1;
        Reg::RegTensor<T> vdExp0Convert;
        Reg::RegTensor<T> vdExp1Convert;
        Reg::RegTensor<float> halfScaleForMulFP32;
        Reg::RegTensor<float> vdExp0ZeroFP32;
        Reg::RegTensor<float> vdExp0OneFP32;
        Reg::RegTensor<float> vdExp1ZeroFP32;
        Reg::RegTensor<float> vdExp1OneFP32;
        Reg::RegTensor<bfloat16_t> vdExp0ZeroBF16;
        Reg::RegTensor<bfloat16_t> vdExp0OneBF16;
        Reg::RegTensor<bfloat16_t> vdExp1ZeroBF16;
        Reg::RegTensor<bfloat16_t> vdExp1OneBF16;

        Reg::RegTensor<bfloat16_t> vdExp0BF16;
        Reg::RegTensor<bfloat16_t> vdExp1BF16;

        Reg::RegTensor<U> vdExp0FP4;
        Reg::RegTensor<U> vdExp1FP4;

        Reg::RegTensor<bfloat16_t> vdBF16Exp0FP4;
        Reg::RegTensor<bfloat16_t> vdBF16Exp1FP4;

        Reg::MaskReg dataMaskB16 = Reg::CreateMask<half>();
        Reg::MaskReg dataMaskB32 = Reg::CreateMask<float>();

        static constexpr Reg::CastTrait castTrait = {
            Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
            Reg::MaskMergeMode::ZEROING, roundMode};
        static constexpr Reg::CastTrait castTraitHalf2Bf16 = {
            Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
            Reg::MaskMergeMode::ZEROING, toBf16RoundMode};
        static constexpr Reg::CastTrait castTraitF16toFp32Zero = {
            Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
            RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitF16toFp32One = {
            Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitFp32toBF16 = {
            Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, roundMode};

        for (uint16_t i = 0; i < LoopNum2VF; i++) {
            Reg::LoadAlign<
                T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, xLocalAddr, vfLen16Double);
            Reg::LoadAlign<
                uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                halfScaleForMul, recipScaleLocalAddr, elementAfterReduce_);

            if constexpr (IsSame<T, half>::value) {
                Reg::Cast<float, bfloat16_t, castTraitF16toFp32Zero>(
                    halfScaleForMulFP32, (Reg::RegTensor<bfloat16_t>&)halfScaleForMul, dataMaskB16);

                // vdExp0
                Reg::Cast<float, T, castTraitF16toFp32Zero>(vdExp0ZeroFP32, vdExp0, dataMaskB16);
                Reg::Cast<float, T, castTraitF16toFp32One>(vdExp0OneFP32, vdExp0, dataMaskB16);

                Reg::Mul(vdExp0ZeroFP32, vdExp0ZeroFP32, halfScaleForMulFP32, dataMaskB32);
                Reg::Mul(vdExp0OneFP32, vdExp0OneFP32, halfScaleForMulFP32, dataMaskB32);
                ComputeFP4FromHalf<toBf16RoundMode, roundMode>(vdExp0ZeroFP32);
                ComputeFP4FromHalf<toBf16RoundMode, roundMode>(vdExp0OneFP32);
                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(
                    vdExp0ZeroBF16, vdExp0ZeroFP32, dataMaskB32);
                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(
                    vdExp0OneBF16, vdExp0OneFP32, dataMaskB32);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(
                    (Reg::RegTensor<uint16_t>&)vdExp0ZeroBF16, (Reg::RegTensor<uint32_t>&)vdExp0ZeroBF16);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(
                    (Reg::RegTensor<uint16_t>&)vdExp0OneBF16, (Reg::RegTensor<uint32_t>&)vdExp0OneBF16);
                Reg::Interleave(vdExp0ZeroBF16, vdExp0OneBF16, vdExp0ZeroBF16, vdExp0OneBF16);

                // vdExp1
                Reg::Cast<float, T, castTraitF16toFp32Zero>(vdExp1ZeroFP32, vdExp1, dataMaskB16);
                Reg::Cast<float, T, castTraitF16toFp32One>(vdExp1OneFP32, vdExp1, dataMaskB16);

                Reg::Mul(vdExp1ZeroFP32, vdExp1ZeroFP32, halfScaleForMulFP32, dataMaskB32);
                Reg::Mul(vdExp1OneFP32, vdExp1OneFP32, halfScaleForMulFP32, dataMaskB32);
                ComputeFP4FromHalf<toBf16RoundMode, roundMode>(vdExp1ZeroFP32);
                ComputeFP4FromHalf<toBf16RoundMode, roundMode>(vdExp1OneFP32);
                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(
                    vdExp1ZeroBF16, vdExp1ZeroFP32, dataMaskB32);
                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(
                    vdExp1OneBF16, vdExp1OneFP32, dataMaskB32);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(
                    (Reg::RegTensor<uint16_t>&)vdExp1ZeroBF16, (Reg::RegTensor<uint32_t>&)vdExp1ZeroBF16);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(
                    (Reg::RegTensor<uint16_t>&)vdExp1OneBF16, (Reg::RegTensor<uint32_t>&)vdExp1OneBF16);
                Reg::Interleave(vdExp1ZeroBF16, vdExp1OneBF16, vdExp1ZeroBF16, vdExp1OneBF16);

                Reg::Interleave(vdExp0ZeroBF16, vdExp1ZeroBF16, vdExp0ZeroBF16, vdExp1ZeroBF16);
                Reg::Cast<U, bfloat16_t, castTrait>(vdExp0FP4, vdExp0ZeroBF16, dataMaskB16);
                Reg::Cast<U, bfloat16_t, castTrait>(vdExp1FP4, vdExp1ZeroBF16, dataMaskB16);
            } else {
                Reg::Mul(vdExp0, vdExp0, (Reg::RegTensor<T>&)halfScaleForMul, dataMaskB16);
                Reg::Mul(vdExp1, vdExp1, (Reg::RegTensor<T>&)halfScaleForMul, dataMaskB16);
                Reg::Interleave(vdExp0, vdExp1, vdExp0, vdExp1);
                Reg::Cast<U, T, castTrait>(vdExp0FP4, vdExp0, dataMaskB16);
                Reg::Cast<U, T, castTrait>(vdExp1FP4, vdExp1, dataMaskB16);
            }

            Reg::StoreAlign<
                int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                yLocalAddr, (Reg::RegTensor<int8_t>&)vdExp0FP4, OUT_ELE_NUM_ONE_BLK, dataMaskB16);
            Reg::StoreAlign<
                int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                yLocalAddr, (Reg::RegTensor<int8_t>&)vdExp1FP4, OUT_ELE_NUM_ONE_BLK, dataMaskB16);
        }
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::Compute(int64_t ubFactorRowBlockNum, int64_t ubFactorColBlockNum)
{
    ubFactorColBlockNum = ops::CeilDiv(ubFactorColBlockNum, DIGIT_TWO) * DIGIT_TWO;
    uint32_t totalBlockNum = ubFactorRowBlockNum * ubFactorColBlockNum;             // 当前Ub内总的Block数量 = 当前Ub内总的计算出Scale数量

    uint16_t LoopNum2VF = ops::CeilDiv(static_cast<uint16_t>(totalBlockNum), elementAfterReduce_);                          // 当前Ub内总的Block数量向上整除8，BlockSize=32，即得到双搬256个X的循环次数
    uint16_t LoopNum1VF = ops::CeilDiv(static_cast<uint16_t>(totalBlockNum), static_cast<uint16_t>(vfLen16));               // 当前Ub内总的Block数量向上整除128，即得到单搬128个max的循环次数
    uint16_t LoopNumHalfVF = ops::CeilDiv(static_cast<uint16_t>(totalBlockNum), static_cast<uint16_t>(vfLen32));            // 当前Ub内总的Block数量向上整除64，即得到单搬64个max的循环次数

    LocalTensor<T> xLocal = inQueue_.DeQue<T>();
    LocalTensor<uint16_t> scaleLocal = mxScaleQueue_.AllocTensor<uint16_t>();
    LocalTensor<int8_t> yLocal = outQueue_.AllocTensor<int8_t>();
    LocalTensor<uint16_t> maxExpLocal = maxExpBuffer_.Get<uint16_t>();
    LocalTensor<uint16_t> recipScaleLocal = recipScaleBuffer_.Get<uint16_t>();

    auto xLocalAddr = reinterpret_cast<__ubuf__ T*>(xLocal.GetPhyAddr());
    auto scaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(scaleLocal.GetPhyAddr());
    auto yLocalAddr = reinterpret_cast<__ubuf__ int8_t*>(yLocal.GetPhyAddr());
    auto maxExpLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpLocal.GetPhyAddr());
    auto recipScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(recipScaleLocal.GetPhyAddr());

    if constexpr (ops::IsSame<U, fp4x2_e1m2_t>::value || (ops::IsSame<U, fp4x2_e2m1_t>::value && SCALE_ALG == DIGIT_ZERO)) {
        ComputeMaxExp(xLocalAddr, maxExpLocalAddr, LoopNum2VF);
        maxExpLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpLocal.GetPhyAddr());
        ComputeScale(maxExpLocalAddr, scaleLocalAddr, recipScaleLocalAddr, LoopNum1VF, totalBlockNum);
        xLocalAddr = reinterpret_cast<__ubuf__ T*>(xLocal.GetPhyAddr());
        scaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(scaleLocal.GetPhyAddr());
        recipScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(recipScaleLocal.GetPhyAddr());
        if (roundMode_ == MODE_RINT) {
            ComputeDataOptimize<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(xLocalAddr, recipScaleLocalAddr, yLocalAddr, LoopNum2VF);
        } else if (roundMode_ == MODE_ROUND) {
            ComputeDataOptimize<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(xLocalAddr, recipScaleLocalAddr, yLocalAddr, LoopNum2VF);
        } else if (roundMode_ == MODE_FLOOR) {
            ComputeData<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR>(xLocalAddr, recipScaleLocalAddr, yLocalAddr, LoopNum2VF);
        }
    } else if constexpr (ops::IsSame<U, fp4x2_e2m1_t>::value && SCALE_ALG == DIGIT_TWO) {
        if (dstTypeMax_ == DIGIT_ZERO_FLOAT || dstTypeMax_ == DIGIT_SIX_FLOAT || dstTypeMax_ == DIGIT_SEVEN_FLOAT) {
            ComputeMaxExpDynamicDtypeRange(xLocalAddr, maxExpLocalAddr, LoopNum2VF);
            maxExpLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpLocal.GetPhyAddr());
            ComputeScaleDynamicDtypeRange(maxExpLocalAddr, scaleLocalAddr, recipScaleLocalAddr, LoopNum1VF, totalBlockNum);
            xLocalAddr = reinterpret_cast<__ubuf__ T*>(xLocal.GetPhyAddr());
            scaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(scaleLocal.GetPhyAddr());
            recipScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(recipScaleLocal.GetPhyAddr());
            if (roundMode_ == MODE_RINT) {
                ComputeDataOptimize<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(xLocalAddr, recipScaleLocalAddr, yLocalAddr, LoopNum2VF);
            } else if (roundMode_ == MODE_ROUND) {
                ComputeDataOptimize<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(xLocalAddr, recipScaleLocalAddr, yLocalAddr, LoopNum2VF);
            } else if (roundMode_ == MODE_FLOOR) {
                ComputeData<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR>(xLocalAddr, recipScaleLocalAddr, yLocalAddr, LoopNum2VF);
            }
        } else {
            ComputeMaxExpcuBLAS(xLocalAddr, maxExpLocalAddr, LoopNum2VF);
            maxExpLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpLocal.GetPhyAddr());
            ComputeScalecuBLASFP4(maxExpLocalAddr, scaleLocalAddr, recipScaleLocalAddr, LoopNumHalfVF, totalBlockNum);
            xLocalAddr = reinterpret_cast<__ubuf__ T*>(xLocal.GetPhyAddr());
            scaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(scaleLocal.GetPhyAddr());
            recipScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(recipScaleLocal.GetPhyAddr());
            if (roundMode_ == MODE_RINT) {
                ComputeDataOptimize<RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(xLocalAddr, recipScaleLocalAddr, yLocalAddr, LoopNum2VF);
            } else if (roundMode_ == MODE_ROUND) {
                ComputeDataOptimize<RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(xLocalAddr, recipScaleLocalAddr, yLocalAddr, LoopNum2VF);
            } else if (roundMode_ == MODE_FLOOR) {
                ComputeData<RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR>(xLocalAddr, recipScaleLocalAddr, yLocalAddr, LoopNum2VF);
            }
        }
    }
    
    inQueue_.FreeTensor(xLocal);
    mxScaleQueue_.EnQue(scaleLocal);
    outQueue_.EnQue(yLocal);
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::FP16Convert(
    Reg::RegTensor<half>& output, Reg::RegTensor<half>& input,
    Reg::MaskReg& mask)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint16_t> specialValueTensor;
        Reg::RegTensor<uint16_t> newMantissa;
        Reg::RegTensor<uint16_t> andResult;
        Reg::RegTensor<uint16_t> newValue;
        Reg::MaskReg specialMask;
        Reg::MaskReg nonzeroMask;
        uint16_t specialValue = SPECIAL_VALUE_E1M2;
        if constexpr (IsSame<U, fp4x2_e2m1_t>::value) {
            specialValue = SPECIAL_VALUE_E2M1;
        }
        Reg::Duplicate(specialValueTensor, specialValue);
        Reg::Duplicate(newMantissa, NEW_MANTISSA);
        Reg::And(andResult, (Reg::RegTensor<uint16_t>&)input, specialValueTensor, mask);
        Reg::CompareScalar<uint16_t, CMPMODE::GT>(nonzeroMask, andResult, 0, mask);
        Reg::CompareScalar<uint16_t, CMPMODE::LT>(specialMask, andResult, NEW_MANTISSA, mask);
        Reg::MaskAnd(specialMask, specialMask, nonzeroMask, mask);
        Reg::Or(newValue, (Reg::RegTensor<uint16_t>&)input, newMantissa, mask);
        Reg::Select<uint16_t>(
            (Reg::RegTensor<uint16_t>&)output, newValue, (Reg::RegTensor<uint16_t>&)input,
            specialMask);
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
template <RoundMode toBf16RoundMode, RoundMode roundMode>
__aicore__ inline void DynamicMxQuantTailAxis<T, U, SCALE_ALG>::ComputeFP4FromHalf(Reg::RegTensor<float>& Reg)
{
    Reg::MaskReg pregAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg zeroMask;
    Reg::MaskReg specialMask;
    Reg::MaskReg negInfMask;

    Reg::RegTensor<int32_t> negZero;
    Reg::RegTensor<int32_t> maxExpFP32;
    Reg::RegTensor<int32_t> exp0FP32;
    Reg::RegTensor<int32_t> exp1FP32;

    Reg::Duplicate(negZero, NEG_ZERO);
    Reg::Compare<int32_t, CMPMODE::EQ>(negInfMask, (Reg::RegTensor<int32_t>&)Reg, negZero, pregAll32);
    if constexpr (IsSameType<U, fp4x2_e1m2_t>::value) {
        Reg::Muls(Reg, Reg, FOUR, pregAll32);
        Reg::CompareScalar<float, CMPMODE::LT>(specialMask, Reg, 0, pregAll32);
        Reg::Truncate<float, roundMode>(Reg, Reg, pregAll32);
        Reg::Muls(Reg, Reg, ONE_FOURTH, pregAll32);
    } else {
        Reg::Duplicate(maxExpFP32, MAX_EXP_FOR_FP32);
        Reg::And(exp0FP32, (Reg::RegTensor<int32_t>&)Reg, maxExpFP32, pregAll32);
        Reg::ShiftRights(exp0FP32, exp0FP32, SHR_NUM_FOR_FP32, pregAll32);
        Reg::Adds(exp0FP32, exp0FP32, FP32_BIAS_NEG, pregAll32);
        Reg::Maxs(exp0FP32, exp0FP32, 0, pregAll32);
        Reg::Adds(exp0FP32, exp0FP32, NEG_ONE, pregAll32);
        Reg::Muls(exp1FP32, exp0FP32, NEG_ONE, pregAll32);
        Reg::Adds(exp1FP32, exp1FP32, FP32_BIAS, pregAll32);
        Reg::ShiftLefts(exp1FP32, exp1FP32, SHR_NUM_FOR_FP32, pregAll32);

        Reg::Mul(Reg, Reg, (Reg::RegTensor<float>&)exp1FP32, pregAll32);
        Reg::Adds(exp0FP32, exp0FP32, FP32_BIAS, pregAll32);
        Reg::ShiftLefts(exp0FP32, exp0FP32, SHR_NUM_FOR_FP32, pregAll32);
        Reg::CompareScalar<float, CMPMODE::LT>(specialMask, Reg, 0, pregAll32);
        Reg::Truncate<float, roundMode>(Reg, Reg, pregAll32);
        Reg::Mul(Reg, Reg, (Reg::RegTensor<float>&)exp0FP32, pregAll32);
    }
    Reg::CompareScalar<float, CMPMODE::EQ>(zeroMask, Reg, 0, pregAll32);
    Reg::MaskAnd(zeroMask, specialMask, zeroMask, pregAll32);
    Reg::MaskOr(zeroMask, negInfMask, zeroMask, pregAll32);
    Reg::Select<int32_t>(
        (Reg::RegTensor<int32_t>&)Reg, negZero, (Reg::RegTensor<int32_t>&)Reg, zeroMask);
}

} // namespace DynamicMxQuant
#endif // DYNAMIC_MX_QUANT_TAIL_AXIS_H
