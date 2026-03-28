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
 * \file dynamic_mx_quant_tail_axis_fp8.h
 * \brief
 */

#ifndef DYNAMIC_MX_QUANT_TAIL_AXIS_FP8_H
#define DYNAMIC_MX_QUANT_TAIL_AXIS_FP8_H
#define FLOAT_OVERFLOW_MODE_CTRL 60
#include "kernel_operator.h"
#include "dynamic_mx_quant_common.h"
namespace DynamicMxQuant {
using namespace AscendC;

template <typename T, typename U, int64_t SCALE_ALG>
class DynamicMxQuantTailAxisFP8 {
public:
    __aicore__ inline DynamicMxQuantTailAxisFP8()
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
    __aicore__ inline void ComputeMaxExpOCP(__ubuf__ T* xLocalAddr, __ubuf__ uint16_t* maxExpAddr, uint16_t LoopNum2VF);
    __aicore__ inline void ComputeScaleOCP(__ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr, uint16_t LoopNum1VF, uint32_t totalScaleInUB);
    __aicore__ inline void ComputeMaxExpcuBLAS(__ubuf__ T* xLocalAddr, __ubuf__ uint16_t* maxExpAddr, uint16_t LoopNum2VF);
    __aicore__ inline void ComputeScalecuBLAS(__ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr, uint16_t LoopNumHalfVF, uint32_t totalScaleInUB);
    __aicore__ inline void ComputeData(__ubuf__ T* xLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr, __ubuf__ int8_t* yLocalAddr, uint16_t LoopNum2VF);

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

    uint16_t f8Emax_ = 0;
    uint32_t dtypeMax = 0;
};

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxisFP8<T, U, SCALE_ALG>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, const DynamicMxQuantTailAxisTilingData* tilingData)
{
#if (__NPU_ARCH__ == 3510)
    SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    ParseTilingData(tilingData);                // 获取TilingData数据

    GetGmParams();                              // 计算核间GM地址偏移

    GetUbParams();                              // 计算核内切分参数

    pipe_.InitBuffer(inQueue_, DB_BUFFER, maxUbBlockNum_ * blockSize_ * sizeof(T));
    pipe_.InitBuffer(outQueue_, DB_BUFFER, maxUbBlockNum_ * blockSize_ * sizeof(uint8_t));
    pipe_.InitBuffer(mxScaleQueue_, DB_BUFFER, maxUbBlockNum_ * sizeof(uint8_t));
    pipe_.InitBuffer(maxExpBuffer_, maxUbBlockNum_ * sizeof(uint16_t));
    pipe_.InitBuffer(recipScaleBuffer_, maxUbBlockNum_ * sizeof(uint16_t));

    xGm_.SetGlobalBuffer((__gm__ T*)x + xGmOffset_);
    yGm_.SetGlobalBuffer((__gm__ uint8_t*)y + xGmOffset_);
    scaleGm_.SetGlobalBuffer((__gm__ uint8_t*)mxScale + scaleGmOffet_);

    if constexpr (IsSame<U, fp8_e4m3fn_t>::value) {
        f8Emax_ = FP8_E4M3_MAX_EXP;
        dtypeMax = FP8_E4M3_MAX;
    } else {
        f8Emax_ = FP8_E5M2_MAX_EXP;
        dtypeMax = FP8_E5M2_MAX;
    }
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxisFP8<T, U, SCALE_ALG>::GetGmParams(){
    coreIdx_ = GetBlockIdx();
    coreColIdx_ = coreIdx_ % colTileNum_;
    coreRowIdx_ = coreIdx_ / colTileNum_;
    xGmOffset_ = coreRowIdx_ * rowNormalBlockNum_ * DIGIT_ONE * colNum_ + coreColIdx_ * colNormalBlockNum_ * DIGIT_EIGHT * blockSize_;
    scaleColNum_ = ops::CeilDiv(ops::CeilDiv(colNum_, blockSize_), DIGIT_TWO) * DIGIT_TWO;
    scaleGmOffet_ = coreRowIdx_ * rowNormalBlockNum_ * DIGIT_ONE * scaleColNum_ + coreColIdx_ * colNormalBlockNum_ * DIGIT_EIGHT;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxisFP8<T, U, SCALE_ALG>::GetUbParams(){
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
__aicore__ inline void DynamicMxQuantTailAxisFP8<T, U, SCALE_ALG>::ParseTilingData(const DynamicMxQuantTailAxisTilingData* tilingData)
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
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxisFP8<T, U, SCALE_ALG>::Process()
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
__aicore__ inline void DynamicMxQuantTailAxisFP8<T, U, SCALE_ALG>::CopyIn(int64_t dim0LoopIdx, int64_t dim1LoopIdx, int64_t ubFactorRowNum, int64_t ubFactorColNum, int64_t ubFactorColBlockNum)
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
__aicore__ inline void DynamicMxQuantTailAxisFP8<T, U, SCALE_ALG>::CopyOut(int64_t dim0LoopIdx, int64_t dim1LoopIdx, int64_t ubFactorRowNum, int64_t ubFactorColNum, int64_t ubFactorColBlockNum)
{
    LocalTensor<uint8_t> scaleLocal = mxScaleQueue_.DeQue<uint8_t>();
    LocalTensor<uint8_t> yLocal = outQueue_.DeQue<uint8_t>();
    int64_t scaleIsNotOdd = ubFactorColBlockNum % DIGIT_TWO;

    DataCopyExtParams copyOutParamY = {
        static_cast<uint16_t>(ubFactorRowNum), static_cast<uint32_t>(ubFactorColNum * sizeof(uint8_t)), static_cast<uint32_t>(scaleIsNotOdd), // 一个BlockSize=32，输出y为1 Bytes，搬入UB需要跳32Bytes，srcStride = 1
        static_cast<uint32_t>((colNum_ - ubFactorColNum) * sizeof(uint8_t)), static_cast<uint32_t>(0)};

    int64_t offset = dim0LoopIdx * ubFactorRowNormalBlockNum_ * colNum_ + dim1LoopIdx * ubFactorColNormalBlockLen_;
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
__aicore__ inline void DynamicMxQuantTailAxisFP8<T, U, SCALE_ALG>::ComputeMaxExpOCP(__ubuf__ T* xLocalAddr, __ubuf__ uint16_t* maxExpAddr, uint16_t LoopNum2VF)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vdExp0;
        Reg::RegTensor<T> vdExp1;
        Reg::RegTensor<uint16_t> vdMaxExp;
        Reg::RegTensor<uint16_t> vdExpSelect0;
        Reg::RegTensor<uint16_t> vdExpSelect1;
        Reg::RegTensor<uint16_t> vdExpExtract0;
        Reg::RegTensor<uint16_t> vdExpExtract1;
        Reg::RegTensor<bfloat16_t> vdExp0BF16;
        Reg::RegTensor<bfloat16_t> vdExp1BF16;

        Reg::RegTensor<uint16_t> expMaskBF16;
        Reg::Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);
        Reg::RegTensor<uint16_t> invalidMaskFP16;
        Reg::Duplicate(invalidMaskFP16, INVALID_FLOAT16);

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
                    vdExpSelect0, (Reg::RegTensor<uint16_t>&)vdExp0, invalidMaskFP16, Mask);
                Reg::And(
                    vdExpSelect1, (Reg::RegTensor<uint16_t>&)vdExp1, invalidMaskFP16, Mask);
                Reg::Compare<uint16_t, CMPMODE::NE>(
                    invalidDataMask0, vdExpSelect0, invalidMaskFP16, Mask);
                Reg::Compare<uint16_t, CMPMODE::NE>(
                    invalidDataMask1, vdExpSelect1, invalidMaskFP16, Mask);
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
__aicore__ inline void DynamicMxQuantTailAxisFP8<T, U, SCALE_ALG>::ComputeMaxExpcuBLAS(__ubuf__ T* xLocalAddr, __ubuf__ uint16_t* maxExpAddr, uint16_t LoopNum2VF)
{
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
            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, xLocalAddr, vfLen16Double);
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
__aicore__ inline void DynamicMxQuantTailAxisFP8<T, U, SCALE_ALG>::ComputeScaleOCP(__ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr, uint16_t LoopNum1VF, uint32_t totalScaleInUB)
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
        Reg::Duplicate(maxExpValue, f8Emax_);
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
        Reg::MaskReg cmpResultSub;
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
__aicore__ inline void DynamicMxQuantTailAxisFP8<T, U, SCALE_ALG>::ComputeScalecuBLAS(
    __ubuf__ uint16_t* maxExpAddr, __ubuf__ uint16_t* mxScaleLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr, uint16_t LoopNumHalfVF, uint32_t totalScaleInUB)
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

        Reg::RegTensor<uint32_t> invMax;
        Reg::Duplicate(invMax, dtypeMax);
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
        Reg::RegTensor<uint32_t> fp8NanRegTensor;
        Reg::Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8_IN_FP32);

        Reg::MaskReg cmpResult;
        Reg::MaskReg zeroMask;
        Reg::MaskReg p0;
        Reg::MaskReg p1;
        Reg::MaskReg p2;
        // Reg::MaskReg maskHalf = Reg::CreateMask<uint16_t>();
        uint32_t SixtyFour = 64;
        Reg::MaskReg dataMaskB16Half = Reg::UpdateMask<uint16_t>(SixtyFour);
        Reg::MaskReg maskFloat = Reg::CreateMask<uint32_t>();

        static constexpr Reg::CastTrait castTraitHalf2Float = {
            Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
            Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        for (uint16_t i = 0; i < LoopNumHalfVF; i++) {
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_UNPACK_B16>(max16, maxExpAddr, vfLen32);         // 单搬 64 个数

            Reg::Cast<float, T, castTraitHalf2Float>(
                (Reg::RegTensor<float>&)max32, (Reg::RegTensor<T>&)max16, maskFloat);
            Reg::Compare<uint32_t, CMPMODE::LT>(cmpResult, max32, expMask, maskFloat);
            Reg::Compare<uint32_t, CMPMODE::NE>(zeroMask, max32, zeroRegTensor32, maskFloat);

            Reg::Mul(
                (Reg::RegTensor<float>&)max32, (Reg::RegTensor<float>&)max32,
                (Reg::RegTensor<float>&)invMax, maskFloat);
            Reg::ShiftRights(exp32, max32, SHR_NUM_FOR_FP32, maskFloat);
            Reg::And(man32, max32, manMaskFP32, maskFloat);

            Reg::CompareScalar<uint32_t, CMPMODE::GT>(p0, exp32, NUMBER_ZERO, maskFloat);
            Reg::CompareScalar<uint32_t, CMPMODE::LT>(p1, exp32, NUMBER_TWO_FIVE_FOUR, maskFloat);
            Reg::CompareScalar<uint32_t, CMPMODE::GT>(p2, man32, NUMBER_ZERO, maskFloat);
            Reg::MaskAnd(p0, p0, p1, maskFloat);
            Reg::MaskAnd(p0, p0, p2, maskFloat);

            Reg::CompareScalar<uint32_t, CMPMODE::EQ>(p1, exp32, NUMBER_ZERO, maskFloat);
            Reg::CompareScalar<uint32_t, CMPMODE::GT>(p2, man32, NUMBER_HALF, maskFloat);
            Reg::MaskAnd(p1, p1, p2, maskFloat);
            Reg::MaskOr(p0, p0, p1, maskFloat);

            Reg::Adds(expAddOne32, exp32, 1, maskFloat);
            Reg::Select(extractExp, expAddOne32, exp32, p0);
            Reg::Select<uint32_t>(extractExp, extractExp, fp8NanRegTensor, cmpResult);
            Reg::Select<uint32_t>(extractExp, extractExp, zeroRegTensor32, zeroMask);
            Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(expOut, extractExp);

            Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_PACK_B16>(
                mxScaleLocalAddr + i * 32, expOut, dataMaskB16Half);               // 64 个 max 计算得到 64 * 1 Bytes = 32 * sizeof(uint16_t) Bytes

            Reg::ShiftLefts(extractExp, extractExp, SHR_NUM_FOR_BF16, maskFloat);
            Reg::Sub(halfScale, scaleBias, extractExp, maskFloat);
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
__aicore__ inline void DynamicMxQuantTailAxisFP8<T, U, SCALE_ALG>::ComputeData(
    __ubuf__ T* xLocalAddr, __ubuf__ uint16_t* recipScaleLocalAddr, __ubuf__ int8_t* yLocalAddr, uint16_t LoopNum2VF)
{
    __VEC_SCOPE__
    {
        Reg::MaskReg dataMask1 = Reg::CreateMask<T>();
        Reg::MaskReg dataMask2 = Reg::CreateMask<T>();
        Reg::MaskReg dataMask3 = Reg::CreateMask<T>();
        Reg::MaskReg dataMask4 = Reg::CreateMask<T>();
        Reg::MaskReg dataMask5 = Reg::CreateMask<U>();
        Reg::MaskReg maskAll =
            Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();

        Reg::RegTensor<uint16_t> halfScaleForMul;
        Reg::RegTensor<float> floatScaleForMul;
        Reg::RegTensor<T> vdExp0;
        Reg::RegTensor<T> vdExp1;
        Reg::RegTensor<T> vdExp0Convert;
        Reg::RegTensor<T> vdExp1Convert;
        Reg::RegTensor<bfloat16_t> vdExp0BF16;
        Reg::RegTensor<bfloat16_t> vdExp1BF16;
        Reg::RegTensor<float> vdExp0FP32Zero;
        Reg::RegTensor<float> vdExp0FP32One;
        Reg::RegTensor<float> vdExp1FP32Zero;
        Reg::RegTensor<float> vdExp1FP32One;
        Reg::RegTensor<U> vdExp0FP8Zero;
        Reg::RegTensor<U> vdExp0FP8One;
        Reg::RegTensor<U> vdExp1FP8Zero;
        Reg::RegTensor<U> vdExp1FP8One;
        Reg::RegTensor<bfloat16_t> vdBF16Exp0FP4;
        Reg::RegTensor<bfloat16_t> vdBF16Exp1FP4;

        static constexpr Reg::CastTrait castTrait = {
            Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
            Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr Reg::CastTrait castTraitHalf2Bf16 = {
            Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN,
            Reg::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
        static constexpr Reg::CastTrait castTraitZero = {
            Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
            Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTraitOne = {
            Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN,
            Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
        static constexpr Reg::CastTrait castTrait32to8 = {
            Reg::RegLayout::ZERO, Reg::SatMode::SAT,
            Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr Reg::CastTrait castTrait32to80 = {
            Reg::RegLayout::ZERO, Reg::SatMode::SAT,
            Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr Reg::CastTrait castTrait32to81 = {
            Reg::RegLayout::ONE, Reg::SatMode::SAT,
            Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr Reg::CastTrait castTrait32to82 = {
            Reg::RegLayout::TWO, Reg::SatMode::SAT,
            Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        static constexpr Reg::CastTrait castTrait32to83 = {
            Reg::RegLayout::THREE, Reg::SatMode::SAT,
            Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        
        for (uint16_t i = 0; i < LoopNum2VF; i++) {
            Reg::LoadAlign<
                T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                vdExp0, vdExp1, xLocalAddr, vfLen16Double);
            Reg::LoadAlign<
                uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
                halfScaleForMul, recipScaleLocalAddr, elementAfterReduce_);
            if constexpr (IsSame<T, half>::value) {
                Reg::Cast<float, T, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMask1);
                Reg::Cast<float, T, castTraitOne>(vdExp0FP32One, vdExp0, dataMask1);
                Reg::Cast<float, bfloat16_t, castTraitZero>(
                    floatScaleForMul, (Reg::RegTensor<bfloat16_t>&)halfScaleForMul, maskAll);
                Reg::Mul(vdExp0FP32Zero, vdExp0FP32Zero, floatScaleForMul, dataMask3);
                Reg::Mul(vdExp0FP32One, vdExp0FP32One, floatScaleForMul, dataMask4);
            
                Reg::Cast<float, T, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMask1);
                Reg::Cast<float, T, castTraitOne>(vdExp1FP32One, vdExp1, dataMask1);
                Reg::Mul(vdExp1FP32Zero, vdExp1FP32Zero, floatScaleForMul, dataMask3);
                Reg::Mul(vdExp1FP32One, vdExp1FP32One, floatScaleForMul, dataMask4);
            
            } else {
                Reg::Mul(vdExp0, vdExp0, (Reg::RegTensor<T>&)halfScaleForMul, dataMask1);
                Reg::Mul(vdExp1, vdExp1, (Reg::RegTensor<T>&)halfScaleForMul, dataMask1);
              
                Reg::Cast<float, T, castTraitZero>(vdExp0FP32Zero, vdExp0, dataMask1);
                Reg::Cast<float, T, castTraitOne>(vdExp0FP32One, vdExp0, dataMask1);
                Reg::Cast<float, T, castTraitZero>(vdExp1FP32Zero, vdExp1, dataMask2);
                Reg::Cast<float, T, castTraitOne>(vdExp1FP32One, vdExp1, dataMask2);
              
            }  
            Reg::Cast<U, float, castTrait32to80>(vdExp0FP8Zero, vdExp0FP32Zero, dataMask3);
            Reg::Cast<U, float, castTrait32to82>(vdExp0FP8One, vdExp0FP32One, dataMask3);
            Reg::Cast<U, float, castTrait32to81>(vdExp1FP8Zero, vdExp1FP32Zero, dataMask4);
            Reg::Cast<U, float, castTrait32to83>(vdExp1FP8One, vdExp1FP32One, dataMask4);
        
            Reg::Add((Reg::RegTensor<uint8_t>&)vdExp0FP8Zero, (Reg::RegTensor<uint8_t>&)vdExp0FP8Zero, (Reg::RegTensor<uint8_t>&)vdExp0FP8One, dataMask5);
            Reg::Add((Reg::RegTensor<uint8_t>&)vdExp1FP8Zero, (Reg::RegTensor<uint8_t>&)vdExp1FP8Zero, (Reg::RegTensor<uint8_t>&)vdExp1FP8One, dataMask5);
            Reg::Add((Reg::RegTensor<uint8_t>&)vdExp0FP8Zero, (Reg::RegTensor<uint8_t>&)vdExp0FP8Zero, (Reg::RegTensor<uint8_t>&)vdExp1FP8Zero, dataMask5);

            Reg::StoreAlign<int8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_NORM_B8>(
                yLocalAddr, (Reg::RegTensor<int8_t>&)vdExp0FP8Zero, vfLen16Double, dataMask5);

        }
    }
    return;
}

template <typename T, typename U, int64_t SCALE_ALG>
__aicore__ inline void DynamicMxQuantTailAxisFP8<T, U, SCALE_ALG>::Compute(int64_t ubFactorRowBlockNum, int64_t ubFactorColBlockNum)
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

    if constexpr (SCALE_ALG == 0) {
        ComputeMaxExpOCP(xLocalAddr, maxExpLocalAddr, LoopNum2VF);
        maxExpLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpLocal.GetPhyAddr());
        ComputeScaleOCP(maxExpLocalAddr, scaleLocalAddr, recipScaleLocalAddr, LoopNum1VF, totalBlockNum);
    } else if constexpr (SCALE_ALG == 1) {
        ComputeMaxExpcuBLAS(xLocalAddr, maxExpLocalAddr, LoopNum2VF);
        maxExpLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpLocal.GetPhyAddr());
        ComputeScalecuBLAS(maxExpLocalAddr, scaleLocalAddr, recipScaleLocalAddr, LoopNumHalfVF, totalBlockNum);
    }
    xLocalAddr = reinterpret_cast<__ubuf__ T*>(xLocal.GetPhyAddr());
    recipScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(recipScaleLocal.GetPhyAddr());
    ComputeData(xLocalAddr, recipScaleLocalAddr, yLocalAddr, LoopNum2VF);

    inQueue_.FreeTensor(xLocal);
    mxScaleQueue_.EnQue(scaleLocal);
    outQueue_.EnQue(yLocal);
    return;
}

} // namespace DynamicMxQuant
#endif // DYNAMIC_MX_QUANT_TAIL_AXIS_FP8_H
