/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_to_dense_simt.h
 * \brief
 */

#ifndef ASCENDC_SPARSE_TO_DENSE_SIMT_H
#define ASCENDC_SPARSE_TO_DENSE_SIMT_H

#include "kernel_operator.h"
#include "sparse_to_dense_struct.h"

namespace SparseToDense {
using namespace AscendC;

constexpr int64_t USED_THREAD = 1024;
constexpr int64_t DOUBLE_BUFFER = 1;
constexpr int64_t DIMENSION_OFFSET_FROM_LAST = 2;

template <typename IDX_T, typename Y_T, typename COMP_T, bool isScalar>
class SparseToDenseSimt
{
public:
    __aicore__ inline SparseToDenseSimt(TPipe& pipe, const SparseToDenseTilingData& __restrict tilingData)
        : pipe_(pipe), tilingData_(tilingData){};
    __aicore__ inline void Init(
        GM_ADDR indices, GM_ADDR outputShape, GM_ADDR values, GM_ADDR defaultValue, GM_ADDR y);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SetDefaultValue();
    static __aicore__ inline void SparseToDenseSimtCompute(
        COMP_T numValues, COMP_T numDims, COMP_T sparseUsedCoreNum, COMP_T sparseBlockOffset, COMP_T sparseDataNum,
        __gm__ IDX_T* indices, __gm__ IDX_T* outputShape, __gm__ Y_T* values, __gm__ Y_T* y);

private:
    GlobalTensor<IDX_T> indices_;
    GlobalTensor<IDX_T> outputShape_;
    GlobalTensor<Y_T> values_;
    GlobalTensor<Y_T> defaultValue_;
    bool validateIndices_; // 不参与本次迭代
    GlobalTensor<Y_T> y_;
    TPipe& pipe_;
    const SparseToDenseTilingData& tilingData_;
    COMP_T blockIdx_;
    COMP_T loop_;
    COMP_T tailLoopSize_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> outputQue_;
};

template <typename IDX_T, typename Y_T, typename COMP_T, bool isScalar>
__aicore__ inline void SparseToDenseSimt<IDX_T, Y_T, COMP_T, isScalar>::Init(
    GM_ADDR indices, GM_ADDR outputShape, GM_ADDR values, GM_ADDR defaultValue, GM_ADDR y)
{
    blockIdx_ = GetBlockIdx();
    indices_.SetGlobalBuffer((__gm__ IDX_T*)(indices));
    outputShape_.SetGlobalBuffer((__gm__ IDX_T*)(outputShape));
    values_.SetGlobalBuffer((__gm__ Y_T*)(values));
    defaultValue_.SetGlobalBuffer((__gm__ Y_T*)(defaultValue));
    y_.SetGlobalBuffer((__gm__ Y_T*)(y));
    pipe_.InitBuffer(outputQue_, DOUBLE_BUFFER, tilingData_.defaultUbFactor * sizeof(Y_T));
    loop_ = tilingData_.normBlockLoop;
    tailLoopSize_ = tilingData_.normBlockTailLoopSize;  // 尾循环处理的数量
    if (blockIdx_ == tilingData_.defaultValueUsedCoreNum - 1) {  //  尾核
        loop_ = tilingData_.tailBlockLoop;
        tailLoopSize_ = tilingData_.tailBlockTailLoopSize;
    }
}

template <typename IDX_T, typename Y_T, typename COMP_T, bool isScalar>
__aicore__ inline void SparseToDenseSimt<IDX_T, Y_T, COMP_T, isScalar>::Process()
{
    COMP_T numValues = static_cast<COMP_T>(tilingData_.numValues);
    COMP_T numDims = static_cast<COMP_T>(tilingData_.numDims);
    COMP_T defaultValueUsedCoreNum = static_cast<COMP_T>(tilingData_.defaultValueUsedCoreNum);
    if (blockIdx_ >= defaultValueUsedCoreNum) {
        return;
    }
    SetDefaultValue();

    SyncAll();

    COMP_T sparseUsedCoreNum = tilingData_.sparseUsedCoreNum;
    if (blockIdx_ >= sparseUsedCoreNum) {
        return;
    }
    COMP_T normCoreSparses = tilingData_.normCoreHandleSparses;
    COMP_T sparseDataNum = tilingData_.normCoreHandleSparses;
    if (blockIdx_ == sparseUsedCoreNum - 1) {
        sparseDataNum = tilingData_.tailCoreHandleSparses;
    }
    COMP_T sparseBlockOffset = blockIdx_ * normCoreSparses;
    AscendC::Simt::VF_CALL<SparseToDenseSimt<IDX_T, Y_T, COMP_T, isScalar>::SparseToDenseSimtCompute>(
        AscendC::Simt::Dim3(USED_THREAD), numValues, numDims, sparseUsedCoreNum, sparseBlockOffset, sparseDataNum,
        (__gm__ IDX_T*)(indices_.GetPhyAddr()), (__gm__ IDX_T*)(outputShape_.GetPhyAddr()),
        (__gm__ Y_T*)(values_.GetPhyAddr()), (__gm__ Y_T*)(y_.GetPhyAddr()));
}

template <typename IDX_T, typename Y_T, typename COMP_T, bool isScalar>
__aicore__ inline void SparseToDenseSimt<IDX_T, Y_T, COMP_T, isScalar>::SetDefaultValue()
{
    LocalTensor<Y_T> outputLocal = outputQue_.AllocTensor<Y_T>();

    COMP_T defaultDataNumBlock = static_cast<COMP_T>(tilingData_.normCoreHandleDefaultValues); //  正常核处理的总数量
    COMP_T defaultUbFactor = static_cast<COMP_T>(tilingData_.defaultUbFactor);                 //  正常循环处理的数量      
    Y_T defaultValueScalar = static_cast<Y_T>(((__gm__ Y_T*)(defaultValue_.GetPhyAddr()))[0]);
    
    DataCopyExtParams outputCopyParamsNorm{1, static_cast<uint32_t>(defaultUbFactor * sizeof(Y_T)), 0, 0, 0};
    Duplicate(outputLocal, defaultValueScalar, defaultUbFactor);
    outputQue_.EnQue<Y_T>(outputLocal);

    outputLocal = outputQue_.DeQue<Y_T>();
    for (int i = 0; i < loop_ - 1; i++) {
        uint64_t outputOffset = blockIdx_ * defaultDataNumBlock + i * defaultUbFactor;
        DataCopyPad(y_[outputOffset], outputLocal, outputCopyParamsNorm); //  正常循环搬出
    }

    DataCopyExtParams outputCopyParamsTail{1, static_cast<uint32_t>(tailLoopSize_ * sizeof(Y_T)), 0, 0, 0};
    uint64_t outputOffset = blockIdx_ * defaultDataNumBlock + (loop_ - 1) * defaultUbFactor;
    DataCopyPad(y_[outputOffset], outputLocal, outputCopyParamsTail); //  尾循环搬出

    outputQue_.FreeTensor(outputLocal);
}

template <typename IDX_T, typename Y_T, typename COMP_T, bool isScalar>
__simt_vf__ __aicore__
LAUNCH_BOUND(USED_THREAD) inline void SparseToDenseSimt<IDX_T, Y_T, COMP_T, isScalar>::SparseToDenseSimtCompute(
    COMP_T numValues, COMP_T numDims, COMP_T sparseUsedCoreNum, COMP_T sparseBlockOffset, COMP_T sparseDataNum,
    __gm__ IDX_T* indices, __gm__ IDX_T* outputShape, __gm__ Y_T* values, __gm__ Y_T* y)
{
    for (COMP_T idx = sparseBlockOffset + Simt::GetThreadIdx(); idx < sparseBlockOffset + sparseDataNum;
         idx += Simt::GetThreadNum()) {
        COMP_T outputIdx = indices[idx * numDims + numDims - 1]; //  idx索引的坐标的最后一个值
        COMP_T strides = 1;                                      //  每个维度对应的偏移步长
        for (int i = numDims - DIMENSION_OFFSET_FROM_LAST; i >= 0; i--) {
            strides *= outputShape[i + 1];
            outputIdx += indices[idx * numDims + i] * strides; //  循环累加，计算输出索引
        }
        if constexpr (isScalar) {
            y[outputIdx] = values[0];
        } else {
            y[outputIdx] = values[idx];
        }
    }
}
} // namespace SparseToDense

#endif