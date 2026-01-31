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
 * \file dequant_swiglu_quant_static_not_full.h
 * \brief
 */

#ifndef DEQUANT_SWIGLU_QUANT_STATIC_NOT_FULL_H
#define DEQUANT_SWIGLU_QUANT_STATIC_NOT_FULL_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "dequant_swiglu_quant_common.h"


namespace DequantSwigluQuantV35Ops {
using namespace AscendC;

template <typename TXtype, typename TActScale, typename TBias, typename TQuantScale, typename TQuantOffset, typename TGroup, typename TYtype>
class DequantSwigluQuantStaticNotFull {
public:
    static constexpr bool hasWeightScale_ = IsSameType<TXtype, int32_t>::value;
    static constexpr bool hasActScale_ = IsSameType<TActScale, float>::value;
    static constexpr bool hasQuantScale_ = IsSameType<TQuantScale, float>::value;
    static constexpr bool hasQuantOffset_ = IsSameType<TQuantOffset, float>::value;
    static constexpr bool hasGroupIndex_ = IsSameType<TGroup, int64_t>::value || IsSameType<TGroup, int32_t>::value;
    // bias标记 bias可支持float，float16，bf16，int32
    static constexpr bool hasBiasIndex_ = IsSameType<TBias, float>::value || IsSameType<TBias, half>::value || IsSameType<TBias, bfloat16_t>::value || IsSameType<TBias, int32_t>::value;
    // x数据类型为int32标记
    static constexpr bool ifXIntIndex_ = IsSameType<TXtype, int32_t>::value;
    // x数据类型为bf16标记
    static constexpr bool ifXBf16Index_ = IsSameType<TXtype, bfloat16_t>::value;
    // x数据类型为float16标记
    static constexpr bool ifXFloat16Index_ = IsSameType<TXtype, half>::value;
    // y数据类型为int8标记
    static constexpr bool ifYInt8Index_ = IsSameType<TYtype, int8_t>::value;
    // y数据类型为float8_e4m3标记
    static constexpr bool ifYFloat8e4m3Index_ = IsSameType<TYtype, fp8_e4m3fn_t>::value;
    // y数据类型为float8_e5m2标记
    static constexpr bool ifYFloat8e5m2Index_ = IsSameType<TYtype, fp8_e5m2_t>::value;
    // y数据类型为float4_e2m1标记
    static constexpr bool ifYFloat4e2m1Index_ = IsSameType<TYtype, fp4x2_e2m1_t>::value;
    // y数据类型为float4_e1m2标记
    static constexpr bool ifYFloat4e1m2Index_ = IsSameType<TYtype, fp4x2_e1m2_t>::value;
    // y数据类型为hifloat8标记
    static constexpr bool ifYHiFloat8Index_ = IsSameType<TYtype, hifloat8_t>::value;

    __aicore__ inline DequantSwigluQuantStaticNotFull(TPipe* pipe) {
        pipe_ = pipe;
    };
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weightScale, GM_ADDR activationScale, GM_ADDR bias, GM_ADDR quantScale,
                                GM_ADDR quantOffset, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR scale,
                                const DequantSwigluQuantV35BaseTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessSingleRow(int64_t groupIdx, int64_t rowIdx);
    template <typename copyDtype>
    __aicore__ inline void CopyIn2HPerRowForSwiGluV1(LocalTensor<copyDtype> dstTensor, GlobalTensor<copyDtype> srcTensor, int64_t rowIdx, int64_t ubloop, int64_t processNum);
    template <typename copyDtype>
    __aicore__ inline void CopyIn2HPerRowForSwiGluV2(LocalTensor<copyDtype> dstTensor, GlobalTensor<copyDtype> srcTensor, int64_t rowIdx, int64_t ubloop, int64_t processNum);
    template <typename copyDtype>
    __aicore__ inline void CopyIn2HPerGroupForSwiGluV1(LocalTensor<copyDtype> dstTensor, GlobalTensor<copyDtype> srcTensor, int64_t groupIdx, int64_t ubloop, int64_t processNum);
    template <typename copyDtype>
    __aicore__ inline void CopyIn2HPerGroupForSwiGluV2(LocalTensor<copyDtype> dstTensor, GlobalTensor<copyDtype> srcTensor, int64_t groupIdx, int64_t ubloop, int64_t processNum);
    template <typename copyDtype>
    __aicore__ inline void CopyInHPerGroup(LocalTensor<copyDtype> dstTensor, GlobalTensor<copyDtype> srcTensor, int64_t groupIdx, int64_t ubloop, int64_t processNum);
    template <typename copyDtype>
    __aicore__ inline void CopyInOnePerGroup(LocalTensor<copyDtype> dstTensor, GlobalTensor<copyDtype> srcTensor, int64_t groupIdx, int64_t ubloop, int64_t processNum);

protected:
    /* global memory address */
    // input global mem
    GlobalTensor<TXtype> xGm_;
    GlobalTensor<float> weightScaleGm_;
    GlobalTensor<float> activationScaleGm_;
    GlobalTensor<int32_t> biasInt32Gm_;
    GlobalTensor<bfloat16_t> biasBf16Gm_;
    GlobalTensor<half> biasFp16Gm_;
    GlobalTensor<float> biasFp32Gm_;
    GlobalTensor<float> quantScaleGm_;
    GlobalTensor<float> quantOffsetGm_;
    GlobalTensor<TGroup> groupIndexGm_; // GroupIndex的类型如何获取？

    // output global mem
    GlobalTensor<TYtype> yGm_;
    GlobalTensor<float> scaleGm_;

    // input global mem
    // x
    LocalTensor<TXtype> xLocal;
    __local_mem__ TXtype* xPtr;
    __local_mem__ TXtype* x2Ptr;
    // weightScale
    LocalTensor<float> weightScaleLocal;
    __local_mem__ float* wScalePtr;
    __local_mem__ float* wScale2Ptr;
    // activationScale
    LocalTensor<float> actScaleLocal;
    __local_mem__ float* aScalePtr;
    // bias区分不同类型
    LocalTensor<int32_t> biasInt32Local;
    __local_mem__ int32_t* biasInt32Ptr;
    __local_mem__ int32_t* bias2Int32Ptr;
    LocalTensor<bfloat16_t> biasBf16Local;
    __local_mem__ bfloat16_t* biasBf16Ptr;
    __local_mem__ bfloat16_t* bias2Bf16Ptr;
    LocalTensor<half> biasFp16Local;
    __local_mem__ half* biasFp16Ptr;
    __local_mem__ half* bias2Fp16Ptr;
    LocalTensor<float> biasLocal;
    __local_mem__ float* biasPtr;
    __local_mem__ float* bias2Ptr;
    //quant_scale
    LocalTensor<float> quantScaleLocal;
    __local_mem__ float* qScalePtr;
    //quant_offset
    LocalTensor<float> quantOffsetLocal;
    __local_mem__ float* qOffsetPtr;

    /* ascendc variable */
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, 1> xQueue_;
    TQue<QuePosition::VECIN, 1> weightScaleQueue_;
    TQue<QuePosition::VECIN, 1> actScaleQueue_;
    TQue<QuePosition::VECIN, 1> biasQueue_;
    TQue<QuePosition::VECIN, 1> quantScaleQueue_;
    TQue<QuePosition::VECIN, 1> quantOffsetQueue_;
    TQue<QuePosition::VECOUT, 1> yQueue_;

    TBuf<> tmpBuffer;

    uint32_t blockIdx_ = GetBlockIdx();
    uint32_t realCoreDim_ = 0;
    int64_t realDimx_ = 0;
    int64_t groupOffset_ = 0;
    int64_t roundMode_ = 0; // 溢出模式的标识
    int64_t usedCoreNum_ = 0;
    // 获取quant_scale/quant_offset尾轴是否为单个元素
    bool ifQuantIsOne_ = true;
    // bias数据类型为int32标记
    bool ifBiasIntIndex_ = false;
    // bias数据类型为float标记
    bool ifBiasFloatIndex_ = false;
    // bias数据类型为float16标记
    bool ifBiasFloat16Index_ = false;
    // bias数据类型为bfloat16标记
    bool ifBiasBfloat16Index_ = false;

    const DequantSwigluQuantV35BaseTilingData* tl_ = nullptr;
};

template <typename TXtype, typename TActScale, typename TBias, typename TQuantScale, typename TQuantOffset, typename TGroup, typename TYtype>
__aicore__ inline void DequantSwigluQuantStaticNotFull<TXtype, TActScale, TBias, TQuantScale, TQuantOffset, TGroup, TYtype>::Init(
    GM_ADDR x, GM_ADDR weightScale, GM_ADDR activationScale, GM_ADDR bias, GM_ADDR quantScale, GM_ADDR quantOffset,
    GM_ADDR groupIndex, GM_ADDR y, GM_ADDR scale, const DequantSwigluQuantV35BaseTilingData* tilingData)
{
    tl_ = tilingData;
    // 获取quant_scale/quant_offset尾轴是否为单个元素
    ifQuantIsOne_ = tl_->quantIsOne == static_cast<int64_t>(1) ? true : false;
    // bias数据类型为int32标记
    ifBiasIntIndex_ = tl_->biasMode == static_cast<int64_t>(1) ? true : false;
    // bias数据类型为bfloat16标记
    ifBiasBfloat16Index_ = tl_->biasMode == static_cast<int64_t>(2) ? true : false;
    // bias数据类型为float16标记
    ifBiasFloat16Index_ = tl_->biasMode == static_cast<int64_t>(3) ? true : false;
    // bias数据类型为float标记
    ifBiasFloatIndex_ = tl_->biasMode == static_cast<int64_t>(4) ? true : false;

    xGm_.SetGlobalBuffer((__gm__ TXtype*)x);
    pipe_->InitBuffer(xQueue_, DOUBLE_BUFFER, (tl_->UbFactorDimy * 2) * sizeof(TXtype));
    if constexpr (hasWeightScale_) {
        weightScaleGm_.SetGlobalBuffer((__gm__ float*)weightScale);
        pipe_->InitBuffer(weightScaleQueue_, DOUBLE_BUFFER, (tl_->UbFactorDimy * 2) * sizeof(float));
    }
    if constexpr (hasActScale_) {
        activationScaleGm_.SetGlobalBuffer((__gm__ float*)activationScale);
        pipe_->InitBuffer(actScaleQueue_, DOUBLE_BUFFER, BLOCK_SIZE);
    }
    if constexpr (hasBiasIndex_) {
        if (ifBiasIntIndex_) {
            biasInt32Gm_.SetGlobalBuffer((__gm__ int32_t*)bias);
            pipe_->InitBuffer(biasQueue_, DOUBLE_BUFFER, (tl_->UbFactorDimy * 2) * sizeof(int32_t));
        } else if (ifBiasBfloat16Index_) {
            biasBf16Gm_.SetGlobalBuffer((__gm__ bfloat16_t*)bias);
            pipe_->InitBuffer(biasQueue_, DOUBLE_BUFFER, (tl_->UbFactorDimy * 2) * sizeof(bfloat16_t));
        } else if (ifBiasFloat16Index_) {
            biasFp16Gm_.SetGlobalBuffer((__gm__ half*)bias);
            pipe_->InitBuffer(biasQueue_, DOUBLE_BUFFER, (tl_->UbFactorDimy * 2) * sizeof(half));
        } else if (ifBiasFloatIndex_) {
            biasFp32Gm_.SetGlobalBuffer((__gm__ float*)bias);
            pipe_->InitBuffer(biasQueue_, DOUBLE_BUFFER, (tl_->UbFactorDimy * 2) * sizeof(float));
        }
    }
    if constexpr (hasQuantScale_) {
        quantScaleGm_.SetGlobalBuffer((__gm__ float*)quantScale);
        pipe_->InitBuffer(quantScaleQueue_, DOUBLE_BUFFER, tl_->UbFactorDimy * sizeof(float));
    }
    if constexpr (hasQuantOffset_) {
        quantOffsetGm_.SetGlobalBuffer((__gm__ float*)quantOffset);
        pipe_->InitBuffer(quantOffsetQueue_, DOUBLE_BUFFER, tl_->UbFactorDimy * sizeof(float));
    }
    if constexpr (hasGroupIndex_) {
        groupIndexGm_.SetGlobalBuffer((__gm__ TGroup*)groupIndex);
    }

    yGm_.SetGlobalBuffer((__gm__ TYtype*)y);
    pipe_->InitBuffer(yQueue_, DOUBLE_BUFFER, tl_->UbFactorDimy * sizeof(TYtype));

    scaleGm_.SetGlobalBuffer((__gm__ float*)scale);

    pipe_->InitBuffer(tmpBuffer, 2 * tl_->UbFactorDimy * sizeof(float));
}

template <typename TXtype, typename TActScale, typename TBias, typename TQuantScale, typename TQuantOffset, typename TGroup, typename TYtype>
__aicore__ inline void DequantSwigluQuantStaticNotFull<TXtype, TActScale, TBias, TQuantScale, TQuantOffset, TGroup, TYtype>::Process()
{
    groupOffset_ = 0;
    int64_t realGroupNums = 1;
    if constexpr (hasGroupIndex_) {
        realGroupNums = tl_->inGroupNum;
    }
    for (int64_t groupIndex = 0; groupIndex < realGroupNums; groupIndex++) {
        if constexpr (!hasGroupIndex_) {
            realDimx_ = tl_->inDimx;
        }
        if (tl_->groupIndexMode == 2) {
            realDimx_ = groupIndexGm_.template ReinterpretCast<int64_t>()(groupIndex);
        } else if (tl_->groupIndexMode == 1) {
            realDimx_ = static_cast<int64_t>(groupIndexGm_(groupIndex));
        }
        realCoreDim_ = Std::min(tl_->usedCoreNum, realDimx_);
        if (realDimx_ < 0) {
            realDimx_ = 0;
        }
        // group内的值累加超过inDimx会出现越界访问,tiling侧无法校验,通过groupOffset_ + rowIndex < tl_->inDimx保证
        for (int64_t rowIndex = blockIdx_; rowIndex < realDimx_ && groupOffset_ + rowIndex < tl_->inDimx; rowIndex += tl_->usedCoreNum) {
            ProcessSingleRow(groupIndex, groupOffset_ + rowIndex);
        }
        groupOffset_ += realDimx_;
    }
}

template <typename TXtype, typename TActScale, typename TBias, typename TQuantScale, typename TQuantOffset, typename TGroup, typename TYtype>
template <typename copyDtype>
__aicore__ inline void DequantSwigluQuantStaticNotFull<TXtype, TActScale, TBias, TQuantScale, TQuantOffset, TGroup, TYtype>::CopyIn2HPerRowForSwiGluV1(LocalTensor<copyDtype> dstTensor, GlobalTensor<copyDtype> srcTensor, int64_t rowIdx, int64_t ubloop, int64_t processNum)
{
    int64_t rowOffsetFor2H = rowIdx * tl_->inDimy;
    int64_t actOffset = tl_->actRight * tl_->outDimy;
    int64_t gateOffset = tl_->outDimy - actOffset;
    DataCopyParams dataCopyPerRowForSwiGluV1Params;
    DataCopyPadParams padParams{false, 0, 0, 0};
    dataCopyPerRowForSwiGluV1Params.blockCount = 1;
    dataCopyPerRowForSwiGluV1Params.blockLen = processNum * sizeof(copyDtype);
    dataCopyPerRowForSwiGluV1Params.srcStride = 0;
    dataCopyPerRowForSwiGluV1Params.dstStride = 0;
    // copy_in: (BS, H) 激活部分
    DataCopyPad(dstTensor[0], srcTensor[rowOffsetFor2H + ubloop * tl_->UbFactorDimy + actOffset], dataCopyPerRowForSwiGluV1Params, padParams);
    // copy_in: (BS, H) 门控部分
    DataCopyPad(dstTensor[tl_->UbFactorDimy], srcTensor[rowOffsetFor2H + ubloop * tl_->UbFactorDimy + gateOffset], dataCopyPerRowForSwiGluV1Params, padParams);
}

template <typename TXtype, typename TActScale, typename TBias, typename TQuantScale, typename TQuantOffset, typename TGroup, typename TYtype>
template <typename copyDtype>
__aicore__ inline void DequantSwigluQuantStaticNotFull<TXtype, TActScale, TBias, TQuantScale, TQuantOffset, TGroup, TYtype>::CopyIn2HPerRowForSwiGluV2(LocalTensor<copyDtype> dstTensor, GlobalTensor<copyDtype> srcTensor, int64_t rowIdx, int64_t ubloop, int64_t processNum)
{
    int64_t rowOffsetFor2H = rowIdx * tl_->inDimy;
    DataCopyParams dataCopy2HPerRowForSwiGluV2Params;
    DataCopyPadParams padParams{false, 0, 0, 0};
    dataCopy2HPerRowForSwiGluV2Params.blockCount = 1;
    dataCopy2HPerRowForSwiGluV2Params.blockLen = 2 * processNum * sizeof(copyDtype);
    dataCopy2HPerRowForSwiGluV2Params.srcStride = 0;
    dataCopy2HPerRowForSwiGluV2Params.dstStride = 0;
    // copy_in: (BS, 2H)
    DataCopyPad(dstTensor[0], srcTensor[rowOffsetFor2H + ubloop * 2 * tl_->UbFactorDimy], dataCopy2HPerRowForSwiGluV2Params, padParams);
}

template <typename TXtype, typename TActScale, typename TBias, typename TQuantScale, typename TQuantOffset, typename TGroup, typename TYtype>
template <typename copyDtype>
__aicore__ inline void DequantSwigluQuantStaticNotFull<TXtype, TActScale, TBias, TQuantScale, TQuantOffset, TGroup, TYtype>::CopyIn2HPerGroupForSwiGluV1(LocalTensor<copyDtype> dstTensor, GlobalTensor<copyDtype> srcTensor, int64_t groupIdx, int64_t ubloop, int64_t processNum)
{
    int64_t groupOffsetFor2H = groupIdx * tl_->inDimy;
    int64_t actOffset = tl_->actRight * tl_->outDimy;
    int64_t gateOffset = tl_->outDimy - actOffset;

    DataCopyParams dataCopy2HPerGroupForSwiGluV1Params;
    DataCopyPadParams padParams{false, 0, 0, 0};
    dataCopy2HPerGroupForSwiGluV1Params.blockCount = 1;
    dataCopy2HPerGroupForSwiGluV1Params.blockLen = processNum * sizeof(copyDtype);
    dataCopy2HPerGroupForSwiGluV1Params.srcStride = 0;
    dataCopy2HPerGroupForSwiGluV1Params.dstStride = 0;
    // copy_in: (G, H) 激活部分
    DataCopyPad(dstTensor[0], srcTensor[groupOffsetFor2H + ubloop * tl_->UbFactorDimy + actOffset], dataCopy2HPerGroupForSwiGluV1Params, padParams);
    // copy_in: (G, H) 门控部分
    DataCopyPad(dstTensor[tl_->UbFactorDimy], srcTensor[groupOffsetFor2H + ubloop * tl_->UbFactorDimy + gateOffset], dataCopy2HPerGroupForSwiGluV1Params, padParams);
}

template <typename TXtype, typename TActScale, typename TBias, typename TQuantScale, typename TQuantOffset, typename TGroup, typename TYtype>
template <typename copyDtype>
__aicore__ inline void DequantSwigluQuantStaticNotFull<TXtype, TActScale, TBias, TQuantScale, TQuantOffset, TGroup, TYtype>::CopyIn2HPerGroupForSwiGluV2(LocalTensor<copyDtype> dstTensor, GlobalTensor<copyDtype> srcTensor, int64_t groupIdx, int64_t ubloop, int64_t processNum)
{
    int64_t groupOffsetFor2H = groupIdx * tl_->inDimy;

    DataCopyParams dataCopy2HPerGroupForSwiGluV2Params;
    DataCopyPadParams padParams{false, 0, 0, 0};
    dataCopy2HPerGroupForSwiGluV2Params.blockCount = 1;
    dataCopy2HPerGroupForSwiGluV2Params.blockLen = 2 * processNum * sizeof(copyDtype);
    dataCopy2HPerGroupForSwiGluV2Params.srcStride = 0;
    dataCopy2HPerGroupForSwiGluV2Params.dstStride = 0;
    // copy_in: (BS, 2H)
    DataCopyPad(dstTensor[0], srcTensor[groupOffsetFor2H + ubloop * 2 * tl_->UbFactorDimy], dataCopy2HPerGroupForSwiGluV2Params, padParams);
}

template <typename TXtype, typename TActScale, typename TBias, typename TQuantScale, typename TQuantOffset, typename TGroup, typename TYtype>
template <typename copyDtype>
__aicore__ inline void DequantSwigluQuantStaticNotFull<TXtype, TActScale, TBias, TQuantScale, TQuantOffset, TGroup, TYtype>::CopyInHPerGroup(LocalTensor<copyDtype> dstTensor, GlobalTensor<copyDtype> srcTensor, int64_t groupIdx, int64_t ubloop, int64_t processNum)
{
    int64_t groupOffsetForH = groupIdx * tl_->outDimy;

    DataCopyParams dataCopyHPerGroup;
    DataCopyPadParams padParams{false, 0, 0, 0};
    dataCopyHPerGroup.blockCount = 1;
    dataCopyHPerGroup.blockLen = processNum * sizeof(copyDtype);
    dataCopyHPerGroup.srcStride = 0;
    dataCopyHPerGroup.dstStride = 0;
    // copy_in: (G, H)
    DataCopyPad(dstTensor[0], srcTensor[groupOffsetForH + ubloop * tl_->UbFactorDimy], dataCopyHPerGroup, padParams);
}

template <typename TXtype, typename TActScale, typename TBias, typename TQuantScale, typename TQuantOffset, typename TGroup, typename TYtype>
template <typename copyDtype>
__aicore__ inline void DequantSwigluQuantStaticNotFull<TXtype, TActScale, TBias, TQuantScale, TQuantOffset, TGroup, TYtype>::CopyInOnePerGroup(LocalTensor<copyDtype> dstTensor, GlobalTensor<copyDtype> srcTensor, int64_t groupIdx, int64_t ubloop, int64_t processNum)
{
    DataCopyParams dataCopyOnePerGroup;
    DataCopyPadParams padParams{false, 0, 0, 0};
    dataCopyOnePerGroup.blockCount = 1;
    dataCopyOnePerGroup.blockLen = 1 * sizeof(copyDtype);
    dataCopyOnePerGroup.srcStride = 0;
    dataCopyOnePerGroup.dstStride = 0;
    // copy_in: (G, 1)
    DataCopyPad(dstTensor[0], srcTensor[groupIdx], dataCopyOnePerGroup, padParams);
}

template <typename TXtype, typename TActScale, typename TBias, typename TQuantScale, typename TQuantOffset, typename TGroup, typename TYtype>
__aicore__ inline void DequantSwigluQuantStaticNotFull<TXtype, TActScale, TBias, TQuantScale, TQuantOffset, TGroup, TYtype>::ProcessSingleRow(int64_t groupIdx, int64_t rowIdx)
{
    // 如果当前核id超过了实际需要的核数，则说明处理完成了
    if (blockIdx_ >= realCoreDim_) {
        return;
    }
    int64_t rowOffsetForH = rowIdx * tl_->outDimy;
    for (int64_t ubloop = 0; ubloop < tl_->loopTimesPerRow; ubloop++) {
        int64_t processNum = ubloop == tl_->loopTimesPerRow - 1 ? tl_->tailPerRow : tl_->UbFactorDimy;
        DataCopyPadParams padParams{false, 0, 0, 0};

        // copy_in: x(BS, 2H) -> (BS, H),(BS, H)
        xLocal = xQueue_.AllocTensor<TXtype>();
        if (tl_->swiGluMode == 0) {
            CopyIn2HPerRowForSwiGluV1<TXtype>(xLocal, xGm_, rowIdx, ubloop, processNum);
        } else {
            CopyIn2HPerRowForSwiGluV2<TXtype>(xLocal, xGm_, rowIdx, ubloop, processNum);
        }
        xQueue_.EnQue(xLocal);
        // copy_in: weight_scale(G, 2H) -> (G, H),(G, H)
        if constexpr (hasWeightScale_) {
            weightScaleLocal = weightScaleQueue_.AllocTensor<float>();
            if (tl_->swiGluMode == 0) {
                CopyIn2HPerGroupForSwiGluV1<float>(weightScaleLocal, weightScaleGm_, groupIdx, ubloop, processNum);
            } else {
                CopyIn2HPerGroupForSwiGluV2<float>(weightScaleLocal, weightScaleGm_, groupIdx, ubloop, processNum);
            }
            weightScaleQueue_.EnQue(weightScaleLocal);
        }
        // copy_in: activation_scale(BS,)
        if constexpr (hasActScale_) {
            actScaleLocal = actScaleQueue_.AllocTensor<TActScale>();
            DataCopyParams dataCopyActScaleParams;
            dataCopyActScaleParams.blockCount = 1;
            dataCopyActScaleParams.blockLen = 1 * sizeof(TActScale);
            dataCopyActScaleParams.srcStride = 0;
            dataCopyActScaleParams.dstStride = 0;
            DataCopyPad(actScaleLocal[0], activationScaleGm_[rowIdx], dataCopyActScaleParams, padParams);
            actScaleQueue_.EnQue(actScaleLocal);
        }
        // copy_in: bias(G, 2H)->(G, H), (G, H)
        if constexpr (hasBiasIndex_) {
            if (ifBiasIntIndex_) {
                biasInt32Local = biasQueue_.AllocTensor<int32_t>();
                if (tl_->swiGluMode == 0) {
                    CopyIn2HPerGroupForSwiGluV1<int32_t>(biasInt32Local, biasInt32Gm_, groupIdx, ubloop, processNum);
                } else {
                    CopyIn2HPerGroupForSwiGluV2<int32_t>(biasInt32Local, biasInt32Gm_, groupIdx, ubloop, processNum);
                }
                biasQueue_.EnQue(biasInt32Local);
            } else if (ifBiasBfloat16Index_) {
                biasBf16Local = biasQueue_.AllocTensor<bfloat16_t>();
                if (tl_->swiGluMode == 0) {
                    CopyIn2HPerGroupForSwiGluV1<bfloat16_t>(biasBf16Local, biasBf16Gm_, groupIdx, ubloop, processNum);
                } else {
                    CopyIn2HPerGroupForSwiGluV2<bfloat16_t>(biasBf16Local, biasBf16Gm_, groupIdx, ubloop, processNum);
                }
                biasQueue_.EnQue(biasBf16Local);
            } else if (ifBiasFloat16Index_) {
                biasFp16Local = biasQueue_.AllocTensor<half>();
                if (tl_->swiGluMode == 0) {
                    CopyIn2HPerGroupForSwiGluV1<half>(biasFp16Local, biasFp16Gm_, groupIdx, ubloop, processNum);
                } else {
                    CopyIn2HPerGroupForSwiGluV2<half>(biasFp16Local, biasFp16Gm_, groupIdx, ubloop, processNum);
                }
                biasQueue_.EnQue(biasFp16Local);
            } else if (ifBiasFloatIndex_) {
                biasLocal = biasQueue_.AllocTensor<float>();
                if (tl_->swiGluMode == 0) {
                    CopyIn2HPerGroupForSwiGluV1<float>(biasLocal, biasFp32Gm_, groupIdx, ubloop, processNum);
                } else {
                    CopyIn2HPerGroupForSwiGluV2<float>(biasLocal, biasFp32Gm_, groupIdx, ubloop, processNum);
                }
                biasQueue_.EnQue(biasLocal);
            }
        }
        // copy_in: quant_scale(G, 1)或(G, H)
        if constexpr (hasQuantScale_) {
            quantScaleLocal = quantScaleQueue_.AllocTensor<TQuantScale>();
            if (!ifQuantIsOne_) {
                CopyInHPerGroup<float>(quantScaleLocal, quantScaleGm_, groupIdx, ubloop, processNum);
            } else {
                CopyInOnePerGroup<float>(quantScaleLocal, quantScaleGm_, groupIdx, ubloop, processNum);
            }
            quantScaleQueue_.EnQue(quantScaleLocal);
        }
        // copy_in: quant_offset(G, 1)或(G, H)
        if constexpr (hasQuantOffset_) {
            quantOffsetLocal = quantOffsetQueue_.AllocTensor<TQuantOffset>();
            if (!ifQuantIsOne_) {
                CopyInHPerGroup<float>(quantOffsetLocal, quantOffsetGm_, groupIdx, ubloop, processNum);
            } else {
                CopyInOnePerGroup<float>(quantOffsetLocal, quantOffsetGm_, groupIdx, ubloop, processNum);
            }
            quantOffsetQueue_.EnQue(quantOffsetLocal);
        }
        xLocal = xQueue_.DeQue<TXtype>();
        xPtr = (__local_mem__ TXtype*)xLocal.GetPhyAddr(0);
        x2Ptr = (__local_mem__ TXtype*)xLocal.GetPhyAddr(tl_->UbFactorDimy);
        LocalTensor<float> tmpLocal = tmpBuffer.Get<float>();
        __local_mem__ float* tmpPtr = (__local_mem__ float*)tmpLocal.GetPhyAddr(0);
        __local_mem__ float* tmp2Ptr = (__local_mem__ float*)tmpLocal.GetPhyAddr(tl_->UbFactorDimy);
        // 当weight_scale、act_scale、bias存在时，才去获取地址
        if constexpr (hasWeightScale_) {
            weightScaleLocal = weightScaleQueue_.DeQue<float>();
            wScalePtr = (__local_mem__ float*)weightScaleLocal.GetPhyAddr(0);
            wScale2Ptr = (__local_mem__ float*)weightScaleLocal.GetPhyAddr(tl_->UbFactorDimy);
        }
        if constexpr (hasActScale_) {
            actScaleLocal = actScaleQueue_.DeQue<float>();
            aScalePtr = (__local_mem__ float*)actScaleLocal.GetPhyAddr(0);
        }
        if constexpr (hasBiasIndex_) {
            if (ifBiasIntIndex_) {
                biasInt32Local = biasQueue_.DeQue<int32_t>();
                biasInt32Ptr = (__local_mem__ int32_t*)biasInt32Local.GetPhyAddr(0);
                bias2Int32Ptr = (__local_mem__ int32_t*)biasInt32Local.GetPhyAddr(tl_->UbFactorDimy);
            } else if (ifBiasBfloat16Index_) {
                biasBf16Local = biasQueue_.DeQue<bfloat16_t>();
                biasBf16Ptr = (__local_mem__ bfloat16_t*)biasBf16Local.GetPhyAddr(0);
                bias2Bf16Ptr = (__local_mem__ bfloat16_t*)biasBf16Local.GetPhyAddr(tl_->UbFactorDimy);
            } else if (ifBiasFloat16Index_) {
                biasFp16Local = biasQueue_.DeQue<half>();
                biasFp16Ptr = (__local_mem__ half*)biasFp16Local.GetPhyAddr(0);
                bias2Fp16Ptr = (__local_mem__ half*)biasFp16Local.GetPhyAddr(tl_->UbFactorDimy);
            } else if (ifBiasFloatIndex_) {
                biasLocal = biasQueue_.DeQue<float>();
                biasPtr = (__local_mem__ float*)biasLocal.GetPhyAddr(0);
                bias2Ptr = (__local_mem__ float*)biasLocal.GetPhyAddr(tl_->UbFactorDimy);
            }
        }
        // 将fp16/bf16统一cast到fp32类型，因此sizePerRepeat使用fp32计算
        constexpr uint32_t sizePerRepeat = AscendC::GetVecLen() / sizeof(float);
        uint32_t repeatTimesPerFactor = CeilDivision(processNum, sizePerRepeat); // 向上取整
        uint32_t repeatTimesPerFactorForV2 = CeilDivision(processNum * 2, sizePerRepeat); // 向上取整
        // SwiGlu Mode不同，UB内数据排布不同，区分处理逻辑
        if (tl_->swiGluMode == 0) {
            // Dequant，根据x的数据类型选择Dequant分支
            if constexpr (ifXIntIndex_) {
                if (ifBiasIntIndex_ || !hasBiasIndex_) {
                    VF_CALL<Int32Dequant<TXtype, int32_t, hasBiasIndex_, hasActScale_>>(xPtr, tmpPtr, wScalePtr, aScalePtr, biasInt32Ptr, repeatTimesPerFactor, sizePerRepeat, processNum);
                    VF_CALL<Int32Dequant<TXtype, int32_t, hasBiasIndex_, hasActScale_>>(x2Ptr, tmp2Ptr, wScale2Ptr, aScalePtr, bias2Int32Ptr, repeatTimesPerFactor, sizePerRepeat, processNum);
                } else if (ifBiasBfloat16Index_) {
                    VF_CALL<Int32Dequant<TXtype, bfloat16_t, hasBiasIndex_, hasActScale_>>(xPtr, tmpPtr, wScalePtr, aScalePtr, biasBf16Ptr, repeatTimesPerFactor, sizePerRepeat, processNum);
                    VF_CALL<Int32Dequant<TXtype, bfloat16_t, hasBiasIndex_, hasActScale_>>(x2Ptr, tmp2Ptr, wScale2Ptr, aScalePtr, bias2Bf16Ptr, repeatTimesPerFactor, sizePerRepeat, processNum);
                } else if (ifBiasFloat16Index_) {
                    VF_CALL<Int32Dequant<TXtype, half, hasBiasIndex_, hasActScale_>>(xPtr, tmpPtr, wScalePtr, aScalePtr, biasFp16Ptr, repeatTimesPerFactor, sizePerRepeat, processNum);
                    VF_CALL<Int32Dequant<TXtype, half, hasBiasIndex_, hasActScale_>>(x2Ptr, tmp2Ptr, wScale2Ptr, aScalePtr, bias2Fp16Ptr, repeatTimesPerFactor, sizePerRepeat, processNum);
                } else if (ifBiasFloatIndex_) {
                    VF_CALL<Int32Dequant<TXtype, float, hasBiasIndex_, hasActScale_>>(xPtr, tmpPtr, wScalePtr, aScalePtr, biasPtr, repeatTimesPerFactor, sizePerRepeat, processNum);
                    VF_CALL<Int32Dequant<TXtype, float, hasBiasIndex_, hasActScale_>>(x2Ptr, tmp2Ptr, wScale2Ptr, aScalePtr, bias2Ptr, repeatTimesPerFactor, sizePerRepeat, processNum);
                }
            } else {
                VF_CALL<FloatDequant<TXtype, ifXFloat16Index_, ifXBf16Index_>>(xPtr, tmpPtr, repeatTimesPerFactor, sizePerRepeat, processNum);
                VF_CALL<FloatDequant<TXtype, ifXFloat16Index_, ifXBf16Index_>>(x2Ptr, tmp2Ptr, repeatTimesPerFactor, sizePerRepeat, processNum);
            }
            xQueue_.FreeTensor(xLocal);
            if constexpr (hasWeightScale_) {
                weightScaleQueue_.FreeTensor(weightScaleLocal);
            }
            if constexpr (hasActScale_) {
                actScaleQueue_.FreeTensor(actScaleLocal);
            }
            if constexpr (hasBiasIndex_) {
                if (ifBiasIntIndex_) {
                    biasQueue_.FreeTensor(biasInt32Local);
                } else if (ifBiasBfloat16Index_) {
                    biasQueue_.FreeTensor(biasBf16Local);
                } else if (ifBiasFloat16Index_) {
                    biasQueue_.FreeTensor(biasFp16Local);
                } else if (ifBiasFloatIndex_) {
                    biasQueue_.FreeTensor(biasLocal);
                }
            }
            if constexpr (hasQuantScale_) {
                quantScaleLocal = quantScaleQueue_.DeQue<float>();
                qScalePtr = (__local_mem__ float*)quantScaleLocal.GetPhyAddr(0);
            }
            // SwiGlu with QuantScale
            VF_CALL<SwigluSingleYWithQuantScale<hasQuantScale_>>(tmpPtr, qScalePtr, tmpPtr, static_cast<uint32_t>(tl_->UbFactorDimy),
                                                repeatTimesPerFactor, sizePerRepeat, processNum, ifQuantIsOne_);
        } else {
            if constexpr (ifXIntIndex_) {
                if (ifBiasIntIndex_ || !hasBiasIndex_) {
                    VF_CALL<Int32Dequant<TXtype, int32_t, hasBiasIndex_, hasActScale_>>(xPtr, tmpPtr, wScalePtr, aScalePtr, biasInt32Ptr, repeatTimesPerFactorForV2, sizePerRepeat, processNum * 2);
                } else if (ifBiasBfloat16Index_) {
                    VF_CALL<Int32Dequant<TXtype, bfloat16_t, hasBiasIndex_, hasActScale_>>(xPtr, tmpPtr, wScalePtr, aScalePtr, biasBf16Ptr, repeatTimesPerFactorForV2, sizePerRepeat, processNum * 2);
                } else if (ifBiasFloat16Index_) {
                    VF_CALL<Int32Dequant<TXtype, half, hasBiasIndex_, hasActScale_>>(xPtr, tmpPtr, wScalePtr, aScalePtr, biasFp16Ptr, repeatTimesPerFactorForV2, sizePerRepeat, processNum * 2);
                } else if (ifBiasFloatIndex_) {
                    VF_CALL<Int32Dequant<TXtype, float, hasBiasIndex_, hasActScale_>>(xPtr, tmpPtr, wScalePtr, aScalePtr, biasPtr, repeatTimesPerFactorForV2, sizePerRepeat, processNum * 2);
                }
            } else {
                VF_CALL<FloatDequant<TXtype, ifXFloat16Index_, ifXBf16Index_>>(xPtr, tmpPtr, repeatTimesPerFactorForV2, sizePerRepeat, processNum * 2);
            }
            xQueue_.FreeTensor(xLocal);
            if constexpr (hasWeightScale_) {
                weightScaleQueue_.FreeTensor(weightScaleLocal);
            }
            if constexpr (hasActScale_) {
                actScaleQueue_.FreeTensor(actScaleLocal);
            }
            if constexpr (hasBiasIndex_) {
                if (ifBiasIntIndex_) {
                    biasQueue_.FreeTensor(biasInt32Local);
                } else if (ifBiasBfloat16Index_) {
                    biasQueue_.FreeTensor(biasBf16Local);
                } else if (ifBiasFloat16Index_) {
                    biasQueue_.FreeTensor(biasFp16Local);
                } else if (ifBiasFloatIndex_) {
                    biasQueue_.FreeTensor(biasLocal);
                }
            }
            if constexpr (hasQuantScale_) {
                quantScaleLocal = quantScaleQueue_.DeQue<float>();
                qScalePtr = (__local_mem__ float*)quantScaleLocal.GetPhyAddr(0);
            }
            // SwiGluV2 with QuantScale
            VF_CALL<SwigluV2SingleYWithQuantScale<hasQuantScale_>>(tmpPtr, qScalePtr, tmpPtr, processNum * 2, 0, 1, repeatTimesPerFactorForV2,
                        sizePerRepeat, 0, ifQuantIsOne_, tl_->clampLimit, tl_->gluAlpha, tl_->gluBias);
        }
        if constexpr (hasQuantScale_) {
            quantScaleQueue_.FreeTensor(quantScaleLocal);
        }

        if constexpr (hasQuantOffset_) {
            quantOffsetLocal = quantOffsetQueue_.DeQue<float>();
            qOffsetPtr = (__local_mem__ float*)quantOffsetLocal.GetPhyAddr(0);
        }
        // static quant with QuantOffset
        if constexpr (hasQuantOffset_) {
            VF_CALL<StaticQuantSingleY<hasQuantOffset_>>(tmpPtr, qOffsetPtr, tmpPtr,
                                                        repeatTimesPerFactor, sizePerRepeat, processNum, ifQuantIsOne_);
        }
        if constexpr (hasQuantOffset_) {
            quantOffsetQueue_.FreeTensor(quantOffsetLocal);
        }
        // cast y to dst_dtype
        // int8,fp8
        LocalTensor<TYtype> yLocal = yQueue_.AllocTensor<TYtype>();
        __local_mem__ TYtype* yPtr = (__local_mem__ TYtype*)yLocal.GetPhyAddr();
        // fp4
        LocalTensor<uint8_t> yFp4Local = yLocal.template ReinterpretCast<uint8_t>();
        __local_mem__ uint8_t* yFp4Ptr = (__local_mem__ uint8_t*)yFp4Local.GetPhyAddr();
        VF_CALL<CastY<TXtype, TYtype, ifYFloat8e4m3Index_, ifYFloat8e5m2Index_, ifYFloat4e2m1Index_, ifYFloat4e1m2Index_, ifYHiFloat8Index_>>(tmpPtr, yPtr,
                    yFp4Ptr, processNum, 1, repeatTimesPerFactor, sizePerRepeat, tl_->roundMode, 1, 1, 1);
        yQueue_.EnQue<TYtype>(yLocal);
        yLocal = yQueue_.DeQue<TYtype>();

        // copy_out: y
        if constexpr (ifYFloat4e2m1Index_ || ifYFloat4e1m2Index_) {
            GlobalTensor<uint8_t> yGmTmp = yGm_.template ReinterpretCast<uint8_t>();
            DataCopyParams dataCopyYParams;
            dataCopyYParams.blockCount = 1;
            dataCopyYParams.blockLen = processNum * sizeof(TYtype) / 2;
            dataCopyYParams.srcStride = 0;
            dataCopyYParams.dstStride = 0;
            DataCopyPad(yGmTmp[(rowOffsetForH + ubloop * tl_->UbFactorDimy) / 2], yFp4Local[0], dataCopyYParams);
        } else {
            DataCopyParams dataCopyYParams;
            dataCopyYParams.blockCount = 1;
            dataCopyYParams.blockLen = processNum * sizeof(TYtype);
            dataCopyYParams.srcStride = 0;
            dataCopyYParams.dstStride = 0;
            DataCopyPad(yGm_[rowOffsetForH + ubloop * tl_->UbFactorDimy], yLocal[0], dataCopyYParams);
        }
        yQueue_.FreeTensor(yLocal);
    }
}
}
#endif