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
 * \file dequant_swiglu_quant_dynamic_not_full.h
 * \brief 动态非全载模板
 */

#ifndef DEQUANT_SWIGLU_QUANT_DYNAMIC_NOT_FULL_H
#define DEQUANT_SWIGLU_QUANT_DYNAMIC_NOT_FULL_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "dequant_swiglu_quant_common.h"
#include "dequant_swiglu_quant_dynamic_not_full_vf.h"

namespace DequantSwigluQuantV35Ops {
using namespace AscendC;

template <typename ActiScaleType, typename QuantScaleType, typename GroupType, typename BiasType, typename XType, typename YType>
class DequantSwigluQuantDynamicNotFull {
public:
    static constexpr bool hasActScale_ = IsSameType<ActiScaleType, float>::value;
    static constexpr bool hasQuantScale_ = IsSameType<QuantScaleType, float>::value;
    static constexpr bool hasGroupIndex_ = IsSameType<GroupType, int64_t>::value || IsSameType<GroupType, int32_t>::value;
    // bias标记 bias可支持float，float16，bf16，int32
    static constexpr bool hasBiasIndex_ = IsSameType<BiasType, float>::value || IsSameType<BiasType, half>::value || IsSameType<BiasType, bfloat16_t>::value || IsSameType<BiasType, int32_t>::value;
    // bias数据类型为int32标记
    static constexpr bool ifBiasIntIndex_ = IsSameType<BiasType, int32_t>::value;
    // bias数据类型为float标记
    static constexpr bool ifBiasFloatIndex_ = IsSameType<BiasType, float>::value;
    // bias数据类型为float16标记
    static constexpr bool ifBiasFloat16Index_ = IsSameType<BiasType, half>::value;
    // bias数据类型为bfloat16标记
    static constexpr bool ifBiasBfloat16Index_ = IsSameType<BiasType, bfloat16_t>::value;
    // x数据类型为int32标记
    static constexpr bool ifXIntIndex_ = IsSameType<XType, int32_t>::value;
    // x数据类型为bf16标记
    static constexpr bool ifXBf16Index_ = IsSameType<XType, bfloat16_t>::value;
    // x数据类型为float16标记
    static constexpr bool ifXFloat16Index_ = IsSameType<XType, half>::value;
    // y数据类型为int8标记
    static constexpr bool ifYInt8Index_ = IsSameType<YType, int8_t>::value;
    // y数据类型为float8_e4m3标记
    static constexpr bool ifYFloat8e4m3Index_ = IsSameType<YType, fp8_e4m3fn_t>::value;
    // y数据类型为float8_e5m2标记
    static constexpr bool ifYFloat8e5m2Index_ = IsSameType<YType, fp8_e5m2_t>::value;
    // y数据类型为float4_e2m1标记
    static constexpr bool ifYFloat4e2m1Index_ = IsSameType<YType, fp4x2_e2m1_t>::value;
    // y数据类型为float4_e1m2标记
    static constexpr bool ifYFloat4e1m2Index_ = IsSameType<YType, fp4x2_e1m2_t>::value;
    // y数据类型为hifloat8标记
    static constexpr bool ifYHiFloat8Index_ = IsSameType<YType, hifloat8_t>::value;

    __aicore__ inline DequantSwigluQuantDynamicNotFull(TPipe* pipe) { pipe_ = pipe; };
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weightScale, GM_ADDR activationScale, GM_ADDR bias, GM_ADDR quantScale,
                                GM_ADDR quantOffset, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR scale,
                                GM_ADDR workspace, const DequantSwigluQuantV35BaseTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SetGlobalTensor(GM_ADDR x, GM_ADDR weightScale, GM_ADDR activationScale, GM_ADDR bias, GM_ADDR quantScale,
                                        GM_ADDR quantOffset, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR scale, GM_ADDR workspace);
    __aicore__ inline void ParseTilingData(const DequantSwigluQuantV35BaseTilingData* tilingData);                   
    __aicore__ inline void CopyIn(int64_t groupIndex, size_t rowIndex, uint64_t loop, uint32_t tileData, int64_t blockLength);
    __aicore__ inline void CopyInV2(int64_t groupIndex, size_t rowIndex, uint64_t loop, uint32_t tileData, int64_t blockLength);
    __aicore__ inline void ProcessSingleRow(int64_t groupIndex, size_t rowIndex);
    __aicore__ inline void PreProcess(uint32_t tileData);
    __aicore__ inline void PreProcessV2(uint32_t tileData);
    __aicore__ inline void UpdateReduceMax(uint32_t tileData);
    __aicore__ inline void CalculateResult(uint32_t tileData);
    __aicore__ inline float GetScalarMaxNum();
    __aicore__ inline void SetAlignLength(uint32_t tileData);
    __aicore__ inline void CopyOutScale(uint64_t gmOffset);
    __aicore__ inline void CopyOutY(uint64_t gmOffset, uint32_t tileData);
    __aicore__ inline void CopyInForXTemp(uint64_t gmOffset, uint32_t tileData);
    __aicore__ inline void CopyOutForXTemp(uint64_t gmOffset, uint32_t tileData);

protected:
    // global memory address
    // input global mem
    // 类型支持int32 fp16, bf16
    GlobalTensor<XType> xGm_;
    // 类型支持float
    GlobalTensor<float> weightScaleGm_;
    // 类型支持float
    GlobalTensor<ActiScaleType> activationScaleGm_;
    // 类型支持 float，float16，bf16，int32
    GlobalTensor<BiasType> biasGm_;
    // 类型支持 float
    GlobalTensor<QuantScaleType> quantScaleGm_;
    // 类型支持int32 int64
    GlobalTensor<GroupType> groupIndexGm_;
    // 反量化和SwiGlu计算的结果
    GlobalTensor<float> xTempGm_;
    // 输出y地址GM
    GlobalTensor<YType> yGm_;
    // 输出scale地址GM
    GlobalTensor<float> scaleGm_;

    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, 1> xQueue_;
    TQue<QuePosition::VECIN, 1> weightScaleQueue_;
    TQue<QuePosition::VECIN, 1> activationScaleQueue_;
    TQue<QuePosition::VECIN, 1> biasQueue_;
    TQue<QuePosition::VECIN, 1> quantScaleQueue_;
    TQue<QuePosition::VECOUT, 1> yQueue_;
    TQue<QuePosition::VECOUT, 1> scaleQueue_;
    TBuf<TPosition::VECCALC> tmpBuffer_;

    LocalTensor<float> xTempLocal_;
    __local_mem__ float* xTempPtr_;
    LocalTensor<float> scaleLocal_;
    __local_mem__ float* scalePtr_;
   
    uint32_t blockIdx_ = GetBlockIdx();
    // 处理一行需要的循环次数
    int64_t loopCount_ = 0;
    // UB每次处理的数据长度，包含尾块的长度
    int64_t blockLength_ = 0;
    int64_t tailBlockLength_ = 0;
    // 输入X的行的长度
    int64_t xLength_ = 0;
    // 输出Y的行的长度
    int64_t yLength_ = 0;
    // 输入输出行数
    int64_t row_ = 0;
    // group_index类型，0：不存在；1：int32；2：int64
    int64_t groupIndexMode_ = 0;
    // group的大小
    int64_t groupNum_ = 0;
    // 使用的核数
    int64_t usedCoreNum_ = 0;
    // 核数
    int64_t coreNum_ = 0;
    // 是否为右激活
    int64_t isRightActivation_ = 0;
    // 溢出模式的标识
    int64_t roundMode_ = 0;
    // SwiGlu模式,默认值0，奇偶为1
    int64_t swiGluMode_ = 0;
    // SwiGlu变体参数
    float clampLimit_ = 0;
    // SwiGlu变体参数
    float gluAlpha_ = 0;
    // SwiGlu变体参数
    float gluBias_ = 0;
    // x blockSize对齐的length
    uint32_t xAlignLength_ = 0;
    // weight scale blockSize对齐的length
    uint32_t weightAlignLength_ = 0;
    // bias blockSize对齐的Length
    uint32_t biasAlignLength_ = 0;
};

template <typename ActiScaleType, typename QuantScaleType, typename GroupType, typename BiasType, typename XType, typename YType>
using DSQDynamicNotFull = DequantSwigluQuantDynamicNotFull<ActiScaleType, QuantScaleType, GroupType, BiasType, XType, YType>;

template <typename ActiScaleType, typename QuantScaleType, typename GroupType, typename BiasType, typename XType, typename YType>
__aicore__ inline void DSQDynamicNotFull<ActiScaleType, QuantScaleType, GroupType, BiasType, XType, YType>::Init(GM_ADDR x, GM_ADDR weightScale, GM_ADDR activationScale,
                                            GM_ADDR bias, GM_ADDR quantScale, GM_ADDR quantOffset,
                                            GM_ADDR groupIndex, GM_ADDR y, GM_ADDR scale,
                                            GM_ADDR workspace, const DequantSwigluQuantV35BaseTilingData* tilingData)
{
    ParseTilingData(tilingData);
    SetGlobalTensor(x, weightScale, activationScale, bias, quantScale, quantOffset, groupIndex, y, scale, workspace);
    pipe_->InitBuffer(xQueue_, DOUBLE_BUFFER, blockLength_ * 2 * sizeof(XType));
    pipe_->InitBuffer(weightScaleQueue_, 1, blockLength_ * 2 * sizeof(float));
    pipe_->InitBuffer(activationScaleQueue_, DOUBLE_BUFFER,  BLOCK_ELEM_B32 * sizeof(float));
    pipe_->InitBuffer(quantScaleQueue_, 1, blockLength_ * sizeof(float));
    pipe_->InitBuffer(yQueue_, DOUBLE_BUFFER,  blockLength_ * sizeof(YType));
    pipe_->InitBuffer(scaleQueue_, 1,  BLOCK_ELEM_B32 * sizeof(float));
    pipe_->InitBuffer(tmpBuffer_, blockLength_ * sizeof(float));
    // 如果有bias入参，则给bias进行地址申请
    if constexpr (hasBiasIndex_) {
        pipe_->InitBuffer(biasQueue_, 1, blockLength_ * 2 * sizeof(BiasType));
    }
}

template <typename ActiScaleType, typename QuantScaleType, typename GroupType, typename BiasType, typename XType, typename YType>
__aicore__ inline void DSQDynamicNotFull<ActiScaleType, QuantScaleType, GroupType, BiasType, XType, YType>::SetGlobalTensor(GM_ADDR x, GM_ADDR weightScale, GM_ADDR activationScale,
                                                        GM_ADDR bias, GM_ADDR quantScale, GM_ADDR quantOffset,
                                                        GM_ADDR groupIndex, GM_ADDR y, GM_ADDR scale, GM_ADDR workspace)
{
    xGm_.SetGlobalBuffer((__gm__ XType*)x);
    weightScaleGm_.SetGlobalBuffer((__gm__ float*)weightScale);
    activationScaleGm_.SetGlobalBuffer((__gm__ ActiScaleType*)activationScale);
    quantScaleGm_.SetGlobalBuffer((__gm__ QuantScaleType*)quantScale);
    if constexpr (hasGroupIndex_) {
        groupIndexGm_.SetGlobalBuffer((__gm__ GroupType*)groupIndex);
    }
    if constexpr (hasBiasIndex_) {
        biasGm_.SetGlobalBuffer((__gm__ BiasType*)bias);
    }
    yGm_.SetGlobalBuffer((__gm__ YType*)y);
    scaleGm_.SetGlobalBuffer((__gm__ float*)scale);

    // 反量化和SwiGlu计算的结果
    xTempGm_.SetGlobalBuffer((__gm__ float*)workspace, usedCoreNum_ * yLength_);
}

template <typename ActiScaleType, typename QuantScaleType, typename GroupType, typename BiasType, typename XType, typename YType>
__aicore__ inline void DSQDynamicNotFull<ActiScaleType, QuantScaleType, GroupType, BiasType, XType, YType>::ParseTilingData(const DequantSwigluQuantV35BaseTilingData* tilingData)
{
    roundMode_ = tilingData->roundMode;
    loopCount_ = tilingData->loopTimesPerRow;
    blockLength_ = tilingData->UbFactorDimy;
    tailBlockLength_ = tilingData->tailPerRow;
    xLength_ = tilingData->inDimy;
    yLength_ = tilingData->outDimy;
    row_ = tilingData->inDimx;
    groupIndexMode_ = tilingData->groupIndexMode;
    groupNum_ = tilingData->inGroupNum;
    coreNum_ = tilingData->maxCoreNum;
    usedCoreNum_ = tilingData->usedCoreNum;
    isRightActivation_ = tilingData->actRight;
    swiGluMode_ = tilingData->swiGluMode;
    clampLimit_ = tilingData->clampLimit;
    gluAlpha_ = tilingData->gluAlpha;
    gluBias_ = tilingData->gluBias;
}

template <typename ActiScaleType, typename QuantScaleType, typename GroupType, typename BiasType, typename XType, typename YType>
__aicore__ inline void DSQDynamicNotFull<ActiScaleType, QuantScaleType, GroupType, BiasType, XType, YType>::CopyIn(int64_t groupIdx, size_t rowIndex, uint64_t loop, uint32_t tileData, int64_t blockLength)
{   // uint32_t tailData = 321;
    int64_t xGmOffset = rowIndex * xLength_;
    // 激活左/右部分偏移，actRight是表示是否激活右半部分
    int64_t actOffset;
    int64_t gateOffset;
    if (isRightActivation_) {
        actOffset = loop * blockLength + yLength_;
        gateOffset = loop * blockLength;
    } else {
        actOffset = loop * blockLength;
        gateOffset = loop * blockLength + yLength_;
    }
    DataCopyPadParams padParams{false, 0, 0, 0};
    // copy_in: x(xDimPerLoop, H)
    LocalTensor<XType> xLocal = xQueue_.AllocTensor<XType>();
    DataCopyParams dataCopyXParams;
    dataCopyXParams.blockCount = 1;
    dataCopyXParams.blockLen = tileData * sizeof(XType);
    dataCopyXParams.srcStride = 0;
    dataCopyXParams.dstStride = 0;
    DataCopyPad(xLocal[0], xGm_[xGmOffset + actOffset], dataCopyXParams, padParams); // 激活部分
    DataCopyPad(xLocal[xAlignLength_], xGm_[xGmOffset + gateOffset], dataCopyXParams, padParams); // 门控部分
    xQueue_.EnQue(xLocal);
    // copy_in: weight_scale(G, H)
    if constexpr (ifXIntIndex_) {
        LocalTensor<float> weightScaleLocal = weightScaleQueue_.AllocTensor<float>();
        DataCopyParams dataCopyWeightScaleParams;
        dataCopyWeightScaleParams.blockCount = 1;
        dataCopyWeightScaleParams.blockLen = tileData * sizeof(float);
        dataCopyWeightScaleParams.srcStride = 0;
        dataCopyWeightScaleParams.dstStride = 0;
        DataCopyPad(weightScaleLocal[0], weightScaleGm_[groupIdx * xLength_ + actOffset], dataCopyWeightScaleParams, padParams); // 激活部分
        DataCopyPad(weightScaleLocal[weightAlignLength_], weightScaleGm_[groupIdx * xLength_ + gateOffset], dataCopyWeightScaleParams, padParams); // 门控部分
        weightScaleQueue_.EnQue(weightScaleLocal);
    }
    // copy_in: activation_scale(BS,)
    if constexpr (hasActScale_) {
        LocalTensor<float> activationScaleLocal = activationScaleQueue_.AllocTensor<float>();
        DataCopyParams dataCopyActScaleParams;
        dataCopyActScaleParams.blockCount = 1;
        dataCopyActScaleParams.blockLen = sizeof(float);
        dataCopyActScaleParams.srcStride = 0;
        dataCopyActScaleParams.dstStride = 0;
        DataCopyPad(activationScaleLocal[0], activationScaleGm_[rowIndex], dataCopyActScaleParams, padParams);
        activationScaleQueue_.EnQue(activationScaleLocal);
    }
    // copy_in: quant_scale(G, H)
    if constexpr (hasQuantScale_) {
        LocalTensor<QuantScaleType> quantScaleLocal = quantScaleQueue_.AllocTensor<QuantScaleType>();
        DataCopyParams dataCopyQuantScaleParams;
        dataCopyQuantScaleParams.blockCount = 1;
        dataCopyQuantScaleParams.blockLen = tileData * sizeof(QuantScaleType);
        dataCopyQuantScaleParams.srcStride = 0;
        dataCopyQuantScaleParams.dstStride = 0;
        DataCopyPad(quantScaleLocal[0], quantScaleGm_[groupIdx * yLength_ + loop * blockLength], dataCopyQuantScaleParams, padParams);
        quantScaleQueue_.EnQue(quantScaleLocal);
    }
    // copy_in: bias(1, 2H)->(1, H), (1, H)
    if constexpr (hasBiasIndex_) {
        LocalTensor<BiasType> biasLocal = biasQueue_.AllocTensor<BiasType>();
        DataCopyParams dataCopyBiasParams;
        dataCopyBiasParams.blockCount = 1;
        dataCopyBiasParams.blockLen = tileData * sizeof(BiasType);
        dataCopyBiasParams.srcStride = 0;
        dataCopyBiasParams.dstStride = 0;
        DataCopyPad(biasLocal[0], biasGm_[groupIdx * xLength_ + actOffset], dataCopyBiasParams, padParams);
        DataCopyPad(biasLocal[biasAlignLength_], biasGm_[groupIdx * xLength_ + gateOffset], dataCopyBiasParams, padParams);
        biasQueue_.EnQue(biasLocal);
    }  
}

template <typename ActiScaleType, typename QuantScaleType, typename GroupType, typename BiasType, typename XType, typename YType>
__aicore__ inline void DSQDynamicNotFull<ActiScaleType, QuantScaleType, GroupType, BiasType, XType, YType>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    if constexpr (!hasGroupIndex_) {
        for (size_t rowIndex = blockIdx_; rowIndex < row_; rowIndex += coreNum_) {
            ProcessSingleRow(0, rowIndex);
        }
        return;
    }

    size_t rowOffset = 0;
    for (int64_t groupIndex = 0; groupIndex < groupNum_; groupIndex++) {
        int64_t realDimx = 0;
        if (groupIndexMode_ == 2) {
            realDimx = groupIndexGm_.template ReinterpretCast<int64_t>()(groupIndex);
        } else {
            realDimx = static_cast<int64_t>(groupIndexGm_(groupIndex));
        }
        
        for (size_t rowIndex = blockIdx_; rowIndex < realDimx && rowOffset + rowIndex < row_; rowIndex += coreNum_) {
            ProcessSingleRow(groupIndex, rowOffset + rowIndex);
        }
        rowOffset += realDimx;
    }
}

template <typename ActiScaleType, typename QuantScaleType, typename GroupType, typename BiasType, typename XType, typename YType>
__aicore__ inline void DSQDynamicNotFull<ActiScaleType, QuantScaleType, GroupType, BiasType, XType, YType>::ProcessSingleRow(int64_t groupIdx, size_t rowIndex)
{
    xTempLocal_ = tmpBuffer_.Get<float>();
    xTempPtr_ = (__local_mem__ float*)xTempLocal_.GetPhyAddr();
    scaleLocal_ = scaleQueue_.AllocTensor<float>();
    Duplicate(scaleLocal_, 0.0f, 1);
    scalePtr_ = (__local_mem__ float*)scaleLocal_.GetPhyAddr();
    // 第一步:求整行的ReduceMax值,并将反量化和SwiGlu的计算结果存放到GM上
    for (size_t loop = 0; loop < loopCount_; loop++) {
        uint32_t tileData = (loop == loopCount_ - 1 && tailBlockLength_ != 0) ? tailBlockLength_ : blockLength_;
        if (swiGluMode_) {
            CopyInV2(groupIdx, rowIndex, loop, tileData, blockLength_);
            event_t eventID_MTE2_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventID_MTE2_V);
            WaitFlag<HardEvent::MTE2_V>(eventID_MTE2_V);
            PreProcessV2(tileData);
        } else {
            SetAlignLength(tileData);
            event_t eventID_S_MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
            SetFlag<HardEvent::S_MTE2>(eventID_S_MTE2);
            WaitFlag<HardEvent::S_MTE2>(eventID_S_MTE2);
            CopyIn(groupIdx, rowIndex, loop, tileData, blockLength_);
            event_t eventID_MTE2_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventID_MTE2_V);
            WaitFlag<HardEvent::MTE2_V>(eventID_MTE2_V);
            PreProcess(tileData);
        }
        UpdateReduceMax(tileData);
        event_t eventID_V_MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventID_V_MTE2);
        WaitFlag<HardEvent::V_MTE2>(eventID_V_MTE2);
        event_t eventID_V_MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventID_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(eventID_V_MTE3);
        uint64_t gmOffset = blockIdx_ * yLength_ + loop * blockLength_;
        CopyOutForXTemp(gmOffset, tileData);
    }
    CopyOutScale(rowIndex);
    event_t eventID_MTE3_MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventID_MTE3_MTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventID_MTE3_MTE2);
    // 第二步: 根据第一步求的ReduceMax值和反量化SwiGlu的计算结果，求量化值
    for (size_t loop = 0; loop < loopCount_; loop++) {
        uint32_t tileData = (loop == loopCount_ - 1 && tailBlockLength_ != 0) ? tailBlockLength_ : blockLength_;
        uint64_t inputGmOffset = blockIdx_ * yLength_ + loop * blockLength_;
        CopyInForXTemp(inputGmOffset, tileData);
        event_t eventID_MTE2_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventID_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(eventID_MTE2_V);
        CalculateResult(tileData);
        event_t eventID_V_MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventID_V_MTE2);
        WaitFlag<HardEvent::V_MTE2>(eventID_V_MTE2);
        uint64_t outputGmOffset = rowIndex * yLength_ + loop * blockLength_;
        CopyOutY(outputGmOffset, tileData);
    }
    scaleQueue_.FreeTensor(scaleLocal_);
}

template <typename ActiScaleType, typename QuantScaleType, typename GroupType, typename BiasType, typename XType, typename YType>
__aicore__ inline void DSQDynamicNotFull<ActiScaleType, QuantScaleType, GroupType, BiasType, XType, YType>::PreProcess(uint32_t tileData)
{
    LocalTensor<XType> xLocal;
    LocalTensor<float> weightScaleLocal;
    LocalTensor<float> activationScaleLocal;
    LocalTensor<BiasType> biasLocal;
    xLocal = xQueue_.DeQue<XType>();
    if constexpr (ifXIntIndex_) {
        weightScaleLocal = weightScaleQueue_.DeQue<float>();
    }
    if constexpr (hasActScale_) {
        activationScaleLocal = activationScaleQueue_.DeQue<float>();
    }
    if constexpr (hasBiasIndex_) {
        biasLocal = biasQueue_.DeQue<BiasType>();
    }
    __local_mem__ XType* x1Ptr = (__local_mem__ XType*)xLocal.GetPhyAddr(0);
    __local_mem__ XType* x2Ptr = (__local_mem__ XType*)xLocal.GetPhyAddr(xAlignLength_);
    __local_mem__ float* wScale1Ptr;
    __local_mem__ float* wScale2Ptr;
    if constexpr (ifXIntIndex_) {
        wScale1Ptr = (__local_mem__ float*)weightScaleLocal.GetPhyAddr(0);
        wScale2Ptr = (__local_mem__ float*)weightScaleLocal.GetPhyAddr(weightAlignLength_);
    }
    __local_mem__ float* aScalePtr = (__local_mem__ float*)activationScaleLocal.GetPhyAddr(0);
    __local_mem__ BiasType* bias1Ptr;
    __local_mem__ BiasType* bias2Ptr;
    if constexpr (hasBiasIndex_) {
        bias1Ptr = (__local_mem__ BiasType*)biasLocal.GetPhyAddr(0);
        bias2Ptr = (__local_mem__ BiasType*)biasLocal.GetPhyAddr(biasAlignLength_);
    }

    VF_CALL<DequantAndSwiGlu<XType, BiasType, ifXIntIndex_, ifXFloat16Index_, ifXBf16Index_, hasBiasIndex_,
                            ifBiasIntIndex_, hasActScale_, ifBiasFloatIndex_, ifBiasFloat16Index_, ifBiasBfloat16Index_>>(x1Ptr, x2Ptr,
                            wScale1Ptr, wScale2Ptr, aScalePtr, bias1Ptr, bias2Ptr, xTempPtr_, tileData);
    
    xQueue_.FreeTensor(xLocal);
    if constexpr (ifXIntIndex_) {
        weightScaleQueue_.FreeTensor(weightScaleLocal);
    }
    if constexpr (hasActScale_) {
        activationScaleQueue_.FreeTensor(activationScaleLocal);
    }
    if constexpr (hasBiasIndex_) {
        biasQueue_.FreeTensor(biasLocal);
    }
}

template <typename ActiScaleType, typename QuantScaleType, typename GroupType, typename BiasType, typename XType, typename YType>
__aicore__ inline void DSQDynamicNotFull<ActiScaleType, QuantScaleType, GroupType, BiasType, XType, YType>::UpdateReduceMax(uint32_t tileData)
{
    LocalTensor<float> quantScaleLocal;
    if constexpr (hasQuantScale_) {
        quantScaleLocal = quantScaleQueue_.DeQue<float>();
    }
    __local_mem__ float* qScalePtr = (__local_mem__ float*)quantScaleLocal.GetPhyAddr(0);
    float scalarMaxValue = GetScalarMaxNum();
    event_t eventID_S_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventID_S_V);
    WaitFlag<HardEvent::S_V>(eventID_S_V);
    VF_CALL<CalculateReduceMax<hasQuantScale_>>(xTempPtr_, qScalePtr, scalePtr_, tileData, scalarMaxValue);
    if constexpr (hasQuantScale_) {
        quantScaleQueue_.FreeTensor(quantScaleLocal);
    }
}

template <typename ActiScaleType, typename QuantScaleType, typename GroupType, typename BiasType, typename XType, typename YType>
__aicore__ inline void DSQDynamicNotFull<ActiScaleType, QuantScaleType, GroupType, BiasType, XType, YType>::CalculateResult(uint32_t tileData)
{
    LocalTensor<YType> yLocal = yQueue_.AllocTensor<YType>();
    LocalTensor<uint8_t> yFp4Local = yLocal.template ReinterpretCast<uint8_t>();
    scaleQueue_.EnQue<float>(scaleLocal_);
    scaleLocal_ = scaleQueue_.DeQue<float>();

    __local_mem__ YType* yPtr = (__local_mem__ YType*)yLocal.GetPhyAddr();
    __local_mem__ uint8_t* yFp4Ptr = (__local_mem__ uint8_t*)yFp4Local.GetPhyAddr();
    __local_mem__ float* scalePtr = (__local_mem__ float*)scaleLocal_.GetPhyAddr();
    VF_CALL<QuantAndCast<YType, ifYFloat8e4m3Index_, ifYFloat8e5m2Index_, ifYFloat4e2m1Index_, ifYFloat4e1m2Index_, ifYHiFloat8Index_>>(xTempPtr_, yPtr, yFp4Ptr, scalePtr, roundMode_, tileData);
    yQueue_.EnQue<YType>(yLocal);
}

template <typename ActiScaleType, typename QuantScaleType, typename GroupType, typename BiasType, typename XType, typename YType>
__aicore__ inline void DSQDynamicNotFull<ActiScaleType, QuantScaleType, GroupType, BiasType, XType, YType>::CopyInForXTemp(uint64_t gmOffset, uint32_t tileData)
{
    DataCopyPadExtParams<float> padParams{true, 0, 0, static_cast<float>(0)};
    DataCopyExtParams dataCopyParam;
    dataCopyParam.blockCount = 1;
    dataCopyParam.blockLen = tileData * sizeof(float);
    dataCopyParam.srcStride = 0;
    dataCopyParam.dstStride = 0;
    DataCopyPad(xTempLocal_, xTempGm_[gmOffset], dataCopyParam, padParams);
}

template <typename ActiScaleType, typename QuantScaleType, typename GroupType, typename BiasType, typename XType, typename YType>
__aicore__ inline void DSQDynamicNotFull<ActiScaleType, QuantScaleType, GroupType, BiasType, XType, YType>::CopyOutForXTemp(uint64_t gmOffset, uint32_t tileData)
{
    DataCopyExtParams dataCopyParam;
    dataCopyParam.blockCount = 1;
    dataCopyParam.blockLen = tileData * sizeof(float);
    dataCopyParam.srcStride = 0;
    dataCopyParam.dstStride = 0;
    DataCopyPad(xTempGm_[gmOffset], xTempLocal_, dataCopyParam);
}

template <typename ActiScaleType, typename QuantScaleType, typename GroupType, typename BiasType, typename XType, typename YType>
__aicore__ inline void DSQDynamicNotFull<ActiScaleType, QuantScaleType, GroupType, BiasType, XType, YType>::SetAlignLength(uint32_t tileData)
{
    if constexpr (ifXBf16Index_ || ifXFloat16Index_) {
        uint32_t BLOCK_ELEM_B32_BF = BLOCK_SIZE / sizeof(XType);
        xAlignLength_ = CeilDivision(tileData, BLOCK_ELEM_B32_BF) * BLOCK_ELEM_B32_BF;
    } else {
        xAlignLength_ = CeilDivision(tileData, BLOCK_ELEM_B32) * BLOCK_ELEM_B32;
    }
    weightAlignLength_ = CeilDivision(tileData, BLOCK_ELEM_B32) * BLOCK_ELEM_B32;
    uint32_t blockElem = BLOCK_SIZE / sizeof(BiasType);
    biasAlignLength_ = CeilDivision(tileData, blockElem) * blockElem;
}

template <typename ActiScaleType, typename QuantScaleType, typename GroupType, typename BiasType, typename XType, typename YType>
__aicore__ inline float DSQDynamicNotFull<ActiScaleType, QuantScaleType, GroupType, BiasType, XType, YType>::GetScalarMaxNum()
{
    //获取指定输出类型对应的最大值
    if constexpr (ifYFloat8e4m3Index_) {
        return 448.0;
    }
    if constexpr (ifYFloat8e5m2Index_) {
        return 57344.0;
    }
    if constexpr (ifYFloat4e2m1Index_) {
        return 6.0;
    }
    if constexpr (ifYFloat4e1m2Index_) {
        return 1.75;
    }
    if constexpr (ifYHiFloat8Index_) {
        return 32768.0;
    }
    
    return 127.0;
}

template <typename ActiScaleType, typename QuantScaleType, typename GroupType, typename BiasType, typename XType, typename YType>
__aicore__ inline void DSQDynamicNotFull<ActiScaleType, QuantScaleType, GroupType, BiasType, XType, YType>::CopyOutScale(uint64_t gmOffset)
{
    scaleQueue_.EnQue<float>(scaleLocal_);
    scaleLocal_ = scaleQueue_.DeQue<float>();
    DataCopyParams dataCopyScaleParams;
    dataCopyScaleParams.blockCount = 1;
    dataCopyScaleParams.blockLen = sizeof(float);
    dataCopyScaleParams.srcStride = 0;
    dataCopyScaleParams.dstStride = 0;
    DataCopyPad(scaleGm_[gmOffset], scaleLocal_[0], dataCopyScaleParams);
}

template <typename ActiScaleType, typename QuantScaleType, typename GroupType, typename BiasType, typename XType, typename YType>
__aicore__ inline void DSQDynamicNotFull<ActiScaleType, QuantScaleType, GroupType, BiasType, XType, YType>::CopyOutY(uint64_t gmOffset, uint32_t tileData)
{
    LocalTensor<YType> yLocal = yQueue_.DeQue<YType>();
    LocalTensor<uint8_t> yFp4Local = yLocal.template ReinterpretCast<uint8_t>();
    if constexpr (ifYFloat4e2m1Index_ || ifYFloat4e1m2Index_) {
        DataCopyParams dataCopyYParams;
        dataCopyYParams.blockCount = 1;
        dataCopyYParams.blockLen = tileData * sizeof(YType) / 2;
        dataCopyYParams.srcStride = 0;
        dataCopyYParams.dstStride = 0;
        DataCopyPad(yGm_.template ReinterpretCast<uint8_t>()[gmOffset / 2], yFp4Local[0], dataCopyYParams);
        yQueue_.FreeTensor(yLocal);
    } else {
        DataCopyParams dataCopyYParams;
        dataCopyYParams.blockCount = 1;
        dataCopyYParams.blockLen = tileData * sizeof(YType);
        dataCopyYParams.srcStride = 0;
        dataCopyYParams.dstStride = 0;
        DataCopyPad(yGm_[gmOffset], yLocal[0], dataCopyYParams);
        yQueue_.FreeTensor(yLocal);
    }
}

template <typename ActiScaleType, typename QuantScaleType, typename GroupType, typename BiasType, typename XType, typename YType>
__aicore__ inline void DSQDynamicNotFull<ActiScaleType, QuantScaleType, GroupType, BiasType, XType, YType>::CopyInV2(int64_t groupIdx, size_t rowIndex, uint64_t loop, uint32_t tileData, int64_t blockLength)
{
    int64_t xGmOffset = rowIndex * xLength_;
    int64_t ubOffset = 2 * loop * blockLength;
    DataCopyPadParams padParams{false, 0, 0, 0};
    // copy_in: x(xDimPerLoop, H)
    LocalTensor<XType> xLocal = xQueue_.AllocTensor<XType>();
    DataCopyParams dataCopyXParams;
    dataCopyXParams.blockCount = 1;
    dataCopyXParams.blockLen = 2 * tileData * sizeof(XType);
    dataCopyXParams.srcStride = 0;
    dataCopyXParams.dstStride = 0;
    DataCopyPad(xLocal[0], xGm_[xGmOffset + ubOffset], dataCopyXParams, padParams); // 激活部分
    xQueue_.EnQue(xLocal);
    // copy_in: weight_scale(G, H)
    if constexpr (ifXIntIndex_) {
        LocalTensor<float> weightScaleLocal = weightScaleQueue_.AllocTensor<float>();
        DataCopyParams dataCopyWeightScaleParams;
        dataCopyWeightScaleParams.blockCount = 1;
        dataCopyWeightScaleParams.blockLen = 2 * tileData * sizeof(float);
        dataCopyWeightScaleParams.srcStride = 0;
        dataCopyWeightScaleParams.dstStride = 0;
        DataCopyPad(weightScaleLocal[0], weightScaleGm_[groupIdx * xLength_ + ubOffset], dataCopyWeightScaleParams, padParams); // 激活部分
        weightScaleQueue_.EnQue(weightScaleLocal);
    }
    // copy_in: activation_scale(BS,)
    if constexpr (hasActScale_) {
        LocalTensor<float> activationScaleLocal = activationScaleQueue_.AllocTensor<float>();
        DataCopyParams dataCopyActScaleParams;
        dataCopyActScaleParams.blockCount = 1;
        dataCopyActScaleParams.blockLen = sizeof(float);
        dataCopyActScaleParams.srcStride = 0;
        dataCopyActScaleParams.dstStride = 0;
        DataCopyPad(activationScaleLocal[0], activationScaleGm_[rowIndex], dataCopyActScaleParams, padParams);
        activationScaleQueue_.EnQue(activationScaleLocal);
    }
    // copy_in: quant_scale(G, H)
    if constexpr (hasQuantScale_) {
        LocalTensor<QuantScaleType> quantScaleLocal = quantScaleQueue_.AllocTensor<QuantScaleType>();
        DataCopyParams dataCopyQuantScaleParams;
        dataCopyQuantScaleParams.blockCount = 1;
        dataCopyQuantScaleParams.blockLen = tileData * sizeof(QuantScaleType);
        dataCopyQuantScaleParams.srcStride = 0;
        dataCopyQuantScaleParams.dstStride = 0;
        DataCopyPad(quantScaleLocal[0], quantScaleGm_[groupIdx * yLength_ + loop * blockLength], dataCopyQuantScaleParams, padParams);
        quantScaleQueue_.EnQue(quantScaleLocal);
    }
    // copy_in: bias(1, 2H)->(1, H), (1, H)
    if constexpr (hasBiasIndex_) {
        LocalTensor<BiasType> biasLocal = biasQueue_.AllocTensor<BiasType>();
        DataCopyParams dataCopyBiasParams;
        dataCopyBiasParams.blockCount = 1;
        dataCopyBiasParams.blockLen = 2 * tileData * sizeof(BiasType);
        dataCopyBiasParams.srcStride = 0;
        dataCopyBiasParams.dstStride = 0;
        DataCopyPad(biasLocal[0], biasGm_[groupIdx * xLength_ + ubOffset], dataCopyBiasParams, padParams);
        biasQueue_.EnQue(biasLocal);
    }  
}

template <typename ActiScaleType, typename QuantScaleType, typename GroupType, typename BiasType, typename XType, typename YType>
__aicore__ inline void DSQDynamicNotFull<ActiScaleType, QuantScaleType, GroupType, BiasType, XType, YType>::PreProcessV2(uint32_t tileData)
{
    LocalTensor<XType> xLocal;
    LocalTensor<float> weightScaleLocal;
    LocalTensor<float> activationScaleLocal;
    LocalTensor<BiasType> biasLocal;
    xLocal = xQueue_.DeQue<XType>();
    if constexpr (ifXIntIndex_) {
        weightScaleLocal = weightScaleQueue_.DeQue<float>();
    }
    if constexpr (hasActScale_) {
        activationScaleLocal = activationScaleQueue_.DeQue<float>();
    }
    if constexpr (hasBiasIndex_) {
        biasLocal = biasQueue_.DeQue<BiasType>();
    }

    __local_mem__ XType* xPtr = (__local_mem__ XType*)xLocal.GetPhyAddr(0);
    __local_mem__ float* wScalePtr;
    if constexpr (ifXIntIndex_) {
        wScalePtr = (__local_mem__ float*)weightScaleLocal.GetPhyAddr(0);
    }
    __local_mem__ float* aScalePtr = (__local_mem__ float*)activationScaleLocal.GetPhyAddr(0);
    __local_mem__ BiasType* biasPtr;
    if constexpr (hasBiasIndex_) {
        biasPtr = (__local_mem__ BiasType*)biasLocal.GetPhyAddr(0);
    }

    VF_CALL<DequantAndSwiGluV2<XType, BiasType, ifXIntIndex_, ifXFloat16Index_, ifXBf16Index_, hasBiasIndex_,
                            ifBiasIntIndex_, hasActScale_, ifBiasFloatIndex_, ifBiasFloat16Index_, ifBiasBfloat16Index_>>(xPtr,
                            wScalePtr, aScalePtr, biasPtr, clampLimit_, gluAlpha_, gluBias_, xTempPtr_, tileData);
    
    xQueue_.FreeTensor(xLocal);
    if constexpr (ifXIntIndex_) {
        weightScaleQueue_.FreeTensor(weightScaleLocal);
    }
    if constexpr (hasActScale_) {
        activationScaleQueue_.FreeTensor(activationScaleLocal);
    }
    if constexpr (hasBiasIndex_) {
        biasQueue_.FreeTensor(biasLocal);
    }
}
}  // namespace DequantSwigluQuantV35Ops
#endif  // DEQUANT_SWIGLU_QUANT_H