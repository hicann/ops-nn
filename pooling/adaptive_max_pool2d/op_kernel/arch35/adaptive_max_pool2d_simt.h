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
 * \file adaptive_max_pool2d_simt.h
 * \brief
 */

#ifndef ADAPTIVE_MAX_POOL_2D_SIMT_H
#define ADAPTIVE_MAX_POOL_2D_SIMT_H

#include "kernel_operator.h"
#include "../inc/load_store_utils.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"

#ifdef __CCE_KT_TEST__
#define LAUNCH_BOUND(threads)
#endif

namespace AdaptiveMaxPool2DWithSimt{
    using namespace AscendC;

    constexpr uint32_t THREAD_DIM = 1024;
    constexpr size_t PARAM_NUM = 4;
    constexpr static uint32_t DIV_HW_IDX = 0;
    constexpr static uint32_t DIV_H_IDX = 2;
    constexpr static uint32_t DIV_W_IDX = 4;

    template <typename DIV_T>
    __aicore__ __attribute__((always_inline)) inline static DIV_T startIndex(DIV_T outIdx, DIV_T magicOutLen, DIV_T shiftOutLen, DIV_T inLen)
    {
        DIV_T pStart = outIdx * inLen;
        return Simt::UintDiv<DIV_T>(pStart, magicOutLen, shiftOutLen);
    }

    template <typename DIV_T>
    __aicore__ __attribute__((always_inline)) inline static DIV_T endIndex(DIV_T outIdx, DIV_T magicOutLen, DIV_T shiftOutLen, DIV_T inLen)
    {
        DIV_T pEnd = ((outIdx + 1) * inLen - 1);
        pEnd = Simt::UintDiv<DIV_T>(pEnd, magicOutLen, shiftOutLen);
        return pEnd + 1;
    }

    template <typename VALUE_T, typename INDICES_T, typename FORMAT_T, typename DIV_T>
    __aicore__ __attribute__((always_inline)) inline static void executeFunc(FORMAT_T count, __gm__ VALUE_T* bottomData, 
                                                                             FORMAT_T ncSize, FORMAT_T height, FORMAT_T width, 
                                                                             FORMAT_T outputNc, FORMAT_T outputHeight, FORMAT_T outputWidth, 
                                                                             __gm__ VALUE_T* valueData, __gm__ INDICES_T* indicesData,
                                                                             DIV_T magicHW, DIV_T shiftHW, DIV_T magicH, DIV_T shiftH, 
                                                                             DIV_T magicW, DIV_T shiftW) {
        for (DIV_T index = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); index < count;
            index += Simt::GetBlockNum() * Simt::GetThreadNum()) {
            // 计算NC索引 (index / (outputHeight * outputWidth))
            DIV_T ncId = Simt::UintDiv<DIV_T>(index, magicHW, shiftHW); 
            DIV_T indexIdx = index - ncId * outputNc;  // 在当前NC内的偏移
            
            // 计算高度和宽度索引 (indexIdx = h * outputWidth + w)
            DIV_T hId = Simt::UintDiv<DIV_T>(indexIdx, magicW, shiftW); 
            DIV_T wId = indexIdx - hId * outputWidth;
            
            // 计算输入窗口的起始和结束位置
            FORMAT_T startInH = startIndex<DIV_T>(hId, magicH, shiftH, height);
            FORMAT_T endInH = endIndex<DIV_T>(hId, magicH, shiftH, height);
            FORMAT_T startInW = startIndex<DIV_T>(wId, magicW, shiftW, width);
            FORMAT_T endInW = endIndex<DIV_T>(wId, magicW, shiftW, width);
            
            VALUE_T maxVal = AscendC::NumericLimits<VALUE_T>::NegativeInfinity();
            FORMAT_T maxIdx = startInH * width + startInW;  // 2D索引: h * width + w
            
            auto ncStartData = bottomData + ncId * ncSize;

            // 2D池化: 只在高度和宽度上滑动
            for (FORMAT_T h = startInH; h < endInH; ++h) {
                for (FORMAT_T w = startInW; w < endInW; ++w) {
                    FORMAT_T idxOffset = h * width + w;  // 2D索引
                    VALUE_T val = static_cast<VALUE_T>(ncStartData[idxOffset]);
                    if ((static_cast<VALUE_T>(val) > maxVal) || Simt::IsNan(static_cast<float>(val))) {
                        maxIdx = idxOffset;
                        maxVal = val;
                    }
                }
            } 
            valueData[index] = static_cast<VALUE_T>(maxVal);
            indicesData[index] = static_cast<INDICES_T>(maxIdx);
        }
    }

template <typename VALUE_T, typename INDICES_T, typename FORMAT_T, typename DIV_T>
class AdaptiveMaxPool2DSimt
{
public:
    __aicore__ inline AdaptiveMaxPool2DSimt(TPipe *pipe, const AdaptiveMaxPool2dTilingData* __restrict tilingData) 
        : pipe_(pipe), tilingData_(tilingData)
    {
    }
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR indices);
    __aicore__ inline void Process();

private:
    TPipe *pipe_;
    AscendC::GlobalTensor<VALUE_T> x_;
    AscendC::GlobalTensor<VALUE_T> y_;
    AscendC::GlobalTensor<INDICES_T> indices_;
    const AdaptiveMaxPool2dTilingData* tilingData_;
    TBuf<TPosition::VECCALC> paramBuf_;
};

template <typename VALUE_T, typename INDICES_T, typename FORMAT_T, typename DIV_T>
__aicore__ inline void AdaptiveMaxPool2DSimt<VALUE_T, INDICES_T, FORMAT_T, DIV_T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR indices)
{
    x_.SetGlobalBuffer((__gm__ VALUE_T*)(x));
    y_.SetGlobalBuffer((__gm__ VALUE_T*)(y));
    indices_.SetGlobalBuffer((__gm__ INDICES_T*)(indices));
    if constexpr (!(std::is_same<FORMAT_T, int32_t>::value && std::is_same<DIV_T, uint32_t>::value)){
       pipe_->InitBuffer(paramBuf_, PARAM_NUM * sizeof(DIV_T));
    }
}

template <typename VALUE_T, typename INDICES_T, typename FORMAT_T, typename DIV_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void AdaptiveMaxPool2DNchwUb(FORMAT_T count, __gm__ VALUE_T* bottomData, 
                                                                                    FORMAT_T ncSize, FORMAT_T height, FORMAT_T width, 
                                                                                    FORMAT_T outputNc, FORMAT_T outputHeight, FORMAT_T outputWidth, 
                                                                                    __gm__ VALUE_T* valueData, __gm__ INDICES_T* indicesData, 
                                                                                    __ubuf__ DIV_T* SimtParam)
{
    DIV_T magicHW = SimtParam[DIV_HW_IDX];
    DIV_T shiftHW = SimtParam[DIV_HW_IDX + 1];
    DIV_T magicH = SimtParam[DIV_H_IDX];
    DIV_T shiftH = SimtParam[DIV_H_IDX + 1];
    DIV_T magicW = SimtParam[DIV_W_IDX];
    DIV_T shiftW = SimtParam[DIV_W_IDX + 1];
    executeFunc<VALUE_T, INDICES_T, FORMAT_T, DIV_T>(count, bottomData, ncSize, height, width, 
                                                     outputNc, outputHeight, outputWidth, 
                                                     valueData, indicesData,
                                                     magicHW, shiftHW, magicH, shiftH, magicW, shiftW); 
}

template <typename VALUE_T, typename INDICES_T, typename FORMAT_T, typename DIV_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void AdaptiveMaxPool2DNchwFunc(FORMAT_T count, __gm__ VALUE_T* bottomData, 
                                                                                      FORMAT_T ncSize, FORMAT_T height, FORMAT_T width, 
                                                                                      FORMAT_T outputNc, FORMAT_T outputHeight, FORMAT_T outputWidth, 
                                                                                      __gm__ VALUE_T* valueData, __gm__ INDICES_T* indicesData,
                                                                                      DIV_T magicHW, DIV_T shiftHW, DIV_T magicH, DIV_T shiftH, 
                                                                                      DIV_T magicW, DIV_T shiftW)
{
    executeFunc<VALUE_T, INDICES_T, FORMAT_T, DIV_T>(count, bottomData, ncSize, height, width, 
                                                     outputNc, outputHeight, outputWidth, 
                                                     valueData, indicesData,
                                                     magicHW, shiftHW, magicH, shiftH, magicW, shiftW);
}

template <typename VALUE_T, typename INDICES_T, typename FORMAT_T, typename DIV_T>
__aicore__ inline void AdaptiveMaxPool2DSimt<VALUE_T, INDICES_T, FORMAT_T, DIV_T>::Process()
{
    auto inputData = (__gm__ VALUE_T*)x_.GetPhyAddr();
    auto outputData = (__gm__ VALUE_T*)y_.GetPhyAddr();
    auto indicesData = (__gm__ INDICES_T*)indices_.GetPhyAddr();
    
    DIV_T hw = tilingData_->Ho * tilingData_->Wo;  // H * W
    FORMAT_T totalSize = hw * tilingData_->N * tilingData_->C;  // N * C * H * W
    FORMAT_T ncSize = tilingData_->Hi * tilingData_->Wi;  // 每个N*C的2D数据大小
    
    DIV_T magicHW = 0;
    DIV_T shiftHW = 0;
    DIV_T magicH = 0;
    DIV_T shiftH = 0;
    DIV_T magicW = 0;
    DIV_T shiftW = 0;
    
    GetUintDivMagicAndShift<DIV_T>(magicHW, shiftHW, hw);
    GetUintDivMagicAndShift<DIV_T>(magicH, shiftH, tilingData_->Ho);
    GetUintDivMagicAndShift<DIV_T>(magicW, shiftW, tilingData_->Wo);
    
    if constexpr (std::is_same<FORMAT_T, int32_t>::value && std::is_same<DIV_T, uint32_t>::value){
        Simt::VF_CALL<AdaptiveMaxPool2DNchwFunc<VALUE_T, INDICES_T, FORMAT_T, DIV_T>>(
            Simt::Dim3(THREAD_DIM), 
            static_cast<FORMAT_T>(totalSize), 
            inputData, 
            ncSize, 
            static_cast<FORMAT_T>(tilingData_->Hi), 
            static_cast<FORMAT_T>(tilingData_->Wi),
            static_cast<FORMAT_T>(hw), 
            static_cast<FORMAT_T>(tilingData_->Ho), 
            static_cast<FORMAT_T>(tilingData_->Wo), 
            outputData, 
            indicesData,
            magicHW, shiftHW, magicH, shiftH, magicW, shiftW
        );
    } else {
        LocalTensor<DIV_T> SimtParam = paramBuf_.Get<DIV_T>();
        SimtParam.SetValue(0, static_cast<DIV_T>(magicHW));
        SimtParam.SetValue(1, static_cast<DIV_T>(shiftHW));
        SimtParam.SetValue(DIV_H_IDX, static_cast<DIV_T>(magicH));
        SimtParam.SetValue(DIV_H_IDX + 1, static_cast<DIV_T>(shiftH));
        SimtParam.SetValue(DIV_W_IDX, static_cast<DIV_T>(magicW));
        SimtParam.SetValue(DIV_W_IDX + 1, static_cast<DIV_T>(shiftW));
        DataSyncBarrier<MemDsbT::UB>();
        
        Simt::VF_CALL<AdaptiveMaxPool2DNchwUb<VALUE_T, INDICES_T, FORMAT_T, DIV_T>>(
            Simt::Dim3(THREAD_DIM),
            static_cast<FORMAT_T>(totalSize), 
            inputData, 
            ncSize, 
            static_cast<FORMAT_T>(tilingData_->Hi), 
            static_cast<FORMAT_T>(tilingData_->Wi),
            static_cast<FORMAT_T>(hw), 
            static_cast<FORMAT_T>(tilingData_->Ho), 
            static_cast<FORMAT_T>(tilingData_->Wo), 
            outputData, 
            indicesData,
            (__ubuf__ DIV_T*)SimtParam.GetPhyAddr()
        );
    } 
}

}

#endif // ADAPTIVE_MAX_POOL_2D_SIMT_H
