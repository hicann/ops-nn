/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file adaptive_max_pool3d_simt.h
 * \brief adaptive_max_pool3d implied by simt
 */

#ifndef ADAPTIVE_MAX_POOL_3D_SIMT_H
#define ADAPTIVE_MAX_POOL_3D_SIMT_H

#include "kernel_operator.h"
#include "../inc/load_store_utils.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "adaptive_pool3d_tiling_struct.h"

#ifdef __CCE_KT_TEST__
#define LAUNCH_BOUND(threads)
#endif

namespace AdaptivePool3DWithSimt{
    using namespace AscendC;

    constexpr uint32_t THREAD_DIM = 1024;
    constexpr size_t PARAM_NUM = 8;
    constexpr static uint32_t DIV_DHW_IDX = 0;
 	constexpr static uint32_t DIV_D_IDX = 2;
 	constexpr static uint32_t DIV_H_IDX = 4;
 	constexpr static uint32_t DIV_W_IDX = 6;

    template <typename DIV_T>
    __aicore__ __attribute__((always_inline)) inline static DIV_T startIndex(DIV_T outIdx, DIV_T magicOutLen, DIV_T shiftOutLen,  DIV_T inLen)
    {
        DIV_T pStart = outIdx * inLen;
        return Simt::UintDiv<DIV_T>(pStart, magicOutLen, shiftOutLen);
    }

    template <typename DIV_T>
    __aicore__ __attribute__((always_inline)) inline static DIV_T endIndex(DIV_T outIdx, DIV_T magicOutLen, DIV_T shiftOutLen,  DIV_T inLen)
    {
        DIV_T pEnd = ((outIdx + 1) * inLen - 1);
        pEnd = Simt::UintDiv<DIV_T>(pEnd, magicOutLen, shiftOutLen);
        return pEnd + 1;
    }

    template <typename VALUE_T, typename INDICES_T, typename FORMAT_T, typename DIV_T>
    __aicore__ __attribute__((always_inline)) inline static void executeFunc(FORMAT_T count, __gm__ VALUE_T* bottomData, FORMAT_T ncSize, FORMAT_T depth,
                                                                                FORMAT_T height, FORMAT_T width, FORMAT_T outputNc, FORMAT_T outputDep,  FORMAT_T outputHeight, FORMAT_T outputWidth, 
                                                                               __gm__ VALUE_T* valueData, __gm__ INDICES_T* indicesData,
                                                                               DIV_T magicDHW, DIV_T shiftDHW, DIV_T magicD, DIV_T shiftD, DIV_T magicH, DIV_T shiftH, DIV_T magicW, DIV_T shiftW) {
        for (DIV_T index = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); index < count;
            index += Simt::GetBlockNum() * Simt::GetThreadNum()) {
            DIV_T ncId = Simt::UintDiv<DIV_T>(index, magicDHW, shiftDHW); 
            DIV_T indexIdx = index - ncId * outputNc;
            DIV_T dim0Idx = Simt::UintDiv<DIV_T>(indexIdx, magicW, shiftW); 
            DIV_T wId = indexIdx - dim0Idx * outputWidth;
            DIV_T dId = Simt::UintDiv<DIV_T>(dim0Idx, magicH, shiftH); 
            DIV_T hId = dim0Idx - dId * outputHeight;

            FORMAT_T startInD = startIndex<DIV_T>(dId, magicD, shiftD, depth);
            FORMAT_T endInD = endIndex<DIV_T>(dId, magicD, shiftD, depth);
            FORMAT_T startInH = startIndex<DIV_T>(hId, magicH, shiftH, height);
            FORMAT_T endInH = endIndex<DIV_T>(hId, magicH, shiftH, height);
            FORMAT_T startInW = startIndex<DIV_T>(wId, magicW, shiftW, width);
            FORMAT_T endInW = endIndex<DIV_T>(wId, magicW, shiftW, width);
            VALUE_T maxVal = AscendC::NumericLimits<VALUE_T>::NegativeInfinity();
            FORMAT_T maxIdx =  startInD * height * width + startInH * width + startInW; 
            auto ncStartData = bottomData + ncId * ncSize;

            for (FORMAT_T d = startInD; d < endInD; ++d) {
                for (FORMAT_T h = startInH; h < endInH; ++h) {
                    for (FORMAT_T w = startInW; w < endInW; ++w) {
                        FORMAT_T idxOffset = d * height * width + h * width + w;
                        VALUE_T val = static_cast<VALUE_T>(ncStartData[idxOffset]);
                        if ((static_cast<VALUE_T>(val) > maxVal) || Simt::IsNan(static_cast<float>(val))) {
                            maxIdx = idxOffset;
                            maxVal = val;
                        }
                    }
                }
            } 
            valueData[index] = static_cast<VALUE_T>(maxVal);
            indicesData[index] = static_cast<INDICES_T>(maxIdx);
        }
    }

template <typename VALUE_T, typename INDICES_T, typename FORMAT_T, typename DIV_T>
class AdaptivePool3DSimt
{
public:
    __aicore__ inline AdaptivePool3DSimt(TPipe *pipe, const AdaptivePool3DTiling::AdaptivePool3DSimtTilingData* __restrict tilingData) 
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
    const AdaptivePool3DTiling::AdaptivePool3DSimtTilingData* tilingData_;
    TBuf<TPosition::VECCALC> paramBuf_;
};

template <typename VALUE_T, typename INDICES_T, typename FORMAT_T, typename DIV_T>
__aicore__ inline void AdaptivePool3DSimt<VALUE_T, INDICES_T, FORMAT_T, DIV_T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR indices)
{
    x_.SetGlobalBuffer((__gm__ VALUE_T*)(x));
    y_.SetGlobalBuffer((__gm__ VALUE_T*)(y));
    indices_.SetGlobalBuffer((__gm__ INDICES_T*)(indices));
    if constexpr (!(std::is_same<FORMAT_T, int32_t>::value && std::is_same<DIV_T, uint32_t>::value)){
       pipe_->InitBuffer(paramBuf_, PARAM_NUM * sizeof(DIV_T));
    }
}

template <typename VALUE_T, typename INDICES_T, typename FORMAT_T, typename DIV_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void AdaptiveMaxPool3DNcdhwUb(FORMAT_T count, __gm__ VALUE_T* bottomData, FORMAT_T ncSize, FORMAT_T depth,
                                                                                FORMAT_T height, FORMAT_T width, FORMAT_T outputNc, FORMAT_T outputDep,  FORMAT_T outputHeight, FORMAT_T outputWidth, 
                                                                               __gm__ VALUE_T* valueData, __gm__ INDICES_T* indicesData, __ubuf__ DIV_T* SimtParam)
{
    DIV_T magicDHW = SimtParam[DIV_DHW_IDX];
    DIV_T shiftDHW = SimtParam[DIV_DHW_IDX + 1];
    DIV_T magicD = SimtParam[DIV_D_IDX];
    DIV_T shiftD = SimtParam[DIV_D_IDX + 1];
    DIV_T magicH = SimtParam[DIV_H_IDX];
    DIV_T shiftH = SimtParam[DIV_H_IDX + 1];
    DIV_T magicW = SimtParam[DIV_W_IDX];
    DIV_T shiftW = SimtParam[DIV_W_IDX + 1];
    executeFunc<VALUE_T, INDICES_T, FORMAT_T, DIV_T>(count, bottomData, ncSize, depth,height, width, outputNc, outputDep,  outputHeight, outputWidth, valueData, indicesData,
                                                        magicDHW, shiftDHW, magicD, shiftD, magicH, shiftH, magicW, shiftW); 
}

template <typename VALUE_T, typename INDICES_T, typename FORMAT_T, typename DIV_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void AdaptiveMaxPool3DNcdhwFunc(FORMAT_T count, __gm__ VALUE_T* bottomData, FORMAT_T ncSize, FORMAT_T depth,
                                                                                FORMAT_T height, FORMAT_T width, FORMAT_T outputNc, FORMAT_T outputDep,  FORMAT_T outputHeight, FORMAT_T outputWidth, 
                                                                               __gm__ VALUE_T* valueData, __gm__ INDICES_T* indicesData,
                                                                               DIV_T magicDHW, DIV_T shiftDHW, DIV_T magicD, DIV_T shiftD, DIV_T magicH, DIV_T shiftH, DIV_T magicW, DIV_T shiftW)
{
    executeFunc<VALUE_T, INDICES_T, FORMAT_T, DIV_T>(count, bottomData, ncSize, depth,height, width, outputNc, outputDep,  outputHeight, outputWidth, valueData, indicesData,
                                                        magicDHW, shiftDHW, magicD, shiftD, magicH, shiftH, magicW, shiftW);
}

template <typename VALUE_T, typename INDICES_T, typename FORMAT_T, typename DIV_T>
__aicore__ inline void AdaptivePool3DSimt<VALUE_T, INDICES_T, FORMAT_T, DIV_T>::Process()
{
    auto inputData = (__gm__ VALUE_T*)x_.GetPhyAddr();
    auto outputData = (__gm__ VALUE_T*)y_.GetPhyAddr();
    auto indicesData = (__gm__ INDICES_T*)indices_.GetPhyAddr();
    DIV_T dhw = tilingData_->dOutDim * tilingData_->hOutDim * tilingData_->wOutDim;
    FORMAT_T totalSize = dhw * tilingData_->nDim * tilingData_->cDim;
    FORMAT_T ncSize = tilingData_->dInDim * tilingData_->hInDim * tilingData_->wInDim;
    DIV_T magicDHW = 0;
    DIV_T shiftDHW = 0;
    DIV_T magicD = 0;
    DIV_T shiftD = 0;
    DIV_T magicH = 0;
    DIV_T shiftH = 0;
    DIV_T magicW = 0;
    DIV_T shiftW = 0;
    GetUintDivMagicAndShift<DIV_T>(magicDHW, shiftDHW, dhw);
    GetUintDivMagicAndShift<DIV_T>(magicD, shiftD, tilingData_->dOutDim);
    GetUintDivMagicAndShift<DIV_T>(magicH, shiftH, tilingData_->hOutDim);
    GetUintDivMagicAndShift<DIV_T>(magicW, shiftW, tilingData_->wOutDim);
    if constexpr (std::is_same<FORMAT_T, int32_t>::value && std::is_same<DIV_T, uint32_t>::value){
        Simt::VF_CALL<AdaptiveMaxPool3DNcdhwFunc<VALUE_T, INDICES_T, FORMAT_T, DIV_T>>(Simt::Dim3(THREAD_DIM), static_cast<FORMAT_T>(totalSize), inputData, ncSize, static_cast<FORMAT_T>(tilingData_->dInDim), static_cast<FORMAT_T>(tilingData_->hInDim), static_cast<FORMAT_T>(tilingData_->wInDim),
                                                                        static_cast<FORMAT_T>(dhw), static_cast<FORMAT_T>(tilingData_->dOutDim), static_cast<FORMAT_T>(tilingData_->hOutDim), static_cast<FORMAT_T>(tilingData_->wOutDim), outputData, indicesData,
                                                                        magicDHW, shiftDHW, magicD, shiftD, magicH, shiftH, magicW, shiftW
                                                                        );
    }else{
        LocalTensor<DIV_T> SimtParam = paramBuf_.Get<DIV_T>();
        SimtParam.SetValue(0, static_cast<DIV_T>(magicDHW));
        SimtParam.SetValue(1, static_cast<DIV_T>(shiftDHW));
        SimtParam.SetValue(DIV_D_IDX, static_cast<DIV_T>(magicD));
        SimtParam.SetValue(DIV_D_IDX + 1, static_cast<DIV_T>(shiftD));
        SimtParam.SetValue(DIV_H_IDX, static_cast<DIV_T>(magicH));
        SimtParam.SetValue(DIV_H_IDX + 1, static_cast<DIV_T>(shiftH));
        SimtParam.SetValue(DIV_W_IDX, static_cast<DIV_T>(magicW));
        SimtParam.SetValue(DIV_W_IDX + 1, static_cast<DIV_T>(shiftW));
        DataSyncBarrier<MemDsbT::UB>();
        Simt::VF_CALL<AdaptiveMaxPool3DNcdhwUb<VALUE_T, INDICES_T, FORMAT_T, DIV_T>>(Simt::Dim3(THREAD_DIM),static_cast<FORMAT_T>(totalSize), inputData, ncSize, static_cast<FORMAT_T>(tilingData_->dInDim), static_cast<FORMAT_T>(tilingData_->hInDim), static_cast<FORMAT_T>(tilingData_->wInDim),
                                                                        static_cast<FORMAT_T>(dhw), static_cast<FORMAT_T>(tilingData_->dOutDim), static_cast<FORMAT_T>(tilingData_->hOutDim), static_cast<FORMAT_T>(tilingData_->wOutDim), outputData, indicesData,
                                                                        (__ubuf__ DIV_T*)SimtParam.GetPhyAddr()
                                                                        );
    } 
}

}

#endif //ADAPTIVE_MAX_POOL_3D_SIMT_H