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
 * \file adaptive_max_pool3d_grad_simt.h
 * \brief adaptive_max_pool3d_grad implied by simt
 */

#ifndef ADAPTIVE_MAX_POOL3D_GRAD_SIMT_H
#define ADAPTIVE_MAX_POOL3D_GRAD_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/load_store_utils.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "adaptive_max_pool3d_grad_struct.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"

using namespace AscendC;

namespace AdaptiveMaxPool3dGradOp {
constexpr static uint32_t THREAD_DIM = 512;
constexpr static uint32_t TILING_DATA_NUM = 10;

template <typename OFFSET_T>
using SimtDivT = typename std::conditional<std::is_same<OFFSET_T, int32_t>::value, uint32_t, uint64_t>::type;

template <typename VALUE_T, typename OFFSET_T>
class AdaptiveMaxPool3dGradSimt {
public:
    __aicore__ inline AdaptiveMaxPool3dGradSimt(TPipe* pipe,
                                                const AdaptiveMaxPool3dGradTilingDataV35* __restrict__ tilingData)
        : pipe_(pipe), tilingData_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    TPipe* pipe_;
    AscendC::GlobalTensor<VALUE_T> xGrad_;
    AscendC::GlobalTensor<VALUE_T> grad_;
    AscendC::GlobalTensor<OFFSET_T> argmax_;
    const AdaptiveMaxPool3dGradTilingDataV35* tilingData_;
    TBuf<TPosition::VECCALC> simtTilingDataBuf_;
};

template <typename VALUE_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void InitOutputZeroSimt(const int64_t outputSize,
                                                                               __gm__ VALUE_T* xGradData)
{
    using INDEX_T = uint64_t;
    const INDEX_T threadStart = static_cast<INDEX_T>(Simt::GetBlockIdx()) * static_cast<INDEX_T>(Simt::GetThreadNum()) +
                                static_cast<INDEX_T>(Simt::GetThreadIdx());
    const INDEX_T threadStride = static_cast<INDEX_T>(Simt::GetBlockNum()) * static_cast<INDEX_T>(Simt::GetThreadNum());

    for (INDEX_T i = threadStart; i < static_cast<INDEX_T>(outputSize); i += threadStride) {
        xGradData[i] = (VALUE_T)0;
    }
    __syncthreads();
    if (Simt::GetThreadIdx() == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
}

template <typename VALUE_T, typename OFFSET_T, bool IsDeterministic>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM) inline void AdaptiveMaxPool3dGradKernelSimt(
    const __gm__ VALUE_T* gradData, const __gm__ OFFSET_T* argmaxData, const SimtDivT<OFFSET_T> nDims,
    const SimtDivT<OFFSET_T> cDims, const SimtDivT<OFFSET_T> inD, const SimtDivT<OFFSET_T> inH,
    const SimtDivT<OFFSET_T> inW, const SimtDivT<OFFSET_T> outD, const SimtDivT<OFFSET_T> outH,
    const SimtDivT<OFFSET_T> outW, const int64_t isOverLap, __gm__ VALUE_T* xGradData)
{
    using DIV_T = SimtDivT<OFFSET_T>;
    using INDEX_T = uint64_t;

    const INDEX_T totalNC = static_cast<INDEX_T>(nDims) * static_cast<INDEX_T>(cDims);
    const INDEX_T dhOutPlaneSize = static_cast<INDEX_T>(inD) * static_cast<INDEX_T>(inH) * static_cast<INDEX_T>(inW);
    const INDEX_T inPlaneSize = static_cast<INDEX_T>(outD) * static_cast<INDEX_T>(outH) * static_cast<INDEX_T>(outW);
    const INDEX_T cStride = static_cast<INDEX_T>(cDims) * inPlaneSize;

    const INDEX_T threadStart = static_cast<INDEX_T>(Simt::GetBlockIdx()) * static_cast<INDEX_T>(Simt::GetThreadNum()) +
                                static_cast<INDEX_T>(Simt::GetThreadIdx());
    const INDEX_T threadStride = static_cast<INDEX_T>(Simt::GetBlockNum()) * static_cast<INDEX_T>(Simt::GetThreadNum());

    if constexpr (IsDeterministic) {
        for (INDEX_T ncIdx = threadStart; ncIdx < totalNC; ncIdx += threadStride) {
            const DIV_T n = static_cast<DIV_T>(ncIdx / static_cast<INDEX_T>(cDims));
            const DIV_T c = static_cast<DIV_T>(ncIdx - static_cast<INDEX_T>(n) * static_cast<INDEX_T>(cDims));
            const INDEX_T ncGradOffset = ncIdx * dhOutPlaneSize;
            const INDEX_T ncXGradOffset = static_cast<INDEX_T>(n) * cStride + static_cast<INDEX_T>(c) * inPlaneSize;

            for (INDEX_T dhwIdx = 0; dhwIdx < dhOutPlaneSize; dhwIdx++) {
                OFFSET_T maxIdx = argmaxData[ncGradOffset + dhwIdx];
                INDEX_T inputIdx = ncXGradOffset + static_cast<INDEX_T>(maxIdx);
                xGradData[inputIdx] += gradData[ncGradOffset + dhwIdx];
            }
        }
    } else {
        const INDEX_T outCount = totalNC * dhOutPlaneSize;
        for (INDEX_T index = threadStart; index < outCount; index += threadStride) {
            const INDEX_T ncOffset = index / dhOutPlaneSize;
            const DIV_T n = static_cast<DIV_T>(ncOffset / static_cast<INDEX_T>(cDims));
            const DIV_T c = static_cast<DIV_T>(ncOffset - static_cast<INDEX_T>(n) * static_cast<INDEX_T>(cDims));

            OFFSET_T maxIdx = argmaxData[index];
            INDEX_T inputIdx = static_cast<INDEX_T>(n) * cStride + static_cast<INDEX_T>(c) * inPlaneSize +
                               static_cast<INDEX_T>(maxIdx);

            VALUE_T gradVal = gradData[index];
            if (isOverLap) {
                asc_atomic_add(&xGradData[inputIdx], gradVal);
            } else {
                xGradData[inputIdx] += gradVal;
            }
        }
    }
}

template <typename VALUE_T, typename OFFSET_T>
__aicore__ inline void AdaptiveMaxPool3dGradSimt<VALUE_T, OFFSET_T>::Init(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax,
                                                                          GM_ADDR y, GM_ADDR workspace)
{
    xGrad_.SetGlobalBuffer((__gm__ VALUE_T*)(y));
    grad_.SetGlobalBuffer((__gm__ VALUE_T*)(grad));
    argmax_.SetGlobalBuffer((__gm__ OFFSET_T*)(argmax));
    pipe_->InitBuffer(simtTilingDataBuf_, TILING_DATA_NUM * sizeof(int64_t));
}

template <typename VALUE_T, typename OFFSET_T>
__aicore__ inline void AdaptiveMaxPool3dGradSimt<VALUE_T, OFFSET_T>::Process()
{
    using DIV_T = SimtDivT<OFFSET_T>;

    LocalTensor<int64_t> simtTilingData = simtTilingDataBuf_.Get<int64_t>();
    const int64_t* tilingPtr = reinterpret_cast<const int64_t*>(tilingData_);
    for (uint32_t i = 0; i < TILING_DATA_NUM; ++i) {
        simtTilingData.SetValue(i, tilingPtr[i]);
    }

    const DIV_T nDims = static_cast<DIV_T>(simtTilingData(0));
    const DIV_T cDims = static_cast<DIV_T>(simtTilingData(1));
    const DIV_T inD = static_cast<DIV_T>(simtTilingData(2));
    const DIV_T inH = static_cast<DIV_T>(simtTilingData(3));
    const DIV_T inW = static_cast<DIV_T>(simtTilingData(4));
    const DIV_T outD = static_cast<DIV_T>(simtTilingData(5));
    const DIV_T outH = static_cast<DIV_T>(simtTilingData(6));
    const DIV_T outW = static_cast<DIV_T>(simtTilingData(7));
    const int64_t isOverLap = simtTilingData(8);
    const int64_t deterministicFlag = simtTilingData(9);

    DataSyncBarrier<MemDsbT::UB>();

    int64_t outputSize = static_cast<int64_t>(nDims) * static_cast<int64_t>(cDims) * static_cast<int64_t>(outD) *
                         static_cast<int64_t>(outH) * static_cast<int64_t>(outW);

    auto xGradData = (__gm__ VALUE_T*)xGrad_.GetPhyAddr();
    auto gradData = (__gm__ VALUE_T*)grad_.GetPhyAddr();
    auto argmaxData = (__gm__ OFFSET_T*)argmax_.GetPhyAddr();

    Simt::VF_CALL<InitOutputZeroSimt<VALUE_T>>(Simt::Dim3(THREAD_DIM), outputSize, xGradData);
    AscendC::SyncAll();

    if (deterministicFlag) {
        Simt::VF_CALL<AdaptiveMaxPool3dGradKernelSimt<VALUE_T, OFFSET_T, true>>(Simt::Dim3(THREAD_DIM), gradData,
                                                                                argmaxData, nDims, cDims, inD, inH, inW,
                                                                                outD, outH, outW, isOverLap, xGradData);
    } else {
        Simt::VF_CALL<AdaptiveMaxPool3dGradKernelSimt<VALUE_T, OFFSET_T, false>>(
            Simt::Dim3(THREAD_DIM), gradData, argmaxData, nDims, cDims, inD, inH, inW, outD, outH, outW, isOverLap,
            xGradData);
    }
}

} // namespace AdaptiveMaxPool3dGradOp

#endif // ADAPTIVE_MAX_POOL3D_GRAD_SIMT_H
