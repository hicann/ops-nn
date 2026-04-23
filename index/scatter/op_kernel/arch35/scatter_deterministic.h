/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_deterministic.h
 * \brief
 */

#ifndef ASCENDC_SCATTER_DETERMINISTIC_H_
#define ASCENDC_SCATTER_DETERMINISTIC_H_

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace SCATTER {
using namespace AscendC;

constexpr int32_t INDICES_DIM2 = 2;

template <typename VAR_T, typename INDICES_T>
class ScatterDeterministic
{
public:
    __aicore__ inline ScatterDeterministic(const ScatterTilingData* tiling, TPipe* pipe) : tilingData_(tiling), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR y);
    __aicore__ inline void Process();
    __aicore__ inline void CopyInIndices(int64_t offset, uint32_t indicesLen);
    __aicore__ inline void CopyInUpdates(int64_t offset, uint32_t updatesLen);
    __aicore__ inline void CopyOutUpdates(int64_t offset, uint32_t updatesLen);
    __aicore__ inline void ProcessUpdates(int64_t outOffsetBase, int64_t updatesOffsetBase, INDICES_T indicesDim1Value);

private:
    GlobalTensor<INDICES_T> indicesGm_;
    GlobalTensor<VAR_T> updatesGm_;
    GlobalTensor<VAR_T> yGm_;
    TQue<QuePosition::VECIN, 1> indicesQue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> updatesQueue_;

    TPipe* pipe_ = nullptr;
    const ScatterTilingData* tilingData_;

    int64_t blockIdx_{0};
    int64_t updatesTailLoopSize_{0};
    int64_t updatesBlockColLoop_{0};
};

template <typename VAR_T, typename INDICES_T>
__aicore__ inline void ScatterDeterministic<VAR_T, INDICES_T>::Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR y)
{
    indicesGm_.SetGlobalBuffer((__gm__ INDICES_T*)(indices));
    updatesGm_.SetGlobalBuffer((__gm__ VAR_T*)(updates));
    yGm_.SetGlobalBuffer((__gm__ VAR_T*)(y));

    pipe_->InitBuffer(indicesQue_, 1, tilingData_->indicesUbFactor * sizeof(INDICES_T));
    pipe_->InitBuffer(updatesQueue_, 1, tilingData_->updatesColUbFactor * sizeof(VAR_T));

    blockIdx_ = GetBlockIdx();
    updatesTailLoopSize_ = tilingData_->updatesNormBlockTailLoopSize;
    updatesBlockColLoop_ = tilingData_->updatesNormBlockColLoop;
    if (blockIdx_ == tilingData_->aivCoreNum - 1) {
        updatesTailLoopSize_ = tilingData_->updatesTailBlockTailLoopSize;
        updatesBlockColLoop_ = tilingData_->updatesTailBlockColLoop;
    }
}

template <typename VAR_T, typename INDICES_T>
__aicore__ inline void ScatterDeterministic<VAR_T, INDICES_T>::CopyInIndices(int64_t offset, uint32_t indicesLen)
{
    LocalTensor<INDICES_T> indicesLocal = indicesQue_.AllocTensor<INDICES_T>();
    DataCopyPadExtParams<INDICES_T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = 1;
    dataCoptExtParams.blockLen = indicesLen * sizeof(INDICES_T);
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(indicesLocal, indicesGm_[offset], dataCoptExtParams, dataCopyPadExtParams);
    indicesQue_.EnQue<INDICES_T>(indicesLocal);
}

template <typename VAR_T, typename INDICES_T>
__aicore__ inline void ScatterDeterministic<VAR_T, INDICES_T>::CopyInUpdates(int64_t offset, uint32_t updatesLen)
{
    LocalTensor<VAR_T> updatesLocal = updatesQueue_.AllocTensor<VAR_T>();
    DataCopyPadExtParams<VAR_T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = 1;
    dataCoptExtParams.blockLen = updatesLen * sizeof(VAR_T);
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(updatesLocal, updatesGm_[offset], dataCoptExtParams, dataCopyPadExtParams);
    updatesQueue_.EnQue<VAR_T>(updatesLocal);
}

template <typename VAR_T, typename INDICES_T>
__aicore__ inline void ScatterDeterministic<VAR_T, INDICES_T>::CopyOutUpdates(int64_t offset, uint32_t updatesLen)
{
    event_t eventIdMTE2toMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(eventIdMTE2toMTE3);
    WaitFlag<HardEvent::MTE2_MTE3>(eventIdMTE2toMTE3);

    LocalTensor<VAR_T> updatesLocal = updatesQueue_.DeQue<VAR_T>();
    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = 1;
    dataCoptExtParams.blockLen = updatesLen * sizeof(VAR_T);
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(yGm_[offset], updatesLocal, dataCoptExtParams);
    updatesQueue_.FreeTensor(updatesLocal);
}

template <typename VAR_T, typename INDICES_T>
__aicore__ inline void ScatterDeterministic<VAR_T, INDICES_T>::ProcessUpdates(
               int64_t outOffsetBase, int64_t updatesOffsetBase, INDICES_T indicesDim1Value)
{
    int64_t updatesLast2DimSize = tilingData_->updatesDim2 * tilingData_->updatesDim3;
    int64_t varLast2DimSize = tilingData_->inputDim2 * tilingData_->inputDim3;
    for (int64_t dim1Idx = 0; dim1Idx < tilingData_->updatesDim1; dim1Idx++) {
        for (int64_t dim2Idx = 0; dim2Idx < tilingData_->updatesDim2; dim2Idx++) {
            for (int64_t j = 0; j < updatesBlockColLoop_; j++) {
                uint32_t updatesCount = j == updatesBlockColLoop_ - 1 ? updatesTailLoopSize_ :
                                                                        tilingData_->updatesColUbFactor;
                int64_t updatesOffset = updatesOffsetBase + dim1Idx * updatesLast2DimSize + dim2Idx * tilingData_->updatesDim3 +
                                        blockIdx_ * tilingData_->normBlockColNum + j * tilingData_->updatesColUbFactor;   
                int64_t outOffset = outOffsetBase + dim1Idx * varLast2DimSize + (indicesDim1Value + dim2Idx) * tilingData_->updatesDim3 +
                                    blockIdx_ * tilingData_->normBlockColNum + j * tilingData_->updatesColUbFactor;   

                CopyInUpdates(updatesOffset, updatesCount);
                CopyOutUpdates(outOffset, updatesCount);
            }
        }
    }
}

template <typename VAR_T, typename INDICES_T>
__aicore__ inline void ScatterDeterministic<VAR_T, INDICES_T>::Process()
{
    if (blockIdx_ >= tilingData_->aivCoreNum) {
        return;
    }

    int64_t updatesLast3DimSize = tilingData_->updatesDim1 * tilingData_->updatesDim2 * tilingData_->updatesDim3;
    int64_t varLast3DimSize = tilingData_->inputDim1 * tilingData_->inputDim2 * tilingData_->inputDim3;
    for (int64_t i = 0; i < tilingData_->indicesLoop; i++) {
        uint32_t indicesCount = i == tilingData_->indicesLoop - 1 ? tilingData_->indicesTailLoopNum :
                                                                   tilingData_->indicesUbFactor;
        int64_t indicesOffset = i * tilingData_->indicesUbFactor;
        CopyInIndices(indicesOffset, indicesCount);

        event_t eventIdMTE2toS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIdMTE2toS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMTE2toS);
        LocalTensor<INDICES_T> indicesLocal = indicesQue_.DeQue<INDICES_T>();

        for (int32_t j = 0; j < indicesCount; j+=INDICES_DIM2) {
            INDICES_T indicesDim0Value = indicesLocal.GetValue(j);
            INDICES_T indicesDim1Value = indicesLocal.GetValue(j + 1);
            int64_t outOffsetBase = indicesDim0Value * varLast3DimSize;
            int64_t updatesOffsetBase = (indicesOffset + j) / INDICES_DIM2 * updatesLast3DimSize;
            ProcessUpdates(outOffsetBase, updatesOffsetBase, indicesDim1Value);
        }

        indicesQue_.FreeTensor(indicesLocal);
    }
}
} // namespace SCATTER

#endif // ASCENDC_SCATTER_DETERMINISTIC_H_
