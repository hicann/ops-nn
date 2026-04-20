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
 * \file scatter_nd_update_deterministic_simd.h
 * \brief
 */

#ifndef SCATTER_ND_UPDATE_DETER_SIMD_H
#define SCATTER_ND_UPDATE_DETER_SIMD_H

#include "kernel_operator.h"
#include "scatter_nd_update_common.h"
#include "op_kernel/math_util.h"
namespace ScatterNdUpdate {
using namespace AscendC;

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T = INDICES_T>
class ScatterNdUpdateDeterministicSimd: public ScatterNdUpdateDeterministicCommon<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>
{
public:
    __aicore__ inline ScatterNdUpdateDeterministicSimd(const ScatterNdUpdateRegBaseTilingData& tilingData, TPipe& pipe) : ScatterNdUpdateDeterministicCommon<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>(tilingData, pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInUpdate(LocalTensor<PARAMS_T>& updateLocal);
    __aicore__ inline void CopyOutUpdate(LocalTensor<PARAMS_T>& updateLocal, uint64_t varGmOffSet);

private:
    TYPE_T updateOffSet = 0;
    TYPE_T indiceOffSet = 0;
    TYPE_T idxLoopSize = 0;
    TYPE_T updateLoopSize = 0;
    int32_t updateCount = 0;
};

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateDeterministicSimd<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::Init(GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y ,GM_ADDR workspace)
{    
    this->InitBase(x, indices, updates, y, workspace);
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateDeterministicSimd<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::Process()
{
    // if input is empty, return directly
    if (this->tiling_.sliceSize == 0) {
        return;
    }
    SyncAll();
    this->CalcMask();
    SyncAll();

    if (this->blockIdx >= this->tiling_.usedCoreNumBefore) {
        return;
    }
    this->InitUpdateBuffer();

    if (this->blockIdx == this->tiling_.usedCoreNumBefore - 1) {
        this->currBlockHandleIdx = this->tiling_.tailCoreIndexCount; 
    } else {
        this->currBlockHandleIdx = this->tiling_.eachCoreIndexCount;
    }
    this->idxLoopSize = Ops::Base::CeilDiv(this->currBlockHandleIdx, static_cast<TYPE_T>(this->tiling_.indicesFactor));
    this->updateLoopSize = this->tiling_.updateLoopSize;
    this->indiceBlockOffSet = this->blockIdx * this->tiling_.eachCoreIndexCount;
    for (TYPE_T i = 0; i < this->idxLoopSize; i++) {
        // simd每次只处理一行
        this->indiceOffSet = this->indiceBlockOffSet + i;

        int64_t globalValRowIdx = this->varIdxGm(this->indiceOffSet); 
        // 越界校验
        if (globalValRowIdx < 0 || globalValRowIdx > this->tiling_.outputStorageShapeSize) {
            continue;
        }

        // 获取行对应varIdx
        if (this->maskGm(globalValRowIdx / this->tiling_.sliceSize) != this->indiceOffSet) {
            continue;
        }

        for (TYPE_T j = 0; j < this->updateLoopSize; j++) {
            LocalTensor<PARAMS_T> updateLocal = this->inQueX.template AllocTensor<PARAMS_T>();
            this->updateOffSet = this->indiceOffSet * this->tiling_.sliceSize + j * this->tiling_.afterAxisFactor;
            uint64_t varGmOffSet = globalValRowIdx + j * this->tiling_.afterAxisFactor;
            if (j == this->updateLoopSize-1) {
                this->updateCount =  this->tiling_.updateTailNum;
            } else {
                this->updateCount = this->tiling_.afterAxisFactor;
            }
            CopyInUpdate(updateLocal);
            event_t eventIdMte2ToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
            SetFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
            WaitFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
            CopyOutUpdate(updateLocal, varGmOffSet);
            event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
    }
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateDeterministicSimd<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::CopyInUpdate(LocalTensor<PARAMS_T>& updateLocal)
{
    DataCopyExtParams xCopyParams{1, static_cast<uint32_t>(this->updateCount * sizeof(PARAMS_T)), 0, 0, 0};
    DataCopyPadExtParams<PARAMS_T> xPadParams{false, 0, 0, 0};
    DataCopyPad(updateLocal, this->updateGm[this->updateOffSet], xCopyParams, xPadParams);
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename OFFSET_T>
__aicore__ inline void ScatterNdUpdateDeterministicSimd<PARAMS_T, INDICES_T, TYPE_T, OFFSET_T>::CopyOutUpdate(LocalTensor<PARAMS_T>& updateLocal, uint64_t varGmOffSet)
{
    DataCopyExtParams xCopyParams{1, static_cast<uint32_t>(this->updateCount * sizeof(PARAMS_T)), 0, 0, 0};
    DataCopyPad(this->outputGm[varGmOffSet], updateLocal[0], xCopyParams);
    this->inQueX.template FreeTensor(updateLocal); 
}
} // namespace ScatterNdUpdate

#endif // SCATTER_ND_UPDATE_DETER_SIMD_H