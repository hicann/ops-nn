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
 * \file foreach_flat_flow_regbase.h
 * \brief Flat RegBase transport for foreach operators with 1-3 input TensorList
 *        flows and an optional shared or per-tensor scalar. Mirrors the unary
 *        flat runner style: the mover owns all TQue resources, the compute
 *        policy only receives LocalTensor handles plus the current scalar.
 */

#ifndef FOREACH_FLAT_FLOW_REGBASE_H
#define FOREACH_FLAT_FLOW_REGBASE_H

#include "foreach_regbase_common.h"

namespace ForeachFlatRegbase {
using namespace AscendC;

/**
 * Dtype-independent tensor-list dispatcher for multi-flow operators. All input
 * flows share the same flat partition, so per-tensor cursors are computed once.
 */
template <typename Tiling, int InputFlows>
class TensorListFlowDispatcher {
public:
    __aicore__ inline void Init(const Tiling* tilingData)
    {
        const uint32_t blockIdx = GetBlockIdx();
        tensorDataCountList_ = const_cast<int64_t*>(tilingData->tensorDataCountList);
        tensorStart_ = tilingData->tensorStartList[blockIdx];
        tensorEnd_ = tilingData->tensorEndList[blockIdx];
        tensorStartOffset_ = tilingData->tensorStartOffsetList[blockIdx];
        tensorEndOffset_ = tilingData->tensorEndOffsetList[blockIdx];
    }

    template <typename TypedMover>
    __aicore__ inline void Process(GM_ADDR* inputs, GM_ADDR outputs, TypedMover& mover)
    {
        ListTensorDesc outputDesc(reinterpret_cast<__gm__ void*>(outputs));
        for (uint16_t tensorIdx = tensorStart_; tensorIdx <= tensorEnd_; ++tensorIdx) {
            int64_t cursorStart = tensorIdx == tensorStart_ ? tensorStartOffset_ : 0;
            int64_t cursorEnd = tensorIdx == tensorEnd_ ? tensorEndOffset_ : tensorDataCountList_[tensorIdx] - 1;
            int64_t dataCount = cursorEnd - cursorStart + 1;
            // Empty tensors are valid. Do not resolve their data pointers or issue a zero/negative DMA.
            if (dataCount <= 0) {
                continue;
            }
            mover.ProcessTensor(inputs, outputDesc, tensorIdx, cursorStart, dataCount);
        }
    }

private:
    int64_t* tensorDataCountList_ = nullptr;
    uint16_t tensorStart_ = 0;
    uint16_t tensorEnd_ = 0;
    int64_t tensorStartOffset_ = 0;
    int64_t tensorEndOffset_ = -1;
};

/**
 * Typed storage mover for InputFlows input lists plus one output list. Owns one
 * double-buffered queue per flow. ComputePolicy::Run receives the input
 * LocalTensor array, the output handle, the current scalar and the tile count.
 */
template <typename T, typename ScalarT, int InputFlows, typename ComputePolicy>
class TypedFlowMover {
public:
    static_assert(!ComputePolicy::kUsesExtraUb, "flat RegBase compute policies must not allocate extra UB");

    __aicore__ inline void Init(uint32_t tileElements, TPipe* pipe)
    {
        tileElements_ = tileElements;
        for (int flow = 0; flow < InputFlows; ++flow) {
            pipe->InitBuffer(inputQueues_[flow], BUFFER_NUM, tileElements_ * sizeof(T));
        }
        pipe->InitBuffer(outputQueue_, BUFFER_NUM, tileElements_ * sizeof(T));
    }

    __aicore__ inline void SetScalar(ScalarT value) { scalar_ = value; }

    __aicore__ inline void SetScalars(__gm__ ScalarT* scalars) { scalars_ = scalars; }

    __aicore__ inline void ProcessTensor(GM_ADDR* inputs, ListTensorDesc& outputDesc, uint16_t tensorIdx,
                                         int64_t cursorStart, int64_t dataCount)
    {
        tensorScalar_ = scalars_ != nullptr ? scalars_[tensorIdx] : scalar_;
        for (int flow = 0; flow < InputFlows; ++flow) {
            ListTensorDesc inputDesc(reinterpret_cast<__gm__ void*>(inputs[flow]));
            inputGlobals_[flow].SetGlobalBuffer(inputDesc.GetDataPtr<__gm__ T>(tensorIdx) + cursorStart);
        }
        outputGlobal_.SetGlobalBuffer(outputDesc.GetDataPtr<__gm__ T>(tensorIdx) + cursorStart);
        int64_t tileOffset = 0;
        while (tileOffset < dataCount) {
            uint32_t currentCount = static_cast<uint32_t>(
                dataCount - tileOffset < tileElements_ ? dataCount - tileOffset : tileElements_);
            CopyIn(tileOffset, currentCount);
            Compute(currentCount);
            CopyOut(tileOffset, currentCount);
            tileOffset += currentCount;
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t tileOffset, uint32_t dataCount)
    {
        DataCopyPadExtParams<T> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0;
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = dataCount * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        for (int flow = 0; flow < InputFlows; ++flow) {
            LocalTensor<T> inputLocal = inputQueues_[flow].template AllocTensor<T>();
            DataCopyPad(inputLocal, inputGlobals_[flow][tileOffset], copyParams, padParams);
            inputQueues_[flow].EnQue(inputLocal);
        }
    }

    __aicore__ inline void Compute(uint32_t dataCount)
    {
        LocalTensor<T> inputLocals[InputFlows];
        for (int flow = 0; flow < InputFlows; ++flow) {
            inputLocals[flow] = inputQueues_[flow].template DeQue<T>();
        }
        LocalTensor<T> outputLocal = outputQueue_.template AllocTensor<T>();
        ComputePolicy::Run(inputLocals, outputLocal, tensorScalar_, dataCount);
        for (int flow = 0; flow < InputFlows; ++flow) {
            inputQueues_[flow].FreeTensor(inputLocals[flow]);
        }
        outputQueue_.EnQue(outputLocal);
    }

    __aicore__ inline void CopyOut(int64_t tileOffset, uint32_t dataCount)
    {
        LocalTensor<T> outputLocal = outputQueue_.template DeQue<T>();
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = dataCount * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(outputGlobal_[tileOffset], outputLocal, copyParams);
        outputQueue_.FreeTensor(outputLocal);
    }

    static constexpr int32_t BUFFER_NUM = 2;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueues_[InputFlows];
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue_;
    GlobalTensor<T> inputGlobals_[InputFlows];
    GlobalTensor<T> outputGlobal_;
    uint32_t tileElements_ = 0;
    ScalarT scalar_ = 0;
    __gm__ ScalarT* scalars_ = nullptr;
    ScalarT tensorScalar_ = 0;
};

/**
 * Flat flow kernel facade. Init mirrors the unary flat kernel; the scalar hooks
 * are only called by operators that take a shared scalar or per-tensor scalars.
 */
template <typename T, typename ScalarT, typename Tiling, int InputFlows, typename ComputePolicy>
class FlatFlowKernel {
public:
    __aicore__ inline void Init(const Tiling* tilingData, TPipe* pipe)
    {
        dispatcher_.Init(tilingData);
        mover_.Init(tilingData->tileElems, pipe);
    }

    __aicore__ inline void InitScalar(GM_ADDR scalar)
    {
        GlobalTensor<ScalarT> scalarGm;
        scalarGm.SetGlobalBuffer(reinterpret_cast<__gm__ ScalarT*>(scalar), 1);
        mover_.SetScalar(scalarGm.GetValue(0));
    }

    __aicore__ inline void InitScalars(GM_ADDR scalars)
    {
        mover_.SetScalars(reinterpret_cast<__gm__ ScalarT*>(scalars));
    }

    __aicore__ inline void Process(GM_ADDR x1, GM_ADDR y)
    {
        static_assert(InputFlows == 1, "single-flow Process on a multi-flow kernel");
        GM_ADDR inputs[] = {x1};
        dispatcher_.Process(inputs, y, mover_);
    }

    __aicore__ inline void Process(GM_ADDR x1, GM_ADDR x2, GM_ADDR y)
    {
        static_assert(InputFlows == 2, "two-flow Process on a non-binary kernel");
        GM_ADDR inputs[] = {x1, x2};
        dispatcher_.Process(inputs, y, mover_);
    }

    __aicore__ inline void Process(GM_ADDR x1, GM_ADDR x2, GM_ADDR x3, GM_ADDR y)
    {
        static_assert(InputFlows == 3, "three-flow Process on a non-ternary kernel");
        GM_ADDR inputs[] = {x1, x2, x3};
        dispatcher_.Process(inputs, y, mover_);
    }

private:
    TensorListFlowDispatcher<Tiling, InputFlows> dispatcher_;
    TypedFlowMover<T, ScalarT, InputFlows, ComputePolicy> mover_;
};
} // namespace ForeachFlatRegbase

#endif // FOREACH_FLAT_FLOW_REGBASE_H
