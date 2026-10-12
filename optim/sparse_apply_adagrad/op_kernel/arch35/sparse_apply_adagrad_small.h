/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sparse_apply_adagrad_small.h
 * \brief
 */

#ifndef SPARSE_APPLY_ADAGRAD_SMALL
#define SPARSE_APPLY_ADAGRAD_SMALL

#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "kernel_operator.h"
#include "../inc/kernel_utils.h"

namespace SparseApplyAdagradOps {
using namespace AscendC;
using namespace AscendC::MicroAPI;

class SparseApplyAdagradSmall {
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr int64_t DATA_BLOCK_COUNT = 8;
    static constexpr uint16_t VECTOR_LENGTH = platform::GetVRegSize();
    static constexpr uint32_t VL_FP32 = VECTOR_LENGTH / sizeof(float);
    static constexpr int64_t BLOCK_SIZE = platform::GetUbBlockSize();
    static constexpr float EPSILON = 1e-8;

public:
    __aicore__ inline SparseApplyAdagradSmall(){};

    __aicore__ inline SparseApplyAdagradSmall(TPipe* pipeIn) { pipe_ = pipeIn; }

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR accum, GM_ADDR lr, GM_ADDR grad, GM_ADDR indices,
                                const SparseApplyAdagradSmallDimTilingData* tilingDataIn)
    {
        tilingData_ = tilingDataIn;
        ParseTilingData();
        int64_t gradDim0Align = (gradDim0 + DATA_BLOCK_COUNT - 1) / DATA_BLOCK_COUNT * DATA_BLOCK_COUNT; // 32B对齐
        varGm_.SetGlobalBuffer((__gm__ float*)var);
        accumGm_.SetGlobalBuffer((__gm__ float*)accum);
        lrGm_.SetGlobalBuffer((__gm__ float*)lr);
        gradGm_.SetGlobalBuffer((__gm__ float*)grad);
        indicesGm_.SetGlobalBuffer((__gm__ int32_t*)indices);

        pipe_->InitBuffer(varBuf_, blockFormer);
        pipe_->InitBuffer(accumBuffer_, blockFormer);
        pipe_->InitBuffer(lrBuf_, BLOCK_SIZE);
        pipe_->InitBuffer(gradQueue_, BUFFER_NUM, blockFormer);
        pipe_->InitBuffer(indicesBuf_, gradDim0Align * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        blockIdx = GetBlockIdx();
        if (blockIdx < blockNum - 1) {
            blockLength = blockFormer / sizeof(float);
        } else {
            blockLength = blockTail / sizeof(float);
        }
        CopyInLrAndIndices();
        for (int32_t gradLoopNum = 0; gradLoopNum < gradDim0; gradLoopNum++) {
            CopyInGrad(gradLoopNum);
            int32_t indiceNum = indicesLocalTensor.GetValue(gradLoopNum);
            if (0 > indiceNum || indiceNum >= dim0) {
                continue;
            }
            CopyInVarAndAccum(indiceNum);
            CalcOutput(gradLoopNum);
            CopyOutVarAndAccum(indiceNum);
        }
    }

private:
    __aicore__ inline void ParseTilingData()
    {
        totalCoreNum = tilingData_->totalCoreNum;
        dim0 = tilingData_->a1_;
        dim1 = tilingData_->r_;
        gradDim0 = tilingData_->gradDim0;
        blockFormer = tilingData_->blockFormer;
        blockNum = tilingData_->blockNum;
        blockTail = tilingData_->blockTail;
        useLocking = tilingData_->useLocking;
        updateSlots = tilingData_->updateSlots;
    }

    __aicore__ inline void CopyInLrAndIndices()
    {
        indicesLocalTensor = indicesBuf_.Get<int32_t>();
        DataCopyExtParams copyIndicesParamsBatch{1, static_cast<uint32_t>(gradDim0 * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> indicesPadExtParams{false, 0, 0, 0};
        DataCopyPad(indicesLocalTensor, indicesGm_, copyIndicesParamsBatch, indicesPadExtParams);

        lrLocalTensor = lrBuf_.Get<float>();
        DataCopyParams copyParamsBatch{1, static_cast<uint16_t>(sizeof(float)), 0, 0};
        DataCopyPad(lrLocalTensor, lrGm_, copyParamsBatch, padParams);
        MTE2ToSSync();
        lr = lrLocalTensor.GetValue(0);
    }

    __aicore__ inline void CopyInGrad(int32_t gradLoopNum)
    {
        LocalTensor<float> gradLocalTensor = gradQueue_.AllocTensor<float>();
        DataCopyExtParams copyGradParamsBatch{static_cast<uint16_t>(1),
                                              static_cast<uint32_t>(blockLength * sizeof(float)), 0, 0, 0};
        DataCopyPad(gradLocalTensor, gradGm_[gradLoopNum * dim1 + blockIdx * (blockFormer / sizeof(float))],
                    copyGradParamsBatch, PadExtParams);
        gradQueue_.EnQue(gradLocalTensor);
    }

    __aicore__ inline void CopyInVarAndAccum(int32_t indiceNum)
    {
        varLocalTensor = varBuf_.Get<float>();
        MTE3ToMTE2Sync();
        uint64_t varOffset = indiceNum * dim1 + blockIdx * (blockFormer / sizeof(float));
        DataCopyExtParams copyParamsBatch{1, static_cast<uint32_t>(blockLength * sizeof(float)), 0, 0, 0};
        DataCopyPad(varLocalTensor, varGm_[varOffset], copyParamsBatch, PadExtParams);
        accumLocalTensor = accumBuffer_.Get<float>();
        DataCopyPad(accumLocalTensor, accumGm_[varOffset], copyParamsBatch, PadExtParams);
        MTE2ToVSync();
    }

    __aicore__ inline void CalcOutput(int64_t indicesLoopNum)
    {
        LocalTensor<float> gradLocalRes = gradQueue_.DeQue<float>();
        uint32_t dtypeSize = sizeof(float);
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = static_cast<uint16_t>(ops::CeilDiv(static_cast<uint32_t>(blockLength), vl));

        __local_mem__ float* varLocal = (__local_mem__ float*)varLocalTensor.GetPhyAddr();
        __local_mem__ float* accumLocal = (__local_mem__ float*)accumLocalTensor.GetPhyAddr();
        __local_mem__ float* gradLocal = (__local_mem__ float*)gradLocalRes.GetPhyAddr();

        if (updateSlots) {
            __VEC_SCOPE__
            {
                AscendC::MicroAPI::RegTensor<float> varReg;
                AscendC::MicroAPI::RegTensor<float> accumReg;
                AscendC::MicroAPI::RegTensor<float> gradReg;
                AscendC::MicroAPI::RegTensor<float> gradSquareReg;
                AscendC::MicroAPI::RegTensor<float> accumAddReg;
                AscendC::MicroAPI::RegTensor<float> accumSqrtReg;
                AscendC::MicroAPI::RegTensor<float> accumSqrtEpsilonReg;
                AscendC::MicroAPI::RegTensor<float> gradDivReg;
                AscendC::MicroAPI::RegTensor<float> lrReg;
                AscendC::MicroAPI::RegTensor<float> epsilonReg;
                AscendC::MicroAPI::RegTensor<float> lrMulReg;
                AscendC::MicroAPI::RegTensor<float> varSubReg;

                AscendC::MicroAPI::MaskReg preg0;
                static constexpr AscendC::MicroAPI::DivSpecificMode mode = {AscendC::MicroAPI::MaskMergeMode::ZEROING,
                                                                            true};
                AscendC::MicroAPI::Duplicate(lrReg, lr);
                AscendC::MicroAPI::Duplicate(epsilonReg, EPSILON);

                uint32_t sreg0 = blockLength;
                for (uint16_t i = 0; i < loopNum; i++) {
                    preg0 = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                    AscendC::MicroAPI::DataCopy(gradReg, gradLocal + i * vl);
                    AscendC::MicroAPI::DataCopy(varReg, varLocal + i * vl);
                    AscendC::MicroAPI::DataCopy(accumReg, accumLocal + i * vl);
                    AscendC::MicroAPI::Mul(gradSquareReg, gradReg, gradReg, preg0);
                    AscendC::MicroAPI::Add(accumAddReg, gradSquareReg, accumReg, preg0);
                    AscendC::MicroAPI::DataCopy(accumLocal + i * vl, accumAddReg, preg0);
                    AscendC::MicroAPI::Sqrt(accumSqrtReg, accumAddReg, preg0);
                    AscendC::MicroAPI::Add(accumSqrtEpsilonReg, accumSqrtReg, epsilonReg, preg0);
                    AscendC::MicroAPI::Mul(lrMulReg, gradReg, lrReg, preg0);
                    AscendC::MicroAPI::Div(gradDivReg, lrMulReg, accumSqrtEpsilonReg, preg0);
                    AscendC::MicroAPI::Sub(varSubReg, varReg, gradDivReg, preg0);
                    AscendC::MicroAPI::DataCopy(varLocal + i * vl, varSubReg, preg0);
                }
            }
        } else {
            __VEC_SCOPE__
            {
                AscendC::MicroAPI::RegTensor<float> varReg;
                AscendC::MicroAPI::RegTensor<float> accumReg;
                AscendC::MicroAPI::RegTensor<float> gradReg;
                AscendC::MicroAPI::RegTensor<float> accumSqrtReg;
                AscendC::MicroAPI::RegTensor<float> accumSqrtEpsilonReg;
                AscendC::MicroAPI::RegTensor<float> gradDivReg;
                AscendC::MicroAPI::RegTensor<float> lrReg;
                AscendC::MicroAPI::RegTensor<float> epsilonReg;
                AscendC::MicroAPI::RegTensor<float> lrMulReg;
                AscendC::MicroAPI::RegTensor<float> varSubReg;

                AscendC::MicroAPI::MaskReg preg0;
                static constexpr AscendC::MicroAPI::DivSpecificMode mode = {AscendC::MicroAPI::MaskMergeMode::ZEROING,
                                                                            true};
                AscendC::MicroAPI::Duplicate(lrReg, lr);
                AscendC::MicroAPI::Duplicate(epsilonReg, EPSILON);

                uint32_t sreg0 = blockLength;
                for (uint16_t i = 0; i < loopNum; i++) {
                    preg0 = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                    AscendC::MicroAPI::DataCopy(gradReg, gradLocal + i * vl);
                    AscendC::MicroAPI::DataCopy(varReg, varLocal + i * vl);
                    AscendC::MicroAPI::DataCopy(accumReg, accumLocal + i * vl);
                    AscendC::MicroAPI::Sqrt(accumSqrtReg, accumReg, preg0);
                    AscendC::MicroAPI::Add(accumSqrtEpsilonReg, accumSqrtReg, epsilonReg, preg0);
                    AscendC::MicroAPI::Mul(lrMulReg, gradReg, lrReg, preg0);
                    AscendC::MicroAPI::Div(gradDivReg, lrMulReg, accumSqrtEpsilonReg, preg0);
                    AscendC::MicroAPI::Sub(varSubReg, varReg, gradDivReg, preg0);
                    AscendC::MicroAPI::DataCopy(varLocal + i * vl, varSubReg, preg0);
                }
            }
        }
        gradQueue_.FreeTensor(gradLocalRes);
    }

    __aicore__ inline void CopyOutVarAndAccum(int32_t indiceNum)
    {
        DataCopyExtParams copyParamsBatch{1, static_cast<uint32_t>(blockLength * sizeof(float)), 0, 0, 0};
        VToMTE3Sync();
        uint64_t varOutOffset = indiceNum * dim1 + blockIdx * (blockFormer / sizeof(float));
        DataCopyPad(varGm_[varOutOffset], varLocalTensor, copyParamsBatch);
        DataCopyPad(accumGm_[varOutOffset], accumLocalTensor, copyParamsBatch);
    }

    __aicore__ inline void MTE2ToSSync()
    {
        event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
    }

    __aicore__ inline void MTE3ToSSync()
    {
        event_t eventIDMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
        WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
    }

    __aicore__ inline void VToMTE3Sync()
    {
        event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    }

    __aicore__ inline void MTE2ToVSync()
    {
        event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    }

    __aicore__ inline void MTE3ToMTE2Sync()
    {
        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    }

private:
    const SparseApplyAdagradSmallDimTilingData* tilingData_;

    TPipe* pipe_;
    TBuf<TPosition::VECCALC> varBuf_;
    TBuf<TPosition::VECCALC> accumBuffer_;
    TBuf<TPosition::VECCALC> lrBuf_;
    TQue<QuePosition::VECIN, 1> gradQueue_;
    TBuf<TPosition::VECCALC> indicesBuf_;

    GlobalTensor<float> varGm_;
    GlobalTensor<float> accumGm_;
    GlobalTensor<float> lrGm_;
    GlobalTensor<float> gradGm_;
    GlobalTensor<int32_t> indicesGm_;

    LocalTensor<float> varLocalTensor;
    LocalTensor<float> accumLocalTensor;
    LocalTensor<float> lrLocalTensor;
    LocalTensor<int32_t> indicesLocalTensor;

    int64_t totalCoreNum;
    int64_t blockIdx;
    int64_t blockLength;
    int64_t dim0;
    int64_t dim1;
    int64_t gradDim0;
    int64_t blockFormer;
    int64_t blockNum;
    int64_t blockTail;
    float lr;
    bool useLocking;
    bool updateSlots;

    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyPadExtParams<float> PadExtParams{false, 0, 0, 0};
};

} // namespace SparseApplyAdagradOps
#endif
