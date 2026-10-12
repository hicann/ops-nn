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
 * \file sparse_apply_adagrad_big.h
 * \brief
 */

#ifndef SPARSE_APPLY_ADAGRAD_BIG
#define SPARSE_APPLY_ADAGRAD_BIG

#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "kernel_operator.h"
#include "../inc/kernel_utils.h"

namespace SparseApplyAdagradOps {
using namespace AscendC;
using namespace AscendC::MicroAPI;

class SparseApplyAdagradBig {
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr int64_t DATA_BLOCK_COUNT = 8;
    static constexpr uint16_t VECTOR_LENGTH = platform::GetVRegSize();
    static constexpr uint32_t VL_FP32 = VECTOR_LENGTH / sizeof(float);
    static constexpr int64_t BLOCK_SIZE = platform::GetUbBlockSize();
    static constexpr float EPSILON = 1e-8;
    static constexpr int64_t MAX_INDICES_BLOCK_NUM = 1024;

public:
    __aicore__ inline SparseApplyAdagradBig(){};

    __aicore__ inline SparseApplyAdagradBig(TPipe* pipeIn) { pipe_ = pipeIn; }

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR accum, GM_ADDR lr, GM_ADDR grad, GM_ADDR indices,
                                const SparseApplyAdagradBigDimTilingData* tilingDataIn)
    {
        tilingData_ = tilingDataIn;
        ParseTilingData();
        gradDim0Align = indicesLoopNum == 1 ? (gradDim0 + DATA_BLOCK_COUNT - 1) / DATA_BLOCK_COUNT * DATA_BLOCK_COUNT :
                                              MAX_INDICES_BLOCK_NUM; // 32B对齐
        varGm_.SetGlobalBuffer((__gm__ float*)var);
        accumGm_.SetGlobalBuffer((__gm__ float*)accum);
        lrGm_.SetGlobalBuffer((__gm__ float*)lr);
        gradGm_.SetGlobalBuffer((__gm__ float*)grad);
        indicesGm_.SetGlobalBuffer((__gm__ int32_t*)indices);

        pipe_->InitBuffer(varInQueue_, BUFFER_NUM, ubFormer);
        pipe_->InitBuffer(accumInQueue_, BUFFER_NUM, ubFormer);
        pipe_->InitBuffer(lrBuf_, BLOCK_SIZE);
        pipe_->InitBuffer(gradInQueue_, BUFFER_NUM, ubFormer);
        pipe_->InitBuffer(indicesBuf_, gradDim0Align * sizeof(int32_t));

        pipe_->InitBuffer(varOutQueue_, 1, ubFormer);
        pipe_->InitBuffer(accumOutQueue_, 1, ubFormer);
    }

    __aicore__ inline void Process()
    {
        blockIdx = GetBlockIdx();
        if (blockIdx < blockNum - 1) {
            blockLength = blockFormer / sizeof(float);
            ubLoop = ubLoopOfFormerBlock;
            ubTailBlockLength = ubTailOfFormerBlock;
        } else {
            blockLength = blockTail / sizeof(float);
            ubLoop = ubLoopOfTailBlock;
            ubTailBlockLength = ubTailOfTailBlock;
        }
        CopyInLr();
        for (int64_t indiceLoop = 0; indiceLoop < indicesLoopNum; indiceLoop++) {
            int64_t indiceBlockNum = indiceLoop == (indicesLoopNum - 1) ? indicesTailBlock : MAX_INDICES_BLOCK_NUM;
            CopyInIndices(indiceLoop, indiceBlockNum);
            for (int32_t indice = 0; indice < indiceBlockNum; indice++) {
                int32_t indiceNum = indicesLocalTensor.GetValue(indice);
                if (0 > indiceNum || indiceNum >= dim0) {
                    continue;
                }
                for (int32_t ubLoopNum = 0; ubLoopNum < ubLoop; ubLoopNum++) {
                    int64_t curTileRLen = ubLoopNum == (ubLoop - 1) ? ubTailBlockLength : ubFormer;
                    CopyInGrad(ubLoopNum, curTileRLen, indice + indiceLoop * MAX_INDICES_BLOCK_NUM);
                    CopyInVarAndAccum(ubLoopNum, curTileRLen, indiceNum);
                    CalcOutput(curTileRLen / sizeof(float));
                    CopyOutVarAndAccum(ubLoopNum, curTileRLen, indiceNum);
                }
            }
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
        ubFormer = tilingData_->ubFormer;
        ubLoopOfFormerBlock = tilingData_->ubLoopOfFormerBlock;
        ubLoopOfTailBlock = tilingData_->ubLoopOfTailBlock;
        ubTailOfFormerBlock = tilingData_->ubTailOfFormerBlock;
        ubTailOfTailBlock = tilingData_->ubTailOfTailBlock;
        indicesLoopNum = tilingData_->indicesLoopNum;
        indicesTailBlock = tilingData_->indicesTailBlock;
        useLocking = tilingData_->useLocking;
        updateSlots = tilingData_->updateSlots;
    }

    __aicore__ inline void CopyInIndices(int64_t indiceLoop, int64_t indiceBlockNum)
    {
        indicesLocalTensor = indicesBuf_.Get<int32_t>();
        DataCopyParams copyIndicesParamsBatch{1, static_cast<uint16_t>(indiceBlockNum * sizeof(float)), 0, 0};
        DataCopyPad(indicesLocalTensor, indicesGm_[indiceLoop * MAX_INDICES_BLOCK_NUM], copyIndicesParamsBatch,
                    padParams);
        MTE2ToSSync();
    }

    __aicore__ inline void CopyInLr()
    {
        lrLocalTensor = lrBuf_.Get<float>();
        DataCopyParams copyParamsBatch{1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0};
        DataCopyPad(lrLocalTensor, lrGm_, copyParamsBatch, padParams);
        MTE2ToSSync();
        lr = lrLocalTensor.GetValue(0);
    }

    __aicore__ inline void CopyInGrad(int32_t ubLoopNum, int64_t ubBlockLength, int32_t indice)
    {
        LocalTensor<float> gradLocalTensor = gradInQueue_.AllocTensor<float>();
        DataCopyExtParams copyGradParamsBatch{1, static_cast<uint32_t>(ubBlockLength), 0, 0, 0};
        uint64_t gradOffset = indice * dim1 + blockIdx * (blockFormer / sizeof(float)) +
                              ubLoopNum * (ubFormer / sizeof(float));
        DataCopyPad(gradLocalTensor, gradGm_[gradOffset], copyGradParamsBatch, padExtParams);
        gradInQueue_.EnQue(gradLocalTensor);
    }

    __aicore__ inline void CopyInVarAndAccum(int32_t ubLoopNum, int64_t ubBlockLength, int32_t indiceNum)
    {
        DataCopyExtParams copyParamsBatch{1, static_cast<uint32_t>(ubBlockLength), 0, 0, 0};
        if (ubLoop <= BUFFER_NUM) {
            MTE3ToMTE2Sync();
        }
        LocalTensor<float> varLocalTensor = varInQueue_.AllocTensor<float>();
        uint64_t varOffset = indiceNum * dim1 + blockIdx * (blockFormer / sizeof(float)) +
                             ubLoopNum * (ubFormer / sizeof(float));
        DataCopyPad(varLocalTensor, varGm_[varOffset], copyParamsBatch, padExtParams);
        varInQueue_.EnQue(varLocalTensor);

        LocalTensor<float> accumLocalTensor = accumInQueue_.AllocTensor<float>();
        DataCopyPad(accumLocalTensor, accumGm_[varOffset], copyParamsBatch, padExtParams);
        accumInQueue_.EnQue(accumLocalTensor);
    }

    __aicore__ inline void CalcOutput(int64_t ubBlockLength)
    {
        uint32_t dtypeSize = sizeof(float);
        uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
        uint16_t loopNum = static_cast<uint16_t>(ops::CeilDiv(static_cast<uint32_t>(ubBlockLength), vl));

        LocalTensor<float> varLocalRes = varInQueue_.DeQue<float>();
        LocalTensor<float> accumLocalRes = accumInQueue_.DeQue<float>();
        LocalTensor<float> gradLocalRes = gradInQueue_.DeQue<float>();

        LocalTensor<float> varOutLocalTensor = varOutQueue_.AllocTensor<float>();
        LocalTensor<float> accumOutLocalTensor = accumOutQueue_.AllocTensor<float>();

        __local_mem__ float* varLocal = (__local_mem__ float*)varLocalRes.GetPhyAddr();
        __local_mem__ float* accumLocal = (__local_mem__ float*)accumLocalRes.GetPhyAddr();
        __local_mem__ float* gradLocal = (__local_mem__ float*)gradLocalRes.GetPhyAddr();

        __local_mem__ float* varOutLocal = (__local_mem__ float*)varOutLocalTensor.GetPhyAddr();
        __local_mem__ float* accumOutLocal = (__local_mem__ float*)accumOutLocalTensor.GetPhyAddr();

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

                uint32_t sreg0 = ubBlockLength;
                for (uint16_t i = 0; i < loopNum; i++) {
                    preg0 = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                    AscendC::MicroAPI::DataCopy(gradReg, gradLocal + i * vl);
                    AscendC::MicroAPI::DataCopy(varReg, varLocal + i * vl);
                    AscendC::MicroAPI::DataCopy(accumReg, accumLocal + i * vl);
                    AscendC::MicroAPI::Mul(gradSquareReg, gradReg, gradReg, preg0);
                    AscendC::MicroAPI::Add(accumAddReg, gradSquareReg, accumReg, preg0);
                    AscendC::MicroAPI::DataCopy(accumOutLocal + i * vl, accumAddReg, preg0);
                    AscendC::MicroAPI::Sqrt(accumSqrtReg, accumAddReg, preg0);
                    AscendC::MicroAPI::Add(accumSqrtEpsilonReg, accumSqrtReg, epsilonReg, preg0);
                    AscendC::MicroAPI::Mul(lrMulReg, gradReg, lrReg, preg0);
                    AscendC::MicroAPI::Div(gradDivReg, lrMulReg, accumSqrtEpsilonReg, preg0);
                    AscendC::MicroAPI::Sub(varSubReg, varReg, gradDivReg, preg0);
                    AscendC::MicroAPI::DataCopy(varOutLocal + i * vl, varSubReg, preg0);
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

                uint32_t sreg0 = ubBlockLength;
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
                    AscendC::MicroAPI::DataCopy(varOutLocal + i * vl, varSubReg, preg0);
                    AscendC::MicroAPI::DataCopy(accumOutLocal + i * vl, accumReg, preg0);
                }
            }
        }

        varOutQueue_.EnQue(varOutLocalTensor);
        accumOutQueue_.EnQue(accumOutLocalTensor);

        varInQueue_.FreeTensor(varLocalRes);
        accumInQueue_.FreeTensor(accumLocalRes);
        gradInQueue_.FreeTensor(gradLocalRes);
    }

    __aicore__ inline void CopyOutVarAndAccum(int32_t ubLoopNum, int64_t ubBlockLength, int32_t indiceNum)
    {
        LocalTensor<float> varOutLocalRes = varOutQueue_.DeQue<float>();
        LocalTensor<float> accumOutLocalRes = accumOutQueue_.DeQue<float>();
        uint64_t varOutOffset = indiceNum * dim1 + blockIdx * (blockFormer / sizeof(float)) +
                                ubLoopNum * (ubFormer / sizeof(float));
        DataCopyExtParams copyParamsBatch{1, static_cast<uint32_t>(ubBlockLength), 0, 0, 0};
        DataCopyPad(varGm_[varOutOffset], varOutLocalRes, copyParamsBatch);
        varOutQueue_.FreeTensor(varOutLocalRes);
        DataCopyPad(accumGm_[varOutOffset], accumOutLocalRes, copyParamsBatch);
        accumOutQueue_.FreeTensor(accumOutLocalRes);
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

    __aicore__ inline void MTE3ToMTE2Sync()
    {
        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    }

private:
    const SparseApplyAdagradBigDimTilingData* tilingData_;

    TPipe* pipe_;
    TQue<QuePosition::VECIN, 1> varInQueue_;
    TQue<QuePosition::VECIN, 1> accumInQueue_;
    TQue<QuePosition::VECIN, 1> gradInQueue_;
    TQue<QuePosition::VECOUT, 1> varOutQueue_;
    TQue<QuePosition::VECOUT, 1> accumOutQueue_;
    TBuf<TPosition::VECCALC> lrBuf_;
    TBuf<TPosition::VECCALC> indicesBuf_;

    GlobalTensor<float> varGm_;
    GlobalTensor<float> accumGm_;
    GlobalTensor<float> lrGm_;
    GlobalTensor<float> gradGm_;
    GlobalTensor<int32_t> indicesGm_;

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
    int64_t ubFormer;
    int64_t ubLoopOfFormerBlock;
    int64_t ubLoopOfTailBlock;
    int64_t ubTailOfFormerBlock;
    int64_t ubTailOfTailBlock;
    int64_t indicesLoopNum;
    int64_t indicesTailBlock;
    int64_t gradDim0Align;
    float lr;
    int32_t ubLoop;
    int32_t ubTailBlockLength;
    bool useLocking;
    bool updateSlots;

    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyPadExtParams<float> padExtParams{false, 0, 0, 0};
};

} // namespace SparseApplyAdagradOps
#endif
