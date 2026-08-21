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
 * \file index_fill_front_n.h
 * \brief front axis and Q >= 64 or N >=64. core will split N.
 */
#ifndef INDEX_FILL_FRONT_N_H
#define INDEX_FILL_FRONT_N_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "index_fill_base.h"
#include <type_traits>

namespace IndexFillNS {
using namespace AscendC;
constexpr uint32_t OFFSET_UB_LENGTH = 8192;

template <typename T, typename U>
class IndexFillFrontN : public IndexFillBase<T, U> {
public:
    __aicore__ inline void Process()
    {
        this->ExecIndicesTask();
        this->ReadValue();
        FrontAxisN();
    }

    __aicore__ inline void FrontAxisN()
    {
        uint64_t PCoreNum = this->coreNum / this->P;
        PCoreNum = PCoreNum == 0 ? 1 : PCoreNum;

        uint64_t frontCoreNums = 0;
        uint64_t tailCoreNums = 0;
        uint64_t frontCoreNNums = 0;
        uint64_t tailCoreNNums = 0;
        this->coreSplit(PCoreNum, this->N, frontCoreNums, tailCoreNums, frontCoreNNums, tailCoreNNums);
        InitGather(frontCoreNNums);

        this->maskCached = false;
        this->singleNBlock = false;

        if (PCoreNum == 1 && this->ubNBlockLength >= frontCoreNNums) {
            FrontAxisNFastPath(frontCoreNNums);
        } else {
            for (uint64_t p = 0; p < this->P; p++) {
                ProcessOneP(frontCoreNums, tailCoreNums, frontCoreNNums, tailCoreNNums, p);
            }
        }
    }

    __aicore__ inline void FrontAxisNFastPath(uint64_t coreNNums)
    {
        uint32_t qCount = this->Q;
        if constexpr (sizeof(T) == 8) {
            qCount = this->Q * 2;
        }

        uint64_t myOffset = (GetBlockIdx() >= this->corePtr) ? (GetBlockIdx() - this->corePtr) :
                                                               (GetBlockIdx() + this->coreNum - this->corePtr);

        for (uint64_t p = myOffset; p < this->P; p += this->coreNum) {
            if (!this->maskCached) {
                this->singleNBlock = true;
                uint64_t ubStartInt8 = OFFSET_UB_LENGTH * sizeof(uint32_t);
                this->wsLocal = this->allUbLocal[ubStartInt8].template ReinterpretCast<half>();
                this->CopyInWsGm(0, this->wsLocal, coreNNums);
                this->PIPE_MTE2_S();

                uint32_t alignedQCount = this->AlignedToTarget(coreNNums * qCount, COMPARE_ALIGNED);
                this->wsLocalQ = this->allUbLocal[this->wsLocalQStart].template ReinterpretCast<half>();
                Gather(this->wsLocalQ, this->wsLocal, this->offsetLocal.template ReinterpretCast<uint32_t>(), 0,
                       alignedQCount);
                this->PIPE_V_S();

                this->compareMaskLocal = this->allUbLocal[this->compareMaskStart];
                CompareScalar(this->compareMaskLocal, this->wsLocalQ, static_cast<half>(0), CMPMODE::EQ, alignedQCount);
                this->PIPE_V_S();
                this->maskCached = true;
            }

            uint32_t alignedQCount = this->AlignedToTarget(coreNNums * qCount, COMPARE_ALIGNED);
            uint32_t alignedOrigCount = this->AlignedToTarget(coreNNums * this->Q, COMPARE_ALIGNED);

            uint64_t gmStartInt8 = (p * this->N * this->Q) * sizeof(T);
            this->CopyIn(this->xGm, gmStartInt8, this->xLocalStart, coreNNums * this->Q * sizeof(T));
            this->PIPE_MTE2_S();

            this->ExecuteDataSelect(this->xLocalStart, this->xLocalHalfStart, this->valueTensorStart,
                                    this->compareMaskLocal, alignedOrigCount, alignedQCount);

            this->CopyOut(this->yGm, gmStartInt8, this->xLocalStart, coreNNums * this->Q * sizeof(T));
            this->PIPE_MTE3_S();
        }
        this->corePtr = this->WrapAdd(this->coreNum, this->corePtr, this->P);
    }

    __aicore__ inline void ProcessOneP(uint64_t frontCoreNums, uint64_t tailCoreNums, uint64_t frontCoreNNums,
                                       uint64_t tailCoreNNums, uint64_t pId)
    {
        uint64_t coreNStart = 0;
        uint64_t coreNNums = 0;
        if (GetBlockIdx() - this->corePtr < frontCoreNums) {
            coreNStart = (GetBlockIdx() - this->corePtr) * frontCoreNNums;
            coreNNums = frontCoreNNums;
        } else if (tailCoreNums > 0) {
            coreNStart = frontCoreNums * frontCoreNNums +
                         (GetBlockIdx() - this->corePtr - frontCoreNums) * tailCoreNNums;
            coreNNums = tailCoreNNums;
        }

        uint32_t qCount = this->Q;
        if constexpr (sizeof(T) == 8) {
            qCount = this->Q * 2;
        }

        bool willWork = this->WillWorkForTask(frontCoreNums + tailCoreNums);
        if (willWork) {
            uint64_t NBlocks = coreNNums / this->ubNBlockLength;
            uint64_t leftN = coreNNums - NBlocks * this->ubNBlockLength;
            if (!this->maskCached) {
                uint64_t effectiveBlocks = (leftN > 0) ? NBlocks + 1 : NBlocks;
                this->singleNBlock = (effectiveBlocks <= 1);
            }
            for (uint64_t i = 0; i <= NBlocks; i++) {
                if (i == NBlocks && leftN == 0)
                    break;
                ProcessNBlock(i, NBlocks, leftN, coreNStart, pId, qCount);
            }
            if (this->singleNBlock) {
                this->maskCached = true;
            }
        }
    }

    __aicore__ inline void ProcessNBlock(uint64_t i, uint64_t NBlocks, uint64_t leftN, uint64_t coreNStart,
                                         uint64_t pId, uint32_t qCount)
    {
        uint64_t NNums = (i == NBlocks ? leftN : this->ubNBlockLength);
        uint32_t alignedQCount = this->AlignedToTarget(NNums * qCount, COMPARE_ALIGNED);
        uint32_t alignedOrigCount = this->AlignedToTarget(NNums * this->Q, COMPARE_ALIGNED);

        if (!this->maskCached) {
            uint64_t ubStartInt8 = OFFSET_UB_LENGTH * sizeof(uint32_t);
            this->wsLocal = this->allUbLocal[ubStartInt8].template ReinterpretCast<half>();
            uint64_t gmStartHalf = (coreNStart + i * this->ubNBlockLength);
            this->CopyInWsGm(gmStartHalf, this->wsLocal, NNums);
            this->PIPE_MTE2_S();

            this->wsLocalQ = this->allUbLocal[this->wsLocalQStart].template ReinterpretCast<half>();
            Gather(this->wsLocalQ, this->wsLocal, this->offsetLocal.template ReinterpretCast<uint32_t>(), 0,
                   alignedQCount);
            this->PIPE_V_S();

            this->compareMaskLocal = this->allUbLocal[this->compareMaskStart];
            CompareScalar(this->compareMaskLocal, this->wsLocalQ, static_cast<half>(0), CMPMODE::EQ, alignedQCount);
            this->PIPE_V_S();
        }

        uint64_t gmStartInt8 = (pId * this->N * this->Q + (coreNStart + i * this->ubNBlockLength) * this->Q) *
                               sizeof(T);
        this->CopyIn(this->xGm, gmStartInt8, this->xLocalStart, NNums * this->Q * sizeof(T));
        this->PIPE_MTE2_S();

        this->ExecuteDataSelect(this->xLocalStart, this->xLocalHalfStart, this->valueTensorStart,
                                this->compareMaskLocal, alignedOrigCount, alignedQCount);

        this->CopyOut(this->yGm, gmStartInt8, this->xLocalStart, NNums * this->Q * sizeof(T));
        this->PIPE_MTE3_S();
    }

    __aicore__ inline void InitGather(uint64_t coreNNums)
    {
        uint32_t qCount = this->Q;
        if constexpr (sizeof(T) == 8) {
            qCount = this->Q * 2;
        }

        this->offsetLocal = this->allUbLocal.template ReinterpretCast<int32_t>();
        int32_t dupTimes = coreNNums >= INT32_ALIGNED_NUM ? INT32_ALIGNED_NUM : coreNNums;

        // 对齐 Duplicate 的 Vector 计算量
        uint32_t dupCalCount = this->AlignedToTarget(qCount, COMPARE_ALIGNED);
        for (int32_t i = 0; i < dupTimes; i++) {
            Duplicate(this->offsetLocal, static_cast<int32_t>(i * sizeof(half)), dupCalCount);
            this->PIPE_V_S();
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(qCount * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(this->gatherGm[i * qCount], this->offsetLocal, copyParams);
            this->PIPE_MTE3_S();
        }

        this->ubNBlockLength = OFFSET_UB_LENGTH / qCount;
        this->ubNBlockLength = this->ubNBlockLength >= coreNNums ? coreNNums : this->ubNBlockLength;

        this->wsLocalQStart = OFFSET_UB_LENGTH * sizeof(uint32_t) +
                              this->AlignedToTarget(this->ubNBlockLength * sizeof(half), INT8_ALIGNED_NUM);
        this->compareMaskStart = this->wsLocalQStart +
                                 this->AlignedToTarget(this->ubNBlockLength * qCount * sizeof(half), COMPARE_ALIGNED);
        this->xLocalStart = this->compareMaskStart +
                            this->AlignedToTarget(this->ubNBlockLength * qCount, COMPARE_ALIGNED);
        this->xLocalStart = (this->xLocalStart == this->compareMaskStart ? this->compareMaskStart + COMPARE_ALIGNED :
                                                                           this->xLocalStart);

        if constexpr (sizeof(T) == 1) {
            this->xLocalHalfStart = this->xLocalStart +
                                    this->AlignedToTarget(this->ubNBlockLength * this->Q * sizeof(T), COMPARE_ALIGNED);
        } else if constexpr (sizeof(T) == 8) {
            this->valueTensorStart = this->xLocalStart +
                                     this->AlignedToTarget(this->ubNBlockLength * this->Q * sizeof(T), COMPARE_ALIGNED);
            uint32_t alignedTotalElements = this->AlignedToTarget(this->ubNBlockLength * this->Q, COMPARE_ALIGNED);
            this->BroadcastValueTensor8B(this->valueTensorStart, alignedTotalElements);
        }

        this->ProcessOffsetGather(this->offsetLocal, this->ubNBlockLength, qCount);
    }

protected:
    LocalTensor<int32_t> offsetLocal;
    LocalTensor<half> wsLocalQ;
    LocalTensor<uint8_t> compareMaskLocal;
    LocalTensor<T> valueTensorLocal;

    uint64_t ubNBlockLength = 0;
    uint64_t wsLocalQStart = 0;
    uint64_t compareMaskStart = 0;
    uint64_t xLocalStart = 0;
    uint64_t xLocalHalfStart = 0;
    uint64_t valueTensorStart = 0;
    bool maskCached = false;
    bool singleNBlock = false;
};

template <typename T, typename U>
__aicore__ void index_fill_front_n(GM_ADDR x, GM_ADDR indices, GM_ADDR value, GM_ADDR y, GM_ADDR workspace,
                                   GM_ADDR tiling, const IndexFillTilingData* tilingData, TPipe* tPipe)
{
    if (sizeof(T) == sizeof(half)) {
        IndexFillFrontN<half, U> op;
        op.Init(x, indices, value, y, workspace, tiling, tilingData, tPipe);
        op.Process();
    } else if (sizeof(T) == sizeof(float)) {
        IndexFillFrontN<float, U> op;
        op.Init(x, indices, value, y, workspace, tiling, tilingData, tPipe);
        op.Process();
    } else if (sizeof(T) == sizeof(int8_t)) {
        IndexFillFrontN<int8_t, U> op;
        op.Init(x, indices, value, y, workspace, tiling, tilingData, tPipe);
        op.Process();
    } else if (sizeof(T) == sizeof(int64_t)) {
        IndexFillFrontN<int64_t, U> op;
        op.Init(x, indices, value, y, workspace, tiling, tilingData, tPipe);
        op.Process();
    }
}
} // namespace IndexFillNS
#endif
