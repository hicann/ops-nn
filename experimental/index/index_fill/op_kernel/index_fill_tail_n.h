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
 * \file index_fill_tail_n.h
 * \brief tail axis and N >= core num * 256. core will split N.
 */
#ifndef INDEX_FILL_TAIL_N_H
#define INDEX_FILL_TAIL_N_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "index_fill_base.h"
#include <type_traits>

namespace IndexFillNS {
using namespace AscendC;
constexpr uint32_t N_UB_LENGTH = 12288;

template <typename T, typename U>
class IndexFillTailN : public IndexFillBase<T, U> {
public:
    __aicore__ inline void Process()
    {
        this->ExecIndicesTask();
        this->ReadValue();
        TailAxisN();
    }

    __aicore__ inline void TailAxisN()
    {
        uint32_t qCount = 1;
        this->nBlockLength = N_UB_LENGTH; // 1B/2B/4B 默认使用 12288

        if constexpr (sizeof(T) == 8) {
            qCount = 2;
            // 【核心修复】强行压低 8B 数据每次进 UB 的处理条数，控制总 UB 大小在 192KB 左右，防止 MPU 越界报错
            this->nBlockLength = 4096;
        }

        if (this->nBlockLength == 0)
            this->nBlockLength = 1;

        // 统筹规划 UB 内存起始点 (修复了 Mask 占用计算，统一使用 1 byte)
        this->offsetLocalStart = 0;
        this->compareMaskStart = this->offsetLocalStart +
                                 this->AlignedToTarget(this->nBlockLength * qCount * sizeof(int32_t), COMPARE_ALIGNED);
        this->wsLocalStart = this->compareMaskStart +
                             this->AlignedToTarget(this->nBlockLength * qCount * sizeof(uint8_t), COMPARE_ALIGNED);
        this->wsLocalQStart = this->wsLocalStart +
                              this->AlignedToTarget(this->nBlockLength * sizeof(half), COMPARE_ALIGNED);
        this->xLocalStart = this->wsLocalQStart +
                            this->AlignedToTarget(this->nBlockLength * qCount * sizeof(half), COMPARE_ALIGNED);

        // xLocalHalf 和 valueTensor 在不同数据位宽下是互斥使用的，所以共享同一块内存起点
        this->xLocalHalfStart = this->xLocalStart +
                                this->AlignedToTarget(this->nBlockLength * qCount * sizeof(T), COMPARE_ALIGNED);
        this->valueTensorStart = this->xLocalStart +
                                 this->AlignedToTarget(this->nBlockLength * qCount * sizeof(T), COMPARE_ALIGNED);

        uint64_t frontCoreNums = 0;
        uint64_t tailCoreNums = 0;
        uint64_t frontCoreNNums = 0;
        uint64_t tailCoreNNums = 0;
        this->coreSplit(this->coreNum, this->N, frontCoreNums, tailCoreNums, frontCoreNNums, tailCoreNNums);

        uint64_t coreNStart = 0;
        uint64_t coreNNums = 0;
        if (GetBlockIdx() < frontCoreNums) {
            coreNStart = GetBlockIdx() * frontCoreNNums;
            coreNNums = frontCoreNNums;
        } else if (tailCoreNums > 0) {
            coreNStart = frontCoreNums * frontCoreNNums + (GetBlockIdx() - frontCoreNums) * tailCoreNNums;
            coreNNums = tailCoreNNums;
        }

        InitGatherAndValue();

        uint64_t NBlocks = coreNNums / this->nBlockLength;
        uint64_t leftN = coreNNums - NBlocks * this->nBlockLength;
        for (uint64_t i = 0; i <= NBlocks; i++) {
            if (i == NBlocks && leftN == 0) {
                break;
            }
            uint64_t nNums = (i == NBlocks ? leftN : this->nBlockLength);
            InitSelect(coreNStart + i * this->nBlockLength, nNums);
            for (uint64_t j = 0; j < this->P; j++) {
                ProcessPN(j, coreNStart + i * this->nBlockLength, nNums);
            }
        }
    }

    __aicore__ inline void InitGatherAndValue()
    {
        uint32_t qCount = 1;
        if constexpr (sizeof(T) == 8) {
            qCount = 2;
            this->offsetLocal = this->allUbLocal[this->offsetLocalStart].template ReinterpretCast<int32_t>();
            int32_t dupTimes = this->nBlockLength >= INT32_ALIGNED_NUM ? INT32_ALIGNED_NUM : this->nBlockLength;
            uint32_t dupCalCount = this->AlignedToTarget(qCount, COMPARE_ALIGNED);

            for (int32_t i = 0; i < dupTimes; i++) {
                Duplicate(this->offsetLocal, static_cast<int32_t>(i * sizeof(half)), dupCalCount);
                this->PIPE_V_S();
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(qCount * sizeof(int32_t)), 0, 0, 0};
                DataCopyPad(this->gatherGm[i * qCount], this->offsetLocal, copyParams);
                this->PIPE_MTE3_S();
            }

            this->ProcessOffsetGather(this->offsetLocal, this->nBlockLength, qCount);

            uint32_t alignedTotalElements = this->AlignedToTarget(this->nBlockLength, COMPARE_ALIGNED);
            this->BroadcastValueTensor8B(this->valueTensorStart, alignedTotalElements);
        }
    }

    __aicore__ inline void InitSelect(uint64_t nId, uint64_t nNums)
    {
        this->wsLocal = this->allUbLocal[this->wsLocalStart].template ReinterpretCast<half>();
        this->compareMaskLocal = this->allUbLocal[this->compareMaskStart];

        this->CopyInWsGm(nId, this->wsLocal, nNums);
        this->PIPE_MTE2_S();

        if constexpr (sizeof(T) == 8) {
            uint32_t alignedQCount = this->AlignedToTarget(nNums * 2, COMPARE_ALIGNED);
            this->wsLocalQ = this->allUbLocal[this->wsLocalQStart].template ReinterpretCast<half>();
            Gather(this->wsLocalQ, this->wsLocal, this->offsetLocal.template ReinterpretCast<uint32_t>(), 0,
                   alignedQCount);
            this->PIPE_V_S();
            CompareScalar(this->compareMaskLocal, this->wsLocalQ, static_cast<half>(0), CMPMODE::EQ, alignedQCount);
            this->PIPE_V_S();
        } else {
            uint32_t alignedOrigNums = this->AlignedToTarget(nNums, COMPARE_ALIGNED);
            CompareScalar(this->compareMaskLocal, this->wsLocal, static_cast<half>(0), CMPMODE::EQ, alignedOrigNums);
            this->PIPE_V_S();
        }
    }

    __aicore__ inline void ProcessPN(uint64_t pId, uint64_t nId, uint64_t nNums)
    {
        uint64_t gmStartInt8 = (pId * this->N + nId) * sizeof(T);
        this->CopyIn(this->xGm, gmStartInt8, this->xLocalStart, nNums * sizeof(T));
        this->PIPE_MTE2_S();

        uint32_t alignedOrigNums = this->AlignedToTarget(nNums, COMPARE_ALIGNED);
        uint32_t alignedQCount = this->AlignedToTarget(nNums * 2, COMPARE_ALIGNED);

        // Use centralized select!
        this->ExecuteDataSelect(this->xLocalStart, this->xLocalHalfStart, this->valueTensorStart,
                                this->compareMaskLocal, alignedOrigNums, alignedQCount);

        this->CopyOut(this->yGm, gmStartInt8, this->xLocalStart, nNums * sizeof(T));
        this->PIPE_MTE3_S();
    }

protected:
    LocalTensor<int32_t> offsetLocal;
    LocalTensor<half> wsLocalQ;
    LocalTensor<uint8_t> compareMaskLocal;
    LocalTensor<T> valueTensorLocal;

    uint64_t nBlockLength = 0;
    uint64_t offsetLocalStart = 0;
    uint64_t compareMaskStart = 0;
    uint64_t wsLocalStart = 0;
    uint64_t wsLocalQStart = 0;
    uint64_t xLocalStart = 0;
    uint64_t xLocalHalfStart = 0;
    uint64_t valueTensorStart = 0;
};

template <typename T, typename U>
__aicore__ void index_fill_tail_n(GM_ADDR x, GM_ADDR indices, GM_ADDR value, GM_ADDR y, GM_ADDR workspace,
                                  GM_ADDR tiling, const IndexFillTilingData* tilingData, TPipe* tPipe)
{
    if (sizeof(T) == sizeof(half)) {
        IndexFillTailN<half, U> op;
        op.Init(x, indices, value, y, workspace, tiling, tilingData, tPipe);
        op.Process();
    } else if (sizeof(T) == sizeof(float)) {
        IndexFillTailN<float, U> op;
        op.Init(x, indices, value, y, workspace, tiling, tilingData, tPipe);
        op.Process();
    } else if (sizeof(T) == sizeof(int8_t)) {
        IndexFillTailN<int8_t, U> op;
        op.Init(x, indices, value, y, workspace, tiling, tilingData, tPipe);
        op.Process();
    } else if (sizeof(T) == sizeof(int64_t)) {
        IndexFillTailN<int64_t, U> op;
        op.Init(x, indices, value, y, workspace, tiling, tilingData, tPipe);
        op.Process();
    }
}
} // namespace IndexFillNS
#endif
