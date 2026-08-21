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
 * \file index_fill_tail_p.h
 * \brief tail axis and N < core num * 256. core will split P.
 */
#ifndef INDEX_FILL_TAIL_P_H
#define INDEX_FILL_TAIL_P_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "index_fill_base.h"
#include <type_traits>

namespace IndexFillNS {
using namespace AscendC;
constexpr uint32_t N_LIMIT = 1024;
constexpr uint32_t P_UB_LENGTH_MIN = 4096;
constexpr uint32_t P_UB_LENGTH_MAX = 12288;

template <typename T, typename U>
class IndexFillTailP : public IndexFillBase<T, U> {
public:
    __aicore__ inline void Process()
    {
        this->ExecIndicesTask();
        this->ReadValue();
        TailAxisP();
    }

    __aicore__ inline void TailAxisP()
    {
        uint32_t qCount = (sizeof(T) == 8) ? 2 : 1;

        // 控制 P 的大小，为 8B/1B 操作腾出充足 UB 空间，避免 OOM
        if constexpr (sizeof(T) == 8) {
            this->pUbLength = 4096;
        } else {
            this->pUbLength = this->N <= N_LIMIT ? P_UB_LENGTH_MIN : P_UB_LENGTH_MAX;
        }

        this->pBlockLength = this->pUbLength / this->N;
        if (this->pBlockLength == 0)
            this->pBlockLength = 1;

        uint64_t corePStart = 0;
        uint64_t corePNums = 0;
        this->SliceP(corePStart, corePNums);

        if (corePNums == 0) {
            return;
        }

        if (this->pBlockLength > corePNums) {
            this->pBlockLength = corePNums;
        }

        // Fast path for very small workloads: bypass mask construction entirely
        // Only beneficial when corePNums is tiny (scalar per-N fill cheaper than InitSelect)
        constexpr uint32_t DIRECT_P_THRESHOLD = 4;
        uint64_t perRowBytes = this->N * sizeof(T);
        if (corePNums <= DIRECT_P_THRESHOLD && perRowBytes >= INT8_ALIGNED_NUM) {
            DirectTailProcess(corePStart, corePNums);
            return;
        }

        // 按最大对齐余量分配 UB，彻底杜绝 Vector 指令尾部越界踩踏
        uint32_t alignedMaxNums = this->AlignedToTarget(this->pBlockLength * this->N, COMPARE_ALIGNED);

        this->compareMaskStart = 0;
        uint64_t compareMaskSize = this->AlignedToTarget(alignedMaxNums * qCount * sizeof(uint8_t), COMPARE_ALIGNED);
        compareMaskSize = (compareMaskSize == 0 ? COMPARE_ALIGNED : compareMaskSize);

        this->valueTensorStart = this->compareMaskStart + compareMaskSize;
        uint64_t valueTensorSize = (sizeof(T) == 8) ?
                                       this->AlignedToTarget(alignedMaxNums * sizeof(T), COMPARE_ALIGNED) :
                                       0;

        this->xLocalHalfStart = this->valueTensorStart + valueTensorSize;
        uint64_t xLocalHalfSize = (sizeof(T) == 1) ?
                                      this->AlignedToTarget(alignedMaxNums * sizeof(half), COMPARE_ALIGNED) :
                                      0;

        this->xLocalStart = this->xLocalHalfStart + xLocalHalfSize;
        uint64_t xLocalSize = this->AlignedToTarget(alignedMaxNums * sizeof(T), COMPARE_ALIGNED);

        this->wsPNLocalStart = this->xLocalStart + xLocalSize;
        uint64_t wsPNSize = this->AlignedToTarget(alignedMaxNums * qCount * sizeof(half), COMPARE_ALIGNED);

        this->wsLocalStart = this->wsPNLocalStart + wsPNSize;

        InitSelect(corePNums);

        uint64_t pBlocks = corePNums / this->pBlockLength;
        uint64_t leftP = corePNums - pBlocks * this->pBlockLength;

        for (uint64_t i = 0; i <= pBlocks; i++) {
            if (i == pBlocks && leftP == 0)
                break;
            ProcessTailPBlock(i, pBlocks, leftP, corePStart);
        }
    }

    __aicore__ inline void DirectTailProcess(uint64_t corePStart, uint64_t corePNums)
    {
        // Fast path: load row, scalar-overwrite fill positions, write back
        // Efficient when N is small and corePNums is tiny (data fits in UB)
        this->wsLocal = this->allUbLocal.template ReinterpretCast<half>();
        this->CopyInWsGm(0, this->wsLocal, this->N);
        this->PIPE_MTE2_S();

        uint64_t dataOffset = this->AlignedToTarget(this->N * sizeof(half), INT8_ALIGNED_NUM);
        uint64_t mteNumsInt8 = this->N * sizeof(T);

        for (uint64_t p = 0; p < corePNums; p++) {
            uint64_t gmStartInt8 = (corePStart + p) * this->N * sizeof(T);
            this->CopyIn(this->xGm, gmStartInt8, dataOffset, mteNumsInt8);
            this->PIPE_MTE2_S();

            auto dataLocal = this->allUbLocal[dataOffset].template ReinterpretCast<T>();
            for (uint64_t n = 0; n < this->N; n++) {
                float wsValue = this->wsLocal.GetValue(n);
                if (wsValue > 0) {
                    dataLocal.SetValue(n, this->value);
                }
            }

            this->CopyOut(this->yGm, gmStartInt8, dataOffset, mteNumsInt8);
            this->PIPE_MTE3_S();
        }
    }

    __aicore__ inline void ProcessTailPBlock(uint64_t i, uint64_t pBlocks, uint64_t leftP, uint64_t corePStart)
    {
        uint64_t pNums = (i == pBlocks ? leftP : this->pBlockLength);
        uint64_t gmStartInt8 = (corePStart * this->N + i * this->pBlockLength * this->N) * sizeof(T);
        uint64_t mteNumsInt8 = pNums * this->N * sizeof(T);

        this->CopyIn(this->xGm, gmStartInt8, this->xLocalStart, mteNumsInt8);
        this->PIPE_MTE2_S();

        uint32_t alignedOrigNums = this->AlignedToTarget(pNums * this->N, COMPARE_ALIGNED);
        uint32_t alignedQCount = this->AlignedToTarget(pNums * this->N * 2, COMPARE_ALIGNED);

        // Use centralized select!
        this->ExecuteDataSelect(this->xLocalStart, this->xLocalHalfStart, this->valueTensorStart,
                                this->compareMaskLocal, alignedOrigNums, alignedQCount);

        this->CopyOut(this->yGm, gmStartInt8, this->xLocalStart, mteNumsInt8);
        this->PIPE_MTE3_S();
    }

    __aicore__ inline void InitSelect(uint64_t corePNums)
    {
        this->compareMaskLocal = this->allUbLocal[this->compareMaskStart].template ReinterpretCast<uint8_t>();
        auto wsPNLocal = this->allUbLocal[this->wsPNLocalStart].template ReinterpretCast<half>();
        this->wsLocal = this->allUbLocal[this->wsLocalStart].template ReinterpretCast<half>();

        uint32_t qCount = (sizeof(T) == 8) ? 2 : 1;
        uint32_t totalElements = this->pBlockLength * this->N * qCount;
        uint32_t alignedElements = this->AlignedToTarget(totalElements, COMPARE_ALIGNED);

        // 1. 搬入初始 Mask Tensor
        this->CopyInWsGm(0, this->wsLocal, this->N);
        this->PIPE_MTE2_S();

        // 预清理目标空间，防止尾部对齐未赋值区域存留脏数据
        Duplicate(wsPNLocal, static_cast<half>(0), alignedElements);
        this->PIPE_V_S();

        if constexpr (sizeof(T) == 8) {
            // ==========================================
            // 8B: scalar loop for mask expansion (barrier-free, cap makes it efficient)
            // With pBlockLength capped to corePNums, loop count is small
            // ==========================================
            for (uint32_t i = 0; i < this->pBlockLength; i++) {
                for (uint32_t j = 0; j < this->N; j++) {
                    half val = this->wsLocal.GetValue(j);
                    wsPNLocal.SetValue(i * this->N * 2 + j * 2, val);
                    wsPNLocal.SetValue(i * this->N * 2 + j * 2 + 1, val);
                }
            }
        } else {
            // ==========================================
            // 非 8B 专属：保持最高性能的 DMA repeatGm 展开
            // ==========================================
            uint32_t copyTimes = HALF_ALIGNED_NUM;
            for (uint32_t i = 0; i < copyTimes; i++) {
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->N * sizeof(half)), 0, 0, 0};
                DataCopyPad(this->repeatGm[i * this->N], this->wsLocal, copyParams);
                this->PIPE_MTE3_S();
            }

            uint64_t mteTimes = this->pBlockLength / copyTimes;
            uint64_t leftP = this->pBlockLength - mteTimes * copyTimes;

            for (uint64_t i = 0; i <= mteTimes; i++) {
                if (i == mteTimes && leftP == 0) {
                    break;
                }
                uint64_t mteLength = (i == mteTimes ? leftP * this->N : copyTimes * this->N);
                DataCopyExtParams copyParams{1, static_cast<uint32_t>(mteLength * sizeof(half)), 0, 0, 0};
                DataCopyPadExtParams<half> padParams{true, 0, 0, 0};
                DataCopyPad(wsPNLocal[i * copyTimes * this->N], this->repeatGm, copyParams, padParams);
                this->PIPE_MTE2_S();
            }
        }

        // 4. Compare 转换成最终的 Mask Tensor
        CompareScalar(this->compareMaskLocal, wsPNLocal, static_cast<half>(0), CMPMODE::EQ, alignedElements);
        this->PIPE_V_S();

        // 5. 8B 的特殊操作：利用剩余空间预构造 valueTensor 的 Vector 级广播
        if constexpr (sizeof(T) == 8) {
            uint32_t alignedTotalElements = this->AlignedToTarget(this->pBlockLength * this->N, COMPARE_ALIGNED);
            this->BroadcastValueTensor8B(this->valueTensorStart, alignedTotalElements);
        }
    }

protected:
    LocalTensor<uint8_t> compareMaskLocal;
    LocalTensor<T> valueTensorLocal;

    uint64_t pUbLength = 0;
    uint64_t pBlockLength = 0;

    uint64_t compareMaskStart = 0;
    uint64_t valueTensorStart = 0;
    uint64_t xLocalHalfStart = 0;
    uint64_t xLocalStart = 0;
    uint64_t wsPNLocalStart = 0;
    uint64_t wsLocalStart = 0;
};

template <typename T, typename U>
__aicore__ void index_fill_tail_p(GM_ADDR x, GM_ADDR indices, GM_ADDR value, GM_ADDR y, GM_ADDR workspace,
                                  GM_ADDR tiling, const IndexFillTilingData* tilingData, TPipe* tPipe)
{
    if (sizeof(T) == sizeof(half)) {
        IndexFillTailP<half, U> op;
        op.Init(x, indices, value, y, workspace, tiling, tilingData, tPipe);
        op.Process();
    } else if (sizeof(T) == sizeof(float)) {
        IndexFillTailP<float, U> op;
        op.Init(x, indices, value, y, workspace, tiling, tilingData, tPipe);
        op.Process();
    } else if (sizeof(T) == sizeof(int8_t)) {
        IndexFillTailP<int8_t, U> op;
        op.Init(x, indices, value, y, workspace, tiling, tilingData, tPipe);
        op.Process();
    } else if (sizeof(T) == sizeof(int64_t)) {
        IndexFillTailP<int64_t, U> op;
        op.Init(x, indices, value, y, workspace, tiling, tilingData, tPipe);
        op.Process();
    }
}
} // namespace IndexFillNS
#endif
