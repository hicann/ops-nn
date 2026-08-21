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
 * \file index_fill_front_p.h
 * \brief front axis and Q < 64 && N < 64. core will split P.
 */
#ifndef INDEX_FILL_FRONT_P_H
#define INDEX_FILL_FRONT_P_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "index_fill_base.h"
#include <type_traits>

namespace IndexFillNS {
using namespace AscendC;
constexpr uint32_t P_UB_LENGTH = 8192;

template <typename T, typename U>
class IndexFillFrontP : public IndexFillBase<T, U> {
public:
    __aicore__ inline void Process()
    {
        this->ExecIndicesTask();
        this->ReadValue();
        FrontAxisP();
    }

    __aicore__ inline void FrontAxisP()
    {
        // 全部核均分P
        uint64_t corePStart = 0;
        uint64_t corePNums = 0;
        this->SliceP(corePStart, corePNums);

        uint32_t qCount = this->Q;
        if constexpr (sizeof(T) == 8) {
            qCount = this->Q * 2;
        }

        if (corePNums > 0) {
            // 计算一次UB一次可以容纳多少个P
            this->ubPBlockLength = P_UB_LENGTH / (this->N * qCount);
            if (this->ubPBlockLength == 0) {
                this->ubPBlockLength = 1;
            }
            if (this->N * qCount < COMPARE_ALIGNED) {
                this->ubPBlockLength = COMPARE_ALIGNED;
            }
            if (this->ubPBlockLength > corePNums) {
                this->ubPBlockLength = corePNums;
            }

            uint64_t totalDataBytes = corePNums * this->N * this->Q * sizeof(T);
            uint64_t perRowBytes = this->Q * sizeof(T);
            if (totalDataBytes <= P_UB_LENGTH * sizeof(T) && corePNums <= this->ubPBlockLength &&
                perRowBytes >= INT8_ALIGNED_NUM) {
                DirectProcessP(corePStart, corePNums);
            } else {
                InitSelect(qCount);

                uint64_t pBlocks = corePNums / this->ubPBlockLength;
                uint64_t leftP = corePNums - pBlocks * this->ubPBlockLength;
                for (uint64_t i = 0; i <= pBlocks; i++) {
                    if (i == pBlocks && leftP == 0)
                        break;
                    ProcessPBlock(i, pBlocks, leftP, corePStart, qCount);
                }
            }
        }
    }

    __aicore__ inline void DirectProcessP(uint64_t corePStart, uint64_t corePNums)
    {
        // Fast path: per-N conditional fill/copy without mask construction
        // Fill positions: CopyOut from immutable value buffer (can batch without barriers)
        // Copy positions: CopyIn + CopyOut with barriers
        uint64_t wsLocalOffset = 0;
        this->wsLocal = this->allUbLocal[wsLocalOffset].template ReinterpretCast<half>();
        this->CopyInWsGm(0, this->wsLocal, this->N);
        this->PIPE_MTE2_S();

        uint64_t valueBufferOffset = this->AlignedToTarget(this->N * sizeof(half), INT8_ALIGNED_NUM);
        uint64_t dataOffset = valueBufferOffset + this->AlignedToTarget(this->Q * sizeof(T), INT8_ALIGNED_NUM);

        // Pre-fill value buffer with Q copies of value using SetValue + doubling
        auto valueBuf = this->allUbLocal[valueBufferOffset].template ReinterpretCast<T>();
        uint32_t fillCount = (this->Q < INT8_ALIGNED_NUM) ? this->Q : INT8_ALIGNED_NUM;
        for (uint32_t i = 0; i < fillCount; i++) {
            valueBuf.SetValue(i, this->value);
        }
        if (this->Q > INT8_ALIGNED_NUM) {
            auto valBufUint8 = this->allUbLocal[valueBufferOffset].template ReinterpretCast<uint8_t>();
            uint32_t baseBytes = INT8_ALIGNED_NUM * sizeof(T);
            uint32_t totalBytes = this->Q * sizeof(T);
            for (uint32_t pos = baseBytes; pos < totalBytes; pos *= 2) {
                uint32_t copyLen = ((pos * 2) > totalBytes) ? (totalBytes - pos) : pos;
                DataCopy(valBufUint8[pos], valBufUint8[0], copyLen);
                PipeBarrier<PIPE_V>();
            }
            this->PIPE_V_S();
        }

        for (uint64_t p = 0; p < corePNums; p++) {
            for (uint64_t n = 0; n < this->N; n++) {
                float wsValue = this->wsLocal.GetValue(n);
                uint64_t gmStartInt8 = ((corePStart + p) * this->N * this->Q + n * this->Q) * sizeof(T);
                uint64_t mteNumsInt8 = this->Q * sizeof(T);
                if (wsValue > 0) {
                    // Fill: queue CopyOut from immutable value buffer (no barrier needed)
                    this->CopyOut(this->yGm, gmStartInt8, valueBufferOffset, mteNumsInt8);
                } else {
                    // Copy: need barriers since dataOffset is reused
                    this->PIPE_MTE3_S();
                    this->CopyIn(this->xGm, gmStartInt8, dataOffset, mteNumsInt8);
                    this->PIPE_MTE2_S();
                    this->CopyOut(this->yGm, gmStartInt8, dataOffset, mteNumsInt8);
                }
            }
        }
        this->PIPE_MTE3_S();
    }

    __aicore__ inline void ProcessPBlock(uint64_t i, uint64_t pBlocks, uint64_t leftP, uint64_t corePStart,
                                         uint32_t qCount)
    {
        uint64_t pNums = (i == pBlocks ? leftP : this->ubPBlockLength);
        uint32_t alignedQCount = this->AlignedToTarget(pNums * this->N * qCount, COMPARE_ALIGNED);
        uint32_t alignedOrigCount = this->AlignedToTarget(pNums * this->N * this->Q, COMPARE_ALIGNED);

        uint64_t gmStartInt8 = (corePStart * this->N * this->Q + i * this->ubPBlockLength * this->N * this->Q) *
                               sizeof(T);
        uint64_t mteNumsInt8 = pNums * this->N * this->Q * sizeof(T);

        this->CopyIn(this->xGm, gmStartInt8, this->xLocalStart, mteNumsInt8);
        this->PIPE_MTE2_S();

        // Use centralized select!
        this->ExecuteDataSelect(this->xLocalStart, this->xLocalHalfStart, this->valueTensorStart,
                                this->compareMaskLocal, alignedOrigCount, alignedQCount);

        this->CopyOut(this->yGm, gmStartInt8, this->xLocalStart, mteNumsInt8);
        this->PIPE_MTE3_S();
    }

    __aicore__ inline void InitSelect(uint32_t qCount)
    {
        // UB布局：compareMaskLocal，repeatMarkerLocal(将被复用)，markerLocal，tempLocal
        uint64_t repeatMarkerLocalStart = this->AlignedToTarget(this->ubPBlockLength * this->N * qCount,
                                                                COMPARE_ALIGNED);
        repeatMarkerLocalStart = (repeatMarkerLocalStart == 0 ? COMPARE_ALIGNED : repeatMarkerLocalStart);

        uint64_t markerLocalStart = repeatMarkerLocalStart +
                                    this->AlignedToTarget(this->ubPBlockLength * this->N * qCount * sizeof(half),
                                                          INT8_ALIGNED_NUM);
        uint64_t tempLocalStart = markerLocalStart + this->AlignedToTarget(this->N * sizeof(half), INT8_ALIGNED_NUM);

        this->compareMaskLocal = this->allUbLocal[0].template ReinterpretCast<uint8_t>();
        auto repeatMarkerLocal = this->allUbLocal[repeatMarkerLocalStart].template ReinterpretCast<half>();
        auto markerLocal = this->allUbLocal[markerLocalStart].template ReinterpretCast<half>();
        auto tempLocal = this->allUbLocal[tempLocalStart].template ReinterpretCast<half>();

        // 搬入标记Tensor
        this->CopyInWsGm(0, markerLocal, this->N);
        this->PIPE_MTE2_S();

        // 每个标记复制 qCount 次后搬出
        uint32_t dupCalCount = this->AlignedToTarget(qCount, COMPARE_ALIGNED);
        for (uint64_t i = 0; i < this->N; i++) {
            half marker = markerLocal.GetValue(i);
            Duplicate(tempLocal, marker, dupCalCount);
            this->PIPE_V_S();
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(qCount * sizeof(half)), 0, 0, 0};
            DataCopyPad(this->repeatGm[i * qCount], tempLocal, copyParams);
            this->PIPE_MTE3_S();
        }

        // 搬入复制后的标记Tensor
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->N * qCount * sizeof(half)), 0, 0, 0};
        DataCopyPadExtParams<half> padParams{true, 0, 0, 0};
        DataCopyPad(repeatMarkerLocal, this->repeatGm, copyParams, padParams);
        this->PIPE_MTE2_S();

        // 复制后的标记tensor再搬出 ubPBlockLength 次
        if (this->ubPBlockLength > 1) {
            for (uint64_t i = 0; i < this->ubPBlockLength; i++) {
                DataCopyPad(this->repeatGm[i * this->N * qCount], repeatMarkerLocal, copyParams);
            }
            this->PIPE_MTE3_S();

            // 搬入完全复制完成的标记Tensor
            DataCopyExtParams copyParamsPNQ{
                1, static_cast<uint32_t>(this->ubPBlockLength * this->N * qCount * sizeof(half)), 0, 0, 0};
            DataCopyPad(repeatMarkerLocal, this->repeatGm, copyParamsPNQ, padParams);
            this->PIPE_MTE2_S();
        }

        // half Tensor转mask Tensor
        uint32_t compareNums = this->AlignedToTarget(this->ubPBlockLength * this->N * qCount, COMPARE_ALIGNED);
        CompareScalar(this->compareMaskLocal, repeatMarkerLocal, static_cast<half>(0), CMPMODE::EQ, compareNums);
        this->PIPE_V_S();

        // 内存复用：释放 repeatMarkerLocal 空间，作为数据操作的起点
        this->xLocalStart = repeatMarkerLocalStart;

        // 预分配辅助空间
        if constexpr (sizeof(T) == 1) {
            this->xLocalHalfStart = this->xLocalStart +
                                    this->AlignedToTarget(this->ubPBlockLength * this->N * this->Q * sizeof(T),
                                                          COMPARE_ALIGNED);
        } else if constexpr (sizeof(T) == 8) {
            this->valueTensorStart = this->xLocalStart +
                                     this->AlignedToTarget(this->ubPBlockLength * this->N * this->Q * sizeof(T),
                                                           COMPARE_ALIGNED);
            uint32_t alignedTotalElements = this->AlignedToTarget(this->ubPBlockLength * this->N * this->Q,
                                                                  COMPARE_ALIGNED);
            this->BroadcastValueTensor8B(this->valueTensorStart, alignedTotalElements);
        }
    }

protected:
    LocalTensor<uint8_t> compareMaskLocal;
    LocalTensor<T> valueTensorLocal;

    uint64_t ubPBlockLength = 0;
    uint64_t xLocalStart = 0;
    uint64_t xLocalHalfStart = 0;
    uint64_t valueTensorStart = 0;
};

template <typename T, typename U>
__aicore__ void index_fill_front_p(GM_ADDR x, GM_ADDR indices, GM_ADDR value, GM_ADDR y, GM_ADDR workspace,
                                   GM_ADDR tiling, const IndexFillTilingData* tilingData, TPipe* tPipe)
{
    if (sizeof(T) == sizeof(half)) {
        IndexFillFrontP<half, U> op;
        op.Init(x, indices, value, y, workspace, tiling, tilingData, tPipe);
        op.Process();
    } else if (sizeof(T) == sizeof(float)) {
        IndexFillFrontP<float, U> op;
        op.Init(x, indices, value, y, workspace, tiling, tilingData, tPipe);
        op.Process();
    } else if (sizeof(T) == sizeof(int8_t)) {
        IndexFillFrontP<int8_t, U> op;
        op.Init(x, indices, value, y, workspace, tiling, tilingData, tPipe);
        op.Process();
    } else if (sizeof(T) == sizeof(int64_t)) {
        IndexFillFrontP<int64_t, U> op;
        op.Init(x, indices, value, y, workspace, tiling, tilingData, tPipe);
        op.Process();
    }
}
} // namespace IndexFillNS
#endif
