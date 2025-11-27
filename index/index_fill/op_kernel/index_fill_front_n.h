/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

namespace IndexFillNS {
using namespace AscendC;
constexpr uint32_t OFFSET_UB_LENGTH = 8192;

template <typename T, typename U>
class IndexFillFrontN : public IndexFillBase<T, U> {
public:
  __aicore__ inline void Process() {
    this->ExecIndicesTask();
    this->ReadValue();
    FrontAxisN();
  }

  __aicore__ inline void FrontAxisN() {
    // UB布局：offsetLocal[8192 * sizeof(uint32)], wsLocal[NBlocks * sizeof(half)], wsLocalQ[NBlocks * Q * sizeof(half)], 
    //         compareMask(NBlocks * Q / 8), xLocal[NBlocks * Q * sizeof(T)]

    // 先计算一个P可以使用多少个核
    uint64_t PCoreNum = this->coreNum / this->P;
    PCoreNum = PCoreNum == 0 ? 1 : PCoreNum;
    // 一个P动用的核来平分N(这块切核，可以和切Q那里合并成一个函数)
    uint64_t frontCoreNums = 0;
    uint64_t tailCoreNums = 0;
    uint64_t frontCoreNNums = 0;
    uint64_t tailCoreNNums = 0;
    this->coreSplit(PCoreNum, this->N, frontCoreNums, tailCoreNums, frontCoreNNums, tailCoreNNums);
    InitGather(frontCoreNNums);
    // 遍历P并处理
    for (uint64_t p = 0; p < this->P; p++) {
        ProcessOneP(frontCoreNums, tailCoreNums, frontCoreNNums, tailCoreNNums, p);
    }
  }

  __aicore__ inline void ProcessOneP(uint64_t frontCoreNums, uint64_t tailCoreNums, uint64_t frontCoreNNums,
                                     uint64_t tailCoreNNums, uint64_t pId) {
    // UB布局：offsetLocal[8192 * sizeof(uint32)], wsLocal[NBlocks * sizeof(half)], wsLocalQ[NBlocks * Q * sizeof(half)], 
    //         compareMask(NBlocks * Q / 8), xLocal[NBlocks * Q * sizeof(T)]
    uint64_t coreNStart = 0;
    uint64_t coreNNums = 0;
    if (GetBlockIdx() - this->corePtr < frontCoreNums) {
        coreNStart = (GetBlockIdx() - this->corePtr) * frontCoreNNums;
        coreNNums = frontCoreNNums;
    } else if(tailCoreNums > 0){
        coreNStart = frontCoreNums * frontCoreNNums + (GetBlockIdx() - this->corePtr - frontCoreNums) * tailCoreNNums;
        coreNNums = tailCoreNNums;
    }

    bool willWork = this->WillWorkForTask(frontCoreNums + tailCoreNums);
    if (willWork) {
        // 处理分给自己的一个或多个N。先把N切块
        uint64_t NBlocks = coreNNums / this->ubNBlockLength;
        uint64_t leftN = coreNNums - NBlocks * this->ubNBlockLength;
        for (uint64_t i = 0; i <= NBlocks; i++) {
            if (i == NBlocks && leftN == 0) {
                break;
            }
            uint64_t NNums = i == NBlocks ? leftN : this->ubNBlockLength;
            // 搬入标记Tensor
            uint64_t ubStartInt8 = OFFSET_UB_LENGTH * sizeof(uint32_t); // 标记Tensor放在offsetLocal之后
            this->wsLocal = this->allUbLocal[ubStartInt8].template ReinterpretCast<half>();
            uint64_t gmStartHalf = (coreNStart + i * this->ubNBlockLength);
            uint64_t mteNumsHalf = NNums;
            this->CopyInWsGm(gmStartHalf, this->wsLocal, mteNumsHalf);
            this->PIPE_MTE2_S();
            // 复制标记Tensor(ub起始位置要对齐)
            uint64_t wsLocalQStart = OFFSET_UB_LENGTH * sizeof(uint32_t) +
                                     this->AlignedToTarget(this->ubNBlockLength * sizeof(half), INT8_ALIGNED_NUM);
            this->wsLocalQ = this->allUbLocal[wsLocalQStart].template ReinterpretCast<half>();
            Gather(this->wsLocalQ, this->wsLocal, this->offsetLocal.template ReinterpretCast<uint32_t>(), 0, NNums * this->Q);
            this->PIPE_V_S();
            // 标记Tensor转select-mask
            uint64_t compareMaskStart = wsLocalQStart + 
                                        this->AlignedToTarget(this->ubNBlockLength * this->Q * sizeof(half), COMPARE_ALIGNED);
            this->compareMaskLocal = this->allUbLocal[compareMaskStart];
            uint32_t compareNums = this->AlignedToTarget(NNums * this->Q, COMPARE_ALIGNED);
            CompareScalar(this->compareMaskLocal, this->wsLocalQ, static_cast<half>(0), CMPMODE::EQ, compareNums);
            this->PIPE_V_S();
            // 搬入xLocal
            uint64_t xLocalStart = compareMaskStart + 
                                   this->AlignedToTarget(this->ubNBlockLength * this->Q, COMPARE_ALIGNED);
            xLocalStart = (xLocalStart == compareMaskStart ? compareMaskStart + COMPARE_ALIGNED : xLocalStart);
            uint64_t gmStartInt8 = (pId * this->N * this->Q + (coreNStart + i * this->ubNBlockLength) * this->Q) * sizeof(T);
            this->CopyIn(this->xGm, gmStartInt8, xLocalStart, NNums * this->Q * sizeof(T));
            this->PIPE_MTE2_S();
            this->xLocal = this->allUbLocal[xLocalStart].template ReinterpretCast<T>();
            // 替换为value
            Select(this->xLocal, this->compareMaskLocal, this->xLocal, this->value,
                  SELMODE::VSEL_TENSOR_SCALAR_MODE, NNums * this->Q);
            this->PIPE_V_S();
            // 搬出xLocal
            this->CopyOut(this->yGm, gmStartInt8, xLocalStart, NNums * this->Q * sizeof(T));
            this->PIPE_MTE3_S();
        }
    }
  }

   __aicore__ inline void InitGather(uint64_t coreNNums) {
    // offsetLocal用于将wsLocal每个数复制Q次，offsetLocal = [0000000 11111111......] * sizeof(half)
    // UB布局：offsetLocal[8192 * sizeof(uint32)], wsLocal[NBlocks * sizeof(half)], wsLocalQ[NBlocks * Q * sizeof(half)], 
    //         compareMask(NBlocks * Q / 8), xLocal[NBlocks * Q * sizeof(T)]
    this->offsetLocal = this->allUbLocal.template ReinterpretCast<int32_t>();
    int32_t dupTimes = coreNNums >= INT32_ALIGNED_NUM ? INT32_ALIGNED_NUM : coreNNums;
    for(int32_t i = 0; i < dupTimes; i++) {
        Duplicate(this->offsetLocal, static_cast<int32_t>(i * sizeof(half)), this->Q);
        this->PIPE_V_S();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->Q * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(this->gatherGm[i * this->Q], this->offsetLocal, copyParams);
        this->PIPE_MTE3_S();
    }

    this->ubNBlockLength = OFFSET_UB_LENGTH / this->Q;
    this->ubNBlockLength = this->ubNBlockLength >= coreNNums ? coreNNums : this->ubNBlockLength;
    uint64_t mteTimes = this->ubNBlockLength / INT32_ALIGNED_NUM;
    uint64_t left = this->ubNBlockLength - mteTimes * INT32_ALIGNED_NUM;
    for (uint64_t i = 0; i <= mteTimes; i++) {
        if (i == mteTimes && left == 0) {
            break;
        }
        uint64_t mteLength = (i == mteTimes ? left * this->Q : INT32_ALIGNED_NUM * this->Q);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(mteLength * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
        DataCopyPad(this->offsetLocal[i * INT32_ALIGNED_NUM * this->Q], this->gatherGm, copyParams, padParams);
        this->PIPE_MTE2_S();
        Adds(this->offsetLocal[i * INT32_ALIGNED_NUM * this->Q], this->offsetLocal[i * INT32_ALIGNED_NUM * this->Q],
            static_cast<int32_t>(i * INT32_ALIGNED_NUM * sizeof(half)), mteLength);
        this->PIPE_V_S();
    }
    this->PIPE_MTE2_S();
   }

protected:
   LocalTensor<int32_t> offsetLocal;
   LocalTensor<half> wsLocalQ;
   LocalTensor<uint8_t> compareMaskLocal;
   uint64_t ubNBlockLength = 0; // 对N切块时，UB内一次性可以容纳的“N”
};

template <typename T, typename U>
__aicore__ void index_fill_front_n(GM_ADDR x, GM_ADDR indices, GM_ADDR value, GM_ADDR y, GM_ADDR workspace,
                                   GM_ADDR tiling, const IndexFillTilingData* tilingData, TPipe* tPipe) {
    if (sizeof(T) == sizeof(half)) {
        IndexFillFrontN<half, U> op;
        op.Init(x, indices, value, y, workspace, tiling, tilingData, tPipe);
        op.Process();
    } else if (sizeof(T) == sizeof(float)) {
        IndexFillFrontN<float, U> op;
        op.Init(x, indices, value, y, workspace, tiling, tilingData, tPipe);
        op.Process();
    }
}
}
#endif