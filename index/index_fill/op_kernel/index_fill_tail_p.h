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
 * \file index_fill_tail_p.h
 * \brief tail axis and N < core num * 256. core will split P.
 */
#ifndef INDEX_FILL_TAIL_P_H
#define INDEX_FILL_TAIL_P_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "index_fill_base.h"

namespace IndexFillNS {
using namespace AscendC;
constexpr uint32_t N_LIMIT = 1024;
constexpr uint32_t P_UB_LENGTH_MIN = 4096;
constexpr uint32_t P_UB_LENGTH_MAX = 12288;

template <typename T, typename U>
class IndexFillTailP : public IndexFillBase<T, U> {
public:
  __aicore__ inline void Process() {
    this->ExecIndicesTask();
    this->ReadValue();
    this->pUbLength = this->N <= N_LIMIT? P_UB_LENGTH_MIN : P_UB_LENGTH_MAX; // pUbLength > N
    this->pBlockLength = this->pUbLength / this->N; // 一次性可处理pBlockLength行
    TailAxisP();
  }

  __aicore__ inline void TailAxisP() {
    uint64_t corePStart = 0;
    uint64_t corePNums = 0;
    this->SliceP(corePStart, corePNums);
    InitSelect(corePNums); // frontCorePNums

    uint64_t pBlocks = corePNums / this->pBlockLength;
    uint64_t leftP = corePNums - pBlocks * this->pBlockLength;
    for (uint64_t i = 0; i <= pBlocks; i++) {
        if (i == pBlocks && leftP == 0) {
            break;
        }
        // UB布局：compareMaskLocal, xLocal
        uint64_t pNums = (i == pBlocks ? leftP : this->pBlockLength);
        // 搬入x
        uint64_t gmStartInt8 = (corePStart * this->N + i * this->pBlockLength * this->N) * sizeof(T);
        uint64_t ubStartInt8 = this->AlignedToTarget(this->pBlockLength * this->N, INT8_ALIGNED_NUM);
        ubStartInt8 = (ubStartInt8 == 0 ? COMPARE_ALIGNED : ubStartInt8);
        uint64_t mteNumsInt8 = pNums * this->N  * sizeof(T);
        this->CopyIn(this->xGm, gmStartInt8, ubStartInt8, mteNumsInt8);
        this->PIPE_MTE2_S();
        // 替换为value
        this->xLocal = this->allUbLocal[ubStartInt8].template ReinterpretCast<T>();
        Select(this->xLocal, this->compareMaskLocal, this->xLocal, this->value,
               SELMODE::VSEL_TENSOR_SCALAR_MODE, pNums * this->N);
        this->PIPE_V_S();
        // 搬出xLocal
        this->CopyOut(this->yGm, gmStartInt8, ubStartInt8, mteNumsInt8);
        this->PIPE_MTE3_S();
    }
  }

  __aicore__ inline void InitSelect(uint64_t corePNums) {
    // 标记Tensor复制pBlockLength份即可。[1, this->N] -> [pBlockLength, this->N]
    // UB划分：compareMaskLocal, wsPNLocal[], wsLocal, tempLocal
    uint64_t wsPNLocalStart = this->AlignedToTarget(this->pBlockLength * this->N, COMPARE_ALIGNED);
    wsPNLocalStart = (wsPNLocalStart == 0 ? COMPARE_ALIGNED : wsPNLocalStart);
    uint64_t wsLocalStart = wsPNLocalStart +
                            this->AlignedToTarget(this->pBlockLength * this->N * sizeof(half), INT8_ALIGNED_NUM);
    uint64_t tempLocalStart = wsLocalStart +
                              this->AlignedToTarget(this->N * sizeof(half), INT8_ALIGNED_NUM);
    this->compareMaskLocal = this->allUbLocal;
    auto wsPNLocal = this->allUbLocal[wsPNLocalStart].template ReinterpretCast<half>();
    this->wsLocal = this->allUbLocal[wsLocalStart].template ReinterpretCast<half>();
    auto tempLocal = this->allUbLocal[tempLocalStart];

    // 搬入wsLocal
    this->CopyInWsGm(0, this->wsLocal, this->N);
    this->PIPE_MTE2_S();
    // 搬出wsLocal 16次
    for (uint32_t i = 0; i < HALF_ALIGNED_NUM; i++) {
      DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->N * sizeof(half)), 0, 0, 0};
      DataCopyPad(this->repeatGm[i * this->N], this->wsLocal, copyParams);
      this->PIPE_MTE3_S();
    }
    
    uint64_t mteTimes = this->pBlockLength / HALF_ALIGNED_NUM;
    uint64_t leftP = this->pBlockLength - mteTimes * HALF_ALIGNED_NUM;
    if (corePNums * this->N < this->pBlockLength) {
      mteTimes = corePNums * this->N / HALF_ALIGNED_NUM;
      leftP = corePNums * this->N - mteTimes * HALF_ALIGNED_NUM;
    }

    for (uint64_t i = 0; i <= mteTimes; i++) {
      if (i == mteTimes && leftP == 0) {
        break;
      }
      uint64_t mteLength = (i == mteTimes ? leftP * this->N : HALF_ALIGNED_NUM * this->N);
      DataCopyExtParams copyParams{1, static_cast<uint32_t>(mteLength * sizeof(half)), 0, 0, 0};
      DataCopyPadExtParams<half> padParams{true, 0, 0, 0};
      DataCopyPad(wsPNLocal[i * HALF_ALIGNED_NUM * this->N], this->repeatGm, copyParams, padParams);
      this->PIPE_MTE2_S();
    }
    // compare出mask
    uint64_t compareNums = this->AlignedToTarget(this->pBlockLength * this->N, COMPARE_ALIGNED);
    CompareScalar(this->compareMaskLocal, wsPNLocal, static_cast<half>(0), CMPMODE::EQ, compareNums);
    this->PIPE_V_S();
  }

protected:
   LocalTensor<uint8_t> compareMaskLocal;
   uint64_t pUbLength = 0;
   uint64_t pBlockLength = 0;
};

template <typename T, typename U>
__aicore__ void index_fill_tail_p(GM_ADDR x, GM_ADDR indices, GM_ADDR value, GM_ADDR y, GM_ADDR workspace,
                                  GM_ADDR tiling, const IndexFillTilingData* tilingData, TPipe* tPipe) {
    if (sizeof(T) == sizeof(half)) {
        IndexFillTailP<half, U> op;
        op.Init(x, indices, value, y, workspace, tiling, tilingData, tPipe);
        op.Process();
    } else if (sizeof(T) == sizeof(float)) {
        IndexFillTailP<float, U> op;
        op.Init(x, indices, value, y, workspace, tiling, tilingData, tPipe);
        op.Process();
    }
}
}
#endif