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
 * \file index_fill_tail_n.h
 * \brief tail axis and N >= core num * 256. core will split N.
 */
#ifndef INDEX_FILL_TAIL_N_H
#define INDEX_FILL_TAIL_N_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "index_fill_base.h"

namespace IndexFillNS {
using namespace AscendC;
constexpr uint32_t N_UB_LENGTH = 12288;

template <typename T, typename U>
class IndexFillTailN : public IndexFillBase<T, U> {
public:
  __aicore__ inline void Process() {
    this->ExecIndicesTask();
    this->ReadValue();
    TailAxisN();
  }

  __aicore__ inline void TailAxisN() {
    // 切N
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
    } else if(tailCoreNums > 0){
        coreNStart = frontCoreNums * frontCoreNNums + (GetBlockIdx() - frontCoreNums) * tailCoreNNums;
        coreNNums = tailCoreNNums;
    }

    uint64_t NBlocks = coreNNums / N_UB_LENGTH;
    uint64_t leftN = coreNNums - NBlocks * N_UB_LENGTH;
    for (uint64_t i = 0; i <= NBlocks; i++) {
        if (i == NBlocks && leftN == 0) {
            break;
        }
        uint64_t NNums = (i == NBlocks ? leftN : N_UB_LENGTH);
        InitSelect(coreNStart + i * N_UB_LENGTH, NNums);
        for (uint64_t j = 0; j < this->P; j++) {
            ProcessPN(j, coreNStart + i * N_UB_LENGTH, NNums);
        }
    }
  }

  __aicore__ inline void InitSelect(uint64_t nId, uint64_t nNums) {
    // 搬入一段标记Tensor
    // ub 布局：compareMaskLocal, wsLocal, xLocal
    uint64_t wsLocalUbStart = this->AlignedToTarget(N_UB_LENGTH, COMPARE_ALIGNED);
    this->wsLocal = this->allUbLocal[wsLocalUbStart].template ReinterpretCast<half>();
    this->CopyInWsGm(nId, this->wsLocal, nNums);
    this->PIPE_MTE2_S();
    // 标记Tensor转compareMaskLocal
    this->compareMaskLocal = this->allUbLocal;
    uint32_t compareNums = this->AlignedToTarget(nNums, COMPARE_ALIGNED);
    CompareScalar(this->compareMaskLocal, this->wsLocal, static_cast<half>(0), CMPMODE::EQ, compareNums);
    this->PIPE_V_S();
  }

  __aicore__ inline void ProcessPN(uint64_t pId, uint64_t nId, uint64_t nNums) {
    //搬入x, select value, 搬出x即可
    uint64_t xUbStart = this->AlignedToTarget(N_UB_LENGTH, COMPARE_ALIGNED);
    uint64_t gmStartInt8 = (pId * this->N + nId) * sizeof(T);
    this->CopyIn(this->xGm, gmStartInt8, xUbStart, nNums * sizeof(T));
    this->PIPE_MTE2_S();
    this->xLocal = this->allUbLocal[xUbStart].template ReinterpretCast<T>();
    Select(this->xLocal, this->compareMaskLocal, this->xLocal, this->value, SELMODE::VSEL_TENSOR_SCALAR_MODE, nNums);
    this->PIPE_V_S();
    // 搬出xLocal
    this->CopyOut(this->yGm, gmStartInt8, xUbStart, nNums * sizeof(T));
    this->PIPE_MTE3_S();
  }

protected:
   LocalTensor<uint8_t> compareMaskLocal;
};

template <typename T, typename U>
__aicore__ void index_fill_tail_n(GM_ADDR x, GM_ADDR indices, GM_ADDR value, GM_ADDR y, GM_ADDR workspace,
                                  GM_ADDR tiling, const IndexFillTilingData* tilingData, TPipe* tPipe) {
    if (sizeof(T) == 2) {
        IndexFillTailN<half, U> op;
        op.Init(x, indices, value, y, workspace, tiling, tilingData, tPipe);
        op.Process();
    } else if (sizeof(T) == 4) {
        IndexFillTailN<float, U> op;
        op.Init(x, indices, value, y, workspace, tiling, tilingData, tPipe);
        op.Process();
    }
}
}
#endif