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
 * \file index_fill_front_p.h
 * \brief front axis and Q < 64 && N < 64. core will split P.
 */
#ifndef INDEX_FILL_FRONT_P_H
#define INDEX_FILL_FRONT_P_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "index_fill_base.h"

namespace IndexFillNS {
using namespace AscendC;
constexpr uint32_t P_UB_LENGTH = 8192;

template <typename T, typename U>
class IndexFillFrontP : public IndexFillBase<T, U> {
public:
  __aicore__ inline void Process() {
    this->ExecIndicesTask();
    this->ReadValue();
    FrontAxisP();
  }

  __aicore__ inline void FrontAxisP() {
    // 全部核均分P
    uint64_t corePStart = 0;
    uint64_t corePNums = 0;
    this->SliceP(corePStart, corePNums);
    if (corePNums > 0) {
        // 计算一次UB一次可以容纳多少个P
        this->ubPBlockLength = P_UB_LENGTH / (this->N * this->Q);
        if (this->N * this->Q < COMPARE_ALIGNED) {
            this->ubPBlockLength = COMPARE_ALIGNED; // 防止N, Q太小，InitSelect()耗时占比高
        }
        InitSelect();

        uint64_t pBlocks = corePNums / this->ubPBlockLength;                                          
        uint64_t leftP = corePNums - pBlocks * this->ubPBlockLength;
        for (uint64_t i = 0; i <= pBlocks; i++) {
            if (i == pBlocks && leftP == 0) {
                break;
            }
            uint64_t pNums = (i == pBlocks ? leftP : this->ubPBlockLength);
            uint64_t gmStartInt8 = (corePStart * this->N * this->Q + i * this->ubPBlockLength * this->N * this->Q) * sizeof(T);
            uint64_t ubStartInt8 = this->AlignedToTarget(this->ubPBlockLength * this->N * this->Q, INT8_ALIGNED_NUM);
            ubStartInt8 = (ubStartInt8 == 0 ? COMPARE_ALIGNED : ubStartInt8);
            uint64_t mteNumsInt8 = pNums * this->N * this->Q * sizeof(T);
            this->CopyIn(this->xGm, gmStartInt8, ubStartInt8, mteNumsInt8);
            this->PIPE_MTE2_S();
            // 替换为value
            this->xLocal = this->allUbLocal[ubStartInt8].template ReinterpretCast<T>();
            Select(this->xLocal, this->compareMaskLocal, this->xLocal, this->value, SELMODE::VSEL_TENSOR_SCALAR_MODE,
                pNums * this->N * this->Q);
            this->PIPE_V_S();
            // 搬出xLocal
            this->CopyOut(this->yGm, gmStartInt8, ubStartInt8, mteNumsInt8);
            this->PIPE_MTE3_S();
        }
    }
  }

  __aicore__ inline void InitSelect() {
    // UB布局：compareMaskLocal，repeatMarkerLocal，markerLocal，tempLocal
    uint64_t repeatMarkerLocalStart = this->AlignedToTarget(this->ubPBlockLength * this->N * this->Q, COMPARE_ALIGNED);
    repeatMarkerLocalStart = (repeatMarkerLocalStart == 0 ? COMPARE_ALIGNED : repeatMarkerLocalStart);
    uint64_t markerLocalStart = repeatMarkerLocalStart + this->AlignedToTarget(this->ubPBlockLength * this->N * this->Q * sizeof(half),
                                INT8_ALIGNED_NUM);
    uint64_t tempLocalStart = markerLocalStart + this->AlignedToTarget(this->N * sizeof(half), INT8_ALIGNED_NUM);
    this->compareMaskLocal = this->allUbLocal[0].template ReinterpretCast<uint8_t>();
    auto repeatMarkerLocal = this->allUbLocal[repeatMarkerLocalStart].template ReinterpretCast<half>();
    auto markerLocal = this->allUbLocal[markerLocalStart].template ReinterpretCast<half>();
    auto tempLocal = this->allUbLocal[tempLocalStart].template ReinterpretCast<half>();
    // 搬入标记Tensor
    this->CopyInWsGm(0, markerLocal, this->N); // wsGm: 标记Tensor+同步Tensor+复制Tensor
    this->PIPE_MTE2_S();
    // 每个标记复制Q次后搬出
    for (uint64_t i = 0; i < this->N; i++) {
      half marker = markerLocal.GetValue(i);
      Duplicate(tempLocal, marker, this->Q);
      this->PIPE_V_S();
      DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->Q * sizeof(half)), 0, 0, 0};
      DataCopyPad(this->repeatGm[i * this->Q], tempLocal, copyParams);
      this->PIPE_MTE3_S();
    }
    // 搬入复制后的标记Tensor
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->N * this->Q * sizeof(half)), 0, 0, 0};
    DataCopyPadExtParams<half> padParams{true, 0, 0, 0};
    DataCopyPad(repeatMarkerLocal, this->repeatGm, copyParams, padParams);
    this->PIPE_MTE2_S();
    // 复制后的标记tensor再搬出ubPBlockLength次（即复制ubPBlockLength次）
    for (uint64_t i = 0; i < this->ubPBlockLength; i++) {
      DataCopyPad(this->repeatGm[i * this->N * this->Q], repeatMarkerLocal, copyParams);
    }
    this->PIPE_MTE3_S();
    // 搬入完全复制完成的标记Tensor
    DataCopyExtParams copyParamsPNQ{1, static_cast<uint32_t>(this->ubPBlockLength * this->N * this->Q * sizeof(half)),
                                    0, 0, 0};
    DataCopyPad(repeatMarkerLocal, this->repeatGm, copyParamsPNQ, padParams);
    this->PIPE_MTE2_S();
    // half Tensor转mask Tensor
    uint32_t compareNums = this->AlignedToTarget(this->ubPBlockLength * this->N * this->Q, COMPARE_ALIGNED);
    CompareScalar(this->compareMaskLocal, repeatMarkerLocal, static_cast<half>(0), CMPMODE::EQ, compareNums);
    this->PIPE_V_S();
  }

protected:
   LocalTensor<uint8_t> compareMaskLocal;
   uint64_t ubPBlockLength = 0;
};

template <typename T, typename U>
__aicore__ void index_fill_front_p(GM_ADDR x, GM_ADDR indices, GM_ADDR value, GM_ADDR y, GM_ADDR workspace,
                                   GM_ADDR tiling, const IndexFillTilingData* tilingData, TPipe* tPipe) {
    if (sizeof(T) == sizeof(half)) {
        IndexFillFrontP<half, U> op;
        op.Init(x, indices, value, y, workspace, tiling, tilingData, tPipe);
        op.Process();
    } else if (sizeof(T) == sizeof(float)) {
        IndexFillFrontP<float, U> op;
        op.Init(x, indices, value, y, workspace, tiling, tilingData, tPipe);
        op.Process();
    }
}
}
#endif