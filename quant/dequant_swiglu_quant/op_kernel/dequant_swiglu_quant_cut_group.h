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
 * \file dequant_swiglu_quant_cut_group.h
 * \brief
 */

#ifndef DEQUANT_SWIGLU_QUANT_CUT_GROUP_H
#define DEQUANT_SWIGLU_QUANT_CUT_GROUP_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "dequant_swiglu_quant.h"

namespace DequantSwigluQuantGroupOps {
using namespace AscendC;
constexpr static int64_t GROUPINDEX_STRIDE = 2;

TEMPLATE_DSQ_DECLARE
class DequantSwigluQuantGroup : public DequantSwigluQuantOps::DequantSwigluQuantBase<TEMPLATE_DSQ_ARGS> {
 public:
 __aicore__ inline DequantSwigluQuantGroup(TPipe* pipe) : DequantSwigluQuantOps::DequantSwigluQuantBase<TEMPLATE_DSQ_ARGS>(pipe)
  {
    this->pipe_ = pipe;
  };
  __aicore__ inline void Process();
};
// 公共函数实现

TEMPLATE_DSQ_DECLARE
__aicore__ inline void DequantSwigluQuantGroup<TEMPLATE_DSQ_ARGS>::Process() {
  this->groupOffset_ = 0;
  int64_t cuGroupIdx = this->blockIdx_;
  for (int32_t groupIdx = 0; groupIdx < this->tl_->inGroupNum; ++groupIdx) {
    int64_t realGroupIdx = static_cast<int64_t>(this->groupIndexGm_(groupIdx*GROUPINDEX_STRIDE));
    this->realDimx_ = static_cast<int64_t>(this->groupIndexGm_(groupIdx*GROUPINDEX_STRIDE + 1));
    if (this->realDimx_ <= 0) {
      break;
    }
    if (groupIdx == cuGroupIdx) {
      this->ProcessSingleGroupPerCore(realGroupIdx, this->realDimx_, this->groupOffset_);
      cuGroupIdx += this->tl_->maxCoreNum;
    }
    this->groupOffset_ += this->realDimx_;
  }
}

}  // namespace DequantSwigluQuantGroupOps
#endif  // DEQUANT_SWIGLU_QUANT_CUT_GROUP_H
