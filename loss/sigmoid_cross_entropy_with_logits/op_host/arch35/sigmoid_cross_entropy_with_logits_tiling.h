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
 * \file sigmoid_cross_entropy_with_logits_tiling.h
 * \brief SigmoidCrossEntropyWithLogits 算子 Tiling 头文件
 */
#ifndef SIGMOID_CROSS_ENTROPY_WITH_LOGITS_TILING_H_
#define SIGMOID_CROSS_ENTROPY_WITH_LOGITS_TILING_H_

#include "atvoss/elewise/elewise_tiling.h"
#include "register/tilingdata_base.h"
#include "../../op_kernel/arch35/sigmoid_cross_entropy_with_logits_dag.h"
#include "../../op_kernel/arch35/sigmoid_cross_entropy_with_logits_struct.h"

namespace optiling {
using namespace Ops::Base;

struct SigmoidCrossEntropyWithLogitsCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

class SigmoidCrossEntropyWithLogitsTiling {
 public:
  explicit SigmoidCrossEntropyWithLogitsTiling(gert::TilingContext *context) : tilingContext_(context) {};

  ge::graphStatus RunTiling();

 protected:
  ge::graphStatus DoElewiseTiling();
  ge::graphStatus CalcInputDtype();
  ge::graphStatus CalcOutputDtype();
  ge::graphStatus CheckShape();
  ge::graphStatus CheckSameShape(int32_t inputIdx, const gert::Shape& input0Shape);

 private:
  gert::TilingContext *tilingContext_;
  ge::DataType inputDtype_ = ge::DT_UNDEFINED;
  ge::DataType outputDtype_ = ge::DT_UNDEFINED;
  uint64_t dType_ = 0;
  uint64_t tilingKey_ = 0;
};
}  // namespace optiling

#endif  // SIGMOID_CROSS_ENTROPY_WITH_LOGITS_TILING_H_
