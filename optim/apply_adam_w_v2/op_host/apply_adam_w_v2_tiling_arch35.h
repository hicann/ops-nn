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
 * \file apply_adam_w_v2_tiling_arch35.h
 * \brief
 */
#ifndef RUNTIME_V2_OP_IMPL_APPLY_ADAM_W_V2_REGBASE_TILING_H_
#define RUNTIME_V2_OP_IMPL_APPLY_ADAM_W_V2_REGBASE_TILING_H_

#include "atvoss/elewise/elewise_tiling.h"
#include "register/tilingdata_base.h"
#include "../op_kernel/arch35/apply_adam_w_v2_dag.h"
#include "../op_kernel/arch35/apply_adam_w_v2_tiling_key.h"
#include "../op_kernel/arch35/apply_adam_w_v2_tiling_data.h"

namespace optiling {
using namespace Ops::Base;
class ApplyAdamWV2RegbaseTiling {
 public:
  explicit ApplyAdamWV2RegbaseTiling(gert::TilingContext *context) : tilingContext_(context) {};

  ge::graphStatus RunTiling();
  ApplyAdamWV2RegbaseTilingData *tiling_ = nullptr;

 protected:
  ge::graphStatus DoElewiseTiling();
  ge::graphStatus DoAmsGradTiling(ElewiseBaseTiling& eleBaseTiling, ge::DataType& varDType, ge::DataType& gradDType, ge::DataType& stepDType);
  ge::graphStatus DoNormTiling(ElewiseBaseTiling& eleBaseTiling, ge::DataType& varDType, ge::DataType& gradDType, ge::DataType& stepDType);
  ge::graphStatus SetTilingData();
  ge::graphStatus CheckScalarInput();
  ge::graphStatus CheckMixAndOptionalInput(const gert::Shape& inputStorageShape);
  ge::graphStatus CheckSameShape(size_t inputIdx, const gert::Shape& input0Shape);
  ge::graphStatus CheckSameDtype(size_t inputIdx, ge::DataType& input0Dtype);
  ge::graphStatus CheckShapeAndType();
  ge::graphStatus GetAttributes();
  void PrintInfo();

 private:
  gert::TilingContext *tilingContext_;
  uint64_t tilingKey = 0;
  uint64_t dType = 0;
  uint64_t schMode = 0;
  float lrAttr_ = 0.1;
  float beta1Attr_ = 0.1;
  float beta2Attr_ = 0.1;
  float weightDecayAttr_ = 0.1;
  float epsAttr_ = 1e-8;
  bool amsgradAttr_ = false;
  bool maximizeAttr_ = false;
  uint64_t amsgrad = 0;
};
}  // namespace optiling

#endif  // RUNTIME_V2_OP_IMPL_APPLY_ADAM_W_V2_REGBASE_TILING_H_