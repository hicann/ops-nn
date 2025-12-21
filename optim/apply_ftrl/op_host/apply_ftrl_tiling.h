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
 * \file apply_ftrl_tiling.h
 * \brief
 */
#ifndef RUNTIME_V2_OP_IMPL_APPLY_FTRL_REGBASE_TILING_H_
#define RUNTIME_V2_OP_IMPL_APPLY_FTRL_REGBASE_TILING_H_

#include "atvoss/elewise/elewise_tiling.h"
#include "register/tilingdata_base.h"
#include "../op_kernel/arch35/apply_ftrl_dag.h"
#include "../op_kernel/arch35/apply_ftrl_tiling_key.h"
#include "../op_kernel/arch35/apply_ftrl_tiling_data.h"

namespace optiling {
using namespace Ops::Base;
using namespace ApplyFtrlOpTiling;
class ApplyFtrlRegbaseTiling {
 public:
  explicit ApplyFtrlRegbaseTiling(gert::TilingContext *context) : tilingContext_(context) {};

  ge::graphStatus RunTiling();
  ApplyFtrlRegbaseTilingData *tiling_ = nullptr;

 protected:
  ge::graphStatus DoElewiseTiling();
  ge::graphStatus CheckScalarShape(int32_t inputIdx);
  ge::graphStatus CheckSameShape(int32_t inputIdx, const gert::Shape& input0Shape);
  ge::graphStatus CheckSameDtype(int32_t inputIdx, const ge::DataType& input0Dtype);
  ge::graphStatus CheckShapeAndType();
  ge::graphStatus SetTilingData();

 private:
  gert::TilingContext *tilingContext_;
  uint64_t tilingKey = 0;
};
}  // namespace optiling

#endif  // RUNTIME_V2_OP_IMPL_APPLY_FTRL_REGBASE_TILING_H_