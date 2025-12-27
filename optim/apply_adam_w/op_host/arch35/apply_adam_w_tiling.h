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
 * \file apply_adam_w_tiling.h
 * \brief
 */
#ifndef RUNTIME_V2_OP_IMPL_APPLY_ADAM_W_TILING_H_
#define RUNTIME_V2_OP_IMPL_APPLY_ADAM_W_TILING_H_

#include "op_common/atvoss/elewise/elewise_tiling.h"
#include "register/tilingdata_base.h"
#include "../../op_kernel/arch35/apply_adam_w_tiling_key.h"
#include "../../op_kernel/arch35/apply_adam_w_tiling_struct.h"

using namespace ApplyAdamWNs;

namespace optiling {

struct ApplyAdamWCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

class ApplyAdamWTiling {
 public:
  explicit ApplyAdamWTiling(gert::TilingContext *context) : tilingContext_(context) {};

  ge::graphStatus RunTiling();
  ApplyAdamWTilingData* tiling_;

 protected:
  ge::graphStatus DoElewiseTiling();
  ge::graphStatus SetTilingData();
  ge::graphStatus CheckIsScalar(size_t inputIdx);
  ge::graphStatus CheckSameShape(size_t inputIdx, const gert::Shape& input0Shape);
  ge::graphStatus CheckSameDtype(size_t inputIdx, const ge::DataType& input0Dtype);
  ge::graphStatus CheckShapeAndType();

 private:
  gert::TilingContext *tilingContext_;
  uint64_t tilingKey = 0;
  bool amsgradAttr_ = false;
  bool maximizeAttr_ = false;
  uint64_t dType = 0;
  uint64_t schMode = 0;
  uint64_t amsgrad = 0;
};
}  // namespace optiling

#endif  // RUNTIME_V2_OP_IMPL_APPLY_ADAM_W_TILING_H_