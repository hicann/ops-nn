/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file smooth_l1_loss_grad_v2_tiling_base.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_SMOOTH_L1_LOSS_GRAD_V2_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_SMOOTH_L1_LOSS_GRAD_V2_H

#include <string>
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "log/log.h"

namespace optiling {
struct SmoothL1LossGradV2CompileInfo {
  std::string reduceMeanCofDtype;
  ge::DataType reduceDtype;
  uint64_t coreNum = 0;
  uint64_t ubSize = 0;
  ge::DataType dtype;
};
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_SMOOTH_L1_LOSS_GRAD_V2_H