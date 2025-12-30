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
 * \file kl_div_loss_grad_tiling_arch35.h
 * \brief
 */
#ifndef KL_DIV_LOSS_GRAD_TILING_ARCH35_h
#define KL_DIV_LOSS_GRAD_TILING_ARCH35_h
#include <cstdint>
#include "atvoss/broadcast/broadcast_tiling.h"
#include "register/op_impl_registry.h"

namespace optiling {
struct KlDivLossGradCompileInfo {
  uint64_t coreNum = 0;
  uint64_t ubSize = 0;
  ge::DataType reduce_mean_cof_dtype;
}; // namespace optiling

}
#endif  // KL_DIV_LOSS_GRAD_TILING_ARCH35_h
