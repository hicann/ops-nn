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
 * \file l1_loss_grad_tiling.h
 * \brief l1_loss_grad head file
 */
#ifndef L1_LOSS_GRAD_TILING_H_
#define L1_LOSS_GRAD_TILING_H_
#include <string>
#include <vector>

namespace optiling {
struct L1LossGradCompileInfo
{
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
    ge::DataType dtype;
};
}  // namespace optiling
#endif  // L1_LOSS_GRAD_H_
