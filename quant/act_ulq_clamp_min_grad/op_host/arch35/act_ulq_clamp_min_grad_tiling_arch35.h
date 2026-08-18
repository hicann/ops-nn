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
 * \file act_ulq_clamp_min_grad_tiling_arch35.h
 * \brief ActULQClampMinGrad host tiling 公共声明（arch35）。
 */
#ifndef OPS_ACT_ULQ_CLAMP_MIN_GRAD_TILING_ARCH35_H_
#define OPS_ACT_ULQ_CLAMP_MIN_GRAD_TILING_ARCH35_H_

#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"

namespace optiling {

// 入图场景依赖：必须定义 CompileInfo 结构体
struct ActULQClampMinGradCompileInfo {};

} // namespace optiling

#endif // OPS_ACT_ULQ_CLAMP_MIN_GRAD_TILING_ARCH35_H_
