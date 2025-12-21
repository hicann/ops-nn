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
 * \file sparse_tensor_dense_mat_mul_tiling.h
 * \brief
 */
#ifndef __SPARSE_TENSOR_DENSE_MAT_MUL_TILING_H__
#define __SPARSE_TENSOR_DENSE_MAT_MUL_TILING_H__

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include "util/math_util.h"
#include "../../op_kernel/arch35/sparse_tensor_dense_mat_mul_tiling_def.h"

#define SPARSE_TENSOR_DENSE_MAT_MUL_TILING_ARCH35_PRIORITY 1000

namespace Ops::NN::Optiling {
struct SparseTensorDenseMatMulCompileInfo {};
} // namespace Ops::NN::Optiling

#endif // __SPARSE_TENSOR_DENSE_MAT_MUL_TILING_H__