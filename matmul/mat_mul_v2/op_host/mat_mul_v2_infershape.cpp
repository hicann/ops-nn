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
 * \file mat_mul_v2_infershape.cpp
 * \brief MatMulV2 infershape: 复用公共batch matmul推导(1D-8D/batch广播/bias N校验)
 */
#include "common/op_host/matmul_common_infershape.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t MATMULV2_BIAS_IDX = 2;
static ge::graphStatus InferShapeForMatMulV2(gert::InferShapeContext* context)
{
    return Ops::NN::InferShapeForBatchMatMul(context, 0, MATMULV2_BIAS_IDX);
}
} // namespace

IMPL_OP(MatMulV2).InferShape(InferShapeForMatMulV2);
} // namespace ops
