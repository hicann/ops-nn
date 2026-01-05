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
 * \file elu_grad_infershape.cpp
 * \brief
 */
#include "infershape_broadcast_util.h"
#include "register/op_impl_registry.h"
using namespace Ops::Base;

namespace ops {
static ge::graphStatus InferShape4EluGrad(gert::InferShapeContext* context) {
  return InferShape4Broadcast(context);
}

IMPL_OP_INFERSHAPE(EluGrad).InferShape(InferShape4EluGrad);
}  // namespace ops