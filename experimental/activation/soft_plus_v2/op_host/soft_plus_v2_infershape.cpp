/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jiamin <@zhou-jiamin-666>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */
/*!
 * \file soft_plus_v2_infer.cpp
 * \brief/
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
static ge::graphStatus InferShape(gert::InferShapeContext *context) {
  const gert::Shape *x_shape = context->GetInputShape(0);
  gert::Shape *z_shape = context->GetOutputShape(0);
  *z_shape = *x_shape;
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SoftPlusV2).InferShape(InferShape);
} // namespace ops