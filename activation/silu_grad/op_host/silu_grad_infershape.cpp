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
 * \file silu_grad_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"
#include "infershape_elewise_util.h"


using namespace ge;
namespace ops
{
static ge::graphStatus InferDataTypeForSiluGrad(gert::InferDataTypeContext *context)
{
    const ge::DataType dtypeDy = context->GetInputDataType(0);
    const ge::DataType dtypeX = context->GetInputDataType(1);
    ge::DataType dtypeDx = (dtypeDy == dtypeX) ? dtypeX : ge::DT_FLOAT;
    ge::graphStatus ret = context->SetOutputDataType(0, dtypeDx);
    return ret;
}

IMPL_OP_INFERSHAPE(SiluGrad).InferShape(Ops::Base::InferShape4Elewise)
                            .InferDataType(InferDataTypeForSiluGrad);
}  // namespace ops
