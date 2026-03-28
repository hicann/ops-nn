/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * \file foreach_div_scalar_def.cpp
 * \brief ForeachDivScalar operator definition
 *
 * Inputs:
 *   - x (DYNAMIC_INPUT): TensorList of tensors to divide -- IR index 0
 *   - scalar (INPUT): Scalar tensor (single element) -- IR index 1
 * Outputs:
 *   - y (DYNAMIC_OUTPUT): TensorList of divided results
 */
#include "register/op_def_registry.h"

namespace ops {
class ForeachDivScalar : public OpDef {
public:
    explicit ForeachDivScalar(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("scalar")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_DOUBLE})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        OpAICoreConfig aicoreConfig910B;
        aicoreConfig910B.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "foreach_div_scalar");
        this->AICore().AddConfig("ascend910b", aicoreConfig910B);
    }
};
OP_ADD(ForeachDivScalar);
} // namespace ops
