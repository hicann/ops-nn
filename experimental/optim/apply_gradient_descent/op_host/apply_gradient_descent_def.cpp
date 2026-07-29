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
 * \file apply_gradient_descent_def.cpp
 * \brief ApplyGradientDescent operator definition (experimental, ascend910b classic path).
 *
 * Inputs:  var (Ref/inplace), alpha (scalar), delta -- all same dtype (bf16/fp16/fp32), ND.
 * Output:  var (inplace with input var): var = var - alpha * delta.
 */

#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> dataType = {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT};

static const std::vector<ge::Format> dataFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class ApplyGradientDescent : public OpDef {
public:
    explicit ApplyGradientDescent(const char* name) : OpDef(name)
    {
        this->Input("var").ParamType(REQUIRED).DataType(dataType).Format(dataFormat).AutoContiguous();
        this->Input("alpha").ParamType(REQUIRED).DataType(dataType).Format(dataFormat).AutoContiguous();
        this->Input("delta").ParamType(REQUIRED).DataType(dataType).Format(dataFormat).AutoContiguous();
        this->Output("var").ParamType(REQUIRED).DataType(dataType).Format(dataFormat).AutoContiguous();

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("opFile.value", "apply_gradient_descent");
        this->AICore().AddConfig("ascend910b", aicoreConfig);
    }
};

OP_ADD(ApplyGradientDescent);
} // namespace ops
