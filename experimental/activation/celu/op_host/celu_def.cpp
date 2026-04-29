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

/*!
 * \file celu_def.cpp
 * \brief Celu operator definition, declaring inputs/outputs and operator configuration
 *
 * Operator formula: y = alpha3 * 3 (x >= 0), y = alpha1 * (exp(x / alpha2) - 1) (x < 0)
 * Supported data types: float16, float
 * Target chip: Ascend950 (arch35)
 */

#include "register/op_def_registry.h"

namespace ops {
class Celu : public OpDef {
public:
    explicit Celu(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Attr("alpha1")
            .AttrType(REQUIRED)
            .Float(1.0f);
        this->Attr("alpha2")
            .AttrType(REQUIRED)
            .Float(1.0f);
        this->Attr("alpha3")
            .AttrType(REQUIRED)
            .Float(1.0f);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        OpAICoreConfig aiCoreConfig;
        aiCoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "celu");
        this->AICore().AddConfig("ascend950", aiCoreConfig);
    }
};
OP_ADD(Celu);
} // namespace ops
