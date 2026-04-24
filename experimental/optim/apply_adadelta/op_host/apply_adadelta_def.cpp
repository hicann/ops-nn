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
 * \file apply_adadelta_def.cpp
 * \brief ApplyAdadelta operator definition
 *
 * Inputs:  var, accum, accumUpdate, grad (Tensor) + lr, rho, epsilon (Scalar via TilingData)
 * Outputs: varOut, accumOut, accumUpdateOut (Tensor, inplace with inputs)
 * Target:  ascend950 (arch35)
 */
#include "register/op_def_registry.h"

namespace ops {

class ApplyAdadelta : public OpDef {
public:
    explicit ApplyAdadelta(const char* name) : OpDef(name)
    {
        // Helper: configure a Tensor Input with standard (FP32/FP16, ND) schema.
        auto cfgInput = [this](const char* n) -> void {
            this->Input(n)
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
                .AutoContiguous();
        };
        auto cfgOutput = [this](const char* n) -> void {
            this->Output(n)
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
                .AutoContiguous();
        };

        // 4 Tensor inputs: var, accum, accumUpdate, grad
        cfgInput("var");
        cfgInput("accum");
        cfgInput("accumUpdate");
        cfgInput("grad");

        // 3 Tensor outputs: varOut, accumOut, accumUpdateOut (inplace)
        cfgOutput("varOut");
        cfgOutput("accumOut");
        cfgOutput("accumUpdateOut");

        // Scalar attrs: lr, rho, epsilon (set by ACLNN L0 API, accessed by index in Tiling)
        this->Attr("lr").AttrType(REQUIRED).Float();
        this->Attr("rho").AttrType(REQUIRED).Float();
        this->Attr("epsilon").AttrType(REQUIRED).Float();

        // ascend950 (arch35) configuration
        OpAICoreConfig aiCoreConfig;
        aiCoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "apply_adadelta");
        this->AICore().AddConfig("ascend950", aiCoreConfig);
    }
};

OP_ADD(ApplyAdadelta);

} // namespace ops
