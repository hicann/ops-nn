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
 * \file apply_adamax_def.cpp
 * \brief ApplyAdamax operator definition
 *
 * Inputs:  var, m, v, grad (Tensor) + beta1Power, lr, beta1, beta2, epsilon (Scalar via Attr)
 * Outputs: varOut, mOut, vOut (Tensor, inplace with var/m/v)
 * Target:  ascend950 (arch35)
 */
#include "register/op_def_registry.h"

namespace ops {

class ApplyAdamax : public OpDef {
public:
    explicit ApplyAdamax(const char* name) : OpDef(name)
    {
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

        // 4 Tensor inputs: var, m, v, grad
        cfgInput("var");
        cfgInput("m");
        cfgInput("v");
        cfgInput("grad");

        // 3 Tensor outputs: varOut, mOut, vOut (inplace)
        cfgOutput("varOut");
        cfgOutput("mOut");
        cfgOutput("vOut");

        // Scalar attrs (5): beta1Power, lr, beta1, beta2, epsilon
        // Indices in Tiling attrs: beta1Power=0, lr=1, beta1=2, beta2=3, epsilon=4
        this->Attr("beta1Power").AttrType(REQUIRED).Float();
        this->Attr("lr").AttrType(REQUIRED).Float();
        this->Attr("beta1").AttrType(REQUIRED).Float();
        this->Attr("beta2").AttrType(REQUIRED).Float();
        this->Attr("epsilon").AttrType(REQUIRED).Float();

        // ascend950 (arch35) configuration
        OpAICoreConfig aiCoreConfig;
        aiCoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "apply_adamax");
        this->AICore().AddConfig("ascend950", aiCoreConfig);
    }
};

OP_ADD(ApplyAdamax);

} // namespace ops
