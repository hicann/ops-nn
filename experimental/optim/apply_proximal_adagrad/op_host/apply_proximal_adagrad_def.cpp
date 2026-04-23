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
 * \file apply_proximal_adagrad_def.cpp
 * \brief ApplyProximalAdagrad operator definition
 *
 * Optimizer-style operator: 6 inputs, 2 inplace outputs.
 *
 * Inputs (in order):
 *   var   (Tensor, float32, ND, shape == accum == grad)   -- inplace updated
 *   accum (Tensor, float32, ND, shape == var   == grad)   -- inplace updated
 *   lr    (Tensor, float32, ND, 0-D or 1-element 1-D)     -- scalar
 *   l1    (Tensor, float32, ND, 0-D or 1-element 1-D)     -- scalar
 *   l2    (Tensor, float32, ND, 0-D or 1-element 1-D)     -- scalar
 *   grad  (Tensor, float32, ND, shape == var == accum)
 *
 * Outputs:
 *   var_out   (Tensor, float32, ND) -- shares storage with input var
 *   accum_out (Tensor, float32, ND) -- shares storage with input accum
 *
 * Target: Ascend950 (arch35 / DAV_3510) only.
 */
#include "register/op_def_registry.h"

namespace ops {
class ApplyProximalAdagrad : public OpDef {
public:
    explicit ApplyProximalAdagrad(const char* name) : OpDef(name)
    {
        // --- Inputs (6) ---------------------------------------------------
        this->Input("var")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("accum")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("lr")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("l1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("l2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        // --- Outputs (2, inplace via shared storage at L2 layer) ----------
        this->Output("var_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("accum_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        // --- Ascend950 (arch35) AI Core configuration ---------------------
        OpAICoreConfig aiCoreConfig;
        aiCoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("opFile.value", "apply_proximal_adagrad");
        this->AICore().AddConfig("ascend950", aiCoreConfig);
    }
};
OP_ADD(ApplyProximalAdagrad);
} // namespace ops
