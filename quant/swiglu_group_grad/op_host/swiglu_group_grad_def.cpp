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
 * \file swiglu_group_grad_def.cpp
 * \brief SwigluGroupGrad operator definition
 *
 * IR definition:
 *   Required inputs:  grad_y (T,H)/(B,S,H) [BF16/FP16/FP32], x (T,2H)/(B,S,2H) [BF16/FP16/FP32]
 *   Optional inputs:  weight (T,1)/(B,S,1) [FP32] and y_origin (T,H)/(B,S,H) [BF16/FP16/FP32]
 *                     must be provided together,
 *                     group_index (G,) [INT64]
 *   Attribute:        clamp_limit (Float, default=0 → no clamp)
 *   Required output:  grad_x (T,2H)/(B,S,2H) [BF16/FP16/FP32]
 *   Optional output:  grad_weight (T,1)/(B,S,1) [FP32]
 *   Platform:         Ascend950 only (arch35/DAV_3510, RegBase or SIMT path)
 */
#include <register/op_def_registry.h>

namespace ops {

class SwigluGroupGrad : public OpDef {
public:
    explicit SwigluGroupGrad(const char* name) : OpDef(name)
    {
        // ── Required inputs ──────────────────────────────────────────────────
        this->Input("grad_y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // ── Optional inputs ──────────────────────────────────────────────────
        this->Input("weight")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("y_origin")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("group_index")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // ── Required output ──────────────────────────────────────────────────
        this->Output("grad_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // ── Optional output ──────────────────────────────────────────────────
        this->Output("grad_weight")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        // ── Attribute ────────────────────────────────────────────────────────
        this->Attr("clamp_limit").AttrType(OPTIONAL).Float(0);

        // ── Platform config: only Ascend950 (RegBase path) ───────────────────
        OpAICoreConfig regbaseCfg;
        regbaseCfg.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend950", regbaseCfg);
    }
};
OP_ADD(SwigluGroupGrad);
} // namespace ops
