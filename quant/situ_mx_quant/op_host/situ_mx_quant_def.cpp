/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file situ_mx_quant_def.cpp
 * \brief Situ activation combined with dynamic MX quantization operator definition
 */

#include <register/op_def_registry.h>

namespace ops {

class SituMxQuant : public OpDef {
public:
    explicit SituMxQuant(const char* name) : OpDef(name)
    {
        // dtype 组合（8种）：
        // 输入 x: FLOAT16, BFLOAT16
        // 输出 y: FP4_E2M1, FP4_E1M2, FP8_E4M3FN, FP8_E5M2
        // 输出 y_scale: FLOAT8_E8M0
        // 组合方式：
        // x(FLOAT16) → y的4种 = 4个
        // x(BFLOAT16) → y的4种 = 4个
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({// x=FLOAT16, y的4种
                       ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                       // x=BFLOAT16, y的4种
                       ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({// x=FLOAT16, y的4种
                       ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                       // x=BFLOAT16, y的4种
                       ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("y_scale")
            .ParamType(REQUIRED)
            .DataType({// 对应8种组合，y_scale都是FLOAT8_E8M0
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                       ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("beta").AttrType(OPTIONAL).Float(1.0f);
        this->Attr("linear_beta").AttrType(OPTIONAL).Float(0.0f);
        this->Attr("activate_left").AttrType(OPTIONAL).Bool(false);
        this->Attr("axis").AttrType(OPTIONAL).Int(-1);
        this->Attr("dst_type").AttrType(OPTIONAL).Int(40);
        this->Attr("round_mode").AttrType(OPTIONAL).String("rint");

        // Ascend 950 (arch35) configuration using Regbase
        OpAICoreConfig regbaseCfg;
        regbaseCfg.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "situ_mx_quant_apt");
        this->AICore().AddConfig("ascend950", regbaseCfg);
    }
};

OP_ADD(SituMxQuant);

} // namespace ops
