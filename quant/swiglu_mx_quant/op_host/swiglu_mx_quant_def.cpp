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
 * \file swiglu_mx_quant_def.cpp
 * \brief SwiGLU activation combined with dynamic MX quantization operator definition
 */

#include <register/op_def_registry.h>

namespace ops {

class SwigluMxQuant : public OpDef {
public:
    explicit SwigluMxQuant(const char* name) : OpDef(name)
    {
        // 算子支持以下 dtype 和 format 组合（16种）：
        // 输入 x: FLOAT16, BFLOAT16
        // 输入 group: INT32, INT64（可选）
        // 输出 y: FP4_E2M1, FP4_E1M2, FP8_E4M3FN, FP8_E5M2
        // 输出 mxscale: FLOAT8_E8M0
        // 所有输入输出支持 Format: ND
        //
        // 组合方式：
        // x(FLOAT16) + group(INT32) → y的4种 = 4个
        // x(FLOAT16) + group(INT64) → y的4种 = 4个
        // x(BFLOAT16) + group(INT32) → y的4种 = 4个
        // x(BFLOAT16) + group(INT64) → y的4种 = 4个
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({
                // x=FLOAT16, group_index=INT32, y的4种
                ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                // x=FLOAT16, group_index=INT64, y的4种
                ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
                // x=BFLOAT16, group_index=INT32, y的4种
                ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16,
                // x=BFLOAT16, group_index=INT64, y的4种
                ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();

        this->Input("group_index")
            .ParamType(OPTIONAL)
            .DataType({
                // 对应 x=FLOAT16 的8种组合
                ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
                // 对应 x=BFLOAT16 的8种组合
                ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
            })
            .AutoContiguous();

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({
                // x=FLOAT16, group=INT32, y的4种
                ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                // x=FLOAT16, group=INT64, y的4种
                ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                // x=BFLOAT16, group=INT32, y的4种
                ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
                // x=BFLOAT16, group=INT64, y的4种
                ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
            });

        this->Output("mxscale")
            .ParamType(REQUIRED)
            .DataType({
                // 对应16种组合，mxscale都是FLOAT8_E8M0
                ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
                ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0
            })
            .Format({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
            });

        this->Attr("activate_dim").AttrType(OPTIONAL).Int(-1);
        this->Attr("activate_left").AttrType(OPTIONAL).Bool(false);
        this->Attr("swiglu_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("clamp_limit").AttrType(OPTIONAL).Float(7.0f);
        this->Attr("glu_alpha").AttrType(OPTIONAL).Float(1.702f);
        this->Attr("glu_bias").AttrType(OPTIONAL).Float(1.0f);
        this->Attr("group_mode").AttrType(OPTIONAL).Int(0);
        this->Attr("axis").AttrType(OPTIONAL).Int(-1);
        this->Attr("dst_type").AttrType(OPTIONAL).Int(40);
        this->Attr("round_mode").AttrType(OPTIONAL).String("rint");
        this->Attr("scale_alg").AttrType(OPTIONAL).Int(0);
        this->Attr("max_dtype_value").AttrType(OPTIONAL).Float(0.0f);

        // Ascend 950 (arch35) configuration using Regbase
        OpAICoreConfig regbaseCfg;
        regbaseCfg.DynamicCompileStaticFlag(true)
                .DynamicRankSupportFlag(true)
                .DynamicShapeSupportFlag(true)
                .ExtendCfgInfo("opFile.value", "swiglu_mx_quant_apt");
        this->AICore().AddConfig("ascend950", regbaseCfg);
    }
};

OP_ADD(SwigluMxQuant);

} // namespace ops
