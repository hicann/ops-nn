/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
constexpr int32_t DEFAULT_DST_TYPE = 36; // FLOAT8_E4M3FN
constexpr int32_t DEFAULT_SCALE_ALG = 1;
constexpr bool DEFAULT_DUAL_AXIS_FLAG = false;

// Supported input/output dtype combinations.
class ClaGateQuant : public OpDef {
public:
    explicit ClaGateQuant(const char* name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> inputTypes = {ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_FLOAT16,
                                                                ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_BF16,
                                                                ge::DT_FLOAT16, ge::DT_BF16};
        const std::initializer_list<ge::Format> formats = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                           ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
        // clang-format off
        this->Input("global_attn")
            .ParamType(REQUIRED)
            .DataType(inputTypes)
            .Format(formats)
            .AutoContiguous();
        this->Input("local_attn")
            .ParamType(REQUIRED)
            .DataType(inputTypes)
            .Format(formats)
            .AutoContiguous();
        this->Input("global_gate_logits")
            .ParamType(REQUIRED)
            .DataType(inputTypes)
            .Format(formats)
            .AutoContiguous();
        this->Input("local_gate_logits")
            .ParamType(REQUIRED)
            .DataType(inputTypes)
            .Format(formats)
            .AutoContiguous();
        const std::initializer_list<ge::DataType> dataTypes = {
            ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E1M2, ge::DT_FLOAT4_E1M2,
            ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2};
        const std::initializer_list<ge::DataType> scaleTypes = {
            ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
            ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0};
        this->Output("row_data")
            .ParamType(REQUIRED)
            .DataType(dataTypes)
            .Format(formats);
        this->Output("row_scale")
            .ParamType(REQUIRED)
            .DataType(scaleTypes)
            .Format(formats);
        this->Output("col_data")
            .ParamType(REQUIRED)
            .DataType(dataTypes)
            .Format(formats);
        this->Output("col_scale")
            .ParamType(REQUIRED)
            .DataType(scaleTypes)
            .Format(formats);
        // Attr order must stay in sync with the tiling / aclnn implementations.
        this->Attr("dst_type").AttrType(OPTIONAL).Int(DEFAULT_DST_TYPE);
        this->Attr("round_mode").AttrType(OPTIONAL).String("rint");
        this->Attr("scale_alg").AttrType(OPTIONAL).Int(DEFAULT_SCALE_ALG);
        this->Attr("input_attn_layout").AttrType(OPTIONAL).String("TND");
        this->Attr("dual_axis_flag").AttrType(OPTIONAL).Bool(DEFAULT_DUAL_AXIS_FLAG);
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "cla_gate_quant_apt");
        this->AICore().AddConfig("ascend950", config);
        // clang-format on
    }
};
OP_ADD(ClaGateQuant);
} // namespace ops
