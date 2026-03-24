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
 * \file add_rms_norm_dynamic_mx_quant_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {

// x1/x2 dtype: 16 combinations (4 output types × 2 input types × 2 gamma types)
static const std::vector<ge::DataType> xDataType = {
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16
};

// gamma/beta dtype: first 8 match x dtype, last 8 are FP32
static const std::vector<ge::DataType> gammaDataType = {
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,    ge::DT_BF16,
    ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
    ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT
};

// y output dtype: E2M1, E1M2, E4M3FN, E5M2 cycling
static const std::vector<ge::DataType> yDataType = {
    ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E1M2,   ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
    ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E1M2,   ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
    ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E1M2,   ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,
    ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E1M2,   ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2
};

// mxscale output dtype: all FP8_E8M0
static const std::vector<ge::DataType> mxscaleDataType = {
    ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
    ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
    ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
    ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0
};

// rstd output dtype: all FP32
static const std::vector<ge::DataType> rstdDataType = {
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT
};

// format: all ND
static const std::vector<ge::Format> formatND = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
};

class AddRmsNormDynamicMxQuant : public OpDef {
public:
    explicit AddRmsNormDynamicMxQuant(const char* name) : OpDef(name)
    {
        // Inputs
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType(xDataType)
            .Format(formatND)
            .UnknownShapeFormat(formatND)
            .AutoContiguous();
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType(xDataType)
            .Format(formatND)
            .UnknownShapeFormat(formatND)
            .AutoContiguous();
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType(gammaDataType)
            .Format(formatND)
            .UnknownShapeFormat(formatND)
            .AutoContiguous();
        this->Input("beta")
            .ParamType(OPTIONAL)
            .DataType(gammaDataType)
            .Format(formatND)
            .UnknownShapeFormat(formatND)
            .AutoContiguous();

        // Outputs
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(yDataType)
            .Format(formatND)
            .UnknownShapeFormat(formatND)
            .AutoContiguous();
        this->Output("x")
            .ParamType(REQUIRED)
            .DataType(xDataType)
            .Format(formatND)
            .UnknownShapeFormat(formatND)
            .AutoContiguous();
        this->Output("mxscale")
            .ParamType(REQUIRED)
            .DataType(mxscaleDataType)
            .Format(formatND)
            .UnknownShapeFormat(formatND)
            .AutoContiguous();
        this->Output("rstd")
            .ParamType(REQUIRED)
            .DataType(rstdDataType)
            .Format(formatND)
            .UnknownShapeFormat(formatND)
            .AutoContiguous();

        // Attributes
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-6);
        this->Attr("scale_alg").AttrType(OPTIONAL).Int(0);     // default: standard MX
        this->Attr("round_mode").AttrType(OPTIONAL).String("rint");
        this->Attr("dst_type").AttrType(OPTIONAL).Int(40);     // default: FP4_E2M1
        this->Attr("output_rstd").AttrType(OPTIONAL).Bool(false);     // default: false

        OpAICoreConfig config950;
        config950.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "add_rms_norm_dynamic_mx_quant_apt");
        this->AICore().AddConfig("ascend950", config950);
    }
};
OP_ADD(AddRmsNormDynamicMxQuant);
} // namespace ops
