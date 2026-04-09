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
 * \file rms_norm_dynamic_mx_quant_def.cpp
 * \brief RmsNormDynamicMxQuant operator definition
 */

#include "register/op_def_registry.h"

namespace ops {
static constexpr int DEAF_DST_TYPE = 40;  // ge::DT_FLOAT4_E2M1
static const std::vector<ge::DataType> xDataTypeRegbase = {
    ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16,
    ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16};

static const std::vector<ge::DataType> gammaDataTypeRegbase = {
    ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_FLOAT16, ge::DT_BF16,
    ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_FLOAT,   ge::DT_FLOAT, ge::DT_FLOAT,   ge::DT_FLOAT,
    ge::DT_FLOAT,   ge::DT_FLOAT, ge::DT_FLOAT,   ge::DT_FLOAT};

static const std::vector<ge::DataType> yDataTypeRegbase = {
    ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E1M2, ge::DT_FLOAT4_E1M2,
    ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2,
    ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E2M1,   ge::DT_FLOAT4_E1M2, ge::DT_FLOAT4_E1M2,
    ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2};

static const std::vector<ge::DataType> mxScaleDataTypeRegbase = {
    ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
    ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
    ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
    ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0};

static const std::vector<ge::DataType> rstdDataTypeRegbase = {
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};

static const std::vector<ge::Format> formatRegbase = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                      ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                      ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                      ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
class RmsNormDynamicMxQuant : public OpDef {
public:
    explicit RmsNormDynamicMxQuant(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(xDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase)
            .AutoContiguous();
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType(gammaDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase)
            .AutoContiguous();
        this->Input("beta")
            .ParamType(OPTIONAL)
            .DataType(gammaDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase)
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(yDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase)
            .AutoContiguous();
        this->Output("mxscale")
            .ParamType(REQUIRED)
            .DataType(mxScaleDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase)
            .AutoContiguous();
        this->Output("rstd")
            .ParamType(REQUIRED)
            .DataType(rstdDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase)
            .AutoContiguous();
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-06);
        this->Attr("scale_alg").AttrType(OPTIONAL).Int(0); // OCP: 0, CUSTOM_NV: 1
        this->Attr("round_mode").AttrType(OPTIONAL).String("rint");
        this->Attr("dst_type").AttrType(OPTIONAL).Int(DEAF_DST_TYPE); // ge::DT_FLOAT4_E2M1
        this->Attr("output_rstd").AttrType(OPTIONAL).Bool(false);
        OpAICoreConfig aicoreConfig950;
        aicoreConfig950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "rms_norm_dynamic_mx_quant_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig950);
    }
};

OP_ADD(RmsNormDynamicMxQuant);
} // namespace ops
