/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file rms_norm_quant_v2_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {

static const std::vector<ge::DataType> xDataTypeRegbase = {
ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT,    ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT16,  ge::DT_BF16,   ge::DT_FLOAT16,  ge::DT_BF16,
ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT,    ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT16,  ge::DT_BF16,   ge::DT_FLOAT16,  ge::DT_BF16,
ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT,    ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT16,  ge::DT_BF16,   ge::DT_FLOAT16,  ge::DT_BF16,
ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT,    ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT16,  ge::DT_BF16,   ge::DT_FLOAT16,  ge::DT_BF16,
ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT,    ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT16,  ge::DT_BF16,   ge::DT_FLOAT16,  ge::DT_BF16
};
static const std::vector<ge::DataType> scalesDataTypeRegbase = {
ge::DT_FLOAT,    ge::DT_FLOAT,    ge::DT_FLOAT,    ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT,    ge::DT_FLOAT,  ge::DT_FLOAT16,  ge::DT_BF16,
ge::DT_FLOAT,    ge::DT_FLOAT,    ge::DT_FLOAT,    ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT,    ge::DT_FLOAT,  ge::DT_FLOAT16,  ge::DT_BF16,
ge::DT_FLOAT,    ge::DT_FLOAT,    ge::DT_FLOAT,    ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT,    ge::DT_FLOAT,  ge::DT_FLOAT16,  ge::DT_BF16,
ge::DT_FLOAT,    ge::DT_FLOAT,    ge::DT_FLOAT,    ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT,    ge::DT_FLOAT,  ge::DT_FLOAT16,  ge::DT_BF16,
ge::DT_FLOAT,    ge::DT_FLOAT,    ge::DT_FLOAT,    ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT,    ge::DT_FLOAT,  ge::DT_FLOAT16,  ge::DT_BF16
};
static const std::vector<ge::DataType> zeroPointsDataTypeRegbase = {
ge::DT_INT32,    ge::DT_INT32,    ge::DT_FLOAT,    ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT,    ge::DT_FLOAT,  ge::DT_INT8,     ge::DT_INT8,
ge::DT_INT32,    ge::DT_INT32,    ge::DT_FLOAT,    ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT,    ge::DT_FLOAT,  ge::DT_INT8,     ge::DT_INT8,
ge::DT_INT32,    ge::DT_INT32,    ge::DT_FLOAT,    ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT,    ge::DT_FLOAT,  ge::DT_INT8,     ge::DT_INT8,
ge::DT_INT32,    ge::DT_INT32,    ge::DT_FLOAT,    ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT,    ge::DT_FLOAT,  ge::DT_INT8,     ge::DT_INT8,
ge::DT_INT32,    ge::DT_INT32,    ge::DT_FLOAT,    ge::DT_FLOAT16,  ge::DT_BF16,     ge::DT_FLOAT,    ge::DT_FLOAT,  ge::DT_INT8,     ge::DT_INT8
};
static const std::vector<ge::DataType> yDataTypeRegbase = {
ge::DT_INT8,     ge::DT_INT8,     ge::DT_INT8,     ge::DT_INT8,     ge::DT_INT8,     ge::DT_INT8,     ge::DT_INT8,   ge::DT_INT8,     ge::DT_INT8,
ge::DT_INT4,     ge::DT_INT4,     ge::DT_INT4,     ge::DT_INT4,     ge::DT_INT4,     ge::DT_INT4,     ge::DT_INT4,   ge::DT_INT4,     ge::DT_INT4,
ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,
ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8
};
static const std::vector<ge::Format> formatRegbase = {
ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND, ge::FORMAT_ND,   ge::FORMAT_ND,
ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND, ge::FORMAT_ND,   ge::FORMAT_ND,
ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND, ge::FORMAT_ND,   ge::FORMAT_ND,
ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND, ge::FORMAT_ND,   ge::FORMAT_ND,
ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND,   ge::FORMAT_ND, ge::FORMAT_ND,   ge::FORMAT_ND
};

class RmsNormQuantV2 : public OpDef {
public:
    explicit RmsNormQuantV2(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(xDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase)
            .AutoContiguous();
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType(xDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase)
            .AutoContiguous();
        this->Input("scales1")
            .ParamType(REQUIRED)
            .DataType(scalesDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase)
            .AutoContiguous();
        this->Input("scales2")
            .ParamType(OPTIONAL)
            .DataType(scalesDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase)
            .AutoContiguous();
        this->Input("zero_points1")
            .ParamType(OPTIONAL)
            .DataType(zeroPointsDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase)
            .AutoContiguous();
        this->Input("zero_points2")
            .ParamType(OPTIONAL)
            .DataType(zeroPointsDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase)
            .AutoContiguous();
        this->Input("beta")
            .ParamType(OPTIONAL)
            .DataType(xDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase)
            .AutoContiguous();
        this->Output("y1")
            .ParamType(REQUIRED)
            .DataType(yDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase)
            .AutoContiguous();
        this->Output("y2")
            .ParamType(REQUIRED)
            .DataType(yDataTypeRegbase)
            .Format(formatRegbase)
            .UnknownShapeFormat(formatRegbase)
            .AutoContiguous();

        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-6);
        this->Attr("div_mode").AttrType(OPTIONAL).Bool(true);
        this->Attr("dst_type").AttrType(OPTIONAL).Int(ge::DT_INT8);

        OpAICoreConfig configRegbase;
        configRegbase.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "rms_norm_quant_v2_apt");
        this->AICore().AddConfig("ascend950", configRegbase);
    }
};
OP_ADD(RmsNormQuantV2);
} // namespace ops