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
 * \file quant_update_scatter_def.cpp
 * \brief quant_update_scatter op host
 */
#include "register/op_def_registry.h"

namespace ops {
constexpr int32_t DEFAULT_QUANT_UPDATE_SCATTER_AXIS = -2;

static const std::vector<ge::DataType> varDataType = {
    ge::DT_INT8,          ge::DT_INT8,          ge::DT_INT8,          ge::DT_INT8,
    ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,
    ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
    ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2};

static const std::vector<ge::DataType> indicesDataType = {
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64};

static const std::vector<ge::DataType> updatesDataType = {
    ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16,
    ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16};

static const std::vector<ge::DataType> quantScalesDataType = {
    ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT,
    ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT};

static const std::vector<ge::DataType> quantZeroPointsDataType = {
    ge::DT_BF16, ge::DT_INT32, ge::DT_BF16, ge::DT_INT32, ge::DT_BF16, ge::DT_INT32, ge::DT_BF16, ge::DT_INT32,
    ge::DT_BF16, ge::DT_INT32, ge::DT_BF16, ge::DT_INT32, ge::DT_BF16, ge::DT_INT32, ge::DT_BF16, ge::DT_INT32};

static const std::vector<ge::Format> inputAndOutputFormat = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class QuantUpdateScatter : public OpDef {
public:
    explicit QuantUpdateScatter(const char* name) : OpDef(name)
    {
        this->Input("var")
            .ParamType(REQUIRED)
            .DataType(varDataType)
            .Format(inputAndOutputFormat)
            .UnknownShapeFormat(inputAndOutputFormat);
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType(indicesDataType)
            .Format(inputAndOutputFormat)
            .UnknownShapeFormat(inputAndOutputFormat);
        this->Input("updates")
            .ParamType(REQUIRED)
            .DataType(updatesDataType)
            .Format(inputAndOutputFormat)
            .UnknownShapeFormat(inputAndOutputFormat);
        this->Input("quant_scales")
            .ParamType(REQUIRED)
            .DataType(quantScalesDataType)
            .Format(inputAndOutputFormat)
            .UnknownShapeFormat(inputAndOutputFormat);
        this->Input("quant_zero_points")
            .ParamType(OPTIONAL)
            .DataType(quantZeroPointsDataType)
            .Format(inputAndOutputFormat)
            .UnknownShapeFormat(inputAndOutputFormat);
        this->Output("var")
            .ParamType(REQUIRED)
            .DataType(varDataType)
            .Format(inputAndOutputFormat)
            .UnknownShapeFormat(inputAndOutputFormat);
        this->Attr("reduce").AttrType(REQUIRED).String();
        this->Attr("axis").AttrType(OPTIONAL).Int(DEFAULT_QUANT_UPDATE_SCATTER_AXIS);
        this->Attr("quant_axis").AttrType(OPTIONAL).Int(-1);
        this->Attr("reciprocal_scale").AttrType(OPTIONAL).Bool(false);
        this->Attr("round_mode").AttrType(OPTIONAL).String("rint");
        OpAICoreConfig config_950;
        config_950.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "quant_update_scatter_apt");
        this->AICore().AddConfig("ascend950", config_950);
    }
};
OP_ADD(QuantUpdateScatter);
} // namespace ops
