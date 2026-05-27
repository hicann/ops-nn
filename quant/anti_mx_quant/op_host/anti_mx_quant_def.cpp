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
 * \file anti_mx_quant.cpp
 * \brief
 */

#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
static constexpr int32_t DEFAULT_AXIS = -1;
static constexpr int32_t DEFAULT_DST_TYPE = 27;

static const std::vector<ge::DataType> antiMxQuantXDataType = {
    ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN,
    ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN,
    ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN};

static const std::vector<ge::DataType> antiMxQuantMxscaleDataType = {
    ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
    ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0,
    ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0};

static const std::vector<ge::DataType> antiMxQuantYDataType = {
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
    ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};

static const std::vector<ge::Format> antiMxQuantNDFormat = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class AntiMxQuant : public OpDef {
public:
    explicit AntiMxQuant(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(antiMxQuantXDataType)
            .Format(antiMxQuantNDFormat)
            .UnknownShapeFormat(antiMxQuantNDFormat)
            .AutoContiguous();
        this->Input("mxscale")
            .ParamType(REQUIRED)
            .DataType(antiMxQuantMxscaleDataType)
            .Format(antiMxQuantNDFormat)
            .UnknownShapeFormat(antiMxQuantNDFormat)
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(antiMxQuantYDataType)
            .Format(antiMxQuantNDFormat)
            .UnknownShapeFormat(antiMxQuantNDFormat);
        this->Attr("axis").AttrType(OPTIONAL).Int(DEFAULT_AXIS);
        this->Attr("dst_type").AttrType(OPTIONAL).Int(DEFAULT_DST_TYPE);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "anti_mx_quant");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(AntiMxQuant);
}  // namespace ops