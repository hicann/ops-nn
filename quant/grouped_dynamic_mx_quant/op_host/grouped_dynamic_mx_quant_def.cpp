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
 * \file grouped_dynamic_mx_quant.cpp
 * \brief
 */

#include <cstdint>
#include "register/op_def_registry.h"

namespace ops {
static constexpr int32_t DEFAULT_DST_TYPE = 35;
static constexpr int32_t DEFAULT_BLOCK_SIZE = 32;
static constexpr int32_t DEFAULT_SCALE_ALG = 0;
static constexpr float DEFAULT_DST_TYPE_MAX_VALUE = 0.0;

static const std::vector<ge::DataType> groupedDynamicMxQuantXDataType = {
    ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16};

static const std::vector<ge::DataType> groupedDynamicMxQuantGroupIndexDataType = {
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};

static const std::vector<ge::DataType> groupedDynamicMxQuantYDataType = {
    ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2,ge::DT_FLOAT8_E5M2};

static const std::vector<ge::DataType> groupedDynamicMxQuantMxScaleDataType = {
    ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0};

static const std::vector<ge::Format> groupedDynamicMxQuantNDFormat = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class GroupedDynamicMxQuant : public OpDef {
public:
    explicit GroupedDynamicMxQuant(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(groupedDynamicMxQuantXDataType)
            .Format(groupedDynamicMxQuantNDFormat)
            .UnknownShapeFormat(groupedDynamicMxQuantNDFormat)
            .AutoContiguous();
        this->Input("group_index")
            .ParamType(REQUIRED)
            .DataType(groupedDynamicMxQuantGroupIndexDataType)
            .Format(groupedDynamicMxQuantNDFormat)
            .UnknownShapeFormat(groupedDynamicMxQuantNDFormat)
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(groupedDynamicMxQuantYDataType)
            .Format(groupedDynamicMxQuantNDFormat)
            .UnknownShapeFormat(groupedDynamicMxQuantNDFormat);
        this->Output("mxscale")
            .ParamType(REQUIRED)
            .DataType(groupedDynamicMxQuantMxScaleDataType)
            .Format(groupedDynamicMxQuantNDFormat)
            .UnknownShapeFormat(groupedDynamicMxQuantNDFormat);
        this->Attr("round_mode").AttrType(OPTIONAL).String("rint");
        this->Attr("dst_type").AttrType(OPTIONAL).Int(DEFAULT_DST_TYPE);
        this->Attr("blocksize").AttrType(OPTIONAL).Int(DEFAULT_BLOCK_SIZE);
        this->Attr("scale_alg").AttrType(OPTIONAL).Int(DEFAULT_SCALE_ALG);
        this->Attr("dst_type_max").AttrType(OPTIONAL).Float(DEFAULT_DST_TYPE_MAX_VALUE);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "grouped_dynamic_mx_quant");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(GroupedDynamicMxQuant);
}  // namespace ops