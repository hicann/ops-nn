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
 * \file grouped_dynamic_block_quant.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {
static constexpr int32_t DEFAULT_ROW_BLOCK_SIZE = 1;
static constexpr int32_t DEFAULT_COL_BLOCK_SIZE = 128;
static constexpr int32_t DEFAULT_DST_TYPE = 35;
static constexpr int32_t DEFAULT_GROUP_LIST_TYPE = 0;

static const std::vector<ge::DataType> groupedDynamicBlockQuantXDataType = {
    ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16};

static const std::vector<ge::DataType> groupedDynamicBlockQuantGroupListDataType = {
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};

static const std::vector<ge::DataType> groupedDynamicBlockQuantYDataType = {ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,
                                                                            ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
                                                                            ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2};

static const std::vector<ge::DataType> groupedDynamicBlockQuantScaleDataType = {
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};

static const std::vector<ge::Format> groupedDynamicBlockQuantNDFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                                         ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class GroupedDynamicBlockQuant : public OpDef {
public:
    explicit GroupedDynamicBlockQuant(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(groupedDynamicBlockQuantXDataType)
            .Format(groupedDynamicBlockQuantNDFormat)
            .UnknownShapeFormat(groupedDynamicBlockQuantNDFormat)
            .AutoContiguous();
        this->Input("group_list")
            .ParamType(REQUIRED)
            .DataType(groupedDynamicBlockQuantGroupListDataType)
            .Format(groupedDynamicBlockQuantNDFormat)
            .UnknownShapeFormat(groupedDynamicBlockQuantNDFormat)
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(groupedDynamicBlockQuantYDataType)
            .Format(groupedDynamicBlockQuantNDFormat)
            .UnknownShapeFormat(groupedDynamicBlockQuantNDFormat);
        this->Output("scale")
            .ParamType(REQUIRED)
            .DataType(groupedDynamicBlockQuantScaleDataType)
            .Format(groupedDynamicBlockQuantNDFormat)
            .UnknownShapeFormat(groupedDynamicBlockQuantNDFormat);
        this->Attr("min_scale").AttrType(OPTIONAL).Float(0.0);
        this->Attr("round_mode").AttrType(OPTIONAL).String("rint");
        this->Attr("dst_type").AttrType(OPTIONAL).Int(DEFAULT_DST_TYPE);
        this->Attr("row_block_size").AttrType(OPTIONAL).Int(DEFAULT_ROW_BLOCK_SIZE);
        this->Attr("col_block_size").AttrType(OPTIONAL).Int(DEFAULT_COL_BLOCK_SIZE);
        this->Attr("group_list_type").AttrType(OPTIONAL).Int(DEFAULT_GROUP_LIST_TYPE);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "grouped_dynamic_block_quant");
        this->AICore().AddConfig("ascend910_95", aicoreConfig);
    }
};

OP_ADD(GroupedDynamicBlockQuant);
} // namespace ops