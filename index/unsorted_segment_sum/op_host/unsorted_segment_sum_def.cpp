/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


/*!
 * \file unsorted_segment_sum_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::Format> format = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
};
static const std::vector<ge::DataType> valueDataTypeXY = {
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,  ge::DT_FLOAT,  ge::DT_FLOAT,
    ge::DT_FLOAT,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,  ge::DT_INT64,  ge::DT_INT64,
    ge::DT_INT64,   ge::DT_INT64,   ge::DT_UINT32,  ge::DT_UINT32,  ge::DT_UINT32, ge::DT_UINT32, ge::DT_UINT64,
    ge::DT_UINT64,  ge::DT_UINT64,  ge::DT_UINT64,  ge::DT_BF16,    ge::DT_BF16,   ge::DT_BF16,   ge::DT_BF16
};
static const std::vector<ge::DataType> valueDataTypeIds = {
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT32,
    ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64
};
static const std::vector<ge::DataType> valueDataTypeNumSeg = {
    ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32,
    ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64,
    ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32,
    ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64
};
class UnsortedSegmentSum : public OpDef {
public:
    explicit UnsortedSegmentSum(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType(valueDataTypeXY).Format(format).UnknownShapeFormat(format);
        this->Input("segment_ids")
            .ParamType(REQUIRED)
            .DataType(valueDataTypeIds)
            .Format(format)
            .UnknownShapeFormat(format);
        this->Input("num_segments")
            .ParamType(REQUIRED)
            .DataType(valueDataTypeNumSeg)
            .Format(format)
            .UnknownShapeFormat(format)
            .ValueDepend(OPTIONAL);
        this->Output("y").ParamType(REQUIRED).DataType(valueDataTypeXY).Format(format).UnknownShapeFormat(format);
        this->Attr("is_preprocessed").AttrType(OPTIONAL).Bool(false);
        this->Attr("check_ids").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "unsorted_segment_sum_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(UnsortedSegmentSum);
} // namespace ops
