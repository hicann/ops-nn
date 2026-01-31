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
 * \file segment_sum_def.cpp
 * \brief segment_sum op host config
 */

#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::Format> format = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
};
static const std::vector<ge::DataType> valueDataTypeXY = {
    ge::DT_FLOAT16, ge::DT_FLOAT16,  ge::DT_FLOAT,  ge::DT_FLOAT,   ge::DT_INT32,   ge::DT_INT32,  ge::DT_INT64,
    ge::DT_INT64,   ge::DT_UINT32,  ge::DT_UINT32,  ge::DT_UINT64,  ge::DT_UINT64,  ge::DT_BF16,   ge::DT_BF16
};
static const std::vector<ge::DataType> valueDataTypeIds = {
    ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64,  ge::DT_INT32, ge::DT_INT64, ge::DT_INT32,
    ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32,  ge::DT_INT64, ge::DT_INT32, ge::DT_INT64
};

class SegmentSum : public OpDef {
public:
    explicit SegmentSum(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType(valueDataTypeXY).Format(format).UnknownShapeFormat(format);
        this->Input("segment_ids")
            .ParamType(REQUIRED)
            .DataType(valueDataTypeIds)
            .Format(format)
            .UnknownShapeFormat(format)
            .ValueDepend(OPTIONAL);
        this->Output("y").ParamType(REQUIRED).DataType(valueDataTypeXY).Format(format).UnknownShapeFormat(format);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "segment_sum_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(SegmentSum);
} // namespace ops
