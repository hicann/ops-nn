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
 * \file masked_scatter_with_position.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> SUPPORT_DTYPE = {
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_DOUBLE, ge::DT_UINT8, ge::DT_INT8,
    ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, ge::DT_BOOL, ge::DT_BF16
};

static const std::vector<ge::DataType> MASK_SUPPORT_DTYPE = {
    ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL,
    ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL
};

static const std::vector<ge::DataType> POSITION_SUPPORT_DTYPE = {
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64
};

static const std::vector<ge::Format> SUPPORT_FORMAT = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
};
class MaskedScatterWithPosition : public OpDef {
public:
    explicit MaskedScatterWithPosition(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(SUPPORT_DTYPE)
            .Format(SUPPORT_FORMAT)
            .UnknownShapeFormat(SUPPORT_FORMAT)
            .AutoContiguous();
        this->Input("mask")
            .ParamType(REQUIRED)
            .DataType(MASK_SUPPORT_DTYPE)
            .Format(SUPPORT_FORMAT)
            .UnknownShapeFormat(SUPPORT_FORMAT)
            .AutoContiguous();
        this->Input("position")
            .ParamType(REQUIRED)
            .DataType(POSITION_SUPPORT_DTYPE)
            .Format(SUPPORT_FORMAT)
            .UnknownShapeFormat(SUPPORT_FORMAT)
            .AutoContiguous();
        this->Input("updates")
            .ParamType(REQUIRED)
            .DataType(SUPPORT_DTYPE)
            .Format(SUPPORT_FORMAT)
            .UnknownShapeFormat(SUPPORT_FORMAT)
            .AutoContiguous();

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(SUPPORT_DTYPE)
            .Format(SUPPORT_FORMAT)
            .UnknownShapeFormat(SUPPORT_FORMAT)
            .AutoContiguous();

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "masked_scatter_with_position");

        this->AICore().AddConfig("ascend910_95", aicoreConfig);
    }
};

OP_ADD(MaskedScatterWithPosition); // 添加算子信息库
} // namespace ops