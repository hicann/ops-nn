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
 * \file max_pool_v2_def.cpp
 * \brief imply for max_pool
 */

#include "register/op_def_registry.h"

namespace ops {
class MaxPoolV2 : public OpDef {
public:
    const std::vector<ge::DataType> maxPoolV2XDataType = {
        ge::DT_FLOAT16};
    const std::vector<ge::DataType> maxPoolV2KsizeDataType = {
        ge::DT_INT32};
    const std::vector<ge::Format> maxPoolV2XFormat = {
        ge::FORMAT_ND};
    explicit MaxPoolV2(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(maxPoolV2XDataType)
            .Format(maxPoolV2XFormat)
            .UnknownShapeFormat(maxPoolV2XFormat)
            .AutoContiguous();
        this->Input("ksize")
            .ParamType(REQUIRED)
            .DataType(maxPoolV2KsizeDataType)
            .ValueDepend(OPTIONAL)
            .Format(maxPoolV2XFormat)
            .UnknownShapeFormat(maxPoolV2XFormat)
            .AutoContiguous();
        this->Input("strides")
            .ParamType(REQUIRED)
            .DataType(maxPoolV2KsizeDataType)
            .ValueDepend(OPTIONAL)
            .Format(maxPoolV2XFormat)
            .UnknownShapeFormat(maxPoolV2XFormat)
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(maxPoolV2XDataType)
            .Format(maxPoolV2XFormat)
            .UnknownShapeFormat(maxPoolV2XFormat)
            .AutoContiguous();
        this->Attr("padding").AttrType(REQUIRED).String();
        this->Attr("data_format").AttrType(OPTIONAL).String("NCHW");
        OpAICoreConfig aiCoreConfig;
        aiCoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "max_pool_v2_apt");
        this->AICore().AddConfig("ascend950", aiCoreConfig);
    }
};

OP_ADD(MaxPoolV2);
} // namespace ops