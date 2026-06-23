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
 * \file avg_pool_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {

class AvgPool : public OpDef {
public:
    const std::vector<ge::DataType> AvgPoolXDataType = {
        ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
    const std::vector<ge::Format> AvgPoolXFormat = {
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
    explicit AvgPool(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(AvgPoolXDataType)
            .Format(AvgPoolXFormat)
            .UnknownShapeFormat(AvgPoolXFormat)
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(AvgPoolXDataType)
            .Format(AvgPoolXFormat)
            .UnknownShapeFormat(AvgPoolXFormat)
            .AutoContiguous();
        this->Attr("ksize").AttrType(REQUIRED).ListInt();
        this->Attr("strides").AttrType(REQUIRED).ListInt();
        this->Attr("padding").AttrType(REQUIRED).String();
        this->Attr("data_format").AttrType(OPTIONAL).String("NHWC");
        OpAICoreConfig aiCoreConfig;
        aiCoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "avg_pool_apt");
        this->AICore().AddConfig("ascend950", aiCoreConfig);
    }
};

OP_ADD(AvgPool);
}  // namespace ops