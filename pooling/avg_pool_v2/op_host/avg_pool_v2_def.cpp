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
 * \file avg_pool_v2_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {

class AvgPoolV2 : public OpDef {
public:
    const std::vector<ge::DataType> AvgPoolV2XDataType = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
    const std::vector<ge::Format> AvgPoolV2XFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
    explicit AvgPoolV2(const char* name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(AvgPoolV2XDataType)
            .Format(AvgPoolV2XFormat)
            .UnknownShapeFormat(AvgPoolV2XFormat)
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(AvgPoolV2XDataType)
            .Format(AvgPoolV2XFormat)
            .UnknownShapeFormat(AvgPoolV2XFormat)
            .AutoContiguous();
        this->Attr("ksize")
            .AttrType(REQUIRED)
            .ListInt();
        this->Attr("strides")
            .AttrType(REQUIRED)
            .ListInt();
        this->Attr("padding_mode")
            .AttrType(OPTIONAL)
            .String("CALCULATED");
        this->Attr("pads")
            .AttrType(OPTIONAL)
            .ListInt({0, 0, 0, 0});
        this->Attr("data_format")
            .AttrType(OPTIONAL)
            .String("NCHW");
        this->Attr("global_pooling")
            .AttrType(OPTIONAL)
            .Bool(false);
        this->Attr("ceil_mode")
            .AttrType(OPTIONAL)
            .Bool(false);
        this->Attr("exclusive")
            .AttrType(OPTIONAL)
            .Bool(true);
        this->Attr("divisor_override")
            .AttrType(OPTIONAL)
            .Int(0);

        OpAICoreConfig aiCoreConfig;
        aiCoreConfig.DynamicCompileStaticFlag(true)
              .DynamicFormatFlag(false)
              .DynamicRankSupportFlag(true)
              .DynamicShapeSupportFlag(true)
              .NeedCheckSupportFlag(false)
              .PrecisionReduceFlag(true)
              .ExtendCfgInfo("opFile.value", "avg_pool_v2_apt");
        this->AICore().AddConfig("ascend950", aiCoreConfig);
    }
};

OP_ADD(AvgPoolV2);
}  // namespace ops
