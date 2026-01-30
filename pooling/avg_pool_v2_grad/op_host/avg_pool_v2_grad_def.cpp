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
* \file avg_pool_v2_grad_def.cpp
* \brief
*/

#include "register/op_def_registry.h"

namespace ops {

class AvgPoolV2Grad : public OpDef {
public:
    const std::vector<ge::DataType> AvgPoolV2GradXDataType = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
    const std::vector<ge::Format> AvgPoolV2GradXFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
    explicit AvgPoolV2Grad(const char* name) : OpDef(name) {
        this->Input("orig_input_shape")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({AvgPoolV2GradXFormat})
            .UnknownShapeFormat({AvgPoolV2GradXFormat})
            .AutoContiguous();
        this->Input("input_grad")
            .ParamType(REQUIRED)
            .DataType(AvgPoolV2GradXDataType)
            .Format(AvgPoolV2GradXFormat)
            .UnknownShapeFormat(AvgPoolV2GradXFormat)
            .AutoContiguous();
        this->Output("out_grad")
            .ParamType(REQUIRED)
            .DataType(AvgPoolV2GradXDataType)
            .Format(AvgPoolV2GradXFormat)
            .UnknownShapeFormat(AvgPoolV2GradXFormat)
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
            .ExtendCfgInfo("opFile.value", "avg_pool_v2_grad_apt");
        this->AICore().AddConfig("ascend950", aiCoreConfig);
    }
};

OP_ADD(AvgPoolV2Grad);
}  // namespace ops