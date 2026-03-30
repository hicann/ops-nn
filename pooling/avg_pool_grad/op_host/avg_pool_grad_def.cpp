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
* \file avg_pool_grad_def.cpp
* \brief
*/

#include "register/op_def_registry.h"

namespace ops {

class AvgPoolGrad : public OpDef {
public:
    const std::vector<ge::DataType> AvgPoolGradXDataType = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
    const std::vector<ge::Format> AvgPoolGradXFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
    explicit AvgPoolGrad(const char* name) : OpDef(name) {
        this->Input("orig_input_shape")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({AvgPoolGradXFormat})
            .UnknownShapeFormat({AvgPoolGradXFormat})
            .AutoContiguous();
        this->Input("input_grad")
            .ParamType(REQUIRED)
            .DataType(AvgPoolGradXDataType)
            .Format(AvgPoolGradXFormat)
            .UnknownShapeFormat(AvgPoolGradXFormat)
            .AutoContiguous();
        this->Output("out_grad")
            .ParamType(REQUIRED)
            .DataType(AvgPoolGradXDataType)
            .Format(AvgPoolGradXFormat)
            .UnknownShapeFormat(AvgPoolGradXFormat)
            .AutoContiguous();
        this->Attr("ksize")
            .AttrType(REQUIRED)
            .ListInt();
        this->Attr("strides")
            .AttrType(REQUIRED)
            .ListInt();
        this->Attr("padding")
            .AttrType(REQUIRED)
            .String();
        this->Attr("data_format")
            .AttrType(OPTIONAL)
            .String("NHWC");

        OpAICoreConfig aiCoreConfig;
        aiCoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "avg_pool_grad_apt");
        this->AICore().AddConfig("ascend950", aiCoreConfig);
    }
};

OP_ADD(AvgPoolGrad);
}  // namespace ops