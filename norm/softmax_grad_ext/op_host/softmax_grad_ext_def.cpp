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
 * \file softmax_grad_ext_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {

static const std::vector<ge::DataType> SoftmaxGradExtDataType = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};

static const std::vector<ge::Format> SoftmaxGradExtFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class SoftmaxGradExt : public OpDef {
public:
    explicit SoftmaxGradExt(const char* name) : OpDef(name)
    {
        this->Input("grad")
            .ParamType(REQUIRED)
            .DataType(SoftmaxGradExtDataType)
            .Format(SoftmaxGradExtFormat)
            .UnknownShapeFormat(SoftmaxGradExtFormat);
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType(SoftmaxGradExtDataType)
            .Format(SoftmaxGradExtFormat)
            .UnknownShapeFormat(SoftmaxGradExtFormat);
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType(SoftmaxGradExtDataType)
            .Format(SoftmaxGradExtFormat)
            .UnknownShapeFormat(SoftmaxGradExtFormat);

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(SoftmaxGradExtDataType)
            .Format(SoftmaxGradExtFormat)
            .UnknownShapeFormat(SoftmaxGradExtFormat);

        this->Attr("axes").AttrType(OPTIONAL).Int(-1);
        this->Attr("keep_dims").AttrType(OPTIONAL).Bool(true);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "softmax_grad_ext_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(SoftmaxGradExt);
} // namespace ops