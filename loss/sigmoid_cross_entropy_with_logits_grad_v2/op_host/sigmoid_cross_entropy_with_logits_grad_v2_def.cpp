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
 * \file sigmoid_cross_entropy_with_logits_grad_v2_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> supportedDtypes = {
    ge::DT_BF16,
    ge::DT_FLOAT16,
    ge::DT_FLOAT,
};
static const std::vector<ge::Format> supportedFormats = {
    ge::FORMAT_ND,
    ge::FORMAT_ND,
    ge::FORMAT_ND,
};

class SigmoidCrossEntropyWithLogitsGradV2 : public OpDef {
public:
    explicit SigmoidCrossEntropyWithLogitsGradV2(const char* name) : OpDef(name)
    {
        this->Input("predict")
            .ParamType(REQUIRED)
            .DataType(supportedDtypes)
            .Format(supportedFormats)
            .UnknownShapeFormat(supportedFormats);
        this->Input("target")
            .ParamType(REQUIRED)
            .DataType(supportedDtypes)
            .Format(supportedFormats)
            .UnknownShapeFormat(supportedFormats);
        this->Input("dout")
            .ParamType(REQUIRED)
            .DataType(supportedDtypes)
            .Format(supportedFormats)
            .UnknownShapeFormat(supportedFormats);
        this->Input("weight")
            .ParamType(OPTIONAL)
            .DataType(supportedDtypes)
            .Format(supportedFormats)
            .UnknownShapeFormat(supportedFormats);
        this->Input("pos_weight")
            .ParamType(OPTIONAL)
            .DataType(supportedDtypes)
            .Format(supportedFormats)
            .UnknownShapeFormat(supportedFormats);
        this->Output("gradient")
            .ParamType(REQUIRED)
            .DataType(supportedDtypes)
            .Format(supportedFormats)
            .UnknownShapeFormat(supportedFormats);
        this->Attr("reduction").AttrType(OPTIONAL).String("mean");
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "sigmoid_cross_entropy_with_logits_grad_v2_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};
OP_ADD(SigmoidCrossEntropyWithLogitsGradV2);
} // namespace ops
