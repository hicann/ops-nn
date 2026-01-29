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
 * \file cross_entropy_loss_grad.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> dataType = {
    ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT
};

static const std::vector<ge::DataType> targetDataType = {
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32
};

static const std::vector<ge::DataType> weightDataType = {
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT
};

static const std::vector<ge::Format> dataFormat = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
};
class CrossEntropyLossGrad : public OpDef {
public:
  explicit CrossEntropyLossGrad(const char* name) : OpDef(name)
  {
    this->Input("grad_loss")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("log_prob")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("target")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("weight")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("grad_zloss")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("lse_for_zloss")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("x_grad")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

    this->Attr("reduction").AttrType(OPTIONAL).String("mean");
    // the default value of the parameter is -100
    this->Attr("ignore_index").AttrType(OPTIONAL).Int(-100);
    this->Attr("label_smoothing").AttrType(OPTIONAL).Float(0.0);
    this->Attr("lse_square_scale_for_zloss").AttrType(OPTIONAL).Float(0.0);

    this->AICore().AddConfig("ascend910b");
    this->AICore().AddConfig("ascend910_93");
    OpAICoreConfig aicoreRegbaseConfig;

    aicoreRegbaseConfig.Input("grad_loss")
        .ParamType(REQUIRED)
        .DataType(dataType)
        .Format(dataFormat)
        .UnknownShapeFormat(dataFormat)
        .AutoContiguous();
    aicoreRegbaseConfig.Input("log_prob")
        .ParamType(REQUIRED)
        .DataType(dataType)
        .Format(dataFormat)
        .UnknownShapeFormat(dataFormat)
        .AutoContiguous();
    aicoreRegbaseConfig.Input("target")
        .ParamType(REQUIRED)
        .DataType(targetDataType)
        .Format(dataFormat)
        .UnknownShapeFormat(dataFormat)
        .AutoContiguous();
    aicoreRegbaseConfig.Input("weight")
        .ParamType(OPTIONAL)
        .DataType(weightDataType)
        .Format(dataFormat)
        .UnknownShapeFormat(dataFormat)
        .AutoContiguous();
    aicoreRegbaseConfig.Input("grad_zloss")
        .ParamType(OPTIONAL)
        .DataType(dataType)
        .Format(dataFormat)
        .UnknownShapeFormat(dataFormat)
        .AutoContiguous();
    aicoreRegbaseConfig.Input("lse_for_zloss")
        .ParamType(OPTIONAL)
        .DataType(dataType)
        .Format(dataFormat)
        .UnknownShapeFormat(dataFormat)
        .AutoContiguous();
    aicoreRegbaseConfig.Output("x_grad")
        .ParamType(REQUIRED)
        .DataType(dataType)
        .UnknownShapeFormat(dataFormat)
        .Format(dataFormat);
    aicoreRegbaseConfig.DynamicCompileStaticFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .ExtendCfgInfo("opFile.value", "cross_entropy_loss_grad_apt");
    this->AICore().AddConfig("ascend950", aicoreRegbaseConfig);
  }
};
OP_ADD(CrossEntropyLossGrad);
}  // namespace ops