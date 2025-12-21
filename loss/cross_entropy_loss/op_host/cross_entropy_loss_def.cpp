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
 * \file cross_entropy_loss_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"
namespace ops {
static const std::vector<ge::DataType> xDtype = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,
                                                 ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
static const std::vector<ge::DataType> targetDtype = {ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                                                      ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};
static const std::vector<ge::DataType> weightDtype = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                                                      ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
static const std::vector<ge::Format> xFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
class CrossEntropyLoss : public OpDef
{
public:
    explicit CrossEntropyLoss(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("target")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT64, ge::DT_INT64})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("loss")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("log_prob")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("zloss")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("lse_for_zloss")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("reduction").AttrType(OPTIONAL).String("mean");
        this->Attr("ignore_index").AttrType(OPTIONAL).Int(-100); // -100: default value of ignore_index
        this->Attr("label_smoothing").AttrType(OPTIONAL).Float(0.0);
        this->Attr("lse_square_scale_for_zloss").AttrType(OPTIONAL).Float(0.0);
        this->Attr("return_zloss").AttrType(OPTIONAL).Bool(false);
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true).DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true);

        this->AICore().AddConfig("ascend910b", aicoreConfig);
        this->AICore().AddConfig("ascend910_93", aicoreConfig);
        OpAICoreConfig regbaseConfig;
        regbaseConfig.Input("input")
            .ParamType(REQUIRED)
            .DataType(xDtype)
            .Format(xFormat)
            .UnknownShapeFormat(xFormat)
            .AutoContiguous();
        regbaseConfig.Input("target")
            .ParamType(REQUIRED)
            .DataType(targetDtype)
            .Format(xFormat)
            .UnknownShapeFormat(xFormat)
            .AutoContiguous();
        regbaseConfig.Input("weight")
            .ParamType(OPTIONAL)
            .DataType(weightDtype)
            .Format(xFormat)
            .UnknownShapeFormat(xFormat)
            .AutoContiguous();
        regbaseConfig.Output("loss").ParamType(REQUIRED).DataType(xDtype).Format(xFormat).UnknownShapeFormat(xFormat);
        regbaseConfig.Output("log_prob")
            .ParamType(REQUIRED)
            .DataType(xDtype)
            .Format(xFormat)
            .UnknownShapeFormat(xFormat);
        regbaseConfig.Output("zloss").ParamType(REQUIRED).DataType(xDtype).Format(xFormat).UnknownShapeFormat(xFormat);
        regbaseConfig.Output("lse_for_zloss")
            .ParamType(REQUIRED)
            .DataType(xDtype)
            .Format(xFormat)
            .UnknownShapeFormat(xFormat);

        regbaseConfig.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "cross_entropy_loss_apt");
        this->AICore().AddConfig("ascend910_95", regbaseConfig);
    }
};
OP_ADD(CrossEntropyLoss);
} // namespace ops
