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
 * \file in_training_update_v2_def.cpp
 * \brief INTrainingUpdateV2 Ascend 950 operator definition.
 */

#include "register/op_def_registry.h"

namespace ops {
class INTrainingUpdateV2 : public OpDef {
public:
    explicit INTrainingUpdateV2(const char* name) : OpDef(name)
    {
        // Two public layouts times two x dtypes. Statistics use ND as their
        // physical format; their public origin format and 4-D logical shape are
        // checked by tiling. This also permits an ignored optional orphan to
        // carry either public layout without constraining the active profile.
        const std::vector<ge::DataType> xDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT};
        const std::vector<ge::DataType> statDtypes(4, ge::DT_FLOAT);
        const std::vector<ge::Format> xFormats = {ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC};
        const std::vector<ge::Format> statFormats(4, ge::FORMAT_ND);

        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(xDtypes)
            .Format(xFormats)
            .UnknownShapeFormat(xFormats)
            .AutoContiguous();
        this->Input("sum")
            .ParamType(REQUIRED)
            .DataType(statDtypes)
            .Format(statFormats)
            .UnknownShapeFormat(statFormats)
            .AutoContiguous();
        this->Input("square_sum")
            .ParamType(REQUIRED)
            .DataType(statDtypes)
            .Format(statFormats)
            .UnknownShapeFormat(statFormats)
            .AutoContiguous();
        this->Input("gamma")
            .ParamType(OPTIONAL)
            .DataType(statDtypes)
            .Format(statFormats)
            .UnknownShapeFormat(statFormats)
            .AutoContiguous();
        this->Input("beta")
            .ParamType(OPTIONAL)
            .DataType(statDtypes)
            .Format(statFormats)
            .UnknownShapeFormat(statFormats)
            .AutoContiguous();
        this->Input("mean")
            .ParamType(OPTIONAL)
            .DataType(statDtypes)
            .Format(statFormats)
            .UnknownShapeFormat(statFormats)
            .AutoContiguous();
        this->Input("variance")
            .ParamType(OPTIONAL)
            .DataType(statDtypes)
            .Format(statFormats)
            .UnknownShapeFormat(statFormats)
            .AutoContiguous();

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(xDtypes)
            .Format(xFormats)
            .UnknownShapeFormat(xFormats)
            .AutoContiguous();
        this->Output("batch_mean")
            .ParamType(REQUIRED)
            .DataType(statDtypes)
            .Format(statFormats)
            .UnknownShapeFormat(statFormats)
            .AutoContiguous();
        this->Output("batch_variance")
            .ParamType(REQUIRED)
            .DataType(statDtypes)
            .Format(statFormats)
            .UnknownShapeFormat(statFormats)
            .AutoContiguous();

        this->Attr("momentum").AttrType(OPTIONAL).Float(0.1f);
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1.0e-5f);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(false)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "in_training_update_v2");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(INTrainingUpdateV2);
} // namespace ops
