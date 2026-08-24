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
 * \file bn3_d_training_update_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {

// 4 formats × 3 x-dtypes = 12 entries
// Order per format: float16, float32, bfloat16
static const std::vector<ge::DataType> xDTypes = {
    ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, // NCHW
    ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, // NCDHW
    ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, // NHWC
    ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, // NDHWC
};

static const std::vector<ge::Format> xFormats = {
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
    ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC,
};

// (C,) stat tensors: all float32, all FORMAT_ND (12 entries to match x length)
static const std::vector<ge::DataType> statDTypes = {
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
};

static const std::vector<ge::Format> statFormats = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
};

class BN3DTrainingUpdate : public OpDef {
public:
    explicit BN3DTrainingUpdate(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(xDTypes)
            .Format(xFormats)
            .UnknownShapeFormat(xFormats)
            .AutoContiguous();
        this->Input("sum")
            .ParamType(REQUIRED)
            .DataType(statDTypes)
            .Format(statFormats)
            .UnknownShapeFormat(statFormats)
            .AutoContiguous();
        this->Input("square_sum")
            .ParamType(REQUIRED)
            .DataType(statDTypes)
            .Format(statFormats)
            .UnknownShapeFormat(statFormats)
            .AutoContiguous();
        this->Input("scale")
            .ParamType(REQUIRED)
            .DataType(statDTypes)
            .Format(statFormats)
            .UnknownShapeFormat(statFormats)
            .AutoContiguous();
        this->Input("offset")
            .ParamType(REQUIRED)
            .DataType(statDTypes)
            .Format(statFormats)
            .UnknownShapeFormat(statFormats)
            .AutoContiguous();
        this->Input("mean")
            .ParamType(REQUIRED)
            .DataType(statDTypes)
            .Format(statFormats)
            .UnknownShapeFormat(statFormats)
            .AutoContiguous();
        this->Input("variance")
            .ParamType(REQUIRED)
            .DataType(statDTypes)
            .Format(statFormats)
            .UnknownShapeFormat(statFormats)
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(xDTypes)
            .Format(xFormats)
            .UnknownShapeFormat(xFormats)
            .AutoContiguous();
        this->Output("mean")
            .ParamType(REQUIRED)
            .DataType(statDTypes)
            .Format(statFormats)
            .UnknownShapeFormat(statFormats)
            .AutoContiguous();
        this->Output("variance")
            .ParamType(REQUIRED)
            .DataType(statDTypes)
            .Format(statFormats)
            .UnknownShapeFormat(statFormats)
            .AutoContiguous();
        this->Output("batch_mean")
            .ParamType(REQUIRED)
            .DataType(statDTypes)
            .Format(statFormats)
            .UnknownShapeFormat(statFormats)
            .AutoContiguous();
        this->Output("batch_variance")
            .ParamType(REQUIRED)
            .DataType(statDTypes)
            .Format(statFormats)
            .UnknownShapeFormat(statFormats)
            .AutoContiguous();
        this->Attr("factor").AttrType(REQUIRED).Float(0.1f);
        this->Attr("epsilon").AttrType(REQUIRED).Float(1.0e-5f);
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "bn3_d_training_update");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(BN3DTrainingUpdate);
} // namespace ops
