/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/* !
 * \file batch_norm_grad_v3_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> dyDataType_95 = {
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16,
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16,
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16,
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16};

static const std::vector<ge::Format> dyFormat_95 = {
    ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW, ge::FORMAT_NCHW,  ge::FORMAT_NCHW,  ge::FORMAT_NCHW,  ge::FORMAT_NCHW,  ge::FORMAT_NCHW,
    ge::FORMAT_NCHW,  ge::FORMAT_NCHW,  ge::FORMAT_NHWC,  ge::FORMAT_NHWC,  ge::FORMAT_NHWC,  ge::FORMAT_NHWC,
    ge::FORMAT_NHWC,  ge::FORMAT_NHWC,  ge::FORMAT_NHWC,  ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC,
    ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC};

static const std::vector<ge::DataType> weightDataType_95 = {
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};

static const std::vector<ge::Format> ndFormat_95 = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

static const std::vector<ge::DataType> runningDataType_95 = {
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};

static const std::vector<ge::DataType> saveDataType_95 = {
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};

static const std::vector<ge::DataType> dyDataType_a2 = {
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16,
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16};

static const std::vector<ge::Format> dyFormat_a2 = {
    ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCHW,  ge::FORMAT_NCHW,  ge::FORMAT_NCHW,
    ge::FORMAT_NCHW,  ge::FORMAT_NCHW,  ge::FORMAT_NCHW,  ge::FORMAT_NCHW};

static const std::vector<ge::DataType> weightDataType_a2 = {
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};

static const std::vector<ge::Format> ndFormat_a2 = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

static const std::vector<ge::DataType> runningDataType_a2 = {
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};

static const std::vector<ge::DataType> saveDataType_a2 = {
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};

class BatchNormGradV3 : public OpDef {
public:
    explicit BatchNormGradV3(const char* name) : OpDef(name)
    {
        // input
        this->Input("dy")
            .ParamType(REQUIRED)
            .DataType(dyDataType_95)
            .Format(dyFormat_95)
            .UnknownShapeFormat(dyFormat_95)
            .AutoContiguous();
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(dyDataType_95)
            .Format(dyFormat_95)
            .UnknownShapeFormat(dyFormat_95)
            .AutoContiguous();
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType(weightDataType_95)
            .Format(ndFormat_95)
            .UnknownShapeFormat(ndFormat_95)
            .AutoContiguous();
        this->Input("running_mean")
            .ParamType(REQUIRED)
            .DataType(runningDataType_95)
            .Format(ndFormat_95)
            .UnknownShapeFormat(ndFormat_95)
            .AutoContiguous();
        this->Input("running_var")
            .ParamType(REQUIRED)
            .DataType(runningDataType_95)
            .Format(ndFormat_95)
            .UnknownShapeFormat(ndFormat_95)
            .AutoContiguous();
        this->Input("save_mean")
            .ParamType(REQUIRED)
            .DataType(saveDataType_95)
            .Format(ndFormat_95)
            .UnknownShapeFormat(ndFormat_95)
            .AutoContiguous();
        this->Input("save_rstd")
            .ParamType(REQUIRED)
            .DataType(saveDataType_95)
            .Format(ndFormat_95)
            .UnknownShapeFormat(ndFormat_95)
            .AutoContiguous();

        // output
        this->Output("dx").ParamType(REQUIRED).DataType(dyDataType_95).Format(dyFormat_95).UnknownShapeFormat(dyFormat_95);
        this->Output("dweight")
            .ParamType(REQUIRED)
            .DataType(weightDataType_95)
            .Format(ndFormat_95)
            .UnknownShapeFormat(ndFormat_95);
        this->Output("dbias")
            .ParamType(REQUIRED)
            .DataType(weightDataType_95)
            .Format(ndFormat_95)
            .UnknownShapeFormat(ndFormat_95);

        this->Attr("is_training").AttrType(OPTIONAL).Bool(true);
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-5);
        this->Attr("version").AttrType(OPTIONAL).Int(0);
        OpAICoreConfig regbaseCfg;
        regbaseCfg.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "batch_norm_grad_v3_apt");
        this->AICore().AddConfig("ascend910_95", regbaseCfg);

        OpAICoreConfig config_a2a3;

        config_a2a3.Input("dy")
            .ParamType(REQUIRED)
            .DataType(dyDataType_a2)
            .Format(dyFormat_a2)
            .UnknownShapeFormat(dyFormat_a2);

        config_a2a3.Input("x")
            .ParamType(REQUIRED)
            .DataType(dyDataType_a2)
            .Format(dyFormat_a2)
            .UnknownShapeFormat(dyFormat_a2);

        config_a2a3.Input("weight")
            .ParamType(REQUIRED)
            .DataType(weightDataType_a2)
            .Format(ndFormat_a2)
            .UnknownShapeFormat(ndFormat_a2);

        config_a2a3.Input("running_mean")
            .ParamType(REQUIRED)
            .DataType(runningDataType_a2)
            .Format(ndFormat_a2)
            .UnknownShapeFormat(ndFormat_a2);

        config_a2a3.Input("running_var")
            .ParamType(REQUIRED)
            .DataType(runningDataType_a2)
            .Format(ndFormat_a2)
            .UnknownShapeFormat(ndFormat_a2);

        config_a2a3.Input("save_mean")
            .ParamType(REQUIRED)
            .DataType(saveDataType_a2)
            .Format(ndFormat_a2)
            .UnknownShapeFormat(ndFormat_a2);

        config_a2a3.Input("save_rstd")
            .ParamType(REQUIRED)
            .DataType(saveDataType_a2)
            .Format(ndFormat_a2)
            .UnknownShapeFormat(ndFormat_a2);

        config_a2a3.Output("dx")
            .ParamType(REQUIRED)
            .DataType(dyDataType_a2)
            .Format(dyFormat_a2)
            .UnknownShapeFormat(dyFormat_a2);

        config_a2a3.Output("dweight")
            .ParamType(REQUIRED)
            .DataType(weightDataType_a2)
            .Format(ndFormat_a2)
            .UnknownShapeFormat(ndFormat_a2);

        config_a2a3.Output("dbias")
            .ParamType(REQUIRED)
            .DataType(weightDataType_a2)
            .Format(ndFormat_a2)
            .UnknownShapeFormat(ndFormat_a2);

        config_a2a3.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend910b", config_a2a3);
        this->AICore().AddConfig("ascend910_93", config_a2a3);
    }
};

OP_ADD(BatchNormGradV3);
} // namespace ops
