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
 * \file apply_adam_def.cpp
 * \brief apply_adam_def source file
 */
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> dataType = {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT};

static const std::vector<ge::DataType> scalarDataType = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};

static const std::vector<ge::Format> dataFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

static const std::vector<ge::Format> paraFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class ApplyAdam : public OpDef {
public:
    explicit ApplyAdam(const char* name) : OpDef(name)
    {
        this->Input("var")
            .ParamType(REQUIRED)
            .DataType(dataType)
            .Format(dataFormat)
            .UnknownShapeFormat(dataFormat)
            .AutoContiguous();
        this->Input("m")
            .ParamType(REQUIRED)
            .DataType(dataType)
            .Format(dataFormat)
            .UnknownShapeFormat(dataFormat)
            .AutoContiguous();
        this->Input("v")
            .ParamType(REQUIRED)
            .DataType(dataType)
            .Format(dataFormat)
            .UnknownShapeFormat(dataFormat)
            .AutoContiguous();
        this->Input("beta1_power")
            .ParamType(REQUIRED)
            .DataType(scalarDataType)
            .Format(paraFormat)
            .UnknownShapeFormat(paraFormat)
            .AutoContiguous();
        this->Input("beta2_power")
            .ParamType(REQUIRED)
            .DataType(scalarDataType)
            .Format(paraFormat)
            .UnknownShapeFormat(paraFormat)
            .AutoContiguous();
        this->Input("lr")
            .ParamType(REQUIRED)
            .DataType(scalarDataType)
            .Format(paraFormat)
            .UnknownShapeFormat(paraFormat)
            .AutoContiguous();
        this->Input("beta1")
            .ParamType(REQUIRED)
            .DataType(scalarDataType)
            .Format(paraFormat)
            .UnknownShapeFormat(paraFormat)
            .AutoContiguous();
        this->Input("beta2")
            .ParamType(REQUIRED)
            .DataType(scalarDataType)
            .Format(paraFormat)
            .UnknownShapeFormat(paraFormat)
            .AutoContiguous();
        this->Input("epsilon")
            .ParamType(REQUIRED)
            .DataType(scalarDataType)
            .Format(paraFormat)
            .UnknownShapeFormat(paraFormat)
            .AutoContiguous();
        this->Input("grad")
            .ParamType(REQUIRED)
            .DataType(dataType)
            .Format(dataFormat)
            .UnknownShapeFormat(dataFormat)
            .AutoContiguous();
        this->Output("var")
            .ParamType(REQUIRED)
            .DataType(dataType)
            .Format(dataFormat)
            .UnknownShapeFormat(dataFormat)
            .AutoContiguous();

        this->Attr("use_locking").AttrType(OPTIONAL).Bool(false);
        this->Attr("use_nesterov").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("opFile.value", "apply_adam_apt");
        this->AICore().AddConfig("ascend950", aicore_config);
    }
};

OP_ADD(ApplyAdam);
} // namespace ops
