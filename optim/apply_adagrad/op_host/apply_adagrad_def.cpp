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
 * \file apply_adagrad_def.cpp
 * \brief apply_adagrad def
 */

#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> dataType = {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT};
static const std::vector<ge::Format> dataFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class ApplyAdagrad : public OpDef {
public:
    explicit ApplyAdagrad(const char* name) : OpDef(name)
    {
        this->Input("var")
            .ParamType(REQUIRED)
            .DataType(dataType)
            .Format(dataFormat)
            .UnknownShapeFormat(dataFormat)
            .AutoContiguous();
        this->Input("accum")
            .ParamType(REQUIRED)
            .DataType(dataType)
            .Format(dataFormat)
            .UnknownShapeFormat(dataFormat)
            .AutoContiguous();
        this->Input("lr")
            .ParamType(REQUIRED)
            .DataType(dataType)
            .Format(dataFormat)
            .UnknownShapeFormat(dataFormat)
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
        this->Attr("update_slots").AttrType(OPTIONAL).Bool(true);
        this->Attr("use_locking").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .PrecisionReduceFlag(false);
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(ApplyAdagrad);
} // namespace ops
