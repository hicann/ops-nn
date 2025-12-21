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
 * \file add_rms_norm_quant_v2_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {

class AddRmsNormQuantV2 : public OpDef {
#define DTYPE_X                                  \
    {                                            \
        ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16 \
    }
#define DTYPE_SCALE                             \
    {                                           \
        ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT \
    }
#define DTYPE_OFFSET                            \
    {                                           \
        ge::DT_INT32, ge::DT_BF16, ge::DT_INT32 \
    }
#define DTYPE_QUANT                           \
    {                                         \
        ge::DT_INT8, ge::DT_INT8, ge::DT_INT8 \
    }
#define FORMAT                                      \
    {                                               \
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND \
    }

public:
    explicit AddRmsNormQuantV2(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType(DTYPE_X)
            .Format(FORMAT)
            .UnknownShapeFormat(FORMAT)
            .AutoContiguous();
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType(DTYPE_X)
            .Format(FORMAT)
            .UnknownShapeFormat(FORMAT)
            .AutoContiguous();
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType(DTYPE_X)
            .Format(FORMAT)
            .UnknownShapeFormat(FORMAT)
            .AutoContiguous();
        this->Input("scales1")
            .ParamType(REQUIRED)
            .DataType(DTYPE_SCALE)
            .Format(FORMAT)
            .UnknownShapeFormat(FORMAT)
            .AutoContiguous();
        this->Input("scales2")
            .ParamType(OPTIONAL)
            .DataType(DTYPE_SCALE)
            .Format(FORMAT)
            .UnknownShapeFormat(FORMAT)
            .AutoContiguous();
        this->Input("zero_points1")
            .ParamType(OPTIONAL)
            .DataType(DTYPE_OFFSET)
            .Format(FORMAT)
            .UnknownShapeFormat(FORMAT)
            .AutoContiguous();
        this->Input("zero_points2")
            .ParamType(OPTIONAL)
            .DataType(DTYPE_OFFSET)
            .Format(FORMAT)
            .UnknownShapeFormat(FORMAT)
            .AutoContiguous();
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType(DTYPE_X)
            .Format(FORMAT)
            .UnknownShapeFormat(FORMAT)
            .AutoContiguous();
        this->Output("y1")
            .ParamType(REQUIRED)
            .DataType(DTYPE_QUANT)
            .Format(FORMAT)
            .UnknownShapeFormat(FORMAT)
            .AutoContiguous();
        this->Output("y2")
            .ParamType(REQUIRED)
            .DataType(DTYPE_QUANT)
            .Format(FORMAT)
            .UnknownShapeFormat(FORMAT)
            .AutoContiguous();
        this->Output("x")
            .ParamType(OPTIONAL)
            .DataType(DTYPE_X)
            .Format(FORMAT)
            .UnknownShapeFormat(FORMAT)
            .AutoContiguous();
        this->Output("resOut")
            .ParamType(OPTIONAL)
            .DataType(DTYPE_X)
            .Format(FORMAT)
            .UnknownShapeFormat(FORMAT)
            .AutoContiguous();
        this->Attr("axis").AttrType(OPTIONAL).Int(-1);
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-6);
        this->Attr("div_mode").AttrType(OPTIONAL).Bool(true);

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");

        OpAICoreConfig config310P;
        config310P.Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        config310P.Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        config310P.Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        config310P.Input("scales1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        config310P.Input("scales2")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        config310P.Input("zero_points1")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        config310P.Input("zero_points2")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        config310P.Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        config310P.Output("y1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        config310P.Output("y2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        config310P.Output("x")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        config310P.Output("resOut")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        config310P.DynamicCompileStaticFlag(true).DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true);
        this->AICore().AddConfig("ascend310p", config310P);
    }
};
OP_ADD(AddRmsNormQuantV2);
} // namespace ops