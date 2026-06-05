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
 * \file rms_norm_grad_quant_def.cpp
 * \brief RmsNormGradQuant cpp file
 */

#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> dyDataType950 = {
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,
    ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_BF16,    ge::DT_FLOAT16,
    ge::DT_BF16,
    ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16,    ge::DT_BF16,
    ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_BF16,    ge::DT_FLOAT16,
    ge::DT_BF16
};
static const std::vector<ge::DataType> dxDataType950 = {
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,    ge::DT_INT8,
    ge::DT_INT8,
    ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8,
    ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8,
    ge::DT_HIFLOAT8
};
static const std::vector<ge::DataType> rstdDataType950 = {
    ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
    ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
    ge::DT_FLOAT,
    ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
    ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
    ge::DT_FLOAT
};
static const std::vector<ge::DataType> gammaDataType950 = {
    ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
    ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_BF16,    ge::DT_FLOAT16,
    ge::DT_BF16,
    ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,
    ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_BF16,    ge::DT_FLOAT16,
    ge::DT_BF16
};
static const std::vector<ge::DataType> scalesXDataType950 = {
    ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_BF16,
    ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT16,
    ge::DT_BF16,
    ge::DT_FLOAT,   ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_BF16,
    ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT,   ge::DT_FLOAT16,
    ge::DT_BF16
};
static const std::vector<ge::DataType> offsetXDataType950 = {
    ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,
    ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,
    ge::DT_INT32,
    ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,
    ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT32,
    ge::DT_INT32
};
static const std::vector<ge::Format> format950 = {
    ge::FORMAT_ND,  ge::FORMAT_ND,  ge::FORMAT_ND,  ge::FORMAT_ND,
    ge::FORMAT_ND,  ge::FORMAT_ND,  ge::FORMAT_ND,  ge::FORMAT_ND,
    ge::FORMAT_ND,
    ge::FORMAT_ND,  ge::FORMAT_ND,  ge::FORMAT_ND,  ge::FORMAT_ND,
    ge::FORMAT_ND,  ge::FORMAT_ND,  ge::FORMAT_ND,  ge::FORMAT_ND,
    ge::FORMAT_ND
};

class RmsNormGradQuant : public OpDef {
public:
    explicit RmsNormGradQuant(const char* name) : OpDef(name)
    {
        this->Input("dy")
            .ParamType(REQUIRED)
            .DataType(dyDataType950)
            .Format(format950)
            .UnknownShapeFormat(format950)
            .AutoContiguous();
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(dyDataType950)
            .Format(format950)
            .UnknownShapeFormat(format950)
            .AutoContiguous();
        this->Input("rstd")
            .ParamType(REQUIRED)
            .DataType(rstdDataType950)
            .Format(format950)
            .UnknownShapeFormat(format950)
            .AutoContiguous();
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType(gammaDataType950)
            .Format(format950)
            .UnknownShapeFormat(format950)
            .AutoContiguous();
        this->Input("scales_x")
            .ParamType(REQUIRED)
            .DataType(scalesXDataType950)
            .Format(format950)
            .UnknownShapeFormat(format950)
            .AutoContiguous();
        this->Input("offset_x")
            .ParamType(OPTIONAL)
            .DataType(offsetXDataType950)
            .Format(format950)
            .UnknownShapeFormat(format950)
            .AutoContiguous();
        this->Output("dx")
            .ParamType(REQUIRED)
            .DataType(dxDataType950)
            .Format(format950)
            .UnknownShapeFormat(format950);
        this->Output("dgamma")
            .ParamType(REQUIRED)
            .DataType(rstdDataType950)
            .Format(format950)
            .UnknownShapeFormat(format950);

        this->Attr("quant_mode").AttrType(REQUIRED).String("static");
        this->Attr("div_mode").AttrType(REQUIRED).Bool(true);
        this->Attr("dst_type").AttrType(OPTIONAL).Int(ge::DT_INT8);

        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(RmsNormGradQuant);
} // namespace ops
