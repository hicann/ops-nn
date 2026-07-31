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
 * \file quant_conv2d_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"
namespace ops {
static const std::vector<ge::DataType> quantConv2dFmpDataType = {
    ge::DT_INT8,          ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,
    ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_INT8,
    ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E4M3FN,
    ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN};
static const std::vector<ge::DataType> quantConv2dWeightDataType = {
    ge::DT_INT8,          ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,
    ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_INT8,
    ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E4M3FN,
    ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN};
static const std::vector<ge::DataType> quantConv2dScaleDataType = {
    ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,
    ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
    ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64};
static const std::vector<ge::DataType> quantConv2dBiasDataType = {
    ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
static const std::vector<ge::DataType> quantConv2dOffsetDataType = {
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
static const std::vector<ge::DataType> quantConv2dOutputDataType = {
    ge::DT_FLOAT16, ge::DT_FLOAT,    ge::DT_FLOAT16,       ge::DT_BF16,    ge::DT_HIFLOAT8, ge::DT_FLOAT,
    ge::DT_FLOAT16, ge::DT_BF16,     ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT16, ge::DT_FLOAT,    ge::DT_FLOAT16,
    ge::DT_BF16,    ge::DT_HIFLOAT8, ge::DT_FLOAT,         ge::DT_FLOAT16, ge::DT_BF16,     ge::DT_FLOAT8_E4M3FN};
static const std::vector<ge::Format> quantConv2dNCHWFormat = {
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
    ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW};
static const std::vector<ge::Format> quantConv2dNDFormat = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
class QuantConv2D : public OpDef {
public:
    explicit QuantConv2D(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(quantConv2dFmpDataType)
            .Format(quantConv2dNCHWFormat)
            .UnknownShapeFormat(quantConv2dNCHWFormat);
        this->Input("filter")
            .ParamType(REQUIRED)
            .DataType(quantConv2dWeightDataType)
            .Format(quantConv2dNCHWFormat)
            .UnknownShapeFormat(quantConv2dNCHWFormat);
        this->Input("scale")
            .ParamType(REQUIRED)
            .DataType(quantConv2dScaleDataType)
            .Format(quantConv2dNDFormat)
            .UnknownShapeFormat(quantConv2dNDFormat);
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType(quantConv2dBiasDataType)
            .Format(quantConv2dNDFormat)
            .UnknownShapeFormat(quantConv2dNDFormat);
        this->Input("offset")
            .ParamType(OPTIONAL)
            .DataType(quantConv2dOffsetDataType)
            .Format(quantConv2dNCHWFormat)
            .UnknownShapeFormat(quantConv2dNCHWFormat);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(quantConv2dOutputDataType)
            .Format(quantConv2dNCHWFormat)
            .UnknownShapeFormat(quantConv2dNCHWFormat);

        this->Attr("dtype").AttrType(REQUIRED).Int(); // output dtype
        this->Attr("strides").AttrType(REQUIRED).ListInt();
        this->Attr("pads").AttrType(OPTIONAL).ListInt({0, 0, 0, 0});
        this->Attr("dilations").AttrType(OPTIONAL).ListInt({1, 1, 1, 1});
        this->Attr("groups").AttrType(OPTIONAL).Int(1);
        this->Attr("data_format").AttrType(OPTIONAL).String("NCHW");
        this->Attr("offset_x").AttrType(OPTIONAL).Int(0);
        this->Attr("round_mode").AttrType(OPTIONAL).String("rint");

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "quant_conv2d")
            .ExtendCfgInfo("opInterface.value", "quant_conv2d")
            .ExtendCfgInfo("jitCompile.flag", "false");

        this->AICore().AddConfig("ascend950", aicore_config);
    }
};

OP_ADD(QuantConv2D);
} // namespace ops
