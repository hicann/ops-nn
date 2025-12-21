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
 * \file quant_conv3d_def.cpp
 * \brief
 */

#include <cstdint>
#include "register/op_def_registry.h"
namespace ops {
static const std::vector<ge::DataType> quantConv3dFmpDataType = {
    ge::DT_INT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN,
    ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_INT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8,
    ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
    ge::DT_FLOAT8_E4M3FN
};
static const std::vector<ge::DataType> quantConv3dWeightDataType = {
    ge::DT_INT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN,
    ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_INT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8,
    ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
    ge::DT_FLOAT8_E4M3FN
};
static const std::vector<ge::DataType> quantConv3dScaleDataType = {
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
    ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64
};
static const std::vector<ge::DataType> quantConv3dBiasDataType = {
    ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT
};
static const std::vector<ge::DataType> quantConv3dOffsetDataType = {
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT
};
static const std::vector<ge::DataType> quantConv3dOutputDataType = {
    ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_HIFLOAT8, ge::DT_FLOAT,
    ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16,
    ge::DT_BF16, ge::DT_HIFLOAT8, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN
};
static const std::vector<ge::Format> quantConv3dNCDHWFormat = {
    ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
    ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW
};
static const std::vector<ge::Format> quantConv3dNDFormat = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND
};
class QuantConv3D : public OpDef {
public:
    explicit QuantConv3D(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(quantConv3dFmpDataType)
            .Format(quantConv3dNCDHWFormat)
            .UnknownShapeFormat(quantConv3dNCDHWFormat);
        this->Input("filter")
            .ParamType(REQUIRED)
            .DataType(quantConv3dWeightDataType)
            .Format(quantConv3dNCDHWFormat)
            .UnknownShapeFormat(quantConv3dNCDHWFormat);
        this->Input("scale")
            .ParamType(REQUIRED)
            .DataType(quantConv3dScaleDataType)
            .Format(quantConv3dNDFormat)
            .UnknownShapeFormat(quantConv3dNDFormat);
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType(quantConv3dBiasDataType)
            .Format(quantConv3dNDFormat)
            .UnknownShapeFormat(quantConv3dNDFormat);
        this->Input("offset")
            .ParamType(OPTIONAL)
            .DataType(quantConv3dOffsetDataType)
            .Format(quantConv3dNCDHWFormat)
            .UnknownShapeFormat(quantConv3dNCDHWFormat);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(quantConv3dOutputDataType)
            .Format(quantConv3dNCDHWFormat)
            .UnknownShapeFormat(quantConv3dNCDHWFormat);

        this->Attr("dtype").AttrType(REQUIRED).Int(); // output dtype
        this->Attr("strides").AttrType(REQUIRED).ListInt();
        this->Attr("pads").AttrType(OPTIONAL).ListInt({0, 0, 0, 0, 0, 0});
        this->Attr("dilations").AttrType(OPTIONAL).ListInt({1, 1, 1, 1, 1});
        this->Attr("groups").AttrType(OPTIONAL).Int(1);
        this->Attr("data_format").AttrType(OPTIONAL).String("NCDHW");
        this->Attr("offset_x").AttrType(OPTIONAL).Int(0);
        this->Attr("round_mode").AttrType(OPTIONAL).String("rint");
        this->Attr("pad_mode").AttrType(OPTIONAL).String("SPECIFIC");

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
                     .DynamicFormatFlag(true)
                     .DynamicRankSupportFlag(true)
                     .DynamicShapeSupportFlag(true)
                     .NeedCheckSupportFlag(false)
                     .PrecisionReduceFlag(true)
                     .ExtendCfgInfo("opFile.value", "quant_conv3d")
                     .ExtendCfgInfo("opInterface.value", "quant_conv3d")
                     .ExtendCfgInfo("jitCompile.flag", "false");

        this->AICore().AddConfig("ascend910_95", aicore_config);
        this->AICore().AddConfig("ascend910_55", aicore_config);
    }
};

OP_ADD(QuantConv3D);
}