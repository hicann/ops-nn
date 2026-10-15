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

#include <cstring>
#include "register/op_def_registry.h"
namespace ops {
static const std::map<std::string, std::vector<ge::DataType>> quantConv3dFmpDataType = {
    {"ascend950",
     {ge::DT_INT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN,
      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_INT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8,
      ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
      ge::DT_FLOAT8_E4M3FN}},
    {"ascend960dt",
     {ge::DT_INT8,          ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,
      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_INT8,
      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E4M3FN,
      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_INT8,          ge::DT_HIFLOAT8,
      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_INT8,          ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,
      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
      ge::DT_FLOAT8_E4M3FN}}};
static const std::map<std::string, std::vector<ge::DataType>> quantConv3dWeightDataType = {
    {"ascend950",
     {ge::DT_INT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN,
      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_INT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8,
      ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
      ge::DT_FLOAT8_E4M3FN}},
    {"ascend960dt",
     {ge::DT_INT8,          ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,
      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_INT8,
      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E4M3FN,
      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_INT8,          ge::DT_HIFLOAT8,
      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_INT8,          ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,
      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
      ge::DT_FLOAT8_E4M3FN}}};
static const std::map<std::string, std::vector<ge::DataType>> quantConv3dScaleDataType = {
    {"ascend950",
     {ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
      ge::DT_INT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
      ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64}},
    {"ascend960dt", {ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,
                     ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                     ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                     ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,
                     ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                     ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64}}};
static const std::map<std::string, std::vector<ge::DataType>> quantConv3dBiasDataType = {
    {"ascend950",
     {ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
      ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
      ge::DT_FLOAT, ge::DT_FLOAT}},
    {"ascend960dt",
     {ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
      ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
      ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
      ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
      ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT}}};
static const std::map<std::string, std::vector<ge::DataType>> quantConv3dOffsetDataType = {
    {"ascend950",
     {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
      ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
      ge::DT_FLOAT, ge::DT_FLOAT}},
    {"ascend960dt",
     {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
      ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
      ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
      ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
      ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT}}};
static const std::map<std::string, std::vector<ge::DataType>> quantConv3dOutputDataType = {
    {"ascend950",
     {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_HIFLOAT8, ge::DT_FLOAT, ge::DT_FLOAT16,
      ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_HIFLOAT8,
      ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN}},
    {"ascend960dt",
     {ge::DT_FLOAT16, ge::DT_FLOAT,    ge::DT_FLOAT16,       ge::DT_BF16,    ge::DT_HIFLOAT8, ge::DT_FLOAT,
      ge::DT_FLOAT16, ge::DT_BF16,     ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT16, ge::DT_FLOAT,    ge::DT_FLOAT16,
      ge::DT_BF16,    ge::DT_HIFLOAT8, ge::DT_FLOAT,         ge::DT_FLOAT16, ge::DT_BF16,     ge::DT_FLOAT8_E4M3FN,
      ge::DT_FLOAT16, ge::DT_FLOAT,    ge::DT_FLOAT16,       ge::DT_BF16,    ge::DT_HIFLOAT8, ge::DT_FLOAT,
      ge::DT_FLOAT16, ge::DT_BF16,     ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT16, ge::DT_FLOAT,    ge::DT_FLOAT16,
      ge::DT_BF16,    ge::DT_HIFLOAT8, ge::DT_FLOAT,         ge::DT_FLOAT16, ge::DT_BF16,     ge::DT_FLOAT8_E4M3FN}}};
static const std::map<std::string, std::vector<ge::Format>> quantConv3dNCDHWFormat = {
    {"ascend950",
     {ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
      ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
      ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW}},
    {"ascend960dt",
     {ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
      ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
      ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
      ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
      ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
      ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW}}};
static const std::map<std::string, std::vector<ge::Format>> quantConv3dNDFormat = {
    {"ascend950",
     {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
      ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
      ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}},
    {"ascend960dt", {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}}};
static const std::map<std::string, std::vector<ge::Format>> quantConv3dweightFormat = {
    {"ascend950",
     {ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
      ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
      ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW}},
    {"ascend960dt",
     {ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
      ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
      ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
      ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC,
      ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC,
      ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC}}};
class QuantConv3D : public OpDef {
public:
    explicit QuantConv3D(const char* name) : OpDef(name)
    {
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

        SetAscendConfig(aicore_config, "ascend950");
        SetAscendConfig(aicore_config, "ascend950", "ascend350");
        SetAscendConfig(aicore_config, "ascend960dt");
        SetAscendConfig(aicore_config, "ascend960dt", "ascend960pr");
    }

private:
    void SetAscendConfig(OpAICoreConfig& aicoreConfig, const char* version, const char* dst_version = "default")
    {
        aicoreConfig.Input("x")
            .ParamType(REQUIRED)
            .DataType(quantConv3dFmpDataType.find(version)->second)
            .Format(quantConv3dNCDHWFormat.find(version)->second)
            .UnknownShapeFormat(quantConv3dNCDHWFormat.find(version)->second);
        aicoreConfig.Input("filter")
            .ParamType(REQUIRED)
            .DataType(quantConv3dWeightDataType.find(version)->second)
            .Format(quantConv3dweightFormat.find(version)->second)
            .UnknownShapeFormat(quantConv3dNCDHWFormat.find(version)->second);
        aicoreConfig.Input("scale")
            .ParamType(REQUIRED)
            .DataType(quantConv3dScaleDataType.find(version)->second)
            .Format(quantConv3dNDFormat.find(version)->second)
            .UnknownShapeFormat(quantConv3dNDFormat.find(version)->second);
        aicoreConfig.Input("bias")
            .ParamType(OPTIONAL)
            .DataType(quantConv3dBiasDataType.find(version)->second)
            .Format(quantConv3dNDFormat.find(version)->second)
            .UnknownShapeFormat(quantConv3dNDFormat.find(version)->second);
        aicoreConfig.Input("offset")
            .ParamType(OPTIONAL)
            .DataType(quantConv3dOffsetDataType.find(version)->second)
            .Format(quantConv3dNCDHWFormat.find(version)->second)
            .UnknownShapeFormat(quantConv3dNCDHWFormat.find(version)->second);
        aicoreConfig.Output("y")
            .ParamType(REQUIRED)
            .DataType(quantConv3dOutputDataType.find(version)->second)
            .Format(quantConv3dNCDHWFormat.find(version)->second)
            .UnknownShapeFormat(quantConv3dNCDHWFormat.find(version)->second);

        if (strcmp(dst_version, "default") != 0) {
            this->AICore().AddConfig(dst_version, aicoreConfig);
        } else {
            this->AICore().AddConfig(version, aicoreConfig);
        }
    }
};

OP_ADD(QuantConv3D);
} // namespace ops
