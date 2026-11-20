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

#include "register/op_def_registry.h"
#include "op_common/op_host/util/op_arch_helper.h"
namespace ops {
static const std::map<std::string, std::vector<ge::DataType>> quantConv3dFmpDataType = {
    {"dav3510",
     {ge::DT_INT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN,
      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_INT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8,
      ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
      ge::DT_FLOAT8_E4M3FN}},
    {"dav9201",
     {ge::DT_INT8,          ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,
      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_INT8,
      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E4M3FN,
      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_INT8,          ge::DT_HIFLOAT8,
      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_INT8,          ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,
      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
      ge::DT_FLOAT8_E4M3FN}}};
static const std::map<std::string, std::vector<ge::DataType>> quantConv3dWeightDataType = {
    {"dav3510",
     {ge::DT_INT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN,
      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_INT8, ge::DT_HIFLOAT8, ge::DT_HIFLOAT8,
      ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
      ge::DT_FLOAT8_E4M3FN}},
    {"dav9201",
     {ge::DT_INT8,          ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,
      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_INT8,
      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E4M3FN,
      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_INT8,          ge::DT_HIFLOAT8,
      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_INT8,          ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,
      ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN,
      ge::DT_FLOAT8_E4M3FN}}};
static const std::map<std::string, std::vector<ge::DataType>> quantConv3dScaleDataType = {
    {"dav3510",
     {ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
      ge::DT_INT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
      ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64}},
    {"dav9201", {ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,
                 ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                 ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                 ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,
                 ge::DT_INT64,  ge::DT_INT64,  ge::DT_INT64,  ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64,
                 ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64, ge::DT_UINT64}}};
static const std::map<std::string, std::vector<ge::DataType>> quantConv3dBiasDataType = {
    {"dav3510",
     {ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
      ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
      ge::DT_FLOAT, ge::DT_FLOAT}},
    {"dav9201",
     {ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
      ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
      ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
      ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
      ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT}}};
static const std::map<std::string, std::vector<ge::DataType>> quantConv3dOffsetDataType = {
    {"dav3510",
     {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
      ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
      ge::DT_FLOAT, ge::DT_FLOAT}},
    {"dav9201",
     {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
      ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
      ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
      ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
      ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT}}};
static const std::map<std::string, std::vector<ge::DataType>> quantConv3dOutputDataType = {
    {"dav3510",
     {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_HIFLOAT8, ge::DT_FLOAT, ge::DT_FLOAT16,
      ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_HIFLOAT8,
      ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT8_E4M3FN}},
    {"dav9201",
     {ge::DT_FLOAT16, ge::DT_FLOAT,    ge::DT_FLOAT16,       ge::DT_BF16,    ge::DT_HIFLOAT8, ge::DT_FLOAT,
      ge::DT_FLOAT16, ge::DT_BF16,     ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT16, ge::DT_FLOAT,    ge::DT_FLOAT16,
      ge::DT_BF16,    ge::DT_HIFLOAT8, ge::DT_FLOAT,         ge::DT_FLOAT16, ge::DT_BF16,     ge::DT_FLOAT8_E4M3FN,
      ge::DT_FLOAT16, ge::DT_FLOAT,    ge::DT_FLOAT16,       ge::DT_BF16,    ge::DT_HIFLOAT8, ge::DT_FLOAT,
      ge::DT_FLOAT16, ge::DT_BF16,     ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT16, ge::DT_FLOAT,    ge::DT_FLOAT16,
      ge::DT_BF16,    ge::DT_HIFLOAT8, ge::DT_FLOAT,         ge::DT_FLOAT16, ge::DT_BF16,     ge::DT_FLOAT8_E4M3FN}}};
static const std::map<std::string, std::vector<ge::Format>> quantConv3dNCDHWFormat = {
    {"dav3510",
     {ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
      ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
      ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW}},
    {"dav9201",
     {ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
      ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
      ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
      ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
      ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
      ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW}}};
static const std::map<std::string, std::vector<ge::Format>> quantConv3dNDFormat = {
    {"dav3510",
     {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
      ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
      ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}},
    {"dav9201", {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}}};
static const std::map<std::string, std::vector<ge::Format>> quantConv3dweightFormat = {
    {"dav3510",
     {ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
      ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
      ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW}},
    {"dav9201",
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

        SetArchConfigInputs(aicore_config, "dav3510");
        opbase::ArchConfigHelper::AddConfigByArch(this->AICore(), Ops::Base::DAV_3510, aicore_config);
        SetArchConfigInputs(aicore_config, "dav9201");
        opbase::ArchConfigHelper::AddConfigByArch(this->AICore(), Ops::Base::DAV_9201, aicore_config);
        opbase::ArchConfigHelper::AddConfigByArch(this->AICore(), Ops::Base::DAV_9202, aicore_config);
    }

private:
    void SetArchConfigInputs(OpAICoreConfig& aicoreConfig, const char* version)
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
    }
};

OP_ADD(QuantConv3D);
} // namespace ops
