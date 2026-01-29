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
 * \file conv2d_v2_def.cpp
 * \brief
 */

#include <cstring>
#include "register/op_def_registry.h"

namespace ops {
static const std::map<std::string, std::vector<ge::DataType>> conv2dv2FmapDataType = {
    {"ascend950", {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_HIFLOAT8,
                      ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT}},
    {"ascend910_55", {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_HIFLOAT8}},
    {"mc62cm12a", {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16}}
};
static const std::map<std::string, std::vector<ge::DataType>> conv2dv2WeightDataType = {
    {"ascend950", {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_HIFLOAT8,
                      ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT}},
    {"ascend910_55", {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_HIFLOAT8}},
    {"mc62cm12a", {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16}}
};
static const std::map<std::string, std::vector<ge::DataType>> conv2dv2BiasDataType = {
    {"ascend950", {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT,
                      ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT}},
    {"ascend910_55", {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT}},
    {"mc62cm12a", {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16}}
};
static const std::map<std::string, std::vector<ge::DataType>> conv2dv2OffsetWDataType = {
    {"ascend950", {ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
                      ge::DT_INT8, ge::DT_INT8, ge::DT_INT8}},
    {"ascend910_55", {ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8}},
    {"mc62cm12a", {ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8}}
};
static const std::map<std::string, std::vector<ge::DataType>> conv2dv2OutputDataType = {
    {"ascend950", {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_HIFLOAT8,
                      ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT}},
    {"ascend910_55", {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_HIFLOAT8}},
    {"mc62cm12a", {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16}}
};
static const std::map<std::string, std::vector<ge::Format>> conv2dV2FmapFormat = {
    {"ascend950", {ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
                      ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC}},
    {"ascend910_55", {ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW}},
    {"mc62cm12a", {ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC}}
};
static const std::map<std::string, std::vector<ge::Format>> conv2dV2WeightFormat = {
    {"ascend950", {ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
                      ge::FORMAT_HWCN, ge::FORMAT_HWCN, ge::FORMAT_HWCN}},
    {"ascend910_55", {ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW}},
    {"mc62cm12a", {ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z_C04, ge::FORMAT_FRACTAL_Z, ge::FORMAT_FRACTAL_Z_C04}}
};
static const std::map<std::string, std::vector<ge::Format>> conv2dV2BiasFormat = {
    {"ascend950", {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                      ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}},
    {"ascend910_55", {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}},
    {"mc62cm12a", {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}}
};
static const std::map<std::string, std::vector<ge::Format>> conv2dV2OffsetWFormat = {
    {"ascend950", {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                      ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}},
    {"ascend910_55", {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}},
    {"mc62cm12a", {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}}
};
static const std::map<std::string, std::vector<ge::Format>> conv2dV2OutputFormat = {
    {"ascend950", {ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
                      ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC}},
    {"ascend910_55", {ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW}},
    {"mc62cm12a", {ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC, ge::FORMAT_NHWC}}
};

class Conv2DV2 : public OpDef {
public:
    explicit Conv2DV2(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_MAX})
            .Format({ge::FORMAT_END})
            .UnknownShapeFormat({ge::FORMAT_END});
        this->Input("filter")
            .ParamType(REQUIRED)
            .DataType({ge::DT_MAX})
            .Format({ge::FORMAT_END})
            .UnknownShapeFormat({ge::FORMAT_END});
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_MAX})
            .Format({ge::FORMAT_END})
            .UnknownShapeFormat({ge::FORMAT_END});
        this->Input("offset_w")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_MAX})
            .Format({ge::FORMAT_END})
            .UnknownShapeFormat({ge::FORMAT_END});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_MAX})
            .Format({ge::FORMAT_END})
            .UnknownShapeFormat({ge::FORMAT_END});

        this->Attr("strides").AttrType(REQUIRED).ListInt();
        this->Attr("pads").AttrType(OPTIONAL).ListInt({0, 0, 0, 0});
        this->Attr("dilations").AttrType(OPTIONAL).ListInt({1, 1, 1, 1});
        this->Attr("groups").AttrType(OPTIONAL).Int(1);
        this->Attr("data_format").AttrType(OPTIONAL).String("NCHW");
        this->Attr("offset_x").AttrType(OPTIONAL).Int(0);
        this->Attr("pad_mode").AttrType(OPTIONAL).String("SPECIFIC");
        this->Attr("enable_hf32").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
                        .DynamicFormatFlag(true)
                        .DynamicRankSupportFlag(true)
                        .DynamicShapeSupportFlag(true)
                        .NeedCheckSupportFlag(false)
                        .PrecisionReduceFlag(true)
                        .ExtendCfgInfo("opFile.value", "conv2d_v2")
                        .ExtendCfgInfo("opInterface.value", "conv2dv2")
                        .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
                        .ExtendCfgInfo("jitCompile.flag", "false");

        SetAscendConfig(aicoreConfig, "ascend950");
        SetAscendConfig(aicoreConfig, "ascend910_55");
        SetAscendConfig(aicoreConfig, "mc62cm12a");
    }

private:
    void SetAscendConfig(OpAICoreConfig& aicoreConfig, const char* version, const char* dst_version = "default")
    {
        aicoreConfig.Input("x")
            .ParamType(REQUIRED)
            .DataType(conv2dv2FmapDataType.find(version)->second)
            .Format(conv2dV2FmapFormat.find(version)->second)
            .UnknownShapeFormat(conv2dV2FmapFormat.find(version)->second);
        aicoreConfig.Input("filter")
            .ParamType(REQUIRED)
            .DataType(conv2dv2WeightDataType.find(version)->second)
            .Format(conv2dV2WeightFormat.find(version)->second)
            .UnknownShapeFormat(conv2dV2WeightFormat.find(version)->second);
        aicoreConfig.Input("bias")
            .ParamType(OPTIONAL)
            .DataType(conv2dv2BiasDataType.find(version)->second)
            .Format(conv2dV2BiasFormat.find(version)->second)
            .UnknownShapeFormat(conv2dV2BiasFormat.find(version)->second);
        aicoreConfig.Input("offset_w")
            .ParamType(OPTIONAL)
            .DataType(conv2dv2OffsetWDataType.find(version)->second)
            .Format(conv2dV2OffsetWFormat.find(version)->second)
            .UnknownShapeFormat(conv2dV2OffsetWFormat.find(version)->second);
        aicoreConfig.Output("y")
            .ParamType(REQUIRED)
            .DataType(conv2dv2OutputDataType.find(version)->second)
            .Format(conv2dV2OutputFormat.find(version)->second)
            .UnknownShapeFormat(conv2dV2OutputFormat.find(version)->second);

        if (strcmp(dst_version, "default") != 0) {
            this->AICore().AddConfig(dst_version, aicoreConfig);
        } else {
            this->AICore().AddConfig(version, aicoreConfig);
        }
    }
};

OP_ADD(Conv2DV2);
}