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
 * \file conv3d_v2_def.cpp
 * \brief
 */

#include <cstring>
#include "register/op_def_registry.h"

namespace ops {
static const std::map<std::string, std::vector<ge::DataType>> conv3dv2FmapDataType = {
    {"ascend910b", {ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT,
                    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8}},
    {"ascend910_95", {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_HIFLOAT8,
                      ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8,
                      ge::DT_INT8, ge::DT_INT8, ge::DT_INT8}},
    {"ascend910_55", {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_HIFLOAT8}}
};
static const std::map<std::string, std::vector<ge::DataType>> conv3dv2WeightDataType = {
    {"ascend910b", {ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT,
                    ge::DT_FLOAT16, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8}},
    {"ascend910_95", {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_HIFLOAT8,
                      ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8,
                      ge::DT_INT8, ge::DT_INT8, ge::DT_INT8}},
    {"ascend910_55", {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_HIFLOAT8}}
};
static const std::map<std::string, std::vector<ge::DataType>> conv3dv2BiasDataType = {
    {"ascend910b", {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT,
                    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16}},
    {"ascend910_95", {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT,
                      ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16,
                      ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT}},
    {"ascend910_55", {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT}}
};
static const std::map<std::string, std::vector<ge::DataType>> conv3dv2ScaleAndOffsetDataType = {
    {"ascend910b", {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                    ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT}},
    {"ascend910_95", {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                      ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                      ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT}},
    {"ascend910_55", {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT}}
};
static const std::map<std::string, std::vector<ge::DataType>> conv3dv2OffsetWDataType = {
    {"ascend910b", {ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
                    ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8}},
    {"ascend910_95", {ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
                      ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
                      ge::DT_INT8, ge::DT_INT8, ge::DT_INT8}},
    {"ascend910_55", {ge::DT_INT8, ge::DT_INT8, ge::DT_INT8, ge::DT_INT8}}
};
static const std::map<std::string, std::vector<ge::DataType>> conv3dv2OutputDataType = {
    {"ascend910b", {ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
                    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT16}},
    {"ascend910_95", {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_HIFLOAT8,
                      ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16,
                      ge::DT_FLOAT16, ge::DT_BF16, ge::DT_BF16}},
    {"ascend910_55", {ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_HIFLOAT8}}
};
static const std::map<std::string, std::vector<ge::Format>> conv3dv2FmapFormat = {
    {"ascend910b", {ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW,
                    ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0,
                    ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0}},
    {"ascend910_95", {ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
                      ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NCDHW,
                      ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW}},
    {"ascend910_55", {ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW}}
};
static const std::map<std::string, std::vector<ge::Format>> conv3dv2WeightFormat = {
    {"ascend910b", {ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_NCDHW,
                    ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_FRACTAL_Z_3D,
                    ge::FORMAT_FRACTAL_Z_3D, ge::FORMAT_FRACTAL_Z_3D}},
    {"ascend910_95", {ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
                      ge::FORMAT_DHWCN, ge::FORMAT_DHWCN, ge::FORMAT_DHWCN, ge::FORMAT_NCDHW,
                      ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW}},
    {"ascend910_55", {ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW}}
};
static const std::map<std::string, std::vector<ge::Format>> conv3dv2BiasFormat = {
    {"ascend910b", {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}},
    {"ascend910_95", {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                      ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                      ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}},
    {"ascend910_55", {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}}
};
static const std::map<std::string, std::vector<ge::Format>> conv3dv2ScaleAndOffsetFormat = {
    {"ascend910b", {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}},
    {"ascend910_95", {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                      ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                      ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}},
    {"ascend910_55", {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}}
};
static const std::map<std::string, std::vector<ge::Format>> conv3dv2OffsetWFormat = {
    {"ascend910b", {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}},
    {"ascend910_95", {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                      ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                      ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}},
    {"ascend910_55", {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}}
};
static const std::map<std::string, std::vector<ge::Format>> conv3dv2OutputFormat = {
    {"ascend910b", {ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0, ge::FORMAT_NCDHW,
                    ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0,
                    ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0}},
    {"ascend910_95", {ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
                      ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC,
                      ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC}},
    {"ascend910_55", {ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW}}
};

class Conv3DV2 : public OpDef {
public:
    explicit Conv3DV2(const char *name) : OpDef(name)
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
        this->Input("scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_MAX})
            .Format({ge::FORMAT_END})
            .UnknownShapeFormat({ge::FORMAT_END});
        this->Input("offset")
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
        this->Attr("pads").AttrType(OPTIONAL).ListInt({0, 0, 0, 0, 0, 0});
        this->Attr("dilations").AttrType(OPTIONAL).ListInt({1, 1, 1, 1, 1});
        this->Attr("groups").AttrType(OPTIONAL).Int(1);
        this->Attr("data_format").AttrType(OPTIONAL).String("NCDHW");
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
                        .ExtendCfgInfo("opFile.value", "conv3d_v2")
                        .ExtendCfgInfo("opInterface.value", "conv3dv2")
                        .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
                        .ExtendCfgInfo("jitCompile.flag", "false");

        SetAscendConfig(aicoreConfig, "ascend910b");
        SetAscendConfig(aicoreConfig, "ascend910b", "ascend910_93");

        OpAICoreConfig aicoreConfig95;
        aicoreConfig95.DynamicCompileStaticFlag(true)
                        .DynamicFormatFlag(true)
                        .DynamicRankSupportFlag(true)
                        .DynamicShapeSupportFlag(true)
                        .NeedCheckSupportFlag(false)
                        .PrecisionReduceFlag(true)
                        .ExtendCfgInfo("opFile.value", "conv3d_v2")
                        .ExtendCfgInfo("opInterface.value", "conv3dv2")
                        .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
                        .ExtendCfgInfo("jitCompile.flag", "false");

        SetAscendConfig(aicoreConfig95, "ascend910_95");
        SetAscendConfig(aicoreConfig95, "ascend910_55");
    }

private:
    void SetAscendConfig(OpAICoreConfig& aicoreConfig, const char* version, const char* dst_version = "default")
    {
        aicoreConfig.Input("x")
            .ParamType(REQUIRED)
            .DataType(conv3dv2FmapDataType.find(version)->second)
            .Format(conv3dv2FmapFormat.find(version)->second)
            .UnknownShapeFormat(conv3dv2FmapFormat.find(version)->second);
        aicoreConfig.Input("filter")
            .ParamType(REQUIRED)
            .DataType(conv3dv2WeightDataType.find(version)->second)
            .Format(conv3dv2WeightFormat.find(version)->second)
            .UnknownShapeFormat(conv3dv2WeightFormat.find(version)->second);
        aicoreConfig.Input("bias")
            .ParamType(OPTIONAL)
            .DataType(conv3dv2BiasDataType.find(version)->second)
            .Format(conv3dv2BiasFormat.find(version)->second)
            .UnknownShapeFormat(conv3dv2BiasFormat.find(version)->second);
        aicoreConfig.Input("scale")
            .ParamType(OPTIONAL)
            .DataType(conv3dv2ScaleAndOffsetDataType.find(version)->second)
            .Format(conv3dv2ScaleAndOffsetFormat.find(version)->second)
            .UnknownShapeFormat(conv3dv2ScaleAndOffsetFormat.find(version)->second);
        aicoreConfig.Input("offset")
            .ParamType(OPTIONAL)
            .DataType(conv3dv2ScaleAndOffsetDataType.find(version)->second)
            .Format(conv3dv2ScaleAndOffsetFormat.find(version)->second)
            .UnknownShapeFormat(conv3dv2ScaleAndOffsetFormat.find(version)->second);
        aicoreConfig.Input("offset_w")
            .ParamType(OPTIONAL)
            .DataType(conv3dv2OffsetWDataType.find(version)->second)
            .Format(conv3dv2OffsetWFormat.find(version)->second)
            .UnknownShapeFormat(conv3dv2OffsetWFormat.find(version)->second);
        aicoreConfig.Output("y")
            .ParamType(REQUIRED)
            .DataType(conv3dv2OutputDataType.find(version)->second)
            .Format(conv3dv2OutputFormat.find(version)->second)
            .UnknownShapeFormat(conv3dv2OutputFormat.find(version)->second);

        if (strcmp(dst_version, "default") != 0) {
            this->AICore().AddConfig(dst_version, aicoreConfig);
        } else {
            this->AICore().AddConfig(version, aicoreConfig);
        }
    }
};

OP_ADD(Conv3DV2);
}