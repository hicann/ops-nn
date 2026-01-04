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
 * \file avg_pool3_d_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {

class AvgPool3D : public OpDef {
public:
    const std::vector<ge::DataType> AvgPool3DXDataType = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
    const std::vector<ge::Format> AvgPool3DXFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
    explicit AvgPool3D(const char* name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("ksize")
            .AttrType(REQUIRED)
            .ListInt();
        this->Attr("strides")
            .AttrType(REQUIRED)
            .ListInt();
        this->Attr("pads")
            .AttrType(REQUIRED)
            .ListInt();
        this->Attr("ceil_mode")
            .AttrType(OPTIONAL)
            .Bool(false);
        this->Attr("count_include_pad")
            .AttrType(OPTIONAL)
            .Bool(true);
        this->Attr("divisor_override")
            .AttrType(OPTIONAL)
            .Int(0);
        this->Attr("data_format")
            .AttrType(OPTIONAL)
            .String("NDHWC");

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
              .DynamicRankSupportFlag(true)
              .DynamicShapeSupportFlag(true)
              .NeedCheckSupportFlag(false);
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);

        OpAICoreConfig aiCoreConfig;
        aiCoreConfig.Input("x")
             .ParamType(REQUIRED)
             .DataType(AvgPool3DXDataType)
             .Format(AvgPool3DXFormat)
             .UnknownShapeFormat(AvgPool3DXFormat)
             .AutoContiguous();
        aiCoreConfig.Output("y")
             .ParamType(REQUIRED)
             .DataType(AvgPool3DXDataType)
             .Format(AvgPool3DXFormat)
             .UnknownShapeFormat(AvgPool3DXFormat)
             .AutoContiguous();

        aiCoreConfig.DynamicCompileStaticFlag(true)
             .DynamicFormatFlag(false)
             .DynamicRankSupportFlag(true)
             .DynamicShapeSupportFlag(true)
             .NeedCheckSupportFlag(false)
             .PrecisionReduceFlag(true)
             .ExtendCfgInfo("opFile.value", "avg_pool3_d_apt");
        this->AICore().AddConfig("ascend910_95", aiCoreConfig);
        this->AICore().AddConfig("mc62cm12a", aiCoreConfig);

        OpAICoreConfig config_kirin = GetKirinCoreConfig();
        this->AICore().AddConfig("kirinx90", config_kirin);
    }

private:
    OpAICoreConfig GetKirinCoreConfig() const
    {
        OpAICoreConfig config_kirin;
        config_kirin.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        config_kirin.Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        config_kirin.Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        return config_kirin;
    }
};

OP_ADD(AvgPool3D);
}  // namespace ops
