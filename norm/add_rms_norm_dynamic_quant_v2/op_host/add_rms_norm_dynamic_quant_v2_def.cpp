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
 * \file add_rms_norm_dynamic_quant_v2_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> xDataType950 = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16,
                                                       ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16,
                                                       ge::DT_FLOAT16, ge::DT_BF16};
static const std::vector<ge::DataType> scalesOutDataType950 = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                                                               ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                                                               ge::DT_FLOAT, ge::DT_FLOAT};
static const std::vector<ge::DataType> yDataType950 = {
    ge::DT_INT8,     ge::DT_INT8,        ge::DT_INT4,        ge::DT_INT4,          ge::DT_HIFLOAT8,
    ge::DT_HIFLOAT8, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN};
static const std::vector<ge::DataType> y3DataType950 = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                                                        ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                                                        ge::DT_FLOAT, ge::DT_FLOAT};
static const std::vector<ge::Format> format950 = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                  ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                  ge::FORMAT_ND, ge::FORMAT_ND};

class AddRmsNormDynamicQuantV2 : public OpDef {
public:
    explicit AddRmsNormDynamicQuantV2(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("smooth_scale1")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("smooth_scale2")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("beta")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y3")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y4")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("scale1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("scale2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-6);
        this->Attr("output_mask").AttrType(OPTIONAL).ListBool({});
        this->Attr("dst_type").AttrType(OPTIONAL).Int(ge::DT_INT8);
        this->AICore().AddConfig("ascend910b");

        OpAICoreConfig config_kirin = GetKirinCoreConfig();
        this->AICore().AddConfig("kirinx90", config_kirin);
        this->AICore().AddConfig("kirin9030", config_kirin);

        OpAICoreConfig config_950 = Get950CoreConfig();
        this->AICore().AddConfig("ascend950", config_950);
    }

private:
    OpAICoreConfig Get950CoreConfig() const
    {
        OpAICoreConfig config_950;
        config_950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        config_950.Input("x1")
            .ParamType(REQUIRED)
            .DataType(xDataType950)
            .Format(format950)
            .UnknownShapeFormat(format950)
            .AutoContiguous();
        config_950.Input("x2")
            .ParamType(REQUIRED)
            .DataType(xDataType950)
            .Format(format950)
            .UnknownShapeFormat(format950)
            .AutoContiguous();
        config_950.Input("gamma")
            .ParamType(REQUIRED)
            .DataType(xDataType950)
            .Format(format950)
            .UnknownShapeFormat(format950)
            .AutoContiguous();
        config_950.Input("smooth_scale1")
            .ParamType(OPTIONAL)
            .DataType(xDataType950)
            .Format(format950)
            .UnknownShapeFormat(format950)
            .AutoContiguous();
        config_950.Input("smooth_scale2")
            .ParamType(OPTIONAL)
            .DataType(xDataType950)
            .Format(format950)
            .UnknownShapeFormat(format950)
            .AutoContiguous();
        config_950.Input("beta")
            .ParamType(OPTIONAL)
            .DataType(xDataType950)
            .Format(format950)
            .UnknownShapeFormat(format950)
            .AutoContiguous();
        config_950.Output("y1")
            .ParamType(REQUIRED)
            .DataType(yDataType950)
            .Format(format950)
            .UnknownShapeFormat(format950)
            .AutoContiguous();
        config_950.Output("y2")
            .ParamType(REQUIRED)
            .DataType(yDataType950)
            .Format(format950)
            .UnknownShapeFormat(format950)
            .AutoContiguous();
        config_950.Output("y3")
            .ParamType(REQUIRED)
            .DataType(y3DataType950)
            .Format(format950)
            .UnknownShapeFormat(format950)
            .AutoContiguous();
        config_950.Output("y4")
            .ParamType(REQUIRED)
            .DataType(xDataType950)
            .Format(format950)
            .UnknownShapeFormat(format950)
            .AutoContiguous();
        config_950.Output("x")
            .ParamType(REQUIRED)
            .DataType(xDataType950)
            .Format(format950)
            .UnknownShapeFormat(format950)
            .AutoContiguous();
        config_950.Output("scale1")
            .ParamType(REQUIRED)
            .DataType(scalesOutDataType950)
            .Format(format950)
            .UnknownShapeFormat(format950)
            .AutoContiguous();
        config_950.Output("scale2")
            .ParamType(REQUIRED)
            .DataType(scalesOutDataType950)
            .Format(format950)
            .UnknownShapeFormat(format950)
            .AutoContiguous();
        return config_950;
    }

    OpAICoreConfig GetKirinCoreConfig() const
    {
        OpAICoreConfig v2KirinConfig;
        v2KirinConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        v2KirinConfig.Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        v2KirinConfig.Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        v2KirinConfig.Input("gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        v2KirinConfig.Input("smooth_scale1")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        v2KirinConfig.Input("smooth_scale2")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        v2KirinConfig.Input("beta")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        v2KirinConfig.Output("y1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        v2KirinConfig.Output("y2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        v2KirinConfig.Output("y3")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        v2KirinConfig.Output("y4")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        v2KirinConfig.Output("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        v2KirinConfig.Output("scale1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        v2KirinConfig.Output("scale2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        return v2KirinConfig;
    }
};
OP_ADD(AddRmsNormDynamicQuantV2);
} // namespace ops
