/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"

namespace ops {
namespace {

constexpr std::initializer_list<ge::DataType> A2_VALUE_TYPES = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,
                                                                ge::DT_FLOAT,   ge::DT_BF16,    ge::DT_BF16};
constexpr std::initializer_list<ge::DataType> ARCH310P_VALUE_TYPES = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT,
                                                                      ge::DT_FLOAT};
constexpr std::initializer_list<ge::DataType> A2_INDEX_TYPES = {ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
                                                                ge::DT_INT8, ge::DT_INT8, ge::DT_INT8};
constexpr std::initializer_list<ge::DataType> ARCH310P_INDEX_TYPES = {ge::DT_INT8, ge::DT_INT8, ge::DT_INT8,
                                                                      ge::DT_INT8};
constexpr std::initializer_list<ge::Format> A2_FORMATS = {ge::FORMAT_ND,   ge::FORMAT_NCHW, ge::FORMAT_ND,
                                                          ge::FORMAT_NCHW, ge::FORMAT_ND,   ge::FORMAT_NCHW};
constexpr std::initializer_list<ge::Format> ARCH310P_FORMATS = {ge::FORMAT_ND, ge::FORMAT_NCHW, ge::FORMAT_ND,
                                                                ge::FORMAT_NCHW};

template <typename ConfigT>
void AddInput(ConfigT& config, std::initializer_list<ge::DataType> dataTypes, std::initializer_list<ge::Format> formats)
{
    config.Input("x")
        .ParamType(REQUIRED)
        .DataType(dataTypes)
        .Format(formats)
        .UnknownShapeFormat(formats)
        .AutoContiguous();
}

template <typename ConfigT>
void AddOutputs(ConfigT& config, std::initializer_list<ge::DataType> valueTypes,
                std::initializer_list<ge::DataType> indexTypes, std::initializer_list<ge::Format> formats)
{
    config.Output("out")
        .ParamType(REQUIRED)
        .DataType(valueTypes)
        .Format(formats)
        .UnknownShapeFormat(formats)
        .AutoContiguous();
    config.Output("indices").ParamType(REQUIRED).DataType(indexTypes).Format(formats).UnknownShapeFormat(formats);
}

template <typename ConfigT>
void SetCommonConfig(ConfigT& config)
{
    config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(false)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(false)
        .ExtendCfgInfo("opFile.value", "max_pool2d_with_mask");
}

} // namespace

class MaxPool2dWithMask : public OpDef {
public:
    explicit MaxPool2dWithMask(const char* name) : OpDef(name)
    {
        AddInput(*this, A2_VALUE_TYPES, A2_FORMATS);
        AddOutputs(*this, A2_VALUE_TYPES, A2_INDEX_TYPES, A2_FORMATS);

        this->Attr("kernelSize").AttrType(REQUIRED).ListInt();
        this->Attr("stride").AttrType(REQUIRED).ListInt();
        this->Attr("padding").AttrType(REQUIRED).ListInt();
        this->Attr("dilation").AttrType(REQUIRED).ListInt();
        this->Attr("ceilMode").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig a2Config;
        SetCommonConfig(a2Config);
        this->AICore().AddConfig("ascend910b", a2Config);

        OpAICoreConfig arch310pConfig;
        AddInput(arch310pConfig, ARCH310P_VALUE_TYPES, ARCH310P_FORMATS);
        AddOutputs(arch310pConfig, ARCH310P_VALUE_TYPES, ARCH310P_INDEX_TYPES, ARCH310P_FORMATS);
        SetCommonConfig(arch310pConfig);
        this->AICore().AddConfig("ascend310p", arch310pConfig);
    }
};

OP_ADD(MaxPool2dWithMask);
} // namespace ops
