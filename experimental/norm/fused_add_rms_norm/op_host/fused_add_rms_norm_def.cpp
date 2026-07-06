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
 * \file fused_add_rms_norm_def.cpp
 * \brief
 */
#include <initializer_list>

#include "register/op_def_registry.h"

namespace ops {
constexpr std::initializer_list<ge::DataType> BASE_DTYPE = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
constexpr std::initializer_list<ge::DataType> BASE_RSTD_DTYPE = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
constexpr std::initializer_list<ge::Format> BASE_FORMAT = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
constexpr std::initializer_list<ge::DataType> ARCH310P_DTYPE = {ge::DT_FLOAT16, ge::DT_FLOAT};
constexpr std::initializer_list<ge::DataType> ARCH310P_RSTD_DTYPE = {ge::DT_FLOAT, ge::DT_FLOAT};
constexpr std::initializer_list<ge::Format> ARCH310P_FORMAT = {ge::FORMAT_ND, ge::FORMAT_ND};

template <typename ConfigT>
static void AddInput(ConfigT& config, const char* name, std::initializer_list<ge::DataType> dataTypes,
    std::initializer_list<ge::Format> formats)
{
    config.Input(name)
        .ParamType(REQUIRED)
        .DataType(dataTypes)
        .Format(formats)
        .UnknownShapeFormat(formats)
        .AutoContiguous();
}

template <typename ConfigT>
static void AddOutput(ConfigT& config, const char* name, std::initializer_list<ge::DataType> dataTypes,
    std::initializer_list<ge::Format> formats)
{
    config.Output(name)
        .ParamType(REQUIRED)
        .DataType(dataTypes)
        .Format(formats)
        .UnknownShapeFormat(formats)
        .AutoContiguous();
}

template <typename ConfigT>
static void AddCommonInputs(ConfigT& config, std::initializer_list<ge::DataType> dataTypes,
    std::initializer_list<ge::Format> formats)
{
    AddInput(config, "x1", dataTypes, formats);
    AddInput(config, "x2", dataTypes, formats);
    AddInput(config, "gamma", dataTypes, formats);
}

template <typename ConfigT>
static void AddCommonOutputs(ConfigT& config, std::initializer_list<ge::DataType> dataTypes,
    std::initializer_list<ge::DataType> rstdTypes, std::initializer_list<ge::Format> formats)
{
    AddOutput(config, "y", dataTypes, formats);
    AddOutput(config, "rstd", rstdTypes, formats);
    AddOutput(config, "x", dataTypes, formats);
}

static void SetBaseConfig(OpDef& op)
{
    AddCommonInputs(op, BASE_DTYPE, BASE_FORMAT);
    AddCommonOutputs(op, BASE_DTYPE, BASE_RSTD_DTYPE, BASE_FORMAT);
    op.Attr("epsilon").AttrType(OPTIONAL).Float(1e-6);
    op.Attr("scale").AttrType(OPTIONAL).Float(1.0);
    op.AICore().AddConfig("ascend910b");
    op.AICore().AddConfig("ascend910_93");
}

static void AddArch310PConfig(OpDef& op)
{
    OpAICoreConfig config;
    AddCommonInputs(config, ARCH310P_DTYPE, ARCH310P_FORMAT);
    AddCommonOutputs(config, ARCH310P_DTYPE, ARCH310P_RSTD_DTYPE, ARCH310P_FORMAT);
    config.DynamicCompileStaticFlag(true).DynamicRankSupportFlag(true).DynamicShapeSupportFlag(true);
    op.AICore().AddConfig("ascend310p", config);
    op.AICore().AddConfig("kirinx90", config);
}

static void AddArch91095Config(OpDef& op)
{
    OpAICoreConfig config;
    AddCommonInputs(config, BASE_DTYPE, BASE_FORMAT);
    AddCommonOutputs(config, BASE_DTYPE, BASE_RSTD_DTYPE, BASE_FORMAT);
    config.DynamicCompileStaticFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .ExtendCfgInfo("opFile.value", "fused_add_rms_norm");
    op.AICore().AddConfig("ascend910_95", config);
}

class FusedAddRmsNorm : public OpDef {
public:
    explicit FusedAddRmsNorm(const char* name) : OpDef(name)
    {
        SetBaseConfig(*this);
        AddArch310PConfig(*this);
        AddArch91095Config(*this);
    }
};
OP_ADD(FusedAddRmsNorm);
}  // namespace ops
