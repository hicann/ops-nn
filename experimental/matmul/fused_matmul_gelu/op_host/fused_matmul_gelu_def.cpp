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

void AddRequiredMatmulInput(OpDef& op, const char* name)
{
    op.Input(name)
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
}

void AddOptionalMatmulInput(OpDef& op, const char* name)
{
    op.Input(name)
        .ParamType(OPTIONAL)
        .DataType({ge::DT_BF16, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
}

void AddMatmulOutput(OpDef& op, const char* name)
{
    op.Output(name)
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16, ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
}

OpAICoreConfig MakeFusedMatmulGeluAicoreConfig()
{
    OpAICoreConfig config;
    config.DynamicCompileStaticFlag(true);
    config.DynamicFormatFlag(false);
    config.DynamicRankSupportFlag(true);
    config.DynamicShapeSupportFlag(true);
    config.NeedCheckSupportFlag(false);
    config.PrecisionReduceFlag(true);
    return config;
}

void AddFusedMatmulGeluAicoreConfig(OpDef& op)
{
    auto config = MakeFusedMatmulGeluAicoreConfig();
    op.AICore().AddConfig("ascend910b", config);
    op.AICore().AddConfig("ascend910_93", config);
    op.AICore().AddConfig("ascend950", config);
}

} // namespace

class FusedMatmulGelu : public OpDef {
public:
    explicit FusedMatmulGelu(const char* name) : OpDef(name)
    {
        AddRequiredMatmulInput(*this, "x");
        AddRequiredMatmulInput(*this, "weight");
        AddOptionalMatmulInput(*this, "bias");
        AddMatmulOutput(*this, "y");

        this->Attr("approximate").AttrType(OPTIONAL).Int(1);
        AddFusedMatmulGeluAicoreConfig(*this);
    }
};

OP_ADD(FusedMatmulGelu);

} // namespace ops
