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
 * \file clipped_swiglu.cpp
 * \brief
 */
#include <register/op_def_registry.h>

namespace ops {
constexpr float DEFAULT_ALPHA = 1.702;
constexpr float DEFAULT_LIMIT = 7.0;

static const std::vector<ge::DataType> xDtype = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
static const std::vector<ge::DataType> groupIndexDtype = {ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};
static const std::vector<ge::Format> xFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class ClippedSwiglu : public OpDef {
public:
    explicit ClippedSwiglu(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType(xDtype).Format(xFormat).UnknownShapeFormat(xFormat);
        this->Input("group_index")
            .ParamType(OPTIONAL)
            .DataType(groupIndexDtype)
            .Format(xFormat)
            .UnknownShapeFormat(xFormat);
        this->Output("y").ParamType(REQUIRED).DataType(xDtype).Format(xFormat).UnknownShapeFormat(xFormat);
        this->Attr("dim").AttrType(OPTIONAL).Int(-1);
        this->Attr("alpha").AttrType(OPTIONAL).Float(DEFAULT_ALPHA);
        this->Attr("limit").AttrType(OPTIONAL).Float(DEFAULT_LIMIT);
        this->Attr("bias").AttrType(OPTIONAL).Float(1.0);
        this->Attr("interleaved").AttrType(OPTIONAL).Bool(true);

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");

        OpAICoreConfig regbaseConfig;
        regbaseConfig.Input("x")
            .ParamType(REQUIRED)
            .DataType(xDtype)
            .Format(xFormat)
            .UnknownShapeFormat(xFormat)
            .AutoContiguous();
        regbaseConfig.Input("group_index")
            .ParamType(OPTIONAL)
            .DataType(groupIndexDtype)
            .Format(xFormat)
            .UnknownShapeFormat(xFormat)
            .AutoContiguous();
        regbaseConfig.Output("y").ParamType(REQUIRED).DataType(xDtype).Format(xFormat).UnknownShapeFormat(xFormat);
        regbaseConfig.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "clipped_swiglu_apt");
        this->AICore().AddConfig("ascend950", regbaseConfig);
    }
};
OP_ADD(ClippedSwiglu);
} // namespace ops
