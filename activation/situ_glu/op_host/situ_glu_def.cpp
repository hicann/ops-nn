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
 * \file situ_glu_def.cpp
 * \brief
 */
#include <register/op_def_registry.h>

namespace ops {
constexpr float DEFAULT_BETA = 1.0;
constexpr float DEFAULT_LINEAR_BETA = 0.0;

static const std::vector<ge::DataType> xDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
static const std::vector<ge::Format> xFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
static const std::vector<ge::Format> xUnknownFormat = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class SituGlu : public OpDef {
public:
    explicit SituGlu(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(xDtype)
            .Format(xFormat)
            .UnknownShapeFormat(xUnknownFormat)
            .AutoContiguous();
        this->Output("y").ParamType(REQUIRED).DataType(xDtype).Format(xFormat).UnknownShapeFormat(xUnknownFormat);
        this->Attr("dim").AttrType(OPTIONAL).Int(-1);
        this->Attr("beta").AttrType(OPTIONAL).Float(DEFAULT_BETA);
        this->Attr("linear_beta").AttrType(OPTIONAL).Float(DEFAULT_LINEAR_BETA);
        this->Attr("activate_left").AttrType(OPTIONAL).Bool(true);

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
        this->AICore().AddConfig("ascend950");
    }
};
OP_ADD(SituGlu);
} // namespace ops
