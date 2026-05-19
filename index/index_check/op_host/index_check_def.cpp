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
 * \file index_check_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace {
    static const std::vector<ge::DataType> boundsType = {ge::DT_INT64, ge::DT_INT64};

    static const std::vector<ge::Format> format = {ge::FORMAT_ND, ge::FORMAT_ND};

    static const std::vector<ge::DataType> indicesType = {ge::DT_INT64, ge::DT_INT32};
}
namespace ops {
class IndexCheck : public OpDef
{
public:
    explicit IndexCheck(const char* name) : OpDef(name)
    {
        this->Input("bounds")
            .ParamType(REQUIRED)
            .DataType(boundsType)
            .Format(format);
        this->Input("indices_list")
            .ParamType(DYNAMIC)
            .DataType(indicesType)
            .Format(format);

        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(IndexCheck);
} // namespace ops
