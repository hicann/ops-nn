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
 * \file index_fill_def.cpp
 * \brief
 */
#include <cstdint>
#include "register/op_def_registry.h"
namespace {
    static const std::vector<ge::DataType> xDataType = {
        ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT64, ge::DT_INT32, ge::DT_BOOL,
        ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT64, ge::DT_INT32, ge::DT_BOOL};
    static const std::vector<ge::Format> formatList = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
    static const std::vector<ge::DataType> indicesDataType = {
            ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
            ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};
}

namespace ops {
class IndexFill : public OpDef {
public:
    explicit IndexFill(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType(xDataType).Format(formatList).UnknownShapeFormat(formatList);
        this->Input("indices").ParamType(REQUIRED).DataType(indicesDataType).Format(formatList).UnknownShapeFormat(formatList);
        this->Input("value").ParamType(REQUIRED).DataType(xDataType).Format(formatList).UnknownShapeFormat(formatList);
        this->Output("y").ParamType(REQUIRED).DataType(xDataType).Format(formatList).UnknownShapeFormat(formatList);
        this->Attr("dim").AttrType(REQUIRED).Int();
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true);
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
    }
};

OP_ADD(IndexFill);
}  // namespace ops