/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file inplace_index_fill_def.cpp
 * \brief
 */
#include <cstdint>
#include "register/op_def_registry.h"
namespace {
// x支持10种数据类型，indices支持int32/int64两种，共10x2=20种合法组合
// 前10组：x各类型 + indices=INT32，后10组：x各类型 + indices=INT64
static const std::vector<ge::DataType> xDataType = {
    ge::DT_FLOAT, ge::DT_DOUBLE, ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_INT8,
    ge::DT_UINT8, ge::DT_INT16,  ge::DT_INT32,   ge::DT_INT64, ge::DT_BOOL,
    ge::DT_FLOAT, ge::DT_DOUBLE, ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_INT8,
    ge::DT_UINT8, ge::DT_INT16,  ge::DT_INT32,   ge::DT_INT64, ge::DT_BOOL};
static const std::vector<ge::Format> formatList = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
static const std::vector<ge::DataType> indicesDataType = {
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};
} // namespace

namespace ops {
class InplaceIndexFill : public OpDef {
public:
    explicit InplaceIndexFill(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(xDataType)
            .Format(formatList)
            .UnknownShapeFormat(formatList);
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType(indicesDataType)
            .Format(formatList)
            .UnknownShapeFormat(formatList);
        this->Input("value")
            .ParamType(REQUIRED)
            .DataType(xDataType)
            .Format(formatList)
            .UnknownShapeFormat(formatList);
        this->Output("x")
            .ParamType(REQUIRED)
            .DataType(xDataType)
            .Format(formatList)
            .UnknownShapeFormat(formatList);
        this->Attr("dim").AttrType(REQUIRED).Int();

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "inplace_index_fill_apt");

        this->AICore().AddConfig("ascend950", aicore_config);
    }
};

OP_ADD(InplaceIndexFill);
} // namespace ops