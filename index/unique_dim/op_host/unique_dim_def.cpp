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
 * \file unique_dim_def.cpp
 * \brief Operator definition for UniqueDim
 */
#include "register/op_def_registry.h"

namespace ops
{
static const std::vector<ge::DataType> DataTypeXY = {
    ge::DT_UINT8, ge::DT_INT8, ge::DT_UINT16, ge::DT_INT16,
    ge::DT_UINT32, ge::DT_INT32, ge::DT_UINT64, ge::DT_INT64,
    ge::DT_DOUBLE, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
    ge::DT_BOOL};
static const std::vector<ge::Format> format = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND};
static const std::vector<ge::DataType> DataTypeIdx = {
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64};
static const std::vector<ge::DataType> DataTypeCount = {
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64};

static ge::graphStatus CheckSupport(const ge::Operator& op, ge::AscendString& result) {
    result = ge::AscendString(R"({"isSupported": "True", "dynamicCompileStatic": "True", "reason": "CheckSupported success."})");
    return ge::GRAPH_SUCCESS;
}

class UniqueDim : public OpDef
{
public:
    explicit UniqueDim(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType(DataTypeXY).Format(format).UnknownShapeFormat(format);
        this->Output("y")
            .OutputShapeDependOnCompute()
            .ParamType(REQUIRED)
            .DataType(DataTypeXY)
            .Format(format)
            .UnknownShapeFormat(format);
        this->Output("idx")
            .OutputShapeDependOnCompute()
            .ParamType(REQUIRED)
            .DataType(DataTypeIdx)
            .Format(format)
            .UnknownShapeFormat(format);
        this->Output("count")
            .OutputShapeDependOnCompute()
            .ParamType(REQUIRED)
            .DataType(DataTypeCount)
            .Format(format)
            .UnknownShapeFormat(format);
        this->Attr("dim").AttrType(OPTIONAL).Int(0);
        this->Attr("sorted").AttrType(OPTIONAL).Bool(true);
        this->Attr("return_inverse").AttrType(OPTIONAL).Bool(false);
        this->Attr("return_counts").AttrType(OPTIONAL).Bool(false);
        this->AICore().SetCheckSupport(CheckSupport);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "unique_dim_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};
OP_ADD(UniqueDim);
// 手动注册opDef.AICore()里设置的CheckSupport函数
// 需要当前目录下的CMakeLists.txt将本_def.cpp加入${OPHOST_NAME}_tiling_obj编译目标内
static int UNIQUE_DIM_REGISTERED = [](const char* name) {
    UniqueDim opDef(name);
    optiling::OpCheckFuncHelper(FUNC_CHECK_SUPPORTED, name, opDef.AICore().GetCheckSupport());
    return 0;
}("UniqueDim");

} // namespace ops
