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
 * \file sparse_tensor_dense_mat_mul_def.cpp
 * \brief
 */
#include <vector>
#include <unordered_set>
#include "register/op_def_registry.h"
#include "graph/operator.h"

namespace {
using namespace ge;

const std::vector<DataType> indexTypes = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32,
                                          DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_INT64};
const std::vector<DataType> valueTypes = {DT_FLOAT16,   DT_FLOAT,      DT_DOUBLE,    DT_INT32,     DT_INT64,
                                          DT_COMPLEX64, DT_COMPLEX128, DT_FLOAT16,   DT_FLOAT,     DT_DOUBLE,
                                          DT_INT32,     DT_INT64,      DT_COMPLEX64, DT_COMPLEX128};

graphStatus CheckIfAICoreSupported(const Operator& op, AscendString& result)
{
    static const std::unordered_set<DataType> AICORE_SUPPORTED_VALUES_TYPES = {DT_FLOAT16, DT_FLOAT, DT_INT32};

    DataType x1ValuesDType = op.GetInputDescByName("x1_values").GetDataType();
    DataType x2DType = op.GetInputDescByName("x2").GetDataType();
    if (AICORE_SUPPORTED_VALUES_TYPES.count(x1ValuesDType) == 0 || AICORE_SUPPORTED_VALUES_TYPES.count(x2DType) == 0) {
        result = AscendString(
            R"({"isSupported": "False", "dynamicCompileStatic": "True", "reason": "Unsupported Values DataType."})");
        return GRAPH_FAILED;
    }

    result = AscendString(
        R"({"isSupported": "True", "dynamicCompileStatic": "True", "reason": "AICore CheckSupport Passed."})");
    return GRAPH_SUCCESS;
}

} // namespace

namespace ops {
class SparseTensorDenseMatMul : public OpDef {
public:
    explicit SparseTensorDenseMatMul(const char* name) : OpDef(name)
    {
        this->Input("x1_indices").ParamType(REQUIRED).DataType(indexTypes).FormatList({ge::FORMAT_ND}).AutoContiguous();

        this->Input("x1_values").ParamType(REQUIRED).DataType(valueTypes).FormatList({ge::FORMAT_ND}).AutoContiguous();

        this->Input("x1_shape")
            .ParamType(REQUIRED)
            .DataTypeList({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND})
            .ValueDepend(REQUIRED);

        this->Input("x2").ParamType(REQUIRED).DataType(valueTypes).FormatList({ge::FORMAT_ND}).AutoContiguous();

        this->Output("y").ParamType(REQUIRED).DataType(valueTypes).FormatList({ge::FORMAT_ND});

        this->Attr("adjoint_a").AttrType(OPTIONAL).Bool(false);
        this->Attr("adjoint_b").AttrType(OPTIONAL).Bool(false);

        this->AICore().SetCheckSupport(CheckIfAICoreSupported);

        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(false)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "sparse_tensor_dense_mat_mul_apt");
        this->AICore().AddConfig("ascend910_95", config);
    }
};

OP_ADD(SparseTensorDenseMatMul);
// 手动注册opDef.AICore()里设置的CheckSupport函数
// 需要当前目录下的CMakeLists.txt将本_def.cpp加入${OPHOST_NAME}_tiling_obj编译目标内
static int g_SparseTensorDenseMatMul_register_check_support = [](const char* name) {
    SparseTensorDenseMatMul opDef(name);
    optiling::OpCheckFuncHelper(FUNC_CHECK_SUPPORTED, name, opDef.AICore().GetCheckSupport());
    return 0;
}("SparseTensorDenseMatMul");

} // namespace ops