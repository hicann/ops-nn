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
 * \file foreach_minimum_scalar_list_def.cpp
 * \brief
 */

#include "../../foreach_utils/op_host/foreach_proto_utils.h"

namespace ops {
class ForeachMinimumScalarList : public OpDef {
public:
    explicit ForeachMinimumScalarList(const char* name) : OpDef(name)
    {
        std::vector<ge::DataType> tensor_dtype_list = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_BF16};
        std::vector<ge::Format> format_list(tensor_dtype_list.size(), ge::FORMAT_ND);
        std::vector<ge::DataType> scalar_dtype_list;
        std::for_each(tensor_dtype_list.cbegin(), tensor_dtype_list.cend(), [&scalar_dtype_list](ge::DataType dtype) {
            scalar_dtype_list.push_back(DtypeTensor2Scalar(dtype));
        });
        this->Input("x")
            .ParamType(DYNAMIC)
            .DataType(tensor_dtype_list)
            .Format(format_list)
            .UnknownShapeFormat(format_list)
            .AutoContiguous();
        this->Input("scalars")
            .ScalarList()
            .ParamType(REQUIRED)
            .DataType(scalar_dtype_list)
            .Format(format_list)
            .UnknownShapeFormat(format_list)
            .AutoContiguous();
        this->Output("y")
            .ParamType(DYNAMIC)
            .DataType(tensor_dtype_list)
            .Format(format_list)
            .UnknownShapeFormat(format_list)
            .AutoContiguous();
        this->AICore().AddConfig("ascend950");
        this->AICore().AddConfig("ascend910_93");
        this->AICore().AddConfig("ascend910b");

        OpAICoreConfig config_kirin = GetKirinCoreConfig();
        this->AICore().AddConfig("kirinx90", config_kirin);
        this->AICore().AddConfig("kirin9030", config_kirin);
    }

private:
    OpAICoreConfig GetKirinCoreConfig() const
    {
        OpAICoreConfig config_kirin;
        config_kirin.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);
        std::vector<ge::DataType> tensor_dtype_list_kirin = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32};
        std::vector<ge::Format> format_list_kirin(tensor_dtype_list_kirin.size(), ge::FORMAT_ND);
        std::vector<ge::DataType> scalar_dtype_list_kirin;
        std::for_each(
            tensor_dtype_list_kirin.cbegin(), tensor_dtype_list_kirin.cend(),
            [&scalar_dtype_list_kirin](ge::DataType dtype) {
                scalar_dtype_list_kirin.push_back(DtypeTensor2Scalar(dtype));
            });
        config_kirin.Input("x")
            .ParamType(DYNAMIC)
            .DataType(tensor_dtype_list_kirin)
            .Format(format_list_kirin)
            .UnknownShapeFormat(format_list_kirin)
            .AutoContiguous();
        config_kirin.Input("scalars")
            .ScalarList()
            .ParamType(REQUIRED)
            .DataType(scalar_dtype_list_kirin)
            .Format(format_list_kirin)
            .UnknownShapeFormat(format_list_kirin)
            .AutoContiguous();
        config_kirin.Output("y")
            .ParamType(DYNAMIC)
            .DataType(tensor_dtype_list_kirin)
            .Format(format_list_kirin)
            .UnknownShapeFormat(format_list_kirin)
            .AutoContiguous();
        return config_kirin;
    }
};

OP_ADD(ForeachMinimumScalarList);
} // namespace ops
