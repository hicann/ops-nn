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
 * \file foreach_binary_op_def.cpp
 * \brief Op definition for unified foreach binary op (arch35/ascend950).
 *        Computes y[i] = x1[i] <op> x2[i] where <op> is selected by attr op_code:
 *        0 = add, 1 = sub, 2 = mul, 3 = div.
 */

#include <vector>
#include "register/op_def_registry.h"

namespace ops {
class ForeachBinaryOp : public OpDef {
public:
    explicit ForeachBinaryOp(const char* name) : OpDef(name)
    {
        std::vector<ge::DataType> tensor_dtype_list = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_BF16};
        std::vector<ge::Format> format_list(tensor_dtype_list.size(), ge::FORMAT_ND);

        this->Input("x1")
            .ParamType(DYNAMIC)
            .DataType(tensor_dtype_list)
            .Format(format_list)
            .UnknownShapeFormat(format_list)
            .AutoContiguous();
        this->Input("x2")
            .ParamType(DYNAMIC)
            .DataType(tensor_dtype_list)
            .Format(format_list)
            .UnknownShapeFormat(format_list)
            .AutoContiguous();
        this->Output("y")
            .ParamType(DYNAMIC)
            .DataType(tensor_dtype_list)
            .Format(format_list)
            .UnknownShapeFormat(format_list)
            .AutoContiguous();
        this->Attr("op_code").AttrType(REQUIRED).Int();

        OpAICoreConfig aicoreConfig950;
        aicoreConfig950.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "foreach_binary_op");
        this->AICore().AddConfig("ascend950", aicoreConfig950);
    }
};

OP_ADD(ForeachBinaryOp);
} // namespace ops
