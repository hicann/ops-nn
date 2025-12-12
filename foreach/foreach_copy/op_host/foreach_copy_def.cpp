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
 * \file foreach_copy_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {

class ForeachCopy : public OpDef {
public:
    explicit ForeachCopy(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(DYNAMIC)
            .DataType(
                {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_BF16, ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16,
                 ge::DT_UINT16, ge::DT_UINT32, ge::DT_DOUBLE, ge::DT_BOOL, ge::DT_INT64})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y")
            .ParamType(DYNAMIC)
            .DataType(
                {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_BF16, ge::DT_INT8, ge::DT_UINT8, ge::DT_INT16,
                 ge::DT_UINT16, ge::DT_UINT32, ge::DT_DOUBLE, ge::DT_BOOL, ge::DT_INT64})
            .Format(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(ForeachCopy);
} // namespace ops
