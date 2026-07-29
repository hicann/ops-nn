/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "register/op_def_registry.h"

namespace ops {
class GaussianNllLoss : public OpDef {
public:
    explicit GaussianNllLoss(const char* name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> types = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType(types)
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("target")
            .ParamType(REQUIRED)
            .DataType(types)
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("var")
            .ParamType(REQUIRED)
            .DataType(types)
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Attr("full").AttrType(OPTIONAL).Bool(false);
        this->Attr("eps").AttrType(OPTIONAL).Float(1e-6);
        this->Attr("reduction").AttrType(OPTIONAL).String("mean");
        this->Output("loss")
            .ParamType(REQUIRED)
            .DataType(types)
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(GaussianNllLoss);
} // namespace ops
