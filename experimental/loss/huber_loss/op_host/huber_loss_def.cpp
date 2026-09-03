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
 * \file huber_loss_def.cpp
 * \brief HuberLoss operator definition
 */
#include "register/op_def_registry.h"
// *_def.cpp is built by a separate opbuild step with only $CANN/include on
// the include path and -std=c++11, so the shared tiling-data header is
// included relatively and must stay C++11-clean.
#include "../op_kernel/huber_loss_tiling_data.h"

namespace ops {

class HuberLoss : public OpDef {
public:
    explicit HuberLoss(const char* name) : OpDef(name)
    {
        // input/target/output carry DTYPE_INPUT / DTYPE_TARGET / DTYPE_LOSS
        // into the kernel build. Same shape and dtype, no broadcast.
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Input("target")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // Attribute order is part of the host<->kernel contract: reduction is
        // index 0, delta is index 1. OpDef and tiling's GetAttrs() both address
        // attributes positionally. The default is mean, matching aten.
        this->Attr("reduction").AttrType(OPTIONAL).Int(HUBER_LOSS_REDUCE_MEAN);
        this->Attr("delta").AttrType(OPTIONAL).Float(1.0);

        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(HuberLoss);

} // namespace ops
