/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sleep_def.cpp
 * \brief Sleep operator definition — A5 (Ascend 950) SIMT implementation
 *
 * Control primitive that inserts a device-side delay into the current stream.
 * Semantics match torch.cuda._sleep: busy-spin on AI Core clock() for the
 * specified number of cycles.
 */

#include "register/op_def_registry.h"

namespace ops {

class Sleep : public OpDef {
public:
    explicit Sleep(const char* name) : OpDef(name)
    {
        this->Input("cycles")
            .ParamType(REQUIRED)
            .ValueDepend(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->AICore().AddConfig("ascend950");
    }
};

OP_ADD(Sleep);
} // namespace ops
