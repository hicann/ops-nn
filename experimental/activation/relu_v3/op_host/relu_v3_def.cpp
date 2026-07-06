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
class ReluV3 : public OpDef {
public:
    explicit ReluV3(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8, ge::DT_UINT8, ge::DT_BF16})
            .Format({ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0,
                     ge::FORMAT_NC1HWC0})
            .UnknownShapeFormat({ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0,
                                 ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8, ge::DT_UINT8, ge::DT_BF16})
            .Format({ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0,
                     ge::FORMAT_NC1HWC0})
            .UnknownShapeFormat({ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0,
                                 ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0});
        this->Output("mask")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8, ge::DT_UINT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat(
                {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(ReluV3);
} // namespace ops
