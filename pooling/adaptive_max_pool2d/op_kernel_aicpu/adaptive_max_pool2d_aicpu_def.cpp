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
#include "../../../common/inc/aicpu/aicpu_op_def.h"

namespace ops {
class AdaptiveMaxPool2d : public OpDef {
public:
    explicit AdaptiveMaxPool2d(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_DOUBLE});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_DOUBLE});
        this->Output("argmax").ParamType(REQUIRED).DataType({ge::DT_INT32, ge::DT_INT64});
        this->Attr("output_size").AttrType(REQUIRED).ListInt();

        ApplyNnAicpuDefaultCfg(*this);
        this->AICPU().ExtendCfgInfo(OP_INFO_FORMAT_AGNOSTIC.c_str(), TRUE_FORMAT_AGNOSTIC.c_str());
        this->AICPU().ExtendCfgInfo(OP_INFO_OPS_FLAG.c_str(), OPEN_OPS_FLAG.c_str());
    }
};

OP_ADD(AdaptiveMaxPool2d);
} // namespace ops
