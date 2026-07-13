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
class Bucketize : public OpDef {
public:
    explicit Bucketize(const char* name) : OpDef(name)
    {
        this->Input("x").DataType({ge::DT_INT32, ge::DT_INT64, ge::DT_DOUBLE, ge::DT_FLOAT});
        this->Output("y").DataType({ge::DT_INT32, ge::DT_INT64});
        this->Attr("boundaries").AttrType(REQUIRED).ListFloat();
        this->Attr("right").AttrType(OPTIONAL).Bool(true);

        ApplyNnAicpuDefaultCfg(*this);
        this->AICPU().ExtendCfgInfo(OP_INFO_OPS_FLAG.c_str(), OPEN_OPS_FLAG.c_str());
        this->AICPU().ExtendCfgInfo(OP_INFO_SUB_TYPE_OF_INFERSHAPE.c_str(), "1");
    }
};

OP_ADD(Bucketize);
} // namespace ops
