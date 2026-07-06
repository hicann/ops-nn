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
class Where : public OpDef {
public:
    explicit Where(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType(
            {ge::DT_DOUBLE, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_UINT8, ge::DT_QINT8, ge::DT_QUINT8,
             ge::DT_INT16, ge::DT_UINT16, ge::DT_INT32, ge::DT_UINT32, ge::DT_QINT32, ge::DT_INT64, ge::DT_UINT64,
             ge::DT_BOOL, ge::DT_COMPLEX64, ge::DT_COMPLEX128});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_INT64});

        ApplyNnAicpuDefaultCfg(*this);
        this->AICPU().OpKernelLib(LEGACY_AICPU_OP_KERNEL_LIB.c_str());
        this->AICPU().UserDefined(false);
        this->AICPU().ExtendCfgInfo(OP_INFO_FORMAT_AGNOSTIC.c_str(), TRUE_FORMAT_AGNOSTIC.c_str());
        this->AICPU().ExtendCfgInfo(OP_INFO_OPS_FLAG.c_str(), OPEN_OPS_FLAG.c_str());
        this->AICPU().ExtendCfgInfo(OP_INFO_SUB_TYPE_OF_INFERSHAPE.c_str(), DEFAULT_SUB_TYPE_OF_INFERSHAPE_3.c_str());
    }
};

OP_ADD(Where);
} // namespace ops
