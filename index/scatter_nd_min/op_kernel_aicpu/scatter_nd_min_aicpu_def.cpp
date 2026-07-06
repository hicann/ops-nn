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
class ScatterNdMin : public OpDef {
public:
    explicit ScatterNdMin(const char* name) : OpDef(name)
    {
        // Keep this list strictly aligned with scatter_nd_min_aicpu.cpp Compute() switch-case.
        const std::vector<ge::DataType> data_types = {ge::DT_DOUBLE, ge::DT_FLOAT,  ge::DT_FLOAT16, ge::DT_INT16,
                                                      ge::DT_INT32,  ge::DT_INT64,  ge::DT_INT8,    ge::DT_UINT16,
                                                      ge::DT_UINT32, ge::DT_UINT64, ge::DT_UINT8};
        this->Input("var").ParamType(REQUIRED).DataType(data_types);
        this->Input("indices").ParamType(REQUIRED).DataType({ge::DT_INT32, ge::DT_INT64});
        this->Input("updates").ParamType(REQUIRED).DataType(data_types);
        this->Output("var").ParamType(REQUIRED).DataType(data_types);

        ApplyNnAicpuDefaultCfg(*this);
    }
};

OP_ADD(ScatterNdMin);
} // namespace ops
