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
class SparseSegmentMean : public OpDef {
public:
    explicit SparseSegmentMean(const char* name) : OpDef(name)
    {
        const std::vector<ge::DataType> dataTypes = {ge::DT_FLOAT, ge::DT_DOUBLE, ge::DT_FLOAT16};
        const std::vector<ge::DataType> indexTypes = {ge::DT_INT32, ge::DT_INT64};
        this->Input("x").ParamType(REQUIRED).DataType(dataTypes);
        this->Input("indices").ParamType(REQUIRED).DataType(indexTypes);
        this->Input("segment_ids").ParamType(REQUIRED).DataType(indexTypes);
        this->Output("y").ParamType(REQUIRED).DataType(dataTypes);

        ApplyNnAicpuDefaultCfg(*this);
        this->AICPU().ExtendCfgInfo(OP_INFO_OPS_FLAG.c_str(), OPEN_OPS_FLAG.c_str());
    }
};

OP_ADD(SparseSegmentMean);
} // namespace ops
