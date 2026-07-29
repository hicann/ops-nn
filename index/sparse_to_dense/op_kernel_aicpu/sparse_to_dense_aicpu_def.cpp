/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../common/inc/aicpu/aicpu_op_def.h"
#include "register/op_def_registry.h"

namespace ops {
class SparseToDense : public OpDef {
public:
    explicit SparseToDense(const char* name) : OpDef(name)
    {
        const std::vector<ge::DataType> indexTypes = {ge::DT_INT32, ge::DT_INT64};
        const std::vector<ge::DataType> valueTypes = {ge::DT_FLOAT,  ge::DT_FLOAT16, ge::DT_INT8,  ge::DT_INT16,
                                                      ge::DT_UINT16, ge::DT_UINT8,   ge::DT_INT32, ge::DT_INT64,
                                                      ge::DT_BOOL,   ge::DT_DOUBLE};
        this->Input("indices").ParamType(REQUIRED).DataType(indexTypes);
        this->Input("output_shape").ParamType(REQUIRED).DataType(indexTypes).ValueDepend(OPTIONAL);
        this->Input("values").ParamType(REQUIRED).DataType(valueTypes);
        this->Input("default_value").ParamType(REQUIRED).DataType(valueTypes);
        this->Output("y").ParamType(REQUIRED).DataType(valueTypes);
        this->Attr("validate_indices").AttrType(OPTIONAL).Bool(true);

        ApplyNnAicpuDefaultCfg(*this);
        this->AICPU().ExtendCfgInfo(OP_INFO_OPS_FLAG.c_str(), OPEN_OPS_FLAG.c_str());
    }
};

OP_ADD(SparseToDense);
} // namespace ops
