/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#include "../../../common/inc/aicpu/aicpu_op_def.h"
#include "register/op_def_registry.h"

#include <vector>

namespace ops {
namespace {
const std::vector<ge::DataType> kValueTypes = {
    ge::DT_BOOL,  ge::DT_COMPLEX128, ge::DT_COMPLEX64, ge::DT_DOUBLE, ge::DT_FLOAT,  ge::DT_FLOAT16, ge::DT_INT16,
    ge::DT_INT32, ge::DT_INT64,      ge::DT_INT8,      ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64,  ge::DT_UINT8};
} // namespace

class SparseFillEmptyRows : public OpDef {
public:
    explicit SparseFillEmptyRows(const char* name) : OpDef(name)
    {
        this->Input("indices").DataType({ge::DT_INT64});
        this->Input("values").DataType(kValueTypes);
        this->Input("dense_shape").DataType({ge::DT_INT64});
        this->Input("default_value").DataType(kValueTypes);
        this->Output("y_indices").DataType({ge::DT_INT64});
        this->Output("y_values").DataType(kValueTypes);
        this->Output("empty_row_indicator").DataType({ge::DT_BOOL});
        this->Output("reverse_index_map").DataType({ge::DT_INT64});

        ApplyNnAicpuDefaultCfg(*this);
        this->AICPU().ExtendCfgInfo(OP_INFO_SUB_TYPE_OF_INFERSHAPE.c_str(), DEFAULT_SUB_TYPE_OF_INFERSHAPE_3.c_str());
    }
};

OP_ADD(SparseFillEmptyRows);
} // namespace ops
