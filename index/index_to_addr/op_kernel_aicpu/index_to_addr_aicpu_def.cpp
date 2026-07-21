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
class IndexToAddr : public OpDef {
public:
    explicit IndexToAddr(const char* name) : OpDef(name)
    {
        this->Input("base_addr").ParamType(REQUIRED).DataType({ge::DT_INT64, ge::DT_UINT64});
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_INT64, ge::DT_UINT64});
        this->Output("addrs_table").ParamType(REQUIRED).DataType({ge::DT_INT64, ge::DT_UINT64});
        this->Attr("ori_shape").AttrType(REQUIRED).ListInt();
        this->Attr("block_size").AttrType(REQUIRED).ListInt();
        this->Attr("ori_storage_mode").AttrType(OPTIONAL).String("Matrix");
        this->Attr("block_storage_mode").AttrType(OPTIONAL).String("Matrix");
        this->Attr("rank_id").AttrType(OPTIONAL).Int(0);
        this->Attr("dtype").AttrType(OPTIONAL).Float(ge::DT_FLOAT);

        ApplyNnAicpuDefaultCfg(*this);
        this->AICPU().ExtendCfgInfo(OP_INFO_OPS_FLAG.c_str(), OPEN_OPS_FLAG.c_str());
        this->AICPU().ExtendCfgInfo(OP_INFO_SUB_TYPE_OF_INFERSHAPE.c_str(), "1");
    }
};

OP_ADD(IndexToAddr);
} // namespace ops
