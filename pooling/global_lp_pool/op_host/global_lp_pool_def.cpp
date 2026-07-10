/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file global_lp_pool_def.cpp
 * \brief GlobalLpPool operator definition
 *
 * GlobalLpPool: y = (sum(|x|^p along spatial dims))^(1/p), keepdims=True.
 * Input x supports bfloat16, float16, float32.
 * Attr p is OPTIONAL float, default 2.0.
 * Output y has the same dtype as x.
 * 4D input (NCHW) → output (N,C,1,1); 5D (NCD0D1D2) → output (N,C,1,1,1).
 */

#include "register/op_def_registry.h"

namespace ops {

class GlobalLpPool : public OpDef {
public:
    explicit GlobalLpPool(const char* name) : OpDef(name)
    {
        // Input x: 4D (NCHW) or 5D (NCD0D1D2), supports bf16, fp16, fp32
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // Attr p: OPTIONAL float, default 2.0
        this->Attr("p").AttrType(OPTIONAL).Float(2.0f);

        // Output y: same dtype as x
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // AI Core compilation config for ascend950
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true);

        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(GlobalLpPool);

} // namespace ops
