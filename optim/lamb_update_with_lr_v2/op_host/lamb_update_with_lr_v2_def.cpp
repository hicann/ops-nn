/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lamb_update_with_lr_v2_def.cpp
 * \brief lamb_update_with_lr_v2 def
 */

#include "register/op_def_registry.h"

namespace ops {
class LambUpdateWithLrV2 : public OpDef {
public:
    explicit LambUpdateWithLrV2(const char* name) : OpDef(name)
    {
        static const char* kIn[] = {"x1", "x2", "x3", "x4", "x5", "greater_y", "select_e"};
        for (auto n : kIn) {
            this->Input(n)
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        }
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("opFile.value", "lamb_update_with_lr_v2");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(LambUpdateWithLrV2);
} // namespace ops
