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
 * \file lamb_next_m_v_def.cpp
 * \brief lamb_next_m_v def
 */

#include "register/op_def_registry.h"

namespace ops {
class LambNextMV : public OpDef {
public:
    explicit LambNextMV(const char* name) : OpDef(name)
    {
        static const char* kIn[] = {"input_mul3", "input_mul2", "input_realdiv1", "input_mul1", "input_mul0",
                                    "input_realdiv0", "input_mul4", "mul0_x", "mul1_sub", "mul2_x", "mul3_sub1",
                                    "mul4_x", "add2_y"};
        for (auto n : kIn) {
            this->Input(n)
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        }
        static const char* kOut[] = {"y1", "y2", "y3", "y4"};
        for (auto n : kOut) {
            this->Output(n)
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        }

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("opFile.value", "lamb_next_m_v");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(LambNextMV);
} // namespace ops
