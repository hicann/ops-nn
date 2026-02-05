/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sigmoid_cross_entropy_with_logits_v2_def.cpp
 * \brief sigmoid_cross_entropy_with_logits_v2_def.cpp
 */

#include "register/op_def_registry.h"

namespace ops {
    class SigmoidCrossEntropyWithLogitsV2 : public OpDef {
    public:
        explicit SigmoidCrossEntropyWithLogitsV2(const char *name) : OpDef(name) {
            this->Input("predict")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("target")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("weight")
                .ParamType(OPTIONAL)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("pos_weight")
                .ParamType(OPTIONAL)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("loss")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("reduction").AttrType(OPTIONAL).String("mean");

            OpAICoreConfig aicoreConfig;
            aicoreConfig.DynamicCompileStaticFlag(true)
                .DynamicRankSupportFlag(true)
                .DynamicShapeSupportFlag(true)
                .PrecisionReduceFlag(false)
                .ExtendCfgInfo("opFile.value", "sigmoid_cross_entropy_with_logits_v2_apt");
            this->AICore().AddConfig("ascend950", aicoreConfig);
        }
    };

    OP_ADD(SigmoidCrossEntropyWithLogitsV2);
} // namespace ops

