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
 * \file in_training_update_grad_def.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {
// arch35-only, GE-only. Two configs: (dy/x fp16) and (dy/x fp32).
// variance/mean and both outputs are always fp32 (mixed-dtype: their dtype column stays fp32
// while dy/x vary). Format is NDC1HWC0 (6D) for every param, aligned with the shipped A2 binary.
class INTrainingUpdateGrad : public OpDef {
public:
    explicit INTrainingUpdateGrad(const char* name) : OpDef(name)
    {
        this->Input("dy")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0})
            .UnknownShapeFormat({ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0});
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0})
            .UnknownShapeFormat({ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0});
        this->Input("variance")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0})
            .UnknownShapeFormat({ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0});
        this->Input("mean")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0})
            .UnknownShapeFormat({ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0});
        this->Output("res_gamma")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0})
            .UnknownShapeFormat({ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0});
        this->Output("res_beta")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0})
            .UnknownShapeFormat({ge::FORMAT_NDC1HWC0, ge::FORMAT_NDC1HWC0});

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .PrecisionReduceFlag(false);
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(INTrainingUpdateGrad);
} // namespace ops
