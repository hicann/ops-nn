/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_focal_loss_grad_def.cpp
 * \brief softmax_focal_loss_grad def
 */

#include "register/op_def_registry.h"

namespace ops {
class SoftmaxFocalLossGrad : public OpDef {
public:
    explicit SoftmaxFocalLossGrad(const char* name) : OpDef(name)
    {
        // 对齐 A2 ini: weight 与 pred 同 dtype, 仅 (f16,f16) / (f32,f32) 两组
        this->Input("pred")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("target")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("dout")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("weight")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("grad")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("gamma").AttrType(OPTIONAL).Float(2.0);
        this->Attr("alpha").AttrType(OPTIONAL).Float(0.25);
        this->Attr("reduction").AttrType(OPTIONAL).String("mean");

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .PrecisionReduceFlag(false);
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(SoftmaxFocalLossGrad);
} // namespace ops
