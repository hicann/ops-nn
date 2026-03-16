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
 * \file max_pool3d_grad.cpp
 * \brief
 */

#include "register/op_def_registry.h"

namespace ops {

class MaxPool3DGrad : public OpDef {
public:
    explicit MaxPool3DGrad(const char* name) : OpDef(name)
    {
        this->Input("orig_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW})
            .UnknownShapeFormat({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW})
            .AutoContiguous();
        this->Input("orig_y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW})
            .UnknownShapeFormat({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW})
            .AutoContiguous();
        this->Input("grads")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW})
            .UnknownShapeFormat({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW})
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW})
            .UnknownShapeFormat({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW});
        this->Attr("ksize").AttrType(REQUIRED).ListInt();
        this->Attr("strides").AttrType(REQUIRED).ListInt();
        this->Attr("padding").AttrType(OPTIONAL).String("SAME");
        this->Attr("pads").AttrType(REQUIRED).ListInt();
        this->Attr("data_format").AttrType(OPTIONAL).String("NCDHW");

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .ExtendCfgInfo("opFile.value", "max_pool3d_grad")
            .ExtendCfgInfo("opInterface.value", "max_pool3d_grad");

        this->AICore().AddConfig("ascend950", aicore_config);
    }
};

OP_ADD(MaxPool3DGrad);
} // namespace ops
