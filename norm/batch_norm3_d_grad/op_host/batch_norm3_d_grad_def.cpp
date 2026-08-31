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
 * \file batch_norm3_d_grad.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class BatchNorm3DGrad : public OpDef {
public:
    explicit BatchNorm3DGrad(const char* name) : OpDef(name)
    {
        // input
        this->Input("y_backprop")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16,
                       ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC,
                     ge::FORMAT_NHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC})
            .UnknownShapeFormat({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC,
                                 ge::FORMAT_NHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC})
            .AutoContiguous();
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16,
                       ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC,
                     ge::FORMAT_NHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC})
            .UnknownShapeFormat({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC,
                                 ge::FORMAT_NHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC})
            .AutoContiguous();
        this->Input("scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("reserve_space_1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("reserve_space_2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // output
        this->Output("x_backprop")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16,
                       ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC,
                     ge::FORMAT_NHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC})
            .UnknownShapeFormat({ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NHWC,
                                 ge::FORMAT_NHWC, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC});
        this->Output("scale_backprop")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("offset_backprop")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("reserve_space_4")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("reserve_space_5")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("epsilon").AttrType(OPTIONAL).Float(1e-4);
        this->Attr("data_format").AttrType(OPTIONAL).String("NCDHW");
        this->Attr("is_training").AttrType(OPTIONAL).Bool(true);
        OpAICoreConfig regbaseCfg;
        regbaseCfg.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "batch_norm3_d_grad")
            .ExtendCfgInfo("opInterface.value", "batch_norm3_d_grad");
        this->AICore().AddConfig("ascend950", regbaseCfg);
    }
};

OP_ADD(BatchNorm3DGrad);
} // namespace ops
