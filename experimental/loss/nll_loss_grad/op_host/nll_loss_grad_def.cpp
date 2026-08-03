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
 * \file nll_loss_grad_def.cpp
 * \brief NllLossGrad 算子定义
 */
#include "register/op_def_registry.h"

namespace ops {

// 9 组 dtype 组合：浮点 {float32, bf16, float16} × target {int32, int64, uint8}
// 组合顺序与 kernel schMode 保持一致：
//   0: float32/int32   1: bf16/int32    2: float32/int64
//   3: bf16/int64      4: float32/uint8 5: bf16/uint8
//   6: float16/int32   7: float16/int64 8: float16/uint8

class NllLossGrad : public OpDef {
public:
    explicit NllLossGrad(const char* name) : OpDef(name)
    {
        std::vector<ge::DataType> floatTypes = {ge::DT_FLOAT,   ge::DT_BF16,    ge::DT_FLOAT,
                                                ge::DT_BF16,    ge::DT_FLOAT,   ge::DT_BF16,
                                                ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16};
        std::vector<ge::DataType> targetTypes = {ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_UINT8,
                                                 ge::DT_UINT8, ge::DT_INT32, ge::DT_INT64, ge::DT_UINT8};
        std::vector<ge::Format> ndFormats(floatTypes.size(), ge::FORMAT_ND);

        this->Input("x")
            .ParamType(REQUIRED)
            .DataType(floatTypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats)
            .AutoContiguous();
        this->Input("y_grad")
            .ParamType(REQUIRED)
            .DataType(floatTypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats)
            .AutoContiguous();
        this->Input("target")
            .ParamType(REQUIRED)
            .DataType(targetTypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats)
            .AutoContiguous();
        this->Input("weight")
            .ParamType(REQUIRED)
            .DataType(floatTypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats)
            .AutoContiguous();
        this->Input("total_weight")
            .ParamType(REQUIRED)
            .DataType(floatTypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats)
            .AutoContiguous();
        this->Output("x_grad")
            .ParamType(REQUIRED)
            .DataType(floatTypes)
            .Format(ndFormats)
            .UnknownShapeFormat(ndFormats)
            .AutoContiguous();
        this->Attr("reduction").AttrType(OPTIONAL).String("mean");
        this->Attr("ignore_index").AttrType(OPTIONAL).Int(-100);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(NllLossGrad);
} // namespace ops
