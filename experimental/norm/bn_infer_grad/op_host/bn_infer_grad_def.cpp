/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file bn_infer_grad_def.cpp
 * \brief BnInferGrad 算子定义，声明输入输出和算子配置
 */
#include "register/op_def_registry.h"

namespace ops {
class BnInferGrad : public OpDef {
public:
    explicit BnInferGrad(const char* name) : OpDef(name)
    {
        // 3 种 grads dtype（fp16/fp32/bf16）× 4 种 format（ND/NCHW/NHWC/NC1HWC0）= 12 种组合
        // ND format 用于 aclnn API 层（aclCreateTensor 默认格式）
        // NCHW/NHWC/NC1HWC0 用于框架层 format 匹配
        // scale/batch_variance 始终为 fp32 + FORMAT_ND
        this->Input("grads")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
                     ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
                     ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
                                ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
                                ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0})
            .AutoContiguous();

        this->Input("scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Input("batch_variance")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                       ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("x_backprop")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
                       ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                     ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
                     ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
                     ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW,
                                ge::FORMAT_NHWC, ge::FORMAT_NHWC, ge::FORMAT_NHWC,
                                ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0, ge::FORMAT_NC1HWC0})
            .AutoContiguous();

        this->Attr("epsilon")
            .AttrType(OPTIONAL)
            .Float(0.0001f);

        // format_mode 属性：指定 grads 的逻辑数据布局。
        // 框架的 AutoContiguous 会将 NHWC 等格式归一化为 ND，
        // 导致 tiling 侧无法通过 GetFormat() 区分 NCHW 与 NHWC。
        // 因此通过显式属性传递格式信息。
        // 取值：0=NCHW(默认) / 1=NHWC / 2=NC1HWC0
        this->Attr("format_mode")
            .AttrType(OPTIONAL)
            .Int(0);

        OpAICoreConfig aicoreConfig910B;
        aicoreConfig910B.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "bn_infer_grad");
        this->AICore().AddConfig("ascend910b", aicoreConfig910B);
    }
};
OP_ADD(BnInferGrad);
} // namespace ops
