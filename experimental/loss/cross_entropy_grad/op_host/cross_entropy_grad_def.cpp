/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Pei Haobo<@xiaopei-1>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cross_entropy_grad_def.cpp
 * \brief CrossEntropyGrad算子的定义
 *
 * 本文件定义了CrossEntropyGrad算子的接口，包括输入输出规格、支持的数据类型、格式以及AI Core编译配置。
 */

#include "register/op_def_registry.h"

namespace ops {
/*!
 * \brief CrossEntropyGrad算子类定义
 *
 * 计算CrossEntropy的梯度：
 * grad_input = softmax(logits) - one_hot(target)
 *
 * 支持2种数据类型组合:
 *   - logits: FLOAT,   target: INT32   (tiling key 0)
 *   - logits: FLOAT16, target: INT32  (tiling key 1)
 *   - logits: BFLOAT16, target: INT32   (tiling key 2)
 */
class CrossEntropyGrad : public OpDef
{
public:
    explicit CrossEntropyGrad(const char* name) : OpDef(name)
    {
        this->Input("logits")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("target")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("grad_input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "cross_entropy_grad");
        this->AICore().AddConfig("ascend910b", aicoreConfig);
    }
};

OP_ADD(CrossEntropyGrad);
} // namespace ops
