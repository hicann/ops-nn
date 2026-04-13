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
 * \file softsign_grad_def.cpp
 * \brief SoftsignGrad 算子定义，声明输入输出和算子配置
 *
 * 数学公式: output = gradients / (1 + |features|)^2
 *
 * 迭代一目标：单 TilingKey 骨架 (D_T=half, BUFFER_MODE=0)
 *            预埋多 dtype/多缓冲骨架，后续迭代扩展
 */
#include "register/op_def_registry.h"

namespace ops {
class SoftsignGrad : public OpDef {
public:
    explicit SoftsignGrad(const char* name) : OpDef(name)
    {
        // 输入 gradients：上游梯度张量
        this->Input("gradients")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // 输入 features：Softsign 前向输入特征
        this->Input("features")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // 输出 output：计算得到的梯度输出
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // Ascend950 (arch35) 配置
        OpAICoreConfig aicoreConf;
        aicoreConf.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "softsign_grad_apt");
        this->AICore().AddConfig("ascend950", aicoreConf);
        this->AICore().AddConfig("ascend910b", aicoreConf);
    }
};
OP_ADD(SoftsignGrad);
} // namespace ops
