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
 * \file p_relu_grad_redue_def.cpp
 * \brief
 */
#include "register/op_def_registry.h"

namespace ops {
class PReluGradReduce : public OpDef
{
public:
    explicit PReluGradReduce(const char* name) : OpDef(name)
    {
        this->Input("grads")                               // 输入x定义
            .ParamType(REQUIRED)                                   // 必选输入
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT}) // 支持数据类型
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})                // 支持format格式
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})    // 未确定大小shape对应format格式
            .AutoContiguous();                                     // 内存自动连续化
        this->Input("features")                                // 输入权重weight定义
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("weights") // 输入上游梯度gradient定义
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("updates") // 输入上游梯度gradient定义
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("da") // 输出y定义
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_FLOAT})
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
            .ExtendCfgInfo("opInterface.value", "prelu_grad_reduce")
            .ExtendCfgInfo("opFile.value", "p_relu_grad_reduce_apt"); // 这里制定的值会对应到kernel入口文件名.cpp
        this->AICore().AddConfig("ascend950", aicoreConfig); // 其他的soc版本补充部分配置项
    }
};
OP_ADD(PReluGradReduce); // 添加算子信息库
} // namespace ops