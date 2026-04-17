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
 * \file relu6_def.cpp
 * \brief Relu6 算子定义，声明输入输出和算子配置
 *
 * 算子公式: y = min(max(x, 0), 6)
 * 支持数据类型: float16, float, int32, bfloat16
 * 目标芯片: Ascend950 (arch35)
 */

#include "register/op_def_registry.h"

namespace ops {
class Relu6 : public OpDef {
public:
    explicit Relu6(const char* name) : OpDef(name)
    {
        this->Input("x")                                        // 输入 x 定义
            .ParamType(REQUIRED)                                // 必选输入
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_BF16})  // 支持数据类型
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}) // 支持 format 格式
            .UnknownShapeFormat({ge::FORMAT_ND})                // 未确定大小 shape 对应 format
            .AutoContiguous();                                  // 内存自动连续化
        this->Output("y")                                       // 输出 y 定义
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        // Ascend950 AI Core 配置
        OpAICoreConfig aiCoreConfig;
        aiCoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "relu6");     // 指定 Kernel 入口文件名
        this->AICore().AddConfig("ascend950", aiCoreConfig);
    }
};
OP_ADD(Relu6);  // 注册算子到算子信息库
} // namespace ops
