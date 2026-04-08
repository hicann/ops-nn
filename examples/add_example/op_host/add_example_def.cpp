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
 * \file add_example_def.cpp
 * \brief Add算子的定义
 * 
 * 本文件定义了Add算子的接口，包括输入输出规格、支持的数据类型、格式以及AI Core编译配置。
 * 算子定义对于在CANN框架中注册和验证算子是必需的。
 */
#include "register/op_def_registry.h"

namespace ops {
/*!
 * \brief Add算子类定义
 * 
 * 该类定义了执行元素级加法的Add算子，接收两个输入张量并产生一个相同形状的输出张量。
 * 
 * 支持的数据类型: FLOAT, INT32
 * 支持的格式: ND (n维格式)
 */
class AddExample : public OpDef
{
public:
    /*!
     * \brief Add算子的构造函数
     * \param name 算子实例的名称
     *
     * 定义算子接口，包括：
     * - 两个必需输入 (x1, x2)
     * - 一个必需输出 (y)
     * - 支持的数据类型和格式
     * - 针对不同SOC版本的AI Core编译配置
     */
    explicit AddExample(const char* name) : OpDef(name)
    {
        // 定义输入x1的规格
        this->Input("x1")                                       // 输入x1名称定义
            .ParamType(REQUIRED)                                // 必选输入参数
            .DataType({ge::DT_FLOAT, ge::DT_INT32})             // 支持的数据类型：浮点和32位整数
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})             // 支持的格式：n维格式
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND}) // 未确定大小shape对应的format格式
            .AutoContiguous();                                  // 内存自动连续化
        // 定义输入x2的规格
        this->Input("x2")                                       // 输入x2名称定义
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();
        // 定义输出y的规格
        this->Output("y") // 输出y名称定义
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // AI Core编译配置
        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)    // 启用静态动态编译标志
            .DynamicFormatFlag(false)                   // 禁用动态格式标志
            .DynamicRankSupportFlag(true)               // 启用动态rank支持
            .DynamicShapeSupportFlag(true)              // 启用动态shape支持
            .NeedCheckSupportFlag(false)                 // 禁用检查支持标志
            .PrecisionReduceFlag(true)                  // 启用精度降低标志
            .ExtendCfgInfo("opFile.value", "add_example");    // 指定的kernel入口文件名
        // 为不同SOC版本添加AI Core配置
        this->AICore().AddConfig("ascend910b", aicoreConfig);  // Ascend 910B芯片配置
        this->AICore().AddConfig("ascend910_93", aicoreConfig); // Ascend 910A芯片配置
        this->AICore().AddConfig("ascend950", aicoreConfig);    // Ascend 950芯片配置
    }
};

// 将Add算子添加到算子库中
OP_ADD(AddExample); 
} // namespace ops