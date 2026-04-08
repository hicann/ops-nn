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
 * \file add_example.cpp
 * \brief Add算子的kernel入口函数
 * 
 * 本文件实现了Add算子在AI Core上的kernel入口函数。
 * 该函数根据tiling key(数据类型)分发到相应的模板实现。
 * kernel使用tiling策略将计算任务划分到多个AI Core上并行执行。
 */

#include "add_example.h"

// 定义Add算子的tiling key枚举
// tiling key用于区分不同数据类型的实现策略
enum class AddExampleTilingKey : uint32_t
{
    TILING_KEY_EXAMPLE_FLOAT = 0,  // 浮点数类型的tiling key
    TILING_KEY_EXAMPLE_INT32 = 1,  // 32位整型类型的tiling key
};

// Add算子的kernel入口函数
// 该函数是AI Core执行的入口点，根据模板参数schMode选择对应的数据类型实现
// @param x: 输入张量x的GM地址
// @param y: 输入张量y的GM地址
// @param z: 输出张量z的GM地址
// @param workspace: 工作空间的GM地址
// @param tiling: tiling数据的GM地址，包含分块和内存管理信息
template <uint32_t schMode>
__global__ __aicore__ void add_example(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    // 注册默认的tiling数据结构
    REGISTER_TILING_DEFAULT(AddExampleTilingData);
    // 从GM内存获取tiling数据
    GET_TILING_DATA_WITH_STRUCT(AddExampleTilingData, tilingData, tiling);
    
    // 根据tiling key(schMode)分发到不同数据类型的实现
    if constexpr (schMode == static_cast<uint32_t>(AddExampleTilingKey::TILING_KEY_EXAMPLE_FLOAT)) {
        // 浮点数类型的实现
        NsAddExample::AddExample<float> op; // 算子kernel实例获取
        op.Init(x, y, z, &tilingData);      // 算子kernel实例初始化
        op.Process();                       // 算子kernel实例执行
    }
    if constexpr (schMode == static_cast<uint32_t>(AddExampleTilingKey::TILING_KEY_EXAMPLE_INT32)) {
        // 32位整型类型的实现
        NsAddExample::AddExample<int32_t> op; // 算子kernel实例获取
        op.Init(x, y, z, &tilingData);        // 算子kernel实例初始化
        op.Process();                         // 算子kernel实例执行
    }
}
