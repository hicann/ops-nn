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
 * \file add_example_aicpu.cpp
 * \brief AICPU版本的Add算子kernel实现
 * 
 * 本文件实现了在AICPU(人工智能处理器)上执行的元素级加法操作kernel。
 * 该kernel支持多种数据类型(浮点、整数)，并通过模板特化实现类型安全的高效计算。
 * AICPU是Ascend芯片上的通用CPU核心，用于执行控制逻辑和无法在AI Core上高效实现的计算。
 */

#include "add_example_aicpu.h"

#include <cmath>
#include <string>
#include "cust_cpu_utils.h"
#include "securec.h"
#include "log.h"
#include "status.h"
#include "utils/kernel_util.h"

namespace {
// 常量定义
const char* const kAddExample = "AddExampleAicpu";      // 算子名称常量
const uint32_t kFirstInputIndex = 0;              // 第一个输入张量的索引
const uint32_t kSecondInputIndex = 1;             // 第二个输入张量的索引
const uint32_t kFirstOutputIndex = 0;             // 输出张量的索引
const uint32_t kSuccess = 0;                       // 成功返回码
const uint32_t kParamInvalid = 1;                 // 参数无效返回码
const uint32_t kError = 2;                         // 错误返回码
} 

namespace aicpu {

/*!
 * \brief AICPU kernel主计算函数
 * 
 * 该函数是AICPU kernel的入口点，负责：
 * 1. 参数验证：检查输入输出张量是否有效
 * 2. 数据类型分发：根据输入数据类型选择对应的模板实现
 * 3. 调用类型化的计算函数：执行实际的加法运算
 * 
 * @param ctx CPU kernel上下文对象，用于访问输入输出张量和日志
 * @return kSuccess表示成功，kParamInvalid表示参数无效，kError表示其他错误
 */
uint32_t AddExampleCpuKernel::Compute(CpuKernelContext& ctx)
{
    // 获取输入和输出张量指针
    Tensor* input0 = ctx.Input(kFirstInputIndex);
    Tensor* input1 = ctx.Input(kSecondInputIndex);
    Tensor* output = ctx.Output(kFirstOutputIndex);

    // 参数有效性检查
    if (input0 == nullptr || input1 == nullptr || output == nullptr) {
        KERNEL_LOG_ERROR("Invalid argument: input or output tensor is null");
        return kParamInvalid;
    }

    // 如果输入张量的数据大小为0，直接返回成功
    if (input0->GetDataSize() == 0 || input1->GetDataSize() == 0) {
        return kSuccess;
    }

    // 获取输入数据类型
    auto data_type = static_cast<DataType>(input0->GetDataType());
    
    // 根据数据类型分发到对应的模板实现
    switch (data_type) {
    case DT_FLOAT:
        // 浮点数类型：调用浮点加法模板
        return AddCompute<float>(ctx);
    case DT_INT32:
        // 32位整型：调用整型加法模板
        return AddCompute<int32_t>(ctx);
    case DT_INT64:
        // 64位整型：调用长整型加法模板
        return AddCompute<int64_t>(ctx);
    default:
        // 不支持的数据类型：返回参数无效错误
        KERNEL_LOG_ERROR("Unsupported data type: %d", data_type);
        return kParamInvalid;
    }
    
    return kSuccess;
}

/*!
 * \brief 类型化的加法计算模板函数
 * 
 * 该模板函数实现了具体的加法计算逻辑：
 * 1. 获取输入输出张量的数据指针
 * 2. 遍历所有元素，执行元素级加法
 * 3. 将结果写入输出张量
 * 
 * 该函数通过模板特化支持多种数据类型，保证类型安全和计算效率。
 * 
 * @tparam T 数据类型(float, int32_t, int64_t等)
 * @param ctx CPU kernel上下文对象
 * @return kSuccess表示成功，kParamInvalid表示参数无效
 */
template <typename T>
uint32_t AddExampleCpuKernel::AddCompute(CpuKernelContext& ctx)
{
    // 获取输入和输出张量指针
    Tensor* input0 = ctx.Input(kFirstInputIndex);
    Tensor* input1 = ctx.Input(kSecondInputIndex);
    Tensor* output = ctx.Output(kFirstOutputIndex);

    // 获取输入数据指针
    T* x0 = reinterpret_cast<T*>(input0->GetData());
    if (x0 == nullptr) {
        KERNEL_LOG_ERROR("Failed to get data pointer for input0");
        return kParamInvalid;
    }
    
    T* x1 = reinterpret_cast<T*>(input1->GetData());
    if (x1 == nullptr) {
        KERNEL_LOG_ERROR("Failed to get data pointer for input1");
        return kParamInvalid;
    }
    
    // 获取输出数据指针
    T* y = reinterpret_cast<T*>(output->GetData());
    if (y == nullptr) {
        KERNEL_LOG_ERROR("Failed to get data pointer for output");
        return kParamInvalid;
    }

    // 获取元素总数
    int64_t num_elements = input0->NumElements();
    KERNEL_LOG_INFO("Num of elements to process: %ld", num_elements);

    // 执行元素级加法：遍历所有元素并计算
    // 这是一个简单的for循环加法，适合在CPU上执行
    for (int64_t i = 0; i < num_elements; i++) {
        y[i] = x0[i] + x1[i];
    }

    return kSuccess;
}

// 注册AICPU kernel
// 将AddExampleCpuKernel类注册到kernel注册表，使其可以被系统调用
REGISTER_CPU_KERNEL(kAddExample, AddExampleCpuKernel);

} 
