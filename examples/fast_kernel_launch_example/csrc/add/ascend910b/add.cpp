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
 * \file add.cpp
 * \brief 基于Ascend AI处理器的张量加法操作实现
 * 
 * 本文件实现了一个自定义的PyTorch扩展，用于在Ascend NPU上执行元素级张量加法操作。
 * 该实现使用了AscendC编程模型，通过流水线和双缓冲技术优化数据传输和计算性能。
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "kernel_operator.h"
#include "platform/platform_ascendc.h"
#include <type_traits>

namespace ascend_ops {
namespace Add {

// 注册算子接口定义，声明函数签名
TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def("add(Tensor x, Tensor y) -> Tensor");
}

// Meta函数：在编译时确定输出张量的形状和数据类型，不进行实际计算
// 该函数在CPU上执行，用于张量形状推断和类型推导
torch::Tensor add_meta(const torch::Tensor &x, const torch::Tensor &y)
{
    // 检查两个输入张量的形状是否相同
    TORCH_CHECK(x.sizes() == y.sizes(), "The shapes of x and y must be the same.");
    // 创建一个与输入张量x形状和类型相同的空张量作为输出
    auto z = torch::empty_like(x);
    return z;
}

// 注册Meta实现，用于形状推导
TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, Meta, m)
{
    m.impl("add", add_meta);
}

// 计算tiling参数，用于优化数据分块和并行处理
// 返回值：(numBlocks, blockLength, tileSize)
//   - numBlocks: 使用的AI Core数量
//   - blockLength: 每个AI Core处理的元素数量
//   - tileSize: 每次tile处理的数据块大小
std::tuple<int64_t, int64_t, int64_t> calc_tiling_params(int64_t totalLength)
{
    // 每个AI Core最少处理的元素数量，避免任务过小导致并行效率低
    constexpr static int64_t MIN_ELEMS_PER_CORE = 1024;
    // 流水线深度，用于实现数据传输与计算的并行
    constexpr static int64_t PIPELINE_DEPTH = 2;
    // 缓冲区数量，用于双缓冲技术
    constexpr static int64_t BUFFER_NUM = 3;
    
    // 获取AscendC平台实例
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubSize;
    // 获取Unified Buffer(UB)的内存大小
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    // 获取AI Core数量
    int64_t coreNum = ascendcPlatform->GetCoreNumAiv();
    TORCH_CHECK(coreNum > 0, "coreNum must be positive.");
    
    // 计算实际使用的AI Core数量，不超过总数据所需的Core数
    int64_t numBlocks = std::min(coreNum, (totalLength + MIN_ELEMS_PER_CORE - 1) / MIN_ELEMS_PER_CORE);
    // 计算每个AI Core需要处理的元素数量
    int64_t blockLength = (totalLength + numBlocks - 1) / numBlocks;
    // 计算tile大小，即每次处理的数据块大小
    int64_t tileSize = ubSize / PIPELINE_DEPTH / BUFFER_NUM;
    
    return std::make_tuple(numBlocks, blockLength, tileSize);
}

// 加法kernel函数，在AI Core上执行元素级加法操作
// 使用双缓冲和流水线技术，实现数据传输与计算的并行
template <typename T>
__global__ __aicore__ void add_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR z, int64_t totalLength, int64_t blockLength, uint32_t tileSize)
{
    // 流水线深度，支持数据传输与计算的重叠
    constexpr static int64_t PIPELINE_DEPTH = 2;
    
    // 创建TPipe对象，用于管理数据流水线
    AscendC::TPipe pipe;
    // 定义全局内存张量，用于访问GM(Global Memory)中的数据
    AscendC::GlobalTensor<T> xGm, yGm, zGm;
    // 定义输入队列，用于VECIN(向量输入)方向的数据传输
    AscendC::TQue<AscendC::QuePosition::VECIN, PIPELINE_DEPTH> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, PIPELINE_DEPTH> inQueueY;
    // 定义输出队列，用于VECOUT(向量输出)方向的数据传输
    AscendC::TQue<AscendC::QuePosition::VECOUT, PIPELINE_DEPTH> outQueueZ;
    
    // 初始化队列缓冲区，分配大小为tileSize的内存空间
    pipe.InitBuffer(inQueueX, PIPELINE_DEPTH, tileSize);
    pipe.InitBuffer(inQueueY, PIPELINE_DEPTH, tileSize);
    pipe.InitBuffer(outQueueZ, PIPELINE_DEPTH, tileSize);
    
    // 设置全局内存缓冲区，每个AI Core处理不同的数据块
    xGm.SetGlobalBuffer((__gm__ T *)x + blockLength * AscendC::GetBlockIdx());
    yGm.SetGlobalBuffer((__gm__ T *)y + blockLength * AscendC::GetBlockIdx());
    zGm.SetGlobalBuffer((__gm__ T *)z + blockLength * AscendC::GetBlockIdx());

    // 计算当前AI Core需要处理的元素数量
    int64_t currentBlockLength = totalLength - AscendC::GetBlockIdx() * blockLength;
    if (currentBlockLength > blockLength) {
      currentBlockLength = blockLength;
    }
    
    // 计算每个tile包含的元素数量
    int64_t elementNumPerTile = tileSize / sizeof(T);
    // 计算完整的tile数量
    int64_t tileNum = currentBlockLength / elementNumPerTile;
    // 计算最后一个不完整tile中的元素数量
    int64_t tailTileElementNum = currentBlockLength - tileNum * elementNumPerTile;

    // 处理完整的tile
    for (int64_t i = 0; i < tileNum; ++i) {
        int64_t offset = i * elementNumPerTile;
        
        // 阶段1：CopyIn - 将数据从GM搬运到本地内存
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = elementNumPerTile * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        
        // 分配本地tensor内存
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        AscendC::LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();
        // 执行数据拷贝：从GM到本地内存
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        AscendC::DataCopyPad(yLocal, yGm[offset], copyParams, padParams);
        // 将本地tensor放入队列
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
        
        // 阶段2：Compute - 执行加法计算
        xLocal = inQueueX.DeQue<T>();
        yLocal = inQueueY.DeQue<T>();
        // 分配输出tensor内存
        AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
        // 执行向量加法操作
        AscendC::Add(zLocal, xLocal, yLocal, elementNumPerTile);
        // 将结果放入输出队列
        outQueueZ.EnQue(zLocal);
        // 释放输入tensor内存
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
        
        // 阶段3：CopyOut - 将结果从本地内存搬运到GM
        zLocal = outQueueZ.DeQue<T>();
        // 执行数据拷贝：从本地内存到GM
        AscendC::DataCopyPad(zGm[offset], zLocal, copyParams);
        // 释放输出tensor内存
        outQueueZ.FreeTensor(zLocal);
    }

    // 处理最后一个不完整的tile
    if (tailTileElementNum > 0) {
        int64_t offset = tileNum * elementNumPerTile;
        
        // CopyIn
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = tailTileElementNum * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        AscendC::LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        AscendC::DataCopyPad(yLocal, yGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
        
        // Compute
        xLocal = inQueueX.DeQue<T>();
        yLocal = inQueueY.DeQue<T>();
        AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
        AscendC::Add(zLocal, xLocal, yLocal, tailTileElementNum);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
        
        // CopyOut
        zLocal = outQueueZ.DeQue<T>();
        AscendC::DataCopyPad(zGm[offset], zLocal, copyParams);
        outQueueZ.FreeTensor(zLocal);
    }
}

// NPU实现函数，在Ascend NPU上执行加法操作
// 负责设备管理、内存分配、kernel启动等任务
torch::Tensor add_npu(const torch::Tensor &x, const torch::Tensor &y)
{
    // OptionalDeviceGuard确保后续操作在正确的设备上下文执行
    // 它会记录当前设备状态，执行完作用域代码后自动恢复
    const c10::OptionalDeviceGuard guard(x.device());
    
    // 调用meta函数获取输出张量的形状
    auto z = add_meta(x, y);
    
    // 获取当前NPU流
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    
    // 计算tiling参数
    int64_t totalLength, numBlocks, blockLength, tileSize;
    totalLength = x.numel();
    std::tie(numBlocks, blockLength, tileSize) = calc_tiling_params(totalLength);
    
    // 获取输入输出张量的设备地址
    auto x_ptr = (GM_ADDR)x.data_ptr();
    auto y_ptr = (GM_ADDR)y.data_ptr();
    auto z_ptr = (GM_ADDR)z.data_ptr();
    
    // 定义ACL调用的lambda函数，在NPU上执行
    auto acl_call = [=]() -> int {
        // 根据输入张量的数据类型进行分发
        AT_DISPATCH_SWITCH(
            x.scalar_type(), "add_npu",
            // Float32类型处理
            AT_DISPATCH_CASE(torch::kFloat32, [&] {
                using scalar_t = float;
                add_kernel<scalar_t><<<numBlocks, nullptr, stream>>>(x_ptr, y_ptr, z_ptr, totalLength, blockLength, tileSize);
            })
            // Float16类型处理
            AT_DISPATCH_CASE(torch::kFloat16, [&] {
                using scalar_t = half;
                add_kernel<scalar_t><<<numBlocks, nullptr, stream>>>(x_ptr, y_ptr, z_ptr, totalLength, blockLength, tileSize);
            })
            // Int32类型处理
            AT_DISPATCH_CASE(torch::kInt32, [&] {
                using scalar_t = int32_t;
                add_kernel<scalar_t><<<numBlocks, nullptr, stream>>>(x_ptr, y_ptr, z_ptr, totalLength, blockLength, tileSize);
            })
        );
        return 0;
    };
    
    // 通过OpCommand执行API调用
    at_npu::native::OpCommand::RunOpApi("Add", acl_call);
    
    return z;
}

// 注册NPU实现
TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("add", add_npu);
}

} // namespace Add
} // namespace ascend_ops
