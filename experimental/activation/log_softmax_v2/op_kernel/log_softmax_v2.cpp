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
 * \file log_softmax_v2.cpp
 * \brief
*/

#include "log_softmax_v2.h"

// ========================== 常量定义  ==========================
namespace {
    // 缓冲区与分片限制
    constexpr uint32_t MAX_L1_WORKSPACE = 8192;
    constexpr uint32_t MAX_CHUNK_SIZE = 2048;
    constexpr uint32_t DEFAULT_ALIGN_BLOCK = 128;

    // 形状判定阈值
    constexpr uint32_t SMALL_AXIS_LIMIT = 16;
    constexpr uint32_t SMALL_BATCH_LIMIT = 1024;
    constexpr uint32_t LARGE_SHAPE_LIMIT = 8192;

    // 不同数据类型的对齐步长 
    struct AlignConfig {
        uint32_t smallBatchAlign; 
        uint32_t defaultAlign;    
    };
}

// ========================== 模板 Traits (类型映射) ==========================
template<typename T> struct LogSoftmaxTraits;

template<> struct LogSoftmaxTraits<float> {
    using Op2D = LogSoftmax;
    using Op3D = LogSoftmaxCol;
    static constexpr AlignConfig ALIGN = {16, 128};
};

template<> struct LogSoftmaxTraits<half> {
    using Op2D = LogSoftmaxHalf;
    using Op3D = LogSoftmaxHalfCol;
    static constexpr AlignConfig ALIGN = {64, 256};
};

template<> struct LogSoftmaxTraits<bfloat16_t> {
    using Op2D = LogSoftmaxBf16;
    using Op3D = LogSoftmaxBF16Col;
    static constexpr AlignConfig ALIGN = {128, 64}; // BF16原逻辑比较特殊
};

// ========================== 核心逻辑实现 ==========================
template<typename T>
__aicore__ inline void ProcessLogSoftmaxInternal(GM_ADDR input, GM_ADDR out, const LogSoftmaxV2TilingData& tilingData) {
    using Traits = LogSoftmaxTraits<T>;
    const uint32_t dims = tilingData.dims;
    const uint64_t* shape = tilingData.shape;
    const int blockIdx = GetBlockIdx();
    const int blockDim = GetBlockNum();

    if (dims == 2) {
        typename Traits::Op2D op;
        op.Init(input, out, shape[0], shape[1]);
        int start = blockIdx * shape[0] / blockDim;
        int end = (blockIdx + 1) * shape[0] / blockDim;
        if (end > start) op.Process1(start, end);
    } 
    else if (dims == 3) {
        typename Traits::Op3D op;
        
        // 逻辑1: 轴较小时的分片处理
        if (shape[1] <= SMALL_AXIS_LIMIT) {
            int size = MAX_L1_WORKSPACE / shape[1];
            size = (int(size / DEFAULT_ALIGN_BLOCK)) * DEFAULT_ALIGN_BLOCK;
            size = mmin((int)MAX_CHUNK_SIZE, size);
            
            op.Init(input, out, shape[1], shape[2], 1);
            int chunk = ((shape[2] + size - 1) / size);
            int total = chunk * shape[0];
            int start = blockIdx * total / blockDim;
            int end = (blockIdx + 1) * total / blockDim;
            
            if (end <= start) return;
            for (int i = start; i != end; i++) {
                op.Process4(i / chunk, i % chunk, (i % chunk) + 1, size);
            }
        } 
        // 逻辑2: 大形状场景 (根据类型逻辑略有不同)
        else if ((std::is_same_v<T, float> && shape[2] >= LARGE_SHAPE_LIMIT) || 
                 (!std::is_same_v<T, float> && shape[2] >= (uint32_t)blockDim * MAX_CHUNK_SIZE)) {
            
            op.Init(input, out, shape[1], shape[2], 1);
            int chunk = ((shape[2] + (MAX_CHUNK_SIZE - 1)) / MAX_CHUNK_SIZE);
            int total = chunk * shape[0];
            int start = blockIdx * total / blockDim;
            int end = (blockIdx + 1) * total / blockDim;
            
            if (end <= start) return;
            for (int i = start; i != end; i++) {
                int batch = i / chunk;
                int index = i % chunk;
                if constexpr (std::is_same_v<T, bfloat16_t>) {
                    op.Process2(batch, index, index + 1);
                } else {
                    op.Process3(batch, index, index + 1);
                }
            }
        } 
        // 逻辑3: 小 Batch 场景对齐逻辑
        else if ((!std::is_same_v<T, bfloat16_t> && shape[0] * shape[2] <= SMALL_BATCH_LIMIT) ||
                 (std::is_same_v<T, bfloat16_t> && shape[0] * shape[2] >= SMALL_BATCH_LIMIT)) {
            
            op.Init(input, out, shape[1], shape[2], 1);
            uint32_t align = Traits::ALIGN.smallBatchAlign;
            int total = ((shape[2] + (align - 1)) / align) * shape[0];
            int start = blockIdx * total / blockDim;
            int end = (blockIdx + 1) * total / blockDim;
            if (end > start) op.Process1(start, end);
        } 
        // 逻辑4: 默认场景
        else {
            op.Init(input, out, shape[1], shape[2], 0);
            uint32_t align = Traits::ALIGN.defaultAlign;
            int total = ((shape[2] + (align - 1)) / align) * shape[0];
            int start = blockIdx * total / blockDim;
            int end = (blockIdx + 1) * total / blockDim;
            if constexpr (std::is_same_v<T, bfloat16_t>) {
                if (end > start) op.Process3(start, end);
            } else {
                if (end > start) op.Process2(start, end);
            }
        }
    }
}

// ========================== Kernel 入口 ==========================
template <uint32_t schMode>
__global__ __aicore__ void log_softmax_v2(GM_ADDR input, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(LogSoftmaxV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(LogSoftmaxV2TilingData, tilingData, tiling);

    if constexpr (std::is_same_v<DTYPE_INPUT, float>) {
        ProcessLogSoftmaxInternal<float>(input, out, tilingData);
    }
    else if constexpr (std::is_same_v<DTYPE_INPUT, half>) {
        ProcessLogSoftmaxInternal<half>(input, out, tilingData);
    }
    else if constexpr (std::is_same_v<DTYPE_INPUT, bfloat16_t>) {
        ProcessLogSoftmaxInternal<bfloat16_t>(input, out, tilingData);
    }
}