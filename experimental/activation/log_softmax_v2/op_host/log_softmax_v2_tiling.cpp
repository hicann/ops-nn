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
 * \file log_softmax_v2_tiling.cpp
 * \brief
 */
#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/log_softmax_v2_tiling_data.h"
#include "../op_kernel/log_softmax_v2_tiling_key.h"

namespace optiling {

using namespace Ops::NN::OpTiling;

// ========================== 常量定义 ==========================
namespace {
    constexpr uint32_t MAX_SHAPE_DIMS = 8;
    constexpr uint32_t INITIAL_DIM_VAL = 1;
    constexpr int64_t AXIS_L_INIT = 9999;
    constexpr int64_t AXIS_INVALID = -1;
    
    // 维度常量
    constexpr uint32_t DIMS_2D = 2;
    constexpr uint32_t DIMS_3D = 3;

    // 阈值与对齐因子
    constexpr uint32_t SMALL_BATCH_THRESHOLD = 1024;
    constexpr uint32_t CHUNK_SIZE_2048 = 2048;
    constexpr uint32_t CHUNK_SIZE_8192 = 8192;
    constexpr uint32_t SMALL_AXIS_THRESHOLD = 16;
    constexpr uint32_t BLOCK_FACTOR_40 = 40;
    
    // 不同类型的对齐单位
    constexpr uint32_t ALIGN_15 = 15;
    constexpr uint32_t ALIGN_16 = 16;
    constexpr uint32_t ALIGN_63 = 63;
    constexpr uint32_t ALIGN_64 = 64;
    constexpr uint32_t ALIGN_127 = 127;
    constexpr uint32_t ALIGN_128 = 128;
    constexpr uint32_t ALIGN_255 = 255;
    constexpr uint32_t ALIGN_256 = 256;
    constexpr uint32_t ALIGN_2047 = 2047;
    
    constexpr int64_t CORE_NUM_MIN_FACTOR = 2;
}

struct LogSoftmaxV2CompileInfo {};

static ge::graphStatus TilingParseForLogSoftmaxV2([[maybe_unused]] gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    coreNum = ascendcPlatform.GetCoreNum();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize <= 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    size_t usrSize = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1); 
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);

    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus LogSoftmaxV2TilingFunc(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);

    // 获取输入shape并增加非空判断
    uint32_t shape[MAX_SHAPE_DIMS];
    for(uint32_t i = 0; i < MAX_SHAPE_DIMS; i++) shape[i] = INITIAL_DIM_VAL;
    
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1_shape);
    
    uint32_t dims = x1_shape->GetStorageShape().GetDimNum();

    // 获取axis属性并增加非空判断
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    
    auto *attr0 = attrs->GetListInt(0);
    int64_t axis = AXIS_INVALID, axisL = AXIS_L_INIT, axisR = AXIS_INVALID;

    if (attr0 != nullptr && attr0->GetSize() > 0) {
        const int64_t* data_ptr = attr0->GetData();
        OP_CHECK_NULL_WITH_CONTEXT(context, data_ptr);

        for(size_t i = 0; i < attr0->GetSize(); i++) {
            int64_t curl = data_ptr[i];
            curl = (curl == AXIS_INVALID) ? static_cast<int64_t>(dims - 1) : curl;
            axisL = curl < axisL ? curl : axisL;
            axisR = curl > axisR ? curl : axisR;
        }
        int index = 0;
        for (uint32_t i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) {
            if(static_cast<int64_t>(i) >= axisL && static_cast<int64_t>(i) <= axisR) {
                shape[index] *= x1_shape->GetStorageShape().GetDim(i);
                if(static_cast<int64_t>(i) == axisR) { axis = index; index++; }
            } else {
                shape[index] = x1_shape->GetStorageShape().GetDim(i);
                index++;
            }
        }
        dims = static_cast<uint32_t>(static_cast<int64_t>(dims) + axisL - axisR);
    }
    
    uint64_t ubSize;
    int64_t coreNum;
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, coreNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    
    LogSoftmaxV2TilingData* tiling = context->GetTilingData<LogSoftmaxV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    
    OP_CHECK_IF(
        memset_s(tiling, sizeof(LogSoftmaxV2TilingData), 0, sizeof(LogSoftmaxV2TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    
    const auto* input_tensor = context->GetInputTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input_tensor);
    auto dt = input_tensor->GetDataType();

    // 重新组织维度逻辑
    if (axis == static_cast<int64_t>(dims - 1)) {
        if (axis == 0) {
            shape[1] = shape[0];
            shape[0] = INITIAL_DIM_VAL;
        } else {
            for (int i = 1; i < axis; i++) {
                shape[0] *= shape[i];
            }
            shape[1] = shape[axis];
        }
        dims = DIMS_2D;
        axis = 1;
        int64_t requiredCore = static_cast<int64_t>(shape[0]) * CORE_NUM_MIN_FACTOR;
        coreNum = requiredCore < coreNum ? requiredCore : coreNum;
    } else {
        for (uint32_t i = static_cast<uint32_t>(axis + 2); i < dims; i++) {
            shape[axis + 1] *= shape[i];
        }
        if (axis == 0) {
            shape[2] = shape[1];
            shape[1] = shape[0];
            shape[0] = INITIAL_DIM_VAL;
        } else {
            for (int i = 1; i < axis; i++) {
                shape[0] *= shape[i];
            }
            shape[1] = shape[axis];
            shape[2] = shape[axis + 1];
        }
        dims = DIMS_3D;
        axis = 1;

        int32_t total = 0;
        if (dt == ge::DT_FLOAT16) {
            if (shape[0] * shape[2] <= SMALL_BATCH_THRESHOLD) {
                total = shape[0] * ((shape[2] + ALIGN_63) / ALIGN_64);
            } else if (shape[2] >= BLOCK_FACTOR_40 * CHUNK_SIZE_2048 || shape[1] <= SMALL_AXIS_THRESHOLD) {
                total = ((shape[2] + ALIGN_2047) / CHUNK_SIZE_2048) * shape[0];
            } else {
                total = shape[0] * ((shape[2] + ALIGN_255) / ALIGN_256);
            }
        } else if (dt == ge::DT_FLOAT) {
            if (shape[2] >= CHUNK_SIZE_8192 || shape[1] <= SMALL_AXIS_THRESHOLD) {
                total = shape[0] * ((shape[2] + ALIGN_2047) / CHUNK_SIZE_2048);
            } else if (shape[0] * shape[2] <= SMALL_BATCH_THRESHOLD) {
                total = ((shape[2] + ALIGN_15) / ALIGN_16) * shape[0];
            } else {
                total = shape[0] * ((shape[2] + ALIGN_127) / ALIGN_128);
            }
        } else {
            // BF16 或其他
            if (shape[2] >= BLOCK_FACTOR_40 * CHUNK_SIZE_2048) {
                total = shape[0] * ((shape[2] + ALIGN_2047) / CHUNK_SIZE_2048);
            } else if (shape[0] * shape[2] >= SMALL_BATCH_THRESHOLD) {
                total = ((shape[2] + ALIGN_127) / ALIGN_128) * shape[0];
            } else {
                total = shape[0] * ((shape[2] + ALIGN_63) / ALIGN_64);
            }
        }
        int64_t requiredCore = static_cast<int64_t>(total) * CORE_NUM_MIN_FACTOR;
        coreNum = requiredCore < coreNum ? requiredCore : coreNum;
    }
    
    tiling->axis = static_cast<uint64_t>(axis);
    tiling->dims = static_cast<uint64_t>(dims);
    tiling->shape[0] = static_cast<uint64_t>(shape[0]);
    tiling->shape[1] = static_cast<uint64_t>(shape[1]);
    tiling->shape[2] = static_cast<uint64_t>(shape[2]);

    context->SetBlockDim(coreNum);
    
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(LogSoftmaxV2).Tiling(LogSoftmaxV2TilingFunc).TilingParse<LogSoftmaxV2CompileInfo>(TilingParseForLogSoftmaxV2);
} // namespace optiling
