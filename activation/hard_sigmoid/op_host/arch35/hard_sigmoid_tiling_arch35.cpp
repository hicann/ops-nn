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
 * \file hard_sigmoid_tiling_arch35.cpp
 * \brief HardSigmoid tiling（arch35 / DAV_3510）
 *
 * 多核切分和核内 UB 切分分别由 blockFactor、ubFactor 描述，并按 input0 dtype 选择模板实例。
 */

#include "register/op_def_registry.h"
#include <string>
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch35/hard_sigmoid_tiling_data.h"
#include "../../op_kernel/arch35/hard_sigmoid_tiling_key.h"

namespace optiling {
namespace {
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;
using Ops::Base::GetVRegSize;

constexpr float DEFAULT_ALPHA = 1.0f / 6.0f;
constexpr float DEFAULT_BETA = 0.5f;
constexpr size_t WORKSPACE_NUM = 1;
constexpr size_t WS_SYS_SIZE = 0U;

constexpr int64_t UB_RESERVE_BYTES = 8192; // 预留系统/对齐余量
constexpr int64_t MIN_COPY_BYTES = 16 * 1024;
constexpr int64_t F32_TEMP_BYTES = static_cast<int64_t>(sizeof(float)); // 非 fp32 路径的 fp32 中间缓冲

struct HardSigmoidCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

int64_t StorageBytesPerElement(ge::DataType dtype) { return (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) ? 2 : 4; }

// 每元素 UB 占用，与 kernel 侧 buffer 布局严格对应（见 hard_sigmoid.cpp InitBuffer）：
//   inQue/outQue 各 HARD_SIGMOID_BUFFER_NUM 份 sizeof(T)；非 fp32 另加 1 份 fp32 中间缓冲。
// 实测值：FLOAT=16、FLOAT16/BFLOAT16=12、INT32=20。
// BUFFER_NUM 取自 Host/Kernel 共享头，避免任一侧改动导致 UB 预算静默失配。
int64_t UbBytesPerElement(ge::DataType dtype)
{
    constexpr int64_t QUEUE_COUNT = 2; // inQue + outQue
    const int64_t queueBytes = HARD_SIGMOID_BUFFER_NUM * QUEUE_COUNT * StorageBytesPerElement(dtype);
    return queueBytes + ((dtype == ge::DT_FLOAT) ? 0 : F32_TEMP_BYTES);
}

// vRegSize 为向量寄存器宽度（平台参数，由调用方经 GetVRegSize 获取），用于对齐 ubFactor。
int64_t ComputeUbFactor(ge::DataType dtype, int64_t ubSize, int64_t ubBlockSize, int64_t vRegSize)
{
    const int64_t usableUbSize = ubSize - UB_RESERVE_BYTES;
    if (usableUbSize <= 0 || ubBlockSize <= 0 || vRegSize <= 0) {
        return 0;
    }

    const int64_t alignBytes = ubBlockSize > vRegSize ? ubBlockSize : vRegSize;
    const int64_t alignElements = alignBytes / StorageBytesPerElement(dtype);
    return FloorAlign(FloorDiv(usableUbSize, UbBytesPerElement(dtype)), alignElements);
}

ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* workspaces = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}
} // namespace

static ge::graphStatus HardSigmoidTilingFunc(gert::TilingContext* context)
{
    auto* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    auto* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dtype = inputDesc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16 && dtype != ge::DT_INT32,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "input_x",
                                          std::to_string(static_cast<int32_t>(dtype)).c_str(),
                                          "DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT32"),
                return ge::GRAPH_FAILED);

    auto* tilingData = context->GetTilingData<HardSigmoidTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    *tilingData = HardSigmoidTilingData{};

    // 激活系数：使用默认值，attrs 存在时按位置 0/1 覆盖。
    tilingData->alpha = DEFAULT_ALPHA;
    tilingData->beta = DEFAULT_BETA;
    const auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float* alpha = attrs->GetFloat(0);
        const float* beta = attrs->GetFloat(1);
        if (alpha != nullptr) {
            tilingData->alpha = *alpha;
        }
        if (beta != nullptr) {
            tilingData->beta = *beta;
        }
    }

    if (GetWorkspaceSize(context) != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "GetWorkspaceSize failed");
        return ge::GRAPH_FAILED;
    }

    const auto& storageShape = inputShape->GetStorageShape();
    for (size_t i = 0; i < storageShape.GetDimNum(); ++i) {
        OP_CHECK_IF(storageShape.GetDim(i) < 0,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "input_x",
                                                             std::to_string(storageShape.GetDim(i)).c_str(),
                                                             "Storage shape dimensions must be non-negative"),
                    return ge::GRAPH_FAILED);
    }
    const int64_t totalElements = storageShape.GetShapeSize();
    OP_CHECK_IF(totalElements < 0,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "input_x",
                                                          std::to_string(totalElements).c_str(),
                                                          "The storage shape size must be non-negative"),
                return ge::GRAPH_FAILED);
    tilingData->totalElements = totalElements;

    // tiling-key：按 input0 dtype 选择编译期计算路径。ge::DataType 与 C_DT_* 共享 c_types.h 枚举值。
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dtype));

    // 空张量：单核占位返回，kernel 不申请 UB。
    if (totalElements == 0) {
        context->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }

    // 平台信息：UB 大小 + AIV 核数。binary tiling 场景下 platformInfo 可能为空，使用 TilingParse 写入的 compileInfo。
    auto platformInfoPtr = context->GetPlatformInfo();
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    if (platformInfoPtr == nullptr) {
        auto compileInfo = context->GetCompileInfo<HardSigmoidCompileInfo>();
        OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
        ubSize = compileInfo->ubSize;
        coreNum = static_cast<int64_t>(compileInfo->coreNum);
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        coreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    }
    if (ubSize == 0) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "The platform UB size must be greater than zero, ubSize: %lu",
                               ubSize);
        return ge::GRAPH_FAILED;
    }
    if (coreNum <= 0) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "The platform AIV core count must be greater than zero, coreNum: %ld", coreNum);
        return ge::GRAPH_FAILED;
    }

    // ubBlockSize / vRegSize 的非法值统一由 ComputeUbFactor 返回 0，再由下方 ubFactor 守卫拦截并报错。
    const int64_t ubBlockSize = GetUbBlockSize(context);
    const int64_t vRegSize = static_cast<int64_t>(GetVRegSize(context));
    tilingData->ubFactor = ComputeUbFactor(dtype, static_cast<int64_t>(ubSize), ubBlockSize, vRegSize);
    const std::string platformValues = std::to_string(ubSize) + "," + std::to_string(ubBlockSize) + "," +
                                       std::to_string(vRegSize) + "," + std::to_string(static_cast<int32_t>(dtype));
    if (tilingData->ubFactor <= 0) {
        OP_LOGE_WITHOUT_REPORT(
            context->GetNodeName(),
            "The UB is too small or the platform parameters are invalid, ubSize,ubBlockSize,vRegSize,dtype: %s",
            platformValues.c_str());
        return ge::GRAPH_FAILED;
    }

    // 限制常规核的 GM<->UB 搬运量不低于 16KB；总量不足时使用单核，最后一个尾核允许更小。
    const int64_t minCopyElements = CeilDiv(MIN_COPY_BYTES, StorageBytesPerElement(dtype));
    const int64_t maxCoreNumByCopy = totalElements / minCopyElements;
    const int64_t actualCoreNum = maxCoreNumByCopy <= 0 ? 1 : (maxCoreNumByCopy < coreNum ? maxCoreNumByCopy : coreNum);
    tilingData->blockFactor = CeilDiv(totalElements, actualCoreNum);
    // blockFactor 随即作为除数使用；此处就地守卫，避免其非零性仅依赖上游 totalElements>0 的跨行推导。
    if (tilingData->blockFactor <= 0) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "The per-core block factor must be greater than zero, blockFactor: %ld",
                               tilingData->blockFactor);
        return ge::GRAPH_FAILED;
    }
    const int64_t usedCoreNum = CeilDiv(totalElements, tilingData->blockFactor);
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForHardSigmoid(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<HardSigmoidCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    if (compileInfo->coreNum == 0 || compileInfo->ubSize == 0) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "The platform AIV core count and UB size must be greater than zero, coreNum: %lu, "
                               "ubSize: %lu",
                               compileInfo->coreNum, compileInfo->ubSize);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(HardSigmoid)
    .Tiling(HardSigmoidTilingFunc)
    .TilingParse<HardSigmoidCompileInfo>(TilingParseForHardSigmoid);
} // namespace optiling
