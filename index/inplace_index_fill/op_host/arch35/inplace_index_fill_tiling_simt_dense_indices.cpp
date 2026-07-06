/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file inplace_index_fill_tiling_simt_dense_indices.cpp
 * \brief DenseIndices tiling: Sort+Unique deduplication for extremely dense indices
 *
 * 适用场景：indicesNum >= N*5（极端稠密，大量重复值）
 * 核心思路：将 indices 分批搬入 UB，Sort+Unique 去重后再构建 mask
 * 优势：从源头消除写冲突，减少无效的 mask 写入
 */

#include "inplace_index_fill_tiling_simt_dense_indices.h"
#include "log/log.h"
#include "../../op_kernel/arch35/inplace_index_fill_tiling_key.h"
#include "op_host/tiling_templates_registry.h"

using namespace InplaceIndexFill;

namespace optiling {

// SIMD 阈值：Q*dtypeSize <= 128 时才选 DenseIndices（Q 大时走 SIMD 模板）
constexpr uint64_t DENSE_SIMD_THRESHOLD = 128;
// 稠密度阈值：indicesNum >= N * 5 才认为是极端稠密
constexpr uint64_t INDICES_DENSE_THRESHOLD = 5;
// UB 对齐值：32 字节
constexpr int64_t DENSE_UB_ALIGN_VALUE = 32;
// 最小重复元素数阈值：每次 UB 加载至少有 256 个重复元素才值得去重
constexpr int64_t DUP_ELEMENTS_MIN_THRESHOLD = 256;
// 最小 indices UB factor 阈值
constexpr int64_t INDICES_UB_FACTOR_MIN_THRESHOLD = 512;
// DCache 预留
constexpr uint64_t DENSE_DCACHE_SIZE = 128 * 1024;
// indices 队列缓冲数量
constexpr uint64_t N_BUFFER = 2;

// ============================================================================
// IsCapable：判断是否适合使用 DenseIndices 模板
// 三个条件同时满足：
// 1. Q*dtypeSize <= 128（Q 不大，不适合 SIMD 模板）
// 2. P*Q*indicesNum > 2048（总计算量足够大，值得去重开销）
// 3. indicesNum >= N*5（极端稠密，重复度高）
// ============================================================================
bool InplaceIndexFillTilingSimtDenseIndices::IsCapable()
{
    if (inputData.numel == 0) {
        return false;
    }

    if (inputData.postDimProduct * inputData.xDtypeSize <= static_cast<int64_t>(DENSE_SIMD_THRESHOLD) &&
        inputData.preDimProduct * inputData.postDimProduct * inputData.indicesNum > 2048 &&
        inputData.indicesNum >= inputData.dimSize * static_cast<int64_t>(INDICES_DENSE_THRESHOLD)) {
        return true;
    }
    return false;
}

// ============================================================================
// GetOptimalIndiceUBFactor：计算最优的 indices UB 加载量
// 策略：indicesUbFactor 要尽量大（去重效果好），同时确保多核并行
// ============================================================================
int64_t InplaceIndexFillTilingSimtDenseIndices::GetOptimalIndiceUBFactor(int64_t indicesUbFactor, int64_t indicesNum,
                                                                         int64_t n)
{
    int64_t indicesFactor = std::min(indicesUbFactor, indicesNum);
    // 逐步缩小 indicesFactor，直到满足多核并行条件
    while (indicesFactor > INDICES_UB_FACTOR_MIN_THRESHOLD && indicesFactor > n + DUP_ELEMENTS_MIN_THRESHOLD) {
        // 计算总块数
        int64_t totalBlocks = (indicesNum + indicesFactor - 1) / indicesFactor;
        // 如果总块数 >= 核数的一半，说明有足够的并行度
        if (totalBlocks * 2 > coreNum_) {
            break;
        }
        // 否则缩小到 max(n + 256, 512)，确保每次 UB 加载至少有 256 个重复元素
        indicesFactor = std::max(n + DUP_ELEMENTS_MIN_THRESHOLD, INDICES_UB_FACTOR_MIN_THRESHOLD);
    }
    return indicesFactor;
}

// ============================================================================
// DoIndicesTilingTask：计算 indices 的 UB 切分参数
// ============================================================================
void InplaceIndexFillTilingSimtDenseIndices::DoIndicesTilingTask()
{
    uint64_t availableUb = ubSize - DENSE_DCACHE_SIZE;
    int64_t indicesDtypeSize = inputData.indicesDtypeSize;

    // UB 可用空间（对齐到 32 字节）
    int64_t maxUbAvailable = static_cast<int64_t>(availableUb / DENSE_UB_ALIGN_VALUE) * DENSE_UB_ALIGN_VALUE;

    // 计算 UB 中最多能放多少个 indices 元素
    // 需要考虑：indicesQueue(双缓冲) + indicesBuff(sort输出) + uniqueIdBuff(去重ID)
    // 简化估算：总共需要约 (N_BUFFER + 2) 倍的 indices 空间
    int64_t nBuffer = static_cast<int64_t>(N_BUFFER + 1);
    int64_t oneBlockNum = DENSE_UB_ALIGN_VALUE / indicesDtypeSize;
    int64_t maxUbIndicesNum = (maxUbAvailable / nBuffer / indicesDtypeSize / oneBlockNum) * oneBlockNum;

    int64_t indicesUbFactor = maxUbIndicesNum;

    // 进一步优化：确保多核并行度
    indicesUbFactor = GetOptimalIndiceUBFactor(indicesUbFactor, inputData.indicesNum, inputData.dimSize);

    inputData.indicesUbFactor = indicesUbFactor;
}

// ============================================================================
// DoOpTiling：主 tiling 入口
// ============================================================================
ge::graphStatus InplaceIndexFillTilingSimtDenseIndices::DoOpTiling()
{
    coreNum_ = coreNum;

    // 计算 SIMT 核数
    simtUsedCoreNum_ = CalcSimtUsedCoreNum();
    usedCoreNum = inputData.numel == 0 ? coreNum_ : simtUsedCoreNum_;

    // indices 的 UB 切分
    DoIndicesTilingTask();
    SetTilingData();

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// PostTiling：设置 blockDim 和 LocalMemorySize
// ============================================================================
// GetWorkspaceSize：申请 N 字节 mask 位图 + 系统 workspace，并设核间同步模式
ge::graphStatus InplaceIndexFillTilingSimtDenseIndices::GetWorkspaceSize()
{
    size_t sysWorkspaceSize = SYS_WORKSPACE_SIZE;
    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
        sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    }

    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    // workspace = N 字节（mask 位图）+ 系统 workspace
    currentWorkspace[0] = (inputData.dimSize * sizeof(int8_t)) + sysWorkspaceSize;

    // 设置核间同步模式（BuildIndicesMask 的 SyncAll 依赖）
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InplaceIndexFillTilingSimtDenseIndices::PostTiling()
{
    context_->SetBlockDim(usedCoreNum);
    // DenseIndices 需要更多 UB 空间用于 Sort+Unique
    context_->SetLocalMemorySize(ubSize - DENSE_DCACHE_SIZE);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// GetTilingKey：生成 DenseIndices 模板的 tiling key
// ============================================================================
uint64_t InplaceIndexFillTilingSimtDenseIndices::GetTilingKey() const
{
    uint64_t templateMode = static_cast<uint64_t>(TPL_MODE_TEMPLATE_SIMT_DENSE_INDICES);
    uint64_t dtypeMode = (inputData.xDtypeSize <= 4) ? static_cast<uint64_t>(TPL_MODE_DTYPE_B32) :
                                                       static_cast<uint64_t>(TPL_MODE_DTYPE_B64);
    return GET_TPL_TILING_KEY(templateMode, dtypeMode);
}

// ============================================================================
// SetTilingData：写入 DenseIndices tiling 数据
// ============================================================================
void InplaceIndexFillTilingSimtDenseIndices::SetTilingData()
{
    auto* tilingData = context_->GetTilingData<InplaceIndexFillSimtDenseIndicesTilingData>();
    tilingData->tilingKeySimt = GetTilingKey();
    tilingData->p = inputData.preDimProduct;
    tilingData->n = inputData.dimSize;
    tilingData->q = inputData.postDimProduct;
    tilingData->indicesNum = inputData.indicesNum;
    tilingData->coreNum = coreNum_;
    tilingData->usedCoreNum = usedCoreNum;
    tilingData->simtUsedCoreNum = simtUsedCoreNum_;
    tilingData->indicesUbFactor = inputData.indicesUbFactor;
}

// ============================================================================
// DumpTilingInfo：打印调试信息
// ============================================================================
void InplaceIndexFillTilingSimtDenseIndices::DumpTilingInfo()
{
    auto* tilingData = context_->GetTilingData<InplaceIndexFillSimtDenseIndicesTilingData>();
    std::ostringstream info;
    info << "DenseIndices tilingKey: " << tilingData->tilingKeySimt << ", p: " << tilingData->p
         << ", n: " << tilingData->n << ", q: " << tilingData->q << ", indicesNum: " << tilingData->indicesNum
         << ", usedCoreNum: " << tilingData->usedCoreNum << ", indicesUbFactor: " << tilingData->indicesUbFactor;
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

// 注册优先级 6：比 SIMT(10) 高，比 SIMD(1) 低
REGISTER_OPS_TILING_TEMPLATE(InplaceIndexFill, InplaceIndexFillTilingSimtDenseIndices, 6);
} // namespace optiling