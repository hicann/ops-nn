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
 * \file inplace_index_fill_tiling_simt.cpp
 * \brief SIMT tiling strategy with dual-path optimization:
 *        - SIMD 搬运 x→y（UB 切分 + 双缓冲）
 *        - SIMT 计算（路径A 稀疏 / 路径B 稠密 mask+顺序写）
 */

#include "inplace_index_fill_tiling_simt.h"
#include "log/log.h"
#include "../../op_kernel/arch35/inplace_index_fill_tiling_key.h"
#include "op_host/tiling_templates_registry.h"

using namespace InplaceIndexFill;

namespace optiling {

// ============================================================================
// IsCapable：SIMT 模板是兜底模板，无条件接受
// ============================================================================
bool InplaceIndexFillTilingSimt::IsCapable()
{
    return true;
}

// ============================================================================
// CalcSimtUsedCoreNum：计算 SIMT 计算侧需要的核数
// 策略：从 MAX_THREAD_NUM 开始，如果核利用率不足 50%，则减半线程数重新计算
// 目的：确保至少使用一半的核，避免核闲置
// ============================================================================
int64_t InplaceIndexFillTilingSimt::CalcSimtUsedCoreNum()
{
    int64_t threadNum = MAX_THREAD_NUM;
    // numel = P * N * Q（x 的总元素数），是路径B的遍历量上界
    int64_t numel = inputData.numel;
    // 每核至少需要 threadNum 个元素才能充分利用线程
    int64_t simtUsedCoreNum = std::min(coreNum_, numel / threadNum);

    // 如果核利用率不足 50%，减半线程数重新计算
    while ((simtUsedCoreNum < (coreNum_ / 2)) && (threadNum > static_cast<int64_t>(MIN_THREAD_NUM))) {
        threadNum = threadNum / 2;
        simtUsedCoreNum = std::min(coreNum_, numel / threadNum);
    }

    return simtUsedCoreNum;
}

// ============================================================================
// DoOpTiling：主 tiling 入口
// ============================================================================
ge::graphStatus InplaceIndexFillTilingSimt::DoOpTiling()
{
    coreNum_ = coreNum;

    // 空 tensor 处理
    if (inputData.preDimProduct == 0 || inputData.dimSize == 0 || inputData.postDimProduct == 0) {
        usedCoreNum = 1;
        simtUsedCoreNum_ = 1;
        SetTilingData();
        return ge::GRAPH_SUCCESS;
    }

    simtUsedCoreNum_ = CalcSimtUsedCoreNum();
    usedCoreNum = inputData.numel == 0 ? coreNum_ : simtUsedCoreNum_;
    SetTilingData();

    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// GetWorkspaceSize：申请 workspace 内存
// 路径B 需要 N 字节的 mask 位图存放在 workspace 上
// ============================================================================
ge::graphStatus InplaceIndexFillTilingSimt::GetWorkspaceSize()
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

    // 设置核间同步模式（BuildIndicesMask 需要 SyncAll）
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================
// GetTilingKey：生成 tiling key
// ============================================================================
uint64_t InplaceIndexFillTilingSimt::GetTilingKey() const
{
    uint64_t templateMode = static_cast<uint64_t>(TPL_MODE_TEMPLATE_SIMT);
    uint64_t dtypeMode = (inputData.xDtypeSize <= 4)
        ? static_cast<uint64_t>(TPL_MODE_DTYPE_B32)
        : static_cast<uint64_t>(TPL_MODE_DTYPE_B64);
    return GET_TPL_TILING_KEY(templateMode, dtypeMode);
}

// ============================================================================
// SetTilingData：将计算结果写入 tiling 数据结构
// ============================================================================
void InplaceIndexFillTilingSimt::SetTilingData()
{
    auto* tilingData = context_->GetTilingData<InplaceIndexFillSimtTilingData>();
    tilingData->tilingKeySimt = GetTilingKey();
    tilingData->p = inputData.preDimProduct;
    tilingData->n = inputData.dimSize;
    tilingData->q = inputData.postDimProduct;
    tilingData->indicesNum = inputData.indicesNum;
    tilingData->coreNum = coreNum_;
    tilingData->usedCoreNum = usedCoreNum;
    tilingData->simtUsedCoreNum = simtUsedCoreNum_;
}

// ============================================================================
// DumpTilingInfo：打印 tiling 信息用于调试
// ============================================================================
void InplaceIndexFillTilingSimt::DumpTilingInfo()
{
    auto* tilingData = context_->GetTilingData<InplaceIndexFillSimtTilingData>();
    std::ostringstream info;
    info << "tilingKeySimt: " << tilingData->tilingKeySimt
         << ", p: " << tilingData->p
         << ", n: " << tilingData->n
         << ", q: " << tilingData->q
         << ", indicesNum: " << tilingData->indicesNum
         << ", usedCoreNum: " << tilingData->usedCoreNum
         << ", simtUsedCoreNum: " << tilingData->simtUsedCoreNum;
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

// ============================================================================
// PostTiling：设置 blockDim
// ============================================================================
ge::graphStatus InplaceIndexFillTilingSimt::PostTiling()
{
    context_->SetBlockDim(usedCoreNum);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(InplaceIndexFill, InplaceIndexFillTilingSimt, 10);
} // namespace optiling