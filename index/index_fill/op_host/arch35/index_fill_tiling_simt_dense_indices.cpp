/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file index_fill_tiling_simt_dense_indices.cpp
 * \brief
 */

#include "op_api/runtime2_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/util/math_util.h"
#include "tiling/tiling_api.h"
#include "op_api/op_util.h"
#include "log/log.h"
#include "op_common/log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "index_fill_tiling_simt_dense_indices.h"

namespace optiling {
using namespace IndexFill;

constexpr uint64_t DCACHE_SIZE_SIMT = 128 * 1024;
constexpr uint64_t SIMD_THRESHOLD = 128;
constexpr uint64_t INDICES_DENSE_THRESHOLD = 5;
constexpr int64_t UB_AGLIN_VALUE = 32;
constexpr int64_t DUP_ELEMENTS_MIN_THRESHOLD = 256;
constexpr int64_t INDICES_UB_FACTOR_MIN_THRESHOLD = 512;

bool IndexFillSimtDenseIndicesTiling::IsCapable()
{
    if (inputData.numel == 0) {
        return false;
    }

    if (inputData.Q * inputData.dtypeSize <= SIMD_THRESHOLD &&
        inputData.P * inputData.Q * inputData.indicesNum > 2048 &&
        inputData.indicesNum >= inputData.N * INDICES_DENSE_THRESHOLD) {
        return true;
    }
    return false;
}

int64_t IndexFillSimtDenseIndicesTiling::GetOptimalIndiceUBFactor(int64_t indicesUbFactor, int64_t indicesNum,
                                                                  int64_t n)
{
    // indicesUbFactor进一步细化设置，indicesUbFactor要尽量大，如此才能更有效的达到去重和防同地址写冲突的效果；同时，尽量多的核并行处理。
    int64_t indicesFactor = std::min(indicesUbFactor, indicesNum);
    int64_t minThreshold = std::max(INDICES_UB_FACTOR_MIN_THRESHOLD, n + DUP_ELEMENTS_MIN_THRESHOLD);
    if (indicesFactor > minThreshold) {
        int64_t totalBlocks = Ops::Base::CeilDiv(indicesNum, indicesFactor);
        if (totalBlocks * TWO <= coreNum_) {
            indicesFactor = minThreshold;
        }
    }
    return indicesFactor;
}

uint32_t IndexFillSimtDenseIndicesTiling::GetSortTmpSize(ge::DataType indicesDtype, uint32_t shapeSize, bool isDescend)
{
    std::vector<int64_t> shapeVec = {shapeSize};
    ge::Shape srcShape(shapeVec);
    AscendC::SortConfig config;
    config.type = AscendC::SortType::RADIX_SORT;
    config.isDescend = isDescend;
    config.hasSrcIndex = false;
    config.hasDstIndex = true;

    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetSortMaxMinTmpSize(srcShape, indicesDtype, ge::DT_UINT32, false, config, maxValue, minValue);
    OP_LOGI(context_->GetNodeName(), "Need tmp buffer %u byte for ascendc sort api", maxValue);
    return maxValue;
}

int64_t IndexFillSimtDenseIndicesTiling::CalcUsedBufSize(int64_t indicesUbFactor, ge::DataType indicesDtype)
{
    int64_t indicesDtypeSize = static_cast<int64_t>(ge::GetSizeByDataType(indicesDtype));
    int64_t vfLen = Ops::Base::GetVRegSize(context_) / indicesDtypeSize;
    int64_t indicesBuffSize = Ops::Base::CeilDiv(indicesUbFactor + 1, vfLen) * vfLen * indicesDtypeSize +
                              UB_AGLIN_VALUE;
    int64_t sortNeedTmpSize = static_cast<int64_t>(GetSortTmpSize(indicesDtype, indicesUbFactor, false));
    int64_t calcBufSize = (Ops::Base::CeilAlign(indicesUbFactor * indicesDtypeSize, UB_AGLIN_VALUE) *
                           N_BUFFER)                                              // indicesQueue占用的大小
                          + Ops::Base::CeilAlign(indicesBuffSize, UB_AGLIN_VALUE) // indicesBuff占用的大小
                          + Ops::Base::CeilAlign(static_cast<int64_t>(indicesUbFactor * sizeof(int32_t)),
                                                 UB_AGLIN_VALUE)                   // uniqueIdBuff占用的大小
                          + Ops::Base::CeilAlign(sortNeedTmpSize, UB_AGLIN_VALUE); // sort排序占用的tmp大小
    return calcBufSize;
}

void IndexFillSimtDenseIndicesTiling::DoIndicesTilingTask()
{
    uint64_t availableUb = ubSize_ - DCACHE_SIZE_SIMT;
    int64_t indicesDtypeSize = static_cast<int64_t>(inputData.indicesDtypeSize);
    int64_t oneBlockNum = Ops::Base::GetUbBlockSize(context_) / indicesDtypeSize;

    int64_t indicesNum = static_cast<int64_t>(inputData.indicesNum);
    int64_t maxUbAvailable = Ops::Base::FloorAlign(static_cast<int64_t>(availableUb),
                                                   static_cast<int64_t>(UB_AGLIN_VALUE));
    int64_t nBuffer = static_cast<int64_t>(N_BUFFER + 1);
    int64_t maxUbIndicesNum = Ops::Base::FloorAlign(maxUbAvailable / nBuffer / indicesDtypeSize, oneBlockNum);
    int64_t indicesUbFactor = maxUbIndicesNum;
    int64_t calcBufSize = CalcUsedBufSize(indicesUbFactor, inputData.indicesDtype);
    while (calcBufSize > maxUbAvailable) {
        indicesUbFactor = indicesUbFactor - 1024;
        if (indicesUbFactor <= 0) {
            indicesUbFactor = INDICES_UB_FACTOR_MIN_THRESHOLD;
            break;
        }
        calcBufSize = CalcUsedBufSize(indicesUbFactor, inputData.indicesDtype);
    }

    indicesUbFactor = GetOptimalIndiceUBFactor(indicesUbFactor, indicesNum, static_cast<int64_t>(inputData.N));
    inputData.indicesUbFactor = indicesUbFactor;

    // 考虑indices切分对核数的要求，使得kernel各个阶段执行过程中，多核并行度尽可能高.
    int64_t totalBlocks = Ops::Base::CeilDiv(indicesNum, indicesUbFactor);
    int64_t indicesUsedCoreNum = totalBlocks >= coreNum_ ? coreNum_ : totalBlocks;
    usedCoreNum_ = std::max(usedCoreNum_, indicesUsedCoreNum);
}

ge::graphStatus IndexFillSimtDenseIndicesTiling::DoOpTiling()
{
    DoUBTiling();
    DoIndicesTilingTask();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexFillSimtDenseIndicesTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    context_->SetLocalMemorySize(ubSize_ - DCACHE_SIZE_SIMT);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexFillSimtDenseIndicesTiling::GetWorkspaceSize()
{
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = (inputData.N * sizeof(int8_t)) +
                          sysWorkspaceSize_; // workspace上申请额外申请一片内存，用于存放索引位图

    // 涉及到核间同步，需要设置该模式.
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

void IndexFillSimtDenseIndicesTiling::SetTilingData()
{
    IndexFill::IndexFillSimtDenseIndicesTilingData*
        tilingData = context_->GetTilingData<IndexFill::IndexFillSimtDenseIndicesTilingData>();
    tilingData->p = static_cast<int64_t>(inputData.P);
    tilingData->q = static_cast<int64_t>(inputData.Q);
    tilingData->n = static_cast<int64_t>(inputData.N);
    tilingData->indicesNum = static_cast<int64_t>(inputData.indicesNum);
    tilingData->coreNum = coreNum_;
    tilingData->usedCoreNum = usedCoreNum_;
    tilingData->simtUsedCoreNum = simtUsedCoreNum_;
    tilingData->simdUsedCoreNum = simdUsedCoreNum_;
    tilingData->frontCoreNum = inputData.frontCoreNum;
    tilingData->blockSize = inputData.blockSize;
    tilingData->tailBlockSize = inputData.tailBlockSize;
    tilingData->loopsPerFrontCore = inputData.loopsPerFrontCore;
    tilingData->loopsPerTailCore = inputData.loopsPerTailCore;
    tilingData->indicesUbFactor = inputData.indicesUbFactor;
    tilingData->tilingKey = GetTilingKey();
}

uint64_t IndexFillSimtDenseIndicesTiling::GetTilingKey() const
{
    uint64_t templateMode = static_cast<uint64_t>(TPL_MODE_TEMPLATE_SIMT_DENSE_INDICES);
    uint64_t dTypeMode = static_cast<uint64_t>(TPL_MODE_DTYPE_B32);
    uint64_t totalDataSize = inputData.numel;
    if (inputData.numel == 0) {
        templateMode = static_cast<uint64_t>(TPL_MODE_TEMPLATE_EMPTY);
    }

    if (totalDataSize > MAX_INT32) {
        dTypeMode = static_cast<uint64_t>(TPL_MODE_DTYPE_B64);
    }
    const uint64_t tilingKey = GET_TPL_TILING_KEY(templateMode, dTypeMode);
    OP_LOGI(context_->GetNodeName(), "IndexFillSimtDenseIndicesTiling::GetTilingKey enter 02");
    return tilingKey;
}

void IndexFillSimtDenseIndicesTiling::DumpTilingInfo()
{
    IndexFill::IndexFillSimtDenseIndicesTilingData*
        tilingData = context_->GetTilingData<IndexFill::IndexFillSimtDenseIndicesTilingData>();
    OP_LOGI(context_->GetNodeName(), "IndexFill tilingInfo is: %s", tilingData->ToString().c_str());
}

REGISTER_OPS_TILING_TEMPLATE(IndexFill, IndexFillSimtDenseIndicesTiling, 6);
} // namespace optiling
