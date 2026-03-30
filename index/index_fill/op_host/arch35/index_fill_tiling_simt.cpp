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
 * \file index_fill_tiling_simt.cpp
 * \brief
 */

#include "op_api/runtime2_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/util/math_util.h"
#include "op_api/op_util.h"
#include "log/log.h"
#include "op_common/log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "index_fill_tiling_simt.h"

namespace optiling
{
using namespace IndexFill;

constexpr uint64_t DCACHE_SIZE_SIMT = 128 * 1024;

bool IndexFillSimtTiling::IsCapable()
{
    return true;
}

int64_t IndexFillSimtTiling::CalcSimtUsedCoreNum()
{
    int64_t threadNum = MAX_THREAD_NUM;
    int64_t simtUsedCoreNum = std::min(coreNum_, inputData.numel / threadNum);
    while ((simtUsedCoreNum < (coreNum_ / 2)) && (threadNum > MIN_THREAD_NUM)) {
        threadNum = threadNum / 2;
        simtUsedCoreNum = std::min(coreNum_, inputData.numel / threadNum);
    }

    return simtUsedCoreNum;
}

void IndexFillSimtTiling::DoUBTiling()
{
    int64_t simdUsedCoreNum = coreNum_;
    int64_t numelPerCore = Ops::Base::CeilDiv(inputData.numel, simdUsedCoreNum); // 平均每个核要处理的元素数
    int64_t numelForTailCore = 0;
    int64_t blockSize = 0;
    int64_t tailBlockSize = 0;
    // 如果所有核均分后，每个核要搬运的数据小于阈值BLOCK_SPLIT_THREHOLD个字节(1k), 则以阈值大小为准，重新计算需要多少核来搬运.
    if (numelPerCore * inputData.dtypeSize <= BLOCK_SPLIT_THREHOLD) {
        uint64_t numelMinThrehold = BLOCK_SPLIT_THREHOLD / inputData.dtypeSize;
        numelPerCore = numelMinThrehold;
        blockSize = numelPerCore;

        int64_t totalSliceNum = Ops::Base::CeilDiv(inputData.numel, blockSize);
        simdUsedCoreNum = totalSliceNum;
        numelForTailCore = inputData.numel - numelPerCore * (simdUsedCoreNum - 1);
        tailBlockSize = numelForTailCore;

        inputData.frontCoreNum = 0;
        inputData.blockSize = blockSize;
        inputData.tailBlockSize = tailBlockSize;
        inputData.loopsPerFrontCore = 1;
        inputData.loopsPerTailCore = 1;
    } else {
        // 判断numelPerCore是否超过ub大小，进行ub切分
        int64_t oneBlockNum = Ops::Base::GetUbBlockSize(context_) / inputData.dtypeSize;
        // maxUbAvailable表示ub最大能承载的元素数
        int64_t maxUbAvailable = Ops::Base::FloorAlign(static_cast<int64_t>((ubSize_ - DCACHE_SIZE_SIMT) / N_BUFFER / inputData.dtypeSize), oneBlockNum);
        int64_t ubFactor = std::min(numelPerCore, maxUbAvailable);

        blockSize = ubFactor;
        // 根据每个块的大小blockSize， 算出可以切分出totalSliceNum个块
        int64_t totalSliceNum = Ops::Base::CeilDiv(inputData.numel, blockSize);
        tailBlockSize = inputData.numel - (totalSliceNum - 1) * blockSize;

        // 对切分的块，分核
        int64_t frontCoreNum = totalSliceNum % simdUsedCoreNum;
        int64_t loopsPerFrontCore = Ops::Base::CeilDiv(totalSliceNum, simdUsedCoreNum);
        int64_t loopsPerTailCore = totalSliceNum /simdUsedCoreNum;
        numelPerCore = loopsPerFrontCore * blockSize;
        numelForTailCore = (loopsPerTailCore - 1) * blockSize + tailBlockSize;

        inputData.frontCoreNum = frontCoreNum;
        inputData.blockSize = blockSize;
        inputData.tailBlockSize = tailBlockSize;
        inputData.loopsPerFrontCore = loopsPerFrontCore;
        inputData.loopsPerTailCore = loopsPerTailCore;
    }

    // 使用的核数取simdUsedCoreNum和simtUsedCoreNum的最大值
    simdUsedCoreNum_ = simdUsedCoreNum;
    simtUsedCoreNum_ = CalcSimtUsedCoreNum();
    usedCoreNum_ = inputData.numel == 0 ? coreNum_ : std::max(simdUsedCoreNum_, simtUsedCoreNum_);
}

ge::graphStatus IndexFillSimtTiling::DoOpTiling()
{
    DoUBTiling();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexFillSimtTiling::GetWorkspaceSize()
{
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sysWorkspaceSize_;

    // 涉及到核间同步，需要设置该模式.
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

uint64_t IndexFillSimtTiling::GetTilingKey() const
{
    uint64_t templateMode = static_cast<uint64_t>(TPL_MODE_TEMPLATE_SIMT);
    uint64_t dTypeMode = static_cast<uint64_t>(TPL_MODE_DTYPE_B32);
    uint64_t totalDataSize = inputData.numel;
    if (inputData.numel == 0) {
        templateMode = static_cast<uint64_t>(TPL_MODE_TEMPLATE_EMPTY);
    }

    if (totalDataSize > MAX_INT32) {
        dTypeMode = static_cast<uint64_t>(TPL_MODE_DTYPE_B64);
    }
    const uint64_t tilingKey = GET_TPL_TILING_KEY(templateMode, dTypeMode);
    return tilingKey;
}

void IndexFillSimtTiling::SetTilingData()
{
    IndexFill::IndexFillSimtTilingData* tilingData = context_->GetTilingData<IndexFill::IndexFillSimtTilingData>();
    tilingData->p = static_cast<int64_t>(inputData.P);
    tilingData->n = static_cast<int64_t>(inputData.N);
    tilingData->q = static_cast<int64_t>(inputData.Q);
    tilingData->indicesNum = static_cast<int64_t>(inputData.indicesNum);
    tilingData->coreNum = coreNum_;
    tilingData->usedCoreNum = usedCoreNum_;
    tilingData->simdUsedCoreNum = simdUsedCoreNum_;
    tilingData->simtUsedCoreNum = simtUsedCoreNum_;
    tilingData->frontCoreNum = inputData.frontCoreNum;
    tilingData->blockSize = inputData.blockSize;
    tilingData->tailBlockSize = inputData.tailBlockSize;
    tilingData->loopsPerFrontCore = inputData.loopsPerFrontCore;
    tilingData->loopsPerTailCore = inputData.loopsPerTailCore;
    tilingData->tilingKey = GetTilingKey();
}

void IndexFillSimtTiling::DumpTilingInfo()
{
    IndexFill::IndexFillSimtTilingData* tilingData = context_->GetTilingData<IndexFill::IndexFillSimtTilingData>();
    OP_LOGI(context_->GetNodeName(), "IndexFill tilingInfo is: %s", tilingData->ToString().c_str());
}

REGISTER_OPS_TILING_TEMPLATE(IndexFill, IndexFillSimtTiling, 10);
}  // namespace optiling
