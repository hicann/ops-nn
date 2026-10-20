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
 * \file index_fill_tiling_simd_dma.cpp
 * \brief
 */

#include "op_common/op_host/util/platform_util.h"
#include "error_util.h"
#include "log/log.h"
#include "op_api/runtime2_util_nn.h"
#include "op_host/util/math_util.h"
#include "op_host/tiling_templates_registry.h"
#include "index_fill_tiling_simd_dma.h"
#include "index_fill_tiling_common.h"

namespace optiling {
using namespace IndexFill;

constexpr uint64_t MAX_MASK_SIZE = 8192;
constexpr uint64_t SIMD_THRESHPLD = 128;

bool IndexFillSimdDmaTiling::IsCapable()
{
    CalcUsedCoreNum();
    CalcUBBlock();
    if (inputData.numel == 0 || inputData.Q * inputData.dtypeSize <= SIMD_THRESHPLD || !inputData.isAscend960DT ||
        !isUseGatherDma_) {
        return false;
    }
    return true;
}

void IndexFillSimdDmaTiling::CalcUsedCoreNum()
{
    // usedCoreNumPN表示PN两个维度使用的核数.
    usedCoreNumPN_ = coreNum_;
    uint64_t pn = inputData.P * inputData.N;
    blockFactorPN_ = pn / usedCoreNumPN_;
    tailBlockNumPN_ = pn - blockFactorPN_ * usedCoreNumPN_;
    if (blockFactorPN_ <= 0) {
        usedCoreNumPN_ = tailBlockNumPN_;
    }

    normBlockQNum_ = blockFactorPN_ + 1;
    tailBlockQNum_ = blockFactorPN_;
    usedCoreNum_ = usedCoreNumPN_;
}

void IndexFillSimdDmaTiling::CalcUBBlock()
{
    ubSize_ = ubSize_ - SIMT_DCACHE_SIZE;
    uint64_t ubBlock = static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_));
    uint64_t oneBlockNum = ubBlock / inputData.dtypeSize;

    uint64_t blockFactorAlignMask = Ops::Base::CeilAlign(inputData.N, ubBlock / ge::GetSizeByDataType(ge::DT_INT8));
    blockFactorUbBufferMask_ = std::min(MAX_MASK_SIZE, blockFactorAlignMask);
    uint64_t remainSize = (ubSize_ - blockFactorUbBufferMask_ - DOUBLE_BUFFER * ubBlock - DOUBLE_BUFFER * ubBlock) /
                          DOUBLE_BUFFER;
    uint64_t blockFactorOneQSize = inputData.Q * inputData.dtypeSize + sizeof(uint64_t);
    if (remainSize >= TWO * blockFactorOneQSize && normBlockQNum_ > TWO) {
        isUseGatherDma_ = true;
        indicesUbFactor_ = remainSize / blockFactorOneQSize;
        indicesUbFactorSizeAlign_ = Ops::Base::CeilAlign(indicesUbFactor_ * sizeof(uint64_t), ubBlock);
        blockFactorAlignQ_ = Ops::Base::CeilAlign(indicesUbFactor_ * inputData.Q, oneBlockNum);

        normBlockQNumLoop_ = Ops::Base::CeilDiv(normBlockQNum_, indicesUbFactor_);
        normBlockQNumTail_ = normBlockQNum_ - (normBlockQNumLoop_ - 1) * indicesUbFactor_;
        tailBlockQNumLoop_ = Ops::Base::CeilDiv(tailBlockQNum_, indicesUbFactor_);
        tailBlockQNumTail_ = tailBlockQNum_ - (tailBlockQNumLoop_ - 1) * indicesUbFactor_;
    }
}

ge::graphStatus IndexFillSimdDmaTiling::DoOpTiling()
{
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexFillSimdDmaTiling::GetWorkspaceSize()
{
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = (inputData.N * sizeof(int8_t)) +
                          sysWorkspaceSize_; // workspace上申请额外申请一片内存，用于存放索引位图

    // 涉及到核间同步，需要设置该模式.
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

uint64_t IndexFillSimdDmaTiling::GetTilingKey() const
{
    uint64_t templateMode = static_cast<uint64_t>(TPL_MODE_TEMPLATE_SIMD_DMA);
    uint64_t dTypeMode = static_cast<uint64_t>(TPL_MODE_DTYPE_B32);
    uint64_t indicesNum = inputData.indicesNum;
    if (indicesNum > MAX_INT32) {
        dTypeMode = static_cast<uint64_t>(TPL_MODE_DTYPE_B64);
    }
    const uint64_t tilingKey = GET_TPL_TILING_KEY(templateMode, dTypeMode);
    return tilingKey;
}

void IndexFillSimdDmaTiling::SetTilingData()
{
    IndexFill::IndexFillSimdDmaTilingData* tilingData = context_
                                                            ->GetTilingData<IndexFill::IndexFillSimdDmaTilingData>();
    tilingData->p = inputData.P;
    tilingData->n = inputData.N;
    tilingData->q = inputData.Q;
    tilingData->indicesNum = inputData.indicesNum;
    tilingData->usedCoreNum = static_cast<uint64_t>(usedCoreNum_);
    tilingData->blockFactorPN = blockFactorPN_;
    tilingData->tailBlockNumPN = tailBlockNumPN_;
    tilingData->blockFactorAlignQ = blockFactorAlignQ_;
    tilingData->blockFactorUbBufferMask = blockFactorUbBufferMask_;
    tilingData->indicesUbFactorSizeAlign = indicesUbFactorSizeAlign_;
    tilingData->indicesUbFactor = indicesUbFactor_;
    tilingData->normBlockQNumLoop = normBlockQNumLoop_;
    tilingData->normBlockQNumTail = normBlockQNumTail_;
    tilingData->tailBlockQNumLoop = tailBlockQNumLoop_;
    tilingData->tailBlockQNumTail = tailBlockQNumTail_;
    tilingData->fp4Mode = inputData.fp4Support;
}

void IndexFillSimdDmaTiling::DumpTilingInfo()
{
    IndexFill::IndexFillSimdDmaTilingData* tilingData = context_
                                                            ->GetTilingData<IndexFill::IndexFillSimdDmaTilingData>();
    std::ostringstream info;
    info << "\n p: " << tilingData->p;
    info << "\n n: " << tilingData->n;
    info << "\n q: " << tilingData->q;
    info << "\n coreNum: " << coreNum_;
    info << "\n dtypeSize: " << inputData.dtypeSize;
    info << "\n usedCoreNum: " << tilingData->usedCoreNum;
    info << "\n blockFactorPN: " << tilingData->blockFactorPN;
    info << "\n tailBlockNumPN: " << tilingData->tailBlockNumPN;
    info << "\n blockFactorAlignQ: " << tilingData->blockFactorAlignQ;
    info << "\n ubSize_: " << ubSize_;
    info << "\n blockFactorUbBufferMask: " << tilingData->blockFactorUbBufferMask;
    info << "\n indicesUbFactorSizeAlign: " << tilingData->indicesUbFactorSizeAlign;
    info << "\n indicesUbFactor: " << tilingData->indicesUbFactor;
    info << "\n normBlockQNumLoop: " << tilingData->normBlockQNumLoop;
    info << "\n normBlockQNumTail: " << tilingData->normBlockQNumTail;
    info << "\n tailBlockQNumLoop: " << tilingData->tailBlockQNumLoop;
    info << "\n tailBlockQNumTail: " << tilingData->tailBlockQNumTail;

    OP_LOGI(context_->GetNodeName(), "IndexFill tilingInfo is: %s", info.str().c_str());
}

REGISTER_OPS_TILING_TEMPLATE(IndexFill, IndexFillSimdDmaTiling, 1);
} // namespace optiling
