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
 * \file inplace_index_add_simt_tiling.cpp
 * \brief
 */

#include "op_common/op_host/util/platform_util.h"
#include "error_util.h"
#include "log/log.h"
#include "op_api/runtime2_util.h"
#include "op_host/util/math_util.h"
#include "op_host/tiling_templates_registry.h"
#include "index_fill_tiling_simd.h"
#include "index_fill_tiling_common.h"

namespace optiling {
using namespace IndexFill;

constexpr uint64_t BUFFER_NUM = 2;
constexpr uint64_t SPLITQ_SIZE = 2048;
constexpr uint64_t MAX_MASK_SIZE = 2048;
constexpr uint64_t SIMD_THRESHPLD = 128;

bool IndexFillSimdTiling::IsCapable()
{
    if (inputData.numel == 0 || inputData.Q * inputData.dtypeSize <= SIMD_THRESHPLD) {
        return false;
    }
    return true;
}

void IndexFillSimdTiling::CalcUsedCoreNum()
{
    // usedCoreNumPN表示PN两个维度使用的核数.
    usedCoreNumPN_ = coreNum_;
    uint64_t pn = inputData.P * inputData.N;
    blockFactorPN_ = pn / usedCoreNumPN_;
    tailBlockNumPN_ = pn - blockFactorPN_ * usedCoreNumPN_;
    if (blockFactorPN_ <= 0) {
        usedCoreNumPN_ = tailBlockNumPN_;
    }

    usedCoreNumQ_ = 1;
    // q较大时需要切分q，计算切分q所需要的分核数，且一定是p*n较小的场景
    while (usedCoreNumPN_ * usedCoreNumQ_ < coreNum_ / TWO &&
           Ops::Base::CeilAlign(inputData.Q, usedCoreNumQ_) * inputData.dtypeSize >= SPLITQ_SIZE) {
        splitQ_ = 1;
        usedCoreNumQ_ += 1;
    }

    blockFactorQ_ = Ops::Base::CeilDiv(inputData.Q, usedCoreNumQ_);
    blockTailQ_ = inputData.Q - blockFactorQ_ * (usedCoreNumQ_ - 1);

    usedCoreNum_ = usedCoreNumPN_ * usedCoreNumQ_;
}

void IndexFillSimdTiling::CalcUBBlock()
{
    ubSize_ = ubSize_ - SIMT_DCACHE_SIZE;
    uint64_t ubBlock = static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_));
    uint64_t oneBlockNum = ubBlock / inputData.dtypeSize;

    uint64_t blockFactorAlignQ = Ops::Base::CeilAlign(blockFactorQ_, oneBlockNum);
    uint64_t blockFactorAlignMask = Ops::Base::CeilAlign(inputData.N, ubBlock / ge::GetSizeByDataType(ge::DT_INT8));
    uint64_t nBuffer = static_cast<uint64_t>(DOUBLE_BUFFER + 1);
    if (blockFactorAlignQ * inputData.dtypeSize * nBuffer + blockFactorAlignMask <= ubSize_) { // 一次全搬入，不需要切分
        blockFactorUbBufferMask_ = blockFactorAlignMask;
        blockFactorUbFactorQ_ = blockFactorAlignQ;
    } else {
        blockFactorUbBufferMask_ = std::min(MAX_MASK_SIZE, blockFactorAlignMask);
        blockFactorUbFactorQ_ = Ops::Base::FloorAlign((ubSize_ - blockFactorUbBufferMask_) / nBuffer / inputData.dtypeSize, oneBlockNum);
        if (blockFactorUbFactorQ_ * TWO > blockFactorQ_) {
            blockFactorUbFactorQ_ = Ops::Base::CeilAlign(Ops::Base::CeilDiv(blockFactorQ_, TWO), oneBlockNum);
        }
    }

    blockFactorUbFactorQ_ = std::min(blockFactorUbFactorQ_, blockFactorQ_);
    blockFactorTileNumQ_ = Ops::Base::CeilDiv(blockFactorQ_, blockFactorUbFactorQ_);
    blockFactorUbTailQ_ = blockFactorQ_ - (blockFactorTileNumQ_ - 1) * blockFactorUbFactorQ_;

    uint64_t blockTailAlignQ = Ops::Base::CeilAlign(blockTailQ_, oneBlockNum);
    uint64_t blockTailAlignMask = Ops::Base::CeilAlign(inputData.N, ubBlock / ge::GetSizeByDataType(ge::DT_INT8));
    if (blockTailAlignQ * inputData.dtypeSize * nBuffer + blockTailAlignMask <= ubSize_) { // 一次全搬入，不需要切分
        blockTailUbBufferMask_ = blockTailAlignMask;
        blockTailUbFactorQ_ = blockTailAlignQ;
    } else {
        blockTailUbBufferMask_ = std::min(MAX_MASK_SIZE, blockTailAlignMask);
        blockTailUbFactorQ_ = Ops::Base::FloorAlign((ubSize_ - blockTailUbBufferMask_) / nBuffer / inputData.dtypeSize, oneBlockNum);
        if (blockTailUbFactorQ_ * TWO > blockTailQ_) {
            blockTailUbFactorQ_ = Ops::Base::CeilAlign(Ops::Base::CeilDiv(blockTailQ_, TWO), oneBlockNum);
        }
    }

    blockTailUbFactorQ_ = std::min(blockTailUbFactorQ_, blockTailQ_);
    blockTailTileNumQ_ = Ops::Base::CeilDiv(blockTailQ_, blockTailUbFactorQ_);
    blockTailUbTailQ_ = blockTailQ_ - (blockTailTileNumQ_ - 1) * blockTailUbFactorQ_;
}

ge::graphStatus IndexFillSimdTiling::DoOpTiling()
{
    CalcUsedCoreNum();
    CalcUBBlock();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexFillSimdTiling::GetWorkspaceSize()
{
    uint32_t sysWorkspace = WS_SYS_SIZE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = (inputData.N * sizeof(int8_t)) + sysWorkspace; // workspace上申请额外申请一片内存，用于存放索引位图

    // 涉及到核间同步，需要设置该模式.
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

uint64_t IndexFillSimdTiling::GetTilingKey() const
{
    uint64_t templateMode = static_cast<uint64_t>(TPL_MODE_TEMPLATE_SIMD);
    uint64_t dTypeMode = static_cast<uint64_t>(TPL_MODE_DTYPE_B32);
    uint64_t indicesNum = inputData.indicesNum;
    if (indicesNum > MAX_INT32) {
        dTypeMode = static_cast<uint64_t>(TPL_MODE_DTYPE_B64);
    }
    const uint64_t tilingKey = GET_TPL_TILING_KEY(templateMode, dTypeMode);
    return tilingKey;
}

void IndexFillSimdTiling::SetTilingData()
{
    IndexFill::IndexFillSimdTilingData* tilingData = context_->GetTilingData<IndexFill::IndexFillSimdTilingData>();
    tilingData->p = static_cast<int64_t>(inputData.P);
    tilingData->n = static_cast<int64_t>(inputData.N);
    tilingData->q = static_cast<int64_t>(inputData.Q);
    tilingData->indicesNum = static_cast<int64_t>(inputData.indicesNum);
    tilingData->splitQ = static_cast<int64_t>(splitQ_);
    tilingData->usedCoreNum = static_cast<int64_t>(usedCoreNum_);
    tilingData->blockFactorPN = static_cast<int64_t>(blockFactorPN_);
    tilingData->usedCoreNumPN = static_cast<int64_t>(usedCoreNumPN_);
    tilingData->tailBlockNumPN = static_cast<int64_t>(tailBlockNumPN_);
    tilingData->usedCoreNumQ = static_cast<int64_t>(usedCoreNumQ_);
    tilingData->blockFactorQ = static_cast<int64_t>(blockFactorQ_);
    tilingData->blockTailQ = static_cast<int64_t>(blockTailQ_);
    tilingData->blockFactorUbBufferMask = static_cast<int64_t>(blockFactorUbBufferMask_);
    tilingData->blockFactorTileNumQ = static_cast<int64_t>(blockFactorTileNumQ_);
    tilingData->blockFactorUbFactorQ = static_cast<int64_t>(blockFactorUbFactorQ_);
    tilingData->blockFactorUbTailQ = static_cast<int64_t>(blockFactorUbTailQ_);
    tilingData->blockTailUbBufferMask = static_cast<int64_t>(blockTailUbBufferMask_);
    tilingData->blockTailTileNumQ = static_cast<int64_t>(blockTailTileNumQ_);
    tilingData->blockTailUbFactorQ = static_cast<int64_t>(blockTailUbFactorQ_);
    tilingData->blockTailUbTailQ = static_cast<int64_t>(blockTailUbTailQ_);
}

void IndexFillSimdTiling::DumpTilingInfo()
{
    IndexFill::IndexFillSimdTilingData* tilingData = context_->GetTilingData<IndexFill::IndexFillSimdTilingData>();
    std::ostringstream info;
    info << "\n p: " << tilingData->p;
    info << "\n n: " << tilingData->n;
    info << "\n q: " << tilingData->q;
    info << "\n splitQ: " << tilingData->splitQ;
    info << "\n coreNum: " << coreNum_;
    info << "\n dtypeSize: " << inputData.dtypeSize;
    info << "\n usedCoreNum: " << tilingData->usedCoreNum;
    info << "\n blockFactorPN: " << tilingData->blockFactorPN;
    info << "\n tailBlockNumPN: " << tilingData->tailBlockNumPN;
    info << "\n usedCoreNumQ: " << tilingData->usedCoreNumQ;
    info << "\n blockFactorQ: " << tilingData->blockFactorQ;
    info << "\n blockTailQ: " << tilingData->blockTailQ;
    info << "\n ubSize_: " << ubSize_;
    info << "\n blockFactorUbBufferMask: " << tilingData->blockFactorUbBufferMask;
    info << "\n blockFactorTileNumQ: " << tilingData->blockFactorTileNumQ;
    info << "\n blockFactorUbFactorQ: " << tilingData->blockFactorUbFactorQ;
    info << "\n blockFactorUbTailQ: " << tilingData->blockFactorUbTailQ;
    info << "\n blockTailUbBufferMask: " << tilingData->blockTailUbBufferMask;
    info << "\n blockTailTileNumQ: " << tilingData->blockTailTileNumQ;
    info << "\n blockTailUbFactorQ: " << tilingData->blockTailUbFactorQ;
    info << "\n blockTailUbTailQ: " << tilingData->blockTailUbTailQ;
    OP_LOGI(context_->GetNodeName(), "IndexFill tilingInfo is: %s", info.str().c_str());
}

REGISTER_OPS_TILING_TEMPLATE(IndexFill, IndexFillSimdTiling, 1);
} // namespace optiling
