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
 * \file reverse_sequence_bs_tiling.cpp
 * \brief
 */

#include "reverse_sequence_bs_tiling.h"
#include <array>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "error_util.h"
#include "../../op_kernel/arch35/reverse_sequence_struct.h"
#include "../../op_kernel/arch35/reverse_sequence_tiling_key.h"

using namespace AscendC;
using namespace ge;

namespace optiling
{
using namespace ReverseSequence;

static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t DIGIT_TWO = 2;
static constexpr int64_t DIGIT_FOUR = 4;

static constexpr int64_t COPY_THRESHOLD = 128;
static constexpr int64_t DCACHE_SIZE = 131072;  // 128k 
static constexpr int64_t SPLIT_DIMA = 1;
static constexpr int64_t SPLIT_DIMS = 3;
static constexpr int64_t SPLIT_DIMB = 2;

static constexpr int64_t UB_RESERVED_SIZE = 512;
static constexpr int64_t BLOCK_SPLIT_THRESHOLD = 8192;
const uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;
static constexpr int64_t TEMPLATE_MODE = 3;
static constexpr int64_t TYPE_BS = 1;
static constexpr int64_t TYPE_ABS = 3;

template <typename T>
static std::string ToString(const T* value, size_t size) {
  std::string r = "[";
  for (size_t i = 0; i < size; i++) {
    r = r + std::to_string(value[i]) + ", ";
  }
  r = r + "]";
  return r;
}

bool ReverseSequenceBSTiling::IsCapable()
{
    if (inputData_.comBineType != TYPE_BS && inputData_.comBineType != TYPE_ABS) {
        return false;
    }

    if (inputData_.comBineType == TYPE_ABS) {
        dimB_ = 1;
        dimS_ = 2;
        aDim_ = inputData_.inputDim[dimA_];
        if (aDim_ * inputData_.inputDim[dimB_] * inputData_.inputDim[dimS_] < COPY_THRESHOLD) {
            // ABS 较小的离散场景不处理
            return false;
        }
        return true;
    }

    if (inputData_.inputDim[dimB_] * inputData_.inputDim[dimS_] < COPY_THRESHOLD) {
        // BS 较小的离散场景不处理
        return false;
    }

    return true;
}

ge::graphStatus ReverseSequenceBSTiling::InitializationVars()
{
    oneBlockNum_ = Ops::Base::GetUbBlockSize(context_) / inputData_.xDtypeSize;
    OP_CHECK_IF((ubSize_ <= DCACHE_SIZE),
                    OP_LOGE(context_->GetNodeName(), "ub size:%lu less than Dcache Size:128k", ubSize_),
                    return ge::GRAPH_FAILED);
    ubSize_  = ubSize_ - DCACHE_SIZE;
    availableUb_ = static_cast<int64_t>(ubSize_) / inputData_.xDtypeSize;
    return ge::GRAPH_SUCCESS;
}

void ReverseSequenceBSTiling::DoBlockTiling()
{
    int64_t totalLoop = bLoop_ * sLoop_ * aLoop_;
    blockFactor_ = totalLoop / static_cast<int64_t>(coreNum_);
    blockTail_ = totalLoop - blockFactor_ * static_cast<int64_t>(coreNum_);
    usedCoreNum_ = blockFactor_ == 0 ? blockTail_ : static_cast<int64_t>(coreNum_);

    inUbSize_ = ubFactorA_ * ubFactorB_ * Ops::Base::CeilAlign(ubFactorS_, oneBlockNum_);
}

int64_t ReverseSequenceBSTiling::CalcBufferSize(int64_t inA, int64_t inB, int64_t inS, bool isSplitA)
{
    int64_t tmpInDataBufferSize = inA * inB * Ops::Base::CeilAlign(inS, oneBlockNum_);

    int64_t tmpOutDataBufferSize = tmpInDataBufferSize;

    int64_t tmpTotalBufferSize = (tmpInDataBufferSize + tmpOutDataBufferSize) * DOUBLE_BUFFER;
    return tmpTotalBufferSize;
}

void ReverseSequenceBSTiling::CalcSplitDimB()
{
    int64_t inDimBLower = 1;
    int64_t inDimBUpper = inputData_.inputDim[dimB_];
    while (inDimBLower < inDimBUpper) {
        int64_t inDimBMid = (inDimBLower + inDimBUpper + 1) / DIGIT_TWO;
        int64_t midBuffer = CalcBufferSize(1, inDimBMid, inputData_.inputDim[dimS_]);
        if (midBuffer <= availableUb_) {
            inDimBLower = inDimBMid;
        } else {
            inDimBUpper = inDimBMid - 1;
        }
    }
    ubFactorA_ = 1;
    ubFactorB_ = inDimBLower;
    ubFactorS_ = inputData_.inputDim[dimS_];
    if (ubFactorB_ <= 0) {
        OP_LOGE(context_, "ReverseSequence ubFactorB_ is %ld.", ubFactorB_);
        return;
    }
    
    aLoop_ = aDim_;
    bLoop_ = (inputData_.inputDim[dimB_] + ubFactorB_ - 1) / ubFactorB_;
    sLoop_ = 1;
    splitMode_ = SPLIT_DIMB;
}

void ReverseSequenceBSTiling::CalcSplitDimS()
{
    ubFactorS_ = std::min(inputData_.inputDim[dimS_], (availableUb_ / DIGIT_FOUR));
    if (ubFactorS_ <= 0) {
        OP_LOGE(context_, "ReverseSequence ubFactorS_ is %ld.", ubFactorS_);
        return;
    }

    ubFactorB_ = 1;
    ubFactorA_ = 1;
    aLoop_ = aDim_;
    bLoop_ = inputData_.inputDim[dimB_];
    sLoop_ = (inputData_.inputDim[dimS_] + ubFactorS_ - 1) / ubFactorS_;
    splitMode_ = SPLIT_DIMS;
}

void ReverseSequenceBSTiling::CalcSplitDimA()
{    
    int64_t inDimALower = 1;
    int64_t inDimAUpper = aDim_;
    while (inDimALower < inDimAUpper) {
        int64_t inDimAMid = (inDimALower + inDimAUpper + 1) / DIGIT_TWO;
        // split_a 
        int64_t midBuffer = CalcBufferSize(inDimAMid, inputData_.inputDim[dimB_], inputData_.inputDim[dimS_], true);
        if (midBuffer <= availableUb_) {
            inDimALower = inDimAMid;
        } else {
            inDimAUpper = inDimAMid - 1;
        }
    }
    
    ubFactorA_ = inDimALower;
    ubFactorB_ = inputData_.inputDim[dimB_];
    ubFactorS_ = inputData_.inputDim[dimS_];
    int64_t inputBufferSize = ubFactorA_ * inputData_.inputDim[dimB_] * inputData_.inputDim[dimS_];
    aLoop_ = (aDim_ + ubFactorA_ - 1) / ubFactorA_;
    bLoop_ = 1;
    sLoop_ = 1;
    splitMode_ = SPLIT_DIMA;
}

void ReverseSequenceBSTiling::DoUBTilingSingle()
{
    int64_t oneABuffer = CalcBufferSize(1, inputData_.inputDim[dimB_], inputData_.inputDim[dimS_], true);
    int64_t oneBatchBuffer = CalcBufferSize(1, 1, inputData_.inputDim[dimS_]);
    if (oneBatchBuffer <= 0 || oneABuffer <= 0) {
        bLoop_ = 0;
        sLoop_ = 0;
        aLoop_ = 0;
        isZero_ = true;
        return;
    }
    // BS 全载
    if (oneABuffer <= availableUb_) {
        CalcSplitDimA();
        return;
    }
    // S 全载
    if (oneBatchBuffer <= availableUb_) {
        CalcSplitDimB();
        return;
    }
    // S 不全载
    CalcSplitDimS();
}

void ReverseSequenceBSTiling::DoUBTiling()
{
    int64_t ubStep = BLOCK_SPLIT_THRESHOLD / inputData_.xDtypeSize;
    do {
        DoUBTilingSingle();
        if (aLoop_ * bLoop_ * sLoop_ >= static_cast<int64_t>(coreNum_) || isZero_) {
            break;
        }
        availableUb_ -= ubStep;
    } while (availableUb_ > ubStep);
}

ge::graphStatus ReverseSequenceBSTiling::DoOpTiling()
{
    OP_CHECK_IF(InitializationVars() != ge::GRAPH_SUCCESS,
                    OP_LOGE(context_->GetNodeName(), "ub size:%lu less than Dcache Size:128k", ubSize_),
                    return ge::GRAPH_FAILED);
    DoUBTiling();
    DoBlockTiling();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReverseSequenceBSTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t ReverseSequenceBSTiling::GetTilingKey() const
{
    OP_LOGD("ReverseSequenceBSTiling::GetTilingKey begin");
    const uint64_t tilingKey = GET_TPL_TILING_KEY(TEMPLATE_MODE, inputData_.xDtypeSize, addrRange_);
    OP_LOGD(context_->GetNodeName(), "tilingKey is: [%lu]", tilingKey);
    return tilingKey;
}

ge::graphStatus ReverseSequenceBSTiling::GetWorkspaceSize()
{
    uint32_t sysWorkspace = WS_SYS_SIZE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sysWorkspace;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReverseSequenceBSTiling::PostTiling()
{
    context_->SetBlockDim(coreNum_);
    if (inputData_.xShapeSize >= INT32_MAX) {
        addrRange_ = 1;
    }
    context_->SetTilingKey(GetTilingKey());
    auto res = context_->SetLocalMemorySize(ubSize_);
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS),
        OP_LOGE(context_->GetNodeName(), "SetLocalMemorySize ubSize = %lu failed.", ubSize_), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void ReverseSequenceBSTiling::SetTilingData() const
{
     ReverseSequence::ReverseSequenceBSTilingData* tilingData =
        context_->GetTilingData<ReverseSequence::ReverseSequenceBSTilingData>();

    tilingData->bDim = inputData_.inputDim[dimB_];
    tilingData->sDim = inputData_.inputDim[dimS_];
    tilingData->aDim = aDim_;
    tilingData->blockFactor = blockFactor_;
    tilingData->blockTail = blockTail_;
    tilingData->ubFactorB = ubFactorB_;
    tilingData->ubFactorS = ubFactorS_;
    tilingData->ubFactorA = ubFactorA_;
    tilingData->bLoop = bLoop_;
    tilingData->sLoop = sLoop_;
    tilingData->aLoop = aLoop_;
    tilingData->inUbSize = inUbSize_;
    tilingData->usedCoreNum = usedCoreNum_;
    tilingData->splitMode = splitMode_;
    tilingData->dtypeSize = inputData_.xDtypeSize;
}

void ReverseSequenceBSTiling::DumpTilingInfo()
{
    ReverseSequenceBSTilingData* tilingData =
        context_->GetTilingData<ReverseSequenceBSTilingData>();
    std::string str;
    str += " bDim:" + std::to_string(tilingData->bDim);
    str += " sDim:" + std::to_string(tilingData->sDim);
    str += " aDim:" + std::to_string(tilingData->aDim);
    str += " blockFactor:" + std::to_string(tilingData->blockFactor);
    str += " blockTail:" + std::to_string(tilingData->blockTail);
    str += " ubFactorA:" + std::to_string(tilingData->ubFactorA);
    str += " ubFactorB:" + std::to_string(tilingData->ubFactorB);
    str += " ubFactorS:" + std::to_string(tilingData->ubFactorS);
    str += " aLoop:" + std::to_string(tilingData->aLoop);
    str += " bLoop:" + std::to_string(tilingData->bLoop);
    str += " sLoop:" + std::to_string(tilingData->sLoop);
    str += " inUbSize:" + std::to_string(tilingData->inUbSize);
    str += " usedCoreNum:" + std::to_string(tilingData->usedCoreNum);
    str += " dtypeSize:" + std::to_string(tilingData->dtypeSize);
    str += " splitMode:" + std::to_string(tilingData->splitMode);
    str += " inputDims:" + ToString(inputData_.inputDim, allDims);
    OP_LOGI(context_, "%s", str.c_str());
}
ge::graphStatus ReverseSequenceBSTiling::GetPlatformInfo()
{
    return GetReverseSequencePlatformInfo(context_, ubSize_, coreNum_);
}

ge::graphStatus ReverseSequenceBSTiling::GetShapeAttrsInfo()
{
    return GetReverseSequenceShapeAttrsInfo(context_, inputData_);
}

REGISTER_TILING_TEMPLATE("ReverseSequence", ReverseSequenceBSTiling, 3);

}  // namespace optiling