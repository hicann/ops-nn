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
 * \file reverse_sequence_bas_tiling.cpp
 * \brief
 */

#include "reverse_sequence_bas_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "error_util.h"
#include "index/reverse_sequence/op_kernel/arch35/reverse_sequence_struct.h"
#include "index/reverse_sequence/op_kernel/arch35/reverse_sequence_tiling_key.h"

using namespace AscendC;
using namespace ge;

namespace optiling
{
using namespace ReverseSequence;

static constexpr int64_t TYPE_BAS = 6;
static constexpr int64_t UB_RESERVED_SIZE = 512;
static constexpr int64_t DCACHE_SIZE = 131072;  // 128k or 32k
static constexpr int64_t BLOCK_SPLIT_THRESHOLD = 8192;
static constexpr int64_t DIM_B = 0;
static constexpr int64_t DIM_A = 1;
static constexpr int64_t DIM_S = 2;

static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t DIGIT_TWO = 2;
static constexpr int64_t DIGIT_FOUR = 4;
static constexpr int64_t SPLIT_DIM_S = 1;
static constexpr int64_t SPLIT_DIM_A = 2;
static constexpr int64_t SPLIT_DIM_B = 3;
const uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;
static constexpr int64_t TEMPLATE_MODE = 2;


void ReverseSequenceBASTiling::InitializationVars()
{
    oneBlockNum_ = Ops::Base::GetUbBlockSize(context_) / inputData_.xDtypeSize;
    availableUb_ = static_cast<int64_t>(ubSize_  - UB_RESERVED_SIZE - DCACHE_SIZE) / inputData_.xDtypeSize;
}

bool ReverseSequenceBASTiling::IsCapable()
{
    if (inputData_.comBineType != TYPE_BAS) {
        return false;
    }

    InitializationVars();
    
    return true;
}


int64_t ReverseSequenceBASTiling::CalcBufferSize(int64_t inB, int64_t inA, int64_t inS)
{
    int64_t tmpInDataBufferSize = Ops::Base::CeilAlign(inB * inA * inS, oneBlockNum_);
    
    int64_t tmpOutDataBufferSize = tmpInDataBufferSize;

    int64_t tmpTotalBufferSize = (tmpInDataBufferSize + tmpOutDataBufferSize) * DOUBLE_BUFFER;
    
    return tmpTotalBufferSize;
}

void ReverseSequenceBASTiling::CalcSplitDimB()
{    
    int64_t inDimBLower = 1;
    int64_t inDimBUpper = inputData_.inputDim[DIM_B];
    while (inDimBLower < inDimBUpper) {
        int64_t inDimBMid = (inDimBLower + inDimBUpper + 1) / DIGIT_TWO;
        // split_b 时单次处理所有Batch
        int64_t midSeqBuf = Ops::Base::CeilAlign(Ops::Base::CeilDiv(inDimBMid * inputData_.seqLengthsDtypeSize, inputData_.xDtypeSize), oneBlockNum_) * DOUBLE_BUFFER;
        int64_t midBuffer = CalcBufferSize(inDimBMid, inputData_.inputDim[DIM_A], inputData_.inputDim[DIM_S]) + midSeqBuf;
        if (midBuffer <= availableUb_) {
            inDimBLower = inDimBMid;
        } else {
            inDimBUpper = inDimBMid - 1;
        }
    }
    ubFactorS_ = inputData_.inputDim[DIM_S];
    ubFactorA_ = inputData_.inputDim[DIM_A];
    ubFactorB_ = inDimBLower;

    bLoop_ = (inputData_.inputDim[DIM_B] + ubFactorB_ - 1) / ubFactorB_;
    aLoop_ = 1;
    sLoop_ = 1;
    splitMode_ = SPLIT_DIM_B;
    seqUbByte_ = Ops::Base::CeilAlign(ubFactorB_ * inputData_.seqLengthsDtypeSize, static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_)));
    if (inputData_.inputDim[DIM_S] * inputData_.inputDim[DIM_A] < threadNumX_) {
        threadNumX_ = inputData_.inputDim[DIM_S] * inputData_.inputDim[DIM_A];
    }
}

void ReverseSequenceBASTiling::CalcSplitDimA()
{
    int64_t inDimALower = 1;
    int64_t inDimAUpper = inputData_.inputDim[DIM_A];
    while (inDimALower < inDimAUpper) {
        int64_t inDimAMid = (inDimALower + inDimAUpper + 1) / DIGIT_TWO;
        int64_t midBuffer = CalcBufferSize(1, inDimAMid, inputData_.inputDim[DIM_S]);
        if (midBuffer <= availableUb_) {
            inDimALower = inDimAMid;
        } else {
            inDimAUpper = inDimAMid - 1;
        }
    }
    ubFactorS_ = inputData_.inputDim[DIM_S];
    ubFactorA_ = inDimALower;

    if (ubFactorA_ <= 0) {
        OP_LOGE(context_, "ReverseSequence ubFactorA_ is %ld.", ubFactorA_);
        return;
    }
    ubFactorB_ = 1;
    bLoop_ = inputData_.inputDim[DIM_B];
    aLoop_ = (inputData_.inputDim[DIM_A] + ubFactorA_ - 1) / ubFactorA_;
    sLoop_ = 1;
    splitMode_ = SPLIT_DIM_A;
}

void ReverseSequenceBASTiling::CalcSplitDimS()
{
    int64_t inDimS = std::min(inputData_.inputDim[DIM_S], (availableUb_ / DIGIT_FOUR));
    ubFactorS_ = inDimS;

    if (ubFactorS_ <= 0) {
        OP_LOGE(context_, "ReverseSequence ubFactorS_ is %ld.", ubFactorS_);
        return;
    }
    ubFactorB_ = 1;
    ubFactorA_ = 1;
    bLoop_ = inputData_.inputDim[DIM_B];
    aLoop_ = inputData_.inputDim[DIM_A];
    sLoop_ = (inputData_.inputDim[DIM_S] + ubFactorS_ - 1) / ubFactorS_;
    splitMode_ = SPLIT_DIM_S;
}

void ReverseSequenceBASTiling::DoUBTilingSingle()
{
    int64_t oneBatchBuffer = CalcBufferSize(1, inputData_.inputDim[DIM_A], inputData_.inputDim[DIM_S]);
    int64_t oneABuffer = CalcBufferSize(1, 1, inputData_.inputDim[DIM_S]);
    if (oneBatchBuffer <= 0 || oneABuffer <= 0) {
        bLoop_ = 0;
        aLoop_ = 0;
        sLoop_ = 0;
        isZero_ = true;
        return;
    }
    int64_t oneSeqBuffer = Ops::Base::CeilAlign(Ops::Base::CeilDiv(inputData_.seqLengthsDtypeSize, inputData_.xDtypeSize), oneBlockNum_) * DOUBLE_BUFFER;
    // AS 全载
    if (oneBatchBuffer + oneSeqBuffer <= availableUb_) {
        CalcSplitDimB();
        return;
    }
    // S 全载
    if (oneABuffer <= availableUb_) {
        CalcSplitDimA();
        return;
    }
    // S 不全载
    CalcSplitDimS();
}

void ReverseSequenceBASTiling::DoUBTiling()
{
    int64_t ubStep = BLOCK_SPLIT_THRESHOLD / inputData_.xDtypeSize;
    do {
        DoUBTilingSingle();
        if (bLoop_ * aLoop_ * sLoop_ >= static_cast<int64_t>(coreNum_) || isZero_) {
            break;
        }
        availableUb_ -= ubStep;
    } while (availableUb_ > ubStep);
}

void ReverseSequenceBASTiling::DoBlockTiling()
{
    int64_t totalLoop = bLoop_ * aLoop_ * sLoop_;
    blockFactor_ = totalLoop / static_cast<int64_t>(coreNum_);
    blockTail_ = totalLoop - blockFactor_ * static_cast<int64_t>(coreNum_);
    usedCoreNum_ = blockFactor_ == 0 ? blockTail_ : static_cast<int64_t>(coreNum_);

    if (splitMode_ == SPLIT_DIM_S) {
        inUbSize_ = ubFactorB_ * ubFactorA_ * Ops::Base::CeilAlign(ubFactorS_, oneBlockNum_);
    } else if (splitMode_ == SPLIT_DIM_A) {
        inUbSize_ = ubFactorB_ * Ops::Base::CeilAlign(ubFactorA_ * ubFactorS_, oneBlockNum_);
    } else {
        inUbSize_ = Ops::Base::CeilAlign(ubFactorB_ * ubFactorA_ * ubFactorS_, oneBlockNum_);
    }
}

ge::graphStatus ReverseSequenceBASTiling::DoOpTiling()
{
    DoUBTiling();
    DoBlockTiling();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReverseSequenceBASTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t ReverseSequenceBASTiling::GetTilingKey() const
{
    OP_LOGD("ReverseSequenceBASTiling::GetTilingKey begin");
    const uint64_t tilingKey = GET_TPL_TILING_KEY(TEMPLATE_MODE, static_cast<uint64_t>(inputData_.xDtypeSize), 1);
    OP_LOGD(context_->GetNodeName(), "tilingKey is: [%lu]", tilingKey);
    return tilingKey;
}

ge::graphStatus ReverseSequenceBASTiling::GetWorkspaceSize()
{
    uint32_t sysWorkspace = WS_SYS_SIZE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sysWorkspace;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReverseSequenceBASTiling::PostTiling()
{
    OP_LOGD("ReverseSequenceBASTiling::PostTiling begin");
    context_->SetBlockDim(usedCoreNum_);

    auto res = context_->SetLocalMemorySize(ubSize_  - UB_RESERVED_SIZE - DCACHE_SIZE);
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS),
        OP_LOGE(context_->GetNodeName(), "SetLocalMemorySize ubSize = %lu failed.", ubSize_  - UB_RESERVED_SIZE - DCACHE_SIZE), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void ReverseSequenceBASTiling::SetTilingData()
{
     ReverseSequence::ReverseSequenceBASTilingData* tilingData =
        context_->GetTilingData<ReverseSequence::ReverseSequenceBASTilingData>();

    tilingData->bDim = inputData_.inputDim[DIM_B];
    tilingData->aDim = inputData_.inputDim[DIM_A];
    tilingData->sDim = inputData_.inputDim[DIM_S];
    tilingData->blockFactor = blockFactor_;
    tilingData->blockTail = blockTail_;
    tilingData->ubFactorB = ubFactorB_;
    tilingData->ubFactorA = ubFactorA_;
    tilingData->ubFactorS = ubFactorS_;
    tilingData->bLoop = bLoop_;
    tilingData->aLoop = aLoop_;
    tilingData->sLoop = sLoop_;
    tilingData->inUbSize = inUbSize_;
    tilingData->seqUbByte = seqUbByte_;
    tilingData->threadNumX = threadNumX_;
    tilingData->usedCoreNum = usedCoreNum_;
    tilingData->splitMode = splitMode_;
    tilingData->dtypeSize = inputData_.xDtypeSize;
}

std::string ReverseSequenceBASTiling::TilingDataToString()
{
    ReverseSequence::ReverseSequenceBASTilingData* tilingData =
        context_->GetTilingData<ReverseSequence::ReverseSequenceBASTilingData>();
    std::string str = " bDim:" + std::to_string(tilingData->bDim);
    str += " aDim:" + std::to_string(tilingData->aDim);
    str += " sDim:" + std::to_string(tilingData->sDim);
    str += " blockFactor:" + std::to_string(tilingData->blockFactor);
    str += " blockTail:" + std::to_string(tilingData->blockTail);
    str += " ubFactorB:" + std::to_string(tilingData->ubFactorB);
    str += " ubFactorA:" + std::to_string(tilingData->ubFactorA);
    str += " ubFactorS:" + std::to_string(tilingData->ubFactorS);
    str += " bLoop:" + std::to_string(tilingData->bLoop);
    str += " aLoop:" + std::to_string(tilingData->aLoop);
    str += " sLoop:" + std::to_string(tilingData->sLoop);
    str += " inUbSize:" + std::to_string(tilingData->inUbSize);
    str += " seqUbByte:" + std::to_string(tilingData->seqUbByte);
    str += " threadNumX:" + std::to_string(tilingData->threadNumX);
    str += " usedCoreNum:" + std::to_string(tilingData->usedCoreNum);
    str += " splitMode:" + std::to_string(tilingData->splitMode);
    str += " dtypeSize:" + std::to_string(tilingData->dtypeSize);
    return str;
}

void ReverseSequenceBASTiling::DumpTilingInfo()
{
    OP_LOGI(context_, "ReverseSequenceBAS tilingInfo is :%s", TilingDataToString().c_str());
}

ge::graphStatus ReverseSequenceBASTiling::GetPlatformInfo()
{
    return GetReverseSequencePlatformInfo(context_, ubSize_, coreNum_);
}

ge::graphStatus ReverseSequenceBASTiling::GetShapeAttrsInfo()
{
    return GetReverseSequenceShapeAttrsInfo(context_, inputData_);
}

REGISTER_TILING_TEMPLATE("ReverseSequence", ReverseSequenceBASTiling, 2);


} // namespace optiling