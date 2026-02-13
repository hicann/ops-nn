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
 * \file reverse_sequence_bsa_tiling.cpp
 * \brief
 */

#include "reverse_sequence_bsa_tiling.h"
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

static constexpr int64_t OUT_BUFFER_LEN = 1024;
static constexpr int64_t BUFFER_NUM = 2;

static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t DIGIT_TWO = 2;
static constexpr int64_t DIGIT_FOUR = 4;
static constexpr int64_t MIN_BLOCK_BYTES = 512;
static constexpr int64_t MAX_INPUT_ELEMENTS = std::numeric_limits<uint16_t>::max();

static constexpr int64_t GATHER_DIM_S = 0;
static constexpr int64_t GATHER_DIM_B = 1;
static constexpr int64_t NOT_GATHER = 1001;
static constexpr int64_t GATHER_THRESHOLD = 128;
static constexpr int64_t SPLIT_DIM_A = 1;
static constexpr int64_t SPLIT_DIM_S = 2;
static constexpr int64_t SPLIT_DIM_B = 3;

static constexpr int64_t UB_RESERVED_SIZE = 512;
static constexpr int64_t BLOCK_SPLIT_THRESHOLD = 8192;
static constexpr int64_t DIM_B = 0;
static constexpr int64_t DIM_S = 1;
static constexpr int64_t DIM_A = 2;
const uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;
static constexpr int64_t TEMPLATE_MODE = 1;
static constexpr int64_t TYPE_BSA = 2;

bool ReverseSequenceBSATiling::IsCapable()
{
    if (inputData_.comBineType != TYPE_BSA) {
        return false;
    }

    InitializationVars();
    if (inputData_.inputDim[DIM_S] * inputData_.inputDim[DIM_A] < GATHER_THRESHOLD) {
        // SA 较小的离散场景不处理
        return false;
    }    
    return true;
}

void ReverseSequenceBSATiling::InitializationVars()
{
    oneBlockNum_ = Ops::Base::GetUbBlockSize(context_) / inputData_.xDtypeSize;
    availableUb_ = static_cast<int64_t>(ubSize_  - UB_RESERVED_SIZE) / inputData_.xDtypeSize;
}

void ReverseSequenceBSATiling::DoBlockTiling()
{
    int64_t totalLoop = bLoop_ * sLoop_ * aLoop_;
    blockFactor_ = totalLoop / static_cast<int64_t>(coreNum_);
    blockTail_ = totalLoop - blockFactor_ * static_cast<int64_t>(coreNum_);
    usedCoreNum_ = blockFactor_ == 0 ? blockTail_ : static_cast<int64_t>(coreNum_);

    if (splitMode_ == SPLIT_DIM_A) {
        inUbSize_ = ubFactorB_ * ubFactorS_ * Ops::Base::CeilAlign(ubFactorA_, oneBlockNum_);
    } else if (splitMode_ == SPLIT_DIM_S) {
        if (gatherMode_ == GATHER_DIM_S) {
            inUbSize_ = ubFactorB_ * Ops::Base::CeilAlign(ubFactorS_ * ubFactorA_, oneBlockNum_);
            gatherUbSize_ = inUbSize_;
        } else {
            inUbSize_ = ubFactorB_ * ubFactorS_ * Ops::Base::CeilAlign(ubFactorA_, oneBlockNum_);
        }
    } else {
        if (gatherMode_ == GATHER_DIM_B) {
            inUbSize_ = ubFactorB_ * Ops::Base::CeilAlign(ubFactorS_ * ubFactorA_, oneBlockNum_);
            gatherUbSize_ = Ops::Base::CeilAlign(ubFactorS_ * ubFactorA_, oneBlockNum_);
        } else {
            inUbSize_ = ubFactorB_ * ubFactorS_ * Ops::Base::CeilAlign(ubFactorA_, oneBlockNum_);
        }
    }
}

int64_t ReverseSequenceBSATiling::CalcBufferSize(int64_t inB, int64_t inS, int64_t inA, bool isSplitB)
{
    int64_t tmpInDataBufferSize = inB * inS * Ops::Base::CeilAlign(inA, oneBlockNum_);
    if (inputData_.inputDim[DIM_A] * inputData_.xDtypeSize < GATHER_THRESHOLD) {
        tmpInDataBufferSize = inB * Ops::Base::CeilAlign(inS * inA, oneBlockNum_);
    }
    int64_t tmpOutDataBufferSize = tmpInDataBufferSize;

    int64_t tmpTotalBufferSize = (tmpInDataBufferSize + tmpOutDataBufferSize) * DOUBLE_BUFFER;
    if (inputData_.inputDim[DIM_A] * inputData_.xDtypeSize < GATHER_THRESHOLD) {
        int64_t gatherSize = 0;
        if (isSplitB) {
            gatherSize = Ops::Base::CeilAlign(inputData_.inputDim[DIM_A] * inputData_.inputDim[DIM_S], oneBlockNum_);
        } else {
            gatherSize = tmpInDataBufferSize;
        }
        if (inputData_.xDtypeSize == 1) {
            gatherSize += gatherSize;
        }
        tmpTotalBufferSize += gatherSize;
    }
    return tmpTotalBufferSize;
}

void ReverseSequenceBSATiling::CalcSplitDimS()
{
    int64_t inDimSLower = 1;
    int64_t inDimSUpper = inputData_.inputDim[DIM_S];
    while (inDimSLower < inDimSUpper) {
        int64_t inDimSMid = (inDimSLower + inDimSUpper + 1) / DIGIT_TWO;
        int64_t midBuffer = CalcBufferSize(1, inDimSMid, inputData_.inputDim[DIM_A]);
        if (midBuffer <= availableUb_) {
            inDimSLower = inDimSMid;
        } else {
            inDimSUpper = inDimSMid - 1;
        }
    }
    ubFactorA_ = inputData_.inputDim[DIM_A];
    ubFactorS_ = inDimSLower;
    int64_t inputBufferSize = Ops::Base::CeilAlign(ubFactorS_ * ubFactorA_, oneBlockNum_);
    if (inputBufferSize > MAX_INPUT_ELEMENTS) {
        ubFactorS_ = MAX_INPUT_ELEMENTS / inputData_.inputDim[DIM_A];
    }
    if (ubFactorS_ <= 0) {
        OP_LOGE(context_, "ReverseSequence ubFactorS_ is %ld.", ubFactorS_);
        return;
    }
    ubFactorB_ = 1;
    bLoop_ = inputData_.inputDim[DIM_B];
    sLoop_ = (inputData_.inputDim[DIM_S] + ubFactorS_ - 1) / ubFactorS_;
    aLoop_ = 1;
    splitMode_ = SPLIT_DIM_S;
}

void ReverseSequenceBSATiling::CalcSplitDimA()
{
    int64_t inDimA = std::min(inputData_.inputDim[DIM_A], (availableUb_ / DIGIT_FOUR));
    ubFactorA_ = inDimA;

    if (ubFactorA_ > MAX_INPUT_ELEMENTS) {
        ubFactorA_ = MAX_INPUT_ELEMENTS;
    }
    if (ubFactorA_ <= 0) {
        OP_LOGE(context_, "ReverseSequence ubFactorA_ is %ld.", ubFactorA_);
        return;
    }
    ubFactorB_ = 1;
    ubFactorS_ = 1;
    bLoop_ = inputData_.inputDim[DIM_B];
    sLoop_ = inputData_.inputDim[DIM_S];
    aLoop_ = (inputData_.inputDim[DIM_A] + ubFactorA_ - 1) / ubFactorA_;
    splitMode_ = SPLIT_DIM_A;
}

void ReverseSequenceBSATiling::CalcSplitDimB()
{    
    int64_t inDimBLower = 1;
    int64_t inDimBUpper = inputData_.inputDim[DIM_B];
    while (inDimBLower < inDimBUpper) {
        int64_t inDimBMid = (inDimBLower + inDimBUpper + 1) / DIGIT_TWO;
        // split_b 时单次gather一个Batch
        int64_t midBuffer = CalcBufferSize(inDimBMid, inputData_.inputDim[DIM_S], inputData_.inputDim[DIM_A], true);
        if (midBuffer <= availableUb_) {
            inDimBLower = inDimBMid;
        } else {
            inDimBUpper = inDimBMid - 1;
        }
    }
    ubFactorS_ = inputData_.inputDim[DIM_S];
    ubFactorA_ = inputData_.inputDim[DIM_A];
    ubFactorB_ = inDimBLower;
    int64_t inputBufferSize = ubFactorB_ * inputData_.inputDim[DIM_A] * inputData_.inputDim[DIM_S];
    if (inputBufferSize > MAX_INPUT_ELEMENTS) {
        ubFactorB_ = MAX_INPUT_ELEMENTS / (inputData_.inputDim[DIM_A] * inputData_.inputDim[DIM_S]);
    }
    bLoop_ = (inputData_.inputDim[DIM_B] + ubFactorB_ - 1) / ubFactorB_;
    sLoop_ = 1;
    aLoop_ = 1;
    splitMode_ = SPLIT_DIM_B;
}

void ReverseSequenceBSATiling::CalcGatherMode()    
{
    if (inputData_.inputDim[DIM_A] * inputData_.xDtypeSize <= GATHER_THRESHOLD &&
        splitMode_ == SPLIT_DIM_B) {
        gatherMode_ = GATHER_DIM_B;
    }  else if (inputData_.inputDim[DIM_A] * inputData_.xDtypeSize <= GATHER_THRESHOLD && (splitMode_ == SPLIT_DIM_S)) {
        gatherMode_ = GATHER_DIM_S;
    } else {
        gatherMode_ = NOT_GATHER;
    }
}


void ReverseSequenceBSATiling::DoUBTilingSingle()
{
    int64_t oneBatchBuffer = CalcBufferSize(1, inputData_.inputDim[DIM_S], inputData_.inputDim[DIM_A]);
    int64_t oneSeqBuffer = CalcBufferSize(1, 1, inputData_.inputDim[DIM_A]);
    if (oneBatchBuffer <= 0 || oneSeqBuffer <= 0) {
        bLoop_ = 0;
        sLoop_ = 0;
        aLoop_ = 0;
        isZero_ = true;
        return;
    }
    // SA 全载
    if (oneBatchBuffer <= availableUb_ && oneBatchBuffer <= MAX_INPUT_ELEMENTS ) {
        CalcSplitDimB();
        return;
    }
    // A 全载
    if (oneSeqBuffer <= availableUb_ && inputData_.inputDim[DIM_A] <= MAX_INPUT_ELEMENTS) {
        CalcSplitDimS();
        return;
    }
    // A 不全载
    CalcSplitDimA();
}

void ReverseSequenceBSATiling::DoUBTiling()
{
    int64_t ubStep = BLOCK_SPLIT_THRESHOLD / inputData_.xDtypeSize;
    do {
        DoUBTilingSingle();
        if (bLoop_ * sLoop_ * aLoop_ >= static_cast<int64_t>(coreNum_) || isZero_) {
            break;
        }
        availableUb_ -= ubStep;
    } while (availableUb_ > ubStep);
}

ge::graphStatus ReverseSequenceBSATiling::DoOpTiling()
{
    DoUBTiling();
    CalcGatherMode();
    DoBlockTiling();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReverseSequenceBSATiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}


uint64_t ReverseSequenceBSATiling::GetTilingKey() const
{
    OP_LOGD("ReverseSequenceBSATiling::GetTilingKey begin");
    const uint64_t tilingKey = GET_TPL_TILING_KEY(TEMPLATE_MODE, inputData_.xDtypeSize, 1);
    OP_LOGD(context_->GetNodeName(), "tilingKey is: [%lu]", tilingKey);
    printf("tiling key is :%ld", tilingKey);
    return tilingKey;
}

ge::graphStatus ReverseSequenceBSATiling::GetWorkspaceSize()
{
    uint32_t sysWorkspace = WS_SYS_SIZE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sysWorkspace;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReverseSequenceBSATiling::PostTiling()
{
    context_->SetBlockDim(coreNum_);
    return ge::GRAPH_SUCCESS;
}

void ReverseSequenceBSATiling::SetTilingData()
{
     ReverseSequence::ReverseSequenceBSATilingData* tilingData =
        context_->GetTilingData<ReverseSequence::ReverseSequenceBSATilingData>();

    tilingData->bDim = inputData_.inputDim[DIM_B];
    tilingData->sDim = inputData_.inputDim[DIM_S];
    tilingData->aDim = inputData_.inputDim[DIM_A];
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
    tilingData->gatherMode = gatherMode_;
    tilingData->splitMode = splitMode_;
    tilingData->gatherUbSize = gatherUbSize_;
    tilingData->dtypeSize = inputData_.xDtypeSize;
}


std::string ReverseSequenceBSATiling::TilingDataToString()
{
    ReverseSequence::ReverseSequenceBSATilingData* tilingData =
        context_->GetTilingData<ReverseSequence::ReverseSequenceBSATilingData>();
    std::string str = " bDim:" + std::to_string(tilingData->bDim);
    str += " sDim:" + std::to_string(tilingData->sDim);
    str += " aDim:" + std::to_string(tilingData->aDim);
    str += " blockFactor:" + std::to_string(tilingData->blockFactor);
    str += " blockTail:" + std::to_string(tilingData->blockTail);
    str += " ubFactorB:" + std::to_string(tilingData->ubFactorB);
    str += " ubFactorS:" + std::to_string(tilingData->ubFactorS);
    str += " ubFactorA:" + std::to_string(tilingData->ubFactorA);
    str += " bLoop:" + std::to_string(tilingData->bLoop);
    str += " sLoop:" + std::to_string(tilingData->sLoop);
    str += " aLoop:" + std::to_string(tilingData->aLoop);
    str += " inUbSize:" + std::to_string(tilingData->inUbSize);
    str += " usedCoreNum:" + std::to_string(tilingData->usedCoreNum);
    str += " gatherMode:" + std::to_string(tilingData->gatherMode);
    str += " splitMode:" + std::to_string(tilingData->splitMode);
    str += " gatherUbSize:" + std::to_string(tilingData->gatherUbSize);
    str += " dtypeSize:" + std::to_string(tilingData->dtypeSize);
    return str;
}


void ReverseSequenceBSATiling::DumpTilingInfo()
{
    OP_LOGI(context_, "ReverseSequenceBSA tilingInfo is :%s", TilingDataToString().c_str());
}

ge::graphStatus ReverseSequenceBSATiling::GetPlatformInfo()
{
    return GetReverseSequencePlatformInfo(context_, ubSize_, coreNum_);
}

ge::graphStatus ReverseSequenceBSATiling::GetShapeAttrsInfo()
{
    return GetReverseSequenceShapeAttrsInfo(context_, inputData_);
}

REGISTER_TILING_TEMPLATE("ReverseSequence", ReverseSequenceBSATiling, 1);

}  // namespace optiling