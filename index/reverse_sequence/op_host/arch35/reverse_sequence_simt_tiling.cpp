/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reverse_sequence_simt_tiling.cpp
 * \brief
 */

#include "reverse_sequence_simt_tiling.h"
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
static constexpr uint32_t WS_SYS_SIZE = 16 * 1024 * 1024;
static constexpr uint32_t RESERVE_BUF_SIZE = 256; //UintDiv buf
static constexpr int64_t MIN_UB_SIZE = 128;
static constexpr int64_t TEMPLATE_MODE = 9;
static constexpr int64_t DCACHE_SIZE = 131072;  // 128k 

template <typename T>
static std::string ToString(const T* value, size_t size) {
  std::string r = "[";
  for (size_t i = 0; i < size; i++) {
    r = r + std::to_string(value[i]) + ", ";
  }
  r = r + "]";
  return r;
}

bool ReverseSequenceSimtTiling::IsCapable()
{
    return true;
}

ge::graphStatus ReverseSequenceSimtTiling::DoReverseSequenceSimtTiling() 
{
    OP_LOGD("ReverseSequenceSimtTiling::DoReverseSequenceSimtTiling begin");
    perCoreHandleNums_ = Ops::Base::CeilDiv(inputData.xShapeSize, static_cast<int64_t>(coreNum));
    perCoreHandleNums_ = std::max(perCoreHandleNums_, MIN_UB_SIZE);
    if (inputData.xShapeSize < MIN_UB_SIZE) {
        perCoreHandleNums_ = inputData.xShapeSize;
    }
    usedCoreNums_ = Ops::Base::CeilDiv(inputData.xShapeSize, perCoreHandleNums_);
    tailCoreHandleNums_ = inputData.xShapeSize - (usedCoreNums_ - 1) * perCoreHandleNums_;

    int64_t oneBlockNum = Ops::Base::GetUbBlockSize(context_) / inputData.xDtypeSize;
    OP_CHECK_IF((ubSize <= DCACHE_SIZE),
                    OP_LOGE(context_->GetNodeName(), "ub size:%lu less than Dcache Size:128k", ubSize),
                    return ge::GRAPH_FAILED);
    ubSize = ubSize - DCACHE_SIZE;
    ubSize = ubSize - RESERVE_BUF_SIZE;
    int64_t halfUbSize = static_cast<int64_t>(ubSize) / DOUBLE_BUFFER;
    int64_t ubFactor = halfUbSize / inputData.xDtypeSize;
    int64_t ubFactorAlign = Ops::Base::FloorAlign(ubFactor, oneBlockNum);
    xUbFactor_ = ubFactorAlign;
    xUbLoop_ = Ops::Base::CeilDiv(perCoreHandleNums_, ubFactorAlign);
    xTailUbLoopSize_ = perCoreHandleNums_ - (xUbLoop_ - 1) * ubFactorAlign;
    xTailCoreLoop_ = Ops::Base::CeilDiv(tailCoreHandleNums_, ubFactorAlign);
    xTailCoreTailLoopSize_ = tailCoreHandleNums_ - (xTailCoreLoop_ - 1) * ubFactorAlign;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReverseSequenceSimtTiling::DoOpTiling()
{
    OP_LOGD("ReverseSequenceSimtTiling::DoOpTiling begin");
    DoReverseSequenceSimtTiling();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReverseSequenceSimtTiling::DoLibApiTiling()
{
    OP_LOGD("ReverseSequenceSimtTiling::DoLibApiTiling begin");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ReverseSequenceSimtTiling::GetWorkspaceSize()
{
    OP_LOGD("ReverseSequenceSimtTiling::GetWorkspaceSize begin");
    uint32_t sysWorkspace = WS_SYS_SIZE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sysWorkspace;

    return ge::GRAPH_SUCCESS;
}

uint64_t ReverseSequenceSimtTiling::GetTilingKey() const 
{
    OP_LOGD("ReverseSequenceSimtTiling::GetTilingKey begin");
    const uint64_t tilingKey = GET_TPL_TILING_KEY(TEMPLATE_MODE, inputData.xDtypeSize, addrRange_);
    OP_LOGD(context_->GetNodeName(), "tilingKey is: [%lu]", tilingKey);
    return tilingKey;
}

ge::graphStatus ReverseSequenceSimtTiling::PostTiling()
{
    OP_LOGD("ReverseSequenceSimtTiling::PostTiling begin");
    context_->SetBlockDim(usedCoreNums_);
    context_->SetScheduleMode(1);
    if (inputData.xShapeSize >= INT32_MAX) {
        addrRange_ = 1;
    }
    context_->SetTilingKey(GetTilingKey());
    auto res = context_->SetLocalMemorySize(ubSize);
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS),
        OP_LOGE(context_->GetNodeName(), "SetLocalMemorySize ubSize = %lu failed.", ubSize), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void ReverseSequenceSimtTiling::SetTilingData() const
{
    OP_LOGD("ReverseSequenceSimtTiling::SetTilingData begin");
    OP_LOGD(context_, "ReverseSequenceSimtTiling::SetTilingData inputData.batchDim=%ld", inputData.batchDim);
    ReverseSequenceSimtTilingData4RegBase* tilingData =
        context_->GetTilingData<ReverseSequenceSimtTilingData4RegBase>();
    tilingData->batchDim = inputData.batchDim;
    tilingData->seqDim = inputData.seqDim;
    tilingData->reverseSize = inputData.reverseSize;
    tilingData->batchSize = inputData.batchSize;
    tilingData->perCoreHandleNums = perCoreHandleNums_;
    tilingData->usedCoreNums = usedCoreNums_;
    tilingData->xUbFactor = xUbFactor_;
    tilingData->xUbLoop = xUbLoop_;
    tilingData->xTailUbLoopSize = xTailUbLoopSize_;
    tilingData->xTailCoreLoop = xTailCoreLoop_;
    tilingData->xTailCoreTailLoopSize = xTailCoreTailLoopSize_;
    OP_LOGD(context_, "ReverseSequenceSimtTiling::SetTilingData tilingData->batchDim=%ld", tilingData->batchDim);
}

void ReverseSequenceSimtTiling::DumpTilingInfo()
{
    ReverseSequenceSimtTilingData4RegBase* tilingData =
        context_->GetTilingData<ReverseSequenceSimtTilingData4RegBase>();
    std::string str;
    str += " batchDim:" + std::to_string(tilingData->batchDim);
    str += " seqDim:" + std::to_string(tilingData->seqDim);
    str += " reverseSize:" + std::to_string(tilingData->reverseSize);
    str += " batchSize:" + std::to_string(tilingData->batchSize);
    str += "perCoreHandleNums:" + std::to_string(tilingData->perCoreHandleNums);
    str += " usedCoreNums:" + std::to_string(tilingData->usedCoreNums);
    str += " xUbFactor:" + std::to_string(tilingData->xUbFactor);
    str += " xUbLoop:" + std::to_string(tilingData->xUbLoop);
    str += " xTailUbLoopSize:" + std::to_string(tilingData->xTailUbLoopSize);
    str += " xTailCoreLoop:" + std::to_string(tilingData->xTailCoreLoop);
    str += " xTailCoreTailLoopSize:" + std::to_string(tilingData->xTailCoreTailLoopSize);
    str += " inputDims:" + ToString(inputData.inputDim, allDims);
    OP_LOGI(context_, "%s", str.c_str());
}

//////////////////////////////// ReverseSequenceSimtTiling /////////////////////////////////
ge::graphStatus ReverseSequenceSimtTiling::GetPlatformInfo()
{
    return GetReverseSequencePlatformInfo(context_, ubSize, coreNum);
}

ge::graphStatus ReverseSequenceSimtTiling::GetShapeAttrsInfo()
{
    return GetReverseSequenceShapeAttrsInfo(context_, inputData);
}

REGISTER_TILING_TEMPLATE("ReverseSequence", ReverseSequenceSimtTiling, 9);

}  // namespace optiling