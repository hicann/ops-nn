/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file instance_norm_tiling_ar_full_reduce_arch35.cpp
 * \brief
 */
#include <vector>
#include <algorithm>
#include "instance_norm_tiling.h"

using namespace ge;

namespace optiling {
constexpr int64_t TILINGKEY_AR_FULL_REDUCE = 200000;
constexpr int64_t RESERVER_FOR_ALIGN = 512;
constexpr int64_t SMALL_BUFFER_NUM_T = 2;
constexpr int64_t SMALL_BUFFER_NUM_FP32 = 6;
constexpr int64_t LARGE_BUFFER_NUM = 2;
constexpr int64_t DOUBLE_BUFFER = 2;

constexpr int64_t AR_BINARY_ADD_THRESHOLD = 8;
constexpr int64_t BINARY_ADD_COEF = 2;
constexpr int64_t BINARY_ADD_COEF_FOUR = 4;
constexpr int64_t FP32_BYTE = 4;
constexpr int64_t FP16_BYTE = 2;
constexpr uint32_t ULONG_BIT_LEN = 64;

constexpr uint32_t NUM_2 = 2;
constexpr int64_t R_MAX_VALUE = 16384;

bool InstanceNormARFullReduceTiling::IsCapable()
{
    // ARA当a0=1的特例纳入此场景
    if (format != FORMAT_NCHW && format != FORMAT_NCDHW && format != FORMAT_ND) {
        if (a0 != 1) {
            return false;
        }
    }
    OP_CHECK_IF(r > R_MAX_VALUE,
        OP_LOGI(context_->GetNodeName(), "AR full load template is not capable. actual r is %ld, larger than %ld", r, R_MAX_VALUE),
        return false);
    uint64_t ubfp32 = ubBlockSize / sizeof(float);
    // 数据类型的元素宽度
    int64_t elemSize = FP32_BYTE;
    if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
        elemSize = FP16_BYTE;
    }
    int64_t gammaElemSize = FP32_BYTE;
    if (gammaDataType == ge::DT_FLOAT16 || gammaDataType == ge::DT_BF16) {
        gammaElemSize = FP16_BYTE;
    }
    int64_t meanElemSize = FP32_BYTE;
    if (meanDataType == ge::DT_FLOAT16 || meanDataType == ge::DT_BF16) {
        meanElemSize = FP16_BYTE;
    }

    uint64_t rAlign = Ops::Base::CeilAlign(r * elemSize, ubBlockSize) / elemSize;
    // 计算 二分累加 分界点
    uint64_t binAddQuotient = rAlign == 0 ? 1 : (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(rAlign)));
    binAddQuotient = (binAddQuotient == rAlign) ? binAddQuotient / NUM_2 : binAddQuotient;
    uint64_t binAddBufferOneline = Ops::Base::CeilAlign((binAddQuotient + vlfp32 - 1) / vlfp32, ubfp32);

    // 输入输出：
    // inQueueX_:       rAlign_ * cInner_ * sizeof(T_X)         # rAlign_ * sizeof(T_X) 已经32B对齐
    // inQueueGamma_:   cInner_ * sizeof(T_BETA)       # 32B 对齐 
    // inQueueBeta_:    cInner_ * sizeof(T_BETA)       # 32B 对齐
    // outQueueY_:      rAlign_ * cInner_ * sizeof(T_X)         # rAlign_ * sizeof(T_X) 已经32B对齐
    // outQueueMean_:   cInner_ * sizeof(T_MEAN)        # 32B 对齐
    // outQueueVariance_:   cInner_ * sizeof(T_MEAN)    # 32B 对齐
    // 缓存:
    // rstdBuf_:        (cInner_ + VL_FP32 - 1) / VL_FP32 * sizeof(float)  # VlSize 对齐
    // meanFp32Buff_:   cInner_ * sizeof(float)   # 32B 对齐
    // binaryAddBuf_:   cInner_ * binAddBufferOneline
    // 计算可全载的行数
    uint64_t cInner = (aicoreParams_.ubSize - RESERVER_FOR_ALIGN) /
        ((rAlign * elemSize * NUM_2 + gammaElemSize * NUM_2 + meanElemSize * NUM_2) * NUM_2 + sizeof(float) +
         binAddBufferOneline);
    if (cInner < 1) {
        return false;
    }
    // 计算可以全载的行数
    cInner = std::min(cInner, static_cast<uint64_t>(a0));
    uint64_t cOuter = (a0 + cInner - 1) / cInner;
    uint64_t cTail = a0 - (cOuter - 1) * cInner;
    uint64_t totalTileCnt = cOuter * a1;
    uint64_t perCoreCnt = Ops::Base::CeilDiv(totalTileCnt, aicoreParams_.blockDim);
    blockNum_ = Ops::Base::CeilDiv(totalTileCnt, perCoreCnt);

    td_.numN  = a1;
    td_.numC  = a0;
    td_.numR  = r;
    td_.rAlign  = rAlign;
    td_.cInner  = cInner;
    td_.cOuter  = cOuter;
    td_.cTail  = cTail;
    td_.binaryAddQuotient  = binAddQuotient;
    td_.perCoreCnt  = perCoreCnt;
    td_.epsilon  = epsilon;
    td_.avgFactor  = 1.0 / r;
    return true;
}


ge::graphStatus InstanceNormARFullReduceTiling::DoOpTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t InstanceNormARFullReduceTiling::GetTilingKey() const
{
    return TILINGKEY_AR_FULL_REDUCE;
}

ge::graphStatus InstanceNormARFullReduceTiling::PostTiling()
{
    context_->SetBlockDim(blockNum_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(
        sizeof(td_) > rawTilingData->GetCapacity(),
        OP_LOGE(
            context_->GetNodeName(), "actual tiling data size %zu > context tiling data size %zu",
            sizeof(td_), rawTilingData->GetCapacity()),
        return ge::GRAPH_FAILED);
    auto capSize = rawTilingData->GetCapacity();
    void* ptrData = rawTilingData->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrData);
    void* ptrStruct = static_cast<void*>(&td_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrStruct);
    OP_CHECK_IF(
        memcpy_s(ptrData, capSize, ptrStruct, sizeof(td_)) != 0,
        OP_LOGE(context_->GetNodeName(), "Set tiling data is failed!"), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(td_));
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(InstanceNorm, InstanceNormARFullReduceTiling, IN_AR_FULL_REDUCE_PRIORITY);
} // namespace optiling


