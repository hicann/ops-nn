/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_v2_ar_recompute_tiling.cpp
 * \brief
 */

#include <algorithm>
#include "op_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "softmax_v2_tiling.h"
#include "util/math_util.h"

using namespace ge;

using namespace Ops::Base;
namespace optiling
{
constexpr int64_t AR_RECOMPUTE_MAX_BUFFER_BTYES = 32;
constexpr int64_t AR_RECOMPUTE_SUM_BUFFER_BTYES = 32;
constexpr int64_t AR_RECOMPUTE_BINARY_TMP_BTYES = 512;
constexpr int64_t AR_RECOMPUTE_BINARY_CACHE_BTYES = 2048;   // sizeof(float) * 8 * 64
constexpr int64_t TRIPLE_BUFFER = 3;
constexpr int64_t BIT_COUNT_64 = 64;

bool SoftmaxV2TilingARRecompute::IsCapable()
{
    // a0_为1的场景走AR模板
    OP_CHECK_IF(a0_ != DIM_NUM_ONE,
                    OP_LOGI(context_->GetNodeName(),
                            "AR recompute template is not capable. axes is: %ld, merged shape is (%ld, %ld, %ld).",
                            reduceAxes_, a1_, r_, a0_),
                    return false);
    return true;
}

int64_t SoftmaxV2TilingARRecompute::FindNearestPower2(const int64_t value)
{
    if (value <= CONST_ONE) {
        return CONST_ZERO;
    } else if (value <= CONST_TWO) {
        return CONST_ONE;
    } else if (value <= CONST_FOUR) {
        return CONST_TWO;
    } else {
        const int64_t num = value - CONST_ONE;
        const int64_t pow = CONST_SIXTY_THREE - __builtin_clzl(num);
        return (CONST_ONE << pow);
    }
}

ge::graphStatus SoftmaxV2TilingARRecompute::BinarySummationTiling()
{
    int64_t basicBlockLoop = FindNearestPower2(aLoopCountCeil_);
    int64_t mainFoldCount = aLoopCountFloor_ - basicBlockLoop;

    tilingData_.set_basicBlockLoop(basicBlockLoop);
    tilingData_.set_mainFoldCount(mainFoldCount);
    OP_LOGI(context_->GetNodeName(), "BinarySummation Tiling Finished. basicBlockLoop: %ld, mainFoldCount: %ld",
            basicBlockLoop, mainFoldCount);
    return ge::GRAPH_SUCCESS;
}

int64_t SoftmaxV2TilingARRecompute::Gcd(int64_t a, int64_t b)
{
    a = std::abs(a);
    b = std::abs(b);

    while (b != 0) {
        int64_t temp = b;
        b = a % b;
        a = temp;
    }

    return a;
}

/*
 *  求最大公约数
 */
int64_t SoftmaxV2TilingARRecompute::Lcm(const int64_t a, const int64_t b)
{
    int64_t gcd = Gcd(a, b);
    return (a / gcd) * b;
}

ge::graphStatus SoftmaxV2TilingARRecompute::DoOpTiling()
{
    // 检查ub空间是否足够
    ubFlexible_ = aicoreParams_.ubSize - AR_RECOMPUTE_MAX_BUFFER_BTYES - AR_RECOMPUTE_SUM_BUFFER_BTYES -
                  AR_RECOMPUTE_BINARY_TMP_BTYES - AR_RECOMPUTE_BINARY_CACHE_BTYES;
    baseFactor_ = xDtypeSize_ * TRIPLE_BUFFER + yDtypeSize_ * DOUBLE_BUFFER + sizeof(float) * DOUBLE_BUFFER;

    OP_CHECK_IF((baseFactor_ > ubFlexible_),
                    OP_LOGI(context_->GetNodeName(),
                            "AR recompute template is not capable. original shape is:(%s), axes is: %ld, merged "
                            "shape is (%ld, %ld, %ld), ub size: %ldB",
                            VectorToString(xShape_).c_str(), reduceAxes_, a1_, r_, a0_, aicoreParams_.ubSize),
                    return ge::GRAPH_PARAM_INVALID);

    OP_LOGI(context_->GetNodeName(),
            "AR recompute template is capable. original shape is:(%s), axes is: %ld, merged shape is "
            "(%ld, %ld, %ld).",
            VectorToString(xShape_).c_str(), reduceAxes_, a1_, r_, a0_);

    // 1、切a：按核均分
    int64_t coreNum = static_cast<int64_t>(aicoreParams_.blockDim);
    int64_t aPerCore = CeilDiv(a1_, coreNum);
    usedCoreNums_ = CeilDiv(a1_, aPerCore);

    // 2、切r：保证X、Y、Tmp的长度相等、32B对齐，且尽可能大
    int64_t ubFactorTmp = ubFlexible_ / baseFactor_;
    int64_t lcm = Lcm(blockSize_, Lcm(blockSize_ / xDtypeSize_, blockSize_ / yDtypeSize_));
    int64_t ubFactor = std::min(r_, (ubFactorTmp / lcm) * lcm);
    aLoopCountCeil_ = CeilDiv(r_, ubFactor);
    aLoopCountFloor_ = FloorDiv(r_, ubFactor);
    int64_t ubFactorTail = r_ - ubFactor * aLoopCountFloor_;

    tilingData_.set_a(a1_);
    tilingData_.set_r(r_);
    tilingData_.set_ubFactor(ubFactor);
    tilingData_.set_ubFactorTail(ubFactorTail);
    tilingData_.set_aBlockFactor(aPerCore);
    tilingData_.set_aLoopCountCeil(aLoopCountCeil_);
    return BinarySummationTiling();
}

uint64_t SoftmaxV2TilingARRecompute::GetTilingKey() const
{
    return TILINGKEY_AR_RECOMPUTE;
}

ge::graphStatus SoftmaxV2TilingARRecompute::PostTiling()
{
    context_->SetBlockDim(usedCoreNums_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(SoftmaxV2, SoftmaxV2TilingARRecompute, TEMPLATE_AR_RECOMPUTE_PRIORITY);
}  // namespace optiling