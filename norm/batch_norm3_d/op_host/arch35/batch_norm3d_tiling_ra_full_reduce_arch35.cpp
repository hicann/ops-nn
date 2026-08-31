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
 * \file batch_norm3d_tiling_ra_full_reduce_arch35.cpp
 * \brief
 */
#include "batch_norm3d_tiling.h"

using namespace ge;

namespace {
constexpr int64_t TILINGKEY_RA_FULL_REDUCE = 400000;

constexpr int64_t SMALL_BUFFER_NUM = 8;
constexpr int64_t SMALL_BUFFER_NUM_FP32 = 8;
constexpr int64_t SMALL_BUFFER_NUM_T = 0;
constexpr int64_t LARGE_BUFFER_NUM = 2;

constexpr int64_t BINARY_ADD_COEF = 2;
constexpr int64_t BINARY_ADD_COEF_FOUR = 4;
constexpr int64_t RA_BINARY_ADD_THRESHOLD = 8;
constexpr int64_t CHANGE_TO_WELFORD_THRESHOLD = 64;

} // namespace

namespace optiling {

class BatchNorm3DRAFullReduceTilingBase : public BatchNorm3DRegbaseTilingBase {
public:
    explicit BatchNorm3DRAFullReduceTilingBase(gert::TilingContext* context) : BatchNorm3DRegbaseTilingBase(context) {}
    ~BatchNorm3DRAFullReduceTilingBase() override = default;

    void Reset(gert::TilingContext* context) override
    {
        BatchNorm3DRegbaseTilingBase::Reset(context);
        binaryAddQuotient = 0;
    }

protected:
    bool IsCapable() override
    {
        // 只支持ra 场景
        if (r0_ != CONST_ONE) {
            return false;
        }
        int64_t elemSize = FLOAT32_BYTES;
        if (xDtype_ == ge::DT_FLOAT16) {
            elemSize = FLOAT16_BYTES;
        }

        int64_t ubCanUseSize = (((aicoreParams_.ubSize / DOUBLE_BUFFER) / blockSize_) * blockSize_);
        int64_t ubSizePerA = (LARGE_BUFFER_NUM * r1_ + 1) * elemSize + SMALL_BUFFER_NUM_T * elemSize +
                             SMALL_BUFFER_NUM_FP32 * FLOAT32_BYTES;
        if (xDtype_ == ge::DT_FLOAT16) {
            ubSizePerA = LARGE_BUFFER_NUM * r1_ * elemSize + (r1_ + 1) * FLOAT32_BYTES + SMALL_BUFFER_NUM_T * elemSize +
                         SMALL_BUFFER_NUM_FP32 * FLOAT32_BYTES;
        }
        int64_t aFactor = ubCanUseSize / ubSizePerA;
        int64_t aFactorAlign = (((aFactor * elemSize) / blockSize_) * blockSize_) / elemSize;
        if (aFactorAlign >= 1) {
            batchNormTilingData.aFactor = aFactorAlign;
            return true;
        }
        return false;
    }
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;
    ge::graphStatus BinaryAddTiling();
    bool IsNeedChangeToWelford(int64_t elemSize);

private:
    int64_t binaryAddQuotient;
    BatchNorm3DRAFullReduceTilingData batchNormTilingData;
};

bool BatchNorm3DRAFullReduceTilingBase::IsNeedChangeToWelford(int64_t elemSize)
{
    int64_t blockFactor = batchNormTilingData.aBlockFactor;
    int64_t ubFactor = batchNormTilingData.aFactor;
    int64_t blockFactorSize = blockFactor * elemSize;
    // 核内last轴大于64B时，如果ub内可以放下全部last轴或者计算带宽可以用满，那么无需切到welford模版
    return ((blockFactorSize >= CHANGE_TO_WELFORD_THRESHOLD) &&
            (ubFactor < std::min(static_cast<int64_t>(vlFp32_), blockFactor)));
}

ge::graphStatus BatchNorm3DRAFullReduceTilingBase::BinaryAddTiling()
{
    int64_t binaryQuotient = RA_BINARY_ADD_THRESHOLD;
    while (binaryQuotient < r1_) {
        binaryQuotient *= BINARY_ADD_COEF;
    }
    binaryQuotient /= BINARY_ADD_COEF;
    batchNormTilingData.binaryAddQuotient = binaryQuotient;
    int64_t binaryAddNum = binaryQuotient / RA_BINARY_ADD_THRESHOLD;
    int64_t binaryAddK = 0;
    int64_t curBinaryAddNum = 1;
    while (curBinaryAddNum < binaryAddNum) {
        binaryAddK++;
        curBinaryAddNum *= BINARY_ADD_COEF_FOUR;
    }
    if (curBinaryAddNum == binaryAddNum) {
        batchNormTilingData.binaryAddK = binaryAddK;
        batchNormTilingData.binaryAddLast = 0;
    } else if (curBinaryAddNum == binaryAddNum * BINARY_ADD_COEF) {
        batchNormTilingData.binaryAddK = binaryAddK - 1;
        batchNormTilingData.binaryAddLast = 1;
    } else {
        OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "Binary add calculate error.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNorm3DRAFullReduceTilingBase::DoOpTiling()
{
    // dim
    batchNormTilingData.r1 = r1_;
    batchNormTilingData.a = a_;
    int64_t powerOfTwoForR = 1;
    while (powerOfTwoForR < r1_) {
        powerOfTwoForR *= BINARY_ADD_COEF;
    }
    batchNormTilingData.powerOfTwoForR = powerOfTwoForR;

    // attr
    batchNormTilingData.epsilon = epsilon_;
    batchNormTilingData.momentum = exponentialAvgFactor_;

    // core num
    int64_t elemSize = FLOAT32_BYTES;
    if (xDtype_ == ge::DT_FLOAT16) {
        elemSize = FLOAT16_BYTES;
    }
    int64_t theLeastAPerCore = blockSize_ / elemSize;
    int64_t blockFactor = (a_ + aicoreParams_.blockDim - 1) / aicoreParams_.blockDim;
    if (blockFactor < theLeastAPerCore) {
        blockFactor = theLeastAPerCore;
    }
    usedCoreNums_ = (a_ + blockFactor - 1) / blockFactor;
    batchNormTilingData.aBlockFactor = blockFactor;
    batchNormTilingData.blockNum = usedCoreNums_;
    batchNormTilingData.useRunningMeanVar = useRunningMeanVar_;

    if (r1_ <= RA_BINARY_ADD_THRESHOLD) {
        return ge::GRAPH_SUCCESS;
    }

    if (IsNeedChangeToWelford(elemSize)) {
        OP_LOGW(context_->GetNodeName(), "Change to welford tiling.");
        return ge::GRAPH_PARAM_INVALID;
    }

    return BinaryAddTiling();
}

uint64_t BatchNorm3DRAFullReduceTilingBase::GetTilingKey() const { return TILINGKEY_RA_FULL_REDUCE; }

ge::graphStatus BatchNorm3DRAFullReduceTilingBase::PostTiling()
{
    context_->SetBlockDim(usedCoreNums_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    auto* tilingDataOut = context_->GetTilingData<BatchNorm3DRAFullReduceTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingDataOut);
    *tilingDataOut = batchNormTilingData;

    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(BatchNorm3D, BatchNorm3DRAFullReduceTilingBase, 10000);
} // namespace optiling
