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
 * \file batch_norm3d_tiling_full_reduce_arch35.cpp
 * \brief
 */
#include "batch_norm3d_tiling.h"

using namespace ge;

namespace {
constexpr int64_t TILINGKEY_FULL_REDUCE = 200000;

constexpr int64_t SMALL_BUFFER_NUM = 8;
constexpr int64_t SMALL_BUFFER_NUM_FP32 = 8;
constexpr int64_t SMALL_BUFFER_NUM_T = 0;
constexpr int64_t LARGE_BUFFER_NUM = 2;
constexpr int64_t BINARY_ADD_COEF = 2;

} // namespace

namespace optiling {
class BatchNorm3DFullReduceTilingBase : public BatchNorm3DRegbaseTilingBase {
public:
    explicit BatchNorm3DFullReduceTilingBase(gert::TilingContext* context) : BatchNorm3DRegbaseTilingBase(context) {}
    ~BatchNorm3DFullReduceTilingBase() override = default;

    void Reset(gert::TilingContext* context) override
    {
        BatchNorm3DRegbaseTilingBase::Reset(context);
        binaryAddQuotient = 0;
    }

protected:
    bool IsCapable() override
    {
        // 不按 format 过滤：基类已把五种 format 统一折算成 (r1_, a_, r0_)，
        // 本模板往下只用这三个值，同 dims 的 ND 与 NCHW 在这里不可区分。
        // NHWC / NDHWC 折算后 r0_ 恒为 1，会先被 r0_ == 1 的模板（优先级 10000/12000/15000）接走。
        int64_t elemSize = FLOAT32_BYTES;
        if (xDtype_ == ge::DT_FLOAT16 || xDtype_ == ge::DT_BF16) {
            elemSize = FLOAT16_BYTES;
        }
        int64_t r1r0 = r0_ * r1_;
        binaryAddQuotient = vlFp32_;
        while (binaryAddQuotient < r1r0) {
            binaryAddQuotient *= BINARY_ADD_COEF;
        }
        binaryAddQuotient /= BINARY_ADD_COEF;
        int64_t quotientVcaddNum = binaryAddQuotient / vlFp32_;
        int64_t quotientVcaddSizeAlign = ((quotientVcaddNum * FLOAT32_BYTES + blockSize_ - 1) / blockSize_) *
                                         blockSize_;
        if (static_cast<uint64_t>(quotientVcaddSizeAlign) >= aicoreParams_.ubSize) {
            return false;
        }
        // reserve 8 block for 8 A tensor alignment
        int64_t ubCanUseSize = ((((aicoreParams_.ubSize - quotientVcaddSizeAlign) / DOUBLE_BUFFER) / blockSize_) *
                                blockSize_);
        if (static_cast<int64_t>(SMALL_BUFFER_NUM * blockSize_) >= ubCanUseSize) {
            return false;
        }
        ubCanUseSize -= SMALL_BUFFER_NUM * blockSize_;
        int64_t r1r0Align = (((r1r0 * elemSize + blockSize_ - 1) / blockSize_) * blockSize_) / elemSize;
        // two AR tensor, two A tensor, six fp32 A tensor
        int64_t ubSizePerA = LARGE_BUFFER_NUM * r1r0Align * elemSize + SMALL_BUFFER_NUM_T * elemSize +
                             SMALL_BUFFER_NUM_FP32 * FLOAT32_BYTES;
        int64_t aFactor = ubCanUseSize / ubSizePerA;
        if (aFactor >= 1) {
            batchNormTilingData.aFactor = aFactor;
            batchNormTilingData.binaryAddQuotient = binaryAddQuotient;
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

private:
    int64_t binaryAddQuotient;
    BatchNorm3DFullReduceRegbaseTilingData batchNormTilingData;
};

ge::graphStatus BatchNorm3DFullReduceTilingBase::DoOpTiling()
{
    // dim
    batchNormTilingData.r1 = r1_;
    batchNormTilingData.a = a_;
    batchNormTilingData.r0 = r0_;
    int64_t rDim = r1_ * r0_;
    int64_t powerOfTwoForR = 1;
    while (powerOfTwoForR < rDim) {
        powerOfTwoForR *= BINARY_ADD_COEF;
    }
    batchNormTilingData.powerOfTwoForR = powerOfTwoForR;

    // attr
    batchNormTilingData.epsilon = epsilon_;
    batchNormTilingData.momentum = exponentialAvgFactor_;

    // core num
    int64_t blockFactor = (a_ + aicoreParams_.blockDim - 1) / aicoreParams_.blockDim;
    usedCoreNums_ = (a_ + blockFactor - 1) / blockFactor;
    batchNormTilingData.aBlockFactor = blockFactor;
    batchNormTilingData.blockNum = usedCoreNums_;

    // vf loop count
    int64_t r1r0LoopCount = ((r1_ * r0_) + vlFp32_ - 1) / vlFp32_;
    batchNormTilingData.r1r0LoopCount = r1r0LoopCount;

    // binary add k
    int64_t vcaddNum = binaryAddQuotient / vlFp32_; // 2的幂次方的数据要做二分
    if (vcaddNum <= static_cast<int64_t>(vlFp32_)) {
        batchNormTilingData.binaryAddK = 0;
        batchNormTilingData.binaryAddLastNum = vcaddNum;
    } else {
        int64_t binaryAddNum = vcaddNum / vlFp32_; // vl为一块，要累加的块，当前肯定是2的幂次方
        int64_t binaryAddK = 0;
        int64_t curBinaryAddNum = 1;
        while (curBinaryAddNum < binaryAddNum) {
            binaryAddK++;
            curBinaryAddNum *= BINARY_ADD_COEF;
        }
        batchNormTilingData.binaryAddK = binaryAddK;
        batchNormTilingData.binaryAddLastNum = vlFp32_;
    }

    batchNormTilingData.useRunningMeanVar = useRunningMeanVar_;
    return ge::GRAPH_SUCCESS;
}

uint64_t BatchNorm3DFullReduceTilingBase::GetTilingKey() const { return TILINGKEY_FULL_REDUCE; }

ge::graphStatus BatchNorm3DFullReduceTilingBase::PostTiling()
{
    context_->SetBlockDim(usedCoreNums_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    auto* tilingDataOut = context_->GetTilingData<BatchNorm3DFullReduceRegbaseTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingDataOut);
    *tilingDataOut = batchNormTilingData;
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(BatchNorm3D, BatchNorm3DFullReduceTilingBase, 20000);
} // namespace optiling
