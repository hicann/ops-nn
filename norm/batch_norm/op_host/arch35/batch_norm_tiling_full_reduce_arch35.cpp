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
 * \file batch_norm_tiling_full_reduce_arch35.cpp
 * \brief
 */
#include "batch_norm_tiling.h"

using namespace ge;

namespace
{
constexpr int64_t TILINGKEY_FULL_REDUCE = 200000;

constexpr int64_t SMALL_BUFFER_NUM = 8;
constexpr int64_t SMALL_BUFFER_NUM_FP32 = 8;
constexpr int64_t SMALL_BUFFER_NUM_T = 0;
constexpr int64_t LARGE_BUFFER_NUM = 2;
constexpr int64_t BINARY_ADD_COEF = 2;

}  // namespace

namespace optiling
{
class BatchNormFullReduceTilingBase : public BatchNormRegbaseTilingBase
{
public:
    explicit BatchNormFullReduceTilingBase(gert::TilingContext* context) : BatchNormRegbaseTilingBase(context)
    {
    }
    ~BatchNormFullReduceTilingBase() override = default;

    void Reset(gert::TilingContext* context) override
    {
        BatchNormRegbaseTilingBase::Reset(context);
        binaryAddQuotient = 0;
    }

protected:
    bool IsCapable() override
    {
        if (xFormat_ != FORMAT_NCHW && xFormat_ != FORMAT_NCDHW) {
            return false;
        }
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
        int64_t quotientVcaddSizeAlign = ((quotientVcaddNum * FLOAT32_BYTES + blockSize_ - 1) / blockSize_) * blockSize_;
        if (static_cast<uint64_t>(quotientVcaddSizeAlign) >= aicoreParams_.ubSize) {
            return false;
        }
        // reserve 8 block for 8 A tensor alignment
        int64_t ubCanUseSize =
            ((((aicoreParams_.ubSize - quotientVcaddSizeAlign) / DOUBLE_BUFFER) / blockSize_) * blockSize_);
        if (static_cast<int64_t>(SMALL_BUFFER_NUM * blockSize_) >= ubCanUseSize) {
            return false;
        }
        ubCanUseSize -= SMALL_BUFFER_NUM * blockSize_;
        int64_t r1r0Align = (((r1r0 * elemSize + blockSize_ - 1) / blockSize_) * blockSize_) / elemSize;
        // two AR tensor, two A tensor, six fp32 A tensor
        int64_t ubSizePerA =
            LARGE_BUFFER_NUM * r1r0Align * elemSize + SMALL_BUFFER_NUM_T * elemSize + SMALL_BUFFER_NUM_FP32 * FLOAT32_BYTES;
        int64_t aFactor = ubCanUseSize / ubSizePerA;
        if (aFactor >= 1) {
            batchNormTilingData.set_aFactor(aFactor);
            batchNormTilingData.set_binaryAddQuotient(binaryAddQuotient);
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
    BatchNormFullReduceRegbaseTilingData batchNormTilingData;
};

ge::graphStatus BatchNormFullReduceTilingBase::DoOpTiling()
{
    // dim
    batchNormTilingData.set_r1(r1_);
    batchNormTilingData.set_a(a_);
    batchNormTilingData.set_r0(r0_);
    int64_t rDim = r1_ * r0_;
    int64_t powerOfTwoForR = 1;
    while (powerOfTwoForR < rDim) {
        powerOfTwoForR *= BINARY_ADD_COEF;
    }
    batchNormTilingData.set_powerOfTwoForR(powerOfTwoForR);

    // attr
    batchNormTilingData.set_epsilon(epsilon_);
    batchNormTilingData.set_momentum(exponentialAvgFactor_);

    // core num
    int64_t blockFactor = (a_ + aicoreParams_.blockDim - 1) / aicoreParams_.blockDim;
    usedCoreNums_ = (a_ + blockFactor - 1) / blockFactor;
    batchNormTilingData.set_aBlockFactor(blockFactor);
    batchNormTilingData.set_blockNum(usedCoreNums_);

    // vf loop count
    int64_t r1r0LoopCount = ((r1_ * r0_) + vlFp32_ - 1) / vlFp32_;
    batchNormTilingData.set_r1r0LoopCount(r1r0LoopCount);

    // binary add k
    int64_t vcaddNum = binaryAddQuotient / vlFp32_;  // 2的幂次方的数据要做二分
    if (vcaddNum <= static_cast<int64_t>(vlFp32_)) {
        batchNormTilingData.set_binaryAddK(0);
        batchNormTilingData.set_binaryAddLastNum(vcaddNum);
    } else {
        int64_t binaryAddNum = vcaddNum / vlFp32_;  // vl为一块，要累加的块，当前肯定是2的幂次方
        int64_t binaryAddK = 0;
        int64_t curBinaryAddNum = 1;
        while (curBinaryAddNum < binaryAddNum) {
            binaryAddK++;
            curBinaryAddNum *= BINARY_ADD_COEF;
        }
        batchNormTilingData.set_binaryAddK(binaryAddK);
        batchNormTilingData.set_binaryAddLastNum(vlFp32_);
    }

    batchNormTilingData.set_useRunningMeanVar(useRunningMeanVar_);
    return ge::GRAPH_SUCCESS;
}

uint64_t BatchNormFullReduceTilingBase::GetTilingKey() const
{
    return TILINGKEY_FULL_REDUCE;
}

ge::graphStatus BatchNormFullReduceTilingBase::PostTiling()
{
    context_->SetBlockDim(usedCoreNums_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    batchNormTilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                                     context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(batchNormTilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(BatchNorm, BatchNormFullReduceTilingBase, 20000);
}  // namespace optiling