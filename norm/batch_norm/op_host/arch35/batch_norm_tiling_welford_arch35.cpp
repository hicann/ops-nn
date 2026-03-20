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
 * \file batch_norm_tiling_welford_arch35.cpp
 * \brief
 */
#include "batch_norm_tiling.h"

using namespace ge;

namespace
{
constexpr int64_t TILINGKEY_WELFORD_REDUCE = 300000;

constexpr int64_t SMALL_BUFFER_NUM = 9;
constexpr int64_t LARGE_BUFFER_NUM_QUEUE = 2;
constexpr int64_t LARGE_BUFFER_NUM_TMP = 2;
constexpr int64_t BINARY_ADD_COEF = 2;
constexpr int64_t MAX_COMMON_PARELLEL = 256;
// 6 for large case, 1 for extra
constexpr int64_t BLOCK_RESERVE_NUMBER = 7;

}  // namespace

namespace optiling
{
class BatchNormWelfordReduceTilingBase : public BatchNormRegbaseTilingBase
{
public:
    explicit BatchNormWelfordReduceTilingBase(gert::TilingContext* context) : BatchNormRegbaseTilingBase(context)
    {
        Reset();
    }
    ~BatchNormWelfordReduceTilingBase() override = default;

    void Reset(gert::TilingContext* context) override
    {
        BatchNormRegbaseTilingBase::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        return true;
    }

    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

    void Reset();

private:
    const char* opName = "BatchNormWelfordReduce";
    int64_t binaryAddQuotient;
    int64_t parallelN;
    BatchNormWelfordRegbaseTilingData tilingData;
};

void BatchNormWelfordReduceTilingBase::Reset()
{
    opName = nullptr;
    binaryAddQuotient = 0;
    parallelN = 0;
    return;
}

inline static int64_t RoundUp(int64_t a, int64_t b)
{
    return Ops::Base::CeilDiv(a, b) * b;
}

ge::graphStatus BatchNormWelfordReduceTilingBase::DoOpTiling()
{
    // block tiling
    tilingData.set_aBlockFactor(Ops::Base::CeilDiv(a_, (int64_t)aicoreParams_.blockDim));
    tilingData.set_realCoreNum(Ops::Base::CeilDiv(a_, tilingData.get_aBlockFactor()));
    tilingData.set_numLastCore(a_ % tilingData.get_aBlockFactor());
    usedCoreNums_ = tilingData.get_realCoreNum();

    tilingData.set_elemNum(r1_ * r0_);
    tilingData.set_vlLenFp32(vlFp32_);

    int64_t elemSize = FLOAT16_BYTES;
    if (xDtype_ == ge::DT_FLOAT) {
        elemSize = FLOAT32_BYTES;
    }
    int64_t elemAlignNum = blockSize_ / elemSize;

    // ub tiling
    int64_t aGatherLimit =
        tilingData.get_aBlockFactor() > MAX_COMMON_PARELLEL ? MAX_COMMON_PARELLEL : tilingData.get_aBlockFactor();
    tilingData.set_aGatherLimit(aGatherLimit);

    int32_t totalUBSize = aicoreParams_.ubSize;
    uint64_t smallUbNum = RoundUp(tilingData.get_aGatherLimit() * FLOAT32_BYTES, blockSize_);
    uint64_t smallUbSize = (smallUbNum * SMALL_BUFFER_NUM * DOUBLE_BUFFER) * FLOAT32_BYTES;

    int64_t binaryAddBufNum =
        (totalUBSize / (DOUBLE_BUFFER * LARGE_BUFFER_NUM_QUEUE * elemSize)) / tilingData.get_vlLenFp32();
    int64_t binaryAddBufSize = ((binaryAddBufNum * FLOAT32_BYTES + blockSize_ - 1) / blockSize_) * blockSize_;

    uint64_t ubRemain = totalUBSize - smallUbSize - binaryAddBufSize - blockSize_ * BLOCK_RESERVE_NUMBER;

    // processSize is max ub size.
    int64_t ubSize =
        ubRemain / (DOUBLE_BUFFER * elemSize * LARGE_BUFFER_NUM_QUEUE + FLOAT32_BYTES * LARGE_BUFFER_NUM_TMP);
    int64_t ubSizeAlign = ubSize / elemAlignNum * elemAlignNum;

    if (r0_ >= ubSizeAlign) {
        tilingData.set_r0Factor(ubSizeAlign);
        tilingData.set_loopR0outer(Ops::Base::CeilDiv(r0_, ubSizeAlign));
        tilingData.set_r1Factor(1);
        tilingData.set_loopR1outer(r1_);
        tilingData.set_ubSize(ubSizeAlign);
        parallelN = ubSizeAlign;
        tilingData.set_parallelN(parallelN);
        tilingData.set_processSize(ubSizeAlign);
        tilingData.set_cutR1OrR0(0);
    } else {
        int64_t r1Factor = ubSizeAlign / r0_;
        r1Factor = r1Factor > r1_ ? r1_ : r1Factor;

        tilingData.set_r0Factor(r0_);
        tilingData.set_loopR0outer(1);
        tilingData.set_r1Factor(r1Factor);
        tilingData.set_loopR1outer(Ops::Base::CeilDiv(r1_, r1Factor));
        int64_t processSize = r0_ * r1Factor;
        ubSizeAlign = (processSize + elemAlignNum - 1) / elemAlignNum * elemAlignNum;
        tilingData.set_ubSize(ubSizeAlign);
        parallelN = processSize;
        tilingData.set_parallelN(parallelN);
        tilingData.set_processSize(processSize);
        tilingData.set_cutR1OrR0(1);
    }

    // binary add param
    int64_t vlLenFp32 = tilingData.get_vlLenFp32();
    binaryAddQuotient = vlLenFp32;
    while (binaryAddQuotient < parallelN) {
        binaryAddQuotient = binaryAddQuotient * BINARY_ADD_COEF;
    }
    binaryAddQuotient = binaryAddQuotient / BINARY_ADD_COEF;
    tilingData.set_binaryAddQuotient(binaryAddQuotient);

    OP_CHECK_IF(vlLenFp32 == 0, OP_LOGE(opName, "vlLenFp32 should not be 0."),
                return ge::GRAPH_FAILED);
    int64_t vcaddNum = binaryAddQuotient / vlLenFp32;
    if (vcaddNum <= vlLenFp32) {
        tilingData.set_binaryAddK(0);
        tilingData.set_binaryAddLastNum(vcaddNum);
    } else {
        int64_t binaryAddNum = vcaddNum / vlLenFp32;
        int64_t binaryAddK = 0;
        int64_t tmpBinaryAddNum = 1;
        while (tmpBinaryAddNum < binaryAddNum) {
            binaryAddK = binaryAddK + 1;
            tmpBinaryAddNum = tmpBinaryAddNum * BINARY_ADD_COEF;
        }
        tilingData.set_binaryAddK(binaryAddK);
        tilingData.set_binaryAddLastNum(vlLenFp32);
    }

    tilingData.set_epsilon(epsilon_);
    tilingData.set_momentum(exponentialAvgFactor_);
    tilingData.set_r1(r1_);
    tilingData.set_a0(a_);
    tilingData.set_r0(r0_);
    tilingData.set_useRunningMeanVar(useRunningMeanVar_);
    return ge::GRAPH_SUCCESS;
}

uint64_t BatchNormWelfordReduceTilingBase::GetTilingKey() const
{
    return TILINGKEY_WELFORD_REDUCE;
}

ge::graphStatus BatchNormWelfordReduceTilingBase::PostTiling()
{
    context_->SetBlockDim(usedCoreNums_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(tilingData.GetDataSize() > rawTilingData->GetCapacity(),
                OP_LOGE(context_->GetNodeName(),
                    "actual tiling data size %zu > context tiling data size %zu",
                    tilingData.GetDataSize(), rawTilingData->GetCapacity()),
                return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(BatchNorm, BatchNormWelfordReduceTilingBase, 30000);
}  // namespace optiling