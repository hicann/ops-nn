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
 * \file batch_norm3d_tiling_welford_arch35.cpp
 * \brief
 */
#include "batch_norm3d_tiling.h"

using namespace ge;

namespace {
constexpr int64_t TILINGKEY_WELFORD_REDUCE = 300000;

constexpr int64_t SMALL_BUFFER_NUM = 9;
constexpr int64_t LARGE_BUFFER_NUM_QUEUE = 2;
constexpr int64_t LARGE_BUFFER_NUM_TMP = 2;
constexpr int64_t BINARY_ADD_COEF = 2;
constexpr int64_t MAX_COMMON_PARELLEL = 256;
// 6 for large case, 1 for extra
constexpr int64_t BLOCK_RESERVE_NUMBER = 7;

} // namespace

namespace optiling {
class BatchNorm3DWelfordReduceTilingBase : public BatchNorm3DRegbaseTilingBase {
public:
    explicit BatchNorm3DWelfordReduceTilingBase(gert::TilingContext* context) : BatchNorm3DRegbaseTilingBase(context)
    {
        Reset();
    }
    ~BatchNorm3DWelfordReduceTilingBase() override = default;

    void Reset(gert::TilingContext* context) override
    {
        BatchNorm3DRegbaseTilingBase::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        // 二分折叠要求 binaryAddQuotient 是整数个 vlFp32_（kernel 侧按 dichotomyAddPower / VL_FP32
        // 取整块循环次数），而它的下限就是一个 vlFp32_。因此归约长度 r1_ * r0_ 不超过一个
        // vlFp32_ 时，不存在合法取值，本模板不支持。
        // 该场景 r1_ * r0_ 很小，BatchNorm3DFullReduceTilingBase（优先级 20000）的 UB 判据必然通过，
        // 会先行接管；其 kernel 侧用 CeilDiv + 掩码处理不足一个 VL 的归约，天然支持这一档。
        if (r1_ * r0_ <= vlFp32_) {
            OP_LOGI(context_->GetNodeName(), "BatchNorm3DWelfordReduce not capable: r1(%ld) * r0(%ld) <= vlFp32(%ld).",
                    r1_, r0_, vlFp32_);
            return false;
        }
        return true;
    }

    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;

    void Reset();

private:
    // 不再声明本类私有的 opName：基类 BatchNorm3DTilingBase 已有同名 protected 成员，
    // 并在 GetPlatformInfo 中赋值为 context_->GetNodeName()。此前的私有声明遮蔽了它，
    // 又在 Reset() 中被置为 nullptr 且无处赋回，导致本文件所有 OP_LOGE 都打成 nil。
    int64_t binaryAddQuotient;
    int64_t parallelN;
    BatchNorm3DWelfordRegbaseTilingData tilingData;
};

void BatchNorm3DWelfordReduceTilingBase::Reset()
{
    binaryAddQuotient = 0;
    parallelN = 0;
    return;
}

inline static int64_t RoundUp(int64_t a, int64_t b) { return Ops::Base::CeilDiv(a, b) * b; }

ge::graphStatus BatchNorm3DWelfordReduceTilingBase::DoOpTiling()
{
    // block tiling
    tilingData.aBlockFactor = Ops::Base::CeilDiv(a_, (int64_t)aicoreParams_.blockDim);
    tilingData.realCoreNum = Ops::Base::CeilDiv(a_, tilingData.aBlockFactor);
    tilingData.numLastCore = a_ % tilingData.aBlockFactor;
    usedCoreNums_ = tilingData.realCoreNum;

    tilingData.elemNum = r1_ * r0_;
    tilingData.vlLenFp32 = vlFp32_;

    int64_t elemSize = FLOAT16_BYTES;
    if (xDtype_ == ge::DT_FLOAT) {
        elemSize = FLOAT32_BYTES;
    }
    int64_t elemAlignNum = blockSize_ / elemSize;

    // ub tiling
    int64_t aGatherLimit = tilingData.aBlockFactor > MAX_COMMON_PARELLEL ? MAX_COMMON_PARELLEL :
                                                                           tilingData.aBlockFactor;
    tilingData.aGatherLimit = aGatherLimit;

    int32_t totalUBSize = aicoreParams_.ubSize;
    uint64_t smallUbNum = RoundUp(tilingData.aGatherLimit * FLOAT32_BYTES, blockSize_);
    uint64_t smallUbSize = (smallUbNum * SMALL_BUFFER_NUM * DOUBLE_BUFFER) * FLOAT32_BYTES;

    int64_t binaryAddBufNum = (totalUBSize / (DOUBLE_BUFFER * LARGE_BUFFER_NUM_QUEUE * elemSize)) /
                              tilingData.vlLenFp32;
    int64_t binaryAddBufSize = ((binaryAddBufNum * FLOAT32_BYTES + blockSize_ - 1) / blockSize_) * blockSize_;

    // smallUbSize / blockSize_ 是无符号数，直接相减会在表达式内下溢（算术转换发生在赋值之前，
    // 只改左值类型无效），这里统一转成有符号数计算。
    int64_t ubRemain = static_cast<int64_t>(totalUBSize) - static_cast<int64_t>(smallUbSize) - binaryAddBufSize -
                       static_cast<int64_t>(blockSize_) * BLOCK_RESERVE_NUMBER;

    // processSize is max ub size.
    int64_t ubSize = ubRemain /
                     (DOUBLE_BUFFER * elemSize * LARGE_BUFFER_NUM_QUEUE + FLOAT32_BYTES * LARGE_BUFFER_NUM_TMP);
    int64_t ubSizeAlign = ubSize / elemAlignNum * elemAlignNum;
    // 放在对齐之后校验：ubRemain 为负时 ubSizeAlign 同样不为正，为正但不足一个对齐单位时会被
    // 抹成 0，这一条同时覆盖两种情况。
    OP_CHECK_IF(ubSizeAlign <= 0,
                OP_LOGE_WITHOUT_REPORT(opName, "ub size %d is not enough, ubSizeAlign %ld.", totalUBSize, ubSizeAlign),
                return ge::GRAPH_FAILED);

    if (r0_ >= ubSizeAlign) {
        tilingData.r0Factor = ubSizeAlign;
        tilingData.loopR0outer = Ops::Base::CeilDiv(r0_, ubSizeAlign);
        tilingData.r1Factor = 1;
        tilingData.loopR1outer = r1_;
        tilingData.ubSize = ubSizeAlign;
        parallelN = ubSizeAlign;
        tilingData.parallelN = parallelN;
        tilingData.processSize = ubSizeAlign;
        tilingData.cutR1OrR0 = 0;
    } else {
        int64_t r1Factor = ubSizeAlign / r0_;
        r1Factor = r1Factor > r1_ ? r1_ : r1Factor;

        tilingData.r0Factor = r0_;
        tilingData.loopR0outer = 1;
        tilingData.r1Factor = r1Factor;
        tilingData.loopR1outer = Ops::Base::CeilDiv(r1_, r1Factor);
        int64_t processSize = r0_ * r1Factor;
        ubSizeAlign = (processSize + elemAlignNum - 1) / elemAlignNum * elemAlignNum;
        tilingData.ubSize = ubSizeAlign;
        parallelN = processSize;
        tilingData.parallelN = parallelN;
        tilingData.processSize = processSize;
        tilingData.cutR1OrR0 = 1;
    }

    // parallelN 才是二分折叠的实际输入：它取 ubSizeAlign（r0_ >= ubSizeAlign 分支）或
    // r0_ * r1Factor（另一分支），两者恒小于等于 r1_ * r0_。UB 紧张时 ubSizeAlign 可以小到
    // 一个 vlFp32_ 以内，此时 IsCapable 里按 r1_ * r0_ 判的守卫拦不住：binaryAddQuotient
    // 仍会退化成半个 VL，vcaddNum 与 binaryAddLastNum 双双为 0，归约结果恒为 0。
    OP_CHECK_IF(
        parallelN <= vlFp32_,
        OP_LOGE_WITHOUT_REPORT(opName, "ub size %d is not enough, parallelN %ld should be greater than vlFp32 %ld.",
                               totalUBSize, parallelN, vlFp32_),
        return ge::GRAPH_FAILED);

    // binary add param
    int64_t vlLenFp32 = tilingData.vlLenFp32;
    binaryAddQuotient = vlLenFp32;
    while (binaryAddQuotient < parallelN) {
        binaryAddQuotient = binaryAddQuotient * BINARY_ADD_COEF;
    }
    binaryAddQuotient = binaryAddQuotient / BINARY_ADD_COEF;
    tilingData.binaryAddQuotient = binaryAddQuotient;

    OP_CHECK_IF(vlLenFp32 == 0, OP_LOGE_WITHOUT_REPORT(opName, "vlLenFp32 should not be 0."), return ge::GRAPH_FAILED);
    int64_t vcaddNum = binaryAddQuotient / vlLenFp32;
    if (vcaddNum <= vlLenFp32) {
        tilingData.binaryAddK = 0;
        tilingData.binaryAddLastNum = vcaddNum;
    } else {
        int64_t binaryAddNum = vcaddNum / vlLenFp32;
        int64_t binaryAddK = 0;
        int64_t tmpBinaryAddNum = 1;
        while (tmpBinaryAddNum < binaryAddNum) {
            binaryAddK = binaryAddK + 1;
            tmpBinaryAddNum = tmpBinaryAddNum * BINARY_ADD_COEF;
        }
        tilingData.binaryAddK = binaryAddK;
        tilingData.binaryAddLastNum = vlLenFp32;
    }

    tilingData.epsilon = epsilon_;
    tilingData.momentum = exponentialAvgFactor_;
    tilingData.r1 = r1_;
    tilingData.a0 = a_;
    tilingData.r0 = r0_;
    tilingData.useRunningMeanVar = useRunningMeanVar_;
    return ge::GRAPH_SUCCESS;
}

uint64_t BatchNorm3DWelfordReduceTilingBase::GetTilingKey() const { return TILINGKEY_WELFORD_REDUCE; }

ge::graphStatus BatchNorm3DWelfordReduceTilingBase::PostTiling()
{
    context_->SetBlockDim(usedCoreNums_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    auto* tilingDataOut = context_->GetTilingData<BatchNorm3DWelfordRegbaseTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingDataOut);
    *tilingDataOut = tilingData;

    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(BatchNorm3D, BatchNorm3DWelfordReduceTilingBase, 30000);
} // namespace optiling
