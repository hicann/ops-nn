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
 * \file batch_norm_grad_tiling_rar_split_core_r0.cpp
 * \brief
 */

#include "batch_norm_grad_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling_base/tiling_templates_registry.h"

using namespace AscendC;

namespace optiling
{
static constexpr uint64_t BATCH_NORM_GRAD_RAR_SPLIT_CORE_R0_TILING_KEY = 1100;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t CONST_TWO = 2;
static constexpr int32_t ULONG_BIT_LEN = 64;
static constexpr int64_t CACHE_BUFFER_COUNT_MAX = 63;  // 按int64最大值代入计算
static constexpr int64_t FLOAT32_BYTES = 4;
static constexpr int64_t FLOAT16_BYTES = 2;
static constexpr int64_t R0_THRESHOLD = 102400;
static constexpr int64_t CACHE_LINE = 512;
static constexpr int64_t SMALL_SHAPES_STG0 = 4;
static constexpr int64_t BIG_SHAPES_STG0 = 2;
static constexpr int64_t SMALL_SHAPES_STG2 = 5;
static constexpr int64_t BIG_SHAPES_STG2 = 3;
static constexpr int64_t REDUCE_TMP_BUF_NUM_FP16 = 2;
static constexpr int64_t REDUCE_TMP_BUF_NUM_FP32 = 1;
static constexpr size_t BNG_WORKSPACE_RESERVED = 16 * 1024 * 1024;

constexpr int64_t CONST_ONE = 1;

class BatchNormGradRARSplitCoreR0 : public BatchNormGradTilingBase
{
public:
    explicit BatchNormGradRARSplitCoreR0(gert::TilingContext* context)
        : BatchNormGradTilingBase(context, tilingData_.baseTilingData)
    {
    }

protected:
    bool IsCapable() override;
    // 计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 计算TilingKey
    uint64_t GetTilingKey() const override;
    // 计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 保存Tiling数据
    ge::graphStatus PostTiling() override;

private:
    ge::graphStatus DoOpTilingStage0();
    ge::graphStatus DoOpTilingStage1();
    ge::graphStatus DoOpTilingStage2();
    int64_t ComputeBinaryAddParams(int64_t fusedR, int64_t lastCoreFusedR);
    int64_t GetCacheID(const int64_t idx);
    BatchNormGradRARSplitCoreR0TilingData tilingData_;

    int64_t usedCoreNums_{0};
    int64_t blockFp32Nums_{0};
    int64_t blockFp16Nums_{0};
    int64_t cacheLineFp32Nums_{0};
    int64_t cacheLineFp16Nums_{0};
    int64_t r0Inner_{0};
    int64_t r0Tail_{0};
    int64_t aDimAligned_{0};
    int64_t dyDtypeSize_{0};
    int64_t dyBaseLen_{0};
    int64_t dyBlockAlignedLen_{0};
};

bool BatchNormGradRARSplitCoreR0::IsCapable()
{
    if (r0Dim < R0_THRESHOLD) {
        return false;
    }

    if (aDim != 1) {
        return false;
    }

    if (r1Dim != 1) {
        return false;
    }

    if (dyDtype == ge::DT_FLOAT) {
        return false;
    }

    return true;
}

ge::graphStatus BatchNormGradRARSplitCoreR0::DoOpTiling()
{
    blockFp32Nums_ = blockSize / FLOAT32_BYTES;
    blockFp16Nums_ = blockSize / FLOAT16_BYTES;
    cacheLineFp32Nums_ = CACHE_LINE / FLOAT32_BYTES;
    cacheLineFp16Nums_ = CACHE_LINE / FLOAT16_BYTES;
    dyBaseLen_ = dyDtype == ge::DT_FLOAT ? cacheLineFp32Nums_ : cacheLineFp16Nums_;
    dyBlockAlignedLen_ = dyDtype == ge::DT_FLOAT ? blockFp32Nums_ : blockFp16Nums_;

    dyDtypeSize_ = dyDtype == ge::DT_FLOAT ? FLOAT32_BYTES : FLOAT16_BYTES;

    r0Inner_ = Ops::Base::CeilDiv(Ops::Base::CeilDiv(r0Dim, dyBaseLen_), static_cast<int64_t>(coreNum)) * dyBaseLen_;
    usedCoreNums_ = Ops::Base::CeilDiv(r0Dim, r0Inner_);
    r0Tail_ = r0Dim - r0Inner_ * (usedCoreNums_ - CONST_ONE);
    aDimAligned_ = Ops::Base::CeilAlign(aDim, blockFp32Nums_);

    OP_CHECK_IF(DoOpTilingStage0() != ge::GRAPH_SUCCESS, , return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(DoOpTilingStage1() != ge::GRAPH_SUCCESS, , return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(DoOpTilingStage2() != ge::GRAPH_SUCCESS, , return ge::GRAPH_PARAM_INVALID);

    tilingData_.set_r1Dim(r1Dim);
    tilingData_.set_aDim(aDim);
    tilingData_.set_aDimAligned(aDimAligned_);
    tilingData_.set_r0Dim(r0Dim);
    tilingData_.set_usedCoreNums(usedCoreNums_);
    tilingData_.set_r0Inner(r0Inner_);
    tilingData_.set_r0Tail(r0Tail_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradRARSplitCoreR0::DoOpTilingStage0()
{
    int64_t aInner = CONST_ONE;
    int64_t aInnerAligned = blockFp32Nums_;
    int64_t cacheBufferCount = CACHE_BUFFER_COUNT_MAX;

    // 计算公式: ubsize >= r0InnerStg0 * r1InnerInner * aInner * (sizeof(dy) * DOUBLE_BUFFER * BIG_SHAPES_STG0 +
    // binAddTmpFactor*sizeof(float)) + aInnerAligned * sizeof(float) * DOUBLE_BUFFER * SMALL_SHAPES_STG0 +
    // cacheBufferCount）

    // dbeta二分预处理，fp32直接用拷贝输入，b16增加cast输出缓存
    int64_t reduceTmpBufNum = dyDtype == ge::DT_FLOAT ? REDUCE_TMP_BUF_NUM_FP32 : REDUCE_TMP_BUF_NUM_FP16;
    // 切r0
    int64_t factorMax =
        (ubSize - aInnerAligned * FLOAT32_BYTES * (DOUBLE_BUFFER * SMALL_SHAPES_STG0 + cacheBufferCount)) / dyBaseLen_ /
        (DOUBLE_BUFFER * dyDtypeSize_ * BIG_SHAPES_STG0 + reduceTmpBufNum * FLOAT32_BYTES);

    OP_CHECK_IF(
        factorMax <= 0,
        OP_LOGE(context_, "BatchNormGrad RAR R0 split core template is not capable. Shape (%ld, %ld, %ld), "
            "factorMax in stage0 is %ld .",
            r1Dim, aDim, r0Dim, factorMax),
        return ge::GRAPH_PARAM_INVALID);

    int64_t r0FactorMax = Ops::Base::CeilDiv(r0Inner_, dyBaseLen_);
    int64_t r0Factor = factorMax <= r0FactorMax ? factorMax : r0FactorMax;
    int64_t r0InnerInner = r0Factor * dyBaseLen_;
    int64_t r0InnerOuter = Ops::Base::CeilDiv(r0Inner_, r0InnerInner);
    int64_t r0InnerTail = r0Inner_ - (r0InnerOuter - CONST_ONE) * r0InnerInner;
    int64_t r0TailOuter = Ops::Base::CeilDiv(r0Tail_, r0InnerInner);
    int64_t r0TailTail = r0Tail_ - (r0TailOuter - CONST_ONE) * r0InnerInner;
    int64_t r0TailTailAligned = Ops::Base::CeilAlign(r0TailTail, dyBlockAlignedLen_);
    // 切r1
    factorMax = factorMax / r0Factor;
    int64_t r1Inner = factorMax <= r1Dim ? factorMax : r1Dim;
    int64_t r1Outer = Ops::Base::CeilDiv(r1Dim, r1Inner);
    int64_t r1Tail = r1Dim - (r1Outer - CONST_ONE) * r1Inner;

    int64_t fusedR = r1Outer * r0InnerOuter;
    int64_t lastCoreFusedR = r1Outer * r0TailOuter;

    cacheBufferCount = ComputeBinaryAddParams(fusedR, lastCoreFusedR);

    OP_LOGD(context_,
            "BatchNormGrad R0 split core template Stage0, r1Outer: %ld, r0InnerOuter: %ld, cache buffer count: %ld.",
            r1Outer, r0InnerOuter, cacheBufferCount);

    // 切a, 取（aInner + (blockFp32Nums_ -1)) == aAligned 简化计算
    int64_t aInnerMax =
        (ubSize -
         (blockFp32Nums_ - CONST_ONE) * FLOAT32_BYTES * (DOUBLE_BUFFER * SMALL_SHAPES_STG0 + cacheBufferCount)) /
        (r1Inner * r0InnerInner * (dyDtypeSize_ * DOUBLE_BUFFER * BIG_SHAPES_STG0 + reduceTmpBufNum * FLOAT32_BYTES) +
         (DOUBLE_BUFFER * SMALL_SHAPES_STG0 + cacheBufferCount + 2) * FLOAT32_BYTES);

    aInner = aInnerMax < aDim ? aInnerMax : aDim;
    aInnerAligned = Ops::Base::CeilAlign(aInner, blockFp32Nums_);
    int64_t aOuter = Ops::Base::CeilDiv(aDim, aInner);
    int64_t aTail = aDim - (aOuter - CONST_ONE) * aInner;

    tilingData_.set_r1InnerStg0(r1Inner);
    tilingData_.set_r1OuterStg0(r1Outer);
    tilingData_.set_r1TailStg0(r1Tail);
    tilingData_.set_r0InnerInnerStg0(r0InnerInner);
    tilingData_.set_r0InnerOuterStg0(r0InnerOuter);
    tilingData_.set_r0InnerTailStg0(r0InnerTail);
    tilingData_.set_r0TailOuterStg0(r0TailOuter);
    tilingData_.set_r0TailTailStg0(r0TailTail);
    tilingData_.set_r0TailTailAlignedStg0(r0TailTailAligned);
    tilingData_.set_aInnerStg0(aInner);
    tilingData_.set_aInnerAlignedStg0(aInnerAligned);
    tilingData_.set_aOuterStg0(aOuter);
    tilingData_.set_aTailStg0(aTail);

    return ge::GRAPH_SUCCESS;
}

int64_t BatchNormGradRARSplitCoreR0::GetCacheID(const int64_t idx)
{
    return __builtin_popcountll(idx ^ (idx + CONST_ONE)) - CONST_ONE;
}

int64_t BatchNormGradRARSplitCoreR0::ComputeBinaryAddParams(int64_t fusedR, int64_t lastCoreFusedR)
{
    // 计算最接近binAddRTotalLoop_的2^k
    int64_t binAddBasicBlockLoop = fusedR > 1 ? (1L << (ULONG_BIT_LEN - 1 - __builtin_clzl(fusedR - CONST_ONE))) : 0;
    int64_t mainFoldCount = fusedR - binAddBasicBlockLoop;
    int64_t binAddCacheBufferCount = CONST_ONE;
    int64_t binAddResultCacheID = 0;
    if (binAddBasicBlockLoop != 0) {
        binAddCacheBufferCount = ULONG_BIT_LEN - __builtin_clzl(binAddBasicBlockLoop);
        binAddResultCacheID = GetCacheID(binAddBasicBlockLoop - CONST_ONE);
    }

    tilingData_.set_binAddBasicBlockLoop(binAddBasicBlockLoop);
    tilingData_.set_binAddMainFoldCount(mainFoldCount);
    tilingData_.set_binAddCacheBufferCount(binAddCacheBufferCount);
    tilingData_.set_binAddResultCacheID(binAddResultCacheID);

    int64_t lastCoreBinAddBasicBlockLoop =
        lastCoreFusedR > 1 ? (1L << (ULONG_BIT_LEN - CONST_ONE - __builtin_clzl(lastCoreFusedR - CONST_ONE))) : 0;
    int64_t lastCoreMainFoldCount = lastCoreFusedR - lastCoreBinAddBasicBlockLoop;
    int64_t lastCoreBinAddResultCacheID = 0;
    if (lastCoreBinAddBasicBlockLoop != 0) {
        lastCoreBinAddResultCacheID = GetCacheID(lastCoreBinAddBasicBlockLoop - CONST_ONE);
    }

    tilingData_.set_lastCoreBinAddBasicBlockLoop(lastCoreBinAddBasicBlockLoop);
    tilingData_.set_lastCoreBinAddMainFoldCount(lastCoreMainFoldCount);
    tilingData_.set_lastCoreBinAddResultCacheID(lastCoreBinAddResultCacheID);

    return binAddCacheBufferCount;
}

ge::graphStatus BatchNormGradRARSplitCoreR0::DoOpTilingStage1()
{
    int64_t weightDtypeSize = weightDtype == ge::DT_FLOAT ? FLOAT32_BYTES : FLOAT16_BYTES;
    int64_t weightBaseLen = weightDtype == ge::DT_FLOAT ? blockFp32Nums_ : blockFp16Nums_;

    // 计算公式: ubSize >= aInnerStg1 * sizeof(float) * usedCoreNums_ * DOUBLE_BUFFER * 2 +
    // aInnerStg1 * DOUBLE_BUFFER * 2 * (sizeof(float) + sizeof(DTYPE_WEIGHT))
    int64_t factorMax = ubSize / (DOUBLE_BUFFER * CONST_TWO * weightBaseLen *
                                  (FLOAT32_BYTES * usedCoreNums_ + (FLOAT32_BYTES + weightDtypeSize)));

    OP_CHECK_IF(factorMax <= 0,
        OP_LOGE(context_, "BatchNormGrad RAR R0 split core template is not capable. Shape (%ld, %ld, %ld), "
            "factorMax in stage1 is %ld .",
            r1Dim, aDim, r0Dim, factorMax),
        return ge::GRAPH_PARAM_INVALID);

    int64_t aFactorMax = Ops::Base::CeilDiv(aDim, weightBaseLen);
    int64_t aFactor = factorMax <= aFactorMax ? factorMax : aFactorMax;
    int64_t aInner = aFactor * weightBaseLen;
    int64_t aOuter = Ops::Base::CeilDiv(aDim, aInner);
    int64_t aTail = aDim - (aOuter - CONST_ONE) * aInner;

    OP_LOGD(context_, "BatchNormGrad R0 split core template Stage1, weightBaseLen: %ld, aFactor: %ld, aInner: %ld.",
            weightBaseLen, aFactor, aInner);

    tilingData_.set_aInnerStg1(aInner);
    tilingData_.set_aOuterStg1(aOuter);
    tilingData_.set_aTailStg1(aTail);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradRARSplitCoreR0::DoOpTilingStage2()
{
    int64_t aInner = CONST_ONE;
    int64_t aInnerAligned = blockFp32Nums_;

    // 计算公式: ubsize >= r0Inner * r1InnerInner * aInner * sizeof(DTYPE_DY) * DOUBLE_BUFFER * 3 + aInnerAligned
    // * sizeof(float) * DOUBLE_BUFFER * 5 切r0
    int64_t factorMax = (ubSize - aInnerAligned * FLOAT32_BYTES * DOUBLE_BUFFER * SMALL_SHAPES_STG2) / dyBaseLen_ /
                        (DOUBLE_BUFFER * FLOAT32_BYTES * BIG_SHAPES_STG2);

    OP_CHECK_IF(factorMax <= 0,
        OP_LOGE(context_, "BatchNormGrad RAR R0 split core template is not capable. Shape (%ld, %ld, %ld), "
            "factorMax in stage2 is %ld .",
            r1Dim, aDim, r0Dim, factorMax),
        return ge::GRAPH_PARAM_INVALID);

    int64_t r0FactorMax = Ops::Base::CeilDiv(r0Inner_, dyBaseLen_);
    int64_t r0Factor = factorMax <= r0FactorMax ? factorMax : r0FactorMax;
    int64_t r0InnerInner = r0Factor * dyBaseLen_;
    int64_t r0InnerOuter = Ops::Base::CeilDiv(r0Inner_, r0InnerInner);
    int64_t r0InnerTail = r0Inner_ - (r0InnerOuter - CONST_ONE) * r0InnerInner;
    int64_t r0TailOuter = Ops::Base::CeilDiv(r0Tail_, r0InnerInner);
    int64_t r0TailTail = r0Tail_ - (r0TailOuter - CONST_ONE) * r0InnerInner;
    int64_t r0TailTailAligned = Ops::Base::CeilAlign(r0TailTail, dyBlockAlignedLen_);

    // 切r1
    factorMax = factorMax / r0Factor;
    int64_t r1Inner = factorMax <= r1Dim ? factorMax : r1Dim;
    int64_t r1Outer = Ops::Base::CeilDiv(r1Dim, r1Inner);
    int64_t r1Tail = r1Dim - (r1Outer - CONST_ONE) * r1Inner;

    // 切a, 取(aInner + (blockFp32Nums_ -1)) == aAligned 简化计算
    int64_t aInnerMax = (ubSize - (blockFp32Nums_ - CONST_ONE) * FLOAT32_BYTES * DOUBLE_BUFFER * SMALL_SHAPES_STG2) /
                        (r1Inner * r0InnerInner * DOUBLE_BUFFER * BIG_SHAPES_STG2 * dyDtypeSize_ +
                         FLOAT32_BYTES * DOUBLE_BUFFER * SMALL_SHAPES_STG2);

    aInner = aInnerMax < aDim ? aInnerMax : aDim;
    aInnerAligned = Ops::Base::CeilAlign(aInner, blockFp32Nums_);
    int64_t aOuter = Ops::Base::CeilDiv(aDim, aInner);
    int64_t aTail = aDim - (aOuter - CONST_ONE) * aInner;

    tilingData_.set_r1InnerStg2(r1Inner);
    tilingData_.set_r1OuterStg2(r1Outer);
    tilingData_.set_r1TailStg2(r1Tail);
    tilingData_.set_r0InnerInnerStg2(r0InnerInner);
    tilingData_.set_r0InnerOuterStg2(r0InnerOuter);
    tilingData_.set_r0InnerTailStg2(r0InnerTail);
    tilingData_.set_r0TailOuterStg2(r0TailOuter);
    tilingData_.set_r0TailTailStg2(r0TailTail);
    tilingData_.set_r0TailTailAlignedStg2(r0TailTailAligned);
    tilingData_.set_aInnerStg2(aInner);
    tilingData_.set_aInnerAlignedStg2(aInnerAligned);
    tilingData_.set_aOuterStg2(aOuter);
    tilingData_.set_aTailStg2(aTail);

    return ge::GRAPH_SUCCESS;
}

uint64_t BatchNormGradRARSplitCoreR0::GetTilingKey() const
{
    return BATCH_NORM_GRAD_RAR_SPLIT_CORE_R0_TILING_KEY;
}

ge::graphStatus BatchNormGradRARSplitCoreR0::GetWorkspaceSize()
{
    workspaceSize_ = BNG_WORKSPACE_RESERVED + usedCoreNums_ * aDimAligned_ * FLOAT32_BYTES * CONST_TWO;

    OP_LOGI(context_, "Workspace size: %ld", workspaceSize_);

    size_t* workspaces = context_->GetWorkspaceSizes(CONST_ONE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradRARSplitCoreR0::PostTiling()
{
    context_->SetBlockDim(usedCoreNums_);
    context_->SetScheduleMode(CONST_ONE);  // Set to batch mode, all cores start simultaneously
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(BatchNormGrad, BatchNormGradRARSplitCoreR0, 2100);

}  // namespace optiling
