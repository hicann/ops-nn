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
 * \file batch_norm_grad_tiling_rar_split_core_r1.cpp
 * \brief
 */

#include "batch_norm_grad_tiling.h"
#include "register/op_impl_registry.h"
#include "tiling_base/tiling_templates_registry.h"

using namespace AscendC;

namespace optiling
{
static constexpr uint64_t BATCH_NORM_GRAD_RAR_SPLIT_CORE_R1_TILING_KEY = 1000;
static constexpr int64_t A_SPLIT_CORE_TEMPLATE_ENABLE_FACTOR = 4;  // a > usedCore/4 走A分核
static constexpr int64_t R1_SPLIT_CORE_TEMPLATE_ENABLE_FACTOR = 5;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t CONST_TWO = 2;
static constexpr int32_t ULONG_BIT_LEN = 64;
static constexpr int64_t CACHE_BUFFER_COUNT_MAX = 63;  // 按int64最大值代入计算
static constexpr int64_t FLOAT32_BYTES = 4;
static constexpr int64_t FLOAT16_BYTES = 2;
static constexpr int64_t SMALL_SHAPES_STG0 = 4;
static constexpr int64_t BIG_SHAPES_STG0 = 2;
static constexpr int64_t SMALL_SHAPES_STG2 = 5;
static constexpr int64_t BIG_SHAPES_STG2 = 3;
static constexpr int64_t REDUCE_TMP_BUF_NUM_FP16 = 2;
static constexpr int64_t REDUCE_TMP_BUF_NUM_FP32 = 1;
static constexpr size_t BNG_WORKSPACE_RESERVED = 16 * 1024 * 1024 * 2;

constexpr int64_t CONST_ONE = 1;

class BatchNormGradRARSplitCoreR1 : public BatchNormGradTilingBase
{
public:
    explicit BatchNormGradRARSplitCoreR1(gert::TilingContext* context)
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
    BatchNormGradRARSplitCoreR1TilingData tilingData_;

    int64_t usedCoreNums_{0};
    int64_t blockFp32Nums_{0};
    int64_t blockFp16Nums_{0};

    int64_t r1Inner_{0};
    int64_t r1Tail_{0};
    int64_t aDimAligned_{0};
    int64_t dyDtypeSize_{0};
    int64_t dyBaseLen_{0};
};

bool BatchNormGradRARSplitCoreR1::IsCapable()
{
    // 走RA模板
    if (r0Dim == CONST_ONE) {
        return false;
    }

    // a轴开核大于1/4
    if (aDim > static_cast<int64_t>(coreNum) / A_SPLIT_CORE_TEMPLATE_ENABLE_FACTOR) {
        OP_LOGD(context_, "RAR split core template not support shape: (%ld, %ld, %ld), aDim > %ld",
                r1Dim, aDim, r0Dim, static_cast<int64_t>(coreNum) / A_SPLIT_CORE_TEMPLATE_ENABLE_FACTOR);
        return false;
    }

    // r1 开核后大于一半
    if (r1Dim < static_cast<int64_t>(coreNum)) {
        OP_LOGD(context_, "RAR split core template not support shape: (%ld, %ld, %ld), r1Dim < %ld",
                r1Dim, aDim, r0Dim, static_cast<int64_t>(coreNum));
        return false;
    }

    // 剔除全载场景
    if (r1Dim * r0Dim * R1_SPLIT_CORE_TEMPLATE_ENABLE_FACTOR < static_cast<int64_t>(ubSize)) {
        OP_LOGD(context_,
                "RAR split core template not support shape: (%ld, %ld, %ld), r0Dim * r1Dim < %ld", r1Dim, aDim, r0Dim,
                ubSize / R1_SPLIT_CORE_TEMPLATE_ENABLE_FACTOR);
        return false;
    }

    return true;
}

ge::graphStatus BatchNormGradRARSplitCoreR1::DoOpTiling()
{
    blockFp32Nums_ = blockSize / FLOAT32_BYTES;
    blockFp16Nums_ = blockSize / FLOAT16_BYTES;

    r1Inner_ = Ops::Base::CeilDiv(r1Dim, static_cast<int64_t>(coreNum));
    usedCoreNums_ = Ops::Base::CeilDiv(r1Dim, r1Inner_);
    r1Tail_ = r1Dim - r1Inner_ * (usedCoreNums_ - CONST_ONE);

    aDimAligned_ = Ops::Base::CeilAlign(aDim, blockFp32Nums_);

    dyDtypeSize_ = dyDtype == ge::DT_FLOAT ? FLOAT32_BYTES : FLOAT16_BYTES;
    dyBaseLen_ = dyDtype == ge::DT_FLOAT ? blockFp32Nums_ : blockFp16Nums_;

    OP_CHECK_IF(DoOpTilingStage0() != ge::GRAPH_SUCCESS, , return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(DoOpTilingStage1() != ge::GRAPH_SUCCESS, , return ge::GRAPH_PARAM_INVALID);
    OP_CHECK_IF(DoOpTilingStage2() != ge::GRAPH_SUCCESS, , return ge::GRAPH_PARAM_INVALID);

    tilingData_.set_r1Dim(r1Dim);
    tilingData_.set_aDim(aDim);
    tilingData_.set_aDimAligned(aDimAligned_);
    tilingData_.set_r0Dim(r0Dim);
    tilingData_.set_usedCoreNums(usedCoreNums_);
    tilingData_.set_r1Inner(r1Inner_);
    tilingData_.set_r1Tail(r1Tail_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradRARSplitCoreR1::DoOpTilingStage0()
{
    int64_t aInner = CONST_ONE;
    int64_t aInnerAligned = blockFp32Nums_;
    int64_t r1InnerInner = CONST_ONE;
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
        OP_LOGE(context_, "BatchNormGrad RAR R1 split core template is not capable. Shape (%ld, %ld, %ld), "
            "factorMax in baseTilingData is %ld .",
            r1Dim, aDim, r0Dim, factorMax),
        return ge::GRAPH_PARAM_INVALID);

    int64_t r0FactorMax = Ops::Base::CeilDiv(r0Dim, dyBaseLen_);
    int64_t r0Factor = factorMax <= r0FactorMax ? factorMax : r0FactorMax;
    int64_t r0Inner = r0Factor * dyBaseLen_;
    int64_t r0Outer = Ops::Base::CeilDiv(r0Dim, r0Inner);
    int64_t r0Tail = r0Dim - (r0Outer - CONST_ONE) * r0Inner;
    int64_t r0TailAligned = Ops::Base::CeilAlign(r0Tail, dyBaseLen_);

    // 切r1
    factorMax = factorMax / r0Factor;
    r1InnerInner = factorMax <= r1Inner_ ? factorMax : r1Inner_;
    int64_t r1InnerOuter = Ops::Base::CeilDiv(r1Inner_, r1InnerInner);
    int64_t r1InnerTail = r1Inner_ - (r1InnerOuter - CONST_ONE) * r1InnerInner;
    int64_t r1TailOuter = Ops::Base::CeilDiv(r1Tail_, r1InnerInner);
    int64_t r1TailTail = r1Tail_ - (r1TailOuter - CONST_ONE) * r1InnerInner;

    int64_t fusedR = r0Outer * r1InnerOuter;
    int64_t lastCoreFusedR = r0Outer * r1TailOuter;

    cacheBufferCount = ComputeBinaryAddParams(fusedR, lastCoreFusedR);

    OP_LOGD(context_,
            "BatchNormGrad R1 split core template Stage0, r0Outer: %ld, r1InnerOuter: %ld, cache buffer count: %ld.",
            r0Outer, r1InnerOuter, cacheBufferCount);

    // 切a, 取（aInner + (blockFp32Nums_ -1)) == aAligned 简化计算
    int64_t aInnerMax =
        (ubSize -
         (blockFp32Nums_ - CONST_ONE) * FLOAT32_BYTES * (DOUBLE_BUFFER * SMALL_SHAPES_STG0 + cacheBufferCount)) /
        (r0Inner * r1InnerInner * (dyDtypeSize_ * DOUBLE_BUFFER * BIG_SHAPES_STG0 + reduceTmpBufNum * FLOAT32_BYTES) +
         (DOUBLE_BUFFER * SMALL_SHAPES_STG0 + cacheBufferCount + 2) * FLOAT32_BYTES);

    aInner = aInnerMax < aDim ? aInnerMax : aDim;
    aInnerAligned = Ops::Base::CeilAlign(aInner, blockFp32Nums_);
    int64_t aOuter = Ops::Base::CeilDiv(aDim, aInner);
    int64_t aTail = aDim - (aOuter - CONST_ONE) * aInner;

    tilingData_.set_r0InnerStg0(r0Inner);
    tilingData_.set_r0OuterStg0(r0Outer);
    tilingData_.set_r0TailStg0(r0Tail);
    tilingData_.set_r0TailAlignedStg0(r0TailAligned);
    tilingData_.set_r1InnerInnerStg0(r1InnerInner);
    tilingData_.set_r1InnerOuterStg0(r1InnerOuter);
    tilingData_.set_r1InnerTailStg0(r1InnerTail);
    tilingData_.set_r1TailOuterStg0(r1TailOuter);
    tilingData_.set_r1TailTailStg0(r1TailTail);
    tilingData_.set_aInnerStg0(aInner);
    tilingData_.set_aInnerAlignedStg0(aInnerAligned);
    tilingData_.set_aOuterStg0(aOuter);
    tilingData_.set_aTailStg0(aTail);

    return ge::GRAPH_SUCCESS;
}

int64_t BatchNormGradRARSplitCoreR1::GetCacheID(const int64_t idx)
{
    return __builtin_popcountll(idx ^ (idx + CONST_ONE)) - CONST_ONE;
}

int64_t BatchNormGradRARSplitCoreR1::ComputeBinaryAddParams(int64_t fusedR, int64_t lastCoreFusedR)
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

    int64_t lastCoreBinAddBasicBlockLoop =
        lastCoreFusedR > 1 ? (1L << (ULONG_BIT_LEN - CONST_ONE - __builtin_clzl(lastCoreFusedR - CONST_ONE))) : 0;
    int64_t lastCoreMainFoldCount = lastCoreFusedR - lastCoreBinAddBasicBlockLoop;
    int64_t lastCoreBinAddResultCacheID = 0;
    if (lastCoreBinAddBasicBlockLoop != 0) {
        lastCoreBinAddResultCacheID = GetCacheID(lastCoreBinAddBasicBlockLoop - CONST_ONE);
    }

    tilingData_.set_binAddBasicBlockLoop(binAddBasicBlockLoop);
    tilingData_.set_binAddMainFoldCount(mainFoldCount);
    tilingData_.set_binAddCacheBufferCount(binAddCacheBufferCount);
    tilingData_.set_binAddResultCacheID(binAddResultCacheID);

    tilingData_.set_lastCoreBinAddBasicBlockLoop(lastCoreBinAddBasicBlockLoop);
    tilingData_.set_lastCoreBinAddMainFoldCount(lastCoreMainFoldCount);
    tilingData_.set_lastCoreBinAddResultCacheID(lastCoreBinAddResultCacheID);

    return binAddCacheBufferCount;
}

ge::graphStatus BatchNormGradRARSplitCoreR1::DoOpTilingStage1()
{
    int64_t weightDtypeSize = weightDtype == ge::DT_FLOAT ? FLOAT32_BYTES : FLOAT16_BYTES;
    int64_t weightBaseLen = weightDtype == ge::DT_FLOAT ? blockFp32Nums_ : blockFp16Nums_;

    // 计算公式: ubSize >= aInnerStg1 * sizeof(float) * usedCoreNums_ * DOUBLE_BUFFER * 2 +
    // aInnerStg1 * DOUBLE_BUFFER * 2 * (sizeof(float) + sizeof(DTYPE_WEIGHT))
    int64_t factorMax = ubSize / (DOUBLE_BUFFER * CONST_TWO * weightBaseLen *
                                  (FLOAT32_BYTES * usedCoreNums_ + (FLOAT32_BYTES + weightDtypeSize)));

    OP_CHECK_IF(factorMax <= 0,
        OP_LOGE(context_, "BatchNormGrad RAR R1 split core template is not capable. Shape (%ld, %ld, %ld), "
            "factorMax in baseTilingData is %ld .",
            r1Dim, aDim, r0Dim, factorMax),
        return ge::GRAPH_PARAM_INVALID);

    int64_t aFactorMax = Ops::Base::CeilDiv(aDim, weightBaseLen);
    int64_t aFactor = factorMax <= aFactorMax ? factorMax : aFactorMax;
    int64_t aInner = aFactor * weightBaseLen;
    int64_t aOuter = Ops::Base::CeilDiv(aDim, aInner);
    int64_t aTail = aDim - (aOuter - CONST_ONE) * aInner;

    OP_LOGD(context_, "BatchNormGrad R1 split core template Stage1, weightBaseLen: %ld, aFactor: %ld, aInner: %ld.",
            weightBaseLen, aFactor, aInner);

    tilingData_.set_aInnerStg1(aInner);
    tilingData_.set_aOuterStg1(aOuter);
    tilingData_.set_aTailStg1(aTail);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradRARSplitCoreR1::DoOpTilingStage2()
{
    int64_t aInner = CONST_ONE;
    int64_t aInnerAligned = blockFp32Nums_;
    int64_t r1InnerInner = CONST_ONE;

    // 计算公式: ubsize >= r0Inner * r1InnerInner * aInner * sizeof(DTYPE_DY) * DOUBLE_BUFFER * 3 + aInnerAligned
    // * sizeof(float) * DOUBLE_BUFFER * 5 切r0
    int64_t factorMax = (ubSize - aInnerAligned * FLOAT32_BYTES * DOUBLE_BUFFER * SMALL_SHAPES_STG2) / dyBaseLen_ /
                        (DOUBLE_BUFFER * FLOAT32_BYTES * BIG_SHAPES_STG2);

    OP_CHECK_IF(factorMax <= 0,
        OP_LOGE(context_, "BatchNormGrad RAR R1 split core template is not capable. Shape (%ld, %ld, %ld), "
            "factorMax in stage2 is %ld .",
            r1Dim, aDim, r0Dim, factorMax),
        return ge::GRAPH_PARAM_INVALID);

    int64_t r0FactorMax = Ops::Base::CeilDiv(r0Dim, dyBaseLen_);
    int64_t r0Factor = factorMax <= r0FactorMax ? factorMax : r0FactorMax;
    int64_t r0Inner = r0Factor * dyBaseLen_;
    int64_t r0Outer = Ops::Base::CeilDiv(r0Dim, r0Inner);
    int64_t r0Tail = r0Dim - (r0Outer - CONST_ONE) * r0Inner;
    int64_t r0TailAligned = Ops::Base::CeilAlign(r0Tail, dyBaseLen_);

    // 切r1
    factorMax = factorMax / r0Factor;
    r1InnerInner = factorMax <= r1Inner_ ? factorMax : r1Inner_;
    int64_t r1InnerOuter = Ops::Base::CeilDiv(r1Inner_, r1InnerInner);
    int64_t r1InnerTail = r1Inner_ - (r1InnerOuter - CONST_ONE) * r1InnerInner;
    int64_t r1TailOuter = Ops::Base::CeilDiv(r1Tail_, r1InnerInner);
    int64_t r1TailTail = r1Tail_ - (r1TailOuter - CONST_ONE) * r1InnerInner;

    // 切a, 取(aInner + (blockFp32Nums_ -1)) == aAligned 简化计算
    int64_t aInnerMax = (ubSize - (blockFp32Nums_ - CONST_ONE) * FLOAT32_BYTES * DOUBLE_BUFFER * SMALL_SHAPES_STG2) /
                        (r0Inner * r1InnerInner * DOUBLE_BUFFER * BIG_SHAPES_STG2 * dyDtypeSize_ +
                         FLOAT32_BYTES * DOUBLE_BUFFER * SMALL_SHAPES_STG2);

    aInner = aInnerMax < aDim ? aInnerMax : aDim;
    aInnerAligned = Ops::Base::CeilAlign(aInner, blockFp32Nums_);
    int64_t aOuter = Ops::Base::CeilDiv(aDim, aInner);
    int64_t aTail = aDim - (aOuter - CONST_ONE) * aInner;

    tilingData_.set_r0InnerStg2(r0Inner);
    tilingData_.set_r0OuterStg2(r0Outer);
    tilingData_.set_r0TailStg2(r0Tail);
    tilingData_.set_r0TailAlignedStg2(r0TailAligned);
    tilingData_.set_r1InnerInnerStg2(r1InnerInner);
    tilingData_.set_r1InnerOuterStg2(r1InnerOuter);
    tilingData_.set_r1InnerTailStg2(r1InnerTail);
    tilingData_.set_r1TailOuterStg2(r1TailOuter);
    tilingData_.set_r1TailTailStg2(r1TailTail);
    tilingData_.set_aInnerStg2(aInner);
    tilingData_.set_aInnerAlignedStg2(aInnerAligned);
    tilingData_.set_aOuterStg2(aOuter);
    tilingData_.set_aTailStg2(aTail);

    return ge::GRAPH_SUCCESS;
}

uint64_t BatchNormGradRARSplitCoreR1::GetTilingKey() const
{
    return BATCH_NORM_GRAD_RAR_SPLIT_CORE_R1_TILING_KEY;
}

ge::graphStatus BatchNormGradRARSplitCoreR1::GetWorkspaceSize()
{
    workspaceSize_ = BNG_WORKSPACE_RESERVED + usedCoreNums_ * aDimAligned_ * FLOAT32_BYTES * CONST_TWO;

    OP_LOGI(context_, "Workspace size: %ld", workspaceSize_);

    size_t* workspaces = context_->GetWorkspaceSizes(CONST_ONE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchNormGradRARSplitCoreR1::PostTiling()
{
    context_->SetBlockDim(usedCoreNums_);
    context_->SetScheduleMode(CONST_ONE);  // Set to batch mode, all cores start simutaneously
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(BatchNormGrad, BatchNormGradRARSplitCoreR1, 2000);

}  // namespace optiling
