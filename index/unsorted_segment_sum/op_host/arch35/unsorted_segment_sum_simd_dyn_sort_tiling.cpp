/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file unsorted_segment_sum_simd_dyn_sort_tiling.cpp
 * \brief unsorted_segment_sum_simd_non_sort_tiling
 */

#include "unsorted_segment_sum_simd_dyn_sort_tiling.h"
#include "util/platform_util.h"
#include "util/math_util.h"



#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"

namespace optiling {
static constexpr uint64_t TEMPLATE_SIMD_DYN_SORT = 7000;
static constexpr uint64_t LAST_DIM_SIMD_COND = 128;
static constexpr uint64_t BUFFER_NUM = 1;
static constexpr uint64_t SIMD_RESERVED_SIZE = 8192;
static constexpr uint64_t BASE_A_SIZE = 512;
static constexpr uint64_t BASE_BLOCK_SIZE = 8192;
static constexpr uint64_t COL_LIMIT_SIZE = 1024;
static constexpr uint64_t DIVISOR_TWO = 2;
static constexpr uint32_t SORT_STAT_PADDING = 64;

static const std::set<ge::DataType> setAtomicNotSupport = {ge::DT_UINT32, ge::DT_INT64, ge::DT_UINT64};

bool UnsortedSegmentSumSimdDynSortTiling::IsCapable()
{
    if (innerDim_ * valueTypeBytes_ >= LAST_DIM_SIMD_COND &&
        setAtomicNotSupport.find(dataType_) == setAtomicNotSupport.end()) {
        return true;
    }
    return false;
}

uint64_t UnsortedSegmentSumSimdDynSortTiling::GetTilingKey() const
{
    uint64_t tilingKey = TEMPLATE_SIMD_DYN_SORT;
    return tilingKey;
}

void UnsortedSegmentSumSimdDynSortTiling::SetTilingData()
{
    tilingData_.set_outputOuterDim(outputOuterDim_);
    tilingData_.set_innerDim(innerDim_);
    tilingData_.set_sTileNum(sTileNum_);
    tilingData_.set_aTileNum(aTileNum_);
    tilingData_.set_normBlockS(normBlockS_);
    tilingData_.set_tailBlockS(tailBlockS_);
    tilingData_.set_normBlockA(normBlockA_);
    tilingData_.set_tailBlockA(tailBlockA_);
    tilingData_.set_baseS(baseS_);
    tilingData_.set_baseA(baseA_);
    tilingData_.set_sortBaseS(sortBaseS_);
    tilingData_.set_sortBaseA(sortBaseA_);
    tilingData_.set_sortSharedBufSize(static_cast<uint64_t>(sortSharedBufSize_));
}

void UnsortedSegmentSumSimdDynSortTiling::DoBlockTiling()
{
    uint64_t colNumAlign = Ops::Base::CeilDiv(innerDim_, baseA_);
    usedCoreNum_ =
        std::min(totalCoreNum_, static_cast<uint64_t>(inputOuterDim_ * colNumAlign * BASE_A_SIZE / BASE_BLOCK_SIZE));
    usedCoreNum_ = usedCoreNum_ == 0 ? 1 : usedCoreNum_;
    std::tie(sTileNum_, aTileNum_) = AutoTiling(usedCoreNum_, colNumAlign, COL_LIMIT_SIZE);

    normBlockS_ = Ops::Base::CeilDiv(inputOuterDim_, sTileNum_);
    sTileNum_ = Ops::Base::CeilDiv(inputOuterDim_, normBlockS_);

    normBlockA_ = Ops::Base::CeilDiv(innerDim_, aTileNum_);
    normBlockA_ = Ops::Base::CeilAlign(normBlockA_, ubBlockSize_ / valueTypeBytes_);
    aTileNum_ = Ops::Base::CeilDiv(innerDim_, normBlockA_);

    usedCoreNum_ = sTileNum_ * aTileNum_;

    tailBlockS_ = inputOuterDim_ - (sTileNum_ - 1UL) * normBlockS_;
    tailBlockA_ = innerDim_ - (aTileNum_ - 1UL) * normBlockA_;
}

/**
 * @brief Find best baseSize in range [baseXoStart, baseXoEnd], use dichotomy algorithm.
 */
uint64_t UnsortedSegmentSumSimdDynSortTiling::CalBestBaseSize(uint64_t baseXoStart, uint64_t baseXoEnd)
{
    uint64_t idsSortBufCnt = 2;
    uint64_t baseXoMid;
    uint64_t tmpTotalSize = 0;
    baseXoEnd = baseXoEnd + 1UL;
    while (baseXoEnd - baseXoStart > 1UL) {
        baseXoMid = (baseXoStart + baseXoEnd) / DIVISOR_TWO;
        uint64_t sortNeedTmpSize = static_cast<uint64_t>(GetSortTmpSize(baseXoMid, false));
        tmpTotalSize = Ops::Base::CeilAlign(baseXoMid * sortBaseA_ * valueTypeBytes_, ubBlockSize_) * BUFFER_NUM + // xQue
                       Ops::Base::CeilAlign(baseXoMid * idTypeBytes_, ubBlockSize_) * BUFFER_NUM +                 // idQue
                       Ops::Base::CeilAlign(baseXoMid * sortBaseA_ * valueTypeBytes_, ubBlockSize_) +              // resBuf
                       Ops::Base::CeilAlign(baseXoMid * idTypeBytes_, ubBlockSize_) +                     // sortedkeyBuf
                       Ops::Base::CeilAlign(baseXoMid * sizeof(uint32_t), ubBlockSize_) * idsSortBufCnt + // sortedIdxBuf
                       SORT_STAT_PADDING + SORT_STAT_PADDING +                                      // sort padding
                       Ops::Base::CeilAlign(sortNeedTmpSize, ubBlockSize_); // sort shared buf size
        if (tmpTotalSize <= ubSize_) {
            baseXoStart = baseXoMid;
        } else {
            baseXoEnd = baseXoMid;
        }
    }
    return baseXoStart;
}

ge::graphStatus UnsortedSegmentSumSimdDynSortTiling::DoOpTiling()
{
    baseA_ = BASE_A_SIZE / valueTypeBytes_;
    DoBlockTiling();

    if (normBlockA_ * valueTypeBytes_ <= COL_LIMIT_SIZE) {
        baseA_ = normBlockA_;
    }
    sortBaseA_ = baseA_;
    ubSize_ -= SIMD_RESERVED_SIZE;
    uint64_t coreMaxS = std::max(normBlockS_, tailBlockS_);

    // ub split for non sort case
    uint64_t maxBaseS = (ubSize_ - BUFFER_NUM * ubBlockSize_) / BUFFER_NUM / (baseA_ * valueTypeBytes_ + idTypeBytes_);
    baseS_ = maxBaseS;
    if (coreMaxS < baseS_) {
        baseS_ = coreMaxS;
        baseA_ =
            (ubSize_ - BUFFER_NUM * (ubBlockSize_ + baseS_ * idTypeBytes_)) / BUFFER_NUM / (baseS_ * valueTypeBytes_);
        baseA_ = Ops::Base::FloorAlign(baseA_, ubBlockSize_ / valueTypeBytes_);
    }

    // ub split for sort case
    sortBaseS_ = CalBestBaseSize(1UL, maxBaseS);
    if (coreMaxS < sortBaseS_) {
        sortBaseS_ = coreMaxS;
        uint64_t idsSortBufCnt = 2;
        uint64_t sortNeedTmpSize = static_cast<uint64_t>(GetSortTmpSize(sortBaseS_, false));
        uint64_t remainSize = ubSize_ - Ops::Base::CeilAlign(sortBaseS_ * idTypeBytes_, ubBlockSize_) * BUFFER_NUM -
                              Ops::Base::CeilAlign(sortBaseS_ * idTypeBytes_, ubBlockSize_) -
                              Ops::Base::CeilAlign(sortBaseS_ * sizeof(uint32_t), ubBlockSize_) * idsSortBufCnt -
                              Ops::Base::CeilAlign(sortNeedTmpSize, ubBlockSize_) - SORT_STAT_PADDING - SORT_STAT_PADDING;
        sortBaseA_ =
            (remainSize - BUFFER_NUM * ubBlockSize_ - ubBlockSize_) / (BUFFER_NUM + 1) / (sortBaseS_ * valueTypeBytes_);
        sortBaseA_ = Ops::Base::FloorAlign(sortBaseA_, ubBlockSize_ / valueTypeBytes_);
    }
    sortSharedBufSize_ = GetSortTmpSize(sortBaseS_, false);
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UnsortedSegmentSumSimdDynSortTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    context_->SetScheduleMode(1);
    if (tilingData_.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void UnsortedSegmentSumSimdDynSortTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "tilingKey: " << GetTilingKey();
    info << ", usedCoreNum: " << usedCoreNum_;
    info << ", inputOuterDim: " << inputOuterDim_;
    info << ", outputOuterDim: " << tilingData_.get_outputOuterDim();
    info << ", innerDim: " << tilingData_.get_innerDim();
    info << ", sTileNum: " << tilingData_.get_sTileNum();
    info << ", aTileNum: " << tilingData_.get_aTileNum();
    info << ", normBlockS: " << tilingData_.get_normBlockS();
    info << ", tailBlockS: " << tilingData_.get_tailBlockS();
    info << ", normBlockA: " << tilingData_.get_normBlockA();
    info << ", tailBlockA: " << tilingData_.get_tailBlockA();
    info << ", baseS: " << tilingData_.get_baseS();
    info << ", baseA: " << tilingData_.get_baseA();
    info << ", sortBaseS: " << tilingData_.get_sortBaseS();
    info << ", sortBaseA: " << tilingData_.get_sortBaseA();
    info << ", sortSharedBufSize: " << tilingData_.get_sortSharedBufSize();
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

REGISTER_OPS_TILING_TEMPLATE(UnsortedSegmentSum, UnsortedSegmentSumSimdDynSortTiling, 50);

} // namespace optiling