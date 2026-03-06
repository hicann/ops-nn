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
 * \file unsorted_segment_sum_tiling_base.cpp
 * \brief unsorted_segment_sum_tiling_base
 */

#include <vector>
#include "unsorted_segment_sum_tiling.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "util/platform_util.h"
#include "util/math_util.h"
#include "unsorted_segment_sum_tiling_arch35.h"

using namespace AscendC;
using namespace ge;

namespace optiling {
static constexpr uint32_t INPUT_DATA_INDEX = 0;
static constexpr uint32_t INPUT_SEGMENT_IDS_INDEX = 1;
static constexpr uint32_t INPUT_NUM_SEGMENTS_INDEX = 2;
static constexpr uint32_t OUTPUT_DATA_INDEX = 0;
static constexpr uint32_t ASCENDC_TOOLS_WORKSPACE = static_cast<uint32_t>(16 * 1024 * 1024);

static const std::map<ge::DataType, uint32_t> dataTypeMap = {{ge::DT_FLOAT, 0}, {ge::DT_FLOAT16, 1}, {ge::DT_BF16, 2},
                                                             {ge::DT_INT32, 3}, {ge::DT_INT64, 4},   {ge::DT_UINT32, 5},
                                                             {ge::DT_UINT64, 6}};

static const std::map<ge::DataType, uint32_t> indexTypeMap = {
    {ge::DT_INT32, 100},
    {ge::DT_INT64, 200},
};

ge::graphStatus UnsortedSegmentSumBaseTiling::GetPlatformInfo()
{
    auto compileInfo =
        static_cast<const TilingPrepareForUnsortedSegmentSumCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    totalCoreNum_ = compileInfo->core_num;
    maxThread_ = compileInfo->max_thread;
    return ge::GRAPH_SUCCESS;
}

bool UnsortedSegmentSumBaseTiling::ShapeStartsWith(const gert::Shape shape, const gert::Shape prefix)
{
    if (shape.GetDimNum() < prefix.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < prefix.GetDimNum(); ++i) {
        if (shape[i] != prefix[i]) {
            return false;
        }
    }
    return true;
}

std::tuple<int64_t, int64_t> UnsortedSegmentSumBaseTiling::FlatInput(const gert::Shape shape, const gert::Shape prefix)
{
    int64_t outer{1};
    int64_t inner{1};
    for (size_t i = 0; i < prefix.GetDimNum(); ++i) {
        outer *= prefix[i];
    }
    for (size_t i = prefix.GetDimNum(); i < shape.GetDimNum(); ++i) {
        inner *= shape[i];
    }
    return std::make_tuple(outer, inner);
}

uint32_t UnsortedSegmentSumBaseTiling::GetSortTmpSize(uint32_t lastAxisNum, bool isDescend)
{
    std::vector<int64_t> shapeVec = {lastAxisNum};
    ge::Shape srcShape(shapeVec);
    AscendC::SortConfig config;
    config.type = AscendC::SortType::RADIX_SORT;
    config.isDescend = isDescend;
    config.hasSrcIndex = false;
    config.hasDstIndex = true;
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetSortMaxMinTmpSize(srcShape, idType_, ge::DT_UINT32, false, config, maxValue, minValue);
    return maxValue;
}

ge::graphStatus UnsortedSegmentSumBaseTiling::CheckInputDtype()
{
    auto dataDtypePtr = context_->GetInputDesc(INPUT_DATA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dataDtypePtr);
    auto segmentIdsDtypePtr = context_->GetInputDesc(INPUT_SEGMENT_IDS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, segmentIdsDtypePtr);

    auto dataTypeIter = dataTypeMap.find(dataDtypePtr->GetDataType());
    auto indexTypeIter = indexTypeMap.find(segmentIdsDtypePtr->GetDataType());
    OP_CHECK_IF(
        dataTypeIter == dataTypeMap.end(),
        OP_LOGE(context_->GetNodeName(), "Not support data's type!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        indexTypeIter == indexTypeMap.end(),
        OP_LOGE(context_->GetNodeName(), "Not support segment_ids's type"),
        return ge::GRAPH_FAILED);

    dataType_ = dataDtypePtr->GetDataType();
    idType_ = segmentIdsDtypePtr->GetDataType();

    valueTypeBytes_ = ge::GetSizeByDataType(dataType_);
    idTypeBytes_ = ge::GetSizeByDataType(idType_);
    OP_CHECK_IF(
        valueTypeBytes_ <= 0UL, OP_LOGE(context_->GetNodeName(), "get dataType size fail."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        idTypeBytes_ <= 0UL, OP_LOGE(context_->GetNodeName(), "get idType size fail."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UnsortedSegmentSumBaseTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(
        CheckInputDtype() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "input dtype check failed."), return ge::GRAPH_FAILED);

    auto dataShapePtr = context_->GetInputShape(INPUT_DATA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dataShapePtr);
    auto dataShape = dataShapePtr->GetStorageShape();

    auto segmentIdsShapePtr = context_->GetInputShape(INPUT_SEGMENT_IDS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dataShapePtr);
    auto segmentIdsShape = segmentIdsShapePtr->GetStorageShape();

    auto numSegmentsShapePtr = context_->GetInputShape(INPUT_NUM_SEGMENTS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, numSegmentsShapePtr);
    auto& numSegmentsShape = Ops::Base::EnsureNotScalar(numSegmentsShapePtr->GetStorageShape());

    auto outputShapePtr = context_->GetOutputShape(OUTPUT_DATA_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShapePtr);
    auto outputShape = outputShapePtr->GetStorageShape();

    OP_CHECK_IF(
        numSegmentsShape.GetDimNum() != 1,
        OP_LOGE(context_->GetNodeName(), "Num_segments should be one dim shape!"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        !ShapeStartsWith(dataShape, segmentIdsShape),
        OP_LOGE(
            context_->GetNodeName(), "Data.shape does not start with segment_ids.shape"),
        return ge::GRAPH_FAILED);

    std::tie(inputOuterDim_, innerDim_) = FlatInput(dataShape, segmentIdsShape);
    // check key value not greater than 2^48.
    uint64_t bound = static_cast<int64_t>(1ULL << 48);
    OP_CHECK_IF(
        inputOuterDim_ > bound,
        OP_LOGE(context_->GetNodeName(), "InputOuterDim out of 2^48!"),
        return ge::GRAPH_FAILED);

    dataShapeSize_ = dataShape.GetShapeSize();
    outputOuterDim_ = outputShape[0];
    ubBlockSize_ = Ops::Base::GetUbBlockSize(context_);
    innerDimAlign_ = Ops::Base::CeilAlign(innerDim_ * valueTypeBytes_, ubBlockSize_) / valueTypeBytes_;
    ratio_ = inputOuterDim_ / outputOuterDim_;
    return ge::GRAPH_SUCCESS;
}

std::tuple<uint64_t, uint64_t> UnsortedSegmentSumBaseTiling::AutoTiling(
    uint64_t usedCoreNum, uint64_t colNumAlign, uint64_t colLimitSize, bool colTileNumMin)
{
    /* 给定Shape[M, N] 和 block core num
    ** M切分成m块，N切分成n块，找到m*n <= usedCoreNum, 且m*n尽可能接近usedCoreNum的所有m和n的可能
    */
    std::set<uint64_t> cutSet = FindUniqueCut(usedCoreNum);

    std::vector<std::vector<uint64_t>> allTiling;

    // 核先按照行方向切分，枚举m的取值
    for (uint64_t m : cutSet) {
        // 行方向分核超过行方向数据量，则说明有空闲核
        if (m > inputOuterDim_) {
            continue;
        }

        uint64_t n = usedCoreNum / m;
        n = n < 1 ? 1 : n;
        // 列方向分核超过列方向数据量，则说明有空闲核
        if (n > colNumAlign) {
            continue;
        }

        // ub重排模板，不会在列方向切分
        if (innerDim_ * valueTypeBytes_ <= colLimitSize && n > 1) {
            continue;
        }
        // 行切分
        uint64_t rowNormalBlock = Ops::Base::CeilDiv(inputOuterDim_, m);
        uint64_t mReal = Ops::Base::CeilDiv(inputOuterDim_, rowNormalBlock);
        uint64_t rowTailBlock = inputOuterDim_ - (mReal - 1) * rowNormalBlock;

        // 列方向按照BASE_A_SIZE基本块切分
        uint64_t colNormalBlock = Ops::Base::CeilDiv(colNumAlign, n);
        uint64_t nReal = Ops::Base::CeilDiv(colNumAlign, colNormalBlock);
        uint64_t colTailBlock = colNumAlign - (nReal - 1) * colNormalBlock;

        uint64_t blockNormal = rowNormalBlock * colNormalBlock;
        uint64_t blockTail = rowTailBlock * colTailBlock;
        uint64_t delta = blockNormal - blockTail;

        allTiling.push_back({m, n, m * n, delta});
    }

    if (colTileNumMin) {
        std::sort(
            allTiling.begin(), allTiling.end(), [](const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
                constexpr int NIndex = 1;
                constexpr int DeltaIndex = 3;
                return std::make_pair(a[DeltaIndex], a[NIndex]) < std::make_pair(b[DeltaIndex], b[NIndex]);
            });
    } else {
        std::sort(
            allTiling.begin(), allTiling.end(), [](const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
                constexpr int MIndex = 0;
                constexpr int DeltaIndex = 3;
                return std::make_pair(a[DeltaIndex], a[MIndex]) < std::make_pair(b[DeltaIndex], b[MIndex]);
            });
    }

    uint64_t rowTileNum = allTiling[0][0];
    uint64_t colTileNum = allTiling[0][1];
    return std::make_tuple(rowTileNum, colTileNum);
}

std::set<uint64_t> UnsortedSegmentSumBaseTiling::FindUniqueCut(uint64_t usedCoreNum)
{
    std::set<uint64_t> result;
    uint64_t upbound = std::ceil(std::sqrt(usedCoreNum) + 1);

    for (uint64_t m = 1; m < upbound; m++) {
        uint64_t y = usedCoreNum / m;
        result.insert(m);
        result.insert(y);
    }
    return result;
}

bool UnsortedSegmentSumBaseTiling::IsCapable()
{
    return true;
}

ge::graphStatus UnsortedSegmentSumBaseTiling::DoOpTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UnsortedSegmentSumBaseTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t UnsortedSegmentSumBaseTiling::GetTilingKey() const
{
    return 0;
}

ge::graphStatus UnsortedSegmentSumBaseTiling::GetWorkspaceSize()
{
    uint32_t sysWorkspace = ASCENDC_TOOLS_WORKSPACE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sysWorkspace + usrWorkspaceSize_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UnsortedSegmentSumBaseTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling