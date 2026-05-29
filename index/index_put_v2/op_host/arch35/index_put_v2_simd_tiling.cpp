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
 * \file index_put_v2_simd_tiling.cpp
 * \brief
 */
#include <algorithm>
#include <cmath>
#include <set>
#include <utility>
#include <vector>
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "log/log.h"
#include "register/op_def_registry.h"
#include "../../op_kernel/arch35/index_put_v2_tiling_key.h"
#include "index_put_v2_simd_tiling.h"

using namespace IndexPutV2;

namespace optiling {
constexpr size_t VALUE_IDX = 1;
constexpr size_t INDICES_IDX = 4;
constexpr size_t MAX_DIM = 8;
constexpr size_t INDEXED_SIZES_IDX = 2;
static constexpr int64_t BASE_BLOCK_ALIGN = 512;
static constexpr int64_t SINGLE_CORE_THRESHOLD = 4 * 1024;
constexpr uint32_t TWO = 2;

bool IndexPutV2SimdTiling::IsCapable()
{
    auto valueType = context_->GetInputDesc(VALUE_IDX)->GetDataType();
    auto indicesType = context_->GetInputDesc(INDICES_IDX)->GetDataType();
    valueTypeSize = ge::GetSizeByDataType(valueType);
    indicesTypeSize = ge::GetSizeByDataType(indicesType);

    bool isContinuous = IsContinuous();
    if (!isContinuous || nonIndexedLength_ * valueTypeSize < 256) {    // 为[1,1,1,1,0,0]且非索引轴长度小于256B
        return false;
    }
    if (!CheckInputDtype()) {
        return false;
    }
    return true;
}

bool IndexPutV2SimdTiling::IsContinuous()
{
    int64_t firstZeroPos = indexedSizesNum_;
    // 找到第一个0值对应的位置
    for (int64_t i = 0; i < indexedSizesNum_; i++) {
        if (indexedSizes_[i] == 0) {
            firstZeroPos = i;
            break;
        }
    }
    // 检查0值前的值是否都为1
    for (int64_t i = 0; i < firstZeroPos; i++) {
        if (indexedSizes_[i] != 1) {
            return false;
        }
    }
    // 检查0值后的值是否都为0
    for (int64_t i = firstZeroPos; i < indexedSizesNum_; i++) {
        if (indexedSizes_[i] != 0) {
            return false;
        }
    }
    return true;
}

bool IndexPutV2SimdTiling::CheckInputDtype()
{
    std::set<ge::DataType> supportType = {ge::DT_BOOL, ge::DT_INT8, ge::DT_UINT8, ge::DT_FLOAT16,
                                          ge::DT_BF16, ge::DT_INT32, ge::DT_FLOAT, ge::DT_INT64};
    std::set<ge::DataType> atomicAddSupportType = {ge::DT_INT8, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32, ge::DT_FLOAT};
    auto const attrs = context_->GetAttrs();
    auto* accumulateMode = attrs->GetAttrPointer<bool>(0);
    auto inputDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    auto inputDtype = inputDesc->GetDataType();
    // accu为true，此时要检查数据类型, 累加场景支持五种数据类型
    if (*accumulateMode) {
        if (atomicAddSupportType.find(inputDtype) == atomicAddSupportType.end()) {
            return false;
        }
    } else {
        if (supportType.find(inputDtype) == supportType.end()) {
            return false;
        }
    }
    return true;
}

ge::graphStatus IndexPutV2SimdTiling::GetShapeAttrsInfo()
{
    OP_LOGD("IndexPutV2", "IndexPutV2SimdTiling is running");

    // 获取输入x的shape及其对应维度
    auto const inputShape = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
    auto const inShapeVal = inputShape->GetStorageShape();
    inputLength_ = inShapeVal.GetShapeSize();
    const size_t inputRank = inShapeVal.GetDimNum();
    inputDimNum_ = inputRank;
    for (size_t i = 0; i < inputRank; ++i) {
        inputShapes_[i] = inShapeVal.GetDim(i);
    }
    OP_LOGD("IndexPutV2Simd", "input dim Num: %u", inputDimNum_);

    // 获取输入value的shape
    auto const valueSize = context_->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, valueSize);
    auto const valueShapeVal = valueSize->GetStorageShape();
    valueLength_ = valueShapeVal.GetShapeSize();
    OP_LOGD("IndexPutV2Simd", "valueLength_: %lu", valueLength_);

    // 获取输入indexedSizes的shape
    auto const indexedSizes = context_->GetInputShape(INDEXED_SIZES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indexedSizes);
    auto const indexedSizesShape = indexedSizes->GetShape();
    indexedSizesNum_ = indexedSizesShape.GetDim(0);
    OP_LOGI("IndexPutV2Simd", "input indexed_sizes size: %ld", indexedSizesNum_);

    // 获取indexedSize的值
    const gert::Tensor* mask_tensor = context_->GetInputTensor(INDEXED_SIZES_IDX);
    const int64_t* mask_arr = mask_tensor->GetData<int64_t>();
    for (int64_t i = 0; i < indexedSizesNum_; i++) {
        indexedSizes_[i] = mask_arr[i];
    }

    // 计算输入张量各维度步长
    indexedStrides_[indexedSizesNum_ - 1] = 1;
    for (int64_t i = static_cast<int64_t>(indexedSizesNum_) - 2; i >= 0; i--) {
        indexedStrides_[i] = inputShapes_[i + 1] * indexedStrides_[i + 1];
    }

    // value的n为非索引轴合轴的维度乘积
    for (int i = 0; i < indexedSizesNum_; i++) {
        if (indexedSizes_[i] == 0) {
            nonIndexedDimNum_++;
            nonIndexedLength_ *= inputShapes_[i];
        }
    }
    indexedDimNum_ = indexedSizesNum_ - nonIndexedDimNum_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexPutV2SimdTiling::SetTilingData()
{
    auto const attrs = context_->GetAttrs();
    auto* accumulateMode = attrs->GetAttrPointer<bool>(0);
    if (*accumulateMode) {
        accumulateMode_ = 1;
    } else {
        accumulateMode_ = 0;
    }

    for (size_t i = 0; i < MAX_DIM; i++) {
        tilingData_->inputShapes[i] = inputShapes_[i];
        tilingData_->indexedStrides[i] = indexedStrides_[i];
    }
    tilingData_->inputLength = inputLength_;
    tilingData_->valueLength = valueLength_;
    tilingData_->inputDimNum = inputDimNum_;
    tilingData_->indexedSizesNum = indexedSizesNum_;
    tilingData_->indexedDimNum = indexedDimNum_;
    tilingData_->nonIndexedDimNum = nonIndexedDimNum_;
    tilingData_->accumulateMode = static_cast<int64_t>(accumulateMode_);
    tilingData_->indexedLength = indexedLength_;
    tilingData_->nonIndexedLength = nonIndexedLength_;
    tilingData_->normalCoreRowsNum = normalCoreRowsNum_;
    tilingData_->normalCoreColsNum = normalCoreColsNum_;
    tilingData_->tailCoreRowsNum = tailCoreRowsNum_;
    tilingData_->tailCoreColsNum = tailCoreColsNum_;
    tilingData_->blockNumInRow = blockNumInRow_;
    tilingData_->blockNumInCol = blockNumInCol_;
    tilingData_->rowsFactor = rowsFactor_;
    tilingData_->colsFactor = colsFactor_;
    tilingData_->coreNum = needCoreNum_;
    return ge::GRAPH_SUCCESS;
}

uint64_t IndexPutV2SimdTiling::GetTilingKey() const {
    return GET_TPL_TILING_KEY(0, 0, 0, 1, 0, static_cast<uint64_t>(accumulateMode_), 0);
}

std::set<int64_t> IndexListFactors(int64_t usedCoreNum)
{
    std::set<int64_t> result;
    int64_t upbound = std::ceil(std::sqrt(usedCoreNum) + 1);

    for (int64_t m = 1; m < upbound; m++) {
        int64_t y = static_cast<int64_t>(usedCoreNum) / m;
        result.insert(m);
        result.insert(y);
    }
    return result;
}

void IndexPutV2SimdTiling::AutoTilingRowCol(int64_t& rowTileNum, int64_t& colTileNum, int64_t usedCoreNum, int64_t rowTotalNum, int64_t colTotalNum)
{
    int64_t tmpEleNum = BASE_BLOCK_ALIGN / valueTypeSize;
    int64_t colBlockTotalNum = (colTotalNum + tmpEleNum - 1) / tmpEleNum;
    usedCoreNum = std::min(usedCoreNum, std::max(int64_t(1), rowTotalNum * colBlockTotalNum * tmpEleNum / (SINGLE_CORE_THRESHOLD)));

    std::set<int64_t> cutSet = IndexListFactors(usedCoreNum);
    std::vector<std::vector<int64_t>> allTiling;

    for (int64_t m : cutSet) {
        if (m > rowTotalNum) {
            continue;
        }

        int64_t n = usedCoreNum / m;
        n = n < 1 ? 1 : n;
        if (n > colBlockTotalNum) {
            continue;
        }

        int64_t rowNormalBlock = Ops::Base::CeilDiv(rowTotalNum, m);
        int64_t mReal = Ops::Base::CeilDiv(rowTotalNum, rowNormalBlock);
        int64_t rowTailBlock = rowTotalNum - (mReal - 1) * rowNormalBlock;

        int64_t colNormalBlock = Ops::Base::CeilDiv(colBlockTotalNum, n);
        int64_t nReal = Ops::Base::CeilDiv(colBlockTotalNum, colNormalBlock);
        int64_t colTailBlock = colBlockTotalNum - (nReal - 1) * colNormalBlock;

        // m、n符合要求且尾块和正常块的大小尽可能接近
        int64_t blockNormal = rowNormalBlock * colNormalBlock;
        int64_t blockTail = rowTailBlock * colTailBlock;
        int64_t delta = blockNormal - blockTail;
        allTiling.push_back({m, n, m * n, delta});
    }

    // 排序逻辑：1、少切n；2、delta尽可能小
    std::sort(allTiling.begin(), allTiling.end(), [](const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
        constexpr int NIndex = 1;
        constexpr int DeltaIndex = 3;
        return std::make_pair(a[NIndex], a[DeltaIndex]) < std::make_pair(b[NIndex], b[DeltaIndex]);
    });

    rowTileNum = static_cast<uint16_t>(allTiling[0][0]);
    colTileNum = static_cast<uint16_t>(allTiling[0][1]);
}

// 核间切UB
void IndexPutV2SimdTiling::DoUBTiling()
{
    uint64_t availableUbsize =  ubSize_ - MAX_DIM * MAX_DIM * TWO;
    int64_t minRows = 8;
    int64_t doubleB = 2;
    int32_t rankDim = 0;
    uint64_t oneRowBuffer = 0;
    uint64_t ubBlockSize = static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_));

    for (int64_t i = 0; i < indexedSizesNum_; i++) {
        if (indexedSizes_[i] == 1) {
            rankDim++;
        }
    }

    minRows = std::min(minRows, normalCoreRowsNum_);
    uint64_t tmpBuf = minRows * Ops::Base::CeilAlign(normalCoreColsNum_ * valueTypeSize, ubBlockSize) * doubleB
                      + Ops::Base::CeilAlign(minRows * indicesTypeSize, ubBlockSize) * (rankDim * doubleB + 1);

    if (tmpBuf < availableUbsize) {
        colsFactor_ = normalCoreColsNum_;
        oneRowBuffer = Ops::Base::CeilAlign(normalCoreColsNum_ * valueTypeSize, ubBlockSize) * doubleB
                     + Ops::Base::CeilAlign(indicesTypeSize, ubBlockSize) * (rankDim * doubleB + 1);
        rowsFactor_ = availableUbsize / oneRowBuffer;
        rowsFactor_ = std::min(rowsFactor_, normalCoreRowsNum_);
    } else {
        rowsFactor_ = minRows;
        colsFactor_ = (availableUbsize - Ops::Base::CeilAlign(rowsFactor_ * indicesTypeSize, ubBlockSize) * (rankDim * doubleB + 1)) / rowsFactor_ / doubleB / valueTypeSize;
        colsFactor_ = Ops::Base::FloorAlign(colsFactor_, 128 / static_cast<int64_t>(valueTypeSize));
        colsFactor_ = std::min(colsFactor_, normalCoreColsNum_);
    }
}

ge::graphStatus IndexPutV2SimdTiling::DoOpTiling()
{
    auto curIndexShape = context_->GetDynamicInputShape(INDICES_IDX, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, curIndexShape);
    auto const indexSizeVal = curIndexShape->GetStorageShape();
    indexedLength_ = indexSizeVal.GetShapeSize();

    AutoTilingRowCol(blockNumInRow_, blockNumInCol_, coreNum_, indexedLength_, nonIndexedLength_);

    normalCoreRowsNum_ = Ops::Base::CeilDiv(indexedLength_, blockNumInRow_);
    blockNumInRow_ = Ops::Base::CeilDiv(indexedLength_, normalCoreRowsNum_);
    tailCoreRowsNum_ = indexedLength_ - normalCoreRowsNum_ * (blockNumInRow_ - 1);

    normalCoreColsNum_ = Ops::Base::CeilDiv(nonIndexedLength_, blockNumInCol_);
    blockNumInCol_ = Ops::Base::CeilDiv(nonIndexedLength_, normalCoreColsNum_);
    tailCoreColsNum_ = nonIndexedLength_ - normalCoreColsNum_ * (blockNumInCol_ - 1);

    needCoreNum_ = blockNumInRow_ * blockNumInCol_;

    tilingData_ = context_->GetTilingData<IndexPutV2SimdTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData_);
    OP_CHECK_IF(
    memset_s(tilingData_, sizeof(IndexPutV2SimdTilingData), 0, sizeof(IndexPutV2SimdTilingData)) != EOK,
    OP_LOGE(context_->GetNodeName(), "set tiling data error"), return ge::GRAPH_FAILED);

    DoUBTiling();
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexPutV2SimdTiling::PostTiling()
{
    context_->SetBlockDim(needCoreNum_);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(IndexPutV2, IndexPutV2SimdTiling, 10);
} // namespace optiling
