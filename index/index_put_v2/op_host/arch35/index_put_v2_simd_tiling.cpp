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
#include "util/platform_util.h"
#include "op_host/tiling_util.h"
#include "index_put_v2_simd_tiling.h"

using namespace AscendC; 

namespace optiling {

constexpr size_t VALUE_IDX = 1;
constexpr size_t INDICES_IDX = 4;
constexpr size_t MAX_DIM = 8;
constexpr size_t INDEXED_SIZES_IDX = 2;
static constexpr int64_t BASE_BLOCK_ALIGN = 512;
static constexpr int64_t SINGLE_CORE_THRESHOLD = 4 * 1024;
static constexpr int64_t NUM_FOUR = 4;
static constexpr uint32_t ASCENDC_TOOLS_WORKSPACE = 16 * 1024 * 1024;

constexpr uint32_t IDX_TYPE_TILING_KEY_WEIGHT = 100;
constexpr uint32_t SIMD_OFFSET = 3000;
constexpr uint32_t ACCU_OFFSET = 1000;
constexpr uint32_t TWO = 2;
constexpr uint64_t DTYPE_UINT8 = 0;
constexpr uint64_t DTYPE_INT8 = 1;
constexpr uint64_t DTYPE_F16 = 2;
constexpr uint64_t DTYPE_BF16 = 3;
constexpr uint64_t DTYPE_INT32 = 4;
constexpr uint64_t DTYPE_F32 = 5;
constexpr uint64_t DTYPE_INT64 = 6;
constexpr uint64_t DTYPE_BOOL = 7;
static std::map<ge::DataType, uint64_t> typeMap =  {{ge::DT_INT64, DTYPE_INT64}, {ge::DT_INT32, DTYPE_INT32}, 
                                            {ge::DT_FLOAT, DTYPE_F32}, {ge::DT_FLOAT16, DTYPE_F16}, 
                                            {ge::DT_BF16, DTYPE_BF16}, {ge::DT_INT8, DTYPE_INT8},
                                            {ge::DT_BOOL, DTYPE_BOOL}, {ge::DT_UINT8, DTYPE_UINT8}};

bool IndexPutV2SimdTiling::IsCapable()
{
    valueType = context_->GetInputDesc(VALUE_IDX)->GetDataType();
    indicesType = context_->GetInputDesc(INDICES_IDX)->GetDataType();
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
    size_t firstZeroPos = indexedSizesNum_;
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

ge::graphStatus IndexPutV2SimdTiling::GetPlatformInfo()
{
    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr =
            reinterpret_cast<const IndexPutV2SimdCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        totalCoreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
        totalCoreNum_ = ascendcPlatform.GetCoreNumAiv();

        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize_ = static_cast<uint64_t>(ubSizePlatform);
    }
    OP_CHECK_IF(
        (totalCoreNum_ <= 0 || ubSize_ <= 0),
        OP_LOGE(
            context_, "coreNum and ubSize should not be smaller than 0, but got coreNum [%ld] and ubSize [%ld]",
            totalCoreNum_, ubSize_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
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

    // 获取索引、索引轴大小
    auto paramIndicesIdx = INDICES_IDX;
    int32_t indicesNum = 0;
    for (size_t i = 0; i < MAX_DIM; ++i) {
        auto curTensor = context_->GetDynamicInputTensor(paramIndicesIdx, i);
        if (curTensor == nullptr) {
            indicesNum = i;
            break;
        }
    }
    if (context_->GetDynamicInputTensor(paramIndicesIdx, 0) != nullptr && indicesNum == 0) {
        indicesNum = MAX_DIM;
    }

    // 获取索引轴的shape及其索引长度
    auto curIndexShape = context_->GetDynamicInputShape(paramIndicesIdx, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, curIndexShape);
    auto const indexShapeVal = curIndexShape->GetStorageShape();
    uint64_t indexLength = indexShapeVal.GetShapeSize();

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
            nonIndexDims_[nonIndexedDimNum_] = inputShapes_[i];
            nonIndexedDimNum_++;
            nonIndexedLength_ *= inputShapes_[i]; 
        }
    }
    indexedDimNum_ = indexedSizesNum_ - nonIndexedDimNum_;

    OP_LOGD("IndexSimd", "indices number: %d, index length: %lu", indicesNum, indexLength);
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

    int64_t inputShapes[MAX_DIM] = {0, 0, 0, 0, 0, 0, 0, 0};
    int64_t indexedStrides[MAX_DIM] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < MAX_DIM; i++) {
        inputShapes[i] = inputShapes_[i];
        indexedStrides[i] = indexedStrides_[i];
    }
    tilingData_.set_inputLength(inputLength_);
    tilingData_.set_valueLength(valueLength_);
    tilingData_.set_inputDimNum(inputDimNum_);
    tilingData_.set_indexedSizesNum(indexedSizesNum_);
    tilingData_.set_indexedDimNum(indexedDimNum_);
    tilingData_.set_nonIndexedDimNum(nonIndexedDimNum_);
    tilingData_.set_accumulateMode(static_cast<int64_t>(accumulateMode_));
    tilingData_.set_indexedLength(indexedLength_);
    tilingData_.set_nonIndexedLength(nonIndexedLength_);
    tilingData_.set_normalCoreRowsNum(normalCoreRowsNum_);
    tilingData_.set_normalCoreColsNum(normalCoreColsNum_);
    tilingData_.set_tailCoreRowsNum(tailCoreRowsNum_);
    tilingData_.set_tailCoreColsNum(tailCoreColsNum_);
    tilingData_.set_blockNumInRow(blockNumInRow_);
    tilingData_.set_blockNumInCol(blockNumInCol_);
    tilingData_.set_rowsFactor(rowsFactor_);
    tilingData_.set_colsFactor(colsFactor_);
    tilingData_.set_coreNum(needCoreNum_);
    tilingData_.set_inputShapes(inputShapes);
    tilingData_.set_indexedStrides(indexedStrides);
    return ge::GRAPH_SUCCESS;
}

void IndexPutV2SimdTiling::GenIndexSimdTilingKey() {
    uint64_t simdKey = 0;
    auto firstInput = context_->GetInputDesc(0);
    auto paramsDtype = firstInput->GetDataType();
    if (typeMap.find(paramsDtype) != typeMap.end()) {
        simdKey = typeMap[paramsDtype];
    } else {
        OP_LOGE("IndexPutV2Simd", "input x dtype error!");
    }
    
    auto paramIndicesIdx = INDICES_IDX;
    auto idxInput = context_->GetInputDesc(paramIndicesIdx);
    auto idxDtype = idxInput->GetDataType();
    if (idxDtype == ge::DT_INT64) {
        simdKey += IDX_TYPE_TILING_KEY_WEIGHT;
    }
    if (accumulateMode_) {
        simdKey += ACCU_OFFSET;
    }
    tilingKey_ = simdKey + SIMD_OFFSET;
    OP_LOGI("IndexPutV2Simd", "tiling key: %lu", tilingKey_);
}

uint64_t IndexPutV2SimdTiling::GetTilingKey() const {
    return tilingKey_;
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
    int64_t availableUbsize =  ubSize_ - MAX_DIM * MAX_DIM * TWO;
    int64_t minRows = 8;
    int64_t doubleB = 2;
    int32_t rankDim = 0;
    int64_t oneRowBuffer = 0;
    uint64_t ubBlockSize = static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_));

    for (int64_t i = 0; i < indexedSizesNum_; i++) {
        if (indexedSizes_[i] == 1) {
            rankDim++;
        }
    }

    minRows = std::min(minRows, normalCoreRowsNum_);
    tmp_buf = minRows * Ops::Base::CeilAlign(normalCoreColsNum_ * valueTypeSize, ubBlockSize) * doubleB
              + Ops::Base::CeilAlign(minRows * indicesTypeSize, ubBlockSize) * (rankDim * doubleB + 1);     // 处理minRows行value需要用到的UB空间

    if (tmp_buf < availableUbsize) {
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

    AutoTilingRowCol(blockNumInRow_, blockNumInCol_, totalCoreNum_, indexedLength_, nonIndexedLength_);

    normalCoreRowsNum_ = Ops::Base::CeilDiv(indexedLength_, blockNumInRow_);
    blockNumInRow_ = Ops::Base::CeilDiv(indexedLength_, normalCoreRowsNum_);
    tailCoreRowsNum_ = indexedLength_ - normalCoreRowsNum_ * (blockNumInRow_ - 1);

    normalCoreColsNum_ = Ops::Base::CeilDiv(nonIndexedLength_, blockNumInCol_);
    blockNumInCol_ = Ops::Base::CeilDiv(nonIndexedLength_, normalCoreColsNum_);
    tailCoreColsNum_ = nonIndexedLength_ - normalCoreColsNum_ * (blockNumInCol_ - 1);

    needCoreNum_ = blockNumInRow_ * blockNumInCol_;

    DoUBTiling();
    SetTilingData();
    GenIndexSimdTilingKey();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexPutV2SimdTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexPutV2SimdTiling::GetWorkspaceSize()
{
    size_t* workspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspace);
    workspace[0] = ASCENDC_TOOLS_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IndexPutV2SimdTiling::PostTiling()
{
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetBlockDim(needCoreNum_);
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(IndexPutV2, IndexPutV2SimdTiling, 10);

} // namespace optiling
