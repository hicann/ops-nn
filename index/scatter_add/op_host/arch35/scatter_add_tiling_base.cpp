/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_add_tiling_base.cpp
 * \brief
 */

#include "scatter_add_tiling_base.h"
#include <vector>
#include "platform/platform_info.h"
#include "scatter_add_tiling.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_common/op_host/util/math_util.h"
#include "tiling_base/tiling_key.h"

using namespace AscendC;

namespace optiling
{
constexpr int64_t VAR_IDX = 0;
constexpr int64_t INDICES_IDX = 1;
constexpr int64_t UPDATES_IDX = 2;
constexpr uint16_t OUTPUT_IDX = 0;

constexpr uint64_t BUFFER_NUM = 2;
constexpr uint64_t BUFFER_NUM_SIMD_SORT = 1;
constexpr int64_t USE_UB_MAX_SIZE = static_cast<int64_t>(64) * 1024;
constexpr int64_t MIN_COL_SIZE = 256;
constexpr uint64_t MIN_BLOCK_SIZE = 1024;
constexpr uint64_t DCACHE_SIZE = static_cast<uint64_t>(128) * 1024;
constexpr uint64_t DCACHE_SIZE1 = static_cast<uint64_t>(32) * 1024;
constexpr int64_t ASCENDC_TOOLS_WORKSPACE = static_cast<int64_t>(16) * 1024 * 1024;
constexpr uint64_t UPDATES_IN_SIMD = 1;
constexpr uint64_t NON_SORT_NON_ATOMIC_SIZE = static_cast<uint64_t>(64) * 128;
constexpr uint64_t GM_ALIGN = 512;
constexpr uint64_t MIN_SIZE_SORT_INDICES_128 = 128;
constexpr uint64_t INT32_TYPE_SIZE = 4;
constexpr uint64_t VAR_TAIL_DIM_SIZE = 128;
constexpr uint32_t TILING_KEY_PLACE_HOLD = 3;
constexpr size_t VAR_SHAPE_LENGTH = 2;
constexpr uint32_t SORT_UTIL_SIZE = 20;
constexpr int32_t UB_AGLIN_VALUE = 32;
constexpr uint32_t FLOAT_BYTES = 4;
constexpr uint32_t UINT32_BYTES = 4;
constexpr uint32_t DOUBLE = 2;
constexpr uint32_t THREE = 3;
constexpr uint32_t TEN = 10;
constexpr uint64_t SIMD_RESERVED_SIZE = static_cast<uint64_t>(8) * 1024;
static constexpr uint64_t BASE_A_SIZE = 512;
static constexpr uint64_t BASE_BLOCK_SIZE = 1024;
static constexpr uint64_t COL_LIMIT_SIZE = 1024;
static constexpr uint32_t SORT_STAT_PADDING = 64;
static constexpr uint64_t CASTMODE1 = 1;   // int32 Cast int16
static constexpr uint64_t CASTMODE2 = 2;   // int64 Cast int32
static constexpr uint64_t CASTMODE3 = 3;   // int64 Cast int16
static constexpr uint64_t CASTMODE4 = 4;   // int32 Cast uint8
static constexpr uint64_t CASTMODE5 = 5;   // int64 Cast uint8
static constexpr uint64_t DETERMIN_THRESHOLD_MID = 128;
static constexpr uint64_t DETERMIN_THRESHOLD_BOT = 16;
static constexpr float SCALE_FACOTR = 0.8;
static constexpr uint64_t INDICES_THRESHOLD = 4096;
static constexpr uint64_t VAR_LASTDIM_10W = 100000;

static const std::unordered_map<ge::DataType, uint64_t> INDICE_DATA_TYPE_TO_INT{{ge::DataType::DT_INT32, 1},
                                                                                {ge::DataType::DT_INT64, 2}};

static const std::unordered_map<ge::DataType, uint64_t> VAR_DATA_TYPE_TO_INT{{ge::DataType::DT_INT32, 1},
                                                                            {ge::DataType::DT_INT8, 2},
                                                                            {ge::DataType::DT_UINT8, 3},
                                                                            {ge::DataType::DT_FLOAT16, 4},
                                                                            {ge::DataType::DT_FLOAT, 5},
                                                                            {ge::DataType::DT_BF16, 6}};

static const std::set<ge::DataType> VAR_DTYPE_SET = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8,
                                                    ge::DT_UINT8, ge::DT_BF16};
static const std::set<ge::DataType> INDICES_DTYPE_SET = {ge::DT_INT32, ge::DT_INT64};
static const std::set<ge::DataType> SIMD_ATOMIC_ADD_NOT_SUPPORT_DTYPE = {ge::DataType::DT_UINT8};
static const std::set<ge::DataType> SIMT_ATOMIC_ADD_NOT_SUPPORT_DTYPE = {ge::DataType::DT_UINT8, ge::DataType::DT_INT8};

static const std::set<ge::DataType> scatterAddDeterminsiticType = { ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};

static std::string ToString(std::set<ge::DataType> supportDtypes)
{
    std::stringstream ss;
    for (const auto& element : supportDtypes) {
        ss << element << " ";
    }

    return ss.str();
}

template <typename T>
static std::string ToString(const T* value, size_t size) {
  std::string r = "[";
  for (size_t i = 0; i < size; i++) {
    r = r + std::to_string(value[i]) + ", ";
  }
  r = r + "]";
  return r;
}

uint64_t ScatterAddTiling::GetSortTmpSize(ge::DataType dataType, uint32_t lastAxisNum, bool isDescend)
{
    std::vector<int64_t> shapeVec = { lastAxisNum };
    ge::Shape srcShape(shapeVec);
    AscendC::SortConfig config;
    config.type = AscendC::SortType::RADIX_SORT;
    config.isDescend = isDescend;
    config.hasSrcIndex = false;
    config.hasDstIndex = true;
    uint32_t maxValue = 0, minValue = 0;
    AscendC::GetSortMaxMinTmpSize(srcShape, dataType, ge::DT_UINT32, false, config, maxValue, minValue);
    OP_LOGI("RadixSortTilingForAscendC", "Need tmp buffer %u byte for ac sort api", maxValue);
    return maxValue;
}

ge::graphStatus ScatterAddTiling::getRestAvailableSize(uint64_t sampleNum, uint64_t valueTypeBytes, uint64_t originalSize, uint64_t postAxisSize,
    ge::DataType idType)
{
    uint64_t indicesDtypeSize = ge::GetSizeByDataType(idType);
    OP_CHECK_IF(indicesDtypeSize <= 0, OP_LOGE(opName, "get indicesType size fail."),
                    return ge::GRAPH_FAILED);
    auto ubBlock = Ops::Base::GetUbBlockSize(context_);
    uint64_t occupy = sampleNum * Ops::Base::CeilAlign(valueTypeBytes * postAxisSize, static_cast<uint64_t>(ubBlock)) + sampleNum * Ops::Base::CeilAlign(FLOAT_BYTES * postAxisSize, static_cast<uint64_t>(ubBlock)) + 
        Ops::Base::CeilAlign(sampleNum * indicesDtypeSize, static_cast<uint64_t>(ubBlock)) * THREE + Ops::Base::CeilAlign(sampleNum * UINT32_BYTES, static_cast<uint64_t>(ubBlock)) + DOUBLE * DOUBLE * UB_AGLIN_VALUE +
        DOUBLE * UB_AGLIN_VALUE + GetSortTmpSize(idType, sampleNum, false) + Ops::Base::CeilAlign(postAxisSize * FLOAT_BYTES, static_cast<uint64_t>(ubBlock));
    return originalSize - occupy;
}

ge::graphStatus ScatterAddTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(opName, "fail to get platform info"),
                    return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((aivNum <= 0), OP_LOGE(opName, "fail to get coreNum."),
                    return ge::GRAPH_FAILED);
    totalCoreNum_ = aivNum;
    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    // UB Size Need reserve space for Dcache / CCEC Compile Stack.
    ubSize_ = ubSizePlatForm;
    if (isSimt_) {
        if (isSort_) {
            ubSize_ = ubSizePlatForm - DCACHE_SIZE1 - SIMD_RESERVED_SIZE;
        } else {
            OP_CHECK_IF((ubSizePlatForm <= DCACHE_SIZE),
                            OP_LOGE(opName, "ub size less than Dcache Size. please check"),
                            return ge::GRAPH_FAILED);
            ubSize_ = ubSizePlatForm - DCACHE_SIZE;
        }
    } else {
        ubSize_ = ubSizePlatForm - SIMD_RESERVED_SIZE;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterAddTiling::GetShapeAttrsInfo()
{
    auto var = context_->GetInputShape(VAR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, var);
    auto varShape = var->GetStorageShape();
    varSize_ = varShape.GetShapeSize();
    varShape_[0] = varShape.GetDim(0);
    if (varSize_ != 0) {
        varShape_[1] = varSize_ / varShape_[0];
    }

    auto indices = context_->GetInputShape(INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indices);
    auto indiceShape = indices->GetStorageShape();
    indicesNum_ = indiceShape.GetShapeSize();

    auto updates = context_->GetInputShape(UPDATES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, updates);
    auto updateShape = updates->GetStorageShape();
    updatesSize_ = updateShape.GetShapeSize();
    uint64_t updateDims = updateShape.GetDimNum();
    if (updateDims == 0) {
        isUpdateScalar_ = 1;
    } else {
        OP_CHECK_IF(CheckUpdatesShape(varShape, indiceShape, updateShape) != ge::GRAPH_SUCCESS,
                        OP_LOGE(opName, "update shape check failed."), return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(CheckInputDtype() != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName, "input dtype check failed."), return ge::GRAPH_FAILED);

    bool supportAtomicAdd =
        isSimt_ ? SIMT_ATOMIC_ADD_NOT_SUPPORT_DTYPE.find(varDtype_) == SIMT_ATOMIC_ADD_NOT_SUPPORT_DTYPE.end()
                : SIMD_ATOMIC_ADD_NOT_SUPPORT_DTYPE.find(varDtype_) == SIMD_ATOMIC_ADD_NOT_SUPPORT_DTYPE.end();
    if (supportAtomicAdd) {
        isSort_ = indicesNum_ > varShape_[0] * TEN ? 1 : 0;
        if (varShape_[1] >= VAR_LASTDIM_10W && indicesNum_ > MIN_SIZE_SORT_INDICES_128) {
            isSort_ = indicesNum_ > varShape_[0] * THREE ? 1 : 0;
        }
    }

    if (context_->GetDeterministic() && !isUpdateScalar_ &&
        scatterAddDeterminsiticType.find(varDtype_) != scatterAddDeterminsiticType.end()) {
        isDeterminTemplate_ = 1;
        isSort_ = 0;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterAddTiling::CheckInputDtype()
{
    auto indicesPtr = context_->GetInputDesc(INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesPtr);
    indicesDtype_ = indicesPtr->GetDataType();
    OP_CHECK_IF(
        (INDICES_DTYPE_SET.find(indicesDtype_) == INDICES_DTYPE_SET.end()),
        OP_LOGE(opName, "indices data dtype only support %s, currently, please check.",
                                        ToString(INDICES_DTYPE_SET).c_str()),
        return ge::GRAPH_FAILED);
    indicesDtypeSize_ = ge::GetSizeByDataType(indicesDtype_);
    OP_CHECK_IF(indicesDtypeSize_ <= 0, OP_LOGE(opName, "get indicesDtype size fail."),
                    return ge::GRAPH_FAILED);
    auto dataPtr = context_->GetInputDesc(VAR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dataPtr);
    varDtype_ = dataPtr->GetDataType();
    OP_CHECK_IF((VAR_DTYPE_SET.find(varDtype_) == VAR_DTYPE_SET.end()),
                    OP_LOGE(opName, "var data dtype only support %s, please check.",
                                                    ToString(VAR_DTYPE_SET).c_str()),
                    return ge::GRAPH_FAILED);
    varTypeSize_ = ge::GetSizeByDataType(varDtype_);
    OP_CHECK_IF(varTypeSize_ <= 0, OP_LOGE(opName, "get dataType size fail."),
                    return ge::GRAPH_FAILED);
    isSimt_ = varShape_[1] * varTypeSize_ < VAR_TAIL_DIM_SIZE;
    auto updatePtr = context_->GetInputDesc(UPDATES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, updatePtr);
    auto updatesType = updatePtr->GetDataType();
    OP_CHECK_IF(
        (VAR_DTYPE_SET.find(updatesType) == VAR_DTYPE_SET.end()),
        OP_LOGE(opName, "updates data dtype only support %s currently, please check.",
                                        ToString(VAR_DTYPE_SET).c_str()),
        return ge::GRAPH_FAILED);
    updatesDtypeSize_ = ge::GetSizeByDataType(updatesType);
    OP_CHECK_IF(
        (updatesType != varDtype_),
        OP_LOGE(opName, "expected updates dtype to be equal to var dtype, please check."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterAddTiling::CheckUpdatesShape(const gert::Shape& varShape, const gert::Shape& indicesShape,
                                                    const gert::Shape& updatesShape)
{
    uint64_t varDimNum = static_cast<uint64_t>(varShape.GetDimNum());
    uint64_t indicesDimNum = static_cast<uint64_t>(indicesShape.GetDimNum());
    uint64_t updatesDimNum = static_cast<uint64_t>(updatesShape.GetDimNum());
    OP_CHECK_IF(
        (updatesDimNum != indicesDimNum + varDimNum - 1),
        OP_LOGE(
            opName, "updatesDimNum must have the same number of indicesDimNum add varDimNum - 1, please check."),
        return ge::GRAPH_FAILED);
    for (uint64_t i = 0; i < indicesDimNum; i++) {
        OP_CHECK_IF(
            (static_cast<uint32_t>(updatesShape.GetDim(i)) != static_cast<uint32_t>(indicesShape.GetDim(i))),
            OP_LOGE(
                opName, "updatesShape should be equal to the shape of 'indices' concats the shape of 'var' except for the first dimension."),
            return ge::GRAPH_FAILED);
    }

    for (uint64_t i = 1; i < varDimNum; i++) {
        OP_CHECK_IF((static_cast<uint32_t>(updatesShape.GetDim(i + indicesDimNum - 1)) !=
                        static_cast<uint32_t>(varShape.GetDim(i))),
                        OP_LOGE(
                            opName, "updatesShape should be equal to the shape of 'indices' concats the shape of 'var' except for the first dimension."),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

std::set<uint64_t> ScatterAddTiling::FindUniqueCut(uint64_t usedCoreNum)
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

std::tuple<uint64_t, uint64_t> ScatterAddTiling::SimdTiling(
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
        if (m > indicesNum_) {
            continue;
        }

        uint64_t n = usedCoreNum / m;
        n = n < 1 ? 1 : n;
        // 列方向分核超过列方向数据量，则说明有空闲核
        if (n > colNumAlign) {
            continue;
        }

        // ub重排模板，不会在列方向切分
        if (varShape_[1] * updatesDtypeSize_ <= colLimitSize && n > 1) {  // 当尾轴长度小于设定最小值时，列方向不应切分
            continue;
        }

        uint64_t rowNormalBlock = Ops::Base::CeilDiv(indicesNum_, m);  // 切块后每块行数
        // 列方向按照BASE_A_SIZE基本块切分
        uint64_t colNormalBlock = Ops::Base::CeilDiv(colNumAlign, n);     // 切块后每块列数

        // 正常切分块和尾块的差值计算
        uint64_t delta = rowNormalBlock * colNormalBlock;
        if (m * n == usedCoreNum) {    // 选择把核尽量用满的组合。没用满时此条件不满足，delta为最大值，后面排序后不会选中此组合
            if (indicesNum_ % m == 0 && colNumAlign % n == 0) {  // 两个方向都能整切，每个块大小相同（理想情况），此时直接返回
                uint64_t rowTileNum = m;
                uint64_t colTileNum = n;
                return std::make_tuple(rowTileNum, colTileNum);
            } else if (indicesNum_ % m == 0) {                   // 行方向整切，列不能整切，计算差值
                delta = static_cast<uint64_t>(delta) - rowNormalBlock * (colNumAlign % colNormalBlock);
            } else if (colNumAlign % n == 0) {                      // 列方向整切，行不能整切，计算差值
                delta = static_cast<uint64_t>(delta) - (indicesNum_ % rowNormalBlock) * colNormalBlock;
            } else {                                                // 两个方向都不能整切，计算差值
                delta = static_cast<uint64_t>(delta) - (indicesNum_ % rowNormalBlock) * (colNumAlign % colNormalBlock);
            }
        }

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
        std::sort(   // 排序，然后取delta最小的m, n值，如果delta相等时取m最小的
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

void ScatterAddTiling::DoBlockTiling(uint64_t baseCol)
{
    uint64_t ubBlock = static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_));
    uint64_t colNumAlign = Ops::Base::CeilDiv(varShape_[1], baseCol);   // 按设定的512B切分，列能切分的份数
    atomicAddCoreNum_ =
        std::min(totalCoreNum_, static_cast<uint64_t>(indicesNum_ * colNumAlign * BASE_A_SIZE / BASE_BLOCK_SIZE));  // 按每个核最小分1024个数计算使用核数
    atomicAddCoreNum_ = atomicAddCoreNum_ == 0 ? 1 : atomicAddCoreNum_;
    std::tie(rowTileNum_, colTileNum_) = SimdTiling(atomicAddCoreNum_, colNumAlign, COL_LIMIT_SIZE, true);  // 分别计算两个方向切分份数，使得满足：1、尽量把核用满；2、尾核与整核数据量差值尽量小

    // 分别计算行和列的整核/尾核长度，整核列长512B对齐
    normBlockRow_ = Ops::Base::CeilDiv(indicesNum_, rowTileNum_);
    normBlockCol_ = Ops::Base::CeilDiv(varShape_[1], colTileNum_);
    normBlockCol_ = Ops::Base::CeilAlign(normBlockCol_, ubBlock / updatesDtypeSize_);
    rowTileNum_ = Ops::Base::CeilDiv(indicesNum_, normBlockRow_);
    colTileNum_ = Ops::Base::CeilDiv(varShape_[1], normBlockCol_);
    atomicAddCoreNum_ = rowTileNum_ * colTileNum_;
    tailBlockRow_ = indicesNum_ - (rowTileNum_ - 1) * normBlockRow_;
    tailBlockCol_ = varShape_[1] - (colTileNum_ - 1) * normBlockCol_;
}

/**
 * @brief Find best baseSize in range [baseXoStart, baseXoEnd], use dichotomy algorithm.
 */
uint64_t ScatterAddTiling::CalBestBaseSize(uint64_t baseXoStart, uint64_t baseXoEnd)
{
    uint64_t indicesSortBufCnt = 2;
    uint64_t baseXoMid;
    uint64_t tmpTotalSize = 0;
    baseXoEnd = baseXoEnd + 1;
    uint64_t ubBlock = static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_));
    while (baseXoEnd - baseXoStart > 1) {
        baseXoMid = (baseXoStart + baseXoEnd) / DOUBLE;
        if (indicesCastMode_ == 0) {
            uint64_t sortNeedTmpSize = static_cast<uint64_t>(GetSortTmpSize(indicesDtype_, baseXoMid, false));
            tmpTotalSize = Ops::Base::CeilAlign(baseXoMid * ubFactorCol_ * updatesDtypeSize_, ubBlock) * BUFFER_NUM_SIMD_SORT +  // updates
                           Ops::Base::CeilAlign(baseXoMid * indicesDtypeSize_, ubBlock) * BUFFER_NUM_SIMD_SORT +                 // indices
                           Ops::Base::CeilAlign(baseXoMid * ubFactorCol_ * updatesDtypeSize_, ubBlock) +               // 存放相加后的临时变量
                           Ops::Base::CeilAlign(baseXoMid * indicesDtypeSize_, ubBlock) +                              // indicessorted
                           Ops::Base::CeilAlign(baseXoMid * sizeof(uint32_t), ubBlock) * indicesSortBufCnt +           // sortedIdxBuf  排序后原始的索引
                           SORT_STAT_PADDING + SORT_STAT_PADDING +                                               // sort padding
                           Ops::Base::CeilAlign(sortNeedTmpSize, ubBlock); // sort shared buf size
        } else {
            uint64_t sortNeedTmpSize = static_cast<uint64_t>(GetSortTmpSize(indicesCastDtype_, baseXoMid, false));
            tmpTotalSize = Ops::Base::CeilAlign(baseXoMid * ubFactorCol_ * updatesDtypeSize_, ubBlock) * BUFFER_NUM_SIMD_SORT +  // updates
                           Ops::Base::CeilAlign(baseXoMid * indicesCastDtypeSize_, ubBlock) +                              // Cast后的indices
                           Ops::Base::CeilAlign(baseXoMid * ubFactorCol_ * updatesDtypeSize_, ubBlock) +                   // 存放相加后的临时变量
                           Ops::Base::CeilAlign(baseXoMid * indicesCastDtypeSize_, ubBlock) +                              // indicessorted
                           Ops::Base::CeilAlign(baseXoMid * sizeof(uint32_t), ubBlock) * indicesSortBufCnt +               // sortedIdxBuf  排序后原始的索引
                           SORT_STAT_PADDING + SORT_STAT_PADDING +                                                   // sort padding
                           Ops::Base::CeilAlign(baseXoMid * indicesDtypeSize_, ubBlock) * BUFFER_NUM_SIMD_SORT +           // Cast前的indices
                           Ops::Base::CeilAlign(sortNeedTmpSize, ubBlock); // sort shared buf size
        }

        if (tmpTotalSize <= ubSize_) {
            baseXoStart = baseXoMid;
        } else {
            baseXoEnd = baseXoMid;
        }
    }
    return baseXoStart;
}

ge::graphStatus ScatterAddTiling::TilingSimdSupportAtomicAddSortCompute()
{
    GetCastType();

    uint64_t baseCol = BASE_A_SIZE / updatesDtypeSize_;             // 列按BASE_A_SIZE切分
    DoBlockTiling(baseCol);                                         // 计算每个核切分的数量，分别按行/列分整核/尾核
    if (normBlockCol_ * updatesDtypeSize_ <= COL_LIMIT_SIZE) {      // 如果整核列数小于设定值（1024B），则将baseCol 与normBlockCol_保持一致
        baseCol = normBlockCol_;
    }
    ubFactorCol_ = baseCol;
    ubSize_ -= SIMD_RESERVED_SIZE;
    uint64_t coreMaxRow = std::max(normBlockRow_, tailBlockRow_);

    // ub split for non sort case
    uint64_t ubBlock = static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_));
    uint64_t maxBaseRow = (ubSize_ - BUFFER_NUM_SIMD_SORT * ubBlock) / BUFFER_NUM_SIMD_SORT /            // 在列长确定时，计算UB最多容纳多少行，此时还未考虑排序空间
                          (baseCol * updatesDtypeSize_ + indicesDtypeSize_ + indicesCastDtypeSize_);
    uint64_t baseRow = maxBaseRow;
    if (coreMaxRow < baseRow) {    // 在单循环行数超过block行数时，将单循环行数缩小至block行数，并重新计算单循环列数
        baseRow = coreMaxRow;
        baseCol = (ubSize_ - BUFFER_NUM_SIMD_SORT * (ubBlock + baseRow * (indicesDtypeSize_ + indicesCastDtypeSize_))) /
                  BUFFER_NUM_SIMD_SORT / (baseRow * updatesDtypeSize_);
        baseCol = Ops::Base::FloorAlign(baseCol, ubBlock / updatesDtypeSize_);
    }

    // ub split for sort case
    ubFactorRow_ = CalBestBaseSize(1, maxBaseRow);   // 通过二分法计算排序情况下的单循环行数
    if (coreMaxRow < ubFactorRow_) {                 // coreMaxRow < ubFactorRow_时刷新单循环行数，并重新计算单循环列数
        ubFactorRow_ = coreMaxRow;                   
        uint64_t indicesSortBufCnt = 2;
        uint64_t remainSize = 0;
        if (indicesCastMode_ == 0) {
            uint64_t sortNeedTmpSize = static_cast<uint64_t>(GetSortTmpSize(indicesDtype_, ubFactorRow_, false));
            remainSize = ubSize_ - Ops::Base::CeilAlign(ubFactorRow_ * indicesDtypeSize_, ubBlock) * BUFFER_NUM_SIMD_SORT -
                                  Ops::Base::CeilAlign(ubFactorRow_ * indicesDtypeSize_, ubBlock) -
                                  Ops::Base::CeilAlign(ubFactorRow_ * sizeof(uint32_t), ubBlock) * indicesSortBufCnt -
                                  Ops::Base::CeilAlign(sortNeedTmpSize, ubBlock) - SORT_STAT_PADDING - SORT_STAT_PADDING;
        } else {
            uint64_t sortNeedTmpSize = static_cast<uint64_t>(GetSortTmpSize(indicesCastDtype_, ubFactorRow_, false));
            remainSize = ubSize_ - Ops::Base::CeilAlign(ubFactorRow_ * indicesCastDtypeSize_, ubBlock) -
                                  Ops::Base::CeilAlign(ubFactorRow_ * indicesDtypeSize_, ubBlock) * BUFFER_NUM_SIMD_SORT -
                                  Ops::Base::CeilAlign(ubFactorRow_ * indicesCastDtypeSize_, ubBlock) -
                                  Ops::Base::CeilAlign(ubFactorRow_ * sizeof(uint32_t), ubBlock) * indicesSortBufCnt -
                                  Ops::Base::CeilAlign(sortNeedTmpSize, ubBlock) - SORT_STAT_PADDING - SORT_STAT_PADDING;
        }

        ubFactorCol_ =
            (remainSize - BUFFER_NUM_SIMD_SORT * ubBlock - ubBlock) / (BUFFER_NUM_SIMD_SORT + 1) / (ubFactorRow_ * updatesDtypeSize_);
        ubFactorCol_ = Ops::Base::FloorAlign(ubFactorCol_, ubBlock / updatesDtypeSize_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterAddTiling::TilingSimdSupportAtomicAddCompute()
{
    // block split, ensure that ub prioritizes a, block split s * (a/maxBaseCol)
    uint64_t minBaseIndices = 1;
    ubSize_ -= SIMD_RESERVED_SIZE;
    uint64_t ubBlock = static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_));

    uint64_t maxBaseCol =       // 计算UB最多能容纳多少个updates, (假设只搬1行且尾轴非常大)  
        (ubSize_ - BUFFER_NUM * (ubBlock + minBaseIndices * indicesDtypeSize_)) / BUFFER_NUM / (minBaseIndices * updatesDtypeSize_);
    maxBaseCol = Ops::Base::FloorAlign(maxBaseCol, ubBlock / updatesDtypeSize_);         // 向下32B对齐

    uint64_t colNumAlign = Ops::Base::CeilDiv(varShape_[1], maxBaseCol);                 // 计算尾轴能被切几份，尾轴比较小时则不切尾轴（切1份）
    // [行数 * 每行切分数量] 不超过coreNum时，相当于每个核分一小块（每小块只有1行，且核可能未用满）
    atomicAddCoreNum_ = std::min(totalCoreNum_, static_cast<uint64_t>(indicesNum_ * colNumAlign));
    while(atomicAddCoreNum_ <= totalCoreNum_ / DOUBLE && maxBaseCol > MIN_COL_SIZE) {
        maxBaseCol = Ops::Base::CeilAlign(maxBaseCol / DOUBLE, ubBlock / updatesDtypeSize_);
        colNumAlign = Ops::Base::CeilDiv(varShape_[1], maxBaseCol);
        atomicAddCoreNum_ = std::min(totalCoreNum_, static_cast<uint64_t>(indicesNum_ * colNumAlign));
    }
    atomicAddCoreNum_ = atomicAddCoreNum_ == 0 ? 1 : atomicAddCoreNum_;

    // 分别计算两个方向切分份数，使得满足：1、尽量把核用满；2、尾核与整核数据量差值尽量小
    std::tie(rowTileNum_, colTileNum_) = SimdTiling(atomicAddCoreNum_, colNumAlign, maxBaseCol * updatesDtypeSize_);

    normBlockRow_ = Ops::Base::CeilDiv(indicesNum_, rowTileNum_);                      // 每个核分块行数
    normBlockCol_ = Ops::Base::CeilDiv(varShape_[1], colTileNum_);                      // 每个核分块列数
    normBlockCol_ = Ops::Base::CeilAlign(normBlockCol_, ubBlock / updatesDtypeSize_);
    rowTileNum_ = Ops::Base::CeilDiv(indicesNum_, normBlockRow_);
    colTileNum_ = Ops::Base::CeilDiv(varShape_[1], normBlockCol_);
    atomicAddCoreNum_ = rowTileNum_ * colTileNum_;
    tailBlockRow_ = indicesNum_ - (rowTileNum_ - 1) * normBlockRow_;              // 行尾核分块行数
    tailBlockCol_ = varShape_[1] - (colTileNum_ - 1) * normBlockCol_;              // 列尾核分块列数

    // ub split
    ubFactorCol_ = std::min(normBlockCol_, maxBaseCol);                            // UB每次循环搬运的列数
    ubFactorRow_ = (ubSize_ - BUFFER_NUM * ubBlock) / BUFFER_NUM / (ubFactorCol_ * updatesDtypeSize_ + indicesDtypeSize_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterAddTiling::TilingSimdNotSupportAtomicAddCompute(bool supportAtomicAdd)
{
    perCoreHandleIndices_ = Ops::Base::CeilDiv(indicesNum_, totalCoreNum_);
    atomicAddCoreNum_ = Ops::Base::CeilDiv(indicesNum_, perCoreHandleIndices_);
    auto tailCoreHandleIndicesNum = indicesNum_ - (atomicAddCoreNum_ - 1) * perCoreHandleIndices_;
    postAxisSize_ = varShape_[1];

    if (!supportAtomicAdd) {
        if (updatesSize_ == 1) {
            updatesDtypeSize_ = INT32_TYPE_SIZE;
        } else {
            updatesDtypeSize_ = INT32_TYPE_SIZE + varTypeSize_;
        }
    }

    auto varUbBlock = Ops::Base::GetUbBlockSize(context_) / varTypeSize_;
    updatesUbFactor_ = Ops::Base::CeilAlign(postAxisSize_, varUbBlock);
    if (updatesUbFactor_ * updatesDtypeSize_ > ((ubSize_ - MIN_BLOCK_SIZE) / BUFFER_NUM)) {
        auto indicesUbSize = std::min(MIN_BLOCK_SIZE, perCoreHandleIndices_ * indicesDtypeSize_);
        indicesUbFactor_ = Ops::Base::CeilAlign(indicesUbSize, static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_))) / indicesDtypeSize_;
        auto maxUbAvailable = (ubSize_ - (indicesUbFactor_ * indicesDtypeSize_)) / BUFFER_NUM / updatesDtypeSize_;
        updatesUbFactor_ = Ops::Base::FloorAlign(std::min(postAxisSize_, maxUbAvailable), varUbBlock);
    } else {
        indicesUbFactor_ = Ops::Base::FloorAlign(ubSize_ - updatesUbFactor_ * updatesDtypeSize_ * BUFFER_NUM,
                                        static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_))) /
                        indicesDtypeSize_;
    }

    indicesLoopSize_ = Ops::Base::CeilDiv(perCoreHandleIndices_, indicesUbFactor_);
    indicesTailUbFactor_ = perCoreHandleIndices_ - (indicesLoopSize_ - 1) * indicesUbFactor_;
    tailCoreIndicesLoopSize_ = Ops::Base::CeilDiv(tailCoreHandleIndicesNum, indicesUbFactor_);
    tailCoreIndicesTailUbFactor_ = tailCoreHandleIndicesNum - (tailCoreIndicesLoopSize_ - 1) * indicesUbFactor_;

    updatesLoopSize_ = Ops::Base::CeilDiv(postAxisSize_, updatesUbFactor_);
    updatesTailUbFactor_ = postAxisSize_ - (updatesLoopSize_ - 1) * updatesUbFactor_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterAddTiling::TilingCopyCompute()
{
    castTypeSize_ = ge::GetSizeByDataType(ge::DT_INT32);
    OP_CHECK_IF(castTypeSize_ <= 0, OP_LOGE(opName, "get dataType size fail."),
                    return ge::GRAPH_FAILED);
    ubFactor_ = Ops::Base::FloorAlign(ubSize_ / BUFFER_NUM / (castTypeSize_ + varTypeSize_),
                                static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_)));

    perCoreHandleVar_ = std::max(Ops::Base::CeilDiv(varSize_, totalCoreNum_), MIN_BLOCK_SIZE / varTypeSize_);
    copyCoreNum_ = Ops::Base::CeilDiv(varSize_, perCoreHandleVar_);
    auto tailCoreHandleSize = varSize_ - perCoreHandleVar_ * (copyCoreNum_ - 1);
    blockFactor_ = Ops::Base::CeilDiv(perCoreHandleVar_, ubFactor_);           // 每个核需要搬入整个UB处理的次数
    tailUbFactor_ = perCoreHandleVar_ - (blockFactor_ - 1) * ubFactor_;  // 尾UB处理的数据size
    tailBlockFactor_ = Ops::Base::CeilDiv(tailCoreHandleSize, ubFactor_);      // 尾核UB循环次数
    tailCoreTailUbFactor_ = tailCoreHandleSize - (tailBlockFactor_ - 1) * ubFactor_;  // 尾核尾UB处理size
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterAddTiling::ScatterAddDeterministicTiling()
{
    templateKey_ = isSimt_ ? 0 : 1;
    uint64_t var1Size = varShape_[1] * varTypeSize_;
     // 尾轴大于128B走列分核；尾轴小于128B时索引小于4096走列分核
    if ((var1Size > DETERMIN_THRESHOLD_MID) || (indicesNum_ < INDICES_THRESHOLD)) {
        // 尽量按尾轴分核，indices在一个核上，没有确定性问题，不需要量化
        perCoreHandleCol_ = Ops::Base::CeilDiv(varShape_[1], totalCoreNum_);
        logicCoreNum_ = Ops::Base::CeilDiv(varShape_[1], perCoreHandleCol_);
        tailCoreHandleCol_ = varShape_[1] - (logicCoreNum_ - 1) * perCoreHandleCol_;
        auto postUbBlock = static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_)) / varTypeSize_;
        if (perCoreHandleCol_ * varTypeSize_ > (ubSize_ - MIN_BLOCK_SIZE)) {
            auto indicesUbSize = std::min(MIN_BLOCK_SIZE, indicesNum_ * indicesDtypeSize_);
            indicesUbFactor_ = Ops::Base::CeilAlign(indicesUbSize, static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_))) / indicesDtypeSize_;
            auto maxUbAvailable = (ubSize_ - indicesUbFactor_ * indicesDtypeSize_) / updatesDtypeSize_;
            updatesUbFactor_ = Ops::Base::FloorAlign(std::min(perCoreHandleCol_, maxUbAvailable), postUbBlock);
        } else {
            updatesUbFactor_ = Ops::Base::CeilAlign(perCoreHandleCol_, postUbBlock);
            indicesUbFactor_ = Ops::Base::FloorAlign(ubSize_ - updatesUbFactor_ * updatesDtypeSize_,
                            static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_))) / indicesDtypeSize_;
        }

        indicesLoopSize_ = Ops::Base::CeilDiv(indicesNum_, indicesUbFactor_);
        indicesTailUbFactor_ = indicesNum_ - (indicesLoopSize_ - 1) * indicesUbFactor_;  
        updatesLoopSize_ = Ops::Base::CeilDiv(perCoreHandleCol_, updatesUbFactor_);
        updatesTailUbFactor_ = perCoreHandleCol_ - (updatesLoopSize_ - 1) * updatesUbFactor_;
        tailCoreColsLoopSize_ = Ops::Base::CeilDiv(tailCoreHandleCol_, updatesUbFactor_);
        tailCoreColsTailUbFactor_ = tailCoreHandleCol_ - (tailCoreColsLoopSize_ - 1) * updatesUbFactor_;

        return ge::GRAPH_SUCCESS;
    } 

    // 量化分支，sort按indices分核，indices搬多少行，update搬多少行
    isDeterministic_ = 1;
    perCoreHandleIndices_ = Ops::Base::CeilDiv(indicesNum_, totalCoreNum_);
    uint32_t shiftOffset = UB_AGLIN_VALUE;
    // * updatesSum * updateFloatSum * indices * sortedIndices * sortedOriginIndex * uniqueIndices * shiftOffset * sortTmp
    uint64_t oneIdOccupyBytes = varTypeSize_ * varShape_[1] + FLOAT_BYTES * varShape_[1] + 
        indicesDtypeSize_ * THREE  + UINT32_BYTES + DOUBLE * shiftOffset * DOUBLE + SORT_UTIL_SIZE;
    uint64_t halfUBSize = ubSize_ / DOUBLE;
    indicesUbFactor_ = halfUBSize / oneIdOccupyBytes;

    int32_t restSize = -1;
    while (restSize <= 0) {
        --indicesUbFactor_;
        restSize = getRestAvailableSize(indicesUbFactor_, varTypeSize_, halfUBSize, varShape_[1], indicesDtype_);
    }
    if (indicesUbFactor_ > indicesNum_) {
        indicesUbFactor_ = indicesNum_;
    }

    indicesUbFactor_ = indicesUbFactor_ > perCoreHandleIndices_ ? perCoreHandleIndices_ : indicesUbFactor_;
    logicCoreNum_ = Ops::Base::CeilDiv(indicesNum_, perCoreHandleIndices_);
    auto tailCoreHandleIndicesNum = indicesNum_ - (logicCoreNum_ - 1) * perCoreHandleIndices_;
    indicesLoopSize_ = Ops::Base::CeilDiv(perCoreHandleIndices_, indicesUbFactor_);
    indicesTailUbFactor_ = perCoreHandleIndices_ - (indicesLoopSize_ - 1) * indicesUbFactor_;
    tailCoreIndicesLoopSize_ = Ops::Base::CeilDiv(tailCoreHandleIndicesNum, indicesUbFactor_);
    tailCoreIndicesTailUbFactor_ = tailCoreHandleIndicesNum - (tailCoreIndicesLoopSize_ - 1) * indicesUbFactor_;
    updatesUbFactor_ = varShape_[1]; 

    // 反量化分支选择
    float xValue = indicesNum_ / varShape_[0];
    float dequantizeBrancThreshold = 1.0;
    if (var1Size >= DETERMIN_THRESHOLD_MID) {
        dequantizeBrancThreshold = SCALE_FACOTR;
    } else {
        dequantizeBrancThreshold = var1Size / (DETERMIN_THRESHOLD_BOT * SCALE_FACOTR);
    }
    dequantizeBranch_ = xValue < dequantizeBrancThreshold ? 1: 0;

    // 反量化分核
    auto ubBlock = static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_));
    if (dequantizeBranch_ == 1) {
        // 反量化分支1，按indides分核
        perCoreHandleRows_ = perCoreHandleIndices_;
        uint32_t oneRowOccupyBytes = varShape_[1] * UINT32_BYTES + varShape_[1] * FLOAT_BYTES + varShape_[1] * FLOAT_BYTES + indicesDtypeSize_;
        rowsInUb_ = ubSize_ / DOUBLE / oneRowOccupyBytes;
        restSize = -1;
        while (restSize <= 0) {
            --rowsInUb_;
            uint32_t occupyBytes = rowsInUb_ * Ops::Base::CeilAlign(varShape_[1] * UINT32_BYTES, ubBlock) + rowsInUb_ * Ops::Base::CeilAlign(varShape_[1] * FLOAT_BYTES, ubBlock) + 
                DOUBLE * UB_AGLIN_VALUE + Ops::Base::CeilAlign(varShape_[1] * FLOAT_BYTES, ubBlock) + Ops::Base::CeilAlign(indicesDtypeSize_ * rowsInUb_, ubBlock);
            restSize = (ubSize_ / DOUBLE) - occupyBytes;
        }
        rowsInUb_ = rowsInUb_ > perCoreHandleRows_ ? perCoreHandleRows_ : rowsInUb_;
        deQuantizeCoreNum_ = logicCoreNum_;
        tailCoreHandleRows_ = indicesNum_ - (deQuantizeCoreNum_ - 1) * perCoreHandleRows_;
    } else {
        // 反量化分支2, 按varShape[0]二次分核
        perCoreHandleRows_ = Ops::Base::CeilDiv(varShape_[0], totalCoreNum_);
        uint32_t oneRowOccupyBytes = varShape_[1] * UINT32_BYTES + varShape_[1] * FLOAT_BYTES;
        rowsInUb_ = ubSize_ / DOUBLE / oneRowOccupyBytes;
        restSize = -1;
        while (restSize <= 0) {
            --rowsInUb_;
            uint32_t occupyBytes = rowsInUb_ * Ops::Base::CeilAlign(varShape_[1] * UINT32_BYTES, ubBlock) + rowsInUb_ * Ops::Base::CeilAlign(varShape_[1] * FLOAT_BYTES, ubBlock) + 
                DOUBLE * UB_AGLIN_VALUE + Ops::Base::CeilAlign(varShape_[1] * FLOAT_BYTES, ubBlock);
            restSize = (ubSize_ / DOUBLE) - occupyBytes;
        }
        rowsInUb_ = rowsInUb_ > perCoreHandleRows_ ? perCoreHandleRows_ : rowsInUb_;
        deQuantizeCoreNum_ = Ops::Base::CeilDiv(varShape_[0], perCoreHandleRows_);
        tailCoreHandleRows_ = varShape_[0] - (deQuantizeCoreNum_ - 1) * perCoreHandleRows_;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterAddTiling::TilingSimtSort()
{
    GetCastType();
    int64_t mid = 0;
    int64_t start = 1;
    int64_t end = static_cast<int64_t>(indicesNum_ + 1);
    int64_t sortTmpSize = 0;
    int64_t ubBlock = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    while(end - start > 1) {
        mid = (end + start) / static_cast<int64_t>(DOUBLE);
        int64_t totalIndexSize = 0;
        // 所需空间：indicesLocal + indicesSortedLocal + updatesOriginIdxLocal + uniqueIdCountLocal
        if (indicesCastMode_ == 0) {
            totalIndexSize = Ops::Base::CeilAlign(mid * static_cast<int64_t>(indicesDtypeSize_), ubBlock) * DOUBLE +
                             Ops::Base::CeilAlign(mid * static_cast<int64_t>(sizeof(uint32_t)), ubBlock) + ubBlock * DOUBLE +
                             Ops::Base::CeilAlign(mid * static_cast<int64_t>(sizeof(int32_t)), ubBlock) + ubBlock * DOUBLE;
            sortTmpSize = static_cast<int64_t>(GetSortTmpSize(indicesDtype_, mid, false));
        } else {
            totalIndexSize = Ops::Base::CeilAlign(mid * static_cast<int64_t>(indicesDtypeSize_), ubBlock) +
                             Ops::Base::CeilAlign(mid * static_cast<int64_t>(indicesCastDtypeSize_), ubBlock) * DOUBLE +
                             Ops::Base::CeilAlign(mid * static_cast<int64_t>(sizeof(uint32_t)), ubBlock) + ubBlock * DOUBLE +
                             Ops::Base::CeilAlign(mid * static_cast<int64_t>(sizeof(int32_t)), ubBlock) + ubBlock * DOUBLE;
            sortTmpSize = static_cast<int64_t>(GetSortTmpSize(indicesCastDtype_, mid, false));
        }

        sortTmpSize = Ops::Base::CeilAlign(sortTmpSize, ubBlock);
        int64_t updateSize = Ops::Base::CeilAlign(static_cast<int64_t>(mid * varShape_[1] * updatesDtypeSize_), ubBlock);
        int64_t tmpTotalSize = totalIndexSize + sortTmpSize + updateSize;
        if (tmpTotalSize <= static_cast<int64_t>(ubSize_)) {
            start = mid;
        } else {
            end = mid;
        }
    }

    indicesFactor_ = static_cast<uint64_t>(start);
    uint64_t totalLoop = Ops::Base::CeilDiv(static_cast<uint64_t>(indicesNum_), indicesFactor_);
    // 按照UB总循环次数分核
    normBlockLoop_ = Ops::Base::CeilDiv(totalLoop, totalCoreNum_);
    usedCoreNum_ = Ops::Base::CeilDiv(totalLoop, normBlockLoop_);
    tailBlockLoop_ = totalLoop - normBlockLoop_ * (usedCoreNum_ - 1);
    tailBlockTail_ = indicesNum_ - indicesFactor_ * normBlockLoop_ * (usedCoreNum_ - 1) - indicesFactor_ * (tailBlockLoop_ - 1);
    while(usedCoreNum_ <= totalCoreNum_ / DOUBLE && indicesFactor_ > 1) {
        indicesFactor_ = indicesFactor_ / DOUBLE;
        totalLoop = Ops::Base::CeilDiv(static_cast<uint64_t>(indicesNum_), indicesFactor_);
        normBlockLoop_ = Ops::Base::CeilDiv(totalLoop, totalCoreNum_);
        usedCoreNum_ = Ops::Base::CeilDiv(totalLoop, normBlockLoop_);
        tailBlockLoop_ = totalLoop - normBlockLoop_ * (usedCoreNum_ - 1);
        tailBlockTail_ = indicesNum_ - indicesFactor_ * normBlockLoop_ * (usedCoreNum_ - 1) - indicesFactor_ * (tailBlockLoop_ - 1);
    }
    normBlockIndices_ = indicesFactor_ * normBlockLoop_;
    return ge::GRAPH_SUCCESS;
}

void ScatterAddTiling::SetTilingData()
{
    tilingData_.set_blockFactor(blockFactor_);
    tilingData_.set_tailBlockFactor(tailBlockFactor_);
    tilingData_.set_ubFactor(ubFactor_);
    tilingData_.set_tailUbFactor(tailUbFactor_);
    tilingData_.set_tailCoreTailUbFactor(tailCoreTailUbFactor_);
    tilingData_.set_indicesSize(indicesNum_);
    tilingData_.set_varShape(varShape_);
    tilingData_.set_indicesUbFactor(indicesUbFactor_);
    tilingData_.set_indicesLoopSize(indicesLoopSize_);
    tilingData_.set_indicesTailUbFactor(indicesTailUbFactor_);
    tilingData_.set_tailCoreIndicesLoopSize(tailCoreIndicesLoopSize_);
    tilingData_.set_tailCoreIndicesTailUbFactor(tailCoreIndicesTailUbFactor_);
    tilingData_.set_updatesUbFactor(updatesUbFactor_);
    tilingData_.set_updatesLoopSize(updatesLoopSize_);
    tilingData_.set_updatesTailUbFactor(updatesTailUbFactor_);
    tilingData_.set_copyCoreNum(copyCoreNum_);
    tilingData_.set_perCoreHandleVar(perCoreHandleVar_);
    tilingData_.set_atomicAddCoreNum(atomicAddCoreNum_);
    tilingData_.set_perCoreHandleIndices(perCoreHandleIndices_);
    tilingData_.set_postAxisSize(postAxisSize_);
    tilingData_.set_perCoreHandleCol(perCoreHandleCol_);
    tilingData_.set_logicCoreNum(logicCoreNum_);
    tilingData_.set_isDeterministic(isDeterministic_);
    tilingData_.set_tailCoreHandleCol(tailCoreHandleCol_);
    tilingData_.set_tailCoreColsLoopSize(tailCoreColsLoopSize_);
    tilingData_.set_tailCoreColsTailUbFactor(tailCoreColsTailUbFactor_);
    tilingData_.set_perCoreHandleRows(perCoreHandleRows_);
    tilingData_.set_tailCoreHandleRows(tailCoreHandleRows_);
    tilingData_.set_rowsInUb(rowsInUb_);
    tilingData_.set_deQuantizeCoreNum(deQuantizeCoreNum_);
    tilingData_.set_isDeterminTemplate(isDeterminTemplate_);
    tilingData_.set_dequantizeBranch(dequantizeBranch_);
    tilingData_.set_normBlockIndices(normBlockIndices_);
    tilingData_.set_indicesFactor(indicesFactor_);
    tilingData_.set_normBlockLoop(normBlockLoop_);
    tilingData_.set_tailBlockLoop(tailBlockLoop_);
    tilingData_.set_tailBlockTail(tailBlockTail_);
    tilingData_.set_sortCoreNum(usedCoreNum_);
    tilingData_.set_rowTileNum(rowTileNum_);
    tilingData_.set_colTileNum(colTileNum_);
    tilingData_.set_normBlockRow(normBlockRow_);
    tilingData_.set_tailBlockRow(tailBlockRow_);
    tilingData_.set_normBlockCol(normBlockCol_);
    tilingData_.set_tailBlockCol(tailBlockCol_);
    tilingData_.set_ubFactorRow(ubFactorRow_);
    tilingData_.set_ubFactorCol(ubFactorCol_);
    tilingData_.set_indicesCastMode(indicesCastMode_);
}

ge::graphStatus ScatterAddTiling::GetCastType()
{
    indicesCastDtype_ = indicesDtype_;

    if (indicesDtype_ == ge::DT_INT32) {
        if (varShape_[0] < UINT8_MAX) {
            indicesCastMode_ = CASTMODE4;          // int32 Cast uint8
            indicesCastDtype_ = ge::DT_UINT8;
        } else if (varShape_[0] < INT16_MAX) {
            indicesCastMode_ = CASTMODE1;          // int32 Cast int16
            indicesCastDtype_ = ge::DT_INT16;
        }
    } else {
        if (varShape_[0] < UINT8_MAX) {
            indicesCastMode_ = CASTMODE5;          // int64 Cast uint8
            indicesCastDtype_ = ge::DT_UINT8;
        } else if (varShape_[0] < INT16_MAX) {
            indicesCastMode_ = CASTMODE3;          // int64 Cast int16
            indicesCastDtype_ = ge::DT_INT16;
        } else if (varShape_[0] < INT32_MAX) {
            indicesCastMode_ = CASTMODE2;          // int64 Cast int32
            indicesCastDtype_ = ge::DT_INT32;
        }
    }

    if (indicesCastMode_ != 0) {
        indicesCastDtypeSize_ = ge::GetSizeByDataType(indicesCastDtype_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterAddTiling::DoOpTiling()
{
    if (varShape_[0] * varShape_[1] * indicesNum_ == 0) {
        usedCoreNum_ = 1;
        SetTilingData();
        return ge::GRAPH_SUCCESS;
    }

    if (isDeterminTemplate_) {
        OP_CHECK_IF(ScatterAddDeterministicTiling() != ge::GRAPH_SUCCESS,
                    OP_LOGE(opName, "ScatterAddDeterministicTiling fail."), return ge::GRAPH_FAILED);
        SetTilingData();
        return ge::GRAPH_SUCCESS;
    }

    bool supportAtomicAdd =
        isSimt_ ? SIMT_ATOMIC_ADD_NOT_SUPPORT_DTYPE.find(varDtype_) == SIMT_ATOMIC_ADD_NOT_SUPPORT_DTYPE.end()
                : SIMD_ATOMIC_ADD_NOT_SUPPORT_DTYPE.find(varDtype_) == SIMD_ATOMIC_ADD_NOT_SUPPORT_DTYPE.end();
    if (!supportAtomicAdd) {      // 不支持AtomicAdd的数据类型，先走 Tiling CopyCompute计算类型转换的tiling数据，需要分配Cast的空间
        OP_CHECK_IF(TilingCopyCompute() != ge::GRAPH_SUCCESS,
                        OP_LOGE(opName, "TilingCopyCompute fail."), return ge::GRAPH_FAILED);
    }

    if (!isSimt_) {
        templateKey_ = UPDATES_IN_SIMD;
        if (!supportAtomicAdd) {
            isSort_ = 0;
            OP_CHECK_IF(TilingSimdNotSupportAtomicAddCompute(supportAtomicAdd) != ge::GRAPH_SUCCESS,
                            OP_LOGE(opName, "TilingSimdNotSupportAtomicAddCompute fail."),
                            return ge::GRAPH_FAILED);
        } else {
            if (isSort_) {
                OP_CHECK_IF(TilingSimdSupportAtomicAddSortCompute() != ge::GRAPH_SUCCESS,
                                OP_LOGE(opName, "TilingSimdSupportAtomicAddSortCompute fail."),
                                return ge::GRAPH_FAILED);
            } else {
                OP_CHECK_IF(TilingSimdSupportAtomicAddCompute() != ge::GRAPH_SUCCESS,
                                OP_LOGE(opName, "TilingSimdSupportAtomicAddCompute fail."),
                                return ge::GRAPH_FAILED);
            }
        }
    } else {
        if (isSort_) {
            OP_CHECK_IF(TilingSimtSort() != ge::GRAPH_SUCCESS,
                            OP_LOGE(opName, "TilingSimtSort fail."),
                            return ge::GRAPH_FAILED);
        }
    }

    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterAddTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t ScatterAddTiling::GetTilingKey() const
{
    if (varShape_[0] * varShape_[1] * indicesNum_ == 0) {
        return 0;
    }
    uint32_t sizeAddrType = ((varShape_[1] * indicesNum_ > INT32_MAX) || (varSize_ > INT32_MAX)) ? 1 : 0;
    if (!isSimt_) {
        sizeAddrType = 0;
    }

    uint64_t tilingKey = 0;
    tilingKey =
        Ops::NN::Optiling::GET_TILINGKEY(isSort_, templateKey_, sizeAddrType, isUpdateScalar_, TILING_KEY_PLACE_HOLD,
                    TILING_KEY_PLACE_HOLD, TILING_KEY_PLACE_HOLD, TILING_KEY_PLACE_HOLD, TILING_KEY_PLACE_HOLD);

    OP_LOGD("ScatterAddTiling", "tilingKey is %lu", tilingKey);
    return tilingKey;
}

ge::graphStatus ScatterAddTiling::GetWorkspaceSize()
{
    workspaceSize_ = ASCENDC_TOOLS_WORKSPACE;
    if(!isSort_) {
        if ((isSimt_ && varDtype_ == ge::DT_INT8) || varDtype_ == ge::DT_UINT8) {
            workspaceSize_ += castTypeSize_ * varSize_;
        }
    }

    if (isDeterminTemplate_ && isDeterministic_) {
        uint64_t postVarAlignSize = Ops::Base::CeilAlign(varShape_[1] * sizeof(float), static_cast<uint64_t>(Ops::Base::GetUbBlockSize(context_))) / sizeof(float);
        workspaceSize_ += logicCoreNum_ * perCoreHandleIndices_ * indicesDtypeSize_ + logicCoreNum_ * perCoreHandleIndices_ * postVarAlignSize * FLOAT_BYTES + 
            varSize_ * DOUBLE * UINT32_BYTES + varShape_[0] * UINT32_BYTES;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterAddTiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    tilingKey_ = GetTilingKey();
    context_->SetTilingKey(tilingKey_);

    if (varShape_[0] * varShape_[1] * indicesNum_ != 0) {
        if (!isSort_) {
            usedCoreNum_ = totalCoreNum_;
        }
        if (!isSimt_ && !isDeterminTemplate_) {
            usedCoreNum_ = copyCoreNum_ > atomicAddCoreNum_ ? copyCoreNum_ : atomicAddCoreNum_;
        }
    }

    context_->SetBlockDim(usedCoreNum_);
    context_->SetScheduleMode(1);
    auto res = context_->SetLocalMemorySize(ubSize_);
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS),
                    OP_LOGE(opName, "SetLocalMemorySize ubSize = %lu failed.", ubSize_),
                    return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void ScatterAddTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "varShape: " << ToString(tilingData_.get_varShape(), VAR_SHAPE_LENGTH).c_str() << std::endl;
    info << "blockFactor: " << tilingData_.get_blockFactor() << std::endl;
    info << "tailBlockFactor: " << tilingData_.get_tailBlockFactor() << std::endl;
    info << "ubFactor: " << tilingData_.get_ubFactor() << std::endl;
    info << "tailUbFactor: " << tilingData_.get_tailUbFactor() << std::endl;
    info << "tailCoreTailUbFactor: " << tilingData_.get_tailCoreTailUbFactor() << std::endl;
    info << "indicesSize: " << tilingData_.get_indicesSize() << std::endl;
    info << "indicesUbFactor: " << tilingData_.get_indicesUbFactor() << std::endl;
    info << "indicesLoopSize: " << tilingData_.get_indicesLoopSize() << std::endl;
    info << "indicesTailUbFactor: " << tilingData_.get_indicesTailUbFactor() << std::endl;
    info << "tailCoreIndicesLoopSize: " << tilingData_.get_tailCoreIndicesLoopSize() << std::endl;
    info << "tailCoreIndicesTailUbFactor: " << tilingData_.get_tailCoreIndicesTailUbFactor() << std::endl;
    info << "updatesUbFactor: " << tilingData_.get_updatesUbFactor() << std::endl;
    info << "updatesLoopSize: " << tilingData_.get_updatesLoopSize() << std::endl;
    info << "updatesTailUbFactor: " << tilingData_.get_updatesTailUbFactor() << std::endl;
    info << "copyCoreNum: " << tilingData_.get_copyCoreNum() << std::endl;
    info << "perCoreHandleVar: " << tilingData_.get_perCoreHandleVar() << std::endl;
    info << "atomicAddCoreNum: " << tilingData_.get_atomicAddCoreNum() << std::endl;
    info << "perCoreHandleIndices: " << tilingData_.get_perCoreHandleIndices() << std::endl;
    info << "postAxisSize: " << tilingData_.get_postAxisSize() << std::endl;
    info << "perCoreHandleCol: " << tilingData_.get_perCoreHandleCol() << std::endl;
    info << "logicCoreNum: " << tilingData_.get_logicCoreNum() << std::endl;
    info << "isDeterministic: " << tilingData_.get_isDeterministic() << std::endl;
    info << "tailCoreHandleCol: " << tilingData_.get_tailCoreHandleCol() << std::endl;
    info << "tailCoreColsLoopSize: " << tilingData_.get_tailCoreColsLoopSize() << std::endl;
    info << "tailCoreColsTailUbFactor: " << tilingData_.get_tailCoreColsTailUbFactor() << std::endl;
    info << "perCoreHandleRows: " << tilingData_.get_perCoreHandleRows() << std::endl;
    info << "tailCoreHandleRows: " << tilingData_.get_tailCoreHandleRows() << std::endl;
    info << "rowsInUb: " << tilingData_.get_rowsInUb() << std::endl;
    info << "deQuantizeCoreNum: " << tilingData_.get_deQuantizeCoreNum() << std::endl;
    info << "isDeterminTemplate:" << tilingData_.get_isDeterminTemplate() << std::endl;
    info << "dequantizeBranch:" << tilingData_.get_dequantizeBranch() << std::endl;
    info << "normBlockIndices:" << tilingData_.get_normBlockIndices() << std::endl;
    info << "indicesFactor:" << tilingData_.get_indicesFactor() << std::endl;
    info << "normBlockLoop:" << tilingData_.get_normBlockLoop() << std::endl;
    info << "tailBlockLoop:" << tilingData_.get_tailBlockLoop() << std::endl;
    info << "tailBlockTail:" << tilingData_.get_tailBlockTail() << std::endl;
    info << "sortCoreNum:" << tilingData_.get_sortCoreNum() << std::endl;
    info << "rowTileNum:" << tilingData_.get_rowTileNum() << std::endl;
    info << "colTileNum:" << tilingData_.get_colTileNum() << std::endl;
    info << "normBlockRow:" << tilingData_.get_normBlockRow() << std::endl;
    info << "tailBlockRow:" << tilingData_.get_tailBlockRow() << std::endl;
    info << "normBlockCol:" << tilingData_.get_normBlockCol() << std::endl;
    info << "tailBlockCol:" << tilingData_.get_tailBlockCol() << std::endl;
    info << "ubFactorRow:" << tilingData_.get_ubFactorRow() << std::endl;
    info << "ubFactorCol:" << tilingData_.get_ubFactorCol() << std::endl;
    info << "indicesCastMode: " << tilingData_.get_indicesCastMode() << std::endl;
    OP_LOGI(opName, "Tiling inf is: %s", info.str().c_str());
}

ge::graphStatus ScatterAddTilingForAscendC(gert::TilingContext* context)
{
    ScatterAddTiling tiling(context);
    return tiling.DoTiling();
}
}  // namespace optiling
