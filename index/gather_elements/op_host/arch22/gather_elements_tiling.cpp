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
 * \file gather_elements_tiling.cpp
 * \brief
 */
#include "gather_elements_tiling.h"
#include "error_util.h"
#include "log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_util.h"
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include <algorithm>
#include <set>
#include <vector>

namespace {
const int32_t BLOCK_SIZE = 32;
const int32_t PARAMS_CUT_INTO_SLICE_UB = 149984;
const int32_t PARAMS_CARRY_BLOCK_UB = 730 * 1024;
const int32_t CONVERT_TO_AICPU_UB = 3000 * 1024;
const int32_t INDICES_NUM_THRESHOULD = 2048;
const int32_t RESERVED_UB_SIZE = 2 * 1024;
const int32_t INT_MAX_NUM = 2147483647;
const int32_t HALF = 2;

// A: params larger than cache_ub
// B: indices larger than the number contained in one block for each core
// C: remaining indices larger than one block

const int64_t TILING_MODE_X_LARGE_INDICES_LARGE = 1;
const int64_t TILING_MODE_X_SMALL_INDICES_LARGE = 2;
const int64_t TILING_MODE_X_SLICE_INDICES_LARGE = 3;
const int64_t TILING_MODE_DIF = 3;
// tiling mode when params and indices are so large that both are cut into slices
const int64_t TILING_MODE_FOR_LAST_AXIS = 7;
const int64_t TILING_MODE_FOR_LAST_AXIS_GATHER = 8;
const int64_t TILING_MODE_FOR_LAST_AXIS_DIFF_SHAPE = 9;
const int64_t TILING_MODE_FOR_LAST_AXIS_CUT_GATHER = 10;

const size_t DIM_0 = 0;
const size_t DIM_1 = 1;
const size_t DIM_2 = 2;
const size_t DIM_3 = 3;
const size_t DIM_4 = 4;
const size_t DIM_5 = 5;
const size_t DIM_6 = 6;
const size_t DIM_7 = 7;

const size_t MAX_DIMS = 8;
const int64_t PARAMS_AXIS_PRE_NONE = 1;
const int64_t LEAST_REPEAT_TIME = 1;

const size_t INDEX_ATTR_AXIS = 0;
const size_t SIZE_INT32 = 4;
const std::string OP_NAME = "GatherElements";

const std::set<ge::DataType> GATHER_DTYPES = {ge::DT_INT16, ge::DT_UINT16, ge::DT_FLOAT16, ge::DT_BF16,
                                              ge::DT_INT32, ge::DT_UINT32, ge::DT_FLOAT};
constexpr int64_t V2_RESERVED_UB_SIZE = 2048;
constexpr int64_t V2_CACHELINE = 512;
constexpr int64_t V2_BLOCK_SIZE = 32;
constexpr int64_t V2_TRANSPOSE_WS_LEN = 128;
constexpr int64_t V2_NUM_TWO = 2;
constexpr int64_t V2_UB_LIMIT = 120000;
constexpr int64_t V2_ASCEND_910B_CORE_NUM = 48;
constexpr int64_t V2_ASCEND_910B_UB = 196608;
constexpr int64_t V2_POST_DIM_LIMIT = 32;
constexpr int64_t V2_TRANS_LEN = 16;
constexpr int64_t V2_MODE_SCALAR = 0;
constexpr int64_t V2_MODE_TRANSPOSE = 1;
constexpr int64_t V2_MODE_LASTDIM = 2;
const std::set<ge::DataType> GATHER_ELEMENTS_V2_DTYPES = {ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
constexpr int64_t V2LD_RESERVED_UB = 10 * 1024;
constexpr int64_t V2LD_NUM_ONE = 1;
constexpr int64_t V2LD_INT8_DSIZE = 1;
constexpr int64_t V2LD_INT6_DSIZE = 2;
constexpr int64_t V2LD_INT32_DSIZE = 4;
constexpr int64_t V2LD_INT64_DSIZE = 8;
constexpr int64_t V2LD_BLOCK_SIZE = 32;
constexpr int64_t V2LD_MAX_SLICE_NUM = 5;
constexpr int64_t V2LD_MAX_SIZE_RATIO = 256;
constexpr int64_t V2LD_DOUBLE_TIME = 2;
constexpr float V2LD_MASK_SIZE_RATE = 1.0f / 4;

template <typename T, typename U>
inline T ceilAlign(T value, U factor)
{
    return (factor == 0) ? value :
                           ((value + static_cast<T>(factor) - 1) / static_cast<T>(factor)) * static_cast<T>(factor);
}

template <typename T, typename U>
inline T ceilDiv(T value, U factor)
{
    return (factor == 0) ? value : (value + static_cast<T>(factor) - 1) / static_cast<T>(factor);
}

template <typename T, typename U>
inline T floorDiv(T value, U factor)
{
    return (factor == 0) ? value : value / static_cast<T>(factor);
}

} // namespace

namespace optiling {

class GatherElementsTiling {
public:
    explicit GatherElementsTiling(gert::TilingContext* context) : tilingContext_(context) {}
    ge::graphStatus Init();
    ge::graphStatus SetKernelTiling();
    void TilingDataPrint();

private:
    ge::graphStatus GetPlatformInfo();
    bool CheckTensorShape() const;
    bool IfParamsIndicesSameShapeExceptAxis() const;
    void FalseAxis(int32_t axis, int32_t paramsDims);
    void RecordTilingCommonInformation(int32_t axis, int32_t paramsDims);
    void RecordShapeInformation();
    void ConfirmCutIntoSliceInformation();
    void ConfirmIndicesLoopInformation(int64_t availableUbSize);
    void ChooseTilingModeSameShapeExceptAxis();
    bool LastAxisCutIntoSlices(bool sameShapeExceptAxisFlag, int64_t lastDimSize, int64_t availableUbSize);
    int64_t CalcuRepeatUnaligned(int32_t indicesAxis, int32_t largeNumPerBlock) const;
    bool ChooseTilingModeForLastAxis(bool ifSameDimValueExceptAxis);
    bool ChooseTilingMode(bool ifSameDimValueExceptAxis);

    gert::TilingContext* tilingContext_ = nullptr;
    const GatherElementsCompileInfo* compileInfo_ = nullptr;
    int64_t coreNumAll_ = 0;
    int64_t ubSize_ = 0;
    GatherElementsTilingData tilingData_;
    CommonInformation commonInformation_;
    gert::Shape paramsShape_;
    gert::Shape indicesShape_;
    gert::Shape yShape_;
    int32_t axis_ = 0;
    int32_t dims_ = 0;
    int32_t paramsDsize_ = 0;
    int32_t indicesDsize_ = 0;
    int32_t supportGather_ = 0;

    bool useV2_ = false;
    int64_t v2UsedCoreNum_ = 0;
    int64_t v2Workspace_ = 0;

    ge::graphStatus TryRouteToV2();
    ge::graphStatus ComputeV2LastDimTiling();
    bool IfUseV2() const;
    bool V2MemCheck() const;
};

// 获取平台信息比如CoreNum、UB资源大小；platformInfo 为空时回退到 compileInfo 中 TilingPrepare 阶段保存的值
ge::graphStatus GatherElementsTiling::GetPlatformInfo()
{
    auto platformInfo = tilingContext_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_CHECK_IF(compileInfo_ == nullptr, OP_LOGE(tilingContext_->GetNodeName(), "compile info is null"),
                    return ge::GRAPH_FAILED);
        coreNumAll_ = compileInfo_->core_num;
        ubSize_ = compileInfo_->ub_size;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNumAll_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = static_cast<int64_t>(ubSizePlatForm);
    }
    return ge::GRAPH_SUCCESS;
}

bool GatherElementsTiling::CheckTensorShape() const
{
    int32_t paramsDims = paramsShape_.GetDimNum();
    int32_t indicesDims = indicesShape_.GetDimNum();
    int32_t yDims = yShape_.GetDimNum();
    if (paramsDims != yDims) {
        OP_LOGE(tilingContext_->GetNodeName(),
                "op [GatherElementsTiling] : CheckTensorShape, x shape range and index shape range is not the same.");
        return false;
    }
    if (yDims != indicesDims) {
        OP_LOGE(tilingContext_->GetNodeName(), "op [GatherElementsTiling] : CheckTensorShape, y Shape is invalid.");
        return false;
    }
    for (int32_t i = 0; i < yDims; i++) {
        if (yShape_.GetDim(i) != indicesShape_.GetDim(i)) {
            OP_LOGE(tilingContext_->GetNodeName(),
                    "op [GatherElementsTiling] : CheckTensorShape, y Shape dim is invalid.");
            return false;
        }
    }
    return true;
}

// confirm information of param_slice.
void GatherElementsTiling::ConfirmCutIntoSliceInformation()
{
    tilingData_.set_slice_num(
        ceilDiv(tilingData_.get_params_total() * commonInformation_.params_dsize, PARAMS_CUT_INTO_SLICE_UB));
    tilingData_.set_slice_thickness_once(ceilDiv(tilingData_.get_params_total(), tilingData_.get_slice_num()));
    tilingData_.set_slice_thickness_once(
        ceilDiv(tilingData_.get_slice_thickness_once(), commonInformation_.params_block_num) *
        commonInformation_.params_block_num);
    tilingData_.set_slice_thickness_last(tilingData_.get_params_total() -
                                         tilingData_.get_slice_thickness_once() * (tilingData_.get_slice_num() - 1));
}

void GatherElementsTiling::ConfirmIndicesLoopInformation(int64_t availableUbSize)
{
    OP_TILING_CHECK(commonInformation_.indices_block_num == 0 || commonInformation_.params_block_num == 0 ||
                        commonInformation_.indices_block_num_large == 0,
                    VECTOR_INNER_ERR_REPORT_TILIING("GatherElements",
                                                    "In the ConfirmIndicesLoopInformation function, the divisor is 0."),
                    return);
    int64_t indicesNumEachLoop = availableUbSize / (static_cast<int64_t>(commonInformation_.indices_dsize) +
                                                    static_cast<int64_t>(commonInformation_.params_dsize));
    indicesNumEachLoop = std::min(
        indicesNumEachLoop / commonInformation_.indices_block_num * commonInformation_.indices_block_num,
        indicesNumEachLoop / commonInformation_.params_block_num * commonInformation_.params_block_num);

    tilingData_.set_indices_num_each_core(tilingData_.get_indices_num() / tilingData_.get_need_core_num() /
                                          commonInformation_.indices_block_num_large *
                                          commonInformation_.indices_block_num_large);
    tilingData_.set_indices_num_remaining(tilingData_.get_indices_num() -
                                          tilingData_.get_need_core_num() * tilingData_.get_indices_num_each_core());
    OP_TILING_CHECK(indicesNumEachLoop == static_cast<int64_t>(0),
                    VECTOR_INNER_ERR_REPORT_TILIING("GatherElements",
                                                    "In the ConfirmIndicesLoopInformation function, the divisor is 0."),
                    return);
    tilingData_.set_indices_loop_num(tilingData_.get_indices_num_each_core() / indicesNumEachLoop);
    tilingData_.set_indices_row_num_once(indicesNumEachLoop);
    tilingData_.set_indices_row_num_last(tilingData_.get_indices_num_each_core() %
                                         tilingData_.get_indices_row_num_once());
    tilingData_.set_remaining_block_remain(tilingData_.get_indices_num_remaining() %
                                           commonInformation_.indices_block_num_large);
    tilingData_.set_remaining_block_num(tilingData_.get_indices_num_remaining() /
                                        commonInformation_.indices_block_num_large);
}

bool GatherElementsTiling::IfParamsIndicesSameShapeExceptAxis() const
{
    for (int32_t i = 0; i < dims_; i++) {
        if ((i != axis_) && (paramsShape_.GetDim(i) != indicesShape_.GetDim(i))) {
            return false;
        }
    }
    return true;
}

// record shape information
void GatherElementsTiling::RecordShapeInformation()
{
    std::vector<int64_t> paramsShapePerDim(MAX_DIMS, 0);
    std::vector<int64_t> indicesShapePerDim(MAX_DIMS, 0);
    for (int32_t i = 0; i < dims_; i++) {
        paramsShapePerDim[i] = paramsShape_.GetDim(i);
        indicesShapePerDim[i] = indicesShape_.GetDim(i);
    }
    tilingData_.set_indices_shape_0(indicesShapePerDim[DIM_0]);
    tilingData_.set_indices_shape_1(indicesShapePerDim[DIM_1]);
    tilingData_.set_indices_shape_2(indicesShapePerDim[DIM_2]);
    tilingData_.set_indices_shape_3(indicesShapePerDim[DIM_3]);
    tilingData_.set_indices_shape_4(indicesShapePerDim[DIM_4]);
    tilingData_.set_indices_shape_5(indicesShapePerDim[DIM_5]);
    tilingData_.set_indices_shape_6(indicesShapePerDim[DIM_6]);
    tilingData_.set_indices_shape_7(indicesShapePerDim[DIM_7]);

    tilingData_.set_params_shape_0(paramsShapePerDim[DIM_0]);
    tilingData_.set_params_shape_1(paramsShapePerDim[DIM_1]);
    tilingData_.set_params_shape_2(paramsShapePerDim[DIM_2]);
    tilingData_.set_params_shape_3(paramsShapePerDim[DIM_3]);
    tilingData_.set_params_shape_4(paramsShapePerDim[DIM_4]);
    tilingData_.set_params_shape_5(paramsShapePerDim[DIM_5]);
    tilingData_.set_params_shape_6(paramsShapePerDim[DIM_6]);
    tilingData_.set_params_shape_7(paramsShapePerDim[DIM_7]);
}

// params shape convert to 3D:[params_pre, params_axis, params_row]
// indices shape convert to 1D:[indices_num]
// output tensor, y shape convert to:[params_pre, indices_num, params_row]
void GatherElementsTiling::FalseAxis(int32_t axis, int32_t paramsDims)
{
    tilingData_.set_params_pre(PARAMS_AXIS_PRE_NONE);
    tilingData_.set_params_row(PARAMS_AXIS_PRE_NONE);
    for (int32_t i = 0; i < axis; i++) {
        tilingData_.set_params_pre(tilingData_.get_params_pre() * paramsShape_.GetDim(i));
    }
    tilingData_.set_params_axis(paramsShape_.GetDim(axis));
    tilingData_.set_indices_axis(indicesShape_.GetDim(axis));
    if (axis + 1 < paramsDims) {
        for (int32_t i = axis + 1; i < paramsDims; i++) {
            tilingData_.set_params_row(tilingData_.get_params_row() * paramsShape_.GetDim(i));
        }
    }
}

void GatherElementsTiling::RecordTilingCommonInformation(int32_t axis, int32_t paramsDims)
{
    FalseAxis(axis, paramsDims);
    commonInformation_.indices_pre = PARAMS_AXIS_PRE_NONE;
    for (int32_t i = 0; i < axis; i++) {
        commonInformation_.indices_pre *= indicesShape_.GetDim(i);
    }
    int64_t paramsExceptPre = 1;
    for (int32_t i = axis; i < dims_; i++) {
        paramsExceptPre *= paramsShape_.GetDim(i);
    }
    commonInformation_.params_except_pre_size = ceilDiv(paramsExceptPre * commonInformation_.params_dsize, BLOCK_SIZE) *
                                                BLOCK_SIZE;
    commonInformation_.params_dsize = paramsDsize_;
    commonInformation_.indices_dsize = indicesDsize_;
    commonInformation_.params_block_num = BLOCK_SIZE / commonInformation_.params_dsize;
    commonInformation_.indices_block_num = BLOCK_SIZE / commonInformation_.indices_dsize;
    commonInformation_.large_num_per_block = std::max(commonInformation_.params_block_num,
                                                      commonInformation_.indices_block_num);
    int32_t paramSmallerThanIndices = std::max(commonInformation_.indices_dsize / commonInformation_.params_dsize,
                                               static_cast<int32_t>(1));
    commonInformation_.indices_block_num_large = paramSmallerThanIndices * commonInformation_.indices_block_num;
    tilingData_.set_params_total(paramsShape_.GetShapeSize());
    tilingData_.set_indices_num(indicesShape_.GetShapeSize());
    tilingData_.set_slice_num(LEAST_REPEAT_TIME);
    commonInformation_.params_total_ceil = ceilDiv(tilingData_.get_params_total(),
                                                   commonInformation_.params_block_num) *
                                           commonInformation_.params_block_num;
    commonInformation_.params_total_ceil_size = commonInformation_.params_total_ceil * commonInformation_.params_dsize;
}

void GatherElementsTiling::ChooseTilingModeSameShapeExceptAxis()
{
    int64_t xUbSize = 0;
    tilingData_.set_need_core_num(coreNumAll_);
    if (commonInformation_.params_total_ceil_size > PARAMS_CARRY_BLOCK_UB) {
        tilingData_.set_tilingMode(TILING_MODE_X_LARGE_INDICES_LARGE);
    } else if (commonInformation_.params_total_ceil_size > PARAMS_CUT_INTO_SLICE_UB &&
               commonInformation_.params_total_ceil_size <= PARAMS_CARRY_BLOCK_UB) {
        ConfirmCutIntoSliceInformation();
        tilingData_.set_tilingMode(TILING_MODE_X_SLICE_INDICES_LARGE);
        xUbSize = tilingData_.get_slice_thickness_once() * commonInformation_.params_dsize;
    } else {
        tilingData_.set_tilingMode(TILING_MODE_X_SMALL_INDICES_LARGE);
        xUbSize = commonInformation_.params_total_ceil_size;
    }
    int64_t availableUbSize = ubSize_ - xUbSize - RESERVED_UB_SIZE;
    ConfirmIndicesLoopInformation(availableUbSize);
}

bool GatherElementsTiling::LastAxisCutIntoSlices(bool sameShapeExceptAxisFlag, int64_t lastDimSize,
                                                 int64_t availableUbSize)
{
    if (sameShapeExceptAxisFlag && lastDimSize >= availableUbSize &&
        tilingData_.get_params_axis() * commonInformation_.params_dsize <= availableUbSize / HALF &&
        tilingData_.get_indices_axis() % commonInformation_.large_num_per_block == 0) {
        const int64_t repeatPerCore = tilingData_.get_repeat_per_core();
        const int64_t paramsAxis = tilingData_.get_params_axis();
        const int64_t paramsDsize = commonInformation_.params_dsize;
        const int64_t indicesDsize = commonInformation_.indices_dsize;
        const int64_t blockSizeX = BLOCK_SIZE / paramsDsize;
        const int64_t xAligned = ceilDiv(paramsAxis, blockSizeX) * blockSizeX;
        const int64_t fixedPerRow = xAligned * paramsDsize;
        const int64_t perElem = indicesDsize + paramsDsize + SIZE_INT32 * 2;
        int64_t cutSlice = (availableUbSize - repeatPerCore * fixedPerRow) / (repeatPerCore * perElem) /
                           commonInformation_.large_num_per_block * commonInformation_.large_num_per_block;
        if (cutSlice <= 0) {
            return false;
        }
        int64_t sliceNum = ceilDiv(tilingData_.get_indices_axis(), cutSlice);
        int64_t lastSlice = tilingData_.get_indices_axis() - (sliceNum - 1) * cutSlice;
        tilingData_.set_slice_num(sliceNum);
        tilingData_.set_slice_thickness_once(cutSlice);
        tilingData_.set_slice_thickness_last(lastSlice);
        tilingData_.set_tilingMode(TILING_MODE_FOR_LAST_AXIS_CUT_GATHER);
        return true;
    }
    return false;
}

int64_t GatherElementsTiling::CalcuRepeatUnaligned(int32_t indicesAxis, int32_t largeNumPerBlock) const
{
    int64_t repeatPerCore = LEAST_REPEAT_TIME;
    if (largeNumPerBlock == static_cast<int32_t>(0)) {
        return repeatPerCore;
    }
    while (repeatPerCore * indicesAxis % largeNumPerBlock != 0) {
        repeatPerCore++;
    }
    return repeatPerCore;
}

static int64_t Gcd(int64_t a, int64_t b)
{
    while (b != 0) {
        int64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

bool GatherElementsTiling::ChooseTilingModeForLastAxis(bool ifSameDimValueExceptAxis)
{
    int64_t repeatPerCore = LEAST_REPEAT_TIME;
    if (!ifSameDimValueExceptAxis && tilingData_.get_indices_axis() < commonInformation_.large_num_per_block) {
        return false;
    }
    // Normal branches
    tilingData_.set_tilingMode(ifSameDimValueExceptAxis ? TILING_MODE_FOR_LAST_AXIS :
                                                          TILING_MODE_FOR_LAST_AXIS_DIFF_SHAPE);
    if (ifSameDimValueExceptAxis && tilingData_.get_indices_axis() % commonInformation_.large_num_per_block != 0) {
        // unaligned cases
        repeatPerCore = CalcuRepeatUnaligned(tilingData_.get_indices_axis(), commonInformation_.large_num_per_block);
    }

    OP_TILING_CHECK(repeatPerCore == 0, OP_LOGW("GatherElements", "op GatherElementsTiling: while repeatPerCore is 0."),
                    return false);
    int64_t availableUbSize = ubSize_ - static_cast<int64_t>(RESERVED_UB_SIZE);

    const int64_t TASKS_PER_CORE_TARGET = 8;
    if (ifSameDimValueExceptAxis && supportGather_ == 0 && availableUbSize > 0) {
        const int64_t indicesAxis = tilingData_.get_indices_axis();
        const int64_t paramsAxis = tilingData_.get_params_axis();
        const int64_t blockSizeX = BLOCK_SIZE / commonInformation_.params_dsize;
        const int64_t blockSizeIdx = BLOCK_SIZE / commonInformation_.indices_dsize;
        const int64_t blockSizeIdx32 = BLOCK_SIZE / static_cast<int64_t>(SIZE_INT32);
        const int64_t xAligned = ceilDiv(paramsAxis, blockSizeX) * blockSizeX;
        const int64_t idxAligned = ceilDiv(indicesAxis, blockSizeIdx) * blockSizeIdx;
        const int64_t resAligned = ceilDiv(indicesAxis, blockSizeX) * blockSizeX;
        const int64_t idx32Aligned = ceilDiv(indicesAxis, blockSizeIdx32) * blockSizeIdx32;
        const int64_t perRowBytes = xAligned * commonInformation_.params_dsize +
                                    idxAligned * commonInformation_.indices_dsize +
                                    resAligned * commonInformation_.params_dsize +
                                    idx32Aligned * 2 * static_cast<int64_t>(SIZE_INT32) +
                                    indicesAxis * static_cast<int64_t>(SIZE_INT32);
        const int64_t maxByUb = perRowBytes > 0 ? (availableUbSize - 1) / perRowBytes : repeatPerCore;
        const int64_t coreNum = coreNumAll_;
        const int64_t maxByRounds = (coreNum > 0) ?
                                        (commonInformation_.indices_pre / (TASKS_PER_CORE_TARGET * coreNum)) :
                                        repeatPerCore;
        const int64_t cap = std::min(maxByUb, maxByRounds > 0 ? maxByRounds : repeatPerCore);
        const int64_t large = commonInformation_.large_num_per_block;
        const int64_t step = (large > 0) ? (large / Gcd(large, indicesAxis)) : 1;
        const int64_t inflated = (cap / step) * step;
        if (inflated > repeatPerCore) {
            repeatPerCore = inflated;
        }
    }

    tilingData_.set_repeat_per_core(repeatPerCore);
    tilingData_.set_rounds(ceilDiv(commonInformation_.indices_pre, repeatPerCore));
    tilingData_.set_rounds_tail(commonInformation_.indices_pre % repeatPerCore);

    int64_t lastDimSize = repeatPerCore * tilingData_.get_params_axis() * commonInformation_.params_dsize +
                          repeatPerCore * tilingData_.get_indices_axis() *
                              (static_cast<int64_t>(commonInformation_.params_dsize) +
                               static_cast<int64_t>(commonInformation_.indices_dsize));
    tilingData_.set_indices_loop_num(tilingData_.get_rounds() / coreNumAll_);
    tilingData_.set_need_core_num(tilingData_.get_rounds() > coreNumAll_ ? coreNumAll_ : tilingData_.get_rounds());
    tilingData_.set_indices_row_num_last(tilingData_.get_rounds() % coreNumAll_);

    // Branch judgment for cutting indices into slices
    if (LastAxisCutIntoSlices(ifSameDimValueExceptAxis, lastDimSize, availableUbSize)) {
        return true;
    }

    const int64_t blockSizeX = BLOCK_SIZE / commonInformation_.params_dsize;
    const int64_t blockSizeIdx = BLOCK_SIZE / commonInformation_.indices_dsize;
    const int64_t blockSizeIdx32 = BLOCK_SIZE / static_cast<int64_t>(SIZE_INT32);
    const int64_t paramsAxis = tilingData_.get_params_axis();
    const int64_t indicesAxis = tilingData_.get_indices_axis();
    const int64_t xAligned = ceilDiv(paramsAxis, blockSizeX) * blockSizeX;
    const int64_t idxAligned = ceilDiv(indicesAxis, blockSizeIdx) * blockSizeIdx;
    const int64_t resAligned = ceilDiv(indicesAxis, blockSizeX) * blockSizeX;
    const int64_t idx32Aligned = ceilDiv(indicesAxis, blockSizeIdx32) * blockSizeIdx32;
    const int64_t lastAxisUbSize = repeatPerCore * (xAligned * commonInformation_.params_dsize +
                                                    idxAligned * commonInformation_.indices_dsize +
                                                    resAligned * commonInformation_.params_dsize +
                                                    idx32Aligned * 2 * static_cast<int64_t>(SIZE_INT32) +
                                                    indicesAxis * static_cast<int64_t>(SIZE_INT32));

    if ((supportGather_ == 1) && (lastAxisUbSize < availableUbSize) && ifSameDimValueExceptAxis) {
        tilingData_.set_tilingMode(TILING_MODE_FOR_LAST_AXIS_GATHER);
        tilingData_.set_dbFlag(lastAxisUbSize < availableUbSize / HALF ? 1 : 0);
    }
    OP_TILING_CHECK(
        lastAxisUbSize >= availableUbSize,
        OP_LOGW("GatherElements", "op GatherElementsTiling: while axis is the last dim, the shape is too large."),
        return false);
    return true;
}

bool GatherElementsTiling::ChooseTilingMode(bool ifSameDimValueExceptAxis)
{
    // 1.tiling mode for last axis.
    // 2.tiling mode for not too large shape.
    // 3.when choose tiling failed, aicpu recommended.
    OP_TILING_CHECK(
        commonInformation_.params_total_ceil_size > INT_MAX_NUM,
        OP_LOGW("GatherElements", "op GatherElementsTiling: x is too large, it's not proper to use aicore."),
        return false);

    OP_TILING_CHECK(tilingData_.get_params_axis() > INT_MAX_NUM / HALF,
                    OP_LOGW(OP_NAME.c_str(), "op [GatherElementsTiling] : shape range of x axis is larger than the "
                                             "threshold, it's not proper to use aicore."),
                    return false);

    bool indicesPreNotEqualOne = commonInformation_.indices_pre != 1;
    bool lastAxisFlag = axis_ == (dims_ - 1);

    if (indicesPreNotEqualOne && lastAxisFlag && ChooseTilingModeForLastAxis(ifSameDimValueExceptAxis)) {
        return true;
    }

    // determine to use aicore or aicpu gatherElements
    OP_TILING_CHECK((commonInformation_.params_total_ceil_size > CONVERT_TO_AICPU_UB &&
                     tilingData_.get_indices_num() > INDICES_NUM_THRESHOULD),
                    OP_LOGW("GatherElements", "op GatherElementsTiling: it's not proper to use aicore."), return false);

    ChooseTilingModeSameShapeExceptAxis();
    if (!ifSameDimValueExceptAxis) {
        tilingData_.set_tilingMode(tilingData_.get_tilingMode() + TILING_MODE_DIF);
    }
    return true;
}

bool GatherElementsTiling::V2MemCheck() const
{
    int64_t idxGatherDim = indicesShape_.GetDim(axis_);
    int64_t selfGatherDim = paramsShape_.GetDim(axis_);
    int64_t selfDtypeSize = paramsDsize_;
    int64_t idxDtypeSize = indicesDsize_;
    int64_t indexShapeProduct = 1;
    int64_t selfShapeProduct = 1;
    int64_t idxPreDim = 1;
    int64_t idxPostDim = 1;
    for (int32_t i = 0; i < dims_; i++) {
        indexShapeProduct *= indicesShape_.GetDim(i);
        selfShapeProduct *= paramsShape_.GetDim(i);
        if (i > axis_) {
            idxPostDim *= indicesShape_.GetDim(i);
        } else if (i < axis_) {
            idxPreDim *= indicesShape_.GetDim(i);
        }
    }
    bool isTransCase = true;
    bool memCheck = true;
    if (selfGatherDim * selfDtypeSize < V2_UB_LIMIT) {
        isTransCase = true;
        memCheck = idxGatherDim >= V2_TRANS_LEN && selfGatherDim >= V2_TRANS_LEN;
    } else if (selfGatherDim * selfDtypeSize > V2_ASCEND_910B_UB) {
        isTransCase = false;
    } else {
        memCheck = false;
    }
    if (memCheck && isTransCase) {
        int64_t tailGroupCoreNum = std::max(static_cast<int64_t>(1), V2_ASCEND_910B_CORE_NUM / idxPreDim);
        int64_t workspaceLen = std::min(V2_CACHELINE / selfDtypeSize,
                                        (idxPostDim + tailGroupCoreNum - 1) / tailGroupCoreNum);
        if (idxPreDim * idxPostDim <= V2_ASCEND_910B_CORE_NUM) {
            memCheck = selfShapeProduct > idxPreDim * idxPostDim * (idxGatherDim + selfGatherDim);
        } else {
            memCheck = selfShapeProduct * selfDtypeSize + indexShapeProduct * idxDtypeSize >
                       indexShapeProduct * selfDtypeSize +
                           V2_ASCEND_910B_CORE_NUM * workspaceLen *
                               (idxGatherDim * idxDtypeSize + selfGatherDim * selfDtypeSize);
        }
    }
    return memCheck;
}

bool GatherElementsTiling::IfUseV2() const
{
    auto inputDesc = tilingContext_->GetInputDesc(0);
    if (inputDesc == nullptr) {
        return false;
    }
    if (GATHER_ELEMENTS_V2_DTYPES.count(inputDesc->GetDataType()) == 0) {
        return false;
    }
    if (axis_ == dims_ - 1) {
        for (int32_t i = 0; i < dims_; i++) {
            if (i != axis_ && paramsShape_.GetDim(i) != indicesShape_.GetDim(i)) {
                return false;
            }
        }
        return true;
    }
    int64_t indexShapeProduct = 1;
    int64_t selfShapeProduct = 1;
    bool dimCheck = true;
    int64_t selfPostDim = 1;
    for (int32_t i = 0; i < dims_; i++) {
        int64_t indexDim = indicesShape_.GetDim(i);
        int64_t selfDim = paramsShape_.GetDim(i);
        indexShapeProduct *= indexDim;
        selfShapeProduct *= selfDim;
        if ((i != 0 && i != axis_ && i != axis_ + 1) && indexDim != selfDim) {
            dimCheck = false;
            break;
        }
        if (i > axis_) {
            selfPostDim *= selfDim;
        }
    }
    bool selfShapeCheck = selfShapeProduct < INT_MAX_NUM;
    dimCheck = dimCheck && selfPostDim > V2_POST_DIM_LIMIT;
    bool memCheck = V2MemCheck();
    return dimCheck && selfShapeCheck && memCheck;
}

ge::graphStatus GatherElementsTiling::ComputeV2LastDimTiling()
{
    const int64_t totalCoreNum = coreNumAll_;
    const int64_t ubSize = ubSize_ - V2LD_RESERVED_UB;

    gert::Shape xShape;
    gert::Shape indexShape;
    int i = 0;
    while (i < dims_) {
        int64_t xd = static_cast<int64_t>(paramsShape_.GetDim(i));
        int64_t id = static_cast<int64_t>(indicesShape_.GetDim(i));
        if (xd != id || i == dims_ - 1) {
            xShape.AppendDim(xd);
            indexShape.AppendDim(id);
            i++;
        } else {
            int j = i;
            while (j < dims_ && paramsShape_.GetDim(j) == indicesShape_.GetDim(j) && j != dims_ - 1) {
                j++;
            }
            if (j - i > 1) {
                int64_t val = 1;
                for (int k = i; k < j; k++) {
                    val *= static_cast<int64_t>(paramsShape_.GetDim(k));
                }
                xShape.AppendDim(val);
                indexShape.AppendDim(val);
            } else {
                xShape.AppendDim(xd);
                indexShape.AppendDim(id);
            }
            i = j - 1;
            i++;
        }
    }
    int64_t dimNum = static_cast<int64_t>(xShape.GetDimNum());

    int64_t nonCollectingAxisSize = 1;
    int64_t xAxisSize = 0;
    int64_t indexAxisSize = 0;
    int64_t specialDataMove = 0;
    int64_t xShapeArray[GATHER_ELEMENTS_V2_TILING_ARRAY_LEN_EIGHT] = {0};
    int64_t indexShapeArray[GATHER_ELEMENTS_V2_TILING_ARRAY_LEN_EIGHT] = {0};
    int64_t xStrideArray[GATHER_ELEMENTS_V2_TILING_ARRAY_LEN_EIGHT] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t indexStrideArray[GATHER_ELEMENTS_V2_TILING_ARRAY_LEN_EIGHT] = {1, 1, 1, 1, 1, 1, 1, 1};
    for (int k = 0; k < dimNum; k++) {
        int64_t xd = xShape.GetDim(k);
        int64_t id = indexShape.GetDim(k);
        xShapeArray[k] = xd;
        indexShapeArray[k] = id;
        if (k < dimNum - 1) {
            nonCollectingAxisSize *= xd;
            if (k != 0 && xd != id) {
                specialDataMove = 1;
            }
        } else {
            xAxisSize = xd;
            indexAxisSize = id;
        }
    }
    for (int k = dimNum - 1; k > 0; k--) {
        xStrideArray[k - 1] = xStrideArray[k] * xShapeArray[k];
        indexStrideArray[k - 1] = indexStrideArray[k] * indexShapeArray[k];
    }

    // GetDSize
    int64_t xDSize = static_cast<int64_t>(paramsDsize_);
    int64_t xRealDsize = xDSize;
    int64_t indexDSize = static_cast<int64_t>(indicesDsize_);
    int64_t indexRealDsize = indexDSize;
    int64_t xDsizeRatio = 1;
    if (xDSize == V2LD_INT8_DSIZE) {
        xDSize = V2LD_INT6_DSIZE;
        xDsizeRatio = V2LD_DOUBLE_TIME;
    }
    if (indexDSize == V2LD_INT32_DSIZE) {
        indexDSize = V2LD_INT64_DSIZE;
    }

    // IfEnableBatch
    int64_t xAlignSize = ceilAlign(xAxisSize * xDSize, V2LD_BLOCK_SIZE * xDsizeRatio);
    int64_t yAlignSize = ceilAlign(indexAxisSize * xDSize, V2LD_BLOCK_SIZE * xDsizeRatio);
    int64_t indexAlignSize = ceilAlign(indexAxisSize * indexDSize, V2LD_BLOCK_SIZE * V2LD_DOUBLE_TIME);
    bool batchProcess = (xAlignSize + yAlignSize + indexAlignSize) <= ubSize / V2LD_DOUBLE_TIME;

    // DoUBSlice
    int64_t eachCalculationLines = 0;
    int64_t xSliceNum = 0;
    int64_t indexSliceNum = 0;
    int64_t xBufferSize = 0;
    int64_t indexBufferSize = 0;
    int64_t yBufferSize = 0;
    int64_t maskBufferSize = 0;
    int64_t reservedXSize = 0;
    int64_t reservedIndexSize = 0;
    int64_t indexAxisSizeEqualOne = 0;
    int64_t dataMoveUBStride = 0;
    if (indexShape.GetDim(dimNum - 1) == 1 && batchProcess && specialDataMove == 0) {
        indexAxisSizeEqualOne = V2LD_NUM_ONE;
        int64_t perGroupSize = xAxisSize * xDSize + indexAxisSize * indexDSize + indexAxisSize * indexDSize;
        eachCalculationLines = floorDiv(ubSize, perGroupSize);
        xBufferSize = ceilAlign(xAxisSize * xDSize * eachCalculationLines, V2LD_BLOCK_SIZE * xDsizeRatio);
        indexBufferSize = ceilAlign(indexAxisSize * indexDSize * eachCalculationLines,
                                    V2LD_BLOCK_SIZE * V2LD_DOUBLE_TIME);
        yBufferSize = ceilAlign(indexAxisSize * indexDSize * eachCalculationLines, V2LD_BLOCK_SIZE * xDsizeRatio);
    } else if (batchProcess) {
        int64_t perGroupSize = xAlignSize + indexAlignSize + yAlignSize;
        eachCalculationLines = floorDiv(ubSize, perGroupSize);
        xBufferSize = xAlignSize * eachCalculationLines;
        yBufferSize = yAlignSize * eachCalculationLines;
        indexBufferSize = indexAlignSize * eachCalculationLines;
        if (ceilDiv(indexAxisSize * indexDSize, V2LD_BLOCK_SIZE) % V2LD_DOUBLE_TIME == 1 &&
            indexRealDsize == V2LD_INT64_DSIZE) {
            dataMoveUBStride = V2LD_NUM_ONE;
        }
    } else if (xAlignSize <= ubSize / V2LD_DOUBLE_TIME) {
        xSliceNum = V2LD_NUM_ONE;
        eachCalculationLines = V2LD_NUM_ONE;
        xBufferSize = xAlignSize;
        indexBufferSize = (ubSize - xBufferSize) / (indexDSize + xDSize) * indexDSize;
        indexBufferSize = ceilAlign(indexBufferSize, V2LD_BLOCK_SIZE * V2LD_DOUBLE_TIME);
        yBufferSize = indexBufferSize / indexDSize * xDSize;
        yBufferSize = ceilAlign(yBufferSize, V2LD_BLOCK_SIZE);
        indexSliceNum = ceilDiv(indexAlignSize, indexBufferSize);
        reservedIndexSize = indexAxisSize - (indexSliceNum - 1) * indexBufferSize / indexDSize;
    } else {
        eachCalculationLines = V2LD_NUM_ONE;
        xBufferSize = ubSize / V2LD_DOUBLE_TIME;
        xSliceNum = ceilDiv(xAlignSize, xBufferSize);
        reservedXSize = xAxisSize - (xSliceNum - 1) * xBufferSize / xDSize;
        indexBufferSize = static_cast<int64_t>(static_cast<float>(xBufferSize) /
                                               (indexDSize + xDSize * V2LD_DOUBLE_TIME + V2LD_MASK_SIZE_RATE) *
                                               indexDSize);
        indexBufferSize = ceilAlign(indexBufferSize, V2LD_BLOCK_SIZE * V2LD_INT64_DSIZE);
        indexSliceNum = ceilDiv(indexAlignSize, indexBufferSize);
        yBufferSize = indexBufferSize / indexDSize * xDSize;
        yBufferSize = ceilAlign(yBufferSize, V2LD_BLOCK_SIZE);
        reservedIndexSize = indexAxisSize - (indexSliceNum - 1) * indexBufferSize / indexDSize;
        maskBufferSize = indexBufferSize / indexDSize / V2LD_INT64_DSIZE;
        maskBufferSize = ceilAlign(maskBufferSize, V2LD_BLOCK_SIZE);
    }

    // DoScalarMode
    int64_t scalarMode = 0;
    int64_t scalarModeLength = 0;
    if (xSliceNum > V2LD_MAX_SLICE_NUM || xAxisSize / indexAxisSize > V2LD_MAX_SIZE_RATIO) {
        scalarMode = V2LD_NUM_ONE;
        int64_t idxBlockSize = ceilDiv(indexAlignSize, V2LD_BLOCK_SIZE * V2LD_DOUBLE_TIME) * V2LD_DOUBLE_TIME;
        int64_t yBlockSize = ceilDiv(indexAxisSize * xRealDsize, V2LD_BLOCK_SIZE);
        eachCalculationLines = ubSize / V2LD_BLOCK_SIZE / (idxBlockSize + yBlockSize);
        bool multRowProcess = eachCalculationLines > 1 && (nonCollectingAxisSize > totalCoreNum ||
                                                           (nonCollectingAxisSize <= totalCoreNum &&
                                                            indexAxisSize < totalCoreNum * V2LD_BLOCK_SIZE));
        if (multRowProcess) {
            indexBufferSize = eachCalculationLines * idxBlockSize * V2LD_BLOCK_SIZE;
            yBufferSize = eachCalculationLines * yBlockSize * V2LD_BLOCK_SIZE;
        } else {
            indexSliceNum = ceilDiv(indexAlignSize, indexBufferSize);
            scalarModeLength = indexSliceNum * nonCollectingAxisSize;
            if (scalarModeLength < totalCoreNum) {
                indexSliceNum = totalCoreNum / nonCollectingAxisSize;
                scalarModeLength = indexSliceNum * nonCollectingAxisSize;
            }
            indexBufferSize = ceilAlign(indexAlignSize / indexSliceNum + V2LD_INT64_DSIZE,
                                        V2LD_BLOCK_SIZE * V2LD_DOUBLE_TIME);
            yBufferSize = indexBufferSize / indexDSize * xRealDsize;
            yBufferSize = ceilAlign(yBufferSize, V2LD_BLOCK_SIZE);
            reservedIndexSize = indexAxisSize - (indexSliceNum - 1) * indexBufferSize / indexDSize;
        }
    }

    // DoNeedUseCore
    int64_t needUsedCore = 0;
    int64_t formerCoreRowNum = 0;
    int64_t formerCoreNum = 0;
    if (scalarModeLength == 0) {
        needUsedCore = nonCollectingAxisSize > totalCoreNum ? totalCoreNum : nonCollectingAxisSize;
        formerCoreRowNum = nonCollectingAxisSize / needUsedCore;
        formerCoreNum = nonCollectingAxisSize % needUsedCore;
    } else {
        needUsedCore = scalarModeLength > totalCoreNum ? totalCoreNum : scalarModeLength;
    }
    if (needUsedCore <= 0) {
        needUsedCore = 1;
    }

    auto& ld = tilingData_.v2Data.lastDimTiling;
    ld.set_xShape(xShapeArray);
    ld.set_indexShape(indexShapeArray);
    ld.set_xStrideArray(xStrideArray);
    ld.set_indexStrideArray(indexStrideArray);
    ld.set_dimNum(dimNum);
    ld.set_specialDataMove(specialDataMove);
    ld.set_xSliceNum(xSliceNum);
    ld.set_indexSliceNum(indexSliceNum);
    ld.set_reservedXSize(reservedXSize);
    ld.set_reservedIndexSize(reservedIndexSize);
    ld.set_indexAxisSizeEqualOne(indexAxisSizeEqualOne);
    ld.set_scalarMode(scalarMode);
    ld.set_formerCoreRowNum(formerCoreRowNum);
    ld.set_formerCoreNum(formerCoreNum);
    ld.set_eachCalculationLines(eachCalculationLines);
    ld.set_xBufferSize(xBufferSize);
    ld.set_indexBufferSize(indexBufferSize);
    ld.set_yBufferSize(yBufferSize);
    ld.set_maskBufferSize(maskBufferSize);
    ld.set_scalarModeLength(scalarModeLength);
    ld.set_dataMoveUBStride(dataMoveUBStride);

    tilingData_.set_v2Mode(V2_MODE_LASTDIM);
    tilingData_.set_useV2(1);
    useV2_ = true;
    v2UsedCoreNum_ = needUsedCore;
    v2Workspace_ = 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherElementsTiling::TryRouteToV2()
{
    useV2_ = false;
    if (!IfUseV2()) {
        return ge::GRAPH_SUCCESS;
    }
    if (axis_ == dims_ - 1) {
        return ComputeV2LastDimTiling();
    }
    const uint64_t xDtypeSize = static_cast<uint64_t>(paramsDsize_);
    const uint64_t idxDtypeSize = static_cast<uint64_t>(indicesDsize_);
    const uint64_t coreNum = static_cast<uint64_t>(coreNumAll_);
    const uint64_t ubSize = static_cast<uint64_t>(ubSize_);

    uint64_t xPreDim = 1;
    uint64_t xGatherDim = 1;
    uint64_t xPostDim = 1;
    uint64_t idxPreDim = 1;
    uint64_t idxGatherDim = 1;
    uint64_t idxPostDim = 1;
    const uint64_t dim = static_cast<uint64_t>(axis_);
    const uint64_t dimNum = static_cast<uint64_t>(dims_);
    idxGatherDim = static_cast<uint64_t>(indicesShape_.GetDim(axis_));
    xGatherDim = static_cast<uint64_t>(paramsShape_.GetDim(axis_));
    for (uint64_t i = 0; i < dim; i++) {
        idxPreDim *= static_cast<uint64_t>(indicesShape_.GetDim(i));
        xPreDim *= static_cast<uint64_t>(paramsShape_.GetDim(i));
    }
    for (uint64_t i = dim + 1; i < dimNum; i++) {
        idxPostDim *= static_cast<uint64_t>(indicesShape_.GetDim(i));
        xPostDim *= static_cast<uint64_t>(paramsShape_.GetDim(i));
    }
    (void)xPreDim;
    (void)xPostDim;

    // Tiling4GatherElementsV2 (row/column core grouping)
    uint64_t usedCoreNum = std::min(idxPreDim * idxPostDim, coreNum);
    uint64_t coreGroupNum = 0;
    uint64_t formerGroupNum = 0;
    uint64_t tailGroupNum = 0;
    uint64_t formerGroupPreDim = 0;
    uint64_t tailGroupPreDim = 0;
    uint64_t formerGroupCoreNum = 0;
    uint64_t tailGroupCoreNum = 0;
    uint64_t formerGroupFormerNum = 0;
    uint64_t formerGroupTailNum = 0;
    uint64_t formerGroupFormerPostDim = 0;
    uint64_t formerGroupTailPostDim = 0;
    uint64_t tailGroupFormerNum = 0;
    uint64_t tailGroupTailNum = 0;
    uint64_t tailGroupFormerPostDim = 0;
    uint64_t tailGroupTailPostDim = 0;
    if (idxPreDim > usedCoreNum) {
        coreGroupNum = usedCoreNum;
        tailGroupNum = (coreGroupNum - idxPreDim % coreGroupNum) % coreGroupNum;
        formerGroupNum = coreGroupNum - tailGroupNum;
        if (usedCoreNum == 0UL) {
            usedCoreNum = 1UL;
        }
        formerGroupPreDim = (idxPreDim + usedCoreNum - 1) / usedCoreNum;
        tailGroupPreDim = idxPreDim / usedCoreNum;
        formerGroupCoreNum = 1;
        tailGroupCoreNum = 1;
        formerGroupTailNum = 0;
        formerGroupFormerNum = 1;
        tailGroupTailNum = 0;
        tailGroupFormerNum = 1;
        formerGroupFormerPostDim = idxPostDim;
        formerGroupTailPostDim = idxPostDim;
        tailGroupFormerPostDim = idxPostDim;
        tailGroupTailPostDim = idxPostDim;
    } else {
        coreGroupNum = idxPreDim;
        tailGroupNum = (coreGroupNum - usedCoreNum % coreGroupNum) % coreGroupNum;
        formerGroupNum = coreGroupNum - tailGroupNum;
        formerGroupPreDim = 1;
        tailGroupPreDim = 1;
        formerGroupCoreNum = (usedCoreNum + coreGroupNum - 1) / coreGroupNum;
        tailGroupCoreNum = usedCoreNum / coreGroupNum;
        formerGroupTailNum = (formerGroupCoreNum - idxPostDim % formerGroupCoreNum) % formerGroupCoreNum;
        formerGroupFormerNum = formerGroupCoreNum - formerGroupTailNum;
        formerGroupFormerPostDim = (idxPostDim + formerGroupCoreNum - 1) / formerGroupCoreNum;
        formerGroupTailPostDim = idxPostDim / formerGroupCoreNum;
        tailGroupTailNum = (tailGroupCoreNum - idxPostDim % tailGroupCoreNum) % tailGroupCoreNum;
        tailGroupFormerNum = tailGroupCoreNum - tailGroupTailNum;
        tailGroupFormerPostDim = (idxPostDim + tailGroupCoreNum - 1) / tailGroupCoreNum;
        tailGroupTailPostDim = idxPostDim / tailGroupCoreNum;
    }

    auto& params = tilingData_.v2Data.params;
    params.set_xPreDim(xPreDim);
    params.set_xGatherDim(xGatherDim);
    params.set_xPostDim(xPostDim);
    params.set_idxPreDim(idxPreDim);
    params.set_idxGatherDim(idxGatherDim);
    params.set_idxPostDim(idxPostDim);
    params.set_coreGroupNum(coreGroupNum);
    params.set_formerGroupNum(formerGroupNum);
    params.set_tailGroupNum(tailGroupNum);
    params.set_formerGroupPreDim(formerGroupPreDim);
    params.set_tailGroupPreDim(tailGroupPreDim);
    params.set_formerGroupCoreNum(formerGroupCoreNum);
    params.set_tailGroupCoreNum(tailGroupCoreNum);
    params.set_formerGroupFormerNum(formerGroupFormerNum);
    params.set_formerGroupTailNum(formerGroupTailNum);
    params.set_formerGroupFormerPostDim(formerGroupFormerPostDim);
    params.set_formerGroupTailPostDim(formerGroupTailPostDim);
    params.set_tailGroupFormerNum(tailGroupFormerNum);
    params.set_tailGroupTailNum(tailGroupTailNum);
    params.set_tailGroupFormerPostDim(tailGroupFormerPostDim);
    params.set_tailGroupTailPostDim(tailGroupTailPostDim);

    // CalcMaxBufferSize to decide transpose vs scalar
    uint64_t carryNumAlign = V2_CACHELINE / xDtypeSize;
    uint64_t xAlign = V2_BLOCK_SIZE / xDtypeSize;
    uint64_t idxAlign = V2_BLOCK_SIZE / idxDtypeSize;
    uint64_t availableUb = ubSize - V2_RESERVED_UB_SIZE;
    uint64_t minIdxGatherDimSlice = V2_CACHELINE;
    uint64_t gatherInBufferSize = ceilAlign(xGatherDim, xAlign) * xDtypeSize +
                                  ceilAlign(minIdxGatherDimSlice, idxAlign) * idxDtypeSize * V2_NUM_TWO;
    uint64_t gatherOutBufferSize = ceilAlign(minIdxGatherDimSlice, xAlign) * xDtypeSize;
    uint64_t transInBufferSize = V2_TRANSPOSE_WS_LEN * V2_CACHELINE;
    uint64_t transOutBufferSize = V2_TRANSPOSE_WS_LEN * V2_CACHELINE;
    uint64_t inBufferSize = std::max(gatherInBufferSize, transInBufferSize);
    uint64_t outBufferSize = std::max(gatherOutBufferSize, transOutBufferSize);
    bool canTrans = (availableUb >= (inBufferSize + outBufferSize));
    uint64_t idxGatherDimSlice = 0;
    if (canTrans) {
        uint64_t ubLeft = availableUb - (std::max(transInBufferSize, ceilAlign(xGatherDim, xAlign) * xDtypeSize) +
                                         transOutBufferSize);
        uint64_t maxIdxGatherDimSlice = ubLeft / (V2_BLOCK_SIZE * V2_NUM_TWO) * idxAlign;
        uint64_t idxGatherDimAlign = ceilAlign(idxGatherDim, idxAlign);
        idxGatherDimSlice = std::min(maxIdxGatherDimSlice, idxGatherDimAlign);
        gatherInBufferSize = ceilAlign(xGatherDim, xAlign) * xDtypeSize +
                             ceilAlign(idxGatherDimSlice, idxAlign) * idxDtypeSize;
        if (idxDtypeSize > static_cast<uint64_t>(sizeof(int32_t))) {
            uint64_t idx32Align = V2_BLOCK_SIZE / sizeof(int32_t);
            gatherInBufferSize = ceilAlign(xGatherDim, xAlign) * xDtypeSize +
                                 (ceilAlign(idxGatherDimSlice, idx32Align) + idxGatherDimSlice) * sizeof(int32_t);
        }
        gatherOutBufferSize = ceilAlign(idxGatherDimSlice, xAlign) * xDtypeSize;
        inBufferSize = std::max(gatherInBufferSize, transInBufferSize);
        outBufferSize = std::max(gatherOutBufferSize, transOutBufferSize);
    }

    if (canTrans) {
        v2UsedCoreNum_ = static_cast<int64_t>(usedCoreNum);
        uint64_t xGatherDimAlign = ceilAlign(xGatherDim * xDtypeSize, idxDtypeSize) / xDtypeSize;
        uint64_t usedWorkspaceLen = std::min(carryNumAlign, std::max(formerGroupFormerPostDim, tailGroupFormerPostDim));
        uint64_t workspacePerBlock = usedWorkspaceLen * (xGatherDimAlign * xDtypeSize + idxGatherDim * idxDtypeSize);
        v2Workspace_ = static_cast<int64_t>(usedCoreNum * workspacePerBlock);

        auto& trans = tilingData_.v2Data.transTiling;
        trans.set_carryNumAlign(carryNumAlign);
        trans.set_xCarryNumAlign(carryNumAlign);
        trans.set_idxCarryNumAlign(V2_CACHELINE / idxDtypeSize);
        trans.set_inBufferSize(inBufferSize);
        trans.set_outBufferSize(outBufferSize);
        trans.set_transGatherDimSlice(V2_TRANSPOSE_WS_LEN);
        trans.set_idxGatherDimSlice(idxGatherDimSlice);
        trans.set_workspacePerBlock(workspacePerBlock);
        tilingData_.set_v2Mode(V2_MODE_TRANSPOSE);
    } else {
        // scalar mode (Tiling4Scalar)
        uint64_t idxDataPerPre = idxGatherDim * idxPostDim;
        uint64_t idxAllData = idxPreDim * idxDataPerPre;
        usedCoreNum = std::min(idxAllData, coreNum);
        uint64_t formerGroupFormerData = 0;
        uint64_t formerGroupTailData = 0;
        uint64_t tailGroupFormerData = 0;
        uint64_t tailGroupTailData = 0;
        if (idxPreDim >= usedCoreNum) {
            formerGroupFormerData = idxDataPerPre;
            formerGroupTailData = idxDataPerPre;
            tailGroupFormerData = idxDataPerPre;
            tailGroupTailData = idxDataPerPre;
        } else {
            coreGroupNum = idxPreDim;
            tailGroupNum = (coreGroupNum - usedCoreNum % coreGroupNum) % coreGroupNum;
            formerGroupNum = coreGroupNum - tailGroupNum;
            formerGroupCoreNum = (usedCoreNum + coreGroupNum - 1) / coreGroupNum;
            tailGroupCoreNum = usedCoreNum / coreGroupNum;
            formerGroupTailNum = (formerGroupCoreNum - idxDataPerPre % formerGroupCoreNum) % formerGroupCoreNum;
            formerGroupFormerNum = formerGroupCoreNum - formerGroupTailNum;
            formerGroupFormerData = (idxDataPerPre + formerGroupCoreNum - 1) / formerGroupCoreNum;
            formerGroupTailData = idxDataPerPre / formerGroupCoreNum;
            tailGroupTailNum = (tailGroupCoreNum - idxDataPerPre % tailGroupCoreNum) % tailGroupCoreNum;
            tailGroupFormerNum = tailGroupCoreNum - tailGroupTailNum;
            tailGroupFormerData = (idxDataPerPre + tailGroupCoreNum - 1) / tailGroupCoreNum;
            tailGroupTailData = idxDataPerPre / tailGroupCoreNum;
        }
        uint64_t maxIdxDataAlign = (ubSize - V2_RESERVED_UB_SIZE) / V2_BLOCK_SIZE * V2_BLOCK_SIZE /
                                   (idxDtypeSize + xDtypeSize);

        v2UsedCoreNum_ = static_cast<int64_t>(usedCoreNum);
        auto& scalar = tilingData_.v2Data.scalarTiling;
        scalar.set_formerGroupFormerData(formerGroupFormerData);
        scalar.set_formerGroupTailData(formerGroupTailData);
        scalar.set_tailGroupFormerData(tailGroupFormerData);
        scalar.set_tailGroupTailData(tailGroupTailData);
        scalar.set_maxIdxDataAlign(maxIdxDataAlign);
        tilingData_.set_v2Mode(V2_MODE_SCALAR);
    }

    tilingData_.set_useV2(1);
    useV2_ = true;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherElementsTiling::Init()
{
    OP_LOGD(tilingContext_->GetNodeName(), "GatherElementsTiling initing");
    compileInfo_ = static_cast<const GatherElementsCompileInfo*>(tilingContext_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, compileInfo_);
    OP_TILING_CHECK(GetPlatformInfo() != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(tilingContext_->GetNodeName(),
                                                    "op GatherElementsTiling: GetPlatformInfo failed."),
                    return ge::GRAPH_FAILED);
    auto inputDesc = tilingContext_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, inputDesc);
    auto indicesDesc = tilingContext_->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, indicesDesc);
    auto attrs = tilingContext_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, attrs);

    uint32_t paramsTypeLen = 0;
    uint32_t indicesTypeLen = 0;
    ge::TypeUtils::GetDataTypeLength(inputDesc->GetDataType(), paramsTypeLen);
    ge::TypeUtils::GetDataTypeLength(indicesDesc->GetDataType(), indicesTypeLen);
    paramsDsize_ = static_cast<int32_t>(paramsTypeLen);
    indicesDsize_ = static_cast<int32_t>(indicesTypeLen);
    OP_CHECK_IF((paramsDsize_ <= 0 || indicesDsize_ <= 0),
                OP_LOGE(tilingContext_->GetNodeName(), "Failed to get data type length"), return ge::GRAPH_FAILED);
    supportGather_ = GATHER_DTYPES.count(inputDesc->GetDataType()) > 0 ? 1 : 0;

    paramsShape_ = tilingContext_->GetInputShape(0)->GetStorageShape();
    indicesShape_ = tilingContext_->GetInputShape(1)->GetStorageShape();
    yShape_ = tilingContext_->GetOutputShape(0)->GetStorageShape();
    dims_ = static_cast<int32_t>(paramsShape_.GetDimNum());

    const int64_t* axisPtr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_AXIS);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext_, axisPtr);
    int32_t axis = static_cast<int32_t>(*axisPtr);

    // check inputs shape
    int32_t indicesDims = static_cast<int32_t>(indicesShape_.GetDimNum());
    OP_TILING_CHECK(dims_ <= 0 || indicesDims <= 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(tilingContext_->GetNodeName(),
                                                    "GatherElementsTiling: params_dims or indices_dims is 0."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        axis < -dims_ || axis >= dims_,
        VECTOR_INNER_ERR_REPORT_TILIING(tilingContext_->GetNodeName(), "op GatherElementsTiling: axis is invalid."),
        return ge::GRAPH_FAILED);
    axis_ = axis < 0 ? axis + dims_ : axis;
    OP_TILING_CHECK(!CheckTensorShape(),
                    VECTOR_INNER_ERR_REPORT_TILIING(tilingContext_->GetNodeName(),
                                                    "op GatherElementsTiling: [checkTensorShape] failed."),
                    return ge::GRAPH_FAILED);

    tilingData_.set_axis(axis_);
    tilingData_.set_dims(dims_);
    RecordTilingCommonInformation(axis_, dims_);
    RecordShapeInformation();

    // 优先尝试路由到 gather_elements_v2 能力（910B/910_93 兼容场景）
    ge::graphStatus v2Ret = TryRouteToV2();
    if (v2Ret != ge::GRAPH_SUCCESS) {
        return v2Ret;
    }
    if (useV2_) {
        OP_LOGD(tilingContext_->GetNodeName(), "GatherElementsTiling routed to gather_elements_v2");
        return ge::GRAPH_SUCCESS;
    }

    OP_TILING_CHECK(!ChooseTilingMode(IfParamsIndicesSameShapeExceptAxis()),
                    OP_LOGW(tilingContext_->GetNodeName(), "choose tiling mode failed, aicpu recommended."),
                    return ge::GRAPH_FAILED);
    OP_LOGD(tilingContext_->GetNodeName(), "GatherElementsTiling inited");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherElementsTiling::SetKernelTiling()
{
    size_t* currentWorkSpace = tilingContext_->GetWorkspaceSizes(1);
    if (useV2_) {
        // v2 路由：block dim 使用 v2 计算的核心数，workspace 合并 v2 的 workspace
        tilingContext_->SetBlockDim(v2UsedCoreNum_);
        currentWorkSpace[0] = compileInfo_->sysWorkspaceSize + static_cast<size_t>(v2Workspace_);
    } else {
        tilingContext_->SetBlockDim(tilingData_.get_need_core_num());
        currentWorkSpace[0] = compileInfo_->sysWorkspaceSize;
    }
    tilingData_.SaveToBuffer(tilingContext_->GetRawTilingData()->GetData(),
                             tilingContext_->GetRawTilingData()->GetCapacity());
    tilingContext_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    TilingDataPrint();
    return ge::GRAPH_SUCCESS;
}

void GatherElementsTiling::TilingDataPrint()
{
    if (tilingData_.get_useV2() != 1) {
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] tilingMode=%ld.", tilingData_.get_tilingMode());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] axis=%ld.", tilingData_.get_axis());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] params_pre=%ld.", tilingData_.get_params_pre());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] params_axis=%ld.", tilingData_.get_params_axis());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] params_row=%ld.", tilingData_.get_params_row());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] params_total=%ld.", tilingData_.get_params_total());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] need_core_num=%ld.", tilingData_.get_need_core_num());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] indices_num=%ld.", tilingData_.get_indices_num());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] indices_axis=%ld.", tilingData_.get_indices_axis());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] indices_num_each_core=%ld.",
                tilingData_.get_indices_num_each_core());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] indices_num_remaining=%ld.",
                tilingData_.get_indices_num_remaining());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] indices_loop_num=%ld.",
                tilingData_.get_indices_loop_num());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] indices_row_num_once=%ld.",
                tilingData_.get_indices_row_num_once());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] indices_row_num_last=%ld.",
                tilingData_.get_indices_row_num_last());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] remaining_block_remain=%ld.",
                tilingData_.get_remaining_block_remain());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] remaining_block_num=%ld.",
                tilingData_.get_remaining_block_num());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] slice_thickness_once=%ld.",
                tilingData_.get_slice_thickness_once());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] slice_thickness_last=%ld.",
                tilingData_.get_slice_thickness_last());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] slice_num=%ld.", tilingData_.get_slice_num());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] indices_slice_thickness_dim1=%ld.",
                tilingData_.get_indices_slice_thickness_dim1());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] indices_slice_thickness_dim1_last=%ld.",
                tilingData_.get_indices_slice_thickness_dim1_last());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] indices_slice_num_dim1=%ld.",
                tilingData_.get_indices_slice_num_dim1());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] params_shape_0=%ld.", tilingData_.get_params_shape_0());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] params_shape_1=%ld.", tilingData_.get_params_shape_1());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] params_shape_2=%ld.", tilingData_.get_params_shape_2());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] params_shape_3=%ld.", tilingData_.get_params_shape_3());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] params_shape_4=%ld.", tilingData_.get_params_shape_4());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] params_shape_5=%ld.", tilingData_.get_params_shape_5());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] params_shape_6=%ld.", tilingData_.get_params_shape_6());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] params_shape_7=%ld.", tilingData_.get_params_shape_7());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] indices_shape_0=%ld.", tilingData_.get_indices_shape_0());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] indices_shape_1=%ld.", tilingData_.get_indices_shape_1());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] indices_shape_2=%ld.", tilingData_.get_indices_shape_2());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] indices_shape_3=%ld.", tilingData_.get_indices_shape_3());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] indices_shape_4=%ld.", tilingData_.get_indices_shape_4());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] indices_shape_5=%ld.", tilingData_.get_indices_shape_5());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] indices_shape_6=%ld.", tilingData_.get_indices_shape_6());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] indices_shape_7=%ld.", tilingData_.get_indices_shape_7());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] dims=%ld.", tilingData_.get_dims());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] repeat_per_core=%ld.", tilingData_.get_repeat_per_core());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] rounds=%ld.", tilingData_.get_rounds());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] rounds_tail=%ld.", tilingData_.get_rounds_tail());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData] dbFlag=%ld.", tilingData_.get_dbFlag());
    }
    if (tilingData_.get_useV2() == 1) {
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2] useV2=%ld, v2Mode=%ld.", tilingData_.get_useV2(),
                tilingData_.get_v2Mode());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2] usedCoreNum=%lu.", v2UsedCoreNum_);
        auto& v2Params = tilingData_.v2Data.params;
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2] xPreDim=%lu.", v2Params.get_xPreDim());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2] xGatherDim=%lu.", v2Params.get_xGatherDim());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2] xPostDim=%lu.", v2Params.get_xPostDim());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2] idxPreDim=%lu.", v2Params.get_idxPreDim());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2] idxGatherDim=%lu.", v2Params.get_idxGatherDim());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2] idxPostDim=%lu.", v2Params.get_idxPostDim());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2] coreGroupNum=%lu.", v2Params.get_coreGroupNum());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2] formerGroupNum=%lu.", v2Params.get_formerGroupNum());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2] tailGroupNum=%lu.", v2Params.get_tailGroupNum());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2] formerGroupPreDim=%lu.",
                v2Params.get_formerGroupPreDim());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2] tailGroupPreDim=%lu.", v2Params.get_tailGroupPreDim());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2] formerGroupCoreNum=%lu.",
                v2Params.get_formerGroupCoreNum());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2] tailGroupCoreNum=%lu.",
                v2Params.get_tailGroupCoreNum());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2] formerGroupFormerNum=%lu.",
                v2Params.get_formerGroupFormerNum());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2] formerGroupTailNum=%lu.",
                v2Params.get_formerGroupTailNum());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2] formerGroupFormerPostDim=%lu.",
                v2Params.get_formerGroupFormerPostDim());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2] formerGroupTailPostDim=%lu.",
                v2Params.get_formerGroupTailPostDim());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2] tailGroupFormerNum=%lu.",
                v2Params.get_tailGroupFormerNum());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2] tailGroupTailNum=%lu.",
                v2Params.get_tailGroupTailNum());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2] tailGroupFormerPostDim=%lu.",
                v2Params.get_tailGroupFormerPostDim());
        OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2] tailGroupTailPostDim=%lu.",
                v2Params.get_tailGroupTailPostDim());
        if (tilingData_.get_v2Mode() == 0) {
            auto& v2Scalar = tilingData_.v2Data.scalarTiling;
            OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2-scalar] formerGroupFormerData=%lu.",
                    v2Scalar.get_formerGroupFormerData());
            OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2-scalar] formerGroupTailData=%lu.",
                    v2Scalar.get_formerGroupTailData());
            OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2-scalar] tailGroupFormerData=%lu.",
                    v2Scalar.get_tailGroupFormerData());
            OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2-scalar] tailGroupTailData=%lu.",
                    v2Scalar.get_tailGroupTailData());
            OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2-scalar] maxIdxDataAlign=%lu.",
                    v2Scalar.get_maxIdxDataAlign());
        } else if (tilingData_.get_v2Mode() == 1) {
            auto& v2Trans = tilingData_.v2Data.transTiling;
            OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2-trans] carryNumAlign=%lu.",
                    v2Trans.get_carryNumAlign());
            OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2-trans] xCarryNumAlign=%lu.",
                    v2Trans.get_xCarryNumAlign());
            OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2-trans] idxCarryNumAlign=%lu.",
                    v2Trans.get_idxCarryNumAlign());
            OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2-trans] inBufferSize=%lu.",
                    v2Trans.get_inBufferSize());
            OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2-trans] outBufferSize=%lu.",
                    v2Trans.get_outBufferSize());
            OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2-trans] transGatherDimSlice=%lu.",
                    v2Trans.get_transGatherDimSlice());
            OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2-trans] idxGatherDimSlice=%lu.",
                    v2Trans.get_idxGatherDimSlice());
            OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2-trans] workspacePerBlock=%lu.",
                    v2Trans.get_workspacePerBlock());
        } else if (tilingData_.get_v2Mode() == 2) {
            auto& v2Last = tilingData_.v2Data.lastDimTiling;
            OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2-lastdim] dimNum=%ld.", v2Last.get_dimNum());
            OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2-lastdim] xShape=%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld.",
                    v2Last.get_xShape()[0], v2Last.get_xShape()[1], v2Last.get_xShape()[2], v2Last.get_xShape()[3],
                    v2Last.get_xShape()[4], v2Last.get_xShape()[5], v2Last.get_xShape()[6], v2Last.get_xShape()[7]);
            OP_LOGD(tilingContext_->GetNodeName(),
                    "[tilingData-v2-lastdim] indexShape=%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld.", v2Last.get_indexShape()[0],
                    v2Last.get_indexShape()[1], v2Last.get_indexShape()[2], v2Last.get_indexShape()[3],
                    v2Last.get_indexShape()[4], v2Last.get_indexShape()[5], v2Last.get_indexShape()[6],
                    v2Last.get_indexShape()[7]);
            OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2-lastdim] xStride=%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld.",
                    v2Last.get_xStrideArray()[0], v2Last.get_xStrideArray()[1], v2Last.get_xStrideArray()[2],
                    v2Last.get_xStrideArray()[3], v2Last.get_xStrideArray()[4], v2Last.get_xStrideArray()[5],
                    v2Last.get_xStrideArray()[6], v2Last.get_xStrideArray()[7]);
            OP_LOGD(
                tilingContext_->GetNodeName(), "[tilingData-v2-lastdim] indexStride=%ld,%ld,%ld,%ld,%ld,%ld,%ld,%ld.",
                v2Last.get_indexStrideArray()[0], v2Last.get_indexStrideArray()[1], v2Last.get_indexStrideArray()[2],
                v2Last.get_indexStrideArray()[3], v2Last.get_indexStrideArray()[4], v2Last.get_indexStrideArray()[5],
                v2Last.get_indexStrideArray()[6], v2Last.get_indexStrideArray()[7]);
            OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2-lastdim] specialDataMove=%ld.",
                    v2Last.get_specialDataMove());
            OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2-lastdim] xSliceNum=%ld.", v2Last.get_xSliceNum());
            OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2-lastdim] indexSliceNum=%ld.",
                    v2Last.get_indexSliceNum());
            OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2-lastdim] reservedXSize=%ld.",
                    v2Last.get_reservedXSize());
            OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2-lastdim] reservedIndexSize=%ld.",
                    v2Last.get_reservedIndexSize());
            OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2-lastdim] indexAxisSizeEqualOne=%ld.",
                    v2Last.get_indexAxisSizeEqualOne());
            OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2-lastdim] scalarMode=%ld.", v2Last.get_scalarMode());
            OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2-lastdim] formerCoreRowNum=%ld.",
                    v2Last.get_formerCoreRowNum());
            OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2-lastdim] formerCoreNum=%ld.",
                    v2Last.get_formerCoreNum());
            OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2-lastdim] eachCalculationLines=%ld.",
                    v2Last.get_eachCalculationLines());
            OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2-lastdim] xBufferSize=%ld.",
                    v2Last.get_xBufferSize());
            OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2-lastdim] indexBufferSize=%ld.",
                    v2Last.get_indexBufferSize());
            OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2-lastdim] yBufferSize=%ld.",
                    v2Last.get_yBufferSize());
            OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2-lastdim] maskBufferSize=%ld.",
                    v2Last.get_maskBufferSize());
            OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2-lastdim] scalarModeLength=%ld.",
                    v2Last.get_scalarModeLength());
            OP_LOGD(tilingContext_->GetNodeName(), "[tilingData-v2-lastdim] dataMoveUBStride=%ld.",
                    v2Last.get_dataMoveUBStride());
        }
    }
}

ge::graphStatus Tiling4GatherElements(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4GatherElements running begin");
    if (context == nullptr) {
        OP_LOGE("GatherElements", "The context is nullptr.");
        return ge::GRAPH_FAILED;
    }
    GatherElementsTiling tilingObject(context);
    OP_CHECK_IF(tilingObject.Init() != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "gather elements tiling init fail"), return ge::GRAPH_FAILED);
    return tilingObject.SetKernelTiling();
}

ge::graphStatus TilingPrepare4GatherElements(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling Prepare For GatherElements start.");
    auto compileInfo = context->GetCompiledInfo<GatherElementsCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->core_num = ascendcPlatform.GetCoreNumAiv();
    uint64_t ub_size;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
    compileInfo->ub_size = static_cast<int64_t>(ub_size);
    OP_CHECK_IF((compileInfo->ub_size <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size"),
                return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "ub_size_platform is %lu", compileInfo->ub_size);
    compileInfo->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_LOGD(context->GetNodeName(), "sysWorkspaceSize is %lu", compileInfo->sysWorkspaceSize);
    OP_LOGD(context->GetNodeName(), "Tiling Prepare For GatherElements end.");
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the GatherElements op.
IMPL_OP_OPTILING(GatherElements)
    .Tiling(Tiling4GatherElements)
    .TilingParse<GatherElementsCompileInfo>(TilingPrepare4GatherElements);
} // namespace optiling
