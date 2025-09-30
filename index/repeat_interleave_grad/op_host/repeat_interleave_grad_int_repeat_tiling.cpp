/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file repeat_interleave_grad_int_repeat_tiling.cc
 * \brief
 */

#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"

#include "../op_kernel/v35/repeat_interleave_grad_dag.h"
#include "../op_kernel/v35/repeat_interleave_grad_tiling_key.h"
#include "../op_kernel/v35/repeat_interleave_grad_david_tiling_data.h"
#include "repeat_interleave_grad_int_repeat_tiling.h"
#include "repeat_interleave_grad.h"

namespace optiling {

namespace RIGintRepeat {

static constexpr size_t INPUT_X_INDEX = 0;
static constexpr size_t INPUT_REPEAT_INDEX = 1;
static constexpr size_t OUTPUT_Y_INDEX = 0;
static constexpr size_t ATTR_AXIS_INDEX = 0;
static constexpr int64_t MAX_YGRAD_DIM = 8;

static constexpr size_t MERGED_AXES_NUM = 3;

using namespace Ops::Base;

inline const gert::Shape& EnsureNotScalar(const gert::Shape& inShape, const gert::Shape& oneShape)
{
    if (inShape.IsScalar()) {
        return oneShape;
    }
    return inShape;
}
static ge::graphStatus convertCompileInfo(
    const RepeatInterleaveGradCompileInfo* compileInfo, ReduceOpCompileInfo* reduceCompileInfo,
    gert::TilingContext* context)
{
    reduceCompileInfo->cacheLineSize = static_cast<uint64_t>(compileInfo->clSize);
    reduceCompileInfo->ubBlockSize = static_cast<uint64_t>(compileInfo->blockSize);
    reduceCompileInfo->vRegSize = static_cast<uint64_t>(compileInfo->vRegSize);
    reduceCompileInfo->vectorCoreNum = static_cast<uint64_t>(compileInfo->coreNum);

    uint64_t ubSize = static_cast<uint64_t>(compileInfo->ubSize);
    OP_CHECK_IF(
        ubSize <= static_cast<uint64_t>(CACHE_BUF_SIZE),
        OP_LOGE(
            context, "ReduceOp GetHardwareInfo Failed, ubSize:%lu, at least:%ld.", ubSize,
            CACHE_BUF_SIZE),
        return ge::GRAPH_FAILED);
    reduceCompileInfo->ubSize = ubSize;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DoTiling(
    gert::TilingContext* context, const ReduceOpCompileInfo* compileInfo, ReduceOpInputParam& opInput,
    ReduceTilingKey& key)
{
    ge::graphStatus status = ge::GRAPH_FAILED;
    ReduceOpTilingData reduceTiling;

    if (ge::GetSizeByDataType(opInput.inputDtype) == sizeof(float)) {
        status = Tiling4ReduceOp<RIG::RIGDag<float, float>::OpDag>(context, opInput, key, compileInfo, &reduceTiling);
    } else {
        status = Tiling4ReduceOp<RIG::RIGDag<half, float>::OpDag>(context, opInput, key, compileInfo, &reduceTiling);
    }
    OP_CHECK_IF(
        (status == ge::GRAPH_FAILED),
        OP_LOGE(
            context,
            "ReduceOp Tiling failed, dtype shoude be in (int8/uint8/bfloat16/float16/float/int32/int64)"),
        return ge::GRAPH_FAILED);

    // set Tiling data
    RepeatInterleaveGradDavidTilingData* tiling = context->GetTilingData<RepeatInterleaveGradDavidTilingData>();
    tiling->reduceTiling = reduceTiling;

    return status;
}

static ge::graphStatus GetRIGDim(gert::TilingContext* context, int64_t& axis_, int64_t yDimNum)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* axis = attrs->GetInt(ATTR_AXIS_INDEX);

    if (axis == nullptr) {
        axis_ = yDimNum - 1;
    } else {
        axis_ = *axis;
        if (axis_ < 0) {
            axis_ = axis_ + yDimNum;
        }
        // check dim range
        OP_CHECK_IF(
            (axis_ < 0 && axis_ >= yDimNum),
            OP_LOGE(context, "RIG dim out of range"), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetRIGRepeat(gert::TilingContext* context, int64_t& repeat, const gert::Shape& yShape)
{
    auto repeatStorage = context->GetInputShape(INPUT_REPEAT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, repeatStorage);
    auto oneShapeRepeat = gert::Shape{1};
    const gert::Shape& repeatShape = EnsureNotScalar(repeatStorage->GetStorageShape(), oneShapeRepeat);
    int64_t repeatDimNum = repeatShape.GetDimNum();

    OP_CHECK_IF(
        (repeatDimNum != 1), OP_LOGE(context, "RIG repeat dim num is not 1"),
        return ge::GRAPH_FAILED);

    int64_t lenRepeat = repeatShape.GetDim(0);

    OP_CHECK_IF(
        (lenRepeat != 1), OP_LOGE(context, "RIG repeat has more than 1 number"),
        return ge::GRAPH_FAILED);

    // use input and output shape to get repeat
    auto xStorage = context->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorage);
    auto oneShapeX = gert::Shape{1};
    const gert::Shape& xShape = EnsureNotScalar(xStorage->GetStorageShape(), oneShapeX);
    int64_t xDimNum = xShape.GetDimNum();
    OP_CHECK_IF(
        xDimNum > MAX_YGRAD_DIM,
        OP_LOGE(context, "The yGrad Dim should not be greater than 8!"),
        return ge::GRAPH_FAILED);

    auto xShapeSize = xShape.GetShapeSize();
    auto yShapeSize = yShape.GetShapeSize();

    OP_CHECK_IF(
        xShapeSize == 0 || yShapeSize == 0,
        OP_LOGE(context, "The input or output shape is empty!"),
        return ge::GRAPH_FAILED);

    auto remain = xShapeSize % yShapeSize;
    OP_LOGI(context, "xShapeSize:%lu, yShapeSize:%lu", xShapeSize, yShapeSize);

    OP_CHECK_IF(
        (remain > 0), OP_LOGE(context, "y_grad is not multiple of x_grad"),
        return ge::GRAPH_FAILED);

    repeat = xShapeSize / yShapeSize;

    return ge::GRAPH_SUCCESS;
}

static void MergeAxis(
    ReduceOpInputParam& opInput, const gert::Shape& yShape, const int64_t repeat, const int64_t repeatDim)
{
    int64_t yDimNum = yShape.GetDimNum();
    int64_t M = 1;
    int64_t N = 1;

    // merge axis
    for (auto i = 0; i <= repeatDim; ++i) {
        M *= yShape[i];
    }
    for (auto i = repeatDim + 1; i < yDimNum; ++i) {
        N *= yShape[i];
    }

    // opInput: (M, R, N)
    opInput.axes.resize(1);
    opInput.axes[0] = 1;
    opInput.shape.resize(MERGED_AXES_NUM);
    opInput.shape[0] = M;
    opInput.shape[1] = repeat;
    opInput.shape[MERGED_AXES_NUM - 1] = N;
}

static ge::graphStatus GetRIGinputParam(gert::TilingContext* context, ReduceOpInputParam& opInput)
{
    // get x_grad output
    auto ytStorage = context->GetOutputShape(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, ytStorage);
    auto oneShape = gert::Shape{1};
    const gert::Shape& yShape = EnsureNotScalar(ytStorage->GetStorageShape(), oneShape);
    int64_t yDimNum = yShape.GetDimNum();

    OP_LOGI(context, "The output x_grad is: %s", ToString(yShape).c_str());

    int64_t repeatDim = 0;
    OP_CHECK_IF(
        (GetRIGDim(context, repeatDim, yDimNum) == ge::GRAPH_FAILED),
        OP_LOGE(context, "RIG get dim failed"), return ge::GRAPH_FAILED);

    int64_t repeat = 0;
    OP_CHECK_IF(
        (GetRIGRepeat(context, repeat, yShape) == ge::GRAPH_FAILED),
        OP_LOGE(context, "RIG get repeat failed"), return ge::GRAPH_FAILED);

    OP_LOGI(context, "dim:%lu, repeat:%lu", repeatDim, repeat);

    OP_CHECK_IF(
        (ReduceOpTmpl::GetInputDtype(context, 0, opInput.inputDtype) == ge::GRAPH_FAILED),
        OP_LOGE(context, "ReduceOp get x input dtype failed"),
        return ge::GRAPH_FAILED);

    MergeAxis(opInput, yShape, repeat, repeatDim);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4RIGIntRepeat(gert::TilingContext* context, const RepeatInterleaveGradCompileInfo* compileInfo)
{
    OP_LOGI(context, "entering Tiling4RIGIntRepeat");

    ReduceOpCompileInfo reduceCompileInfo;
    OP_CHECK_IF(
        (convertCompileInfo(compileInfo, &reduceCompileInfo, context) == ge::GRAPH_FAILED),
        OP_LOGE(context, "convert compile info Failed for Reduce template"),
        return ge::GRAPH_FAILED);

    ReduceOpInputParam opInput;
    OP_CHECK_IF(
        (GetRIGinputParam(context, opInput) == ge::GRAPH_FAILED),
        OP_LOGE(context, "ReduceOp get x input param failed"),
        return ge::GRAPH_FAILED);

    repeatInterleaveGradTilingKey key;
    OP_CHECK_IF(
        (DoTiling(context, &reduceCompileInfo, opInput, key.ReduceTiling) == ge::GRAPH_FAILED),
        OP_LOGE(context, "DoTiling Failed for RepeatInterleaveGrad"),
        return ge::GRAPH_FAILED);

    key.templateNum = RIG::IS_REDUCE_T;

    const uint64_t tilingKey = GET_TPL_TILING_KEY(
        key.ReduceTiling.patternID, key.ReduceTiling.loopARCount, key.ReduceTiling.loopInnerARCount, key.templateNum);
    OP_LOGI(
        context, "patternID:%u, loopARCount:%u, loopInnerARCount:%u, Tiling Key is:%lu",
        key.ReduceTiling.patternID, key.ReduceTiling.loopARCount, key.ReduceTiling.loopInnerARCount, tilingKey);
    context->SetTilingKey(tilingKey);

    OP_LOGI(context, "leaving Tiling4RIGIntRepeat");
    return ge::GRAPH_SUCCESS;
}

bool IsIntRepeat(const gert::TilingContext* context)
{
    auto repeatStorage = context->GetInputShape(INPUT_REPEAT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, repeatStorage);
    auto oneShape = gert::Shape{1};
    const gert::Shape& repeatShape = EnsureNotScalar(repeatStorage->GetStorageShape(), oneShape);

    int64_t repeatDimNum = repeatShape.GetDimNum();
    int64_t lenRepeat = repeatShape.GetDim(0);

    return repeatDimNum == 1 && lenRepeat == 1;
}

} // namespace RIGintRepeat

} // namespace optiling
