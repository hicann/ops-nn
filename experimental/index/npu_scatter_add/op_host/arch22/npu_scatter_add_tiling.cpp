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
 * \file npu_scatter_add_tiling.cpp
 * \brief NpuScatterAdd tiling 实现：按核切分行任务，tilingKey 编码 dtype/缩放/精度模式
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "npu_scatter_add_tiling.h"

namespace optiling {
// tiling keys
// low precision
constexpr uint64_t TILING_KEY_BF16 = 0;
constexpr uint64_t TILING_KEY_FP16 = 1;
constexpr uint64_t TILING_KEY_BF16_NO_SCALE = 2;
constexpr uint64_t TILING_KEY_FP16_NO_SCALE = 3;
// high precision
constexpr uint64_t TILING_KEY_BF16_HIGH_PRECISION = 4;
constexpr uint64_t TILING_KEY_FP16_HIGH_PRECISION = 5;
constexpr uint64_t TILING_KEY_BF16_NO_SCALE_HIGH_PRECISION = 6;
constexpr uint64_t TILING_KEY_FP16_NO_SCALE_HIGH_PRECISION = 7;
constexpr uint64_t TILING_KEY_HIGH_PRECISION_OFFSET = 4;

// input indices
constexpr size_t X_INDEX = 0;
constexpr size_t Y_INDEX = 1;
constexpr size_t OPTIONAL_S_INDEX = 2;
constexpr size_t IDX_INDEX = 3;
constexpr size_t SORT_IDX_INDEX = 4;
constexpr size_t OPTIONAL_VALID_TOKEN_NUM_INDEX = 5;

// attr index
constexpr size_t USE_HIGH_PRECISION_INDEX = 0;

// some constants
constexpr uint32_t MAX_UB_SIZE = 180 * 1024;
constexpr uint32_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr uint32_t BYTES_ALIGN = 32;
constexpr uint32_t BYTES_PER_ELEMENT = 2; // bf16/fp16

static ge::graphStatus Tiling4NpuScatterAdd(gert::TilingContext* context)
{
    const gert::Shape& xShape = context->GetInputShape(X_INDEX)->GetStorageShape();
    const gert::Shape& yShape = context->GetInputShape(Y_INDEX)->GetStorageShape();
    const gert::Shape& idxShape = context->GetInputShape(IDX_INDEX)->GetStorageShape();
    const gert::Shape& sortIdxShape = context->GetInputShape(SORT_IDX_INDEX)->GetStorageShape();
    const gert::Shape& outShape = context->GetOutputShape(0)->GetStorageShape();

    // check dim
    if (xShape.GetDimNum() != 2 || yShape.GetDimNum() != 2 || idxShape.GetDimNum() != 1 ||
        sortIdxShape.GetDimNum() != 1 || outShape.GetDimNum() != 2) {
        OP_LOGE(context,
                "Expected dim of [x, y, idx, sort_idx, out] should be [2,2,1,1,2], but now is [%zu,%zu,%zu,%zu,%zu].",
                xShape.GetDimNum(), yShape.GetDimNum(), idxShape.GetDimNum(), sortIdxShape.GetDimNum(),
                outShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    // check size
    if (xShape.GetDim(1) != yShape.GetDim(1) || yShape.GetDim(1) != outShape.GetDim(1)) {
        OP_LOGE(context, "Expect x,y,output's dim[1] to be same, but now is [%ld,%ld,%ld].", xShape.GetDim(1),
                yShape.GetDim(1), outShape.GetDim(1));
        return ge::GRAPH_FAILED;
    }

    if (xShape.GetDim(0) != idxShape.GetDim(0) || xShape.GetDim(0) != sortIdxShape.GetDim(0)) {
        OP_LOGE(context, "Expect x,idx,sort_idx's dim[0] to be same, but now is [%ld,%ld,%ld].", xShape.GetDim(0),
                idxShape.GetDim(0), sortIdxShape.GetDim(0));
        return ge::GRAPH_FAILED;
    }

    // check dtype
    auto xDtype = context->GetInputDesc(X_INDEX)->GetDataType();
    auto yDtype = context->GetInputDesc(Y_INDEX)->GetDataType();
    auto idxDtype = context->GetInputDesc(IDX_INDEX)->GetDataType();
    auto sortIdxDtype = context->GetInputDesc(SORT_IDX_INDEX)->GetDataType();
    auto outDtype = context->GetOutputDesc(0)->GetDataType();

    if (xDtype != yDtype || yDtype != outDtype) {
        OP_LOGE(context, "Expect x,y,out's dtype to be same.");
        return ge::GRAPH_FAILED;
    }

    bool useScale = false;
    // check scale info
    auto optionalScale = context->GetOptionalInputDesc(OPTIONAL_S_INDEX);
    if (optionalScale != nullptr) {
        useScale = true;
        // check dim
        const gert::Shape& scaleShape = context->GetOptionalInputShape(OPTIONAL_S_INDEX)->GetStorageShape();
        if (scaleShape.GetDimNum() != 1) {
            OP_LOGE(context, "scale's dim should be 1, but get %zu.", scaleShape.GetDimNum());
            return ge::GRAPH_FAILED;
        }
        // check size
        if (scaleShape.GetDim(0) != xShape.GetDim(0)) {
            OP_LOGE(context, "scale's dim[0](%ld) should be same as x's dim[0](%ld).", scaleShape.GetDim(0),
                    xShape.GetDim(0));
            return ge::GRAPH_FAILED;
        }
        // check dtype
        auto scaleDtype = optionalScale->GetDataType();
        if (scaleDtype != xDtype) {
            OP_LOGE(context, "scale's dtype should be same as x's dtype.");
            return ge::GRAPH_FAILED;
        }
    }

    auto optionalValid = context->GetOptionalInputDesc(OPTIONAL_VALID_TOKEN_NUM_INDEX);
    bool hasValidNum = optionalValid != nullptr;
    if (hasValidNum && optionalValid->GetDataType() != ge::DT_INT32) {
        OP_LOGE(context, "valid_token_num's dtype should be int32.");
        return ge::GRAPH_FAILED;
    }
    if (idxDtype != ge::DT_INT32 || sortIdxDtype != ge::DT_INT32) {
        OP_LOGE(context, "indices and sort_idx's dtype should be int32.");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    context->SetBlockDim(coreNum);

    const bool useHighPrecision = *(context->GetAttrs()->GetAttrPointer<bool>(USE_HIGH_PRECISION_INDEX));

    auto getTilingKey = [&](const uint64_t x) {
        if (useHighPrecision) {
            return x + TILING_KEY_HIGH_PRECISION_OFFSET;
        }
        return x;
    };

    // set tiling key
    if (useScale) {
        if (xDtype == ge::DT_BF16) {
            context->SetTilingKey(getTilingKey(TILING_KEY_BF16));
        } else if (xDtype == ge::DT_FLOAT16) {
            context->SetTilingKey(getTilingKey(TILING_KEY_FP16));
        } else {
            OP_LOGE(context, "Dtype of input is not in fp16 / bf16.");
            return ge::GRAPH_FAILED;
        }
    } else { // no mul scale
        if (xDtype == ge::DT_BF16) {
            context->SetTilingKey(getTilingKey(TILING_KEY_BF16_NO_SCALE));
        } else if (xDtype == ge::DT_FLOAT16) {
            context->SetTilingKey(getTilingKey(TILING_KEY_FP16_NO_SCALE));
        } else {
            OP_LOGE(context, "Dtype of input is not in fp16 / bf16.");
            return ge::GRAPH_FAILED;
        }
    }

    NpuScatterAddTilingData tiling;
    const uint32_t totalRows = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t hiddenState = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t bytesPerEle = (xDtype == ge::DT_BF16 || xDtype == ge::DT_FLOAT16) ? BYTES_PER_ELEMENT : 0;
    const uint32_t alignHiddenState = (hiddenState * bytesPerEle + BYTES_ALIGN - 1) / BYTES_ALIGN * BYTES_ALIGN /
                                      bytesPerEle;

    // Kernel allocates 6 UB buffers: 3 x DataType + 3 x float, each of size
    // alignHiddenState, plus a 32-byte reduce_row_id buffer.
    if (alignHiddenState * (bytesPerEle * 3 + sizeof(float) * 3) + BYTES_ALIGN > MAX_UB_SIZE) {
        OP_LOGE(context, "x's dim[1] is too large. Currently not supported.");
        return ge::GRAPH_FAILED;
    }

    tiling.set_totalRows(totalRows);
    tiling.set_usedCoreNum(coreNum);
    tiling.set_hiddenState(hiddenState);
    tiling.set_alignHiddenState(alignHiddenState);
    tiling.set_withValid(hasValidNum ? 1 : 0);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // set workspace
    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = SYS_WORKSPACE_SIZE + coreNum * alignHiddenState * bytesPerEle // reduce space
                    + coreNum * sizeof(int32_t);                                  // idx space

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(NpuScatterAdd).Tiling(Tiling4NpuScatterAdd);
} // namespace optiling
