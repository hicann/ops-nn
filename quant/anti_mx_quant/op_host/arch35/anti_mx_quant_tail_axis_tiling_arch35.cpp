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
 * \file anti_mx_quant_tail_axis_tiling_arch35.cpp
 * \brief Tail-axis tiling implementation for AntiMxQuant operator.
 */

#include "anti_mx_quant_tiling_arch35.h"
#include "quant/anti_mx_quant/op_kernel/arch35/anti_mx_quant_tilingdata.h"
#include "quant/anti_mx_quant/op_kernel/arch35/anti_mx_quant_struct.h"
#include <cmath>
#include "platform/platform_info.h"

namespace optiling {
using namespace ge;
using namespace AntiMxQuantOp;

constexpr int64_t DIGIT_ONE = 1;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t SPLIT_M = 1;
constexpr int64_t SPLIT_N = 512;
constexpr int64_t DB_BUFFER = 2;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t SCALE_NUM = 16; // Ceil(512 / 32)

// Per-512-element data sizes (bytes) for different dtypes
constexpr int64_t BYTES_OF_INPUT_FP4_PER_512 = 256;         // 512 * 0.5
constexpr int64_t BYTES_OF_INPUT_FP8_PER_512 = 512;         // 512 * 1
constexpr int64_t BYTES_OF_OUTPUT_FP16_BF16_PER_512 = 1024; // 512 * 2
constexpr int64_t BYTES_OF_OUTPUT_FP32_PER_512 = 2048;      // 512 * 4
constexpr int64_t BYTES_OF_SCALE_PER_BLOCK = 16;            // 16 * 1 (FP8_E8M0)
constexpr int64_t BYTES_OF_UINT32_BUF_PER_BLOCK = 64;       // 16 * 4
constexpr int64_t BYTES_OF_UINT16_BUF_PER_BLOCK = 32;       // 16 * 2
constexpr int64_t NUM_OF_BLOCKSIZE_PER_512 = 16;            // 512 / 32

ge::graphStatus AntiMxQuantTailAxisTiling::SetTilingParams() const
{
    tilingData_->ubSize = tilingParam_.ubSize;
    tilingData_->dstType = tilingParam_.dstType;
    tilingData_->totalCoreNum = tilingParam_.totalCoreNum;
    tilingData_->usedCoreNum = tilingParam_.usedCoreNum;
    tilingData_->rowTileNum = tilingParam_.rowTileNum;
    tilingData_->colTileNum = tilingParam_.colTileNum;
    tilingData_->rowNum = tilingParam_.rowNum;
    tilingData_->colNum = tilingParam_.colNum;
    tilingData_->colNormalBlockNum = tilingParam_.colNormalBlockNum;
    tilingData_->colTailLen = tilingParam_.colTailLen;
    tilingData_->rowNormalBlockNum = tilingParam_.rowNormalBlockNum;
    tilingData_->rowTailLen = tilingParam_.rowTailLen;
    tilingData_->maxUbBlockNum = tilingParam_.maxUbBlockNum;
    return ge::GRAPH_SUCCESS;
}

void AntiMxQuantTailAxisTiling::PrintTilingDataForTailAxis() const
{
    OP_LOGI(context_->GetNodeName(),
            "tilingData is ubSize:%ld, dstType:%ld, "
            "totalCoreNum:%ld, usedCoreNum:%ld, rowTileNum:%ld, colTileNum:%ld, "
            "rowNum:%ld, colNum:%ld, colNormalBlockNum:%ld, colTailLen:%ld, "
            "rowNormalBlockNum:%ld, rowTailLen:%ld, maxUbBlockNum:%ld.",
            tilingData_->ubSize, tilingData_->dstType, tilingData_->totalCoreNum, tilingData_->usedCoreNum,
            tilingData_->rowTileNum, tilingData_->colTileNum, tilingData_->rowNum, tilingData_->colNum,
            tilingData_->colNormalBlockNum, tilingData_->colTailLen, tilingData_->rowNormalBlockNum,
            tilingData_->rowTailLen, tilingData_->maxUbBlockNum);
}

std::set<int64_t> AntiMxQuantTailAxisTiling::FindSplitCombo(int64_t usedCoreNum) const
{
    std::set<int64_t> result;
    int64_t upbound = std::ceil(std::sqrt(usedCoreNum) + 1);

    for (int64_t m = 1; m < upbound; m++) {
        int64_t y = usedCoreNum / m;
        result.insert(m);
        result.insert(y);
    }
    return result;
}

ge::graphStatus AntiMxQuantTailAxisTiling::AutoTiling()
{
    // Calculate available cores
    tilingParam_.usedCoreNum = std::min(tilingParam_.totalCoreNum,
                                        tilingParam_.rowBlockLoopNum * tilingParam_.colBlockLoopNum);
    tilingParam_.usedCoreNum = tilingParam_.usedCoreNum == 0 ? 1 : tilingParam_.usedCoreNum;

    // Find split combinations
    std::set<int64_t> cutSet = FindSplitCombo(tilingParam_.usedCoreNum);
    std::vector<std::vector<int64_t>> allTiling;

    // Enumerate row-direction split (m)
    for (int64_t m : cutSet) {
        if (m > tilingParam_.rowBlockLoopNum) {
            break;
        }
        int64_t n = tilingParam_.usedCoreNum / m;
        n = n < 1 ? 1 : n;
        if (n > tilingParam_.colBlockLoopNum) {
            continue;
        }

        int64_t rowNormalBlockNum = Ops::Base::CeilDiv(tilingParam_.rowBlockLoopNum, m);
        int64_t colNormalBlockNum = Ops::Base::CeilDiv(tilingParam_.colBlockLoopNum, n);
        int64_t delta = rowNormalBlockNum * colNormalBlockNum;
        if (m * n == static_cast<int64_t>(tilingParam_.usedCoreNum)) {
            if (tilingParam_.rowBlockLoopNum % m == 0 && tilingParam_.colBlockLoopNum % n == 0) {
                delta = 0;
            } else if (tilingParam_.rowBlockLoopNum % m == 0) {
                delta = delta - rowNormalBlockNum * (tilingParam_.colBlockLoopNum % colNormalBlockNum);
            } else if (tilingParam_.colBlockLoopNum % n == 0) {
                delta = delta - (tilingParam_.rowBlockLoopNum % rowNormalBlockNum) * colNormalBlockNum;
            } else {
                delta = delta - (tilingParam_.rowBlockLoopNum % rowNormalBlockNum) *
                                    (tilingParam_.colBlockLoopNum % colNormalBlockNum);
            }
        }

        allTiling.push_back({m, n, m * n, delta});
    }

    if (allTiling.empty()) {
        OP_LOGE(context_->GetNodeName(), "Failed to find valid tiling split combination.");
        return ge::GRAPH_FAILED;
    }

    // Sort: prefer smaller delta (more balanced), then prefer larger row split
    std::sort(allTiling.begin(), allTiling.end(), [](const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
        constexpr int FirstIndex = 1; // Prefer row split
        constexpr int DeltaIndex = 3;
        return std::make_pair(a[DeltaIndex], a[FirstIndex]) < std::make_pair(b[DeltaIndex], b[FirstIndex]);
    });

    tilingParam_.rowTileNum = static_cast<uint16_t>(allTiling[0][0]);
    tilingParam_.colTileNum = static_cast<uint16_t>(allTiling[0][1]);

    return ge::GRAPH_SUCCESS;
}

int64_t AntiMxQuantTailAxisTiling::CalcBytesPerBlock() const
{
    auto inputXPtr = context_->GetInputDesc(0);
    auto xDtype = inputXPtr->GetDataType();
    auto outputYPtr = context_->GetOutputDesc(0);
    auto yDtype = outputYPtr->GetDataType();

    int64_t inputBytes = 0;
    int64_t outputBytes = 0;

    bool isFp4 = false;
    // Determine input bytes per 512 elements
    if (xDtype == ge::DT_FLOAT4_E2M1 || xDtype == ge::DT_FLOAT4_E1M2) {
        inputBytes = BYTES_OF_INPUT_FP4_PER_512;
        isFp4 = true;
    } else { // FP8
        inputBytes = BYTES_OF_INPUT_FP8_PER_512;
    }

    // Determine output bytes per 512 elements
    if (yDtype == ge::DT_FLOAT16 || yDtype == ge::DT_BF16) {
        outputBytes = BYTES_OF_OUTPUT_FP16_BF16_PER_512;
    } else { // FP32
        outputBytes = BYTES_OF_OUTPUT_FP32_PER_512;
    }

    // Formula from requirement:
    // DB_BUFFER * (inputBytes + outputBytes + scaleBytes) + extraBufBytes
    // FP8: extraBufBytes = scaleNum * sizeof(uint32_t) = 16 * 4 = 64
    // FP4: extraBufBytes = scaleNum * sizeof(uint16_t) = 16 * 2 = 32
    int64_t extraBufBytes = isFp4 ? BYTES_OF_UINT16_BUF_PER_BLOCK : BYTES_OF_UINT32_BUF_PER_BLOCK;
    return DB_BUFFER * (inputBytes + outputBytes + BYTES_OF_SCALE_PER_BLOCK) + extraBufBytes;
}

void AntiMxQuantTailAxisTiling::CalcTilingKey() const
{
    // Only one template param: axis mode (0=tail axis, 1=non-tail axis)
    uint64_t axisMode = TPL_AXIS_TAIL;
    uint64_t tilingKey = GET_TPL_TILING_KEY(axisMode);
    context_->SetTilingKey(tilingKey);
}

void AntiMxQuantTailAxisTiling::CalcAxisSize(const gert::Shape& xShape)
{
    for (size_t i = 0; i < xShape.GetDimNum() - 1; i++) {
        tilingParam_.rowNum *= xShape.GetDim(i);
    }
    tilingParam_.colNum = xShape.GetDim(xShape.GetDimNum() - 1);
}

ge::graphStatus AntiMxQuantTailAxisTiling::DoTiling()
{
    OP_LOGD(context_->GetNodeName(), "AntiMxQuantTailAxisTiling DoTiling entering.");

    auto xShapePtr = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    CalcAxisSize(xShape);

    // AntiMxQuant: sliceSize is fixed at 512 (splitN=512, splitM=1)
    tilingParam_.rowBlockLoopNum = Ops::Base::CeilDiv(tilingParam_.rowNum, SPLIT_M);
    tilingParam_.colBlockLoopNum = Ops::Base::CeilDiv(tilingParam_.colNum, SPLIT_N);

    OP_CHECK_IF(AutoTiling() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "The auto tiling failed."),
                return ge::GRAPH_FAILED);

    tilingParam_.rowNormalBlockNum = Ops::Base::CeilDiv(tilingParam_.rowBlockLoopNum, tilingParam_.rowTileNum);
    tilingParam_.colNormalBlockNum = Ops::Base::CeilDiv(tilingParam_.colBlockLoopNum, tilingParam_.colTileNum);
    tilingParam_.rowTileNum = Ops::Base::CeilDiv(tilingParam_.rowBlockLoopNum, tilingParam_.rowNormalBlockNum);
    tilingParam_.colTileNum = Ops::Base::CeilDiv(tilingParam_.colBlockLoopNum, tilingParam_.colNormalBlockNum);
    tilingParam_.usedCoreNum = tilingParam_.rowTileNum * tilingParam_.colTileNum;
    tilingParam_.rowTailLen = tilingParam_.rowNum -
                              (tilingParam_.rowNormalBlockNum * (tilingParam_.rowTileNum - DIGIT_ONE));
    tilingParam_.colTailLen = tilingParam_.colNum -
                              (tilingParam_.colNormalBlockNum * SPLIT_N * (tilingParam_.colTileNum - DIGIT_ONE));

    // Calculate maxUbBlockNum based on dtype combination
    int64_t bytesPerBlock = CalcBytesPerBlock();
    int64_t maxUbBlockNum512 = tilingParam_.ubSize / bytesPerBlock;
    tilingParam_.maxUbBlockNum = maxUbBlockNum512 * NUM_OF_BLOCKSIZE_PER_512;

    CalcTilingKey();
    tilingData_ = context_->GetTilingData<AntiMxQuantTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tilingData_);

    OP_CHECK_IF(SetTilingParams() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_, "AntiMxQuantTailAxisTiling set tiling data failed."), return ge::GRAPH_FAILED);

    context_->SetBlockDim(tilingData_->usedCoreNum);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = 0; // AntiMxQuant does not need workspace for tail axis

    PrintTilingDataForTailAxis();
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
