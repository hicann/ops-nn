/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_batch_matmul_v4_arch31_tiling.cpp
 * \brief KirinX90/Kirin9030 arch31 S8S4 tiling
 */

#include "quant_batch_matmul_v4_arch31_tiling.h"

#include <cstring>
#include <limits>

#include "error_util.h"

#include "../../../op_kernel/arch31/quant_batch_matmul_v4_tiling_data.h"

namespace optiling {

namespace {
constexpr size_t IDX_X1 = 0;
constexpr size_t IDX_X2 = 1;
constexpr size_t IDX_BIAS = 2;
constexpr size_t IDX_X1_SCALE = 3;
constexpr size_t IDX_X2_SCALE = 4;
constexpr int32_t ATTR_IDX_TRANSPOSE_X1 = 2;
constexpr int32_t ATTR_IDX_TRANSPOSE_X2 = 3;
constexpr int32_t ATTR_IDX_GROUP_SIZE = 4;
constexpr int64_t TILING_M = 64;
constexpr int64_t TILING_N = 128;
constexpr int64_t GROUP_SIZE_MIN = 128;
constexpr int64_t GROUP_SIZE_MAX = 256;
constexpr int64_t K_FRACTAL_ALIGN = 64;
constexpr int64_t N_FRACTAL_ALIGN = 16;
constexpr int64_t B_FRAC_K = 64;
constexpr int64_t B_FRAC_N = 16;
constexpr int32_t BYTES_PER_INT32 = 4;
constexpr uint32_t BLOCK_DIM = 1U;
constexpr int64_t UB_SIZE_BYTES = 131072;
constexpr int64_t UB_FIXED_BYTES = 116480;
constexpr int64_t GT_MAX_L1 = 960;

template <typename T>
bool IsNd(const T* desc)
{
    if (desc == nullptr) {
        return false;
    }
    const auto fmt = static_cast<ge::Format>(ge::GetPrimaryFormat(desc->GetStorageFormat()));
    return fmt == ge::FORMAT_ND || fmt == ge::FORMAT_NCHW;
}

template <typename T>
bool IsFractalZ(const T* desc)
{
    if (desc == nullptr) {
        return false;
    }
    const auto fmt = static_cast<ge::Format>(ge::GetPrimaryFormat(desc->GetStorageFormat()));
    return fmt == ge::FORMAT_FRACTAL_Z;
}
} // namespace

ge::graphStatus QuantBatchMatmulV4Arch31Tiling(gert::TilingContext* context)
{
    OP_TILING_CHECK(context == nullptr, VECTOR_INNER_ERR_REPORT_TILIING("QuantBatchMatmulV4", "tilingContext is null"),
                    return ge::GRAPH_FAILED);
    const char* opName = context->GetNodeName();
    const auto* x1Shape = context->GetInputShape(IDX_X1);
    const auto* x2Shape = context->GetInputShape(IDX_X2);
    const auto* x1ScaleShape = context->GetOptionalInputShape(IDX_X1_SCALE);
    const auto* x2ScaleShape = context->GetOptionalInputShape(IDX_X2_SCALE);
    const auto* yShape = context->GetOutputShape(0);
    const auto* x1Desc = context->GetInputDesc(IDX_X1);
    const auto* x2Desc = context->GetInputDesc(IDX_X2);
    const auto* x1ScaleDesc = context->GetOptionalInputDesc(IDX_X1_SCALE);
    const auto* x2ScaleDesc = context->GetOptionalInputDesc(IDX_X2_SCALE);
    const auto* yDesc = context->GetOutputDesc(0);
    const auto* biasShape = context->GetOptionalInputShape(IDX_BIAS);
    const auto* biasDesc = context->GetOptionalInputDesc(IDX_BIAS);
    const auto* attrs = context->GetAttrs();
    OP_TILING_CHECK(x1Shape == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opName, "arch31 s8s4 x1 shape is null"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(x2Shape == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opName, "arch31 s8s4 x2 shape is null"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        x1ScaleShape == nullptr,
        VECTOR_INNER_ERR_REPORT_TILIING(opName, "arch31 s8s4 x1_scale shape is null (idx %zu)", IDX_X1_SCALE),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        x2ScaleShape == nullptr,
        VECTOR_INNER_ERR_REPORT_TILIING(opName, "arch31 s8s4 x2_scale shape is null (idx %zu)", IDX_X2_SCALE),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(yShape == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opName, "arch31 s8s4 y shape is null"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(x1Desc == nullptr || x2Desc == nullptr || x1ScaleDesc == nullptr || x2ScaleDesc == nullptr ||
                        yDesc == nullptr || attrs == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName, "input/output desc or attrs is null"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(x1Desc->GetDataType() != ge::DT_INT8 || !IsNd(x1Desc),
                    VECTOR_INNER_ERR_REPORT_TILIING(opName, "arch31 s8s4 requires x1 to be int8 ND, got dtype %d",
                                                    static_cast<int32_t>(x1Desc->GetDataType())),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(x2Desc->GetDataType() != ge::DT_INT4 || !IsFractalZ(x2Desc),
                    VECTOR_INNER_ERR_REPORT_TILIING(opName, "arch31 s8s4 requires x2 to be int4 FRACTAL_Z"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(x1ScaleDesc->GetDataType() != ge::DT_FLOAT || !IsNd(x1ScaleDesc),
                    VECTOR_INNER_ERR_REPORT_TILIING(opName, "arch31 s8s4 requires x1_scale to be float32 ND"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(x2ScaleDesc->GetDataType() != ge::DT_UINT64 || !IsNd(x2ScaleDesc),
                    VECTOR_INNER_ERR_REPORT_TILIING(opName, "arch31 s8s4 requires x2_scale to be uint64 ND"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(yDesc->GetDataType() != ge::DT_FLOAT16 || !IsNd(yDesc),
                    VECTOR_INNER_ERR_REPORT_TILIING(opName, "arch31 s8s4 requires y to be float16 ND"),
                    return ge::GRAPH_FAILED);
    const bool hasBias = (biasShape != nullptr) && (biasDesc != nullptr);
    OP_TILING_CHECK(hasBias && (biasDesc->GetDataType() != ge::DT_FLOAT16 || !IsNd(biasDesc)),
                    VECTOR_INNER_ERR_REPORT_TILIING(opName, "arch31 s8s4 requires bias to be float16 ND"),
                    return ge::GRAPH_FAILED);
    const bool* transposeX1Ptr = attrs->GetAttrPointer<bool>(ATTR_IDX_TRANSPOSE_X1);
    const int64_t* groupSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_IDX_GROUP_SIZE);
    const bool transposeX1 = (transposeX1Ptr != nullptr) && (*transposeX1Ptr);
    const int64_t groupSize = (groupSizePtr != nullptr) ? (*groupSizePtr) : (-1);
    OP_TILING_CHECK(transposeX1, VECTOR_INNER_ERR_REPORT_TILIING(opName, "arch31 s8s4 requires transpose_x1=false"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        groupSize != GROUP_SIZE_MIN && groupSize != GROUP_SIZE_MAX,
        VECTOR_INNER_ERR_REPORT_TILIING(opName, "arch31 s8s4 requires group_size in {128, 256}, got %ld", groupSize),
        return ge::GRAPH_FAILED);

    const auto& x1Dims = x1Shape->GetStorageShape();
    const auto& x2Dims = x2Shape->GetStorageShape();
    int64_t m = 0;
    int64_t kFromX1 = 0;
    if (x1Dims.GetDimNum() == 3U && x1Dims.GetDim(0) == 1) {
        m = x1Dims.GetDim(1);
        kFromX1 = x1Dims.GetDim(2);
    } else if (x1Dims.GetDimNum() == 2U) {
        m = x1Dims.GetDim(0);
        kFromX1 = x1Dims.GetDim(1);
    } else {
        VECTOR_INNER_ERR_REPORT_TILIING(opName, "arch31 s8s4 requires x1 shape (1,M,K) or (M,K), got dim num %zu",
                                        x1Dims.GetDimNum());
        return ge::GRAPH_FAILED;
    }
    OP_TILING_CHECK(x2Dims.GetDimNum() != 4U,
                    VECTOR_INNER_ERR_REPORT_TILIING(
                        opName, "arch31 s8s4 requires x2 FRACTAL_Z shape [K/64, N/16, 16, 64], got dim num %zu",
                        x2Dims.GetDimNum()),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(x2Dims.GetDim(2) != B_FRAC_N || x2Dims.GetDim(3) != B_FRAC_K,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName,
                                                    "arch31 s8s4 requires x2 FRACTAL_Z shape [K/64, N/16, 16, 64], "
                                                    "got dim2=%ld dim3=%ld",
                                                    x2Dims.GetDim(2), x2Dims.GetDim(3)),
                    return ge::GRAPH_FAILED);
    int64_t n = 0;
    int64_t kFromX2 = 0;
    kFromX2 = x2Dims.GetDim(0) * B_FRAC_K;
    n = x2Dims.GetDim(1) * B_FRAC_N;

    OP_TILING_CHECK(
        m <= 0 || n <= 0 || kFromX1 <= 0,
        VECTOR_INNER_ERR_REPORT_TILIING(opName, "arch31 s8s4 requires M/N/K > 0, got M=%ld N=%ld K=%ld", m, n, kFromX1),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        kFromX1 != kFromX2,
        VECTOR_INNER_ERR_REPORT_TILIING(opName, "arch31 s8s4 K mismatch: x1 K=%ld, x2 K=%ld (dim0 = %ld * 64)", kFromX1,
                                        kFromX2, x2Dims.GetDim(0)),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(kFromX1 % K_FRACTAL_ALIGN != 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName, "arch31 s8s4 requires K %% 64 == 0, got K=%ld", kFromX1),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(n % N_FRACTAL_ALIGN != 0,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName, "arch31 s8s4 requires N %% 16 == 0, got N=%ld", n),
                    return ge::GRAPH_FAILED);

    const auto& x1ScaleDims = x1ScaleShape->GetStorageShape();
    int64_t x1ScaleNums = 1;
    for (size_t i = 0; i < x1ScaleDims.GetDimNum(); i++) {
        x1ScaleNums *= x1ScaleDims.GetDim(i);
    }
    OP_TILING_CHECK(
        x1ScaleNums != m,
        VECTOR_INNER_ERR_REPORT_TILIING(opName, "arch31 s8s4 requires x1_scale numel %ld == M %ld", x1ScaleNums, m),
        return ge::GRAPH_FAILED);

    const int64_t groupNum = (kFromX1 + groupSize - 1) / groupSize;
    const auto& x2ScaleDims = x2ScaleShape->GetStorageShape();
    OP_TILING_CHECK(x2ScaleDims.GetDimNum() != 2U || x2ScaleDims.GetDim(0) != groupNum || x2ScaleDims.GetDim(1) != n,
                    VECTOR_INNER_ERR_REPORT_TILIING(
                        opName, "arch31 s8s4 requires x2_scale shape (ceil(K/gs), N) = (%ld, %ld)", groupNum, n),
                    return ge::GRAPH_FAILED);

    if (hasBias) {
        OP_TILING_CHECK(
            n < 16,
            VECTOR_INNER_ERR_REPORT_TILIING(opName, "arch31 s8s4 requires N >= 16 when bias present, got N=%ld", n),
            return ge::GRAPH_FAILED);
        const auto& biasDims = biasShape->GetStorageShape();
        int64_t biasNums = 1;
        for (size_t i = 0; i < biasDims.GetDimNum(); i++) {
            biasNums *= biasDims.GetDim(i);
        }
        OP_TILING_CHECK(
            biasNums != n,
            VECTOR_INNER_ERR_REPORT_TILIING(opName, "arch31 s8s4 requires bias numel %ld == N %ld", biasNums, n),
            return ge::GRAPH_FAILED);
    }

    const auto& yDims = yShape->GetStorageShape();
    int64_t yM = 0;
    int64_t yN = 0;
    if (yDims.GetDimNum() == 3U && yDims.GetDim(0) == 1) {
        yM = yDims.GetDim(1);
        yN = yDims.GetDim(2);
    } else if (yDims.GetDimNum() == 2U) {
        yM = yDims.GetDim(0);
        yN = yDims.GetDim(1);
    }
    OP_TILING_CHECK(yM != m || yN != n,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName, "arch31 s8s4 requires y shape (M,N)=(%ld,%ld)", m, n),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(m > std::numeric_limits<int32_t>::max() || n > std::numeric_limits<int32_t>::max() ||
                        kFromX1 > std::numeric_limits<int32_t>::max(),
                    VECTOR_INNER_ERR_REPORT_TILIING(opName, "arch31 s8s4 M/N/K exceed int32 range"),
                    return ge::GRAPH_FAILED);
    const int64_t ubFixedBytes = UB_FIXED_BYTES;
    OP_TILING_CHECK(ubFixedBytes > UB_SIZE_BYTES,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName,
                                                    "arch31 s8s4 UB overflow: fixed buffers %ld B > ub_size %ld B "
                                                    "(TILE_M=%ld TILE_N=%ld)",
                                                    ubFixedBytes, UB_SIZE_BYTES, TILING_M, TILING_N),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(groupNum > GT_MAX_L1,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName,
                                                    "arch31 s8s4 requires GT = ceil(K/gs) <= %ld (sL1 deq buffer "
                                                    "capacity), got K=%ld gs=%ld GT=%ld",
                                                    GT_MAX_L1, kFromX1, groupSize, groupNum),
                    return ge::GRAPH_FAILED);
    QuantBatchMatmulV4Arch31TilingData tilingData{};
    tilingData.m = static_cast<int32_t>(m);
    tilingData.n = static_cast<int32_t>(n);
    tilingData.k = static_cast<int32_t>(kFromX1);
    tilingData.groupSize = static_cast<int32_t>(groupSize);
    tilingData.tilingM = static_cast<int32_t>(TILING_M);
    tilingData.tilingN = static_cast<int32_t>(TILING_N);
    tilingData.blockDim = static_cast<int32_t>(BLOCK_DIM);
    tilingData.hasBias = hasBias ? 1 : 0;

    auto* raw = context->GetRawTilingData();
    OP_TILING_CHECK(raw == nullptr || raw->GetData() == nullptr || raw->GetCapacity() < sizeof(tilingData),
                    VECTOR_INNER_ERR_REPORT_TILIING(opName, "arch31 s8s4 raw tiling data unavailable"),
                    return ge::GRAPH_FAILED);
    errno_t ret = memcpy_s(raw->GetData(), raw->GetCapacity(), &tilingData, sizeof(tilingData));
    OP_TILING_CHECK(ret != EOK,
                    VECTOR_INNER_ERR_REPORT_TILIING(opName, "arch31 s8s4 memcpy tiling data failed, ret=%d",
                                                    static_cast<int32_t>(ret)),
                    return ge::GRAPH_FAILED);
    raw->SetDataSize(sizeof(tilingData));

    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(opName, "arch31 s8s4 workspace is null"),
                    return ge::GRAPH_FAILED);
    const size_t partialBytes = 2U * static_cast<size_t>(TILING_M) * static_cast<size_t>(TILING_N) * sizeof(uint16_t);
    size_t x2StreamBytes = 0;
    workspaces[0] = partialBytes + x2StreamBytes;

    context->SetBlockDim(BLOCK_DIM);
    if (m == 1) {
        context->SetTilingKey(1U);
    } else {
        context->SetTilingKey(0U);
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
