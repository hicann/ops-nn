/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#include "sparse_fill_empty_rows_aicpu.h"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <string>
#include <vector>

#include "cpu_kernel_utils.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"
#include "utils/log.h"

namespace {
using namespace aicpu;

const char* const kSparseFillEmptyRows = "SparseFillEmptyRows";
const uint32_t kInputNum = 4;
const uint32_t kOutputNum = 4;
constexpr int32_t kIndicesInput = 0;
constexpr int32_t kValuesInput = 1;
constexpr int32_t kDenseShapeInput = 2;
constexpr int32_t kDefaultValueInput = 3;
constexpr int32_t kYIndicesOutput = 0;
constexpr int32_t kYValuesOutput = 1;
constexpr int32_t kEmptyRowIndicatorOutput = 2;
constexpr int32_t kReverseIndexMapOutput = 3;

#define SPARSE_FILL_EMPTY_ROWS_DATA_TYPE_CASE(DTYPE, TYPE)                                      \
    case (DTYPE): {                                                                             \
        ret = ComputeSparseFillEmptyRows<TYPE>(ctx, indices, values, denseShape, defaultValue); \
        break;                                                                                  \
    }
} // namespace

namespace aicpu {
template <typename T>
void SparseFillEmptyRowsCpuKernel::FillValue(const TTypes<int64_t>::Matrix& indicesMatrix,
                                             std::vector<int64_t>& csrOffset,
                                             TTypes<int64_t>::Matrix& outputIndicesMatrix, const T* valuesData,
                                             T* outputValuesData, int64_t* reverseIndexMapData, const T defaultValue)
{
    const int64_t n = indicesMatrix.dimension(0);
    const int64_t rank = indicesMatrix.dimension(1);
    const int64_t denseRows = static_cast<int64_t>(csrOffset.size());
    std::vector<int64_t> filledCount(denseRows, 0);

    for (int64_t i = 0; i < n; ++i) {
        const int64_t row = indicesMatrix(i, 0);
        int64_t& offset = filledCount[row];
        const int64_t outputI = ((row == 0) ? 0 : csrOffset[row - 1]) + offset;
        ++offset;
        (void)std::copy_n(&indicesMatrix(i, 0), rank, &outputIndicesMatrix(outputI, 0));
        outputValuesData[outputI] = valuesData[i];
        reverseIndexMapData[i] = outputI;
    }

    for (int64_t row = 0; row < denseRows; ++row) {
        if (filledCount[row] != 0) {
            continue;
        }
        const int64_t startingIndex = (row == 0) ? 0 : csrOffset[row - 1];
        outputIndicesMatrix(startingIndex, 0) = row;
        for (int64_t col = 1; col < rank; ++col) {
            outputIndicesMatrix(startingIndex, col) = 0;
        }
        outputValuesData[startingIndex] = defaultValue;
    }
}

template <typename T>
KernelStatus SparseFillEmptyRowsCpuKernel::ComputeSparseFillEmptyRows(const CpuKernelContext& ctx, Tensor* indices,
                                                                      const Tensor* values, const Tensor* denseShape,
                                                                      const Tensor* defaultValueTensor)
{
    Tensor* outputIndices = ctx.Output(kYIndicesOutput);
    Tensor* outputValues = ctx.Output(kYValuesOutput);
    Tensor* emptyRowIndicator = ctx.Output(kEmptyRowIndicatorOutput);
    Tensor* reverseIndexMap = ctx.Output(kReverseIndexMapOutput);

    const T defaultValue = reinterpret_cast<T*>(defaultValueTensor->GetData())[0];
    const int64_t n = indices->GetTensorShape()->GetDimSize(0);
    const int64_t denseRows = reinterpret_cast<int64_t*>(denseShape->GetData())[0];
    const int64_t rank = indices->GetTensorShape()->GetDimSize(1);
    if (denseRows == 0) {
        if (n != 0) {
            KERNEL_LOG_ERROR("dense_shape[0] = 0 but indices.shape[0] = %ld", n);
            return KERNEL_STATUS_PARAM_INVALID;
        }
        outputIndices->GetTensorShape()->SetDimSizes({0, rank});
        outputValues->GetTensorShape()->SetDimSizes({0});
        return KERNEL_STATUS_OK;
    }

    bool rowsAreOrdered = true;
    int64_t lastIndicesRow = 0;
    std::vector<int64_t> csrOffset(denseRows, 0);

    EigenTensor indicesEigen(indices, indices->GetData());
    const auto indicesMatrix = indicesEigen.matrix<int64_t>();
    auto* valuesData = reinterpret_cast<T*>(values->GetData());
    auto* emptyRowIndicatorData = reinterpret_cast<bool*>(emptyRowIndicator->GetData());
    auto* reverseIndexMapData = reinterpret_cast<int64_t*>(reverseIndexMap->GetData());

    EigenTensor outputIndicesEigen(outputIndices, outputIndices->GetData());
    auto outputIndicesMatrix = outputIndicesEigen.matrix<int64_t>();
    auto* outputValuesData = reinterpret_cast<T*>(outputValues->GetData());
    for (int64_t i = 0; i < n; ++i) {
        const int64_t row = indicesMatrix(i, 0);
        if (row < 0 || row >= denseRows) {
            KERNEL_LOG_ERROR("indices(%ld, 0) value is %ld which is invalid.", i, row);
            return KERNEL_STATUS_PARAM_INVALID;
        }
        ++csrOffset[row];
        rowsAreOrdered = rowsAreOrdered && (row >= lastIndicesRow);
        lastIndicesRow = row;
    }

    bool allRowsFull = true;
    for (int64_t row = 0; row < denseRows; ++row) {
        const bool rowEmpty = (csrOffset[row] == 0);
        emptyRowIndicatorData[row] = rowEmpty;
        allRowsFull = allRowsFull && !rowEmpty;
        csrOffset[row] = std::max(csrOffset[row], static_cast<int64_t>(1));
        if (row > 0) {
            csrOffset[row] += csrOffset[row - 1];
        }
    }

    if (allRowsFull && rowsAreOrdered) {
        (void)std::copy_n(&indicesMatrix(0, 0), n * rank, &outputIndicesMatrix(0, 0));
        (void)std::copy_n(valuesData, n, outputValuesData);
        for (int64_t i = 0; i < n; ++i) {
            reverseIndexMapData[i] = i;
        }
    } else {
        FillValue(indicesMatrix, csrOffset, outputIndicesMatrix, valuesData, outputValuesData, reverseIndexMapData,
                  defaultValue);
    }
    outputIndices->GetTensorShape()->SetDimSizes({csrOffset[denseRows - 1], rank});
    outputValues->GetTensorShape()->SetDimSizes({csrOffset[denseRows - 1]});
    return KERNEL_STATUS_OK;
}

uint32_t SparseFillEmptyRowsCpuKernel::Compute(CpuKernelContext& ctx)
{
    KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum),
                        "SparseFillEmptyRows check input and output number failed.");

    Tensor* indices = ctx.Input(kIndicesInput);
    Tensor* values = ctx.Input(kValuesInput);
    Tensor* denseShape = ctx.Input(kDenseShapeInput);
    Tensor* defaultValue = ctx.Input(kDefaultValueInput);

    KernelStatus ret = KERNEL_STATUS_OK;
    switch (values->GetDataType()) {
        SPARSE_FILL_EMPTY_ROWS_DATA_TYPE_CASE(DT_BOOL, bool)
        SPARSE_FILL_EMPTY_ROWS_DATA_TYPE_CASE(DT_COMPLEX128, std::complex<double>)
        SPARSE_FILL_EMPTY_ROWS_DATA_TYPE_CASE(DT_COMPLEX64, std::complex<float>)
        SPARSE_FILL_EMPTY_ROWS_DATA_TYPE_CASE(DT_DOUBLE, double)
        SPARSE_FILL_EMPTY_ROWS_DATA_TYPE_CASE(DT_FLOAT, float)
        SPARSE_FILL_EMPTY_ROWS_DATA_TYPE_CASE(DT_FLOAT16, Eigen::half)
        SPARSE_FILL_EMPTY_ROWS_DATA_TYPE_CASE(DT_INT16, int16_t)
        SPARSE_FILL_EMPTY_ROWS_DATA_TYPE_CASE(DT_INT32, int32_t)
        SPARSE_FILL_EMPTY_ROWS_DATA_TYPE_CASE(DT_INT64, int64_t)
        SPARSE_FILL_EMPTY_ROWS_DATA_TYPE_CASE(DT_INT8, int8_t)
        SPARSE_FILL_EMPTY_ROWS_DATA_TYPE_CASE(DT_UINT16, uint16_t)
        SPARSE_FILL_EMPTY_ROWS_DATA_TYPE_CASE(DT_UINT32, uint32_t)
        SPARSE_FILL_EMPTY_ROWS_DATA_TYPE_CASE(DT_UINT64, uint64_t)
        SPARSE_FILL_EMPTY_ROWS_DATA_TYPE_CASE(DT_UINT8, uint8_t)
        default:
            KERNEL_LOG_ERROR("SparseFillEmptyRows doesn't support data type [%s]",
                             DTypeStr(values->GetDataType()).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
    }
    if (ret != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("SparseFillEmptyRows compute failed.");
    }
    return static_cast<uint32_t>(ret);
}

REGISTER_CPU_KERNEL(kSparseFillEmptyRows, SparseFillEmptyRowsCpuKernel);
} // namespace aicpu
