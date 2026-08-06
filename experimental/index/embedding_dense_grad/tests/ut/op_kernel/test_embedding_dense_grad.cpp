/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "data_utils.h"
#include "string.h"
#include "tikicpulib.h"
#endif

#include "../../../op_host/embedding_dense_grad_tiling.h"
#include "../../../op_kernel/embedding_dense_grad_tiling_key.h"
#include "../../../op_kernel/embedding_dense_grad.cpp"

using namespace std;

namespace {
constexpr size_t kBlockBytes = 32;

size_t AlignUp(size_t size, size_t align) { return ((size + align - 1) / align) * align; }

template <typename T>
void CopyToGm(uint8_t* gm, const vector<T>& data)
{
    auto ptr = reinterpret_cast<T*>(gm);
    for (size_t i = 0; i < data.size(); ++i) {
        ptr[i] = data[i];
    }
}
} // namespace

class EmbeddingDenseGradKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "EmbeddingDenseGradKernelTest SetUp" << endl; }

    static void TearDownTestCase() { cout << "EmbeddingDenseGradKernelTest TearDown" << endl; }
};

// Scenario: SINGLE_ROW, fp32, scaleGradByFreq=false
// grad = [[1.0, 2.0], [3.0, 4.0]]  (batchSize=2, dimSize=2)
// indices = [0, 2], numWeights=4, paddingIdx=-1
// expected y = [[1.0, 2.0], [0.0, 0.0], [3.0, 4.0], [0.0, 0.0]]
TEST_F(EmbeddingDenseGradKernelTest, embedding_dense_grad_float32_single_row_smoke)
{
    constexpr size_t batchSize = 2;
    constexpr size_t dimSize = 2;
    constexpr size_t numWeights = 4;
    constexpr int32_t paddingIdx = -1;
    constexpr uint32_t blockDim = 1;

    const vector<float> gradData{1.0f, 2.0f, 3.0f, 4.0f};
    const vector<int32_t> indicesData{0, 2};
    const vector<float> expected{1.0f, 2.0f, 0.0f, 0.0f, 3.0f, 4.0f, 0.0f, 0.0f};

    constexpr size_t kAlignElems = 8;
    constexpr size_t kFloatBytes = sizeof(float);
    size_t gradAlignElems = AlignUp(batchSize * dimSize, kAlignElems) + kAlignElems;
    size_t yAlignElems = AlignUp(numWeights * dimSize, kAlignElems) + kAlignElems;
    size_t gradByteSize = gradAlignElems * kFloatBytes;
    size_t indicesByteSize = AlignUp(batchSize * sizeof(int32_t), kBlockBytes);
    size_t yByteSize = yAlignElems * kFloatBytes;
    size_t align8NumWeights = (numWeights + 7) / 8 * 8;
    size_t userWorkspaceByteSize = 2 * 1024 + align8NumWeights * sizeof(int32_t) + 64 * sizeof(int32_t);
    size_t workspaceByteSize = AlignUp(4 * 1024 + userWorkspaceByteSize, kBlockBytes);
    size_t tilingDataSize = 1024;

    uint8_t* grad = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(gradByteSize));
    uint8_t* indices = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(indicesByteSize));
    uint8_t* y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(yByteSize));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(workspaceByteSize));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tilingDataSize));

    ASSERT_NE(grad, nullptr);
    ASSERT_NE(indices, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    memset(grad, 0, gradByteSize);
    memset(indices, 0, indicesByteSize);
    memset(y, 0, yByteSize);
    memset(workspace, 0, workspaceByteSize);
    memset(tiling, 0, tilingDataSize);

    CopyToGm(grad, gradData);
    CopyToGm(indices, indicesData);

    optiling::EmbeddingDenseGradTilingData tilingData;
    tilingData.set_dimSize(dimSize);
    tilingData.set_numWeights(static_cast<int32_t>(numWeights));
    tilingData.set_paddingIdx(paddingIdx);
    tilingData.set_scaleGradByFreq(0);
    tilingData.set_formerCoreNum(blockDim);
    tilingData.set_formerBatchSize(batchSize);
    tilingData.set_tailBatchSize(0);
    tilingData.set_scaleFormerCoreNum(0);
    tilingData.set_scaleFormerBatchSize(0);
    tilingData.set_scaleTailBatchSize(0);
    tilingData.set_ubProcessNum(static_cast<int64_t>(dimSize));
    tilingData.set_scaleUbProcessNum(0);
    tilingData.SaveToBuffer(tiling, tilingDataSize);

    auto kernel = embedding_dense_grad<EMBEDDING_DENSE_GRAD_SCH_MODE_SINGLE_ROW>;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, blockDim, grad, indices, y, workspace, tiling);

    auto yData = reinterpret_cast<float*>(y);
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(yData[i], expected[i]) << "mismatch at index " << i;
    }

    AscendC::GmFree(grad);
    AscendC::GmFree(indices);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// Scenario: SINGLE_ROW, fp32, scaleGradByFreq=false, duplicate index accumulation
// grad = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]  (batchSize=3, dimSize=2)
// indices = [1, 3, 1], numWeights=4, paddingIdx=-1
// expected y = [[0, 0], [6.0, 8.0], [0, 0], [3.0, 4.0]]  (index 1 accumulated twice)
TEST_F(EmbeddingDenseGradKernelTest, embedding_dense_grad_float32_duplicate_index_smoke)
{
    constexpr size_t batchSize = 3;
    constexpr size_t dimSize = 2;
    constexpr size_t numWeights = 4;
    constexpr int32_t paddingIdx = -1;
    constexpr uint32_t blockDim = 1;

    const vector<float> gradData{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const vector<int32_t> indicesData{1, 3, 1};
    const vector<float> expected{0.0f, 0.0f, 6.0f, 8.0f, 0.0f, 0.0f, 3.0f, 4.0f};

    constexpr size_t kAlignElems = 8;
    constexpr size_t kFloatBytes = sizeof(float);
    size_t gradAlignElems = AlignUp(batchSize * dimSize, kAlignElems) + kAlignElems;
    size_t yAlignElems = AlignUp(numWeights * dimSize, kAlignElems) + kAlignElems;
    size_t gradByteSize = gradAlignElems * kFloatBytes;
    size_t indicesByteSize = AlignUp(batchSize * sizeof(int32_t), kBlockBytes);
    size_t yByteSize = yAlignElems * kFloatBytes;
    size_t align8NumWeights = (numWeights + 7) / 8 * 8;
    size_t userWorkspaceByteSize = 2 * 1024 + align8NumWeights * sizeof(int32_t) + 64 * sizeof(int32_t);
    size_t workspaceByteSize = AlignUp(4 * 1024 + userWorkspaceByteSize, kBlockBytes);
    size_t tilingDataSize = 1024;

    uint8_t* grad = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(gradByteSize));
    uint8_t* indices = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(indicesByteSize));
    uint8_t* y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(yByteSize));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(workspaceByteSize));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tilingDataSize));

    ASSERT_NE(grad, nullptr);
    ASSERT_NE(indices, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    memset(grad, 0, gradByteSize);
    memset(indices, 0, indicesByteSize);
    memset(y, 0, yByteSize);
    memset(workspace, 0, workspaceByteSize);
    memset(tiling, 0, tilingDataSize);

    CopyToGm(grad, gradData);
    CopyToGm(indices, indicesData);

    optiling::EmbeddingDenseGradTilingData tilingData;
    tilingData.set_dimSize(dimSize);
    tilingData.set_numWeights(static_cast<int32_t>(numWeights));
    tilingData.set_paddingIdx(paddingIdx);
    tilingData.set_scaleGradByFreq(0);
    tilingData.set_formerCoreNum(blockDim);
    tilingData.set_formerBatchSize(batchSize);
    tilingData.set_tailBatchSize(0);
    tilingData.set_scaleFormerCoreNum(0);
    tilingData.set_scaleFormerBatchSize(0);
    tilingData.set_scaleTailBatchSize(0);
    tilingData.set_ubProcessNum(static_cast<int64_t>(dimSize));
    tilingData.set_scaleUbProcessNum(0);
    tilingData.SaveToBuffer(tiling, tilingDataSize);

    auto kernel = embedding_dense_grad<EMBEDDING_DENSE_GRAD_SCH_MODE_SINGLE_ROW>;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, blockDim, grad, indices, y, workspace, tiling);

    auto yData = reinterpret_cast<float*>(y);
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(yData[i], expected[i]) << "mismatch at index " << i;
    }

    AscendC::GmFree(grad);
    AscendC::GmFree(indices);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// Scenario: SINGLE_ROW, fp32, paddingIdx skip
// grad = [[1.0, 2.0], [3.0, 4.0]]  (batchSize=2, dimSize=2)
// indices = [0, 1], numWeights=4, paddingIdx=1
// expected y = [[1.0, 2.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]  (index 1 skipped by paddingIdx)
TEST_F(EmbeddingDenseGradKernelTest, embedding_dense_grad_float32_padding_idx_smoke)
{
    constexpr size_t batchSize = 2;
    constexpr size_t dimSize = 2;
    constexpr size_t numWeights = 4;
    constexpr int32_t paddingIdx = 1;
    constexpr uint32_t blockDim = 1;

    const vector<float> gradData{1.0f, 2.0f, 3.0f, 4.0f};
    const vector<int32_t> indicesData{0, 1};
    const vector<float> expected{1.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    constexpr size_t kAlignElems = 8;
    constexpr size_t kFloatBytes = sizeof(float);
    size_t gradAlignElems = AlignUp(batchSize * dimSize, kAlignElems) + kAlignElems;
    size_t yAlignElems = AlignUp(numWeights * dimSize, kAlignElems) + kAlignElems;
    size_t gradByteSize = gradAlignElems * kFloatBytes;
    size_t indicesByteSize = AlignUp(batchSize * sizeof(int32_t), kBlockBytes);
    size_t yByteSize = yAlignElems * kFloatBytes;
    size_t align8NumWeights = (numWeights + 7) / 8 * 8;
    size_t userWorkspaceByteSize = 2 * 1024 + align8NumWeights * sizeof(int32_t) + 64 * sizeof(int32_t);
    size_t workspaceByteSize = AlignUp(4 * 1024 + userWorkspaceByteSize, kBlockBytes);
    size_t tilingDataSize = 1024;

    uint8_t* grad = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(gradByteSize));
    uint8_t* indices = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(indicesByteSize));
    uint8_t* y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(yByteSize));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(workspaceByteSize));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tilingDataSize));

    ASSERT_NE(grad, nullptr);
    ASSERT_NE(indices, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    memset(grad, 0, gradByteSize);
    memset(indices, 0, indicesByteSize);
    memset(y, 0, yByteSize);
    memset(workspace, 0, workspaceByteSize);
    memset(tiling, 0, tilingDataSize);

    CopyToGm(grad, gradData);
    CopyToGm(indices, indicesData);

    optiling::EmbeddingDenseGradTilingData tilingData;
    tilingData.set_dimSize(dimSize);
    tilingData.set_numWeights(static_cast<int32_t>(numWeights));
    tilingData.set_paddingIdx(paddingIdx);
    tilingData.set_scaleGradByFreq(0);
    tilingData.set_formerCoreNum(blockDim);
    tilingData.set_formerBatchSize(batchSize);
    tilingData.set_tailBatchSize(0);
    tilingData.set_scaleFormerCoreNum(0);
    tilingData.set_scaleFormerBatchSize(0);
    tilingData.set_scaleTailBatchSize(0);
    tilingData.set_ubProcessNum(static_cast<int64_t>(dimSize));
    tilingData.set_scaleUbProcessNum(0);
    tilingData.SaveToBuffer(tiling, tilingDataSize);

    auto kernel = embedding_dense_grad<EMBEDDING_DENSE_GRAD_SCH_MODE_SINGLE_ROW>;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, blockDim, grad, indices, y, workspace, tiling);

    auto yData = reinterpret_cast<float*>(y);
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(yData[i], expected[i]) << "mismatch at index " << i;
    }

    AscendC::GmFree(grad);
    AscendC::GmFree(indices);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
