/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "data_utils.h"
#include "string.h"
#include "tikicpulib.h"
#endif

#include "../../../op_kernel/masked_scatter.cpp"
#include "../../../op_kernel/masked_scatter_tiling_data.h"

using namespace std;

namespace {
constexpr size_t kBlockBytes = 32;

size_t AlignUp(size_t size, size_t align) { return ((size + align - 1) / align) * align; }
} // namespace

class MaskedScatterKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "MaskedScatterKernelTest SetUp" << endl; }

    static void TearDownTestCase() { cout << "MaskedScatterKernelTest TearDown" << endl; }
};

template <typename T>
void CopyToGm(uint8_t* gm, const vector<T>& data)
{
    auto ptr = reinterpret_cast<T*>(gm);
    for (size_t i = 0; i < data.size(); ++i) {
        ptr[i] = data[i];
    }
}

TEST_F(MaskedScatterKernelTest, masked_scatter_float32_smoke)
{
    constexpr size_t elemNum = 8;
    constexpr size_t updatesNum = 3;
    uint32_t blockDim = 1;
    const vector<float> expected{10.0f, 1.0f, 2.0f, 11.0f, 4.0f, 12.0f, 6.0f, 7.0f};

    size_t xByteSize = AlignUp(elemNum * sizeof(float), kBlockBytes);
    size_t maskByteSize = AlignUp(elemNum * sizeof(uint8_t), kBlockBytes);
    size_t updatesByteSize = AlignUp(updatesNum * sizeof(float), kBlockBytes);
    size_t yByteSize = AlignUp(elemNum * sizeof(float), kBlockBytes);
    size_t workspaceByteSize = 1024 * 1024;
    size_t tilingDataSize = sizeof(MaskedScatterTilingData);

    uint8_t* x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(xByteSize));
    uint8_t* mask = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(maskByteSize));
    uint8_t* updates = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(updatesByteSize));
    uint8_t* y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(yByteSize));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(workspaceByteSize));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(tilingDataSize));

    ASSERT_NE(x, nullptr);
    ASSERT_NE(mask, nullptr);
    ASSERT_NE(updates, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    memset(x, 0, xByteSize);
    memset(mask, 0, maskByteSize);
    memset(updates, 0, updatesByteSize);
    memset(y, 0, yByteSize);
    memset(workspace, 0, workspaceByteSize);
    memset(tiling, 0, tilingDataSize);

    CopyToGm(x, vector<float>{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f});
    CopyToGm(mask, vector<uint8_t>{1, 0, 0, 1, 0, 1, 0, 0});
    CopyToGm(updates, vector<float>{10.0f, 11.0f, 12.0f});

    auto tilingData = reinterpret_cast<MaskedScatterTilingData*>(tiling);
    tilingData->numElemX = elemNum;
    tilingData->numElemMask = elemNum;
    tilingData->numElemUpdates = updatesNum;
    tilingData->tilingCoreNum = blockDim;

    auto kernel = [](GM_ADDR x, GM_ADDR mask, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::masked_scatter<0>(x, mask, updates, y, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, blockDim, x, mask, updates, y, workspace, tiling);

    auto yData = reinterpret_cast<float*>(y);
    for (size_t i = 0; i < expected.size(); ++i) {
        ASSERT_FLOAT_EQ(yData[i], expected[i]) << "mismatch at index " << i;
    }

    AscendC::GmFree(x);
    AscendC::GmFree(mask);
    AscendC::GmFree(updates);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
