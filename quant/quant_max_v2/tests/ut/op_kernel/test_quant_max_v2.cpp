/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include "gtest/gtest.h"
#include "tikicpulib.h"

// 与 op_kernel/arch35/quant_max_v2_tiling_data.h 的 QuantMaxV2TilingData 保持同构。
struct TestTilingData {
    int64_t totalCoreNum;
    int64_t actualCoreNum;
    int64_t blockTailCoreNum;
    int64_t blockFactor;
    int64_t blockTailFactor;
    int64_t ubFactor;
    int64_t dim1;
    int64_t numGroups;
    uint32_t coreDivReciprocal;
    uint32_t tileDivReciprocal;
    float dstTypeMax;
    int64_t tensorBlockLoopNum;
    int64_t tensorBlockLoopTail;
    int64_t tensorBlockTileCount;
    int64_t tensorTailLoopNum;
    int64_t tensorTailLoopTail;
    int64_t tensorTailTileCount;
    int64_t tensorTailCoreStart;
    int64_t tensorBf16TileMaxBufferSize;
};

// TilingKey 模板分量取值（与 quant_max_v2_struct.h 的宏一致）：
// roundMode(2bit): 0=rint 1=round 2=hybrid；quantMode(1bit): 0=static 1=dynamic；
// hasGroup(1bit): 0=tensor 1=group。
constexpr uint64_t MakeTilingKey(uint64_t roundMode, uint64_t quantMode, uint64_t hasGroup)
{
    return (roundMode << 2) | (quantMode << 1) | hasGroup;
}

// 直接包含 kernel 源（仓内 template 入口 UT 范式，参考 celu_v2）：模板与本测试同 TU 编译，
// lambda 内显式实例化调用（roundMode×quantMode×hasGroup 三模板参）。
#include "../../../op_kernel/quant_max_v2.cpp"

namespace {

class quant_max_v2_test : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

template <typename T>
void GenerateInputData(void* buffer, size_t count, float minValue, float maxValue)
{
    T* data = static_cast<T*>(buffer);
    for (size_t i = 0; i < count; i++) {
        const float val = minValue + (maxValue - minValue) * (static_cast<float>(i % 97) / 97.0f);
        data[i] = static_cast<T>(val);
    }
}

// 动态单张量（fp32 → E5M2, rint）：amax 与宿主侧逐元素 |max| 一致。
TEST_F(quant_max_v2_test, dynamic_tensor_fp32_amax)
{
    constexpr int64_t rows = 256;
    constexpr int64_t cols = 128;
    constexpr int64_t count = rows * cols;
    constexpr uint32_t blockDim = 2;

    const size_t xSize = count * sizeof(float);
    const size_t ySize = count;
    const size_t amaxSize = sizeof(float);
    const size_t yScaleSize = sizeof(float);
    const size_t workspaceSize = 16 * 1024 * 1024;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* amax = (uint8_t*)AscendC::GmAlloc(amaxSize);
    uint8_t* yScale = (uint8_t*)AscendC::GmAlloc(yScaleSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(TestTilingData));

    GenerateInputData<float>(x, count, -2.0f, 2.0f);

    const int64_t blockFactor = count / blockDim;
    auto* td = reinterpret_cast<TestTilingData*>(tiling);
    td->totalCoreNum = blockDim;
    td->actualCoreNum = blockDim;
    td->blockTailCoreNum = 0;
    td->blockFactor = blockFactor;
    td->blockTailFactor = blockFactor;
    td->ubFactor = 4096;
    td->dim1 = cols;
    td->numGroups = 1;
    td->coreDivReciprocal = 0;
    td->tileDivReciprocal = 0;
    td->dstTypeMax = 57344.0f;
    td->tensorBlockLoopNum = blockFactor / 4096;
    td->tensorBlockLoopTail = blockFactor % 4096;
    td->tensorBlockTileCount = td->tensorBlockLoopNum + (td->tensorBlockLoopTail != 0);
    td->tensorTailLoopNum = 0;
    td->tensorTailLoopTail = 0;
    td->tensorTailTileCount = 0;
    td->tensorTailCoreStart = 0;
    td->tensorBf16TileMaxBufferSize = 0;

    ICPU_SET_TILING_KEY(MakeTilingKey(0 /*rint*/, 1 /*dynamic*/, 0 /*tensor*/));
    auto kernel = [](GM_ADDR x, GM_ADDR scale, GM_ADDR groupList, GM_ADDR y, GM_ADDR amax, GM_ADDR yScale,
                     GM_ADDR workspace, GM_ADDR tiling) {
        quant_max_v2<0 /*rint*/, 1 /*dynamic*/, 0 /*tensor*/>(x, scale, groupList, y, amax, yScale, workspace, tiling);
    };
    ICPU_RUN_KF(kernel, blockDim, x, nullptr, nullptr, y, amax, yScale, workspace, tiling);

    float hostAmax = 0.0f;
    const float* xData = reinterpret_cast<const float*>(x);
    for (int64_t i = 0; i < count; ++i) {
        hostAmax = std::max(hostAmax, std::fabs(xData[i]));
    }
    EXPECT_NEAR(*reinterpret_cast<float*>(amax), hostAmax, 0.01f);
    // y_scale = amax / dstTypeMax（dstTypeMax 在 tiling 内被覆盖为 E5M2 类型最大值 57344）。
    EXPECT_NEAR(*reinterpret_cast<float*>(yScale), hostAmax / 57344.0f, 1e-6f);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(amax);
    AscendC::GmFree(yScale);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// 静态单张量（fp32 → E5M2, rint）：scale 乘性量化，amax 仍逐元素最大。
TEST_F(quant_max_v2_test, static_tensor_fp32_amax)
{
    constexpr int64_t rows = 128;
    constexpr int64_t cols = 128;
    constexpr int64_t count = rows * cols;
    constexpr uint32_t blockDim = 2;

    const size_t xSize = count * sizeof(float);
    const size_t ySize = count;
    const size_t amaxSize = sizeof(float);
    const size_t scaleSize = sizeof(float);
    const size_t workspaceSize = 16 * 1024 * 1024;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* amax = (uint8_t*)AscendC::GmAlloc(amaxSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(TestTilingData));

    GenerateInputData<float>(x, count, -1.5f, 1.5f);
    const float scaleVal = 448.0f; // 量化系数（乘性）：x*448 → E5M2 满量程附近
    std::memcpy(scale, &scaleVal, scaleSize);

    const int64_t blockFactor = count / blockDim;
    auto* td = reinterpret_cast<TestTilingData*>(tiling);
    td->totalCoreNum = blockDim;
    td->actualCoreNum = blockDim;
    td->blockTailCoreNum = 0;
    td->blockFactor = blockFactor;
    td->blockTailFactor = blockFactor;
    td->ubFactor = 4096;
    td->dim1 = cols;
    td->numGroups = 1;
    td->coreDivReciprocal = 0;
    td->tileDivReciprocal = 0;
    td->dstTypeMax = 448.0f;
    td->tensorBlockLoopNum = blockFactor / 4096;
    td->tensorBlockLoopTail = blockFactor % 4096;
    td->tensorBlockTileCount = td->tensorBlockLoopNum + (td->tensorBlockLoopTail != 0);
    td->tensorTailLoopNum = 0;
    td->tensorTailLoopTail = 0;
    td->tensorTailTileCount = 0;
    td->tensorTailCoreStart = 0;
    td->tensorBf16TileMaxBufferSize = 0;

    ICPU_SET_TILING_KEY(MakeTilingKey(0 /*rint*/, 0 /*static*/, 0 /*tensor*/));
    auto kernel = [](GM_ADDR x, GM_ADDR scale, GM_ADDR groupList, GM_ADDR y, GM_ADDR amax, GM_ADDR yScale,
                     GM_ADDR workspace, GM_ADDR tiling) {
        quant_max_v2<0 /*rint*/, 0 /*static*/, 0 /*tensor*/>(x, scale, groupList, y, amax, yScale, workspace, tiling);
    };
    ICPU_RUN_KF(kernel, blockDim, x, scale, nullptr, y, amax, nullptr, workspace, tiling);

    float hostAmax = 0.0f;
    const float* xData = reinterpret_cast<const float*>(x);
    for (int64_t i = 0; i < count; ++i) {
        hostAmax = std::max(hostAmax, std::fabs(xData[i]));
    }
    EXPECT_NEAR(*reinterpret_cast<float*>(amax), hostAmax, 0.01f);

    AscendC::GmFree(x);
    AscendC::GmFree(scale);
    AscendC::GmFree(y);
    AscendC::GmFree(amax);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

} // namespace
