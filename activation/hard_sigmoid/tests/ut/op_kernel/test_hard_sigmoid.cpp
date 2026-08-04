/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <memory>
#include "data_utils.h"
#include "kernel_ut_data_executor.h"
#include "kernel_ut_data_helper.h"
#include "tikicpulib.h"

#include "../../../op_kernel/arch35/hard_sigmoid.cpp"

namespace {
constexpr int64_t UB_ELEMENT_COUNT = 1024;

struct GmDeleter {
    void operator()(uint8_t* ptr) const
    {
        if (ptr != nullptr) {
            AscendC::GmFree(ptr);
        }
    }
};

using GmBuffer = std::unique_ptr<uint8_t, GmDeleter>;

void RunHardSigmoidFloat(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    const auto* tilingData = reinterpret_cast<const HardSigmoidTilingData*>(tiling);
    HardSigmoidKernel<float> kernel;
    kernel.Init(input, output, tilingData);
    kernel.Process();
}

void InitTiling(HardSigmoidTilingData* tilingData, int64_t totalElements, int64_t blockFactor, float alpha, float beta)
{
    tilingData->totalElements = totalElements;
    tilingData->blockFactor = blockFactor;
    tilingData->ubFactor = UB_ELEMENT_COUNT;
    tilingData->alpha = alpha;
    tilingData->beta = beta;
}

void RunKernelCase(int64_t elementCount, int64_t blockFactor, uint32_t blockDim, float alpha, float beta)
{
    size_t tensorBytes = elementCount * sizeof(float);
    GmBuffer x(static_cast<uint8_t*>(AscendC::GmAlloc(tensorBytes)));
    GmBuffer y(static_cast<uint8_t*>(AscendC::GmAlloc(tensorBytes)));
    GmBuffer workspace(static_cast<uint8_t*>(AscendC::GmAlloc(16 * 1024 * 1024)));
    GmBuffer tiling(static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(HardSigmoidTilingData))));
    ASSERT_NE(x.get(), nullptr);
    ASSERT_NE(y.get(), nullptr);
    ASSERT_NE(workspace.get(), nullptr);
    ASSERT_NE(tiling.get(), nullptr);

    kernel_ut::SetupTestEnvironment("activation/hard_sigmoid/tests/ut/op_kernel/hard_sigmoid_data",
                                    "hard_sigmoid_data");
    kernel_ut::RunGenData("./hard_sigmoid_data",
                          {"'(" + std::to_string(elementCount) + ")'", std::to_string(alpha), std::to_string(beta)});
    const std::string path = kernel_ut::GetTestWorkDir();
    ReadFile(path + "/hard_sigmoid_data/input_x.bin", tensorBytes, x.get(), tensorBytes);

    auto* tilingData = reinterpret_cast<HardSigmoidTilingData*>(tiling.get());
    InitTiling(tilingData, elementCount, blockFactor, alpha, beta);
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(RunHardSigmoidFloat, blockDim, x.get(), y.get(), workspace.get(), tiling.get());
    WriteFile(path + "/hard_sigmoid_data/output.bin", y.get(), tensorBytes);

    ASSERT_TRUE(kernel_ut::RunCompareData("./hard_sigmoid_data", {}));
}

void RunEmptyKernelCase()
{
    constexpr size_t bufferBytes = 32;
    GmBuffer x(static_cast<uint8_t*>(AscendC::GmAlloc(bufferBytes)));
    GmBuffer y(static_cast<uint8_t*>(AscendC::GmAlloc(bufferBytes)));
    GmBuffer workspace(static_cast<uint8_t*>(AscendC::GmAlloc(bufferBytes)));
    GmBuffer tiling(static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(HardSigmoidTilingData))));
    ASSERT_NE(x.get(), nullptr);
    ASSERT_NE(y.get(), nullptr);
    ASSERT_NE(workspace.get(), nullptr);
    ASSERT_NE(tiling.get(), nullptr);

    auto* tilingData = reinterpret_cast<HardSigmoidTilingData*>(tiling.get());
    *tilingData = HardSigmoidTilingData{};
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(RunHardSigmoidFloat, 1, x.get(), y.get(), workspace.get(), tiling.get());
}
} // namespace

class HardSigmoidKernelTest : public testing::Test {
protected:
    static void TearDownTestCase() { kernel_ut::CleanGeneratedBinFiles("./hard_sigmoid_data"); }
};

TEST_F(HardSigmoidKernelTest, DefaultAttributes) { RunKernelCase(256, 256, 1, 1.0f / 6.0f, 0.5f); }

TEST_F(HardSigmoidKernelTest, CustomAttributes) { RunKernelCase(256, 256, 1, 0.2f, 0.4f); }

TEST_F(HardSigmoidKernelTest, MultiCoreTailBlock) { RunKernelCase(257, 129, 2, 1.0f / 6.0f, 0.5f); }

TEST_F(HardSigmoidKernelTest, EmptyTensor) { RunEmptyKernelCase(); }
