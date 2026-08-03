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
#include <cstring>
#include <iostream>
#include <string>

#include "data_utils.h"
#include "fused_patch_mlp_tiling_def.h"
#include "gtest/gtest.h"
#include "kernel_ut_data_executor.h"
#include "kernel_ut_data_helper.h"
#include "tikicpulib.h"
#include "../../../op_kernel/fused_patch_mlp.h"

#define FUSED_PATCH_MLP_UT_IMPL(T, USE_MDL, PIPELINE_GELU)                \
    do {                                                                  \
        FusedPatchMlp::KernelFusedPatchMlp<T, USE_MDL, PIPELINE_GELU> op; \
        op.Init(x, weights, biases, y, workspace, &tilingData);           \
        op.Process();                                                     \
    } while (0)

// The production entry is a MIX kernel. TmSim executes its AIC and AIV tasks as separate serial processes, which
// cannot model the per-layer Matmul -> GELU dependency used by the __CCE_KT_TEST__ implementation. Keep that
// production metadata out of the CPU UT and exercise the same kernel classes through a single-core AIC wrapper.
#define DEFINE_FUSED_PATCH_MLP_UT_KERNEL(NAME, T)                                                     \
    extern "C" __global__ __aicore__ void NAME(GM_ADDR x, GM_ADDR weights, GM_ADDR biases, GM_ADDR y, \
                                               GM_ADDR workspace, GM_ADDR tiling)                     \
    {                                                                                                 \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);                                               \
        if ASCEND_IS_AIV {                                                                            \
            return;                                                                                   \
        }                                                                                             \
        GET_TILING_DATA(tilingData, tiling);                                                          \
        if (TILING_KEY_IS(1)) {                                                                       \
            FUSED_PATCH_MLP_UT_IMPL(T, false, false);                                                 \
        } else if (TILING_KEY_IS(21)) {                                                               \
            FUSED_PATCH_MLP_UT_IMPL(T, false, false);                                                 \
        } else if (TILING_KEY_IS(11)) {                                                               \
            FUSED_PATCH_MLP_UT_IMPL(T, true, false);                                                  \
        } else if (TILING_KEY_IS(31)) {                                                               \
            FUSED_PATCH_MLP_UT_IMPL(T, true, true);                                                   \
        }                                                                                             \
    }

DEFINE_FUSED_PATCH_MLP_UT_KERNEL(fused_patch_mlp_fp16_test_kernel, half)
DEFINE_FUSED_PATCH_MLP_UT_KERNEL(fused_patch_mlp_bf16_test_kernel, bfloat16_t)
DEFINE_FUSED_PATCH_MLP_UT_KERNEL(fused_patch_mlp_fp32_test_kernel, float)

#undef DEFINE_FUSED_PATCH_MLP_UT_KERNEL
#undef FUSED_PATCH_MLP_UT_IMPL

namespace {

constexpr uint64_t SYS_WORKSPACE = 16UL * 1024UL * 1024UL;

std::string GetSourceDataDirectory()
{
    std::string sourcePath = __FILE__;
    for (char& ch : sourcePath) {
        if (ch == '\\') {
            ch = '/';
        }
    }
    const size_t repoRelativeBegin = sourcePath.find("experimental/");
    const size_t fileNameBegin = sourcePath.rfind('/');
    if (repoRelativeBegin == std::string::npos || fileNameBegin == std::string::npos ||
        fileNameBegin <= repoRelativeBegin) {
        return "experimental/matmul/fused_patch_mlp/tests/ut/op_kernel/fused_patch_mlp_data";
    }
    return sourcePath.substr(repoRelativeBegin, fileNameBegin - repoRelativeBegin) + "/fused_patch_mlp_data";
}

void FillCubeTiling(TCubeTiling& tiling, uint32_t m, uint32_t n, uint32_t k, uint32_t elementSize)
{
    constexpr uint32_t BLOCK = 16;
    auto alignUp = [](uint32_t value, uint32_t alignment) { return (value + alignment - 1U) / alignment * alignment; };

    memset(&tiling, 0, sizeof(TCubeTiling));
    tiling.usedCoreNum = 1;
    tiling.M = m;
    tiling.N = n;
    tiling.Ka = k;
    tiling.Kb = k;
    tiling.singleCoreM = m;
    tiling.singleCoreN = n;
    tiling.singleCoreK = k;
    tiling.baseM = alignUp(m, BLOCK);
    tiling.baseN = alignUp(n, BLOCK);
    tiling.baseK = alignUp(k, BLOCK);
    tiling.depthA1 = 1;
    tiling.depthB1 = 1;
    tiling.stepM = 1;
    tiling.stepN = 1;
    tiling.stepKa = 1;
    tiling.stepKb = 1;
    tiling.isBias = 1;
    tiling.transLength = 0;
    tiling.iterateOrder = 0;
    tiling.shareMode = 0;
    tiling.shareL1Size = 6144 * (elementSize / 2U);
    tiling.shareL0CSize = 2048;
    tiling.shareUbSize = 0;
    tiling.batchM = 1;
    tiling.batchN = 1;
    tiling.singleBatchM = 1;
    tiling.singleBatchN = 1;
    tiling.dbL0A = 2;
    tiling.dbL0B = 2;
    tiling.dbL0C = 1;
    tiling.BatchNum = 0;
}

void GenerateTiling(uint32_t m, uint32_t hidden, uint32_t patch, uint32_t layers, uint32_t elementSize,
                    uint32_t geluMode, uint32_t geluTileSize, FusedPatchMlpTilingData& tiling)
{
    tiling.totalN = m;
    tiling.inFeatures = patch;
    tiling.hiddenSize = hidden;
    tiling.geluTileSize = geluTileSize;
    tiling.geluMode = geluMode;
    tiling.numLayers = layers;
    FillCubeTiling(tiling.mm0Tiling, m, hidden, patch, elementSize);
    FillCubeTiling(tiling.mmHTiling, m, hidden, layers >= 2 ? hidden : patch, elementSize);
}

class FusedPatchMlpKernelTest : public testing::Test {
protected:
    static void TearDownTestCase() { kernel_ut::CleanGeneratedBinFiles("./fused_patch_mlp_data"); }
};

void RunOneCase(const std::string& dtype, uint64_t tilingKey, uint32_t elementSize, uint32_t geluMode,
                uint32_t geluTileSize)
{
    const std::string directory = "./fused_patch_mlp_data";
    ASSERT_TRUE(kernel_ut::SetupTestEnvironment(GetSourceDataDirectory(), "fused_patch_mlp_data"));
    ASSERT_TRUE(kernel_ut::RunGenData(directory, {"2,4,16", "64", "3", dtype}));

    constexpr uint32_t m = 8;
    constexpr uint32_t patch = 16;
    constexpr uint32_t hidden = 64;
    constexpr uint32_t layers = 3;
    const size_t totalWeights = static_cast<size_t>(patch) * hidden +
                                static_cast<size_t>(layers - 1U) * hidden * hidden;
    const uint32_t biasElementSize = dtype == "bfloat16" ? 4U : elementSize;
    const size_t xSize = static_cast<size_t>(m) * patch * elementSize;
    const size_t weightsSize = totalWeights * elementSize;
    const size_t biasesSize = static_cast<size_t>(layers) * hidden * biasElementSize;
    const size_t ySize = static_cast<size_t>(m) * hidden * elementSize;
    const size_t workspaceSize = SYS_WORKSPACE + 2UL * m * hidden * elementSize;

    auto* x = static_cast<uint8_t*>(AscendC::GmAlloc(xSize));
    auto* weights = static_cast<uint8_t*>(AscendC::GmAlloc(weightsSize));
    auto* biases = static_cast<uint8_t*>(AscendC::GmAlloc(biasesSize));
    auto* y = static_cast<uint8_t*>(AscendC::GmAlloc(ySize));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(workspaceSize));
    auto* rawTiling = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(FusedPatchMlpTilingData)));

    size_t fileSize = 0;
    ASSERT_TRUE(ReadFile(directory + "/" + dtype + "_x_fused_patch_mlp.bin", fileSize, x, xSize));
    ASSERT_EQ(fileSize, xSize);
    ASSERT_TRUE(ReadFile(directory + "/" + dtype + "_weights_fused_patch_mlp.bin", fileSize, weights, weightsSize));
    ASSERT_EQ(fileSize, weightsSize);
    ASSERT_TRUE(ReadFile(directory + "/" + dtype + "_biases_fused_patch_mlp.bin", fileSize, biases, biasesSize));
    ASSERT_EQ(fileSize, biasesSize);

    auto* tiling = reinterpret_cast<FusedPatchMlpTilingData*>(rawTiling);
    GenerateTiling(m, hidden, patch, layers, elementSize, geluMode, geluTileSize, *tiling);
    ICPU_SET_TILING_KEY(tilingKey);
    if (dtype == "float16") {
        ICPU_RUN_KF(fused_patch_mlp_fp16_test_kernel, 1, x, weights, biases, y, workspace, rawTiling);
    } else if (dtype == "bfloat16") {
        ICPU_RUN_KF(fused_patch_mlp_bf16_test_kernel, 1, x, weights, biases, y, workspace, rawTiling);
    } else {
        ICPU_RUN_KF(fused_patch_mlp_fp32_test_kernel, 1, x, weights, biases, y, workspace, rawTiling);
    }
    WriteFile(directory + "/" + dtype + "_output_fused_patch_mlp.bin", y, ySize);

    AscendC::GmFree(x);
    AscendC::GmFree(weights);
    AscendC::GmFree(biases);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(rawTiling);
    kernel_ut::RunCompareData(directory, {"'" + dtype + "'"});
}

TEST_F(FusedPatchMlpKernelTest, Float16RowWise) { RunOneCase("float16", 1, 2, 0, 64); }

TEST_F(FusedPatchMlpKernelTest, Bfloat16RowWise) { RunOneCase("bfloat16", 1, 2, 0, 64); }

TEST_F(FusedPatchMlpKernelTest, Float32RowWise) { RunOneCase("float32", 1, 4, 0, 64); }

TEST_F(FusedPatchMlpKernelTest, Float16FlatTiled) { RunOneCase("float16", 1, 2, 1, 128); }

TEST_F(FusedPatchMlpKernelTest, Float16MdlBaseBlock)
{
    // Exercise the externally scheduled MDL branch with a single simulator core. Production selects this key
    // only for large hidden sizes; the compact shape keeps the kernel UT fast while covering the same control flow.
    RunOneCase("float16", 11, 2, 1, 128);
}

} // namespace
