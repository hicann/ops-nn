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
 * \file test_turbo_quant_compress_latent.cpp
 * \brief
 */

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

extern "C" __global__ __aicore__ void turbo_quant_compress_latent(GM_ADDR latent, GM_ADDR centroids, GM_ADDR slot,
                                                                  GM_ADDR workspace, GM_ADDR tiling);

namespace {
constexpr uint32_t HEAD_DIM = 512;
constexpr uint32_t SLOT_SIZE = 320;
constexpr uint32_t COMPACT_SLOT_SIZE = 258;
constexpr uint32_t N_CENT = 16;
constexpr size_t WORKSPACE_SIZE = 32;
constexpr char DATA_DIR[] = "./turbo_quant_compress_latent_data";
} // namespace

class turbo_quant_compress_latent_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "turbo_quant_compress_latent_test SetUp\n" << std::endl; }
    static void TearDownTestCase() { std::cout << "turbo_quant_compress_latent_test TearDown\n" << std::endl; }
};

static void RunCase(uint32_t numTokens, uint32_t tokensPerCore, uint32_t tokensPerBatch, const char* dist = "gauss_lat",
                    uint32_t outputMode = 0)
{
    const uint32_t blockDim = numTokens == 0 ? 1 : (numTokens + tokensPerCore - 1) / tokensPerCore;

    system("cp -rf "
           "../../../../experimental/quant/turbo_quant_compress_latent/tests/ut/op_kernel/"
           "turbo_quant_compress_latent_data ./");
    system("chmod -R 755 ./turbo_quant_compress_latent_data/");
    std::string genCmd = std::string("cd ") + DATA_DIR + " && python3 gen_data.py " + std::to_string(numTokens) + " " +
                         dist + " " + std::to_string(outputMode);
    ASSERT_EQ(system(genCmd.c_str()), 0);

    size_t latentByteSize = static_cast<size_t>(numTokens) * HEAD_DIM * sizeof(float);
    size_t centByteSize = N_CENT * sizeof(float);
    const uint32_t outputSlotSize = outputMode == 1 ? COMPACT_SLOT_SIZE : SLOT_SIZE;
    size_t slotByteSize = static_cast<size_t>(numTokens) * outputSlotSize;

    // GmAlloc(0) is not meaningful; the empty-tensor case still needs valid pointers to hand the kernel
    uint8_t* latent = (uint8_t*)AscendC::GmAlloc(latentByteSize == 0 ? 1 : latentByteSize);
    uint8_t* centroids = (uint8_t*)AscendC::GmAlloc(centByteSize);
    uint8_t* slot = (uint8_t*)AscendC::GmAlloc(slotByteSize == 0 ? 1 : slotByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(WORKSPACE_SIZE);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(TurboQuantCompressLatentTilingData));

    if (latentByteSize > 0) {
        ReadFile(std::string(DATA_DIR) + "/input_latent.bin", latentByteSize, latent, latentByteSize);
    }
    ReadFile(std::string(DATA_DIR) + "/input_centroids.bin", centByteSize, centroids, centByteSize);

    TurboQuantCompressLatentTilingData* tilingData = reinterpret_cast<TurboQuantCompressLatentTilingData*>(tiling);
    tilingData->numTokens = numTokens;
    tilingData->tokensPerCore = tokensPerCore;
    tilingData->headDim = HEAD_DIM;
    tilingData->slotSize = outputSlotSize;
    tilingData->tokensPerBatch = tokensPerBatch;
    tilingData->outputMode = outputMode;

    ICPU_RUN_KF(turbo_quant_compress_latent, blockDim, latent, centroids, slot, workspace, (uint8_t*)(tilingData));

    WriteFile(std::string(DATA_DIR) + "/output_slot.bin", slot, slotByteSize == 0 ? 1 : slotByteSize);

    AscendC::GmFree((void*)latent);
    AscendC::GmFree((void*)centroids);
    AscendC::GmFree((void*)slot);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    std::string cmpCmd = std::string("cd ") + DATA_DIR + " && python3 compare_data.py " + std::to_string(numTokens) +
                         " " + std::to_string(outputMode);
    EXPECT_EQ(system(cmpCmd.c_str()), 0);
}

TEST_F(turbo_quant_compress_latent_test, single_token_single_core) { RunCase(1, 1, 1); }

// One token per core: exercises the tokStart_ == tokEnd_ - 1 path on every core.
TEST_F(turbo_quant_compress_latent_test, one_token_per_core) { RunCase(8, 1, 1); }

// Ragged split: the last core gets fewer tokens than tokensPerCore, so tokEnd_ is clamped.
TEST_F(turbo_quant_compress_latent_test, ragged_last_core) { RunCase(33, 8, 1); }

TEST_F(turbo_quant_compress_latent_test, multi_token_per_core) { RunCase(64, 16, 1); }

// Batched: every core runs whole batches of 4.
TEST_F(turbo_quant_compress_latent_test, batched_exact) { RunCase(64, 16, 4); }

// Two kinds of tail at once: 8 tokens per core in batches of 3 (3+3+2), and the last core owns 1 token.
TEST_F(turbo_quant_compress_latent_test, batched_ragged_tail) { RunCase(33, 8, 3); }

// Batch at the compile-time maximum.
TEST_F(turbo_quant_compress_latent_test, batched_max) { RunCase(24, 12, 12); }

// Compact corrected mode uses the same split/batch machinery but adds centroid-norm correction and
// writes a non-aligned 258-byte GM row. Cover each control-flow boundary independently.
TEST_F(turbo_quant_compress_latent_test, compact_single_token) { RunCase(1, 1, 1, "gauss_lat", 1); }

TEST_F(turbo_quant_compress_latent_test, compact_one_token_per_core) { RunCase(8, 1, 1, "gauss_lat", 1); }

TEST_F(turbo_quant_compress_latent_test, compact_ragged_last_core) { RunCase(33, 8, 1, "gauss_lat", 1); }

TEST_F(turbo_quant_compress_latent_test, compact_batched_ragged_tail) { RunCase(33, 8, 3, "gauss_lat", 1); }

TEST_F(turbo_quant_compress_latent_test, compact_batched_max) { RunCase(24, 12, 12, "gauss_lat", 1); }

TEST_F(turbo_quant_compress_latent_test, compact_empty_tensor) { RunCase(0, 1, 1, "gauss_lat", 1); }

TEST_F(turbo_quant_compress_latent_test, compact_uniform_wide_range) { RunCase(33, 8, 8, "uniform", 1); }

TEST_F(turbo_quant_compress_latent_test, compact_all_centroid_offsets) { RunCase(1, 1, 1, "all_centroids", 1); }

TEST_F(turbo_quant_compress_latent_test, compact_non_finite) { RunCase(9, 4, 4, "mixinf", 1); }

// Batch larger than the tokens a core owns: the count clamps on the very first batch.
TEST_F(turbo_quant_compress_latent_test, batch_larger_than_tokens) { RunCase(5, 5, 12); }

// Empty tensor: tiling still hands out one block, the kernel must write nothing and not fault.
TEST_F(turbo_quant_compress_latent_test, empty_tensor) { RunCase(0, 1, 1); }

// Value-range coverage. The codebook was fitted to N(0, 1/headDim), so anything wider saturates into the
// outermost buckets after normalization; these two make sure that path is exercised rather than assumed.
TEST_F(turbo_quant_compress_latent_test, uniform_wide_range) { RunCase(33, 8, 8, "uniform"); }

TEST_F(turbo_quant_compress_latent_test, gaussian_shifted_wide) { RunCase(33, 8, 8, "gauss_wide"); }

// Non-finite inputs. NaN makes every comparison false, so the nibbles collapse to 0; +-INF makes the L2
// norm infinite, so 1/norm is 0 and every finite lane normalizes to 0 while the infinite lane becomes NaN.
// The stored fp16 norm carries the NaN/INF through.
TEST_F(turbo_quant_compress_latent_test, input_nan) { RunCase(9, 4, 4, "nan"); }

TEST_F(turbo_quant_compress_latent_test, input_pos_inf) { RunCase(9, 4, 4, "posinf"); }

TEST_F(turbo_quant_compress_latent_test, input_neg_inf) { RunCase(9, 4, 4, "neginf"); }

TEST_F(turbo_quant_compress_latent_test, input_mixed_non_finite) { RunCase(9, 4, 4, "mixinf"); }
