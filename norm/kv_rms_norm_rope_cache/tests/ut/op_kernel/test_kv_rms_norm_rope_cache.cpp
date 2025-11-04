/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_kv_rms_norm_rope_cache.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "kv_rms_norm_rope_cache_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void kv_rms_norm_rope_cache(
    GM_ADDR kv, GM_ADDR gamma, GM_ADDR cos, GM_ADDR sin, GM_ADDR index, GM_ADDR k_cache, GM_ADDR ckv_cache,
    GM_ADDR k_rope_scale, GM_ADDR c_kv_scale, GM_ADDR k_rope_offset, GM_ADDR c_kv_offset, GM_ADDR k_cache_out,
    GM_ADDR c_kv_offset_out, GM_ADDR k_rope, GM_ADDR c_kv, GM_ADDR workspace, GM_ADDR tiling);

class kv_rms_norm_rope_cache_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "kv_rms_norm_rope_cache_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "kv_rms_norm_rope_cache_test TearDown\n" << endl;
    }
};

TEST_F(kv_rms_norm_rope_cache_test, test_case_0000)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    size_t kvSize = 64 * 1 * 1 * 576 * sizeof(half);
    size_t gammaSize = 512 * sizeof(half);
    size_t indexSize = 64 * sizeof(uint64_t);
    size_t cosSize = 64 * 1 * 1 * 64 * sizeof(half);
    size_t sinSize = 64 * 1 * 1 * 64 * sizeof(half);
    size_t kCacheSize = 576 * 128 * 1 * 64 * sizeof(half);
    size_t ckvCacheSize = 576 * 128 * 1 * 512 * sizeof(half);
    size_t kRopeScaleSize = 64 * sizeof(float);
    size_t ckvScaleSize = 512 * sizeof(float);
    size_t kRopeSize = 64 * 1 * 1 * 64 * sizeof(half);
    size_t ckvSize = 64 * 1 * 1 * 512 * sizeof(half);
    size_t tiling_data_size = sizeof(KvRmsNormRopeCacheTilingData);
    uint32_t blockDim = 32;

    uint8_t* kv = (uint8_t*)AscendC::GmAlloc(kvSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaSize);
    uint8_t* index = (uint8_t*)AscendC::GmAlloc(indexSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(sinSize);
    uint8_t* kCache = (uint8_t*)AscendC::GmAlloc(kCacheSize);
    uint8_t* ckvCache = (uint8_t*)AscendC::GmAlloc(ckvCacheSize);
    uint8_t* kRopeScale = (uint8_t*)AscendC::GmAlloc(kRopeScaleSize);
    uint8_t* ckvScale = (uint8_t*)AscendC::GmAlloc(ckvScaleSize);
    uint8_t* kRope = (uint8_t*)AscendC::GmAlloc(kRopeSize);
    uint8_t* ckv = (uint8_t*)AscendC::GmAlloc(ckvSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    KvRmsNormRopeCacheTilingData* tilingDatafromBin = reinterpret_cast<KvRmsNormRopeCacheTilingData*>(tiling);

    tilingDatafromBin->blockDim = 32;
    tilingDatafromBin->rowsPerBlock = 0;
    tilingDatafromBin->cacheLength = 1;
    tilingDatafromBin->batchSize = 64;
    tilingDatafromBin->numHead = 1;
    tilingDatafromBin->seqLength = 1;
    tilingDatafromBin->blockSize = 128;
    tilingDatafromBin->blockFactor = 2;
    tilingDatafromBin->ubFactor = 16;
    tilingDatafromBin->epsilon = 1e-05;
    tilingDatafromBin->reciprocal = 1.0f / 512.0f;
    tilingDatafromBin->isOutputKv = true;
    tilingDatafromBin->isKQuant = 1;
    tilingDatafromBin->isVQuant = 1;

#define DTYPE_KV half
#define DTYPE_K_CACHE int8_t
#define DTYPE_CKV_CACHE int8_t

    ICPU_SET_TILING_KEY(5011);
    ICPU_RUN_KF(
        kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, kRopeScale, ckvScale,
        nullptr, nullptr, kCache, ckvCache, kRope, ckv, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(kv);
    AscendC::GmFree(gamma);
    AscendC::GmFree(index);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(kCache);
    AscendC::GmFree(ckvCache);
    AscendC::GmFree(kRopeScale);
    AscendC::GmFree(ckvScale);
    AscendC::GmFree(kRope);
    AscendC::GmFree(ckv);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}


TEST_F(kv_rms_norm_rope_cache_test, test_case_5010)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    size_t kvSize = 64 * 1 * 1 * 576 * sizeof(half);
    size_t gammaSize = 512 * sizeof(half);
    size_t indexSize = 64 * sizeof(uint64_t);
    size_t cosSize = 64 * 1 * 1 * 64 * sizeof(half);
    size_t sinSize = 64 * 1 * 1 * 64 * sizeof(half);
    size_t kCacheSize = 576 * 128 * 1 * 64 * sizeof(half);
    size_t ckvCacheSize = 576 * 128 * 1 * 512 * sizeof(half);
    size_t kRopeScaleSize = 64 * sizeof(float);
    size_t ckvScaleSize = 512 * sizeof(float);
    size_t kRopeSize = 64 * 1 * 1 * 64 * sizeof(half);
    size_t ckvSize = 64 * 1 * 1 * 512 * sizeof(half);
    size_t tiling_data_size = sizeof(KvRmsNormRopeCacheTilingData);
    uint32_t blockDim = 32;

    uint8_t* kv = (uint8_t*)AscendC::GmAlloc(kvSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaSize);
    uint8_t* index = (uint8_t*)AscendC::GmAlloc(indexSize);
    uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosSize);
    uint8_t* sin = (uint8_t*)AscendC::GmAlloc(sinSize);
    uint8_t* kCache = (uint8_t*)AscendC::GmAlloc(kCacheSize);
    uint8_t* ckvCache = (uint8_t*)AscendC::GmAlloc(ckvCacheSize);
    uint8_t* kRopeScale = (uint8_t*)AscendC::GmAlloc(kRopeScaleSize);
    uint8_t* ckvScale = (uint8_t*)AscendC::GmAlloc(ckvScaleSize);
    uint8_t* kRope = (uint8_t*)AscendC::GmAlloc(kRopeSize);
    uint8_t* ckv = (uint8_t*)AscendC::GmAlloc(ckvSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    KvRmsNormRopeCacheTilingData* tilingDatafromBin = reinterpret_cast<KvRmsNormRopeCacheTilingData*>(tiling);

    tilingDatafromBin->blockDim = 32;
    tilingDatafromBin->rowsPerBlock = 0;
    tilingDatafromBin->cacheLength = 1;
    tilingDatafromBin->batchSize = 64;
    tilingDatafromBin->numHead = 1;
    tilingDatafromBin->seqLength = 1;
    tilingDatafromBin->blockSize = 128;
    tilingDatafromBin->blockFactor = 2;
    tilingDatafromBin->ubFactor = 16;
    tilingDatafromBin->epsilon = 1e-05;
    tilingDatafromBin->reciprocal = 1.0f / 512.0f;
    tilingDatafromBin->isOutputKv = false;
    tilingDatafromBin->isKQuant = 1;
    tilingDatafromBin->isVQuant = 1;

#define DTYPE_KV half
#define DTYPE_K_CACHE int8_t
#define DTYPE_CKV_CACHE int8_t
    
    ICPU_SET_TILING_KEY(5010);
    ICPU_RUN_KF(
        kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, kRopeScale, ckvScale,
        nullptr, nullptr, kCache, ckvCache, kRope, ckv, workspace, (uint8_t*)(tilingDatafromBin));
    

    AscendC::GmFree(kv);
    AscendC::GmFree(gamma);
    AscendC::GmFree(index);
    AscendC::GmFree(cos);
    AscendC::GmFree(sin);
    AscendC::GmFree(kCache);
    AscendC::GmFree(ckvCache);
    AscendC::GmFree(kRopeScale);
    AscendC::GmFree(ckvScale);
    AscendC::GmFree(kRope);
    AscendC::GmFree(ckv);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}



// TEST_F(kv_rms_norm_rope_cache_test, test_case_0001)
// {
//     size_t kvSize = 64 * 1 * 1 * 576 * sizeof(half);
//     size_t gammaSize = 512 * sizeof(half);
//     size_t indexSize = 64 * 1 * sizeof(uint64_t);
//     size_t cosSize = 64 * 1 * 1 * 64 * sizeof(half);
//     size_t sinSize = 64 * 1 * 1 * 64 * sizeof(half);
//     size_t kCacheSize = 64 * 128 * 1 * 576 * sizeof(half);
//     size_t ckvCacheSize = 64 * 128 * 1 * 576 * sizeof(half);
//     // size_t kRopeScaleSize = 64 * sizeof(float);
//     // size_t ckvScaleSize = 512 * sizeof(float);
//     // size_t kRopeOffsetSize = 64 * sizeof(float);
//     // size_t ckvOffsetSize = 512 * sizeof(float);
//     size_t kRopeSize = 64 * 1 * 1 * 64 * sizeof(half);
//     size_t ckvSize = 64 * 1 * 1 * 512 * sizeof(half);
//     size_t tiling_data_size = sizeof(KvRmsNormRopeCacheTilingData);
//     uint32_t blockDim = 16;

//     uint8_t* kv = (uint8_t*)AscendC::GmAlloc(kvSize);
//     uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaSize);
//     uint8_t* index = (uint8_t*)AscendC::GmAlloc(indexSize);
//     uint8_t* cos = (uint8_t*)AscendC::GmAlloc(cosSize);
//     uint8_t* sin = (uint8_t*)AscendC::GmAlloc(sinSize);
//     uint8_t* kCache = (uint8_t*)AscendC::GmAlloc(kCacheSize);
//     uint8_t* ckvCache = (uint8_t*)AscendC::GmAlloc(ckvCacheSize);
//     // uint8_t* kRopeScale = (uint8_t*)AscendC::GmAlloc(kRopeScaleSize);
//     // uint8_t* ckvScale = (uint8_t*)AscendC::GmAlloc(ckvScaleSize);
//     // uint8_t* kRopeOffset = (uint8_t*)AscendC::GmAlloc(kRopeOffsetSize);
//     // uint8_t* ckvOffset = (uint8_t*)AscendC::GmAlloc(ckvOffsetSize);
//     uint8_t* kRope = (uint8_t*)AscendC::GmAlloc(kRopeSize);
//     uint8_t* ckv = (uint8_t*)AscendC::GmAlloc(ckvSize);
//     uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

//     char* path_ = get_current_dir_name();
//     string path(path_);

//     KvRmsNormRopeCacheTilingData* tilingDatafromBin = reinterpret_cast<KvRmsNormRopeCacheTilingData*>(tiling);

//     tilingDatafromBin->blockDim = 16;
//     tilingDatafromBin->rowsPerBlock = 4;
//     tilingDatafromBin->cacheLength = 1;
//     tilingDatafromBin->batchSize = 64;
//     tilingDatafromBin->numHead = 1;
//     tilingDatafromBin->seqLength = 1;
//     tilingDatafromBin->blockSize = 128;
//     tilingDatafromBin->blockFactor = 0;
//     tilingDatafromBin->ubFactor = 0;
//     tilingDatafromBin->epsilon = 1e-05;
//     tilingDatafromBin->reciprocal = 1.0f / 512.0f;
//     tilingDatafromBin->isOutputKv = true;
//     tilingDatafromBin->isKQuant = 0;
//     tilingDatafromBin->isVQuant = 0;

// #define DTYPE_KV half
// #define DTYPE_K_CACHE int8_t
// #define DTYPE_CKV_CACHE int8_t
//     ICPU_SET_TILING_KEY(1000);
//     ICPU_RUN_KF(
//         kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, nullptr, nullptr,
//         nullptr, nullptr, nullptr, nullptr, kRope, ckv, workspace, (uint8_t*)(tilingDatafromBin));

//     // ICPU_SET_TILING_KEY(1001);
//     // ICPU_RUN_KF(
//     //     kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, kRopeScale, ckvScale,
//     //     kRopeOffset, ckvOffset, nullptr, nullptr, kRope, ckv, workspace, (uint8_t*)(tilingDatafromBin));
//     // ICPU_SET_TILING_KEY(2000);
//     // ICPU_RUN_KF(
//     //     kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, kRopeScale, ckvScale,
//     //     kRopeOffset, ckvOffset, nullptr, nullptr, kRope, ckv, workspace, (uint8_t*)(tilingDatafromBin));
//     // ICPU_SET_TILING_KEY(2001);
//     // ICPU_RUN_KF(
//     //     kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, kRopeScale, ckvScale,
//     //     kRopeOffset, ckvOffset, nullptr, nullptr, kRope, ckv, workspace, (uint8_t*)(tilingDatafromBin));
//     // ICPU_SET_TILING_KEY(3000);
//     // ICPU_RUN_KF(
//     //     kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, kRopeScale, ckvScale,
//     //     kRopeOffset, ckvOffset, nullptr, nullptr, kRope, ckv, workspace, (uint8_t*)(tilingDatafromBin));
//     // ICPU_SET_TILING_KEY(3001);
//     // ICPU_RUN_KF(
//     //     kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, kRopeScale, ckvScale,
//     //     kRopeOffset, ckvOffset, nullptr, nullptr, kRope, ckv, workspace, (uint8_t*)(tilingDatafromBin));
//     // ICPU_SET_TILING_KEY(4000);
//     // ICPU_RUN_KF(
//     //     kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, kRopeScale, ckvScale,
//     //     kRopeOffset, ckvOffset, nullptr, nullptr, kRope, ckv, workspace, (uint8_t*)(tilingDatafromBin));
//     // ICPU_SET_TILING_KEY(4001);
//     // ICPU_RUN_KF(
//     //     kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, kRopeScale, ckvScale,
//     //     kRopeOffset, ckvOffset, nullptr, nullptr, kRope, ckv, workspace, (uint8_t*)(tilingDatafromBin));
//     // ICPU_SET_TILING_KEY(5000);
//     // ICPU_RUN_KF(
//     //     kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, kRopeScale, ckvScale,
//     //     kRopeOffset, ckvOffset, nullptr, nullptr, kRope, ckv, workspace, (uint8_t*)(tilingDatafromBin));
//     // ICPU_SET_TILING_KEY(5001);
//     // ICPU_RUN_KF(
//     //     kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, kRopeScale, ckvScale,
//     //     kRopeOffset, ckvOffset, nullptr, nullptr, kRope, ckv, workspace, (uint8_t*)(tilingDatafromBin));
//     // ICPU_SET_TILING_KEY(4010);
//     // ICPU_RUN_KF(
//     //     kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, kRopeScale, ckvScale,
//     //     kRopeOffset, ckvOffset, nullptr, nullptr, kRope, ckv, workspace, (uint8_t*)(tilingDatafromBin));
//     // ICPU_SET_TILING_KEY(4011);
//     // ICPU_RUN_KF(
//     //     kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, kRopeScale, ckvScale,
//     //     kRopeOffset, ckvOffset, nullptr, nullptr, kRope, ckv, workspace, (uint8_t*)(tilingDatafromBin));
//     // ICPU_SET_TILING_KEY(5010);
//     // ICPU_RUN_KF(
//     //     kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, kRopeScale, ckvScale,
//     //     kRopeOffset, ckvOffset, nullptr, nullptr, kRope, ckv, workspace, (uint8_t*)(tilingDatafromBin));
//     // ICPU_SET_TILING_KEY(5011);
//     // ICPU_RUN_KF(
//     //     kv_rms_norm_rope_cache, blockDim, kv, gamma, cos, sin, index, kCache, ckvCache, kRopeScale, ckvScale,
//     //     kRopeOffset, ckvOffset, nullptr, nullptr, kRope, ckv, workspace, (uint8_t*)(tilingDatafromBin));

//     AscendC::GmFree(kv);
//     AscendC::GmFree(gamma);
//     AscendC::GmFree(index);
//     AscendC::GmFree(cos);
//     AscendC::GmFree(sin);
//     AscendC::GmFree(kCache);
//     AscendC::GmFree(ckvCache);
//     // AscendC::GmFree(kRopeScale);
//     // AscendC::GmFree(kRopeOffset);
//     AscendC::GmFree(kRope);
//     AscendC::GmFree(ckv);
//     AscendC::GmFree(workspace);
//     AscendC::GmFree(tiling);
//     free(path_);
// }

