/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include <cstdint>
#include "../../../op_kernel/bucketize_v2_apt.cpp"
#include "../../../op_kernel/arch35/bucketize_v2_struct.h"

using namespace std;
class bucketize_v2_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "bucketize_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "bucketize_v2_test TearDown\n" << endl;
    }
};

TEST_F(bucketize_v2_test, test_case_fp32_right_false)
{
    size_t inputByteSize = 2048 * sizeof(float);
    size_t outputByteSize = 2048 * sizeof(int32_t);
    size_t boundByteSize = 512 * sizeof(float);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* boundaries = (uint8_t*)AscendC::GmAlloc(boundByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint64_t tilingKey = 1;
    uint32_t blockDim = 1;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    size_t tilingDataSize = sizeof(BucketizeV2FullLoadTilingData);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../index/bucketize_v2/tests/ut/op_kernel/bucketize_v2_data ./");
    system("chmod -R 755 ./bucketize_v2_data/");
    system("cd ./bucketize_v2_data/ && rm -rf ./*bin");
    system("cd ./bucketize_v2_data/ && python3 gen_data.py case0");

    char* path_ = get_current_dir_name();
    string path(path_);

    ReadFile(path + "/bucketize_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/bucketize_v2_data/input_boundaries.bin", boundByteSize, boundaries, boundByteSize);
    BucketizeV2FullLoadTilingData* tilingDatafromBin = reinterpret_cast<BucketizeV2FullLoadTilingData*>(tiling);

    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->blockFactor = 2048;
    tilingDatafromBin->blockTail = 2048;
    tilingDatafromBin->ubFactor = 15648;
    tilingDatafromBin->boundBufSize = 2048;
    tilingDatafromBin->inUbSize = 62720;
    tilingDatafromBin->outUbSize = 62976;
    tilingDatafromBin->boundSize = 512;
    tilingDatafromBin->maxIter = 10;

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto KernelBucketizeV2 = [](GM_ADDR x, GM_ADDR boundaries, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::bucketize_v2<1,false>(x, boundaries, y, workspace, tiling);
    };
    ICPU_RUN_KF(KernelBucketizeV2, blockDim,
                x, boundaries, y, workspace, tiling);

    WriteFile(path + "/bucketize_v2_data/res_output.bin", y, outputByteSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)boundaries);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);

    int ret = system(("cd " + path + "/bucketize_v2_data && python3 compare_data.py int32").c_str());
    std::cout << "compare exit = " << ret << std::endl;
    ASSERT_EQ(ret, 0);
    free(path_);
}

TEST_F(bucketize_v2_test, test_case_fp32_right_true)
{
    size_t inputByteSize = 2048 * sizeof(float);
    size_t outputByteSize = 2048 * sizeof(int32_t);
    size_t boundByteSize = 512 * sizeof(float);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* boundaries = (uint8_t*)AscendC::GmAlloc(boundByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint64_t tilingKey = 5;
    uint32_t blockDim = 1;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    size_t tilingDataSize = sizeof(BucketizeV2FullLoadTilingData);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    system("cp -r ../../../../index/bucketize_v2/tests/ut/op_kernel/bucketize_v2_data ./");
    system("chmod -R 755 ./bucketize_v2_data/");
    system("cd ./bucketize_v2_data/ && rm -rf ./*bin");
    system("cd ./bucketize_v2_data/ && python3 gen_data.py case1");

    char* path_ = get_current_dir_name();
    string path(path_);

    ReadFile(path + "/bucketize_v2_data/input_x.bin", inputByteSize, x, inputByteSize);
    ReadFile(path + "/bucketize_v2_data/input_boundaries.bin", boundByteSize, boundaries, boundByteSize);
    BucketizeV2FullLoadTilingData* tilingDatafromBin = reinterpret_cast<BucketizeV2FullLoadTilingData*>(tiling);

    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->blockFactor = 2048;
    tilingDatafromBin->blockTail = 2048;
    tilingDatafromBin->ubFactor = 15648;
    tilingDatafromBin->boundBufSize = 2048;
    tilingDatafromBin->inUbSize = 62720;
    tilingDatafromBin->outUbSize = 62976;
    tilingDatafromBin->boundSize = 512;
    tilingDatafromBin->maxIter = 10;

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto KernelBucketizeV2 = [](GM_ADDR x, GM_ADDR boundaries, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::bucketize_v2<1,true>(x, boundaries, y, workspace, tiling);
    };
    ICPU_RUN_KF(KernelBucketizeV2, blockDim,
                x, boundaries, y, workspace, tiling);

    WriteFile(path + "/bucketize_v2_data/res_output.bin", y, outputByteSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)boundaries);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);

    int ret = system(("cd " + path + "/bucketize_v2_data && python3 compare_data.py int32").c_str());
    std::cout << "compare exit = " << ret << std::endl;
    ASSERT_EQ(ret, 0);
    free(path_);
}