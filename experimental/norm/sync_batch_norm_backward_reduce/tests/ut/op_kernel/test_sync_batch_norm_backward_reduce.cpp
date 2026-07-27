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
 * \file test_sync_batch_norm_backward_reduce.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#endif

#include "../../../op_kernel/sync_batch_norm_backward_reduce.cpp"

using namespace std;

extern "C" __global__ __aicore__ void sync_batch_norm_backward_reduce(GM_ADDR sum_dy, GM_ADDR sum_dy_dx_pad,
                                                                      GM_ADDR mean, GM_ADDR invert_std,
                                                                      GM_ADDR sum_dy_xmu, GM_ADDR y, GM_ADDR workspace,
                                                                      GM_ADDR tiling);

class SyncBatchNormBackwardReduceTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "sync_batch_norm_backward_reduce_test SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        ASSERT_EQ(system(cmd.c_str()), 0);
        ASSERT_EQ(system("chmod -R 755 ./sync_batch_norm_backward_reduce_data/"), 0);
    }
    static void TearDownTestCase() { std::cout << "sync_batch_norm_backward_reduce_test TearDown" << std::endl; }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string SyncBatchNormBackwardReduceTest::rootPath = "../../../../";
const std::string SyncBatchNormBackwardReduceTest::dataPath = rootPath +
                                                              "experimental/norm/sync_batch_norm_backward_reduce/tests/"
                                                              "ut/op_kernel/sync_batch_norm_backward_reduce_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

TEST_F(SyncBatchNormBackwardReduceTest, test_case_float32)
{
    uint32_t blockDim = 1;
    uint32_t dataCount = 256;
    ASSERT_EQ(system("cd ./sync_batch_norm_backward_reduce_data/ && python3 gen_data.py '(256,)' 'float32'"), 0);

    size_t byteSize = dataCount * sizeof(float);
    std::string sumDyFile = "./sync_batch_norm_backward_reduce_data/float32_input_sum_dy.bin";
    std::string sumDyDxPadFile = "./sync_batch_norm_backward_reduce_data/float32_input_sum_dy_dx_pad.bin";
    std::string meanFile = "./sync_batch_norm_backward_reduce_data/float32_input_mean.bin";
    std::string invStdFile = "./sync_batch_norm_backward_reduce_data/float32_input_invert_std.bin";

    uint8_t* sumDy = (uint8_t*)AscendC::GmAlloc(CeilAlign(byteSize, 32));
    ASSERT_NE(sumDy, nullptr);
    uint8_t* sumDyDxPad = (uint8_t*)AscendC::GmAlloc(CeilAlign(byteSize, 32));
    ASSERT_NE(sumDyDxPad, nullptr);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(CeilAlign(byteSize, 32));
    ASSERT_NE(mean, nullptr);
    uint8_t* invStd = (uint8_t*)AscendC::GmAlloc(CeilAlign(byteSize, 32));
    ASSERT_NE(invStd, nullptr);
    ReadFile(sumDyFile, byteSize, sumDy, byteSize);
    ReadFile(sumDyDxPadFile, byteSize, sumDyDxPad, byteSize);
    ReadFile(meanFile, byteSize, mean, byteSize);
    ReadFile(invStdFile, byteSize, invStd, byteSize);

    uint8_t* sumDyXmu = (uint8_t*)AscendC::GmAlloc(CeilAlign(byteSize, 32));
    ASSERT_NE(sumDyXmu, nullptr);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(byteSize, 32));
    ASSERT_NE(y, nullptr);

    size_t workspaceSize = 32 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    ASSERT_NE(workspace, nullptr);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(SyncBatchNormBackwardReduceTilingData));
    ASSERT_NE(tiling, nullptr);

    SyncBatchNormBackwardReduceTilingData* tilingData = reinterpret_cast<SyncBatchNormBackwardReduceTilingData*>(
        tiling);
    tilingData->coreNum = 1;
    tilingData->bufferNum = 1;
    tilingData->tailElems = 0;
    tilingData->epochs = 1;
    tilingData->epochsForLastCore = 1;
    tilingData->coreLength = dataCount;
    tilingData->tileLength = dataCount;
    tilingData->tailTileLength = 0;
    tilingData->tailTileLengthForLastCore = 0;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = sync_batch_norm_backward_reduce<SYNCBNBR_TPL_SCH_MODE_1>;
    ICPU_RUN_KF(func, blockDim, sumDy, sumDyDxPad, mean, invStd, sumDyXmu, y, workspace, (uint8_t*)(tilingData));

    WriteFile("./sync_batch_norm_backward_reduce_data/float32_output_sum_dy_xmu.bin", sumDyXmu, byteSize);
    WriteFile("./sync_batch_norm_backward_reduce_data/float32_output_y.bin", y, byteSize);

    AscendC::GmFree((void*)(sumDy));
    AscendC::GmFree((void*)(sumDyDxPad));
    AscendC::GmFree((void*)(mean));
    AscendC::GmFree((void*)(invStd));
    AscendC::GmFree((void*)(sumDyXmu));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    EXPECT_EQ(system("cd ./sync_batch_norm_backward_reduce_data/ && python3 compare_data.py 'float32'"), 0);
}
