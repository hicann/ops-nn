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
#include "gtest/gtest.h"
#include "test_ifmr.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif
#include <cstdint>
#include "../../../op_kernel/ifmr_tiling_data.h"

using namespace std;

extern "C" __global__ __aicore__ void ifmr(GM_ADDR data, GM_ADDR data_min, GM_ADDR data_max, GM_ADDR cumsum,
                                            GM_ADDR scale, GM_ADDR offset, GM_ADDR workspace, GM_ADDR tiling);

class ifmr_test : public testing::Test {
    protected:
    static void SetUpTestCase() {
        cout << "ifmr_test SetUp\n" << endl;
    }
    static void TearDownTestCase() {
        cout << "ifmr_test TearDown\n" << endl;
    }
};

IfmrTilingData* FakeGetTilingData(uint64_t tilingKey, uint8_t *tiling, uint32_t blockDim) {
    IfmrTilingData* tilingData = reinterpret_cast<IfmrTilingData*>(tiling);

    tilingData->minPercentile = 0.999999;
    tilingData->maxPercentile = 0.999999;
    tilingData->searchRange[0] = 0.7;
    tilingData->searchRange[1] = 1.3;
    tilingData->searchStep = 0.01;
    tilingData->withOffset = true;
    tilingData->quantBits = 8;
    tilingData->dataLength = 320000;
    tilingData->cumsumLength = 512;

    return tilingData;
}

TEST_F(ifmr_test, test_case_0) {
    int64_t totalLength = 320000;
    int64_t bins = 512;

    // inputs
    size_t inputs_size = totalLength * sizeof(float);
    size_t min_size = sizeof(float);
    size_t max_size = sizeof(float);
    size_t cumsum_size = bins * sizeof(float);
    size_t scale_size = sizeof(float);
    size_t offset_size = sizeof(float);
    size_t tiling_data_size = sizeof(IfmrTilingData);

    uint8_t *inputs = (uint8_t*)AscendC::GmAlloc(inputs_size);
    uint8_t *min = (uint8_t*)AscendC::GmAlloc(min_size);
    uint8_t *max = (uint8_t*)AscendC::GmAlloc(max_size);
    uint8_t *cumsum = (uint8_t*)AscendC::GmAlloc(cumsum_size);
    uint8_t *scale = (uint8_t*)AscendC::GmAlloc(scale_size);
    uint8_t *offset = (uint8_t*)AscendC::GmAlloc(offset_size);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1; //cpu模拟使用单核
    system("cp -r ../ifmr_data ./");
    system("chmod -R 755 ./ifmr_data/");
    system("cd ./ifmr_data/ && rm -rf ./*bin");
    system("cd ./ifmr_data/ && python3 gen_data.py 320000 0 1 512");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/ifmr_data/inputs.bin", inputs_size, inputs, inputs_size);
    ReadFile(path + "/ifmr_data/inputs_min.bin", min_size, min, min_size);
    ReadFile(path + "/ifmr_data/inputs_max.bin", max_size, max, max_size);
    ReadFile(path + "/ifmr_data/cumsum.bin", cumsum_size, cumsum, cumsum_size);
    uint64_t tilingKey = 0;
    auto tilingData = FakeGetTilingData(tilingKey, tiling, blockDim);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(ifmr, blockDim, inputs, min, max, cumsum, scale, offset, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(inputs);
    AscendC::GmFree(min);
    AscendC::GmFree(max);
    AscendC::GmFree(cumsum);
    AscendC::GmFree(scale);
    AscendC::GmFree(offset);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}