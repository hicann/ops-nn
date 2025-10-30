/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>
#include "trans_quant_param_v2_tiling_def.h"

using namespace std;

extern "C" __global__ __aicore__ void trans_quant_param_v2(
    GM_ADDR scale, GM_ADDR offset, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class trans_quant_param_v2_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "trans_quant_param_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "trans_quant_param_v2_test TearDown\n" << endl;
    }
};

TEST_F(trans_quant_param_v2_test, test_case_2_1)
{
    size_t shape_scale = 2 * sizeof(float);
    size_t shape_offset = 2 * sizeof(float);
    size_t shape_y = 2 * sizeof(uint64_t);
    size_t tiling_data_size = 3 * sizeof(uint32_t);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(shape_scale);
    uint8_t* offset = (uint8_t*)AscendC::GmAlloc(shape_offset);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(shape_y);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    system(
        "cp -r ../../../../quant/trans_quant_param_v2/tests/ut/op_kernel/trans_quant_param_data "
        "./");
    system("chmod -R 755 ./trans_quant_param_data/");
    system("cd ./trans_quant_param_data/ && rm -rf ./*bin");
    system("cd ./trans_quant_param_data/ && python3 gen_data.py 2 1");

    char* path_ = get_current_dir_name();
    string path(path_);
    TransQuantParamV2TilingData* tilingDatafromBin = reinterpret_cast<TransQuantParamV2TilingData*>(tiling);
    tilingDatafromBin->scaleLength = 2;
    tilingDatafromBin->offsetLength = 2;
    tilingDatafromBin->roundMode = 0;

    ReadFile(path + "/trans_quant_param_data/scale.bin", shape_scale, scale, shape_scale);
    ReadFile(path + "/trans_quant_param_data/offset.bin", shape_offset, offset, shape_offset);

    ICPU_RUN_KF(trans_quant_param_v2, 1, scale, offset, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(scale);
    AscendC::GmFree(offset);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}