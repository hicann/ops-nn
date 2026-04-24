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
 * \file test_foreach_div_scalar_apt.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../../../foreach_abs/tests/ut/op_kernel/foreach_abs_tiling_function.h"
#include "../tensor_list_operate.h"
#include "../../../../../foreach_abs/tests/ut/op_kernel/foreach_abs_tiling_def.h"

extern "C" __global__ __aicore__ void foreach_div_scalar(GM_ADDR x, GM_ADDR scalar, GM_ADDR y, GM_ADDR workspace,
                                                                                GM_ADDR tiling);

class foreach_div_scalar_apt_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "foreach_div_scalar_apt_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "foreach_div_scalar_apt_test TearDown\n" << std::endl;
    }
};

TEST_F(foreach_div_scalar_apt_test, test_case_float_1) {
    std::vector<std::vector<uint64_t>> shapeInfos = {{18, 1}, {1, 1}, {1, 1}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_div_scalar/tests/ut/op_kernel/div_scalar_data ./");
    system("chmod -R 755 ./div_scalar_data/");
    system("cd ./div_scalar_data/ && python3 gen_data.py '{{18, 1}, {1, 1}, {1, 1}}' 3 'float32'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 1;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    ForeachSoloTilingDataRegbase* tilingDatafromBin = reinterpret_cast<ForeachSoloTilingDataRegbase*>(tiling);

    tilingDatafromBin->inputsTensorUbSize = 2048;
    tilingDatafromBin->tensorDataCountList[0] = 18;
    tilingDatafromBin->tensorDataCountList[1] = 1;
    tilingDatafromBin->tensorDataCountList[2] = 1;
    tilingDatafromBin->tensorStartList[0] = 0;
    tilingDatafromBin->tensorEndList[0] = 2;
    tilingDatafromBin->tensorStartOffsetList[0] = 0;
    tilingDatafromBin->tensorEndOffsetList[0] = 0;

    uint8_t* x1 = CreateListTensorForeachDivScalar<float>(shapeInfos, "float32");
    uint8_t* x2 = CreateListTensorForeachDivScalar<float>(shapeInfos, "float32");
    uint8_t* x3 = (uint8_t*)AscendC::GmAlloc(sizeof(float));

    *((float*)x3) = 3;

    ICPU_SET_TILING_KEY(10002);
    ICPU_RUN_KF(foreach_div_scalar, blockDim, x1, x3, x2, workspace, (uint8_t*)(tilingDatafromBin));

    FreeTensorListForeachDivScalar<float>(x2, shapeInfos, "float32");
    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)x3);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    int ret = system("cd ./div_scalar_data && python compare_data.py 3 'float32'");
    std::cout << "exit = " << ret << std::endl;
}

TEST_F(foreach_div_scalar_apt_test, test_case_float16_2) {
    std::vector<std::vector<uint64_t>> shapeInfos = {{1, 3, 3, 4, 2, 1, 4, 4}, {1, 3, 3, 4, 2, 1, 4, 4}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_div_scalar/tests/ut/op_kernel/div_scalar_data ./");
    system("chmod -R 755 ./div_scalar_data/");
    system("cd ./div_scalar_data/ && python3 gen_data.py '{{1, 3, 3, 4, 2, 1, 4, 4}, {1, 3, 3, 4, 2, 1, 4, 4}}' 3 'float16'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 2;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    ForeachSoloTilingDataRegbase* tilingDatafromBin = reinterpret_cast<ForeachSoloTilingDataRegbase*>(tiling);

    tilingDatafromBin->inputsTensorUbSize = 21152;
    tilingDatafromBin->tensorDataCountList[0] = 1152;
    tilingDatafromBin->tensorDataCountList[1] = 1152;
    tilingDatafromBin->tensorStartList[0] = 0;
    tilingDatafromBin->tensorStartList[1] = 1;
    tilingDatafromBin->tensorEndList[0] = 1;
    tilingDatafromBin->tensorEndList[1] = 1;
    tilingDatafromBin->tensorStartOffsetList[0] = 0;
    tilingDatafromBin->tensorStartOffsetList[1] = 896;
    tilingDatafromBin->tensorEndOffsetList[0] = 895;
    tilingDatafromBin->tensorEndOffsetList[1] = 1151;

    uint8_t* x1 = CreateListTensorForeachDivScalar<float>(shapeInfos, "float16");
    uint8_t* x2 = CreateListTensorForeachDivScalar<float>(shapeInfos, "float16");
    uint8_t* x3 = (uint8_t*)AscendC::GmAlloc(sizeof(float));

    *((float*)x3) = 3;

    ICPU_SET_TILING_KEY(10001);
    ICPU_RUN_KF(foreach_div_scalar, blockDim, x1, x3, x2, workspace, (uint8_t*)(tilingDatafromBin));

    FreeTensorListForeachDivScalar<half>(x2, shapeInfos, "float16");
    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)x3);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    int ret = system("cd ./div_scalar_data && python compare_data.py 'float16'");
    std::cout << "exit = " << ret << std::endl;
}