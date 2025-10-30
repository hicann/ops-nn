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
 * \file test_foreach_addcdiv_scalar_list.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../../foreach_abs/tests/ut/op_kernel/foreach_abs_tiling_function.h"
#include "tensor_list_operate.h"

extern "C" __global__ __aicore__ void foreach_addcdiv_scalar_list(GM_ADDR inputs, GM_ADDR tensor1,
                                                            GM_ADDR tensor2, GM_ADDR scalar,
                                                            GM_ADDR outputs,
                                                            GM_ADDR workspace, GM_ADDR tiling);

class foreach_addcdiv_scalar_list_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "foreach_addcdiv_scalar_list_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "foreach_addcdiv_scalar_list_test TearDown\n" << std::endl;
    }
};

TEST_F(foreach_addcdiv_scalar_list_test, test_case_float_1) {
    std::vector<std::vector<uint64_t>> shapeInfos = {{128, 64}, {16, 128}, {32, 128}};
    system(
        "cp -rf "
        "../../../../foreach/foreach_addcdiv_scalar_list/tests/ut/op_kernel/addcdiv_scalar_list_data ./");
    system("chmod -R 755 ./addcdiv_scalar_list_data/");
    system("cd ./addcdiv_scalar_list_data/ && python3 gen_data.py '{{128, 64}, {16, 128}, {32, 128}}' 'float32'");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ForeachCommonTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    optiling::ForeachCommonTiling tilingFuncObj;
    tilingFuncObj.Init(shapeInfos, 1, 4);
    tilingFuncObj.RunBigKernelTiling(blockDim);
    tilingFuncObj.FillTilingData(reinterpret_cast<ForeachCommonTilingData*>(tiling));

    uint8_t* x1 = CreateTensorListForeachAddcdivScalarList<float>(shapeInfos, "float32", "input");
    uint8_t* tensor1 = CreateTensorListForeachAddcdivScalarList<float>(shapeInfos, "float32", "tensor1");
    uint8_t* tensor2 = CreateTensorListForeachAddcdivScalarList<float>(shapeInfos, "float32", "tensor2");
    uint8_t* out = CreateTensorListForeachAddcdivScalarList<float>(shapeInfos, "float32", "input");
    float* scalar = (float*)AscendC::GmAlloc(sizeof(float)*3);

    for (int i = 0; i < 3; i++) {
        scalar[i] = i;
    }

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(foreach_addcdiv_scalar_list, blockDim, x1, tensor1, tensor2, (uint8_t*)scalar, out, workspace, tiling);

    FreeTensorListForeachAddcdivScalarList<float>(out, shapeInfos, "float32");
    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)scalar);
    AscendC::GmFree((void*)tensor1);
    AscendC::GmFree((void*)tensor2);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./addcdiv_scalar_list_data/ && python3 compare_data.py 'float32'");
}
