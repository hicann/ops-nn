/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <unistd.h>
#include <array>
#include <vector>

#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__

#include <iostream>
#include <string>

#include "tikicpulib.h"
#include "data_utils.h"
#include "transpose_batch_mat_mul_tiling_def.h"
#include "string.h"
#endif

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

using namespace std;

extern "C" __global__ __aicore__ void transpose_batch_mat_mul(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM,
                                            GM_ADDR offsetWGM, GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM);

class transpose_batch_mat_mul_test : public testing::Test {
   protected:
    static void SetUpTestCase() { cout << "transpose_batch_mat_mul_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "transpose_batch_mat_mul_test TearDown\n" << endl; }
};

struct HcclCombinOpParam {
    uint64_t WorkSpace;
    uint64_t WorkSpaceSize;
    uint32_t rankId;
    uint32_t rankDim;
};
