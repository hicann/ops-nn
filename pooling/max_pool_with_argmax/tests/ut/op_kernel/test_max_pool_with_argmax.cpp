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
#include <limits>
#include <float.h>
#include <numeric>
#include <unistd.h>
#include <type_traits>
#include "../../../op_kernel/arch35/max_pool_with_argmax_struct_common.h"
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

#include <cstdint>
using namespace std;

extern "C" __global__ __aicore__ void max_pool_with_argmax(GM_ADDR x, GM_ADDR y, GM_ADDR argmax,
                                                           GM_ADDR workspace, GM_ADDR tiling);

class max_pool_with_argmax_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "max_pool_with_argmax_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "max_pool_with_argmax_test TearDown\n" << endl;
    }
};

inline int64_t GetShapeSize(const std::vector<int64_t> &shape) {
    return std::accumulate(shape.begin(), shape.end(), 1LL, std::multiplies<int64_t>());
}

template <typename T>
void ExcuteTestCase(const std::vector<int64_t> &inputShape,
                    const std::vector<int64_t> &outputShape,
                    const std::string &input_Type,
                    const std::string &indices_Type,
                    const std::string &caseName,
                    uint64_t tilingKey,
                    const T &tilingConfig)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    uint32_t inputElemSize = (input_Type == "float32" ? 4 : 2);
    uint32_t indexElemSize = (indices_Type == "int32" ? 4 : 8);

    size_t inputByteSize = GetShapeSize(inputShape) * inputElemSize;
    size_t outputByteSize = GetShapeSize(outputShape) * inputElemSize;
    size_t argmaxByteSize = GetShapeSize(outputShape) * indexElemSize;
    size_t workspaceSize = 16 * 1024 * 1024;
    size_t tilingSize = sizeof(T);

    uint8_t *input = (uint8_t *)AscendC::GmAlloc((inputByteSize + 31) / 32 * 32);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc((outputByteSize + 31) / 32 * 32);
    uint8_t *argmax = (uint8_t *)AscendC::GmAlloc((argmaxByteSize + 31) / 32 * 32);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    memset_s(input, (inputByteSize + 31) / 32 * 32, 0, (inputByteSize + 31) / 32 * 32);
    memset_s(out, (outputByteSize + 31) / 32 * 32, 0, (outputByteSize + 31) / 32 * 32);
    memset_s(argmax, (argmaxByteSize + 31) / 32 * 32, 0, (argmaxByteSize + 31) / 32 * 32);
    memset_s(workspace, workspaceSize, 0, workspaceSize);

    std::string path = std::string(get_current_dir_name()) + "/max_pool_data/";
    system(("rm -rf " + path + "*.bin").c_str());

    system("cp -r ../../../../pooling/max_pool_with_argmax/tests/ut/op_kernel/max_pool_data ./");
    system("chmod -R 755 ./max_pool_data/");

    std::string cmd = "cd ./max_pool_data/ && python3 gen_data.py " + caseName + " " + input_Type + " " + indices_Type;
    system(cmd.c_str());
    cmd = "cd ./max_pool_data/ && python3 gen_tiling.py " + caseName;
    system(cmd.c_str());

    ReadFile(path + "input.bin", inputByteSize, input, inputByteSize);

    T* tilingDataPtr = reinterpret_cast<T*>(tiling);
    *tilingDataPtr = tilingConfig;

    uint64_t blockDim = 1;
    if constexpr (std::is_same_v<T, MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxNHWCTilingCommonData>) {
        blockDim = tilingConfig.usedCoreNum;
    } else {
        blockDim = tilingConfig.blockNums;
    }

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(max_pool_with_argmax, blockDim, input, out, argmax, workspace, tiling);

    WriteFile(path + "cce_cpu_y.bin", out, outputByteSize);
    cmd = "cmp " + path + "cce_cpu_y.bin " + path + "y_golden.bin";
    (void)system(cmd.c_str());

    AscendC::GmFree(input);
    AscendC::GmFree(out);
    AscendC::GmFree(argmax);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(max_pool_with_argmax_test, test_case_nhwc_bigc_800001) {
    MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxNHWCTilingCommonData tiling = {
        11, 14, 8, 4, 1, 5, 6, 3, 6, 9, 0, 1, 1, 17, 1, 1, 4, 1, 1, 1, 11, 11, 1, 2, 2, 34, 960, 32, 128, 0, 0, 0, 0, 0, 0, 0
    };
    ExcuteTestCase({17, 14, 8, 11}, {17, 4, 1, 11}, "float32", "int32", "test_case_nhwc_bigc_800001", 800001UL, tiling);
}

TEST_F(max_pool_with_argmax_test, test_case_nhwc_bigc_800011) {
    MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxNHWCTilingCommonData tiling = {
        2, 2, 2, 1, 1, 2, 2, 8303, 2, 0, 0, 5, 4, 69, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 35, 640, 160, 320, 0, 0, 0, 0, 0, 0, 0
    };
    ExcuteTestCase({344, 2, 2, 2}, {344, 1, 1, 2}, "float32", "int32", "test_case_nhwc_bigc_800011", 800011UL, tiling);
}

TEST_F(max_pool_with_argmax_test, test_case_nhwc_bigc_800002) {
    MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxNHWCTilingCommonData tiling = {
        19, 5, 7, 2, 7, 4, 2, 4, 1, 0, 1, 1, 1, 17, 1, 1, 2, 6, 1, 2, 19, 19, 1, 2, 2, 34, 1792, 384, 1536, 1, 0, 0, 0, 0, 0, 0
    };
    ExcuteTestCase({17, 5, 7, 19}, {17, 2, 7, 19}, "float32", "int64", "test_case_nhwc_bigc_800002", 800002UL, tiling);
}

TEST_F(max_pool_with_argmax_test, test_case_nhwc_bigc_800012) {
    MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxNHWCTilingCommonData tiling = {
        79828, 2, 1, 1, 1, 2, 2, 2, 2, 0, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2560, 468, 32, 1, 1, 64, 20480, 5120, 20480, 1, 0, 0, 0, 0, 0, 0
    };
    ExcuteTestCase({2, 2, 1, 79828}, {2, 1, 1, 79828}, "float32", "int64", "test_case_nhwc_bigc_800012", 800012UL, tiling);
}

TEST_F(max_pool_with_argmax_test, test_case_nhwc_smallc_700001) {
    MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxNHWCTilingCommonData tiling = {
        3, 7, 5871, 3, 587, 2, 2, 2, 10, 0, 0, 1, 1, 3, 1, 1, 3, 83, 6, 8, 3, 3, 1, 2, 2, 36, 52608, 2656, 10624, 0, 0, 0, 0, 0, 0, 0
    };
    ExcuteTestCase({3, 7, 5871, 3}, {3, 3, 587, 3}, "float32", "int64", "test_case_nhwc_smallc_700001", 700001UL, tiling);
}

TEST_F(max_pool_with_argmax_test, test_case_nhwc_smallc_700011) {
    MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxNHWCTilingCommonData tiling = {
        15, 2218, 4, 1109, 1, 2, 2, 2, 1068, 0, 0, 1, 1, 4, 73, 14, 16, 1, 1, 1, 15, 15, 1, 1, 1, 64, 9344, 2336, 4672, 0, 0, 0, 0, 0, 0, 0
    };
    ExcuteTestCase({4, 2218, 4, 15}, {4, 1109, 1, 15}, "float32", "int32", "test_case_nhwc_smallc_700011", 700011UL, tiling);
}

TEST_F(max_pool_with_argmax_test, test_case_nchw_simt_50101) {
    MaxPoolWithArgmaxCommonStructNameSpace::MaxPoolWithArgmaxSimtTilingCommonData tiling = {
        256, 64, 3, 12, 3, 3311, 2, 828, 2, 2, 2, 4, 0, 0, 0
    };
    ExcuteTestCase({3, 12, 3, 3311}, {3, 12, 2, 828}, "float32", "int64", "test_case_nchw_simt_500101", 500101UL, tiling);
}