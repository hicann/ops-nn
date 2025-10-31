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
#include "../../../../conv3d_transpose_v2/op_kernel/conv3d_transpose_v2.cpp"
#include "../../../../conv3d_backprop_input_v2/op_kernel/arch32/conv3d_backprop_input_v2_tiling_data.h"
#include <cstdint>

using namespace std;
using namespace AscendC;

class conv3d_transpose_v2_test : public testing::Test {
    protected:
    static void SetUpTestCase() {
        cout << "conv3d_transpose_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase() {
        cout << "conv3d_transpose_v2_test TearDown\n" << endl;
    }
};


TEST_F(conv3d_transpose_v2_test, test_conv3d_dx_17) {
    size_t input_shape_size = 5 * sizeof(int32_t); // 5 len of input shape
    size_t filter_size = 32 * 32 * 16 * 16 * sizeof(int16_t); // [dk*cin1*hk*wk, cout1, 16, 16] bf16
    size_t x_size = 1 * 5 * 32 * 32 * 32 * 16 * sizeof(int16_t); // NDC1HW0 bf16
    size_t y_size = 1 * 5 * 32 * 32 * 32 * 16 * sizeof(int16_t); // NDC1HWC0 bf16
    size_t tiling_data_size = sizeof(Conv3DBackpropInputV2TilingData);

    uint8_t *input_shape = (uint8_t *)AscendC::GmAlloc(input_shape_size);
    uint8_t *filter = (uint8_t *)AscendC::GmAlloc(filter_size);
    uint8_t *x = (uint8_t *)AscendC::GmAlloc(x_size);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(y_size);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);

    memset(workspace, 0, 16 * 1024 * 1024);

    system("cp -r ../../../../conv/conv3d_transpose_v2/tests/ut/op_kernel/conv3d_transpose_v2_data ./");
    system("chmod -R 755 ./conv3d_transpose_v2_data/");
    system("cd ./conv3d_transpose_v2_data/ && rm -rf ./*bin");
    system("cd ./conv3d_transpose_v2_data/ && python3 gen_data.py 1 512 512 5 32 32 5 32 32 1 1 1");
    system("cd ./conv3d_transpose_v2_data/ && python3 gen_tiling.py test_conv3d_dx_17");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/conv3d_transpose_v2_data/input_size.bin", input_shape_size, input_shape, input_shape_size);
    ReadFile(path + "/conv3d_transpose_v2_data/filter.bin", filter_size, filter, filter_size);
    ReadFile(path + "/conv3d_transpose_v2_data/x.bin", x_size, x, x_size);
    ReadFile(path + "/conv3d_transpose_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    auto Conv3DTrransposeKernel = [](GM_ADDR input_size, GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR offset_w, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling) {
        ::conv3d_transpose_v2<1,1,1>(input_size, x, filter, bias, offset_w, y, workSpace, tiling);
    };
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(Conv3DTrransposeKernel, 24, input_shape, x, filter, nullptr, nullptr, y, workspace, tiling);

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(Conv3DTrransposeKernel, 1, input_shape, x, filter, nullptr, nullptr, y, workspace, tiling);

    ICPU_SET_TILING_KEY(10);
    ICPU_RUN_KF(Conv3DTrransposeKernel, 1, input_shape, x, filter, nullptr, nullptr, y, workspace, tiling);

    ICPU_SET_TILING_KEY(11);
    ICPU_RUN_KF(Conv3DTrransposeKernel, 1, input_shape, x, filter, nullptr, nullptr, y, workspace, tiling);

    ICPU_SET_TILING_KEY(100);
    ICPU_RUN_KF(Conv3DTrransposeKernel, 1, input_shape, x, filter, nullptr, nullptr, y, workspace, tiling);

    ICPU_SET_TILING_KEY(101);
    ICPU_RUN_KF(Conv3DTrransposeKernel, 1, input_shape, x, filter, nullptr, nullptr, y, workspace, tiling);

    ICPU_SET_TILING_KEY(110);
    ICPU_RUN_KF(Conv3DTrransposeKernel, 1, input_shape, x, filter, nullptr, nullptr, y, workspace, tiling);

    ICPU_SET_TILING_KEY(111);
    ICPU_RUN_KF(Conv3DTrransposeKernel, 1, input_shape, x, filter, nullptr, nullptr, y, workspace, tiling);

    ICPU_SET_TILING_KEY(200);
    ICPU_RUN_KF(Conv3DTrransposeKernel, 1, input_shape, x, filter, nullptr, nullptr, y, workspace, tiling);

    ICPU_SET_TILING_KEY(201);
    ICPU_RUN_KF(Conv3DTrransposeKernel, 1, input_shape, x, filter, nullptr, nullptr, y, workspace, tiling);

    ICPU_SET_TILING_KEY(210);
    ICPU_RUN_KF(Conv3DTrransposeKernel, 1, input_shape, x, filter, nullptr, nullptr, y, workspace, tiling);

    ICPU_SET_TILING_KEY(211);
    ICPU_RUN_KF(Conv3DTrransposeKernel, 1, input_shape, x, filter, nullptr, nullptr, y, workspace, tiling);

    ICPU_SET_TILING_KEY(11010);
    ICPU_RUN_KF(Conv3DTrransposeKernel, 1, input_shape, x, filter, nullptr, nullptr, y, workspace, tiling);

    ICPU_SET_TILING_KEY(11011);
    ICPU_RUN_KF(Conv3DTrransposeKernel, 1, input_shape, x, filter, nullptr, nullptr, y, workspace, tiling);

    AscendC::GmFree(input_shape);
    AscendC::GmFree(filter);
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(conv3d_transpose_v2_test, params_conv3d_dx_17_hf32) {
    size_t input_shape_size = 5 * sizeof(int32_t); // 5 len of input shape
    size_t filter_size = 32 * 32 * 16 * 16 * sizeof(int16_t); // [dk*cin1*hk*wk, cout1, 16, 16] bf16
    size_t x_size = 1 * 5 * 32 * 32 * 32 * 16 * sizeof(int16_t); // NDC1HW0 bf16
    size_t y_size = 1 * 5 * 32 * 32 * 32 * 16 * sizeof(int16_t); // NDC1HWC0 bf16
    size_t tiling_data_size = sizeof(Conv3DBackpropInputV2TilingData);

    uint8_t *input_shape = (uint8_t *)AscendC::GmAlloc(input_shape_size);
    uint8_t *filter = (uint8_t *)AscendC::GmAlloc(filter_size);
    uint8_t *x = (uint8_t *)AscendC::GmAlloc(x_size);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(y_size);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);

    memset(workspace, 0, 16 * 1024 * 1024);

    system("cp -r ../../../../conv/conv3d_transpose_v2/tests/ut/op_kernel/conv3d_transpose_v2_data ./");
    system("chmod -R 755 ./conv3d_transpose_v2_data/");
    system("cd ./conv3d_transpose_v2_data/ && rm -rf ./*bin");
    system("cd ./conv3d_transpose_v2_data/ && python3 gen_data.py 1 512 512 5 32 32 5 32 32 1 1 1");
    system("cd ./conv3d_transpose_v2_data/ && python3 gen_tiling.py params_conv3d_dx_17_hf32");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/conv3d_transpose_v2_data/input_size.bin", input_shape_size, input_shape, input_shape_size);
    ReadFile(path + "/conv3d_transpose_v2_data/filter.bin", filter_size, filter, filter_size);
    ReadFile(path + "/conv3d_transpose_v2_data/x.bin", x_size, x, x_size);
    ReadFile(path + "/conv3d_transpose_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    auto Conv3DTrransposeKernel = [](GM_ADDR input_size, GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR offset_w, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling) {
        ::conv3d_transpose_v2<1,1,1>(input_size, x, filter, bias, offset_w, y, workSpace, tiling);
    };
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(Conv3DTrransposeKernel, 24, input_shape, x, filter, nullptr, nullptr, y, workspace, tiling);

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(Conv3DTrransposeKernel, 1, input_shape, x, filter, nullptr, nullptr, y, workspace, tiling);

    ICPU_SET_TILING_KEY(10);
    ICPU_RUN_KF(Conv3DTrransposeKernel, 1, input_shape, x, filter, nullptr, nullptr, y, workspace, tiling);

    ICPU_SET_TILING_KEY(11);
    ICPU_RUN_KF(Conv3DTrransposeKernel, 1, input_shape, x, filter, nullptr, nullptr, y, workspace, tiling);

    ICPU_SET_TILING_KEY(100);
    ICPU_RUN_KF(Conv3DTrransposeKernel, 1, input_shape, x, filter, nullptr, nullptr, y, workspace, tiling);

    ICPU_SET_TILING_KEY(101);
    ICPU_RUN_KF(Conv3DTrransposeKernel, 1, input_shape, x, filter, nullptr, nullptr, y, workspace, tiling);

    ICPU_SET_TILING_KEY(110);
    ICPU_RUN_KF(Conv3DTrransposeKernel, 1, input_shape, x, filter, nullptr, nullptr, y, workspace, tiling);

    ICPU_SET_TILING_KEY(111);
    ICPU_RUN_KF(Conv3DTrransposeKernel, 1, input_shape, x, filter, nullptr, nullptr, y, workspace, tiling);

    ICPU_SET_TILING_KEY(200);
    ICPU_RUN_KF(Conv3DTrransposeKernel, 1, input_shape, x, filter, nullptr, nullptr, y, workspace, tiling);

    ICPU_SET_TILING_KEY(201);
    ICPU_RUN_KF(Conv3DTrransposeKernel, 1, input_shape, x, filter, nullptr, nullptr, y, workspace, tiling);

    ICPU_SET_TILING_KEY(210);
    ICPU_RUN_KF(Conv3DTrransposeKernel, 1, input_shape, x, filter, nullptr, nullptr, y, workspace, tiling);

    ICPU_SET_TILING_KEY(211);
    ICPU_RUN_KF(Conv3DTrransposeKernel, 1, input_shape, x, filter, nullptr, nullptr, y, workspace, tiling);

    ICPU_SET_TILING_KEY(11010);
    ICPU_RUN_KF(Conv3DTrransposeKernel, 1, input_shape, x, filter, nullptr, nullptr, y, workspace, tiling);

    ICPU_SET_TILING_KEY(11011);
    ICPU_RUN_KF(Conv3DTrransposeKernel, 1, input_shape, x, filter, nullptr, nullptr, y, workspace, tiling);

    AscendC::GmFree(input_shape);
    AscendC::GmFree(filter);
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}