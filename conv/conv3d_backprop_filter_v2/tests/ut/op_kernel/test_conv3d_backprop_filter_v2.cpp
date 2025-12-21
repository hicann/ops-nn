/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif
#include "../../../op_kernel/conv3d_backprop_filter_v2.cpp"
#include "../../../op_kernel/arch32/conv3d_backprop_filter_v2_tiling_data.h"
#include <cstdint>

using namespace std;
using namespace AscendC;

class conv3d_backprop_filter_v2_test : public testing::Test {
    protected:
    static void SetUpTestCase() {
        cout << "conv3d_backprop_filter_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase() {
        cout << "conv3d_backprop_filter_v2_test TearDown\n" << endl;
    }
};


TEST_F(conv3d_backprop_filter_v2_test, conv_stdit_01_bf16) {
    size_t shape_x = 1 * 1 * 1 * 32 * 32 * 16 * sizeof(int16_t);
    size_t shape_filter_size = 5 * sizeof(int32_t);
    size_t shape_dedy = 1 * 1 * 72 * 16 * 16 * 16 * sizeof(int16_t);
    size_t shape_y = 1 * 1152 * 1 * 16 * 16 * sizeof(float);
    size_t tiling_data_size = sizeof(Conv3DBackpropFilterV2TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(shape_x);
    uint8_t *filter_size = (uint8_t *)AscendC::GmAlloc(shape_filter_size);
    uint8_t *dedy = (uint8_t *)AscendC::GmAlloc(shape_dedy);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(shape_y);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);

    memset(workspace, 0, 16 * 1024 * 1024);

    system("cp -r ../../../../conv/conv3d_backprop_filter_v2/tests/ut/op_kernel/conv3d_backprop_filter_v2_data ./");
    system("chmod -R 755 ./conv3d_backprop_filter_v2_data/");
    system("cd ./conv3d_backprop_filter_v2_data/ && rm -rf ./*bin");
    system("cd ./conv3d_backprop_filter_v2_data/ && python3 gen_data.py 1 4 1152 1 16 16 1 32 32 1 2 2");
    system("cd ./conv3d_backprop_filter_v2_data/ && python3 gen_tiling.py conv_stdit_01_bf16");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/conv3d_backprop_filter_v2_data/x.bin", shape_x, x, shape_x);
    ReadFile(path + "/conv3d_backprop_filter_v2_data/filter_size.bin", shape_filter_size, filter_size, shape_filter_size);
    ReadFile(path + "/conv3d_backprop_filter_v2_data/dedy.bin", shape_dedy, dedy, shape_dedy);
    ReadFile(path + "/conv3d_backprop_filter_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    auto Conv3DDWKernel = [](GM_ADDR x, GM_ADDR filter_size, GM_ADDR out_backprop, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling) {
        ::conv3d_backprop_filter_v2<0>(x, filter_size, out_backprop, y, workSpace, tiling);
    };
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(Conv3DDWKernel, 24, x, filter_size, dedy, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(filter_size);
    AscendC::GmFree(dedy);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(conv3d_backprop_filter_v2_test, c256_depthwise) {
    size_t shape_x = 1 * 16 * 11 * 64 * 64 * 16 * sizeof(int16_t);
    size_t shape_filter_size = 5 * sizeof(int32_t);
    size_t shape_dedy = 1 * 16 * 9 * 64 * 64 * 16 * sizeof(int16_t);
    size_t shape_y = 432 * 1 * 16 * 16 * sizeof(float);
    size_t tiling_data_size = sizeof(Conv3DBackpropFilterV2TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(shape_x);
    uint8_t *filter_size = (uint8_t *)AscendC::GmAlloc(shape_filter_size);
    uint8_t *dedy = (uint8_t *)AscendC::GmAlloc(shape_dedy);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(shape_y);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);

    memset(workspace, 0, 16 * 1024 * 1024);

    system("cp -r ../../../../conv/conv3d_backprop_filter_v2/tests/ut/op_kernel/conv3d_backprop_filter_v2_data ./");
    system("chmod -R 755 ./conv3d_backprop_filter_v2_data/");
    system("cd ./conv3d_backprop_filter_v2_data/ && rm -rf ./*bin");
    system("cd ./conv3d_backprop_filter_v2_data/ && python3 gen_data.py 1 256 256 9 64 64 11 64 64 3 3 3");
    system("cd ./conv3d_backprop_filter_v2_data/ && python3 gen_tiling.py c256_depthwise");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/conv3d_backprop_filter_v2_data/x.bin", shape_x, x, shape_x);
    ReadFile(path + "/conv3d_backprop_filter_v2_data/filter_size.bin", shape_filter_size, filter_size, shape_filter_size);
    ReadFile(path + "/conv3d_backprop_filter_v2_data/dedy.bin", shape_dedy, dedy, shape_dedy);
    ReadFile(path + "/conv3d_backprop_filter_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    auto Conv3DDWKernel = [](GM_ADDR x, GM_ADDR filter_size, GM_ADDR out_backprop, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling) {
        ::conv3d_backprop_filter_v2<0>(x, filter_size, out_backprop, y, workSpace, tiling);
    };
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(Conv3DDWKernel, 20, x, filter_size, dedy, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(filter_size);
    AscendC::GmFree(dedy);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(conv3d_backprop_filter_v2_test, conv_net_ID_03_b16) {
    size_t shape_x = 4 * 4 * 16 * 64 * 64 * 16 * sizeof(int16_t);
    size_t shape_filter_size = 5 * sizeof(int32_t);
    size_t shape_dedy = 4 * 4 * 86 * 64 * 64 * 16 * sizeof(int16_t);
    size_t shape_y = 16 * 86 * 16 * 16 * sizeof(float);
    size_t tiling_data_size = sizeof(Conv3DBackpropFilterV2TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(shape_x);
    uint8_t *filter_size = (uint8_t *)AscendC::GmAlloc(shape_filter_size);
    uint8_t *dedy = (uint8_t *)AscendC::GmAlloc(shape_dedy);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(shape_y);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);

    memset(workspace, 0, 16 * 1024 * 1024);

    system("cp -r ../../../../conv/conv3d_backprop_filter_v2/tests/ut/op_kernel/conv3d_backprop_filter_v2_data ./");
    system("chmod -R 755 ./conv3d_backprop_filter_v2_data/");
    system("cd ./conv3d_backprop_filter_v2_data/ && rm -rf ./*bin");
    system("cd ./conv3d_backprop_filter_v2_data/ && python3 gen_data.py 4 256 1364 4 64 64 4 64 64 1 1 1");
    system("cd ./conv3d_backprop_filter_v2_data/ && python3 gen_tiling.py conv_net_ID_03_b16");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/conv3d_backprop_filter_v2_data/x.bin", shape_x, x, shape_x);
    ReadFile(path + "/conv3d_backprop_filter_v2_data/filter_size.bin", shape_filter_size, filter_size, shape_filter_size);
    ReadFile(path + "/conv3d_backprop_filter_v2_data/dedy.bin", shape_dedy, dedy, shape_dedy);
    ReadFile(path + "/conv3d_backprop_filter_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    auto Conv3DDWKernel = [](GM_ADDR x, GM_ADDR filter_size, GM_ADDR out_backprop, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling) {
        ::conv3d_backprop_filter_v2<2>(x, filter_size, out_backprop, y, workSpace, tiling);
    };
    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(Conv3DDWKernel, 20, x, filter_size, dedy, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(filter_size);
    AscendC::GmFree(dedy);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(conv3d_backprop_filter_v2_test, conv_03_b16) {
    size_t shape_x = 1 * 17 * 16 * 256 * 256 * 16 * sizeof(int16_t);
    size_t shape_filter_size = 5 * sizeof(int32_t);
    size_t shape_dedy = 1 * 17 * 8 * 256 * 256 * 16 * sizeof(int16_t);
    size_t shape_y = 16 * 8 * 16 * 16 * sizeof(float);
    size_t tiling_data_size = sizeof(Conv3DBackpropFilterV2TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(shape_x);
    uint8_t *filter_size = (uint8_t *)AscendC::GmAlloc(shape_filter_size);
    uint8_t *dedy = (uint8_t *)AscendC::GmAlloc(shape_dedy);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(shape_y);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);

    memset(workspace, 0, 16 * 1024 * 1024);

    system("cp -r ../../../../conv/conv3d_backprop_filter_v2/tests/ut/op_kernel/conv3d_backprop_filter_v2_data ./");
    system("chmod -R 755 ./conv3d_backprop_filter_v2_data/");
    system("cd ./conv3d_backprop_filter_v2_data/ && rm -rf ./*bin");
    system("cd ./conv3d_backprop_filter_v2_data/ && python3 gen_data.py 1 256 128 17 128 128 17 128 128 1 1 1");
    system("cd ./conv3d_backprop_filter_v2_data/ && python3 gen_tiling.py conv_03_b16");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/conv3d_backprop_filter_v2_data/x.bin", shape_x, x, shape_x);
    ReadFile(path + "/conv3d_backprop_filter_v2_data/filter_size.bin", shape_filter_size, filter_size, shape_filter_size);
    ReadFile(path + "/conv3d_backprop_filter_v2_data/dedy.bin", shape_dedy, dedy, shape_dedy);
    ReadFile(path + "/conv3d_backprop_filter_v2_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    auto Conv3DDWKernel = [](GM_ADDR x, GM_ADDR filter_size, GM_ADDR out_backprop, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling) {
        ::conv3d_backprop_filter_v2<1>(x, filter_size, out_backprop, y, workSpace, tiling);
    };
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(Conv3DDWKernel, 24, x, filter_size, dedy, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(filter_size);
    AscendC::GmFree(dedy);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(conv3d_backprop_filter_v2_test, conv_stdit_01_fp32) {
    size_t shape_x = 1 * 1 * 1 * 32 * 32 * 16 * sizeof(int16_t);
    size_t shape_filter_size = 5 * sizeof(int32_t);
    size_t shape_dedy = 1 * 1 * 72 * 16 * 16 * 16 * sizeof(int16_t);
    size_t shape_y = 1 * 1152 * 1 * 16 * 16 * sizeof(float);
    size_t tiling_data_size = sizeof(Conv3DBackpropFilterV2TilingData);

    uint8_t *x = (uint8_t *)AscendC::GmAlloc(shape_x);
    uint8_t *filter_size = (uint8_t *)AscendC::GmAlloc(shape_filter_size);
    uint8_t *dedy = (uint8_t *)AscendC::GmAlloc(shape_dedy);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(shape_y);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size);

    memset(workspace, 0, 16 * 1024 * 1024);

    system("cp -r ../../../../conv/conv3d_backprop_filter_v2/tests/ut/op_kernel/conv3d_backprop_filter_v2_fp32_data ./");
    system("chmod -R 755 ./conv3d_backprop_filter_v2_fp32_data/");
    system("cd ./conv3d_backprop_filter_v2_fp32_data/ && rm -rf ./*bin");
    system("cd ./conv3d_backprop_filter_v2_fp32_data/ && python3 gen_data.py 1 4 1152 1 16 16 1 32 32 1 2 2");
    system("cd ./conv3d_backprop_filter_v2_fp32_data/ && python3 gen_tiling.py conv_stdit_01_fp32");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/conv3d_backprop_filter_v2_fp32_data/x.bin", shape_x, x, shape_x);
    ReadFile(path + "/conv3d_backprop_filter_v2_fp32_data/filter_size.bin", shape_filter_size, filter_size, shape_filter_size);
    ReadFile(path + "/conv3d_backprop_filter_v2_fp32_data/dedy.bin", shape_dedy, dedy, shape_dedy);
    ReadFile(path + "/conv3d_backprop_filter_v2_fp32_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    auto Conv3DDWKernel = [](GM_ADDR x, GM_ADDR filter_size, GM_ADDR out_backprop, GM_ADDR y, GM_ADDR workSpace, GM_ADDR tiling) {
        ::conv3d_backprop_filter_v2<0>(x, filter_size, out_backprop, y, workSpace, tiling);
    };
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(Conv3DDWKernel, 24, x, filter_size, dedy, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(filter_size);
    AscendC::GmFree(dedy);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}