/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

class TEST_AVGPOOL1D_AVG_MATRIX_UT : public testing::Test {};

#define CREATE_NODEDEF_WITH_FORMAT(shapes, dataTypes, datas, ksize, strides, pads, ceilMode, countIncludePad, format) \
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();                                                   \
    NodeDefBuilder(nodeDef.get(), "AvgPool1DAvgMatrix", "AvgPool1DAvgMatrix")                                         \
        .Input({"x", dataTypes[0], shapes[0], datas[0], format})                                                      \
        .Output({"y", dataTypes[1], shapes[1], datas[1], FORMAT_NC1HWC0})                                             \
        .Attr("ksize", ksize)                                                                                         \
        .Attr("strides", strides)                                                                                     \
        .Attr("pads", pads)                                                                                           \
        .Attr("ceil_mode", ceilMode)                                                                                  \
        .Attr("count_include_pad", countIncludePad)

#define CREATE_NODEDEF(shapes, dataTypes, datas, ksize, strides, pads, ceilMode, countIncludePad) \
    CREATE_NODEDEF_WITH_FORMAT(shapes, dataTypes, datas, ksize, strides, pads, ceilMode, countIncludePad, FORMAT_NCHW)

TEST_F(TEST_AVGPOOL1D_AVG_MATRIX_UT, FLOAT_SUCC)
{
    vector<DataType> dataTypes = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1, 1, 1, 4}, {1, 16, 1, 3}};
    float input[4] = {1, 2, 3, 4};
    float output[48] = {0};
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    CREATE_NODEDEF(shapes, dataTypes, datas, 2, 2, std::vector<int64_t>({1, 2}), false, true);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);
    EXPECT_FLOAT_EQ(output[0], 0.5F);
}

TEST_F(TEST_AVGPOOL1D_AVG_MATRIX_UT, NHWC_FLOAT_SUCC)
{
    vector<DataType> dataTypes = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1, 1, 4, 1}, {1, 16, 1, 3}};
    float input[4] = {1, 2, 3, 4};
    float output[48] = {0};
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    CREATE_NODEDEF_WITH_FORMAT(shapes, dataTypes, datas, 2, 2, std::vector<int64_t>({1, 2}), false, true, FORMAT_NHWC);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);
    EXPECT_FLOAT_EQ(output[0], 0.5F);
}

TEST_F(TEST_AVGPOOL1D_AVG_MATRIX_UT, FLOAT16_SUCC)
{
    vector<DataType> dataTypes = {DT_FLOAT16, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{1, 1, 1, 4}, {1, 16, 1, 3}};
    Eigen::half input[4] = {Eigen::half(1.0F), Eigen::half(2.0F), Eigen::half(3.0F), Eigen::half(4.0F)};
    Eigen::half output[48] = {Eigen::half(0.0F)};
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    CREATE_NODEDEF(shapes, dataTypes, datas, 2, 2, std::vector<int64_t>({1, 2}), false, true);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);
    EXPECT_FLOAT_EQ(static_cast<float>(output[0]), 0.5F);
}

TEST_F(TEST_AVGPOOL1D_AVG_MATRIX_UT, DOUBLE_SUCC)
{
    vector<DataType> dataTypes = {DT_DOUBLE, DT_DOUBLE};
    vector<vector<int64_t>> shapes = {{1, 1, 1, 4}, {1, 16, 1, 3}};
    double input[4] = {1.0, 2.0, 3.0, 4.0};
    double output[48] = {0};
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    CREATE_NODEDEF(shapes, dataTypes, datas, 2, 2, std::vector<int64_t>({1, 2}), false, true);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);
    EXPECT_DOUBLE_EQ(output[0], 0.5);
}

TEST_F(TEST_AVGPOOL1D_AVG_MATRIX_UT, INT32_SUCC)
{
    vector<DataType> dataTypes = {DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{1, 1, 1, 4}, {1, 16, 1, 3}};
    int32_t input[4] = {1, 2, 3, 4};
    int32_t output[48] = {0};
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    CREATE_NODEDEF(shapes, dataTypes, datas, 2, 2, std::vector<int64_t>({1, 2}), false, true);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);
    EXPECT_EQ(output[0], 0);
    EXPECT_EQ(output[16], 0);
    EXPECT_EQ(output[32], 0);
}

TEST_F(TEST_AVGPOOL1D_AVG_MATRIX_UT, UINT8_SUCC)
{
    vector<DataType> dataTypes = {DT_UINT8, DT_UINT8};
    vector<vector<int64_t>> shapes = {{1, 1, 1, 4}, {1, 16, 1, 4}};
    uint8_t input[4] = {1, 2, 3, 4};
    uint8_t output[64] = {0};
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    CREATE_NODEDEF(shapes, dataTypes, datas, 1, 1, std::vector<int64_t>({0, 0}), false, true);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);
    EXPECT_EQ(output[0], 1);
}

TEST_F(TEST_AVGPOOL1D_AVG_MATRIX_UT, COUNT_EXCLUDE_PAD_SUCCESS)
{
    vector<DataType> dataTypes = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1, 1, 1, 4}, {1, 16, 1, 3}};
    float input[4] = {1, 2, 3, 4};
    float output[48] = {0};
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    CREATE_NODEDEF(shapes, dataTypes, datas, 2, 2, std::vector<int64_t>({1, 2}), false, false);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);
    EXPECT_FLOAT_EQ(output[0], 1.0F);
    EXPECT_FLOAT_EQ(output[16], 0.5F);
    EXPECT_FLOAT_EQ(output[32], 1.0F);
}

TEST_F(TEST_AVGPOOL1D_AVG_MATRIX_UT, CEIL_MODE_SUCCESS)
{
    vector<DataType> dataTypes = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1, 1, 1, 4}, {1, 16, 1, 2}};
    float input[4] = {1, 2, 3, 4};
    float output[32] = {0};
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    CREATE_NODEDEF(shapes, dataTypes, datas, 3, 3, std::vector<int64_t>({0, 0}), true, true);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);
    EXPECT_FLOAT_EQ(output[0], 1.0F / 3.0F);
    EXPECT_FLOAT_EQ(output[16], 1.0F);
}

TEST_F(TEST_AVGPOOL1D_AVG_MATRIX_UT, STRIDES_ZERO_FAIL)
{
    vector<DataType> dataTypes = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1, 1, 1, 4}, {1, 16, 1, 3}};
    float input[4] = {1, 2, 3, 4};
    float output[48] = {0};
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    CREATE_NODEDEF(shapes, dataTypes, datas, 2, 0, std::vector<int64_t>({1, 2}), false, true);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_AVGPOOL1D_AVG_MATRIX_UT, KSIZE_ZERO_FAIL)
{
    vector<DataType> dataTypes = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1, 1, 1, 4}, {1, 16, 1, 3}};
    float input[4] = {1, 2, 3, 4};
    float output[48] = {0};
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    CREATE_NODEDEF(shapes, dataTypes, datas, 0, 2, std::vector<int64_t>({1, 2}), false, true);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_AVGPOOL1D_AVG_MATRIX_UT, PADS_SIZE_INVALID_FAIL)
{
    vector<DataType> dataTypes = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1, 1, 1, 4}, {1, 16, 1, 3}};
    float input[4] = {1, 2, 3, 4};
    float output[48] = {0};
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    CREATE_NODEDEF(shapes, dataTypes, datas, 2, 2, std::vector<int64_t>({1}), false, true);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_AVGPOOL1D_AVG_MATRIX_UT, FORMAT_INVALID_FAIL)
{
    vector<DataType> dataTypes = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1, 1, 1, 4}, {1, 16, 1, 3}};
    float input[4] = {1, 2, 3, 4};
    float output[48] = {0};
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    CREATE_NODEDEF_WITH_FORMAT(shapes, dataTypes, datas, 2, 2, std::vector<int64_t>({1, 2}), false, true, FORMAT_ND);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_AVGPOOL1D_AVG_MATRIX_UT, DEGENERATE_WINDOW_FAIL)
{
    vector<DataType> dataTypes = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1, 1, 1, 4}, {1, 16, 1, 2}};
    float input[4] = {1, 2, 3, 4};
    float output[32] = {0};
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    CREATE_NODEDEF(shapes, dataTypes, datas, 2, 5, std::vector<int64_t>({0, 0}), true, false);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_AVGPOOL1D_AVG_MATRIX_UT, OUTPUT_SHAPE_TOO_SMALL_FAIL)
{
    vector<DataType> dataTypes = {DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1, 1, 1, 4}, {1, 16, 1, 2}};
    float input[4] = {1, 2, 3, 4};
    float output[32] = {0};
    vector<void*> datas = {static_cast<void*>(input), static_cast<void*>(output)};
    CREATE_NODEDEF(shapes, dataTypes, datas, 2, 2, std::vector<int64_t>({1, 2}), false, true);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}
