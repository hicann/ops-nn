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

using namespace std;
using namespace aicpu;
namespace {
constexpr int64_t kExpectedOutputElementCount = 16;
}
class TEST_INDEX_TO_ADDR_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, dataTypes, datas)                          \
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();       \
    NodeDefBuilder(nodeDef.get(), "IndexToAddr", "IndexToAddr")           \
        .Input({"base_addr", (dataTypes)[0], (shapes)[0], (datas)[0]})    \
        .Input({"x", (dataTypes)[1], (shapes)[1], (datas)[1]})            \
        .Output({"addrs_table", (dataTypes)[2], (shapes)[2], (datas)[2]}) \
        .Attr("ori_shape", oriShape)                                      \
        .Attr("ori_storage_mode", std::string("Matrix"))                  \
        .Attr("block_size", blockSize)                                    \
        .Attr("block_storage_mode", std::string("Matrix"))                \
        .Attr("rank_id", 0)                                               \
        .Attr("dtype", DT_FLOAT)

TEST_F(TEST_INDEX_TO_ADDR_UT, Row0Col3Success)
{
    vector<DataType> dataTypes = {DT_INT64, DT_INT64, DT_INT64};
    vector<vector<int64_t>> shapes = {{2}, {2}, {4, 4}};
    int64_t input0[2] = {20, 40};
    int64_t input1[2] = {0, 3};
    int64_t output[16] = {0};
    vector<void*> datas = {static_cast<void*>(input0), static_cast<void*>(input1), static_cast<void*>(output)};
    vector<int64_t> oriShape = {16, 16};
    vector<int64_t> blockSize = {4, 4};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);
    int64_t outputExp[16] = {0, 68, 88, 16, 0, 132, 152, 16, 0, 196, 216, 16, 0, 260, 280, 16};
    EXPECT_EQ(CompareResult(output, outputExp, kExpectedOutputElementCount), true);
}

TEST_F(TEST_INDEX_TO_ADDR_UT, Row1Col1Success)
{
    vector<DataType> dataTypes = {DT_INT64, DT_INT64, DT_INT64};
    vector<vector<int64_t>> shapes = {{2}, {2}, {4, 4}};
    int64_t input0[2] = {20, 40};
    int64_t input1[2] = {1, 1};
    int64_t output[16] = {0};
    vector<void*> datas = {static_cast<void*>(input0), static_cast<void*>(input1), static_cast<void*>(output)};
    vector<int64_t> oriShape = {16, 16};
    vector<int64_t> blockSize = {4, 4};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);
    int64_t outputExp[16] = {0, 292, 312, 16, 0, 356, 376, 16, 0, 420, 440, 16, 0, 484, 504, 16};
    EXPECT_EQ(CompareResult(output, outputExp, kExpectedOutputElementCount), true);
}

TEST_F(TEST_INDEX_TO_ADDR_UT, DtypeMismatchFail)
{
    vector<DataType> dataTypes = {DT_INT64, DT_UINT64, DT_INT64};
    vector<vector<int64_t>> shapes = {{2}, {2}, {4, 4}};
    int64_t input0[2] = {20, 40};
    uint64_t input1[2] = {1, 1};
    int64_t output[16] = {0};
    vector<void*> datas = {static_cast<void*>(input0), static_cast<void*>(input1), static_cast<void*>(output)};
    vector<int64_t> oriShape = {16, 16};
    vector<int64_t> blockSize = {4, 4};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_INDEX_TO_ADDR_UT, ShapeMismatchFail)
{
    vector<DataType> dataTypes = {DT_INT64, DT_INT64, DT_INT64};
    vector<vector<int64_t>> shapes = {{2}, {3}, {4, 4}};
    int64_t input0[2] = {20, 40};
    int64_t input1[3] = {1, 1, 1};
    int64_t output[16] = {0};
    vector<void*> datas = {static_cast<void*>(input0), static_cast<void*>(input1), static_cast<void*>(output)};
    vector<int64_t> oriShape = {16, 16};
    vector<int64_t> blockSize = {4, 4};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_INDEX_TO_ADDR_UT, UnsupportedDtypeFail)
{
    vector<DataType> dataTypes = {DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2}, {2}, {4, 4}};
    int32_t input0[2] = {20, 40};
    int32_t input1[2] = {1, 1};
    int32_t output[16] = {0};
    vector<void*> datas = {static_cast<void*>(input0), static_cast<void*>(input1), static_cast<void*>(output)};
    vector<int64_t> oriShape = {16, 16};
    vector<int64_t> blockSize = {4, 4};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}
