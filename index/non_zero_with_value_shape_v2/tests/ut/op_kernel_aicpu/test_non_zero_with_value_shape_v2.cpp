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

class TEST_NON_ZERO_WITH_VALUE_SHAPE_V2_UT : public testing::Test {};

template <typename T>
void CalcExpectFunc(const NodeDef& nodeDef, T& dims0, T& dims1)
{
    auto shape0 = nodeDef.MutableOutputs(0)->GetTensorShape();
    auto shape1 = nodeDef.MutableOutputs(1)->GetTensorShape();
    dims0 = shape0->GetDimSizes();
    dims1 = shape1->GetDimSizes();
}

#define CREATE_NODEDEF(shapes, dataTypes, datas)                                        \
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();                     \
    NodeDefBuilder(nodeDef.get(), "NonZeroWithValueShapeV2", "NonZeroWithValueShapeV2") \
        .Input({"value", (dataTypes)[0], (shapes)[0], (datas)[0]})                      \
        .Input({"index", (dataTypes)[1], (shapes)[1], (datas)[1]})                      \
        .Input({"count", (dataTypes)[2], (shapes)[2], (datas)[2]})                      \
        .Output({"value", (dataTypes)[3], (shapes)[3], (datas)[3]})                     \
        .Output({"index", (dataTypes)[4], (shapes)[4], (datas)[4]})

TEST_F(TEST_NON_ZERO_WITH_VALUE_SHAPE_V2_UT, DataSuccess)
{
    vector<DataType> dataTypes = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2}, {4}, {1}, {}, {}};

    int32_t input0[2] = {1, 2};
    int32_t input1[4] = {1, 2, 3, 4};
    int32_t input2[1] = {2};
    int32_t output0[2] = {1, 2};
    int32_t output1[4] = {1, 2, 3, 4};
    vector<void*> datas = {static_cast<void*>(input0), static_cast<void*>(input1), static_cast<void*>(input2),
                           static_cast<void*>(output0), static_cast<void*>(output1)};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);
    std::vector<int64_t> dims0;
    std::vector<int64_t> dims1;
    CalcExpectFunc(*nodeDef.get(), dims0, dims1);
    std::vector<int64_t> outputExp0ShapeDims = {2};
    std::vector<int64_t> outputExp1ShapeDims = {2, 2};
    EXPECT_EQ(dims0, outputExp0ShapeDims);
    EXPECT_EQ(dims1, outputExp1ShapeDims);
}

TEST_F(TEST_NON_ZERO_WITH_VALUE_SHAPE_V2_UT, IndexDtypeInvalid)
{
    vector<DataType> dataTypes = {DT_INT32, DT_FLOAT, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2}, {4}, {1}, {}, {}};

    int32_t input0[2] = {1, 2};
    int32_t input1[4] = {1, 2, 3, 4};
    int32_t input2[1] = {2};
    int32_t output0[2] = {1, 2};
    int32_t output1[4] = {1, 2, 3, 4};
    vector<void*> datas = {static_cast<void*>(input0), static_cast<void*>(input1), static_cast<void*>(input2),
                           static_cast<void*>(output0), static_cast<void*>(output1)};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_NON_ZERO_WITH_VALUE_SHAPE_V2_UT, CountDtypeInvalid)
{
    vector<DataType> dataTypes = {DT_INT32, DT_INT32, DT_FLOAT, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2}, {4}, {1}, {}, {}};

    int32_t input0[2] = {1, 2};
    int32_t input1[4] = {1, 2, 3, 4};
    int32_t input2[1] = {2};
    int32_t output0[2] = {1, 2};
    int32_t output1[4] = {1, 2, 3, 4};
    vector<void*> datas = {static_cast<void*>(input0), static_cast<void*>(input1), static_cast<void*>(input2),
                           static_cast<void*>(output0), static_cast<void*>(output1)};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_NON_ZERO_WITH_VALUE_SHAPE_V2_UT, CountEmptyInvalid)
{
    vector<DataType> dataTypes = {DT_INT32, DT_INT32, DT_INT32, DT_INT32, DT_INT32};
    vector<vector<int64_t>> shapes = {{2}, {4}, {0}, {}, {}};

    int32_t input0[2] = {1, 2};
    int32_t input1[4] = {1, 2, 3, 4};
    int32_t output0[2] = {1, 2};
    int32_t output1[4] = {1, 2, 3, 4};
    vector<void*> datas = {static_cast<void*>(input0), static_cast<void*>(input1), nullptr, static_cast<void*>(output0),
                           static_cast<void*>(output1)};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}
