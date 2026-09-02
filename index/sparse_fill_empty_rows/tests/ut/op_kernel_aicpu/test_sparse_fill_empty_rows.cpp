/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "utils/aicpu_test_utils.h"
#undef private
#undef protected

#include <complex>
#include <cstdint>
#include <vector>

#include "unsupported/Eigen/CXX11/Tensor"

using namespace std;
using namespace aicpu;

class TEST_SPARSE_FILL_EMPTY_ROWS_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, dataTypes, datas)                                  \
    auto nodeDef = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();               \
    NodeDefBuilder(nodeDef.get(), "SparseFillEmptyRows", "SparseFillEmptyRows")   \
        .Input({"indices", (dataTypes)[0], (shapes)[0], (datas)[0]})              \
        .Input({"values", (dataTypes)[1], (shapes)[1], (datas)[1]})               \
        .Input({"dense_shape", (dataTypes)[2], (shapes)[2], (datas)[2]})          \
        .Input({"default_value", (dataTypes)[3], (shapes)[3], (datas)[3]})        \
        .Output({"y_indices", (dataTypes)[4], (shapes)[4], (datas)[4]})           \
        .Output({"y_values", (dataTypes)[5], (shapes)[5], (datas)[5]})            \
        .Output({"empty_row_indicator", (dataTypes)[6], (shapes)[6], (datas)[6]}) \
        .Output({"reverse_index_map", (dataTypes)[7], (shapes)[7], (datas)[7]})

#define ADD_FILL_EMPTY_ROW_CASE(aicpuType, baseType)                                                               \
    TEST_F(TEST_SPARSE_FILL_EMPTY_ROWS_UT, fill_empty_row_##aicpuType)                                             \
    {                                                                                                              \
        vector<DataType> dataTypes = {                                                                             \
            DT_INT64, aicpuType, DT_INT64, aicpuType, DT_INT64, aicpuType, DT_BOOL, DT_INT64};                     \
        vector<vector<int64_t>> shapes = {{5, 2}, {5}, {2}, {}, {6, 2}, {6}, {5}, {5}};                            \
        int64_t inputIndices[10] = {0, 1, 0, 2, 2, 3, 3, 4, 4, 5};                                                 \
        baseType inputValues[5] = {static_cast<baseType>(0), static_cast<baseType>(1), static_cast<baseType>(2),   \
                                   static_cast<baseType>(3), static_cast<baseType>(4)};                            \
        int64_t denseShape[2] = {5, 6};                                                                            \
        baseType defaultValue[1] = {static_cast<baseType>(77)};                                                    \
        int64_t yIndices[12] = {0};                                                                                \
        baseType yValues[6] = {static_cast<baseType>(0)};                                                          \
        bool emptyRowIndicator[5] = {false};                                                                       \
        int64_t reverseIndexMap[5] = {0};                                                                          \
        vector<void*> datas = {static_cast<void*>(inputIndices),      static_cast<void*>(inputValues),             \
                               static_cast<void*>(denseShape),        static_cast<void*>(defaultValue),            \
                               static_cast<void*>(yIndices),          static_cast<void*>(yValues),                 \
                               static_cast<void*>(emptyRowIndicator), static_cast<void*>(reverseIndexMap)};        \
        CREATE_NODEDEF(shapes, dataTypes, datas);                                                                  \
        RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);                                                               \
        int64_t expectIndices[12] = {0, 1, 0, 2, 1, 0, 2, 3, 3, 4, 4, 5};                                          \
        baseType expectValues[6] = {static_cast<baseType>(0), static_cast<baseType>(1), static_cast<baseType>(77), \
                                    static_cast<baseType>(2), static_cast<baseType>(3), static_cast<baseType>(4)}; \
        bool expectEmptyRows[5] = {false, true, false, false, false};                                              \
        int64_t expectReverseIndexMap[5] = {0, 1, 3, 4, 5};                                                        \
        EXPECT_EQ(CompareResult<int64_t>(yIndices, expectIndices, 12), true);                                      \
        EXPECT_EQ(CompareResult<baseType>(yValues, expectValues, 6), true);                                        \
        EXPECT_EQ(CompareResult<bool>(emptyRowIndicator, expectEmptyRows, 5), true);                               \
        EXPECT_EQ(CompareResult<int64_t>(reverseIndexMap, expectReverseIndexMap, 5), true);                        \
    }

ADD_FILL_EMPTY_ROW_CASE(DT_BOOL, bool)
ADD_FILL_EMPTY_ROW_CASE(DT_COMPLEX128, std::complex<double>)
ADD_FILL_EMPTY_ROW_CASE(DT_COMPLEX64, std::complex<float>)
ADD_FILL_EMPTY_ROW_CASE(DT_DOUBLE, double)
ADD_FILL_EMPTY_ROW_CASE(DT_FLOAT, float)
ADD_FILL_EMPTY_ROW_CASE(DT_FLOAT16, Eigen::half)
ADD_FILL_EMPTY_ROW_CASE(DT_INT16, int16_t)
ADD_FILL_EMPTY_ROW_CASE(DT_INT32, int32_t)
ADD_FILL_EMPTY_ROW_CASE(DT_INT64, int64_t)
ADD_FILL_EMPTY_ROW_CASE(DT_INT8, int8_t)
ADD_FILL_EMPTY_ROW_CASE(DT_UINT16, uint16_t)
ADD_FILL_EMPTY_ROW_CASE(DT_UINT32, uint32_t)
ADD_FILL_EMPTY_ROW_CASE(DT_UINT64, uint64_t)
ADD_FILL_EMPTY_ROW_CASE(DT_UINT8, uint8_t)

TEST_F(TEST_SPARSE_FILL_EMPTY_ROWS_UT, geir_sample_success)
{
    vector<DataType> dataTypes = {DT_INT64, DT_INT32, DT_INT64, DT_INT32, DT_INT64, DT_INT32, DT_BOOL, DT_INT64};
    vector<vector<int64_t>> shapes = {{2, 2}, {2}, {2}, {}, {3, 2}, {3}, {3}, {2}};
    int64_t inputIndices[4] = {0, 1, 2, 3};
    int32_t inputValues[2] = {10, 20};
    int64_t denseShape[2] = {3, 4};
    int32_t defaultValue[1] = {1};
    int64_t yIndices[6] = {0};
    int32_t yValues[3] = {0};
    bool emptyRowIndicator[3] = {false};
    int64_t reverseIndexMap[2] = {0};
    vector<void*> datas = {inputIndices, inputValues, denseShape,        defaultValue,
                           yIndices,     yValues,     emptyRowIndicator, reverseIndexMap};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);

    int64_t expectIndices[6] = {0, 1, 1, 0, 2, 3};
    int32_t expectValues[3] = {10, 1, 20};
    bool expectEmptyRows[3] = {false, true, false};
    int64_t expectReverseIndexMap[2] = {0, 2};
    EXPECT_EQ(CompareResult<int64_t>(yIndices, expectIndices, 6), true);
    EXPECT_EQ(CompareResult<int32_t>(yValues, expectValues, 3), true);
    EXPECT_EQ(CompareResult<bool>(emptyRowIndicator, expectEmptyRows, 3), true);
    EXPECT_EQ(CompareResult<int64_t>(reverseIndexMap, expectReverseIndexMap, 2), true);
}

#define ADD_ALL_ROWS_FULL_CASE(aicpuType, baseType)                                                              \
    TEST_F(TEST_SPARSE_FILL_EMPTY_ROWS_UT, all_rows_full_##aicpuType)                                            \
    {                                                                                                            \
        vector<DataType> dataTypes = {                                                                           \
            DT_INT64, aicpuType, DT_INT64, aicpuType, DT_INT64, aicpuType, DT_BOOL, DT_INT64};                   \
        vector<vector<int64_t>> shapes = {{5, 2}, {5}, {2}, {}, {5, 2}, {5}, {5}, {5}};                          \
        int64_t inputIndices[10] = {0, 1, 1, 2, 2, 3, 3, 4, 4, 5};                                               \
        baseType inputValues[5] = {static_cast<baseType>(0), static_cast<baseType>(1), static_cast<baseType>(2), \
                                   static_cast<baseType>(3), static_cast<baseType>(4)};                          \
        int64_t denseShape[2] = {5, 6};                                                                          \
        baseType defaultValue[1] = {static_cast<baseType>(77)};                                                  \
        int64_t yIndices[10] = {0};                                                                              \
        baseType yValues[5] = {static_cast<baseType>(0)};                                                        \
        bool emptyRowIndicator[5] = {false};                                                                     \
        int64_t reverseIndexMap[5] = {0};                                                                        \
        vector<void*> datas = {inputIndices, inputValues, denseShape,        defaultValue,                       \
                               yIndices,     yValues,     emptyRowIndicator, reverseIndexMap};                   \
        CREATE_NODEDEF(shapes, dataTypes, datas);                                                                \
        RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);                                                             \
        bool expectEmptyRows[5] = {false, false, false, false, false};                                           \
        int64_t expectReverseIndexMap[5] = {0, 1, 2, 3, 4};                                                      \
        EXPECT_EQ(CompareResult<int64_t>(yIndices, inputIndices, 10), true);                                     \
        EXPECT_EQ(CompareResult<baseType>(yValues, inputValues, 5), true);                                       \
        EXPECT_EQ(CompareResult<bool>(emptyRowIndicator, expectEmptyRows, 5), true);                             \
        EXPECT_EQ(CompareResult<int64_t>(reverseIndexMap, expectReverseIndexMap, 5), true);                      \
    }

ADD_ALL_ROWS_FULL_CASE(DT_BOOL, bool)
ADD_ALL_ROWS_FULL_CASE(DT_COMPLEX128, std::complex<double>)
ADD_ALL_ROWS_FULL_CASE(DT_COMPLEX64, std::complex<float>)
ADD_ALL_ROWS_FULL_CASE(DT_DOUBLE, double)
ADD_ALL_ROWS_FULL_CASE(DT_FLOAT, float)
ADD_ALL_ROWS_FULL_CASE(DT_FLOAT16, Eigen::half)
ADD_ALL_ROWS_FULL_CASE(DT_INT16, int16_t)
ADD_ALL_ROWS_FULL_CASE(DT_INT32, int32_t)
ADD_ALL_ROWS_FULL_CASE(DT_INT64, int64_t)
ADD_ALL_ROWS_FULL_CASE(DT_INT8, int8_t)
ADD_ALL_ROWS_FULL_CASE(DT_UINT16, uint16_t)
ADD_ALL_ROWS_FULL_CASE(DT_UINT32, uint32_t)
ADD_ALL_ROWS_FULL_CASE(DT_UINT64, uint64_t)
ADD_ALL_ROWS_FULL_CASE(DT_UINT8, uint8_t)

TEST_F(TEST_SPARSE_FILL_EMPTY_ROWS_UT, dense_rows_negative_fail)
{
    vector<DataType> dataTypes = {DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_BOOL, DT_INT64};
    vector<vector<int64_t>> shapes = {{0, 2}, {0}, {2}, {}, {0, 2}, {0}, {0}, {0}};
    int64_t inputIndices[1] = {0};
    int64_t inputValues[1] = {0};
    int64_t denseShape[2] = {-1, 6};
    int64_t defaultValue[1] = {-1};
    int64_t yIndices[1] = {0};
    int64_t yValues[1] = {0};
    bool emptyRowIndicator[1] = {false};
    int64_t reverseIndexMap[1] = {0};
    vector<void*> datas = {inputIndices, inputValues, denseShape,        defaultValue,
                           yIndices,     yValues,     emptyRowIndicator, reverseIndexMap};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSE_FILL_EMPTY_ROWS_UT, dense_rows_zero_with_nonempty_indices_fail)
{
    vector<DataType> dataTypes = {DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_BOOL, DT_INT64};
    vector<vector<int64_t>> shapes = {{5, 2}, {5}, {2}, {}, {6, 2}, {6}, {5}, {5}};
    int64_t inputIndices[10] = {0, 1, 0, 2, 2, 3, 3, 4, 4, 5};
    int64_t inputValues[5] = {0, 1, 2, 3, 4};
    int64_t denseShape[2] = {0, 6};
    int64_t defaultValue[1] = {-1};
    int64_t yIndices[12] = {0};
    int64_t yValues[6] = {0};
    bool emptyRowIndicator[5] = {false};
    int64_t reverseIndexMap[5] = {0};
    vector<void*> datas = {inputIndices, inputValues, denseShape,        defaultValue,
                           yIndices,     yValues,     emptyRowIndicator, reverseIndexMap};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSE_FILL_EMPTY_ROWS_UT, dense_rows_zero_with_empty_indices_success)
{
    vector<DataType> dataTypes = {DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_BOOL, DT_INT64};
    vector<vector<int64_t>> shapes = {{0, 2}, {0}, {2}, {}, {0, 2}, {0}, {0}, {0}};
    int64_t inputIndices[1] = {0};
    int64_t inputValues[1] = {0};
    int64_t denseShape[2] = {0, 6};
    int64_t defaultValue[1] = {-1};
    int64_t yIndices[1] = {0};
    int64_t yValues[1] = {0};
    bool emptyRowIndicator[1] = {false};
    int64_t reverseIndexMap[1] = {0};
    vector<void*> datas = {inputIndices, inputValues, denseShape,        defaultValue,
                           yIndices,     yValues,     emptyRowIndicator, reverseIndexMap};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_OK);
}

TEST_F(TEST_SPARSE_FILL_EMPTY_ROWS_UT, row_index_out_of_range_fail)
{
    vector<DataType> dataTypes = {DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_BOOL, DT_INT64};
    vector<vector<int64_t>> shapes = {{5, 2}, {5}, {2}, {}, {6, 2}, {6}, {5}, {5}};
    int64_t inputIndices[10] = {6, 1, 0, 2, 2, 3, 3, 4, 4, 5};
    int64_t inputValues[5] = {0, 1, 2, 3, 4};
    int64_t denseShape[2] = {5, 6};
    int64_t defaultValue[1] = {-1};
    int64_t yIndices[12] = {0};
    int64_t yValues[6] = {0};
    bool emptyRowIndicator[5] = {false};
    int64_t reverseIndexMap[5] = {0};
    vector<void*> datas = {inputIndices, inputValues, denseShape,        defaultValue,
                           yIndices,     yValues,     emptyRowIndicator, reverseIndexMap};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPARSE_FILL_EMPTY_ROWS_UT, unsupported_dtype_fail)
{
    vector<DataType> dataTypes = {DT_INT64, DT_UNDEFINED, DT_INT64, DT_INT64, DT_INT64, DT_INT64, DT_BOOL, DT_INT64};
    vector<vector<int64_t>> shapes = {{5, 2}, {5}, {2}, {}, {6, 2}, {6}, {5}, {5}};
    int64_t inputIndices[10] = {0, 1, 0, 2, 2, 3, 3, 4, 4, 5};
    int64_t inputValues[5] = {0, 1, 2, 3, 4};
    int64_t denseShape[2] = {5, 6};
    int64_t defaultValue[1] = {-1};
    int64_t yIndices[12] = {0};
    int64_t yValues[6] = {0};
    bool emptyRowIndicator[5] = {false};
    int64_t reverseIndexMap[5] = {0};
    vector<void*> datas = {inputIndices, inputValues, denseShape,        defaultValue,
                           yIndices,     yValues,     emptyRowIndicator, reverseIndexMap};
    CREATE_NODEDEF(shapes, dataTypes, datas);
    RUN_KERNEL(nodeDef, HOST, KERNEL_STATUS_PARAM_INVALID);
}
