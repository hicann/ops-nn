/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <cstring>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../../op_kernel/arch35/data_format_dim_map_tiling_data.h"

namespace DataFormatDimMapUT {
using namespace std;
using namespace ge;
using namespace gert;

static const std::string OP_NAME = "DataFormatDimMap";

struct DataFormatDimMapCompileInfo {
} compileInfo;

// Helper: build TilingContextPara with string ATTR for src_format/dst_format
static gert::TilingContextPara MakeTilingPara(
    std::initializer_list<int64_t> xShape,
    ge::DataType dtype,
    const char* srcFormat,
    const char* dstFormat,
    uint64_t coreNum = 20,
    uint64_t ubSize = 253952,
    uint64_t tilingDataSize = 4096)
{
    gert::StorageShape xStorageShape = {xShape, xShape};
    gert::StorageShape yStorageShape = {xShape, xShape};

    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc({
        {xStorageShape, dtype, ge::FORMAT_ND}
    });
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc({
        {yStorageShape, dtype, ge::FORMAT_ND}
    });

    std::vector<gert::TilingContextPara::OpAttr> attrs({
        {"src_format", Ops::NN::AnyValue::CreateFrom<std::string>(std::string(srcFormat))},
        {"dst_format", Ops::NN::AnyValue::CreateFrom<std::string>(std::string(dstFormat))}
    });

    return gert::TilingContextPara(
        OP_NAME, inputTensorDesc, outputTensorDesc, attrs,
        &compileInfo, coreNum, ubSize, tilingDataSize);
}

// Helper: build expected expanded table for verification
static void BuildExpectedTable(const char* srcFormat, const char* dstFormat,
                               int32_t* table, int32_t& formatLen) {
    formatLen = static_cast<int32_t>(strlen(srcFormat));
    for (int i = 0; i < formatLen; i++) {
        table[i] = 0;
        for (int j = 0; j < formatLen; j++) {
            if (dstFormat[j] == srcFormat[i]) {
                table[i] = j;
                break;
            }
        }
    }
    for (int i = 0; i < formatLen; i++) {
        table[formatLen + i] = table[i];
    }
}

class DataFormatDimMapTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "DataFormatDimMapTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "DataFormatDimMapTilingTest TearDown." << std::endl;
    }
};

// ============================================================================
// P0: Mapping table construction - NHWC -> NCHW
// ============================================================================
TEST_F(DataFormatDimMapTilingTest, test_nhwc_to_nchw_mapping_table)
{
    auto para = MakeTilingPara({8}, ge::DT_INT32, "NHWC", "NCHW");
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    auto* td = reinterpret_cast<const DataFormatDimMapTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td->formatLen, 4);
    // NHWC -> NCHW: N->0, H->2, W->3, C->1
    int32_t expectedTable[10] = {0, 2, 3, 1, 0, 2, 3, 1, 0, 0};
    for (int i = 0; i < 8; i++) {
        EXPECT_EQ(td->expandedTable[i], expectedTable[i])
            << "expandedTable[" << i << "] mismatch";
    }
}

// ============================================================================
// P0: 5-dim format mapping - NDHWC -> NCDHW
// ============================================================================
TEST_F(DataFormatDimMapTilingTest, test_ndhwc_to_ncdhw_mapping_table)
{
    auto para = MakeTilingPara({16}, ge::DT_INT32, "NDHWC", "NCDHW");
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    auto* td = reinterpret_cast<const DataFormatDimMapTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td->formatLen, 5);
    int32_t expectedTable[10] = {0, 2, 3, 4, 1, 0, 2, 3, 4, 1};
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(td->expandedTable[i], expectedTable[i])
            << "expandedTable[" << i << "] mismatch";
    }
}

// ============================================================================
// P0: Small data
// ============================================================================
TEST_F(DataFormatDimMapTilingTest, test_small_data)
{
    auto para = MakeTilingPara({100}, ge::DT_INT32, "NHWC", "NCHW");
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    auto* td = reinterpret_cast<const DataFormatDimMapTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td->totalNum, 100);
    EXPECT_GT(td->blockFactor, 0);
    EXPECT_GT(td->ubFactor, 0);
}

// ============================================================================
// P0: Large data
// ============================================================================
TEST_F(DataFormatDimMapTilingTest, test_large_data)
{
    auto para = MakeTilingPara({2048}, ge::DT_INT32, "NHWC", "NCHW");
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    auto* td = reinterpret_cast<const DataFormatDimMapTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td->totalNum, 2048);
    EXPECT_GT(td->blockFactor, 0);
    EXPECT_GT(td->ubFactor, 0);
}

// ============================================================================
// P0: Boundary at 1024
// ============================================================================
TEST_F(DataFormatDimMapTilingTest, test_boundary_1024)
{
    auto para = MakeTilingPara({1024}, ge::DT_INT32, "NHWC", "NCHW");
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    auto* td = reinterpret_cast<const DataFormatDimMapTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td->totalNum, 1024);
}

// ============================================================================
// P0: Same tilingKey for same dtype regardless of data size
// ============================================================================
TEST_F(DataFormatDimMapTilingTest, test_same_tiling_key_for_same_dtype)
{
    auto para1 = MakeTilingPara({1}, ge::DT_INT32, "NHWC", "NCHW");
    auto para1024 = MakeTilingPara({1024}, ge::DT_INT32, "NHWC", "NCHW");
    auto para100000 = MakeTilingPara({100000}, ge::DT_INT32, "NHWC", "NCHW");
    TilingInfo info1, info1024, info100000;
    ASSERT_TRUE(ExecuteTiling(para1, info1));
    ASSERT_TRUE(ExecuteTiling(para1024, info1024));
    ASSERT_TRUE(ExecuteTiling(para100000, info100000));

    EXPECT_EQ(info1.tilingKey, info1024.tilingKey);
    EXPECT_EQ(info1024.tilingKey, info100000.tilingKey);
}

// ============================================================================
// P0: Empty tensor (totalNum=0)
// ============================================================================
TEST_F(DataFormatDimMapTilingTest, test_empty_tensor)
{
    auto para = MakeTilingPara({0}, ge::DT_INT32, "NHWC", "NCHW");
    TilingInfo tilingInfo;
    bool result = ExecuteTiling(para, tilingInfo);
    if (result) {
        auto* td = reinterpret_cast<const DataFormatDimMapTilingData*>(tilingInfo.tilingData.get());
        EXPECT_EQ(td->totalNum, 0);
    }
}

// ============================================================================
// P0: Multi-dimensional shape
// ============================================================================
TEST_F(DataFormatDimMapTilingTest, test_multidim_shape)
{
    auto para = MakeTilingPara({4, 8}, ge::DT_INT32, "NHWC", "NCHW");
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    auto* td = reinterpret_cast<const DataFormatDimMapTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td->totalNum, 32);
}

// ============================================================================
// P0: NCHW -> NHWC (reverse mapping)
// ============================================================================
TEST_F(DataFormatDimMapTilingTest, test_nchw_to_nhwc_mapping_table)
{
    auto para = MakeTilingPara({8}, ge::DT_INT32, "NCHW", "NHWC");
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    auto* td = reinterpret_cast<const DataFormatDimMapTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td->formatLen, 4);
    int32_t expectedTable[8] = {0, 3, 1, 2, 0, 3, 1, 2};
    for (int i = 0; i < 8; i++) {
        EXPECT_EQ(td->expandedTable[i], expectedTable[i])
            << "expandedTable[" << i << "] mismatch";
    }
}

// ============================================================================
// P0: blockFactor alignment and multi-core split
// ============================================================================
TEST_F(DataFormatDimMapTilingTest, test_blockfactor_multicore)
{
    auto para = MakeTilingPara({10000}, ge::DT_INT32, "NHWC", "NCHW", 20);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    auto* td = reinterpret_cast<const DataFormatDimMapTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td->totalNum, 10000);
    EXPECT_GE(td->blockFactor, 500);
    std::cout << "blockFactor=" << td->blockFactor << std::endl;
}

// ============================================================================
// Iteration 2: Boundary and data size coverage
// ============================================================================

TEST_F(DataFormatDimMapTilingTest, test_boundary_1023_tiling)
{
    auto para = MakeTilingPara({1023}, ge::DT_INT32, "NHWC", "NCHW");
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    auto* td = reinterpret_cast<const DataFormatDimMapTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td->totalNum, 1023);
    EXPECT_GT(td->ubFactor, 0);

    auto para1024 = MakeTilingPara({1024}, ge::DT_INT32, "NHWC", "NCHW");
    TilingInfo tilingInfo1024;
    ASSERT_TRUE(ExecuteTiling(para1024, tilingInfo1024));

    EXPECT_EQ(tilingInfo.tilingKey, tilingInfo1024.tilingKey)
        << "Same dtype should produce the same tilingKey regardless of data size";
}

TEST_F(DataFormatDimMapTilingTest, test_boundary_1024_tiling)
{
    auto para = MakeTilingPara({1024}, ge::DT_INT32, "NHWC", "NCHW");
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    auto* td = reinterpret_cast<const DataFormatDimMapTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td->totalNum, 1024);
    EXPECT_GT(td->ubFactor, 0);
}

TEST_F(DataFormatDimMapTilingTest, test_tiny_data_same_key)
{
    auto para = MakeTilingPara({1}, ge::DT_INT32, "NHWC", "NCHW");
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    auto* td = reinterpret_cast<const DataFormatDimMapTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td->totalNum, 1);

    auto para1023 = MakeTilingPara({1023}, ge::DT_INT32, "NHWC", "NCHW");
    TilingInfo tilingInfo1023;
    ASSERT_TRUE(ExecuteTiling(para1023, tilingInfo1023));
    EXPECT_EQ(tilingInfo.tilingKey, tilingInfo1023.tilingKey)
        << "Same dtype should produce the same tilingKey regardless of data size";
}

// ============================================================================
// Iteration 2: Multi-format length branch coverage
// ============================================================================

TEST_F(DataFormatDimMapTilingTest, test_n4_format_expanded_table_size)
{
    auto para = MakeTilingPara({256}, ge::DT_INT32, "NHWC", "NCHW");
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    auto* td = reinterpret_cast<const DataFormatDimMapTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td->formatLen, 4);
    EXPECT_EQ(td->expandedTable[8], 0);
    EXPECT_EQ(td->expandedTable[9], 0);
}

TEST_F(DataFormatDimMapTilingTest, test_n5_format_expanded_table_size)
{
    auto para = MakeTilingPara({256}, ge::DT_INT32, "NDHWC", "NCDHW");
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    auto* td = reinterpret_cast<const DataFormatDimMapTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td->formatLen, 5);
    int32_t expected[10] = {0, 2, 3, 4, 1, 0, 2, 3, 4, 1};
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(td->expandedTable[i], expected[i])
            << "expandedTable[" << i << "] mismatch for N=5 format";
    }
}

TEST_F(DataFormatDimMapTilingTest, test_n5_format_large_data)
{
    auto para = MakeTilingPara({4096}, ge::DT_INT32, "NDHWC", "NCDHW");
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    auto* td = reinterpret_cast<const DataFormatDimMapTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td->totalNum, 4096);
    EXPECT_EQ(td->formatLen, 5);
    EXPECT_GT(td->ubFactor, 0);
    EXPECT_GT(td->blockFactor, 0);
}

TEST_F(DataFormatDimMapTilingTest, test_n4_n5_same_tiling_key_different_formatlen)
{
    auto para4 = MakeTilingPara({100}, ge::DT_INT32, "NHWC", "NCHW");
    auto para5 = MakeTilingPara({100}, ge::DT_INT32, "NDHWC", "NCDHW");

    TilingInfo info4, info5;
    ASSERT_TRUE(ExecuteTiling(para4, info4));
    ASSERT_TRUE(ExecuteTiling(para5, info5));

    auto* td4 = reinterpret_cast<const DataFormatDimMapTilingData*>(info4.tilingData.get());
    auto* td5 = reinterpret_cast<const DataFormatDimMapTilingData*>(info5.tilingData.get());

    EXPECT_EQ(info4.tilingKey, info5.tilingKey);
    EXPECT_EQ(td4->formatLen, 4);
    EXPECT_EQ(td5->formatLen, 5);
}

// ============================================================================
// Iteration 2: Multi-core split branch coverage
// ============================================================================

TEST_F(DataFormatDimMapTilingTest, test_small_data_single_core)
{
    auto para = MakeTilingPara({8}, ge::DT_INT32, "NHWC", "NCHW", 20);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    auto* td = reinterpret_cast<const DataFormatDimMapTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td->totalNum, 8);
    EXPECT_GE(td->blockFactor, td->totalNum);
    EXPECT_EQ(tilingInfo.blockNum, 1u);
}

TEST_F(DataFormatDimMapTilingTest, test_large_data_multicore_split)
{
    auto para = MakeTilingPara({100000}, ge::DT_INT32, "NHWC", "NCHW", 20);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    auto* td = reinterpret_cast<const DataFormatDimMapTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td->totalNum, 100000);
    EXPECT_GT(tilingInfo.blockNum, 1u);
    EXPECT_GE(static_cast<int64_t>(tilingInfo.blockNum) * td->blockFactor, td->totalNum);

    std::cout << "Multi-core: blockNum=" << tilingInfo.blockNum
              << " blockFactor=" << td->blockFactor << std::endl;
}

TEST_F(DataFormatDimMapTilingTest, test_blockfactor_alignment)
{
    auto para = MakeTilingPara({10007}, ge::DT_INT32, "NHWC", "NCHW", 20);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    auto* td = reinterpret_cast<const DataFormatDimMapTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td->blockFactor % 8, 0)
        << "blockFactor=" << td->blockFactor << " should be aligned to ubBlockSize";
}

TEST_F(DataFormatDimMapTilingTest, test_blockfactor_alignment_int64)
{
    auto para = MakeTilingPara({10007}, ge::DT_INT64, "NHWC", "NCHW", 20);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    auto* td = reinterpret_cast<const DataFormatDimMapTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td->blockFactor % 4, 0)
        << "blockFactor=" << td->blockFactor << " should be aligned to ubBlockSize for int64";
}

TEST_F(DataFormatDimMapTilingTest, test_exact_core_divisible)
{
    auto para = MakeTilingPara({20000}, ge::DT_INT32, "NHWC", "NCHW", 20);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    auto* td = reinterpret_cast<const DataFormatDimMapTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td->totalNum, 20000);
    EXPECT_GE(tilingInfo.blockNum, 10u);
}

// ============================================================================
// Iteration 3: int64 dtype Tiling paths
// ============================================================================

TEST_F(DataFormatDimMapTilingTest, test_int64_small_data)
{
    auto para = MakeTilingPara({100}, ge::DT_INT64, "NHWC", "NCHW");
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    auto* td = reinterpret_cast<const DataFormatDimMapTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td->totalNum, 100);
    EXPECT_GT(td->blockFactor, 0);
    EXPECT_GT(td->ubFactor, 0);
    EXPECT_EQ(td->ubFactor % 4, 0) << "int64 ubFactor should be aligned to ubBlockSize=4";

    std::cout << "int64 small data tilingKey=" << tilingInfo.tilingKey
              << " ubFactor=" << td->ubFactor << std::endl;
}

TEST_F(DataFormatDimMapTilingTest, test_int64_large_data)
{
    auto para = MakeTilingPara({2048}, ge::DT_INT64, "NHWC", "NCHW");
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    auto* td = reinterpret_cast<const DataFormatDimMapTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td->totalNum, 2048);
    EXPECT_GT(td->ubFactor, 0);
    EXPECT_EQ(td->ubFactor % 4, 0) << "int64 ubFactor should be aligned to ubBlockSize=4";

    std::cout << "int64 large data tilingKey=" << tilingInfo.tilingKey
              << " ubFactor=" << td->ubFactor << std::endl;
}

TEST_F(DataFormatDimMapTilingTest, test_int64_ubfactor_calculation)
{
    auto paraI32 = MakeTilingPara({100}, ge::DT_INT32, "NHWC", "NCHW");
    auto paraI64 = MakeTilingPara({100}, ge::DT_INT64, "NHWC", "NCHW");
    TilingInfo infoI32, infoI64;
    ASSERT_TRUE(ExecuteTiling(paraI32, infoI32));
    ASSERT_TRUE(ExecuteTiling(paraI64, infoI64));

    auto* tdI32 = reinterpret_cast<const DataFormatDimMapTilingData*>(infoI32.tilingData.get());
    auto* tdI64 = reinterpret_cast<const DataFormatDimMapTilingData*>(infoI64.tilingData.get());

    EXPECT_GT(tdI32->ubFactor, tdI64->ubFactor)
        << "int32 ubFactor should be larger than int64 ubFactor (int64 uses more bytes per element)";

    EXPECT_NE(infoI32.tilingKey, infoI64.tilingKey)
        << "int32 and int64 should have different tilingKeys";
}

TEST_F(DataFormatDimMapTilingTest, test_int64_boundary_1023_vs_1024)
{
    auto para1023 = MakeTilingPara({1023}, ge::DT_INT64, "NHWC", "NCHW");
    auto para1024 = MakeTilingPara({1024}, ge::DT_INT64, "NHWC", "NCHW");
    TilingInfo info1023, info1024;
    ASSERT_TRUE(ExecuteTiling(para1023, info1023));
    ASSERT_TRUE(ExecuteTiling(para1024, info1024));

    auto* td1023 = reinterpret_cast<const DataFormatDimMapTilingData*>(info1023.tilingData.get());
    auto* td1024 = reinterpret_cast<const DataFormatDimMapTilingData*>(info1024.tilingData.get());

    EXPECT_EQ(info1023.tilingKey, info1024.tilingKey)
        << "Same dtype should produce the same tilingKey regardless of data size";

    EXPECT_EQ(td1023->ubFactor, td1024->ubFactor)
        << "Same dtype and UB size should produce the same ubFactor";
}

// ============================================================================
// Iteration 3: Additional boundary scenarios
// ============================================================================

TEST_F(DataFormatDimMapTilingTest, test_single_element_tensor)
{
    auto para = MakeTilingPara({1}, ge::DT_INT32, "NHWC", "NCHW");
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    auto* td = reinterpret_cast<const DataFormatDimMapTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td->totalNum, 1);
    EXPECT_GT(td->blockFactor, 0);
    EXPECT_GT(td->ubFactor, 0);
    EXPECT_EQ(tilingInfo.blockNum, 1u);
}

TEST_F(DataFormatDimMapTilingTest, test_exactly_1024_elements)
{
    auto para = MakeTilingPara({1024}, ge::DT_INT32, "NHWC", "NCHW");
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    auto* td = reinterpret_cast<const DataFormatDimMapTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td->totalNum, 1024);

    auto para2048 = MakeTilingPara({2048}, ge::DT_INT32, "NHWC", "NCHW");
    TilingInfo info2048;
    ASSERT_TRUE(ExecuteTiling(para2048, info2048));
    EXPECT_EQ(tilingInfo.tilingKey, info2048.tilingKey)
        << "1024 and 2048 should share the same tilingKey (same dtype int32)";
}

TEST_F(DataFormatDimMapTilingTest, test_empty_tensor_int64)
{
    auto para = MakeTilingPara({0}, ge::DT_INT64, "NHWC", "NCHW");
    TilingInfo tilingInfo;
    bool result = ExecuteTiling(para, tilingInfo);
    if (result) {
        auto* td = reinterpret_cast<const DataFormatDimMapTilingData*>(tilingInfo.tilingData.get());
        EXPECT_EQ(td->totalNum, 0);
    }
}

// ============================================================================
// Iteration 3: All format combinations
// ============================================================================

TEST_F(DataFormatDimMapTilingTest, test_nchw_to_nhwc_int64)
{
    auto para = MakeTilingPara({256}, ge::DT_INT64, "NCHW", "NHWC");
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    auto* td = reinterpret_cast<const DataFormatDimMapTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td->formatLen, 4);
    int32_t expectedTable[8] = {0, 3, 1, 2, 0, 3, 1, 2};
    for (int i = 0; i < 8; i++) {
        EXPECT_EQ(td->expandedTable[i], expectedTable[i])
            << "expandedTable[" << i << "] mismatch for NCHW->NHWC int64";
    }
}

TEST_F(DataFormatDimMapTilingTest, test_ndhwc_to_ncdhw_int64)
{
    auto para = MakeTilingPara({256}, ge::DT_INT64, "NDHWC", "NCDHW");
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    auto* td = reinterpret_cast<const DataFormatDimMapTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td->formatLen, 5);
    int32_t expected[10] = {0, 2, 3, 4, 1, 0, 2, 3, 4, 1};
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(td->expandedTable[i], expected[i])
            << "expandedTable[" << i << "] mismatch for NDHWC->NCDHW int64";
    }
}

TEST_F(DataFormatDimMapTilingTest, test_hwnc_to_nhwc_mapping_table)
{
    auto para = MakeTilingPara({64}, ge::DT_INT32, "HWNC", "NHWC");
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    auto* td = reinterpret_cast<const DataFormatDimMapTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td->formatLen, 4);
    int32_t expectedTable[8] = {1, 2, 0, 3, 1, 2, 0, 3};
    for (int i = 0; i < 8; i++) {
        EXPECT_EQ(td->expandedTable[i], expectedTable[i])
            << "expandedTable[" << i << "] mismatch for HWNC->NHWC";
    }
}

TEST_F(DataFormatDimMapTilingTest, test_hwnc_to_nhwc_int64_large)
{
    auto para = MakeTilingPara({2048}, ge::DT_INT64, "HWNC", "NHWC");
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(para, tilingInfo));

    auto* td = reinterpret_cast<const DataFormatDimMapTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(td->formatLen, 4);
    EXPECT_EQ(td->totalNum, 2048);
    EXPECT_GT(td->ubFactor, 0);
    int32_t expectedTable[8] = {1, 2, 0, 3, 1, 2, 0, 3};
    for (int i = 0; i < 8; i++) {
        EXPECT_EQ(td->expandedTable[i], expectedTable[i])
            << "expandedTable[" << i << "] mismatch";
    }
}

TEST_F(DataFormatDimMapTilingTest, test_invalid_format_permutation)
{
    auto para = MakeTilingPara({64}, ge::DT_INT32, "NHWC", "NNNN");
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo))
        << "Should fail when dstFormat is not a valid permutation of srcFormat";
}

TEST_F(DataFormatDimMapTilingTest, test_invalid_format_partial_mismatch)
{
    auto para = MakeTilingPara({64}, ge::DT_INT32, "NHWC", "NCHX");
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteTiling(para, tilingInfo))
        << "Should fail when dstFormat contains chars not in srcFormat";
}

} // namespace DataFormatDimMapUT
