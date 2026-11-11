/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../../op_host/arch35/dequantize_tiling_arch35.h"
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class DequantizeTilingTest : public testing::Test {};

// case1: int8 1D [128], scalar min/max range, MIN_COMBINED
TEST_F(DequantizeTilingTest, case1_int8_1d)
{
    optiling::DequantizeCompileInfo ci = {64, 262144};
    gert::TilingContextPara ctx(
        "Dequantize",
        {{{{{128}, {128}}, DT_INT8, FORMAT_ND}, {{{1}, {1}}, DT_FLOAT, FORMAT_ND}, {{{1}, {1}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{128}, {128}}, DT_FLOAT, FORMAT_ND}}},
        {gert::TilingContextPara::OpAttr("mode", Ops::NN::AnyValue::CreateFrom<std::string>("MIN_COMBINED"))}, &ci);
    string expect = "3 128 1 128 1 1 1 0 1 52288 1 1 1 128 3 1 1 1 1 128 1 1 1 1 1 1 1 1 "
                    "0 0 0 1 0 0 0 0 0 0 0 0 1 1 1 128 0 0 0 1 4863606123707498496 128 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{0}});
}

// case2: uint8 2D [4,8], scalar min/max range, MIN_COMBINED
TEST_F(DequantizeTilingTest, case2_uint8_2d)
{
    optiling::DequantizeCompileInfo ci = {64, 262144};
    gert::TilingContextPara ctx(
        "Dequantize",
        {{{{{4, 8}, {4, 8}}, DT_UINT8, FORMAT_ND},
          {{{1}, {1}}, DT_FLOAT, FORMAT_ND},
          {{{1}, {1}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{4, 8}, {4, 8}}, DT_FLOAT, FORMAT_ND}}},
        {gert::TilingContextPara::OpAttr("mode", Ops::NN::AnyValue::CreateFrom<std::string>("MIN_COMBINED"))}, &ci);
    string expect = "2 4 1 4 1 1 1 0 2 52288 1 1 4 8 3 1 1 1 4 8 1 1 1 1 1 1 1 1 "
                    "0 0 8 1 0 0 0 0 0 0 0 0 1 1 4 8 0 0 8 1 4863606122583425024 32 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{0}});
}

// case3: int32 2D [4,8], scalar min/max range, MIN_COMBINED
TEST_F(DequantizeTilingTest, case3_int32_2d)
{
    optiling::DequantizeCompileInfo ci = {64, 262144};
    gert::TilingContextPara ctx(
        "Dequantize",
        {{{{{4, 8}, {4, 8}}, DT_INT32, FORMAT_ND},
          {{{1}, {1}}, DT_FLOAT, FORMAT_ND},
          {{{1}, {1}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{4, 8}, {4, 8}}, DT_FLOAT, FORMAT_ND}}},
        {gert::TilingContextPara::OpAttr("mode", Ops::NN::AnyValue::CreateFrom<std::string>("MIN_COMBINED"))}, &ci);
    string expect = "2 4 1 4 1 1 1 0 2 52288 1 1 4 8 3 1 1 1 4 8 1 1 1 1 1 1 1 1 "
                    "0 0 8 1 0 0 0 0 0 0 0 0 1 1 4 8 0 0 8 1 5728578727340670976 32 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{0}});
}

// case4: int8 3D broadcast [2,4,8]+[1,1,8]+[1,1,8], MIN_COMBINED
TEST_F(DequantizeTilingTest, case4_int8_3d_broadcast)
{
    optiling::DequantizeCompileInfo ci = {64, 262144};
    gert::TilingContextPara ctx(
        "Dequantize",
        {{{{{2, 4, 8}, {2, 4, 8}}, DT_INT8, FORMAT_ND},
          {{{1, 1, 8}, {1, 1, 8}}, DT_FLOAT, FORMAT_ND},
          {{{1, 1, 8}, {1, 1, 8}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{2, 4, 8}, {2, 4, 8}}, DT_FLOAT, FORMAT_ND}}},
        {gert::TilingContextPara::OpAttr("mode", Ops::NN::AnyValue::CreateFrom<std::string>("MIN_COMBINED"))}, &ci);
    string expect = "1 2 1 2 1 1 1 0 3 52288 1 2 4 8 3 1 1 2 4 8 1 1 1 8 1 1 1 8 "
                    "0 32 8 1 0 0 0 1 0 0 0 1 1 2 4 8 0 32 8 1 4863606123707498496 64 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{0}});
}

// case5: int8 4D [2,3,4,8], broadcast min/max, MIN_COMBINED
TEST_F(DequantizeTilingTest, case5_int8_4d)
{
    optiling::DequantizeCompileInfo ci = {64, 262144};
    gert::TilingContextPara ctx(
        "Dequantize",
        {{{{{2, 3, 4, 8}, {2, 3, 4, 8}}, DT_INT8, FORMAT_ND},
          {{{2, 3, 4, 8}, {2, 3, 4, 8}}, DT_FLOAT, FORMAT_ND},
          {{{2, 3, 4, 8}, {2, 3, 4, 8}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{2, 3, 4, 8}, {2, 3, 4, 8}}, DT_FLOAT, FORMAT_ND}}},
        {gert::TilingContextPara::OpAttr("mode", Ops::NN::AnyValue::CreateFrom<std::string>("MIN_COMBINED"))}, &ci);
    string expect = "0 2 1 2 1 1 1 0 4 52288 2 3 4 8 3 1 2 3 4 8 2 3 4 8 2 3 4 8 "
                    "96 32 8 1 96 32 8 1 96 32 8 1 2 3 4 8 96 32 8 1 4863606123707498496 192 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{0}});
}

// case6: int8 large [64,1024,256], scalar min/max, MIN_COMBINED
TEST_F(DequantizeTilingTest, case6_int8_large)
{
    optiling::DequantizeCompileInfo ci = {64, 262144};
    gert::TilingContextPara ctx(
        "Dequantize",
        {{{{{64, 1024, 256}, {64, 1024, 256}}, DT_INT8, FORMAT_ND},
          {{{1}, {1}}, DT_FLOAT, FORMAT_ND},
          {{{1}, {1}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{64, 1024, 256}, {64, 1024, 256}}, DT_FLOAT, FORMAT_ND}}},
        {gert::TilingContextPara::OpAttr("mode", Ops::NN::AnyValue::CreateFrom<std::string>("MIN_COMBINED"))}, &ci);
    string expect = "2 51 21 4 64 1344 21 0 3 52288 1 64 1024 256 3 1 1 64 1024 256 1 1 1 1 1 1 1 1 "
                    "0 262144 256 1 0 0 0 0 0 0 0 0 1 64 1024 256 0 262144 256 1 4863606123707498496 16777216 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{0}});
}

// case7: int8 5D [2,2,3,4,2], scalar min/max, MIN_COMBINED -> kRank=8
TEST_F(DequantizeTilingTest, case7_int8_5d)
{
    optiling::DequantizeCompileInfo ci = {64, 262144};
    gert::TilingContextPara ctx(
        "Dequantize",
        {{{{{2, 2, 3, 4, 2}, {2, 2, 3, 4, 2}}, DT_INT8, FORMAT_ND},
          {{{2, 2, 3, 4, 2}, {2, 2, 3, 4, 2}}, DT_FLOAT, FORMAT_ND},
          {{{2, 2, 3, 4, 2}, {2, 2, 3, 4, 2}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{2, 2, 3, 4, 2}, {2, 2, 3, 4, 2}}, DT_FLOAT, FORMAT_ND}}},
        {gert::TilingContextPara::OpAttr("mode", Ops::NN::AnyValue::CreateFrom<std::string>("MIN_COMBINED"))}, &ci);
    string expect = "3 2 1 2 1 1 1 0 5 52288 1 1 1 2 2 3 4 2 3 1 1 1 1 2 2 3 4 2 1 1 1 2 2 3 4 2 1 1 1 2 2 3 4 2 "
                    "0 0 0 48 24 8 2 1 0 0 0 48 24 8 2 1 0 0 0 48 24 8 2 1 1 1 1 2 2 3 4 2 "
                    "0 0 0 48 24 8 2 1 4863606123707498496 96 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 4, expect, {{0}});
}

// case8: int8 broadcast multi [2,4,8]+[4,1]+[1,8], MIN_COMBINED
TEST_F(DequantizeTilingTest, case8_int8_broadcast_multi)
{
    optiling::DequantizeCompileInfo ci = {64, 262144};
    gert::TilingContextPara ctx(
        "Dequantize",
        {{{{{2, 4, 8}, {2, 4, 8}}, DT_INT8, FORMAT_ND},
          {{{4, 1}, {4, 1}}, DT_FLOAT, FORMAT_ND},
          {{{1, 8}, {1, 8}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{2, 4, 8}, {2, 4, 8}}, DT_FLOAT, FORMAT_ND}}},
        {gert::TilingContextPara::OpAttr("mode", Ops::NN::AnyValue::CreateFrom<std::string>("MIN_COMBINED"))}, &ci);
    string expect = "1 2 1 2 1 1 1 0 3 52288 1 2 4 8 3 1 1 2 4 8 1 1 4 1 1 1 1 8 "
                    "0 32 8 1 0 0 1 0 0 0 0 1 1 2 4 8 0 32 8 1 4863606123707498496 64 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{0}});
}

// case9: int8 2D [4,8], scalar ranges, SCALED mode
TEST_F(DequantizeTilingTest, case9_int8_scaled)
{
    optiling::DequantizeCompileInfo ci = {64, 262144};
    gert::TilingContextPara ctx(
        "Dequantize",
        {{{{{4, 8}, {4, 8}}, DT_INT8, FORMAT_ND},
          {{{1}, {1}}, DT_FLOAT, FORMAT_ND},
          {{{1}, {1}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{4, 8}, {4, 8}}, DT_FLOAT, FORMAT_ND}}},
        {gert::TilingContextPara::OpAttr("mode", Ops::NN::AnyValue::CreateFrom<std::string>("SCALED"))}, &ci);
    string expect = "2 4 1 4 1 1 1 0 2 52288 1 1 4 8 3 1 1 1 4 8 1 1 1 1 1 1 1 1 "
                    "0 0 8 1 0 0 0 0 0 0 0 0 1 1 4 8 0 0 8 1 4827295850587750400 32 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 2, expect, {{0}});
}

// case10: int8 2D [4,8], scalar ranges, MIN_FIRST mode
TEST_F(DequantizeTilingTest, case10_int8_min_first)
{
    optiling::DequantizeCompileInfo ci = {64, 262144};
    gert::TilingContextPara ctx(
        "Dequantize",
        {{{{{4, 8}, {4, 8}}, DT_INT8, FORMAT_ND},
          {{{1}, {1}}, DT_FLOAT, FORMAT_ND},
          {{{1}, {1}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{4, 8}, {4, 8}}, DT_FLOAT, FORMAT_ND}}},
        {gert::TilingContextPara::OpAttr("mode", Ops::NN::AnyValue::CreateFrom<std::string>("MIN_FIRST"))}, &ci);
    string expect = "2 4 1 4 1 1 1 0 2 52288 1 1 4 8 3 1 1 1 4 8 1 1 1 1 1 1 1 1 "
                    "0 0 8 1 0 0 0 0 0 0 0 0 1 1 4 8 0 0 8 1 4863606123707498496 32 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 1, expect, {{0}});
}

// case11: rank 9 input/output is rejected
TEST_F(DequantizeTilingTest, case11_rank9_rejected)
{
    optiling::DequantizeCompileInfo ci = {64, 262144};
    gert::TilingContextPara ctx(
        "Dequantize",
        {{{{{1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}}, DT_INT8, FORMAT_ND},
          {{{1}, {1}}, DT_FLOAT, FORMAT_ND},
          {{{1}, {1}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}}, DT_FLOAT, FORMAT_ND}}},
        {gert::TilingContextPara::OpAttr("mode", Ops::NN::AnyValue::CreateFrom<std::string>("MIN_COMBINED"))}, &ci);
    ExecuteTestCase(ctx, GRAPH_FAILED);
}

// case12: rank 8 input/output is accepted
TEST_F(DequantizeTilingTest, case12_rank8_ok)
{
    optiling::DequantizeCompileInfo ci = {64, 262144};
    gert::TilingContextPara ctx(
        "Dequantize",
        {{{{{1, 2, 1, 2, 1, 2, 1, 2}, {1, 2, 1, 2, 1, 2, 1, 2}}, DT_INT8, FORMAT_ND},
          {{{1}, {1}}, DT_FLOAT, FORMAT_ND},
          {{{1}, {1}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{1, 2, 1, 2, 1, 2, 1, 2}, {1, 2, 1, 2, 1, 2, 1, 2}}, DT_FLOAT, FORMAT_ND}}},
        {gert::TilingContextPara::OpAttr("mode", Ops::NN::AnyValue::CreateFrom<std::string>("MIN_COMBINED"))}, &ci);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(ctx, tilingInfo));
    EXPECT_GT(tilingInfo.blockNum, 0);
}

// case13: x dim 1 cannot be broadcast up to min_range dim > 1
TEST_F(DequantizeTilingTest, case13_broadcast_to_x_rejected)
{
    optiling::DequantizeCompileInfo ci = {64, 262144};
    gert::TilingContextPara ctx(
        "Dequantize",
        {{{{{1, 3}, {1, 3}}, DT_INT8, FORMAT_ND},
          {{{5, 3}, {5, 3}}, DT_FLOAT, FORMAT_ND},
          {{{1, 3}, {1, 3}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{1, 3}, {1, 3}}, DT_FLOAT, FORMAT_ND}}},
        {gert::TilingContextPara::OpAttr("mode", Ops::NN::AnyValue::CreateFrom<std::string>("MIN_COMBINED"))}, &ci);
    ExecuteTestCase(ctx, GRAPH_FAILED);
}

// case14: output shape must equal x shape
TEST_F(DequantizeTilingTest, case14_output_must_equal_x)
{
    optiling::DequantizeCompileInfo ci = {64, 262144};
    gert::TilingContextPara ctx(
        "Dequantize",
        {{{{{5}, {5}}, DT_INT8, FORMAT_ND}, {{{1}, {1}}, DT_FLOAT, FORMAT_ND}, {{{1}, {1}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{1}, {1}}, DT_FLOAT, FORMAT_ND}}},
        {gert::TilingContextPara::OpAttr("mode", Ops::NN::AnyValue::CreateFrom<std::string>("MIN_COMBINED"))}, &ci);
    ExecuteTestCase(ctx, GRAPH_FAILED);
}

// case15: input tensor count must be 3
TEST_F(DequantizeTilingTest, case15_invalid_input_num)
{
    optiling::DequantizeCompileInfo ci = {64, 262144};
    gert::TilingContextPara ctx(
        "Dequantize", {{{{{4, 8}, {4, 8}}, DT_INT8, FORMAT_ND}, {{{1}, {1}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{4, 8}, {4, 8}}, DT_FLOAT, FORMAT_ND}}},
        {gert::TilingContextPara::OpAttr("mode", Ops::NN::AnyValue::CreateFrom<std::string>("MIN_COMBINED"))}, &ci);
    ExecuteTestCase(ctx, GRAPH_FAILED);
}

// case16: output tensor count must be 1
TEST_F(DequantizeTilingTest, case16_invalid_output_num)
{
    optiling::DequantizeCompileInfo ci = {64, 262144};
    gert::TilingContextPara ctx(
        "Dequantize",
        {{{{{4, 8}, {4, 8}}, DT_INT8, FORMAT_ND},
          {{{1}, {1}}, DT_FLOAT, FORMAT_ND},
          {{{1}, {1}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{4, 8}, {4, 8}}, DT_FLOAT, FORMAT_ND}, {{{4, 8}, {4, 8}}, DT_FLOAT, FORMAT_ND}}},
        {gert::TilingContextPara::OpAttr("mode", Ops::NN::AnyValue::CreateFrom<std::string>("MIN_COMBINED"))}, &ci);
    ExecuteTestCase(ctx, GRAPH_FAILED);
}

// case17: min_range dtype must be float32
TEST_F(DequantizeTilingTest, case17_min_dtype_invalid)
{
    optiling::DequantizeCompileInfo ci = {64, 262144};
    gert::TilingContextPara ctx(
        "Dequantize",
        {{{{{4, 8}, {4, 8}}, DT_INT8, FORMAT_ND}, {{{1}, {1}}, DT_INT8, FORMAT_ND}, {{{1}, {1}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{4, 8}, {4, 8}}, DT_FLOAT, FORMAT_ND}}},
        {gert::TilingContextPara::OpAttr("mode", Ops::NN::AnyValue::CreateFrom<std::string>("MIN_COMBINED"))}, &ci);
    ExecuteTestCase(ctx, GRAPH_FAILED);
}

// case18: output dtype must be float32
TEST_F(DequantizeTilingTest, case18_output_dtype_invalid)
{
    optiling::DequantizeCompileInfo ci = {64, 262144};
    gert::TilingContextPara ctx(
        "Dequantize",
        {{{{{4, 8}, {4, 8}}, DT_INT8, FORMAT_ND},
          {{{1}, {1}}, DT_FLOAT, FORMAT_ND},
          {{{1}, {1}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{4, 8}, {4, 8}}, DT_FLOAT16, FORMAT_ND}}},
        {gert::TilingContextPara::OpAttr("mode", Ops::NN::AnyValue::CreateFrom<std::string>("MIN_COMBINED"))}, &ci);
    ExecuteTestCase(ctx, GRAPH_FAILED);
}

// case19: negative/unknown dim must be rejected at tiling
TEST_F(DequantizeTilingTest, case19_negative_dim_rejected)
{
    optiling::DequantizeCompileInfo ci = {64, 262144};
    gert::TilingContextPara ctx(
        "Dequantize",
        {{{{{2, -1}, {2, -1}}, DT_INT8, FORMAT_ND},
          {{{1}, {1}}, DT_FLOAT, FORMAT_ND},
          {{{1}, {1}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{2, -1}, {2, -1}}, DT_FLOAT, FORMAT_ND}}},
        {gert::TilingContextPara::OpAttr("mode", Ops::NN::AnyValue::CreateFrom<std::string>("MIN_COMBINED"))}, &ci);
    ExecuteTestCase(ctx, GRAPH_FAILED);
}

// case20: valid broadcast to x
TEST_F(DequantizeTilingTest, case20_valid_broadcast_to_x)
{
    optiling::DequantizeCompileInfo ci = {64, 262144};
    gert::TilingContextPara ctx(
        "Dequantize",
        {{{{{2, 4}, {2, 4}}, DT_INT8, FORMAT_ND},
          {{{1, 4}, {1, 4}}, DT_FLOAT, FORMAT_ND},
          {{{2, 1}, {2, 1}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{2, 4}, {2, 4}}, DT_FLOAT, FORMAT_ND}}},
        {gert::TilingContextPara::OpAttr("mode", Ops::NN::AnyValue::CreateFrom<std::string>("MIN_COMBINED"))}, &ci);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(ctx, tilingInfo));
    EXPECT_GT(tilingInfo.blockNum, 0);
}

// case21: coreNum must be positive
TEST_F(DequantizeTilingTest, case21_core_num_zero)
{
    optiling::DequantizeCompileInfo ci = {0, 262144};
    gert::TilingContextPara ctx(
        "Dequantize",
        {{{{{4, 8}, {4, 8}}, DT_INT8, FORMAT_ND},
          {{{1}, {1}}, DT_FLOAT, FORMAT_ND},
          {{{1}, {1}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{4, 8}, {4, 8}}, DT_FLOAT, FORMAT_ND}}},
        {gert::TilingContextPara::OpAttr("mode", Ops::NN::AnyValue::CreateFrom<std::string>("MIN_COMBINED"))}, &ci);
    ExecuteTestCase(ctx, GRAPH_FAILED);
}

// case22: ubSize must be greater than metadata
TEST_F(DequantizeTilingTest, case22_ub_size_too_small)
{
    optiling::DequantizeCompileInfo ci = {64, 0};
    gert::TilingContextPara ctx(
        "Dequantize",
        {{{{{4, 8}, {4, 8}}, DT_INT8, FORMAT_ND},
          {{{1}, {1}}, DT_FLOAT, FORMAT_ND},
          {{{1}, {1}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{4, 8}, {4, 8}}, DT_FLOAT, FORMAT_ND}}},
        {gert::TilingContextPara::OpAttr("mode", Ops::NN::AnyValue::CreateFrom<std::string>("MIN_COMBINED"))}, &ci);
    ExecuteTestCase(ctx, GRAPH_FAILED);
}

// case23: empty x launches one core and exits in kernel
TEST_F(DequantizeTilingTest, case23_empty_x)
{
    optiling::DequantizeCompileInfo ci = {64, 262144};
    gert::TilingContextPara ctx(
        "Dequantize",
        {{{{{0, 3}, {0, 3}}, DT_INT8, FORMAT_ND},
          {{{1}, {1}}, DT_FLOAT, FORMAT_ND},
          {{{1}, {1}}, DT_FLOAT, FORMAT_ND}}},
        {{{{{0, 3}, {0, 3}}, DT_FLOAT, FORMAT_ND}}},
        {gert::TilingContextPara::OpAttr("mode", Ops::NN::AnyValue::CreateFrom<std::string>("MIN_COMBINED"))}, &ci);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(ctx, tilingInfo));
    EXPECT_EQ(tilingInfo.blockNum, 1);
}

// case24: rank-0 scalar x is one element and uses the normal tiling path
TEST_F(DequantizeTilingTest, case24_scalar_x)
{
    optiling::DequantizeCompileInfo ci = {64, 262144};
    gert::TilingContextPara ctx(
        "Dequantize",
        {{{{}, {}}, DT_INT8, FORMAT_ND}, {{{}, {}}, DT_FLOAT, FORMAT_ND}, {{{}, {}}, DT_FLOAT, FORMAT_ND}},
        {{{{}, {}}, DT_FLOAT, FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("mode", Ops::NN::AnyValue::CreateFrom<std::string>("MIN_COMBINED"))}, &ci);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(ctx, tilingInfo));
    EXPECT_GT(tilingInfo.blockNum, 0);
}
