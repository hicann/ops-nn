/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../../op_host/arch35/acts_ulq_tiling_arch35.h"
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class ActsUlqTilingTest : public testing::Test {};

// case1: FP32 1D [128], clamp scalar, fixed_min=true
TEST_F(ActsUlqTilingTest, case1_fp32_1d_fixedTrue) {
    optiling::ActsUlqCompileInfo ci = {64, 262144};
    gert::TilingContextPara ctx("ActsULQ",
        {{{{{128},{128}},DT_FLOAT,FORMAT_ND},{{{1},{1}},DT_FLOAT,FORMAT_ND},
          {{{1},{1}},DT_FLOAT,FORMAT_ND}}},
        {{{{{128},{128}},DT_FLOAT,FORMAT_ND},{{{128},{128}},DT_FLOAT,FORMAT_ND},
          {{{128},{128}},DT_FLOAT,FORMAT_ND},{{{128},{128}},DT_FLOAT,FORMAT_ND}}},
        {gert::TilingContextPara::OpAttr("fixed_min", Ops::NN::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(8))},
        &ci);
    string expect = "3 128 1 128 1 1 1 0 1 43680 10920 1 1 1 128 3 4 1 1 1 128 1 1 1 1 1 1 1 1 0 0 0 1 0 0 0 0 0 0 0 0 1 1 1 128 1 1 1 128 1 1 1 128 1 1 1 128 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1 1 8 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{16777216}});
}

// case2: FP16 2D [4,8], clamp scalar, fixed_min=false
TEST_F(ActsUlqTilingTest, case2_fp16_2d_fixedFalse) {
    optiling::ActsUlqCompileInfo ci = {64, 262144};
    gert::TilingContextPara ctx("ActsULQ",
        {{{{{4,8},{4,8}},DT_FLOAT16,FORMAT_ND},{{{1},{1}},DT_FLOAT16,FORMAT_ND},
          {{{1},{1}},DT_FLOAT16,FORMAT_ND}}},
        {{{{{4,8},{4,8}},DT_FLOAT16,FORMAT_ND},{{{4,8},{4,8}},DT_FLOAT16,FORMAT_ND},
          {{{4,8},{4,8}},DT_FLOAT16,FORMAT_ND},{{{4,8},{4,8}},DT_FLOAT16,FORMAT_ND}}},
        {gert::TilingContextPara::OpAttr("fixed_min", Ops::NN::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(8))},
        &ci);
    string expect = "2 4 1 4 1 1 1 0 2 37440 9360 1 1 4 8 3 4 1 1 4 8 1 1 1 1 1 1 1 1 0 0 8 1 0 0 0 0 0 0 0 0 1 1 4 8 1 1 4 8 1 1 4 8 1 1 4 8 0 0 8 1 0 0 8 1 0 0 8 1 0 0 8 1 0 8 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{16777216}});
}

// case3: FP32 3D broadcast [2,4,8]+[1,1,8]+[1,1,8], fixed_min=true
TEST_F(ActsUlqTilingTest, case3_fp32_3d_broadcast) {
    optiling::ActsUlqCompileInfo ci = {64, 262144};
    gert::TilingContextPara ctx("ActsULQ",
        {{{{{2,4,8},{2,4,8}},DT_FLOAT,FORMAT_ND},{{{1,1,8},{1,1,8}},DT_FLOAT,FORMAT_ND},
          {{{1,1,8},{1,1,8}},DT_FLOAT,FORMAT_ND}}},
        {{{{{2,4,8},{2,4,8}},DT_FLOAT,FORMAT_ND},{{{2,4,8},{2,4,8}},DT_FLOAT,FORMAT_ND},
          {{{2,4,8},{2,4,8}},DT_FLOAT,FORMAT_ND},{{{2,4,8},{2,4,8}},DT_FLOAT,FORMAT_ND}}},
        {gert::TilingContextPara::OpAttr("fixed_min", Ops::NN::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(8))},
        &ci);
    string expect = "1 2 1 2 1 1 1 0 3 43680 10920 1 2 4 8 3 4 1 2 4 8 1 1 1 8 1 1 1 8 0 32 8 1 0 0 0 1 0 0 0 1 1 2 4 8 1 2 4 8 1 2 4 8 1 2 4 8 0 32 8 1 0 32 8 1 0 32 8 1 0 32 8 1 1 8 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{16777216}});
}

// case4: FP16 4D [2,3,4,8], fixed_min=false
TEST_F(ActsUlqTilingTest, case4_fp16_4d) {
    optiling::ActsUlqCompileInfo ci = {64, 262144};
    gert::TilingContextPara ctx("ActsULQ",
        {{{{{2,3,4,8},{2,3,4,8}},DT_FLOAT16,FORMAT_ND},{{{2,3,4,8},{2,3,4,8}},DT_FLOAT16,FORMAT_ND},
          {{{2,3,4,8},{2,3,4,8}},DT_FLOAT16,FORMAT_ND}}},
        {{{{{2,3,4,8},{2,3,4,8}},DT_FLOAT16,FORMAT_ND},{{{2,3,4,8},{2,3,4,8}},DT_FLOAT16,FORMAT_ND},
          {{{2,3,4,8},{2,3,4,8}},DT_FLOAT16,FORMAT_ND},{{{2,3,4,8},{2,3,4,8}},DT_FLOAT16,FORMAT_ND}}},
        {gert::TilingContextPara::OpAttr("fixed_min", Ops::NN::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(8))},
        &ci);
    string expect = "0 2 1 2 1 1 1 0 4 37440 9360 2 3 4 8 3 4 2 3 4 8 2 3 4 8 2 3 4 8 96 32 8 1 96 32 8 1 96 32 8 1 2 3 4 8 2 3 4 8 2 3 4 8 2 3 4 8 96 32 8 1 96 32 8 1 96 32 8 1 96 32 8 1 0 8 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{16777216}});
}

// case5: FP32 scalar, fixed_min=true
TEST_F(ActsUlqTilingTest, case5_fp32_scalar) {
    optiling::ActsUlqCompileInfo ci = {64, 262144};
    gert::TilingContextPara ctx("ActsULQ",
        {{{{{},{}},DT_FLOAT,FORMAT_ND},{{{},{}},DT_FLOAT,FORMAT_ND},{{{},{}},DT_FLOAT,FORMAT_ND}}},
        {{{{{},{}},DT_FLOAT,FORMAT_ND},{{{},{}},DT_FLOAT,FORMAT_ND},
          {{{},{}},DT_FLOAT,FORMAT_ND},{{{},{}},DT_FLOAT,FORMAT_ND}}},
        {gert::TilingContextPara::OpAttr("fixed_min", Ops::NN::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(8))},
        &ci);
    string expect = "3 1 1 1 1 1 1 0 1 43680 10920 1 1 1 1 3 4 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 8 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{16777216}});
}

// case6: FP32 large shape [64,1024,256], clamp scalar, fixed_min=false
TEST_F(ActsUlqTilingTest, case6_fp32_large_shape) {
    optiling::ActsUlqCompileInfo ci = {64, 262144};
    gert::TilingContextPara ctx("ActsULQ",
        {{{{{64,1024,256},{64,1024,256}},DT_FLOAT,FORMAT_ND},{{{1},{1}},DT_FLOAT,FORMAT_ND},
          {{{1},{1}},DT_FLOAT,FORMAT_ND}}},
        {{{{{64,1024,256},{64,1024,256}},DT_FLOAT,FORMAT_ND},{{{64,1024,256},{64,1024,256}},DT_FLOAT,FORMAT_ND},
          {{{64,1024,256},{64,1024,256}},DT_FLOAT,FORMAT_ND},{{{64,1024,256},{64,1024,256}},DT_FLOAT,FORMAT_ND}}},
        {gert::TilingContextPara::OpAttr("fixed_min", Ops::NN::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(8))},
        &ci);
    string expect = "2 42 25 16 64 1600 25 0 3 43680 10920 1 64 1024 256 3 4 1 64 1024 256 1 1 1 1 1 1 1 1 0 262144 256 1 0 0 0 0 0 0 0 0 1 64 1024 256 1 64 1024 256 1 64 1024 256 1 64 1024 256 0 262144 256 1 0 262144 256 1 0 262144 256 1 0 262144 256 1 0 8 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{16777216}});
}

// case7: FP16 5D [2,2,3,4,2], fixed_min=true → kRank=8
TEST_F(ActsUlqTilingTest, case7_fp16_5d) {
    optiling::ActsUlqCompileInfo ci = {64, 262144};
    gert::TilingContextPara ctx("ActsULQ",
        {{{{{2,2,3,4,2},{2,2,3,4,2}},DT_FLOAT16,FORMAT_ND},{{{2,2,3,4,2},{2,2,3,4,2}},DT_FLOAT16,FORMAT_ND},
          {{{2,2,3,4,2},{2,2,3,4,2}},DT_FLOAT16,FORMAT_ND}}},
        {{{{{2,2,3,4,2},{2,2,3,4,2}},DT_FLOAT16,FORMAT_ND},{{{2,2,3,4,2},{2,2,3,4,2}},DT_FLOAT16,FORMAT_ND},
          {{{2,2,3,4,2},{2,2,3,4,2}},DT_FLOAT16,FORMAT_ND},{{{2,2,3,4,2},{2,2,3,4,2}},DT_FLOAT16,FORMAT_ND}}},
        {gert::TilingContextPara::OpAttr("fixed_min", Ops::NN::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(8))},
        &ci);
    string expect = "3 2 1 2 1 1 1 0 5 37440 9360 1 1 1 2 2 3 4 2 3 4 1 1 1 2 2 3 4 2 1 1 1 2 2 3 4 2 1 1 1 2 2 3 4 2 0 0 0 48 24 8 2 1 0 0 0 48 24 8 2 1 0 0 0 48 24 8 2 1 1 1 1 2 2 3 4 2 1 1 1 2 2 3 4 2 1 1 1 2 2 3 4 2 1 1 1 2 2 3 4 2 0 0 0 48 24 8 2 1 0 0 0 48 24 8 2 1 0 0 0 48 24 8 2 1 0 0 0 48 24 8 2 1 1 8 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 1, expect, {{16777216}});
}

// case8: FP32 8D [2,2,2,2,2,2,2,2], fixed_min=false → kRank=8
TEST_F(ActsUlqTilingTest, case8_fp32_8d) {
    optiling::ActsUlqCompileInfo ci = {64, 262144};
    gert::TilingContextPara ctx("ActsULQ",
        {{{{{2,2,2,2,2,2,2,2},{2,2,2,2,2,2,2,2}},DT_FLOAT,FORMAT_ND},
          {{{2,2,2,2,2,2,2,2},{2,2,2,2,2,2,2,2}},DT_FLOAT,FORMAT_ND},
          {{{2,2,2,2,2,2,2,2},{2,2,2,2,2,2,2,2}},DT_FLOAT,FORMAT_ND}}},
        {{{{{2,2,2,2,2,2,2,2},{2,2,2,2,2,2,2,2}},DT_FLOAT,FORMAT_ND},
          {{{2,2,2,2,2,2,2,2},{2,2,2,2,2,2,2,2}},DT_FLOAT,FORMAT_ND},
          {{{2,2,2,2,2,2,2,2},{2,2,2,2,2,2,2,2}},DT_FLOAT,FORMAT_ND},
          {{{2,2,2,2,2,2,2,2},{2,2,2,2,2,2,2,2}},DT_FLOAT,FORMAT_ND}}},
        {gert::TilingContextPara::OpAttr("fixed_min", Ops::NN::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(8))},
        &ci);
    string expect = "0 2 1 2 1 1 1 0 8 43680 10920 2 2 2 2 2 2 2 2 3 4 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 128 64 32 16 8 4 2 1 128 64 32 16 8 4 2 1 128 64 32 16 8 4 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 128 64 32 16 8 4 2 1 128 64 32 16 8 4 2 1 128 64 32 16 8 4 2 1 128 64 32 16 8 4 2 1 0 8 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 1, expect, {{16777216}});
}

// case9: FP32 broadcast multi [2,4,8]+[4,1]+[1,8], fixed_min=true
TEST_F(ActsUlqTilingTest, case9_fp32_broadcast_multi) {
    optiling::ActsUlqCompileInfo ci = {64, 262144};
    gert::TilingContextPara ctx("ActsULQ",
        {{{{{2,4,8},{2,4,8}},DT_FLOAT,FORMAT_ND},{{{4,1},{4,1}},DT_FLOAT,FORMAT_ND},
          {{{1,8},{1,8}},DT_FLOAT,FORMAT_ND}}},
        {{{{{2,4,8},{2,4,8}},DT_FLOAT,FORMAT_ND},{{{2,4,8},{2,4,8}},DT_FLOAT,FORMAT_ND},
          {{{2,4,8},{2,4,8}},DT_FLOAT,FORMAT_ND},{{{2,4,8},{2,4,8}},DT_FLOAT,FORMAT_ND}}},
        {gert::TilingContextPara::OpAttr("fixed_min", Ops::NN::AnyValue::CreateFrom<bool>(true)),
         gert::TilingContextPara::OpAttr("num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(8))},
        &ci);
    string expect = "1 2 1 2 1 1 1 0 3 43680 10920 1 2 4 8 3 4 1 2 4 8 1 1 4 1 1 1 1 8 0 32 8 1 0 0 1 0 0 0 0 1 1 2 4 8 1 2 4 8 1 2 4 8 1 2 4 8 0 32 8 1 0 32 8 1 0 32 8 1 0 32 8 1 1 8 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{16777216}});
}

// case10: FP16 6D [2,2,2,3,2,2], fixed_min=false → kRank=8
TEST_F(ActsUlqTilingTest, case10_fp16_6d) {
    optiling::ActsUlqCompileInfo ci = {64, 262144};
    gert::TilingContextPara ctx("ActsULQ",
        {{{{{2,2,2,3,2,2},{2,2,2,3,2,2}},DT_FLOAT16,FORMAT_ND},
          {{{2,2,2,3,2,2},{2,2,2,3,2,2}},DT_FLOAT16,FORMAT_ND},
          {{{2,2,2,3,2,2},{2,2,2,3,2,2}},DT_FLOAT16,FORMAT_ND}}},
        {{{{{2,2,2,3,2,2},{2,2,2,3,2,2}},DT_FLOAT16,FORMAT_ND},
          {{{2,2,2,3,2,2},{2,2,2,3,2,2}},DT_FLOAT16,FORMAT_ND},
          {{{2,2,2,3,2,2},{2,2,2,3,2,2}},DT_FLOAT16,FORMAT_ND},
          {{{2,2,2,3,2,2},{2,2,2,3,2,2}},DT_FLOAT16,FORMAT_ND}}},
        {gert::TilingContextPara::OpAttr("fixed_min", Ops::NN::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(8))},
        &ci);
    string expect = "2 2 1 2 1 1 1 0 6 37440 9360 1 1 2 2 2 3 2 2 3 4 1 1 2 2 2 3 2 2 1 1 2 2 2 3 2 2 1 1 2 2 2 3 2 2 0 0 48 24 12 4 2 1 0 0 48 24 12 4 2 1 0 0 48 24 12 4 2 1 1 1 2 2 2 3 2 2 1 1 2 2 2 3 2 2 1 1 2 2 2 3 2 2 1 1 2 2 2 3 2 2 0 0 48 24 12 4 2 1 0 0 48 24 12 4 2 1 0 0 48 24 12 4 2 1 0 0 48 24 12 4 2 1 0 8 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 1, expect, {{16777216}});
}
