/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file test_adam_tiling.cpp
 * \brief
 */
#include "optim/adam_apply_one_assign/op_host/arch35/adam_apply_one_assign_tiling_arch35.h"
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class AdamTilingTest : public testing::Test {};

// ============================================================
// PadAndSqueeze (3 cases)
// ============================================================

/*
// case1: 混合 rank (4D/3D/2D/1D/标量)，主测补1拉齐 + broadcast stride
TEST_F(AdamTilingTest, case1_mixed_rank) {
    optiling::AdamCompileInfo ci = {56, 256*1024};
    gert::TilingContextPara ctx("AdamApplyOneAssign",
        {{{{{4,6,3,5},{4,6,3,5}},DT_FLOAT,FORMAT_ND},{{{6,3,5},{6,3,5}},DT_FLOAT,FORMAT_ND},
          {{{3,5},{3,5}},DT_FLOAT,FORMAT_ND},{{{5},{5}},DT_FLOAT,FORMAT_ND},
          {{{4,1,1,5},{4,1,1,5}},DT_FLOAT,FORMAT_ND},{{{4,6,1,1},{4,6,1,1}},DT_FLOAT,FORMAT_ND},
          {{{},{}},DT_FLOAT,FORMAT_ND},{{{4,1,1,1},{4,1,1,1}},DT_FLOAT,FORMAT_ND},
          {{{1,1,1,5},{1,1,1,5}},DT_FLOAT,FORMAT_ND},{{{4,6,3,5},{4,6,3,5}},DT_FLOAT,FORMAT_ND}}},
        {{{{{4,6,3,5},{4,6,3,5}},DT_FLOAT,FORMAT_ND},{{{4,6,3,5},{4,6,3,5}},DT_FLOAT,FORMAT_ND},
          {{{4,6,3,5},{4,6,3,5}},DT_FLOAT,FORMAT_ND}}}, &ci);
    string expect = "0 4 1 4 1 1 1 0 4 52416 4 6 3 5 10 3 4 6 3 5 1 6 3 5 1 1 3 5 1 1 1 5 4 1 1 5 4 6 1 1 1 1 1 1 4 1 1
1 1 1 1 5 4 6 3 5 90 15 5 1 0 15 5 1 0 0 5 1 0 0 0 1 5 0 0 1 6 1 0 0 0 0 0 0 1 0 0 0 0 0 0 1 90 15 5 1 4 6 3 5 4 6 3 5 4
6 3 5 90 15 5 1 90 15 5 1 90 15 5 1 "; ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{16777216}});
}

// case2: 去1 — 所有入参轴0全为1 → squeeze掉
TEST_F(AdamTilingTest, case2_de1) {
    optiling::AdamCompileInfo ci = {56, 256*1024};
    gert::TilingContextPara ctx("AdamApplyOneAssign",
        {{{{{1,4,6},{1,4,6}},DT_FLOAT,FORMAT_ND},{{{1,4,6},{1,4,6}},DT_FLOAT,FORMAT_ND},
          {{{1,4,6},{1,4,6}},DT_FLOAT,FORMAT_ND},{{{1,4,6},{1,4,6}},DT_FLOAT,FORMAT_ND},
          {{{1,4,6},{1,4,6}},DT_FLOAT,FORMAT_ND},{{{1,4,6},{1,4,6}},DT_FLOAT,FORMAT_ND},
          {{{1,4,6},{1,4,6}},DT_FLOAT,FORMAT_ND},{{{1,4,6},{1,4,6}},DT_FLOAT,FORMAT_ND},
          {{{1,4,6},{1,4,6}},DT_FLOAT,FORMAT_ND},{{{1,4,6},{1,4,6}},DT_FLOAT,FORMAT_ND}}},
        {{{{{1,4,6},{1,4,6}},DT_FLOAT,FORMAT_ND},{{{1,4,6},{1,4,6}},DT_FLOAT,FORMAT_ND},
          {{{1,4,6},{1,4,6}},DT_FLOAT,FORMAT_ND}}}, &ci);
    string expect = "0 4 1 4 1 1 1 0 2 52416 4 6 0 0 10 3 4 6 0 0 4 6 0 0 4 6 0 0 4 6 0 0 4 6 0 0 4 6 0 0 4 6 0 0 4 6 0
0 4 6 0 0 4 6 0 0 6 1 0 0 6 1 0 0 6 1 0 0 6 1 0 0 6 1 0 0 6 1 0 0 6 1 0 0 6 1 0 0 6 1 0 0 6 1 0 0 4 6 0 0 4 6 0 0 4 6 0
0 6 1 0 0 6 1 0 0 6 1 0 0 "; ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{16777216}});
}

// case3: 全标量 → max_bro_shape=(1)
TEST_F(AdamTilingTest, case3_all_scalar) {
    optiling::AdamCompileInfo ci = {56, 256*1024};
    gert::TilingContextPara ctx("AdamApplyOneAssign",
        {{{{{},{}},DT_FLOAT,FORMAT_ND},{{{},{}},DT_FLOAT,FORMAT_ND},{{{},{}},DT_FLOAT,FORMAT_ND},
          {{{},{}},DT_FLOAT,FORMAT_ND},{{{},{}},DT_FLOAT,FORMAT_ND},{{{},{}},DT_FLOAT,FORMAT_ND},
          {{{},{}},DT_FLOAT,FORMAT_ND},{{{},{}},DT_FLOAT,FORMAT_ND},{{{},{}},DT_FLOAT,FORMAT_ND},
          {{{},{}},DT_FLOAT,FORMAT_ND}}},
        {{{{{},{}},DT_FLOAT,FORMAT_ND},{{{},{}},DT_FLOAT,FORMAT_ND},{{{},{}},DT_FLOAT,FORMAT_ND}}}, &ci);
    string expect = "0 1 1 1 1 1 1 0 1 52416 1 0 0 0 10 3 1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1 0 0 0 1 0 0
0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 1 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 "; ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{16777216}});
}

// ============================================================
// FindSplitAxis (2 cases)
// ============================================================

// case4: 不整切 — (1,1,203,200) → axis=2, a_i=65, a_o=4, a_i_tail=8
TEST_F(AdamTilingTest, case4_non_aligned_split) {
    optiling::AdamCompileInfo ci = {56, 256*1024};
    gert::TilingContextPara ctx("AdamApplyOneAssign",
        {{{{{1,1,203,200},{1,1,203,200}},DT_FLOAT,FORMAT_ND},{{{1,1,203,200},{1,1,203,200}},DT_FLOAT,FORMAT_ND},
          {{{1,1,203,200},{1,1,203,200}},DT_FLOAT,FORMAT_ND},{{{1,1,203,200},{1,1,203,200}},DT_FLOAT,FORMAT_ND},
          {{{1,1,203,200},{1,1,203,200}},DT_FLOAT,FORMAT_ND},{{{1,1,203,200},{1,1,203,200}},DT_FLOAT,FORMAT_ND},
          {{{1,1,203,200},{1,1,203,200}},DT_FLOAT,FORMAT_ND},{{{1,1,203,200},{1,1,203,200}},DT_FLOAT,FORMAT_ND},
          {{{1,1,203,200},{1,1,203,200}},DT_FLOAT,FORMAT_ND},{{{1,1,203,200},{1,1,203,200}},DT_FLOAT,FORMAT_ND}}},
        {{{{{1,1,203,200},{1,1,203,200}},DT_FLOAT,FORMAT_ND},{{{1,1,203,200},{1,1,203,200}},DT_FLOAT,FORMAT_ND},
          {{{1,1,203,200},{1,1,203,200}},DT_FLOAT,FORMAT_ND}}}, &ci);
    string expect = "0 65 4 8 4 4 1 0 2 52416 203 200 0 0 10 3 203 200 0 0 203 200 0 0 203 200 0 0 203 200 0 0 203 200 0
0 203 200 0 0 203 200 0 0 203 200 0 0 203 200 0 0 203 200 0 0 200 1 0 0 200 1 0 0 200 1 0 0 200 1 0 0 200 1 0 0 200 1 0
0 200 1 0 0 200 1 0 0 200 1 0 0 200 1 0 0 203 200 0 0 203 200 0 0 203 200 0 0 200 1 0 0 200 1 0 0 200 1 0 0 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{16777216}});
}

// case5: 全量入UB — (3,16,32) 不触发切分
TEST_F(AdamTilingTest, case5_no_split) {
    optiling::AdamCompileInfo ci = {56, 256*1024};
    gert::TilingContextPara ctx("AdamApplyOneAssign",
        {{{{{3,16,32},{3,16,32}},DT_FLOAT,FORMAT_ND},{{{1,16,32},{1,16,32}},DT_FLOAT,FORMAT_ND},
          {{{1,16,32},{1,16,32}},DT_FLOAT,FORMAT_ND},{{{3,16,32},{3,16,32}},DT_FLOAT,FORMAT_ND},
          {{{3,16,32},{3,16,32}},DT_FLOAT,FORMAT_ND},{{{3,16,32},{3,16,32}},DT_FLOAT,FORMAT_ND},
          {{{1,16,32},{1,16,32}},DT_FLOAT,FORMAT_ND},{{{1,16,32},{1,16,32}},DT_FLOAT,FORMAT_ND},
          {{{3,16,32},{3,16,32}},DT_FLOAT,FORMAT_ND},{{{3,16,32},{3,16,32}},DT_FLOAT,FORMAT_ND}}},
        {{{{{3,16,32},{3,16,32}},DT_FLOAT,FORMAT_ND},{{{3,16,32},{3,16,32}},DT_FLOAT,FORMAT_ND},
          {{{3,16,32},{3,16,32}},DT_FLOAT,FORMAT_ND}}}, &ci);
    string expect = "0 3 1 3 1 1 1 0 3 52416 3 16 32 0 10 3 3 16 32 0 1 16 32 0 1 16 32 0 3 16 32 0 3 16 32 0 3 16 32 0
1 16 32 0 1 16 32 0 3 16 32 0 3 16 32 0 512 32 1 0 0 32 1 0 0 32 1 0 512 32 1 0 512 32 1 0 512 32 1 0 0 32 1 0 0 32 1 0
512 32 1 0 512 32 1 0 3 16 32 0 3 16 32 0 3 16 32 0 512 32 1 0 512 32 1 0 512 32 1 0 "; ExecuteTestCase(ctx,
GRAPH_SUCCESS, 0, expect, {{16777216}});
}

// ============================================================
// MultiCoreSplit (2 cases)
// ============================================================

// case6: 多核均分 — outer_prod=91, total_tiles=364, num_cores=56
TEST_F(AdamTilingTest, case6_multi_core) {
    optiling::AdamCompileInfo ci = {56, 256*1024};
    gert::TilingContextPara ctx("AdamApplyOneAssign",
        {{{{{7,13,203,200},{7,13,203,200}},DT_FLOAT,FORMAT_ND},{{{7,13,203,200},{7,13,203,200}},DT_FLOAT,FORMAT_ND},
          {{{7,13,203,200},{7,13,203,200}},DT_FLOAT,FORMAT_ND},{{{7,13,203,200},{7,13,203,200}},DT_FLOAT,FORMAT_ND},
          {{{7,13,203,200},{7,13,203,200}},DT_FLOAT,FORMAT_ND},{{{7,13,203,200},{7,13,203,200}},DT_FLOAT,FORMAT_ND},
          {{{7,13,203,200},{7,13,203,200}},DT_FLOAT,FORMAT_ND},{{{7,13,203,200},{7,13,203,200}},DT_FLOAT,FORMAT_ND},
          {{{7,13,203,200},{7,13,203,200}},DT_FLOAT,FORMAT_ND},{{{7,13,203,200},{7,13,203,200}},DT_FLOAT,FORMAT_ND}}},
        {{{{{7,13,203,200},{7,13,203,200}},DT_FLOAT,FORMAT_ND},{{{7,13,203,200},{7,13,203,200}},DT_FLOAT,FORMAT_ND},
          {{{7,13,203,200},{7,13,203,200}},DT_FLOAT,FORMAT_ND}}}, &ci);
    string expect = "2 65 4 8 56 364 6 28 4 52416 7 13 203 200 10 3 7 13 203 200 7 13 203 200 7 13 203 200 7 13 203 200
7 13 203 200 7 13 203 200 7 13 203 200 7 13 203 200 7 13 203 200 7 13 203 200 527800 40600 200 1 527800 40600 200 1
527800 40600 200 1 527800 40600 200 1 527800 40600 200 1 527800 40600 200 1 527800 40600 200 1 527800 40600 200 1 527800
40600 200 1 527800 40600 200 1 7 13 203 200 7 13 203 200 7 13 203 200 527800 40600 200 1 527800 40600 200 1 527800 40600
200 1 "; ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{16777216}});
}

// case7: 单核 — total_tiles=1, num_cores=1
TEST_F(AdamTilingTest, case7_single_core) {
    optiling::AdamCompileInfo ci = {56, 256*1024};
    gert::TilingContextPara ctx("AdamApplyOneAssign",
        {{{{{1,1,100,100},{1,1,100,100}},DT_FLOAT,FORMAT_ND},{{{1,1,100,100},{1,1,100,100}},DT_FLOAT,FORMAT_ND},
          {{{1,1,100,100},{1,1,100,100}},DT_FLOAT,FORMAT_ND},{{{1,1,100,100},{1,1,100,100}},DT_FLOAT,FORMAT_ND},
          {{{1,1,100,100},{1,1,100,100}},DT_FLOAT,FORMAT_ND},{{{1,1,100,100},{1,1,100,100}},DT_FLOAT,FORMAT_ND},
          {{{1,1,100,100},{1,1,100,100}},DT_FLOAT,FORMAT_ND},{{{1,1,100,100},{1,1,100,100}},DT_FLOAT,FORMAT_ND},
          {{{1,1,100,100},{1,1,100,100}},DT_FLOAT,FORMAT_ND},{{{1,1,100,100},{1,1,100,100}},DT_FLOAT,FORMAT_ND}}},
        {{{{{1,1,100,100},{1,1,100,100}},DT_FLOAT,FORMAT_ND},{{{1,1,100,100},{1,1,100,100}},DT_FLOAT,FORMAT_ND},
          {{{1,1,100,100},{1,1,100,100}},DT_FLOAT,FORMAT_ND}}}, &ci);
    string expect = "0 100 1 100 1 1 1 0 2 52416 100 100 0 0 10 3 100 100 0 0 100 100 0 0 100 100 0 0 100 100 0 0 100
100 0 0 100 100 0 0 100 100 0 0 100 100 0 0 100 100 0 0 100 100 0 0 100 1 0 0 100 1 0 0 100 1 0 0 100 1 0 0 100 1 0 0
100 1 0 0 100 1 0 0 100 1 0 0 100 1 0 0 100 1 0 0 100 100 0 0 100 100 0 0 100 100 0 0 100 1 0 0 100 1 0 0 100 1 0 0 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{16777216}});
}

// ============================================================
// Stride (2 cases)
// ============================================================

// case8: broadcast stride — 单轴broad + 多轴broad + 无broad 三种在一组
TEST_F(AdamTilingTest, case8_broadcast_strides) {
    optiling::AdamCompileInfo ci = {56, 256*1024};
    gert::TilingContextPara ctx("AdamApplyOneAssign",
        {{{{{4,6,195,200},{4,6,195,200}},DT_FLOAT,FORMAT_ND},{{{4,1,195,200},{4,1,195,200}},DT_FLOAT,FORMAT_ND},
          {{{1,1,1,200},{1,1,1,200}},DT_FLOAT,FORMAT_ND},{{{4,6,195,200},{4,6,195,200}},DT_FLOAT,FORMAT_ND},
          {{{4,6,195,200},{4,6,195,200}},DT_FLOAT,FORMAT_ND},{{{4,1,195,200},{4,1,195,200}},DT_FLOAT,FORMAT_ND},
          {{{1,1,1,200},{1,1,1,200}},DT_FLOAT,FORMAT_ND},{{{4,6,195,200},{4,6,195,200}},DT_FLOAT,FORMAT_ND},
          {{{4,6,195,200},{4,6,195,200}},DT_FLOAT,FORMAT_ND},{{{4,6,195,200},{4,6,195,200}},DT_FLOAT,FORMAT_ND}}},
        {{{{{4,6,195,200},{4,6,195,200}},DT_FLOAT,FORMAT_ND},{{{4,6,195,200},{4,6,195,200}},DT_FLOAT,FORMAT_ND},
          {{{4,6,195,200},{4,6,195,200}},DT_FLOAT,FORMAT_ND}}}, &ci);
    string expect = "2 65 3 65 56 72 1 16 4 52416 4 6 195 200 10 3 4 6 195 200 4 1 195 200 1 1 1 200 4 6 195 200 4 6 195
200 4 1 195 200 1 1 1 200 4 6 195 200 4 6 195 200 4 6 195 200 234000 39000 200 1 39000 0 200 1 0 0 0 1 234000 39000 200
1 234000 39000 200 1 39000 0 200 1 0 0 0 1 234000 39000 200 1 234000 39000 200 1 234000 39000 200 1 4 6 195 200 4 6 195
200 4 6 195 200 234000 39000 200 1 234000 39000 200 1 234000 39000 200 1 "; ExecuteTestCase(ctx, GRAPH_SUCCESS, 0,
expect, {{16777216}});
}

// case9: 无broadcast stride — 全量 strides 全非0
TEST_F(AdamTilingTest, case9_no_broadcast_strides) {
    optiling::AdamCompileInfo ci = {56, 256*1024};
    gert::TilingContextPara ctx("AdamApplyOneAssign",
        {{{{{4,6,195,200},{4,6,195,200}},DT_FLOAT,FORMAT_ND},{{{4,6,195,200},{4,6,195,200}},DT_FLOAT,FORMAT_ND},
          {{{4,6,195,200},{4,6,195,200}},DT_FLOAT,FORMAT_ND},{{{4,6,195,200},{4,6,195,200}},DT_FLOAT,FORMAT_ND},
          {{{4,6,195,200},{4,6,195,200}},DT_FLOAT,FORMAT_ND},{{{4,6,195,200},{4,6,195,200}},DT_FLOAT,FORMAT_ND},
          {{{4,6,195,200},{4,6,195,200}},DT_FLOAT,FORMAT_ND},{{{4,6,195,200},{4,6,195,200}},DT_FLOAT,FORMAT_ND},
          {{{4,6,195,200},{4,6,195,200}},DT_FLOAT,FORMAT_ND},{{{4,6,195,200},{4,6,195,200}},DT_FLOAT,FORMAT_ND}}},
        {{{{{4,6,195,200},{4,6,195,200}},DT_FLOAT,FORMAT_ND},{{{4,6,195,200},{4,6,195,200}},DT_FLOAT,FORMAT_ND},
          {{{4,6,195,200},{4,6,195,200}},DT_FLOAT,FORMAT_ND}}}, &ci);
    string expect = "2 65 3 65 56 72 1 16 4 52416 4 6 195 200 10 3 4 6 195 200 4 6 195 200 4 6 195 200 4 6 195 200 4 6
195 200 4 6 195 200 4 6 195 200 4 6 195 200 4 6 195 200 4 6 195 200 234000 39000 200 1 234000 39000 200 1 234000 39000
200 1 234000 39000 200 1 234000 39000 200 1 234000 39000 200 1 234000 39000 200 1 234000 39000 200 1 234000 39000 200 1
234000 39000 200 1 4 6 195 200 4 6 195 200 4 6 195 200 234000 39000 200 1 234000 39000 200 1 234000 39000 200 1 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{16777216}});
}

// ============================================================
// 贯通 (1 case)
// ============================================================

// case10: 10入3出全开 — broadcast混合+切分+多核
TEST_F(AdamTilingTest, case10_through) {
    optiling::AdamCompileInfo ci = {56, 256*1024};
    gert::TilingContextPara ctx("AdamApplyOneAssign",
        {{{{{7,13,203,200},{7,13,203,200}},DT_FLOAT,FORMAT_ND},{{{13,203,200},{13,203,200}},DT_FLOAT,FORMAT_ND},
          {{{203,200},{203,200}},DT_FLOAT,FORMAT_ND},{{{200},{200}},DT_FLOAT,FORMAT_ND},
          {{{7,1,203,200},{7,1,203,200}},DT_FLOAT,FORMAT_ND},{{{7,13,1,200},{7,13,1,200}},DT_FLOAT,FORMAT_ND},
          {{{7,1,1,200},{7,1,1,200}},DT_FLOAT,FORMAT_ND},{{{1,1,1,200},{1,1,1,200}},DT_FLOAT,FORMAT_ND},
          {{{7,13,203,200},{7,13,203,200}},DT_FLOAT,FORMAT_ND},{{{7,13,203,200},{7,13,203,200}},DT_FLOAT,FORMAT_ND}}},
        {{{{{7,13,203,200},{7,13,203,200}},DT_FLOAT,FORMAT_ND},{{{7,13,203,200},{7,13,203,200}},DT_FLOAT,FORMAT_ND},
          {{{7,13,203,200},{7,13,203,200}},DT_FLOAT,FORMAT_ND}}}, &ci);
    string expect = "2 65 4 8 56 364 6 28 4 52416 7 13 203 200 10 3 7 13 203 200 1 13 203 200 1 1 203 200 1 1 1 200 7 1
203 200 7 13 1 200 7 1 1 200 1 1 1 200 7 13 203 200 7 13 203 200 527800 40600 200 1 0 40600 200 1 0 0 200 1 0 0 0 1
40600 0 200 1 2600 200 0 1 200 0 0 1 0 0 0 1 527800 40600 200 1 527800 40600 200 1 7 13 203 200 7 13 203 200 7 13 203
200 527800 40600 200 1 527800 40600 200 1 527800 40600 200 1 "; ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect,
{{16777216}});
}
*/

// case0: 错题集 Case0 — 全同形 [256,128,64]，FP32
TEST_F(AdamTilingTest, case0_uniform_shape)
{
    optiling::AdamCompileInfo ci = {56, 256 * 1024};
    gert::TilingContextPara ctx("AdamApplyOneAssign",
                                {{{{{256, 128, 64}, {256, 128, 64}}, DT_FLOAT, FORMAT_ND},
                                  {{{256, 128, 64}, {256, 128, 64}}, DT_FLOAT, FORMAT_ND},
                                  {{{256, 128, 64}, {256, 128, 64}}, DT_FLOAT, FORMAT_ND},
                                  {{{256, 128, 64}, {256, 128, 64}}, DT_FLOAT, FORMAT_ND},
                                  {{{256, 128, 64}, {256, 128, 64}}, DT_FLOAT, FORMAT_ND},
                                  {{{256, 128, 64}, {256, 128, 64}}, DT_FLOAT, FORMAT_ND},
                                  {{{256, 128, 64}, {256, 128, 64}}, DT_FLOAT, FORMAT_ND},
                                  {{{256, 128, 64}, {256, 128, 64}}, DT_FLOAT, FORMAT_ND},
                                  {{{256, 128, 64}, {256, 128, 64}}, DT_FLOAT, FORMAT_ND},
                                  {{{256, 128, 64}, {256, 128, 64}}, DT_FLOAT, FORMAT_ND}}},
                                {{{{{256, 128, 64}, {256, 128, 64}}, DT_FLOAT, FORMAT_ND},
                                  {{{256, 128, 64}, {256, 128, 64}}, DT_FLOAT, FORMAT_ND},
                                  {{{256, 128, 64}, {256, 128, 64}}, DT_FLOAT, FORMAT_ND}}},
                                &ci);
    string expect = "1 1 256 1 56 256 4 32 3 52416 1 256 128 64 10 3 1 256 128 64 1 256 128 64 1 256 128 64 1 256 128 "
                    "64 1 256 128 64 1 256 128 64 1 256 128 64 1 256 128 64 1 256 128 64 1 256 128 64 0 8192 64 1 0 "
                    "8192 64 1 0 8192 64 1 0 8192 64 1 0 8192 64 1 0 8192 64 1 0 8192 64 1 0 8192 64 1 0 8192 64 1 0 "
                    "8192 64 1 1 256 128 64 1 256 128 64 1 256 128 64 0 8192 64 1 0 8192 64 1 0 8192 64 1 ";
    ExecuteTestCase(ctx, GRAPH_SUCCESS, 0, expect, {{16777216}});
}
