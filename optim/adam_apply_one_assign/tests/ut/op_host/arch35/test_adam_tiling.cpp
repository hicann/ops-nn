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
#include "experiment/metadef/common/util/error_manager/error_manager.h"
#include "tiling_context_faker.h"
#include "register/op_impl_registry.h"
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

namespace {
void CheckNormalizedShape(const std::vector<std::vector<int64_t>>& inputDims, const std::vector<int64_t>& expectedShape,
                          ge::DataType dtype, const std::vector<std::vector<int64_t>>& expectedInputs = {})
{
    auto implementation = gert::OpImplRegistry::GetInstance().GetOpImpl("AdamApplyOneAssign");
    ASSERT_NE(implementation, nullptr);
    ASSERT_NE(implementation->tiling, nullptr);
    std::vector<gert::StorageShape> shapes(inputDims.size());
    std::vector<gert::StorageShape*> inputs;
    for (size_t i = 0; i < inputDims.size(); ++i) {
        for (auto dim : inputDims[i]) {
            shapes[i].MutableOriginShape().AppendDim(dim);
            shapes[i].MutableStorageShape().AppendDim(dim);
        }
        inputs.push_back(&shapes[i]);
    }
    const auto originalShapes = shapes;
    optiling::AdamCompileInfo compileInfo{56, 253952};
    auto data = gert::TilingData::CreateCap(4096);
    auto workspace = gert::ContinuousVector::Create<size_t>(1);
    fe::PlatFormInfos platform;
    ASSERT_TRUE(platform.Init());
    gert::TilingContextFaker builder;
    builder.SetOpType("AdamApplyOneAssign")
        .NodeIoNum(10, 3)
        .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
        .CompileInfo(&compileInfo)
        .InputShapes(inputs)
        .OutputShapes({&shapes[1], &shapes[2], &shapes[3]})
        .PlatformInfo(&platform)
        .TilingData(data.get())
        .Workspace(reinterpret_cast<gert::ContinuousVector*>(workspace.get()));
    for (size_t i = 0; i < inputs.size(); ++i) {
        builder.NodeInputTd(i, dtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    for (size_t i = 0; i < 3; ++i) {
        builder.NodeOutputTd(i, dtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = builder.Build();
    auto context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(implementation->tiling(context), ge::GRAPH_SUCCESS);
    auto check = [&](const auto* tiling, size_t rank) {
        EXPECT_EQ(tiling->rank, expectedShape.size());
        const size_t padding = rank - expectedShape.size();
        for (size_t d = 0; d < rank; ++d) {
            const int64_t dim = d < padding ? 1 : expectedShape[d - padding];
            EXPECT_EQ(tiling->max_bro_shape[d], dim);
            for (size_t i = 0; i < inputDims.size(); ++i) {
                const auto& expected = expectedInputs.empty() ? expectedShape : expectedInputs[i];
                EXPECT_EQ(tiling->input_shapes[i][d], d < padding ? 1 : expected[d - padding]);
            }
            for (size_t i = 0; i < 3; ++i) {
                EXPECT_EQ(tiling->output_shapes[i][d], dim);
            }
        }
        EXPECT_GT(tiling->split.a_i, 0);
        EXPECT_GT(tiling->split.a_i_tail, 0);
        EXPECT_LE(tiling->split.a_i_tail, tiling->split.a_i);
    };
    const auto* raw = context->GetRawTilingData();
    if (expectedShape.size() <= 4) {
        ASSERT_EQ(raw->GetDataSize(), sizeof(AdamTilingData<4>));
        check(static_cast<const AdamTilingData<4>*>(raw->GetData()), 4);
    } else {
        ASSERT_EQ(raw->GetDataSize(), sizeof(AdamTilingData<8>));
        check(static_cast<const AdamTilingData<8>*>(raw->GetData()), 8);
    }
    EXPECT_EQ(shapes, originalShapes);
}
} // namespace

TEST_F(AdamTilingTest, case0_uniform_shape)
{
    CheckNormalizedShape(std::vector<std::vector<int64_t>>(10, {256, 128, 64}), {2097152}, ge::DT_FLOAT);
}

TEST_F(AdamTilingTest, validation_reports_specific_error_once)
{
    auto implementation = gert::OpImplRegistry::GetInstance().GetOpImpl("AdamApplyOneAssign");
    ASSERT_NE(implementation, nullptr);
    ASSERT_NE(implementation->tiling, nullptr);
    auto& errors = ErrorManager::GetInstance();
    ASSERT_EQ(errors.Init(), 0);
    const char* expectedCodes[] = {"EZ0019", "EZ0018", "EZ0012", "EZ0010", "EZ0004"};
    for (size_t scenario = 0; scenario < 5; ++scenario) {
        SCOPED_TRACE(scenario);
        gert::StorageShape tensor = {{2, 3}, {2, 3}};
        gert::StorageShape state = {{1, 3}, {1, 3}};
        gert::StorageShape rankNine = {{1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}};
        std::vector<gert::StorageShape*> inputs(10, &tensor);
        if (scenario == 2) {
            inputs[4] = &rankNine;
        } else if (scenario == 3) {
            inputs[1] = &state;
        }
        optiling::AdamCompileInfo compileInfo{64, 256 * 1024};
        auto data = gert::TilingData::CreateCap(4096);
        auto workspace = gert::ContinuousVector::Create<size_t>(1);
        fe::PlatFormInfos platform;
        ASSERT_TRUE(platform.Init());
        gert::TilingContextFaker builder;
        builder.SetOpType("AdamApplyOneAssign")
            .NodeIoNum(10, 3)
            .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
            .CompileInfo(&compileInfo)
            .InputShapes(inputs)
            .OutputShapes({&tensor, &tensor, &tensor})
            .PlatformInfo(&platform)
            .TilingData(data.get())
            .Workspace(reinterpret_cast<gert::ContinuousVector*>(workspace.get()));
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto dtype = scenario == 0 && i == 0 ? ge::DT_INT32 : ge::DT_FLOAT;
            auto format = scenario == 1 && i == 4 ? ge::FORMAT_NCHW : ge::FORMAT_ND;
            builder.NodeInputTd(i, dtype, format, format);
        }
        for (size_t i = 0; i < 3; ++i) {
            builder.NodeOutputTd(i, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
        }
        auto holder = builder.Build();
        auto context = holder.GetContext<gert::TilingContext>();
        ASSERT_NE(context, nullptr);
        if (scenario == 4) {
            auto input = holder.GetContext<gert::KernelContext>()->MutableInput(5);
            ASSERT_NE(input, nullptr);
            input->Set(nullptr, nullptr);
            ASSERT_EQ(context->GetInputShape(5), nullptr);
        }
        errors.GetRawErrorMessages();
        EXPECT_EQ(implementation->tiling(context), ge::GRAPH_FAILED);
        auto reports = errors.GetRawErrorMessages();
        ASSERT_EQ(reports.size(), 1U);
        EXPECT_EQ(reports[0].error_id, expectedCodes[scenario]);
        EXPECT_NE(reports[0].error_message.find("AdamApplyOneAssign"), std::string::npos);
    }
}

TEST_F(AdamTilingTest, null_context_reports_stage_error_once)
{
    auto implementation = gert::OpImplRegistry::GetInstance().GetOpImpl("AdamApplyOneAssign");
    ASSERT_NE(implementation, nullptr);
    ASSERT_NE(implementation->tiling, nullptr);
    auto& errors = ErrorManager::GetInstance();
    ASSERT_EQ(errors.Init(), 0);
    errors.GetRawErrorMessages();
    EXPECT_EQ(implementation->tiling(nullptr), ge::GRAPH_FAILED);
    auto reports = errors.GetRawErrorMessages();
    ASSERT_EQ(reports.size(), 1U);
    EXPECT_EQ(reports[0].error_id, "EZ1008");
}

TEST_F(AdamTilingTest, dense_inputs_coalesce_for_every_supported_rank_and_dtype)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT}) {
        for (size_t rank = 0; rank <= 8; ++rank) {
            SCOPED_TRACE(rank);
            SCOPED_TRACE(static_cast<int>(dtype));
            std::vector<int64_t> shape(rank, 2);
            int64_t count = 1;
            if (rank != 0) {
                shape.back() = 5;
                count = 5;
                for (size_t axis = 1; axis < rank; ++axis) {
                    count *= 2;
                }
            }
            CheckNormalizedShape(std::vector<std::vector<int64_t>>(10, shape), {count}, dtype);
        }
    }
}

TEST_F(AdamTilingTest, broadcast_roles_merge_only_compatible_adjacent_axes)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT}) {
        std::vector<std::vector<int64_t>> inputs(10, {2, 3, 5, 7});
        inputs[4] = {1, 1, 5, 7};
        inputs[6] = {};
        std::vector<std::vector<int64_t>> expected(10, {6, 35});
        expected[4] = {1, 35};
        expected[6] = {1, 1};
        CheckNormalizedShape(inputs, {6, 35}, dtype, expected);

        inputs.assign(10, {2, 3, 5, 7});
        expected.assign(10, {210});
        for (size_t i = 4; i < 10; ++i) {
            inputs[i] = {};
            expected[i] = {1};
        }
        CheckNormalizedShape(inputs, {210}, dtype, expected);
    }
}

TEST_F(AdamTilingTest, alternating_broadcast_roles_preserve_rank_eight)
{
    std::vector<std::vector<int64_t>> inputs(10, {2, 3, 2, 3, 2, 3, 2, 3});
    inputs[4] = {2, 1, 2, 1, 2, 1, 2, 1};
    CheckNormalizedShape(inputs, {2, 3, 2, 3, 2, 3, 2, 3}, ge::DT_FLOAT16, inputs);
}

TEST_F(AdamTilingTest, singleton_axes_and_full_issue59_shape_coalesce)
{
    CheckNormalizedShape(std::vector<std::vector<int64_t>>(10, {1, 2, 1, 3, 1, 5, 1, 7}), {210}, ge::DT_FLOAT);
    CheckNormalizedShape(std::vector<std::vector<int64_t>>(10, {2, 2, 351971, 2, 2, 2, 2, 2}), {45052288},
                         ge::DT_FLOAT16);
}
