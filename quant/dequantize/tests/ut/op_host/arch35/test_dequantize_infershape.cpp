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
 * \file test_dequantize_infershape.cpp
 * \brief Dequantize 图模式 infershape UT：unknown rank(-2)、unknown shape(-1)、broadcast 校验。
 */

#include <gtest/gtest.h>

#include "infershape_case_executor.h"

class DequantizeInferShapeTest : public testing::Test {};

// 已知 shape：y 与 x 相同
TEST_F(DequantizeInferShapeTest, known_shape)
{
    gert::InfershapeContextPara para("Dequantize",
                                     {
                                         {{{2, 3}, {2, 3}}, ge::DT_INT8, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 3}});
}

// x unknown rank：输出透传 [-2]，跳过 broadcast 校验
TEST_F(DequantizeInferShapeTest, x_unknown_rank)
{
    gert::InfershapeContextPara para("Dequantize",
                                     {
                                         {{{-2}, {-2}}, ge::DT_INT8, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{-2}});
}

// min unknown rank：输出 x，跳过 broadcast 校验
TEST_F(DequantizeInferShapeTest, min_unknown_rank)
{
    gert::InfershapeContextPara para("Dequantize",
                                     {
                                         {{{2, 3}, {2, 3}}, ge::DT_INT8, ge::FORMAT_ND},
                                         {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 3}});
}

// max unknown rank：输出 x，跳过 broadcast 校验
TEST_F(DequantizeInferShapeTest, max_unknown_rank)
{
    gert::InfershapeContextPara para("Dequantize",
                                     {
                                         {{{2, 3}, {2, 3}}, ge::DT_INT8, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 3}});
}

// min/max 都 unknown rank：输出 x
TEST_F(DequantizeInferShapeTest, both_range_unknown_rank)
{
    gert::InfershapeContextPara para("Dequantize",
                                     {
                                         {{{2, 3}, {2, 3}}, ge::DT_INT8, ge::FORMAT_ND},
                                         {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 3}});
}

// min unknown rank 且 max 已知不兼容：短路返回成功，不校验 max
TEST_F(DequantizeInferShapeTest, min_unknown_rank_short_circuits_incompatible_max)
{
    gert::InfershapeContextPara para("Dequantize",
                                     {
                                         {{{2, 3}, {2, 3}}, ge::DT_INT8, ge::FORMAT_ND},
                                         {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 3}});
}

// x unknown rank，min/max 已知不兼容：短路返回成功，输出 [-2]
TEST_F(DequantizeInferShapeTest, x_unknown_rank_short_circuits_incompatible_ranges)
{
    gert::InfershapeContextPara para("Dequantize",
                                     {
                                         {{{-2}, {-2}}, ge::DT_INT8, ge::FORMAT_ND},
                                         {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{5}, {5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{-2}});
}

// x unknown shape(-1)：-1 维透传，其余维校验
TEST_F(DequantizeInferShapeTest, x_unknown_shape)
{
    gert::InfershapeContextPara para("Dequantize",
                                     {
                                         {{{-1, 3}, {-1, 3}}, ge::DT_INT8, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{-1, 3}});
}

// min/max unknown shape(-1)：-1 维跳过，其余维校验
TEST_F(DequantizeInferShapeTest, range_unknown_shape)
{
    gert::InfershapeContextPara para("Dequantize",
                                     {
                                         {{{2, 3}, {2, 3}}, ge::DT_INT8, ge::FORMAT_ND},
                                         {{{-1, 1}, {-1, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{1, -1}, {1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{2, 3}});
}

// 已知不兼容：min 无法广播
TEST_F(DequantizeInferShapeTest, known_incompatible_min)
{
    gert::InfershapeContextPara para("Dequantize",
                                     {
                                         {{{2, 3}, {2, 3}}, ge::DT_INT8, ge::FORMAT_ND},
                                         {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

// 已知不兼容：max 无法广播
TEST_F(DequantizeInferShapeTest, known_incompatible_max)
{
    gert::InfershapeContextPara para("Dequantize",
                                     {
                                         {{{2, 3}, {2, 3}}, ge::DT_INT8, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

// x unknown shape 但已知维不兼容：最后一维 3 vs 4
TEST_F(DequantizeInferShapeTest, x_unknown_shape_known_dim_incompatible)
{
    gert::InfershapeContextPara para("Dequantize",
                                     {
                                         {{{-1, 3}, {-1, 3}}, ge::DT_INT8, ge::FORMAT_ND},
                                         {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

// rank 0 标量
TEST_F(DequantizeInferShapeTest, scalar_shape)
{
    gert::InfershapeContextPara para("Dequantize",
                                     {
                                         {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                     });
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{}});
}
