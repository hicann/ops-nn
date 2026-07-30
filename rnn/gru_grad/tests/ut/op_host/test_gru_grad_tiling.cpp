/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_gru_grad_tiling.cpp
 * \brief GRU 反向算子 (GruGrad) host 侧 tiling 单元测试
 *
 * 说明: 正例 (tiling 成功) 路径会进入 matmul tiling, 该库在 UT 伪平台下会段错误
 *       (CANN 框架限制, 非本算子问题), 因此本用例覆盖 tiling 校验路径:
 *       传入不支持的 direction=BIDIRECTIONAL, 验证 CheckAttr 正确拒绝并返回 GRAPH_FAILED。
 */

#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "op_host/gru_grad_tiling.h"

namespace {
using TensorDesc = gert::TilingContextPara::TensorDescription;

constexpr int64_t GATE_NUM = 3;
const std::vector<uint32_t> kInputIr{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0}; // 12 输入, batch_sizes 缺省
const std::vector<uint32_t> kOutputIr{1, 1, 1, 1, 1, 1};                  // 6 输出
} // namespace

class GruGradTiling : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// 校验: direction=BIDIRECTIONAL 不被支持, tiling 应返回 GRAPH_FAILED
TEST_F(GruGradTiling, gru_grad_tiling_unsupported_direction)
{
    int64_t T = 4;
    int64_t B = 8;
    int64_t I = 16;
    int64_t H = 32;
    int64_t threeH = GATE_NUM * H;
    ge::DataType dtype = ge::DT_FLOAT;

    std::vector<TensorDesc> inputs = {
        {{{T, B, I}, {T, B, I}}, dtype, ge::FORMAT_ND},     {{{threeH, I}, {threeH, I}}, dtype, ge::FORMAT_ND},
        {{{threeH, H}, {threeH, H}}, dtype, ge::FORMAT_ND}, {{{1, B, H}, {1, B, H}}, dtype, ge::FORMAT_ND},
        {{{T, B, H}, {T, B, H}}, dtype, ge::FORMAT_ND},     {{{T, B, H}, {T, B, H}}, dtype, ge::FORMAT_ND},
        {{{T, B, H}, {T, B, H}}, dtype, ge::FORMAT_ND},     {{{T, B, H}, {T, B, H}}, dtype, ge::FORMAT_ND},
        {{{T, B, H}, {T, B, H}}, dtype, ge::FORMAT_ND},     {{{T, B, H}, {T, B, H}}, dtype, ge::FORMAT_ND},
        {{{1, B, H}, {1, B, H}}, dtype, ge::FORMAT_ND},     {{{}, {}}, dtype, ge::FORMAT_ND},
    };
    std::vector<TensorDesc> outputs = {
        {{{T, B, I}, {T, B, I}}, dtype, ge::FORMAT_ND},     {{{1, B, H}, {1, B, H}}, dtype, ge::FORMAT_ND},
        {{{threeH, I}, {threeH, I}}, dtype, ge::FORMAT_ND}, {{{threeH, H}, {threeH, H}}, dtype, ge::FORMAT_ND},
        {{{threeH}, {threeH}}, dtype, ge::FORMAT_ND},       {{{threeH}, {threeH}}, dtype, ge::FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"direction", Ops::NN::AnyValue::CreateFrom<std::string>("BIDIRECTIONAL")},
    };

    optiling::GruGradCompileInfo compileInfo{};
    gert::TilingContextPara para("GruGrad", inputs, outputs, attrs, kInputIr, kOutputIr, &compileInfo, 1U, 196608ULL,
                                 8192U);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
