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
 *       输出 shape 在该 UT 框架中不由 infershape 回填, CheckParamsShape 会以 shape 契约
 *       拒绝并上报正式码 EZ0010; 用例校验失败返回、正式码以及不再产生 EZ9999/E89999。
 */

#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "op_host/arch22/gru_grad_tiling.h"

namespace {
using TensorDesc = gert::TilingContextPara::TensorDescription;

constexpr int64_t GATE_NUM = 3;
const std::vector<uint32_t> kInputIr{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0}; // 12 输入, batch_sizes 缺省
const std::vector<uint32_t> kOutputIr{1, 1, 1, 1, 1, 1};                  // 6 输出
} // namespace

// 通过弱符号截获 ErrorManager 上报，校验负向路径使用正式错误码且不再产生内部码。
namespace error_message {
static thread_local std::string g_last_code;
static thread_local std::string g_last_detail;
static thread_local std::vector<std::string> g_all_codes;

__attribute__((weak)) void ResetCapture()
{
    g_last_code.clear();
    g_last_detail.clear();
    g_all_codes.clear();
}
__attribute__((weak)) std::string GetLastError() { return g_last_code; }
__attribute__((weak)) std::string GetLastDetail() { return g_last_detail; }
__attribute__((weak)) bool HasCode(const std::string& c)
{
    for (const auto& code : g_all_codes) {
        if (code == c) {
            return true;
        }
    }
    return false;
}
__attribute__((weak)) int32_t ReportPredefinedErrMsg(const char* error_code, const std::vector<const char*>&,
                                                     const std::vector<const char*>& values)
{
    g_last_code = error_code ? error_code : "";
    if (error_code != nullptr) {
        g_all_codes.push_back(error_code);
    }
    std::string detail;
    for (const auto& v : values) {
        detail += std::string(v ? v : "") + " | ";
    }
    g_last_detail = detail;
    return 0;
}
__attribute__((weak)) int32_t ReportPredefinedErrMsg(const char* error_code)
{
    g_last_code = error_code ? error_code : "";
    if (error_code != nullptr) {
        g_all_codes.push_back(error_code);
    }
    return 0;
}
__attribute__((weak)) int32_t ReportInnerErrMsg(const char*, const char*, uint32_t, const char* error_code, const char*,
                                                ...)
{
    g_last_code = error_code ? error_code : "";
    if (error_code != nullptr) {
        g_all_codes.push_back(error_code);
    }
    return 0;
}
} // namespace error_message

class GruGradTiling : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// 校验: shape 契约不满足时 tiling 返回 GRAPH_FAILED 并上报正式码 EZ0010, 不产生内部码
TEST_F(GruGradTiling, gru_grad_tiling_invalid_shape_reports_formal_code)
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
        {"direction", Ops::NN::AnyValue::CreateFrom<std::string>("UNIDIRECTIONAL")},
    };

    optiling::GruGradCompileInfo compileInfo{};
    gert::TilingContextPara para("GruGrad", inputs, outputs, attrs, kInputIr, kOutputIr, &compileInfo, 1U, 196608ULL,
                                 8192U);
    error_message::ResetCapture();
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    EXPECT_EQ(error_message::GetLastError(), "EZ0010") << error_message::GetLastDetail();
    EXPECT_TRUE(error_message::HasCode("EZ0010"));
    EXPECT_FALSE(error_message::HasCode("EZ9999"));
    EXPECT_FALSE(error_message::HasCode("E89999"));
}
