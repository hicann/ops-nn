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

#include <cstdint>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "base/err_mgr.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "op_host/arch22/gru_grad_tiling.h"

namespace {
using TensorDesc = gert::TilingContextPara::TensorDescription;

constexpr int64_t GATE_NUM = 3;
const std::vector<uint32_t> kInputIr{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0}; // 12 输入, batch_sizes 缺省
const std::vector<uint32_t> kOutputIr{1, 1, 1, 1, 1, 1};                  // 6 输出

// 使用独立的 ErrorManager 上下文隔离本用例记录，避免共享 Host UT 进程内串扰。
// 不定义全局 weak 上报替身，否则会截获同一可执行文件中其他算子的错误。
class ScopedErrorContext {
public:
    ScopedErrorContext() : savedContext_(error_message::GetErrMgrContext())
    {
        auto context = savedContext_;
        context.work_stream_id = reinterpret_cast<std::uintptr_t>(this);
        error_message::SetErrMgrContext(context);
        (void)error_message::GetErrMgrRawErrorMessages();
    }

    ~ScopedErrorContext()
    {
        (void)error_message::GetErrMgrRawErrorMessages();
        error_message::SetErrMgrContext(savedContext_);
    }

private:
    error_message::ErrorManagerContext savedContext_;
};

struct CapturedError {
    size_t count = 0;
    std::string code;
    std::vector<std::string> keys;
    std::vector<std::string> values;
};

// EZ0010 由 third_party/opbase 的格式注册对象提供；单算子 UT 构建时该对象可能因静态库
// 按需链接未被拉入。这里在用例内显式注册一次，避免改动公共 UT 配置或引入 weak 桩。
constexpr const char kEz0010ErrorMessage[] = R"json({"error_info_list":[
    {"errClass":"Operator Errors","errTitle":"Invalid_Argument_Tensor_Shape",
     "ErrCode":"EZ0010",
     "ErrMessage":"Parameters %s of %s have incorrect shapes %s. Reason: %s.",
     "Arglist":"param_names, op_name, incorrect_shapes, reason",
     "suggestion":{"Possible Cause":"N/A","Solution":"Check whether the shapes of input/output tensors meet the relationship."}}
]})json";

CapturedError CaptureError()
{
    CapturedError captured;
    const auto errors = error_message::GetErrMgrRawErrorMessages();
    captured.count = errors.size();
    if (errors.empty()) {
        return captured;
    }
    const auto& error = errors.back();
    captured.code = error.error_id == nullptr ? "" : error.error_id.get();
    for (const auto& key : error.args_key) {
        captured.keys.emplace_back(key == nullptr ? "" : key.get());
    }
    for (const auto& value : error.args_value) {
        captured.values.emplace_back(value == nullptr ? "" : value.get());
    }
    return captured;
}
} // namespace

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
    (void)error_message::RegisterFormatErrorMessage(kEz0010ErrorMessage, sizeof(kEz0010ErrorMessage) - 1);
    ScopedErrorContext errorContext;
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    const auto error = CaptureError();
    ASSERT_EQ(error.count, 1U);
    EXPECT_EQ(error.code, "EZ0010");
    // 截获的是真实 ErrorManager 记录，共享可执行文件中不再产生内部兜底码。
    EXPECT_NE(error.code, "EZ9999");
    EXPECT_NE(error.code, "E89999");
}
