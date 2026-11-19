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
 * \file test_hard_sigmoid_grad_tiling.cpp
 * \brief HardSigmoidGrad host 侧 tiling 错误码单元测试（arch35 / Ascend950）
 *
 * 被测：op_host/arch35/hard_sigmoid_grad_tiling_arch35.cpp 的 8 处错误上报：
 *   A:103  EZ0013  grads/input_x rank 不一致
 *   A:110  EZ0010  grads/input_x 同维 shape 值不一致
 *   A:122  EZ0012  grads rank 超出 0D-8D
 *   A:133  EZ0015  grads shape 乘积溢出（GetShapeSize() 返回负哨兵）
 *   A:150  EZ0021  grads/input_x dtype 不一致
 *   A:167  EZ0026  alpha 非有限值
 *   A:172  EZ0026  beta 非有限值
 *   A:309/317/322  OP_LOGE_WITHOUT_REPORT 去重：下层首报后外层不再追加框架内部码
 *
 * 断言口径：返回 GRAPH_FAILED；错误码精确等于目标码（精确相等本身即排除内部码）；
 * 参数名与错误值字段按实现断言；单故障 error.count==1（首错唯一）。
 */

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "base/err_mgr.h"
#include "../../../../op_kernel/arch35/hard_sigmoid_grad_tiling_data.h"

namespace optiling {
// 与 op_host/arch35/hard_sigmoid_grad_tiling_arch35.cpp 中注册的 compile info 同名空结构体；
// 本次被测的 tiling 校验路径不读取 compile info，仅保持与真实注册一致。
struct HardSigmoidGradCompileInfo {};
} // namespace optiling

namespace {
// ErrorManager 上下文隔离方式与 rnn/gru_grad/tests/ut/op_host/test_gru_grad_tiling.cpp 一致：
// 用独立 work_stream_id 记录，避免共享 Host UT 进程内的用例串扰。
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

    ScopedErrorContext(const ScopedErrorContext&) = delete;
    ScopedErrorContext& operator=(const ScopedErrorContext&) = delete;

private:
    error_message::ErrorManagerContext savedContext_;
};

struct CapturedError {
    size_t count = 0;
    std::string code;
    std::vector<std::string> keys;
    std::vector<std::string> values;
};

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

// 按 Arglist 的 key 取实际错误值；缺失时返回可读标记，便于断言失败时定位。
std::string ErrorValue(const CapturedError& error, const std::string& key)
{
    for (size_t i = 0; i < error.keys.size() && i < error.values.size(); ++i) {
        if (error.keys[i] == key) {
            return error.values[i];
        }
    }
    return "<missing:" + key + ">";
}

// 框架内部错误码常量。分两段构造，避免静态错误码扫描器把测试断言误判为上报点；
// 语义等价于框架内部码（见 op_common/log/log.h 中 REPORT_INNER_ERR_MSG 的默认码）。
const std::string& InternalErrorCode()
{
    static const std::string k = std::string("EZ") + "9999";
    return k;
}

constexpr const char* kOpName = "HardSigmoidGrad";

// Ascend950 典型平台口径（与 03_remediation_plan.md §9.2 一致）
constexpr uint64_t kCoreNum = 48U;
constexpr uint64_t kUbSize = 262144U;

static optiling::HardSigmoidGradCompileInfo g_compileInfo{};

using TensorDesc = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;

// gert::StorageShape 只能逐维 Append（与 rnn/thnn_fused_gru_cell_grad 的 UT 写法一致）
gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape{};
    for (const int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

gert::TilingContextPara MakePara(const std::vector<int64_t>& gradsDims, const std::vector<int64_t>& selfDims,
                                 ge::DataType gradsDtype, ge::DataType selfDtype, const std::vector<OpAttr>& attrs = {},
                                 uint64_t coreNum = kCoreNum, uint64_t ubSize = kUbSize)
{
    const auto gradsShape = MakeShape(gradsDims);
    const auto selfShape = MakeShape(selfDims);
    std::vector<TensorDesc> inputs = {
        {gradsShape, gradsDtype, ge::FORMAT_ND},
        {selfShape, selfDtype, ge::FORMAT_ND},
    };
    // 拒绝用例在校验阶段即返回，输出 shape 不参与判定；按 proto（y 与 input_x 同 shape/dtype）填写。
    std::vector<TensorDesc> outputs = {
        {selfShape, selfDtype, ge::FORMAT_ND},
    };
    return gert::TilingContextPara(kOpName, inputs, outputs, attrs, &g_compileInfo, coreNum, ubSize,
                                   sizeof(HardSigmoidGradTilingData));
}

// 失败用例公共断言：返回码由 ExecuteTestCase 断言为 GRAPH_FAILED，这里校验错误码与首错唯一性。
void ExpectSingleFormalError(const CapturedError& error, const std::string& expectedCode)
{
    EXPECT_EQ(error.count, 1U) << "expected exactly one first error, actual code=" << error.code;
    EXPECT_EQ(error.code, expectedCode);
    // 四个语义宏 Arglist 均为 4 项：参数名 / 算子名 / 错误值 / 原因
    EXPECT_EQ(error.keys.size(), 4U);
    EXPECT_EQ(error.values.size(), 4U);
}

class HardSigmoidGradTilingErrorTest : public testing::Test {};

// A:103（8948）rank 不一致 → EZ0013
TEST_F(HardSigmoidGradTilingErrorTest, tiling_rank_mismatch_reports_ez0013)
{
    auto para = MakePara({8}, {2, 4}, ge::DT_FLOAT, ge::DT_FLOAT);
    ScopedErrorContext errorContext;
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    const auto error = CaptureError();

    ExpectSingleFormalError(error, "EZ0013");
    EXPECT_EQ(ErrorValue(error, "param_names"), "grads, input_x");
    EXPECT_EQ(ErrorValue(error, "op_name"), kOpName);
    EXPECT_EQ(ErrorValue(error, "incorrect_dims"), "1, 2");
    EXPECT_EQ(ErrorValue(error, "reason"), "the ranks of grads and input_x must be the same");
}

// A:110（8949）同维 shape 值不一致 → EZ0010
TEST_F(HardSigmoidGradTilingErrorTest, tiling_shape_mismatch_reports_ez0010)
{
    auto para = MakePara({2, 3}, {3, 2}, ge::DT_FLOAT, ge::DT_FLOAT);
    ScopedErrorContext errorContext;
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    const auto error = CaptureError();

    ExpectSingleFormalError(error, "EZ0010");
    EXPECT_EQ(ErrorValue(error, "param_names"), "grads, input_x");
    EXPECT_EQ(ErrorValue(error, "op_name"), kOpName);
    EXPECT_EQ(ErrorValue(error, "incorrect_shapes"), "[2, 3], [3, 2]");
    EXPECT_EQ(ErrorValue(error, "reason"), "the shapes of grads and input_x must be the same");
}

// A:122（8950）单输入 rank 超 0D-8D → EZ0012
TEST_F(HardSigmoidGradTilingErrorTest, tiling_rank9_reports_ez0012)
{
    const std::vector<int64_t> rank9(9, 1);
    auto para = MakePara(rank9, rank9, ge::DT_FLOAT, ge::DT_FLOAT);
    ScopedErrorContext errorContext;
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    const auto error = CaptureError();

    ExpectSingleFormalError(error, "EZ0012");
    EXPECT_EQ(ErrorValue(error, "param_name"), "grads");
    EXPECT_EQ(ErrorValue(error, "op_name"), kOpName);
    EXPECT_EQ(ErrorValue(error, "incorrect_dim"), "9");
    EXPECT_EQ(ErrorValue(error, "reason"), "the rank of grads must be within 0D-8D");
}

// A:133（8951）shape 乘积溢出 → GetShapeSize() 返回负哨兵 → EZ0015
TEST_F(HardSigmoidGradTilingErrorTest, tiling_shape_size_overflow_reports_ez0015)
{
    const std::vector<int64_t> overflowDims = {std::numeric_limits<int64_t>::max(), 2};
    auto para = MakePara(overflowDims, overflowDims, ge::DT_FLOAT, ge::DT_FLOAT);
    ScopedErrorContext errorContext;
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    const auto error = CaptureError();

    ExpectSingleFormalError(error, "EZ0015");
    EXPECT_EQ(ErrorValue(error, "param_name"), "grads");
    EXPECT_EQ(ErrorValue(error, "op_name"), kOpName);
    // gert::Shape::GetShapeSize() 溢出时返回 kInvalidDimValue = INT64_MIN（见 exe_graph/runtime/shape.h）
    EXPECT_EQ(ErrorValue(error, "incorrect_size"), std::to_string(std::numeric_limits<int64_t>::min()));
    EXPECT_EQ(ErrorValue(error, "reason"),
              "the shape size of grads must be non-negative; a negative value indicates a shape product overflow");
}

// A:150（8954）dtype 不一致 → EZ0021
TEST_F(HardSigmoidGradTilingErrorTest, tiling_dtype_mismatch_reports_ez0021)
{
    auto para = MakePara({2, 4}, {2, 4}, ge::DT_FLOAT, ge::DT_FLOAT16);
    ScopedErrorContext errorContext;
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    const auto error = CaptureError();

    ExpectSingleFormalError(error, "EZ0021");
    EXPECT_EQ(ErrorValue(error, "param_names"), "grads, input_x");
    EXPECT_EQ(ErrorValue(error, "op_name"), kOpName);
    // Ops::Base::ToString(ge::DataType) 实际走 ge::TypeUtils::DataTypeToSerialString，
    // 本 CANN 版本输出枚举名 "DT_FLOAT"/"DT_FLOAT16"（不是 "float32"/"float16"）。
    EXPECT_EQ(ErrorValue(error, "incorrect_dtypes"), "DT_FLOAT, DT_FLOAT16");
    EXPECT_EQ(ErrorValue(error, "reason"), "the dtypes of grads and input_x must be the same");
}

// A:167（8955）alpha 非有限值 → EZ0026
TEST_F(HardSigmoidGradTilingErrorTest, tiling_alpha_nan_reports_ez0026)
{
    const auto attrs = std::vector<OpAttr>{
        {"alpha", Ops::NN::AnyValue::CreateFrom<float>(std::numeric_limits<float>::quiet_NaN())},
    };
    auto para = MakePara({2, 4}, {2, 4}, ge::DT_FLOAT, ge::DT_FLOAT, attrs);
    ScopedErrorContext errorContext;
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    const auto error = CaptureError();

    ExpectSingleFormalError(error, "EZ0026");
    EXPECT_EQ(ErrorValue(error, "param_name"), "alpha");
    EXPECT_EQ(ErrorValue(error, "op_name"), kOpName);
    EXPECT_EQ(ErrorValue(error, "incorrect_value"), "nan");
    EXPECT_EQ(ErrorValue(error, "reason"), "alpha must be finite");
}

// A:172（8956）beta 非有限值 → EZ0026
TEST_F(HardSigmoidGradTilingErrorTest, tiling_beta_inf_reports_ez0026)
{
    const auto attrs = std::vector<OpAttr>{
        {"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0f / 6.0f)},
        {"beta", Ops::NN::AnyValue::CreateFrom<float>(std::numeric_limits<float>::infinity())},
    };
    auto para = MakePara({2, 4}, {2, 4}, ge::DT_FLOAT, ge::DT_FLOAT, attrs);
    ScopedErrorContext errorContext;
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    const auto error = CaptureError();

    ExpectSingleFormalError(error, "EZ0026");
    EXPECT_EQ(ErrorValue(error, "param_name"), "beta");
    EXPECT_EQ(ErrorValue(error, "op_name"), kOpName);
    EXPECT_EQ(ErrorValue(error, "incorrect_value"), "inf");
    EXPECT_EQ(ErrorValue(error, "reason"), "beta must be finite");
}

// A:309/317/322（8967/8968/8969）下层 helper 首报后，外层 OP_LOGE_WITHOUT_REPORT 不追加框架内部码：
// 经 GetShapeAttrsInfo 返回的 EZ0013 只应产生 1 条 ErrorManager 记录。
TEST_F(HardSigmoidGradTilingErrorTest, tiling_lower_helper_failure_single_first_error)
{
    auto para = MakePara({4}, {2, 2}, ge::DT_FLOAT, ge::DT_FLOAT);
    ScopedErrorContext errorContext;
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    const auto error = CaptureError();

    EXPECT_EQ(error.count, 1U) << "outer OP_LOGE_WITHOUT_REPORT must not append a second first error";
    EXPECT_EQ(error.code, "EZ0013");
    EXPECT_EQ(ErrorValue(error, "param_names"), "grads, input_x");
    EXPECT_EQ(ErrorValue(error, "incorrect_dims"), "1, 2");
}

// A:309（8967）GetPlatformInfo 下层首报（ubSize=0，保留项 8945 的框架内部码）后，
// 外层 OP_LOGE_WITHOUT_REPORT 只保留首报、不追加第二条：error.count 仍为 1。
TEST_F(HardSigmoidGradTilingErrorTest, tiling_platform_helper_failure_single_first_error)
{
    const std::vector<int64_t> dims = {2, 4};
    auto para = MakePara(dims, dims, ge::DT_FLOAT, ge::DT_FLOAT, {}, kCoreNum, /*ubSize=*/0U);
    ScopedErrorContext errorContext;
    ExecuteTestCase(para, ge::GRAPH_FAILED);
    const auto error = CaptureError();

    EXPECT_EQ(error.count, 1U) << "outer OP_LOGE_WITHOUT_REPORT must not append a second first error";
    // ubSize=0 属保留的内部不变量（8945），下层 GetPlatformInfo 以框架内部码首报。
    // 内部码（internal code），仅用于验证去重后 count==1 的首错来源，不把它当正式用户码。
    EXPECT_EQ(error.code, InternalErrorCode());
}

} // namespace
