/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "log/log.h"
#include "base/err_mgr.h"
#include "../../../op_graph/grouped_dynamic_block_quant_proto.h"

using namespace ge;
using namespace op;

class GroupedDynamicBlockQuantProto : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GroupedDynamicBlockQuantProto SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "GroupedDynamicBlockQuantProto TearDown" << std::endl; }
};

namespace {
// ErrorManager 上下文隔离方式与 rnn/gru_grad/tests/ut/op_host/test_gru_grad_tiling.cpp 一致：
// 用独立 work_stream_id 记录，避免共享 Host UT 进程内的用例串扰；RAII 析构恢复原上下文。
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

// GetErrMgrRawErrorMessages 读取即消费，必须先快照再断言
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
} // namespace

TEST_F(GroupedDynamicBlockQuantProto, inferDtype_case_1)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("GroupedDynamicBlockQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GroupedDynamicBlockQuant")->infer_datatype;
    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_FLOAT16;
        ge::DataType input_group_list_ref = ge::DT_INT32;
        ge::DataType output_y_ref = ge::DT_FLOAT8_E4M3FN;
        ge::DataType output_scale_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(2)
                                  .NodeIoNum(2, 2)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({
                                      {"min_scale", Ops::NN::AnyValue::CreateFrom<int64_t>(0.0)},
                                      {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                      {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(36)},
                                      {"row_block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(128)},
                                      {"col_block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(128)},
                                      {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  })
                                  .InputDataTypes({&input_x_ref, &input_group_list_ref})
                                  .OutputDataTypes({&output_y_ref, &output_scale_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_y_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_scale_ref);
    }
}

TEST_F(GroupedDynamicBlockQuantProto, inferDtype_case_2)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("GroupedDynamicBlockQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GroupedDynamicBlockQuant")->infer_datatype;
    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_FLOAT16;
        ge::DataType input_group_list_ref = ge::DT_INT32;
        ge::DataType output_y_ref = ge::DT_FLOAT8_E5M2;
        ge::DataType output_scale_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(2)
                                  .NodeIoNum(2, 2)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({
                                      {"min_scale", Ops::NN::AnyValue::CreateFrom<int64_t>(0.0)},
                                      {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                                      {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(35)},
                                      {"row_block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(128)},
                                      {"col_block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(128)},
                                      {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  })
                                  .InputDataTypes({&input_x_ref, &input_group_list_ref})
                                  .OutputDataTypes({&output_y_ref, &output_scale_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_y_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_scale_ref);
    }
}

TEST_F(GroupedDynamicBlockQuantProto, inferDtype_case_3)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("GroupedDynamicBlockQuant"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GroupedDynamicBlockQuant")->infer_datatype;
    if (data_type_func != nullptr) {
        ge::DataType input_x_ref = ge::DT_FLOAT16;
        ge::DataType input_group_list_ref = ge::DT_INT32;
        ge::DataType output_y_ref = ge::DT_HIFLOAT8;
        ge::DataType output_scale_ref = ge::DT_FLOAT;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(2)
                                  .NodeIoNum(2, 2)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_HIFLOAT8, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeAttrs({
                                      {"min_scale", Ops::NN::AnyValue::CreateFrom<int64_t>(0.0)},
                                      {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("round")},
                                      {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(34)},
                                      {"row_block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(128)},
                                      {"col_block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(128)},
                                      {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  })
                                  .InputDataTypes({&input_x_ref, &input_group_list_ref})
                                  .OutputDataTypes({&output_y_ref, &output_scale_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_y_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_scale_ref);
    }
}

TEST_F(GroupedDynamicBlockQuantProto, infershape_case_1)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GroupedDynamicBlockQuant")->infer_shape;

    int64_t B = 6;
    int64_t H = 2000;
    int64_t W = 2000;
    int64_t groupNum = 10;
    int64_t rowBlockSize = 128;
    int64_t colBlockSize = 128;
    gert::StorageShape xShape = {{B, H, W}, {B, H, W}};
    gert::StorageShape groupListShape = {{groupNum}, {groupNum}};
    gert::StorageShape yShape = {{B, H, W}, {B, H, W}};
    gert::StorageShape scaleShape = {{B, H / rowBlockSize + groupNum, (W + colBlockSize - 1) / colBlockSize},
                                     {B, H / rowBlockSize + groupNum, (W + colBlockSize - 1) / colBlockSize}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT8_E4M3FN, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({
                          {"min_scale", Ops::NN::AnyValue::CreateFrom<int64_t>(0.0)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                          {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(34)},
                          {"row_block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(rowBlockSize)},
                          {"col_block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(colBlockSize)},
                          {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .InputShapes({&xShape, &groupListShape})
                      .OutputShapes({&yShape, &scaleShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* YShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape* ScaleShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(Ops::Base::ToString(*YShape), "[6, 2000, 2000]");
    ASSERT_EQ(Ops::Base::ToString(*ScaleShape), "[6, 25, 16]");
}

TEST_F(GroupedDynamicBlockQuantProto, infershape_rejects_scalar_group_list)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GroupedDynamicBlockQuant")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    constexpr int64_t rowBlockSize = 128;
    constexpr int64_t colBlockSize = 128;
    gert::StorageShape xShape = {{2000, 2000}, {2000, 2000}};
    gert::StorageShape groupListShape = {{}, {}};
    gert::StorageShape yShape = {{2000, 2000}, {2000, 2000}};
    gert::StorageShape scaleShape = {{1, 1}, {1, 1}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT8_E4M3FN, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({
                          {"min_scale", Ops::NN::AnyValue::CreateFrom<int64_t>(0.0)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                          {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(36)},
                          {"row_block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(rowBlockSize)},
                          {"col_block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(colBlockSize)},
                          {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .InputShapes({&xShape, &groupListShape})
                      .OutputShapes({&yShape, &scaleShape})
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(GroupedDynamicBlockQuantProto, infershape_case_2)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("GroupedDynamicBlockQuant")->infer_shape;

    int64_t H = 2000;
    int64_t W = 2000;
    int64_t groupNum = 10;
    int64_t rowBlockSize = 128;
    int64_t colBlockSize = 128;
    gert::StorageShape xShape = {{H, W}, {H, W}};
    gert::StorageShape groupListShape = {{groupNum}, {groupNum}};
    gert::StorageShape yShape = {{H, W}, {H, W}};
    gert::StorageShape scaleShape = {{H / rowBlockSize + groupNum, (W + colBlockSize - 1) / colBlockSize},
                                     {H / rowBlockSize + groupNum, (W + colBlockSize - 1) / colBlockSize}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT8_E4M3FN, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({
                          {"min_scale", Ops::NN::AnyValue::CreateFrom<int64_t>(0.0)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                          {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(34)},
                          {"row_block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(rowBlockSize)},
                          {"col_block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(colBlockSize)},
                          {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .InputShapes({&xShape, &groupListShape})
                      .OutputShapes({&yShape, &scaleShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* YShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape* ScaleShape = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(Ops::Base::ToString(*YShape), "[2000, 2000]");
    ASSERT_EQ(Ops::Base::ToString(*ScaleShape), "[25, 16]");
}

// 负向场景1：必选输入 x 的 shape 槽位缺失（IrInstanceNum 将 x 实例数置 0 且不构造任何输入链），
// GetInputShape(0) 按下标越界规则返回空指针，整改后应经 OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON
// 上报 EZ0037（param_name="x"）首错并返回 ge::GRAPH_FAILED，无 EZ9999 残留。
// 注意：InferShapeContextFaker 的单参数 IrInstanceNum 是空操作，且 NodeIoNum 与双参数
// IrInstanceNum 互斥，负向用例仅用双参数 IrInstanceNum 控制实例数（对齐 mse_loss_grad_v2 口径）。
TEST_F(GroupedDynamicBlockQuantProto, infershape_error_x_shape_missing_ez0037)
{
    ScopedErrorContext errorContext;
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("GroupedDynamicBlockQuant");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape, nullptr);
    auto inferShapeFunc = opImpl->infer_shape;

    // x 实例数置 0 且不提供任何输入链（不调用 NodeInputTd / InputShapes），输入链数为 0
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("GroupedDynamicBlockQuant")
                      .IrInstanceNum({0, 1}, {2})
                      .NodeAttrs({
                          {"min_scale", Ops::NN::AnyValue::CreateFrom<int64_t>(0.0)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                          {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(34)},
                          {"row_block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(128)},
                          {"col_block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(128)},
                          {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(context->GetInputShape(0), nullptr);
    ASSERT_EQ(context->GetInputShape(1), nullptr);

    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_FAILED);
    const auto error = CaptureError();
    EXPECT_EQ(error.count, 1U); // 一次故障只有一个 ErrorManager 首错，无内部码残留
    EXPECT_EQ(error.code, "EZ0037");
    ASSERT_EQ(error.keys.size(), error.values.size());
    ASSERT_EQ(error.keys.size(), 3U); // EZ0037: param_name / op_name / reason
    std::map<std::string, std::string> arguments;
    for (size_t i = 0; i < error.keys.size(); ++i) {
        EXPECT_FALSE(error.keys[i].empty());
        EXPECT_FALSE(error.values[i].empty());
        EXPECT_TRUE(arguments.emplace(error.keys[i], error.values[i]).second);
    }
    EXPECT_EQ(ErrorValue(error, "param_name"), "x");
    EXPECT_FALSE(ErrorValue(error, "op_name").empty());
    EXPECT_FALSE(ErrorValue(error, "reason").empty());
    EXPECT_EQ(CaptureError().count, 0U); // 无重复上报
}

// 负向场景2：必选输入 group_list 的 shape 槽位缺失（IrInstanceNum 将 group_list 实例数置 0，
// 仅保留 x 一条输入链），GetInputShape(0) 有效而 GetInputShape(1) 按下标越界规则返回空指针，
// 整改后应上报 EZ0037（param_name="group_list"）首错并返回 ge::GRAPH_FAILED，无 EZ9999 残留
TEST_F(GroupedDynamicBlockQuantProto, infershape_error_group_list_shape_missing_ez0037)
{
    ScopedErrorContext errorContext;
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("GroupedDynamicBlockQuant");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_shape, nullptr);
    auto inferShapeFunc = opImpl->infer_shape;

    gert::StorageShape xShape = {{2000, 2000}, {2000, 2000}};

    // group_list 实例数置 0，InputShapes 第 1 位 nullptr 占位（faker 不为该位创建输入链），
    // 仅保留 x 一条输入链
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("GroupedDynamicBlockQuant")
                      .IrInstanceNum({1, 0}, {2})
                      .InputShapes({&xShape, nullptr})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({
                          {"min_scale", Ops::NN::AnyValue::CreateFrom<int64_t>(0.0)},
                          {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                          {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(34)},
                          {"row_block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(128)},
                          {"col_block_size", Ops::NN::AnyValue::CreateFrom<int64_t>(128)},
                          {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                      })
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_NE(context->GetInputShape(0), nullptr);
    ASSERT_EQ(context->GetInputShape(1), nullptr);

    EXPECT_EQ(inferShapeFunc(context), ge::GRAPH_FAILED);
    const auto error = CaptureError();
    EXPECT_EQ(error.count, 1U); // 一次故障只有一个 ErrorManager 首错，无内部码残留
    EXPECT_EQ(error.code, "EZ0037");
    ASSERT_EQ(error.keys.size(), error.values.size());
    ASSERT_EQ(error.keys.size(), 3U); // EZ0037: param_name / op_name / reason
    std::map<std::string, std::string> arguments;
    for (size_t i = 0; i < error.keys.size(); ++i) {
        EXPECT_FALSE(error.keys[i].empty());
        EXPECT_FALSE(error.values[i].empty());
        EXPECT_TRUE(arguments.emplace(error.keys[i], error.values[i]).second);
    }
    EXPECT_EQ(ErrorValue(error, "param_name"), "group_list");
    EXPECT_FALSE(ErrorValue(error, "op_name").empty());
    EXPECT_FALSE(ErrorValue(error, "reason").empty());
    EXPECT_EQ(CaptureError().count, 0U); // 无重复上报
}
