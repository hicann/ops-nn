/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <initializer_list>
#include <string>
#include "base/err_msg.h"
#include "experiment/metadef/common/util/error_manager/error_manager.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_range_context.h"
#include "kernel_run_context_facker.h"

namespace error_message {
// Other operator UTs may replace the reporting API with a weak capture implementation.
void ResetCapture() __attribute__((weak));
std::string GetLastError() __attribute__((weak));
int GetReportCount() __attribute__((weak));
} // namespace error_message

class WhereUT : public testing::Test {
protected:
    // EZ0012 由 third_party/opbase 的格式注册对象提供；单算子 UT 构建时该对象可能因静态库
    // 未被引用而未链入，导致宏上报被静默丢弃（同 gru_grad !13929 的场景），此处显式注册兜底。
    static constexpr const char kEz0012ErrorMessage[] = R"json({"error_info_list":[
        {"errClass":"Operator Errors","errTitle":"Invalid_Argument_Tensor_Shape",
         "ErrCode":"EZ0012",
         "ErrMessage":"The dim num of %s in %s is %s. Reason: %s.",
         "Arglist":"param_name, op_name, incorrect_dim, reason",
         "suggestion":{"Possible Cause":"N/A","Solution":"Check whether the dim num of the tensor meets the requirement."}}
    ]})json";

    static void ResetErrorReports()
    {
        (void)error_message::RegisterFormatErrorMessage(kEz0012ErrorMessage, sizeof(kEz0012ErrorMessage) - 1);
        (void)ErrorManager::GetInstance().GetRawErrorMessages();
        if (error_message::ResetCapture != nullptr) {
            error_message::ResetCapture();
        }
    }

    static void ExpectSingleErrorReport(const std::string& expectedCode,
                                        const std::initializer_list<std::string>& messageParts)
    {
        const auto errors = ErrorManager::GetInstance().GetRawErrorMessages();
        if (error_message::GetLastError != nullptr && error_message::GetReportCount != nullptr &&
            error_message::GetReportCount() != 0) {
            const auto reportCount = error_message::GetReportCount();
            const auto reportedCode = error_message::GetLastError();
            if (error_message::ResetCapture != nullptr) {
                error_message::ResetCapture();
            }
            EXPECT_EQ(reportCount, 1);
            EXPECT_EQ(reportedCode, expectedCode);
            if (errors.empty()) {
                // The legacy weak collector exposes the code and count, but no formatted message.
                return;
            }
        }
        ASSERT_EQ(errors.size(), 1U);
        EXPECT_EQ(errors[0].error_id, expectedCode);
        if (expectedCode == "EZ1008" || expectedCode == "EZ1014") {
            EXPECT_NE(errors[0].error_message.find("Preserve the execution logs"), std::string::npos);
            for (const auto* detail : {"compileInfo", "compile info", "compiled info", "platformInfo", "platform info",
                                       "attrs is null", "shape is null", "totalCoreNum"}) {
                EXPECT_EQ(errors[0].error_message.find(detail), std::string::npos);
            }
        }
        for (const auto& part : messageParts) {
            EXPECT_NE(errors[0].error_message.find(part), std::string::npos) << errors[0].error_message;
        }
    }

    static void SetUpTestCase() { std::cout << "Where SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "Where TearDown" << std::endl; }
};

TEST_F(WhereUT, infershape_range_test)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Where"), nullptr);
    auto infer_shape_range_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Where")->infer_shape_range;

    gert::Shape min1{2, 2, 3, 8};
    gert::Shape max1{2, -1, 3, 8};
    gert::Shape outmin{0, 4};
    gert::Shape outmax{-1, 4};
    gert::Shape null1{};
    gert::Shape null2{};

    gert::Range<gert::Shape> in_shape_range(&min1, &max1);
    gert::Range<gert::Shape> out_shape_range(&null1, &null2);
    gert::Range<gert::Shape> exp_out_shape_range(&outmin, &outmax);

    auto context_holder = gert::InferShapeRangeContextFaker()
                              .IrInputNum(1)
                              .NodeIoNum(1, 1)
                              .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputShapeRanges({&in_shape_range})
                              .OutputShapeRanges({&out_shape_range})
                              .Build();

    auto context = context_holder.GetContext<gert::InferShapeRangeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(infer_shape_range_func(context), ge::GRAPH_SUCCESS);

    EXPECT_EQ(context->GetOutputShapeRange(0)->GetMin()->GetDim(0), 0);
    EXPECT_EQ(context->GetOutputShapeRange(0)->GetMin()->GetDim(1), 4);
    EXPECT_EQ(context->GetOutputShapeRange(0)->GetMax()->GetDim(0), -1);
    EXPECT_EQ(context->GetOutputShapeRange(0)->GetMax()->GetDim(1), 4);
}

TEST_F(WhereUT, invalid_rank_reports_one_formal_error)
{
    auto inferShape = gert::OpImplRegistry::GetInstance().GetOpImpl("Where")->infer_shape;
    ASSERT_NE(inferShape, nullptr);
    for (const auto& shape : std::vector<gert::Shape>{{}, {1, 1, 1, 1, 1, 1, 1, 1, 1}}) {
        auto input = shape;
        gert::Shape output;
        auto holder = gert::InferShapeContextFaker()
                          .NodeIoNum(1, 1)
                          .IrInputNum(1)
                          .InputShapes({&input})
                          .OutputShapes({&output})
                          .Build();
        ResetErrorReports();
        EXPECT_EQ(inferShape(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
        ExpectSingleErrorReport("EZ0012", {"x", std::to_string(input.GetDimNum()), "rank"});
    }
}

TEST_F(WhereUT, missing_output_reports_shape_inference_execution_failure)
{
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl("Where");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_shape, nullptr);
    gert::Shape input{2, 3};
    gert::Shape output;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInputNum(1)
                      .InputShapes({&input})
                      .OutputShapes({&output})
                      .Build();
    auto* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context, nullptr);
    auto* kernelContext = reinterpret_cast<gert::KernelContext*>(context);
    kernelContext->GetOutput(0)->Set(nullptr, nullptr);
    ASSERT_EQ(context->GetOutputShape(0), nullptr);
    ResetErrorReports();
    EXPECT_EQ(impl->infer_shape(context), ge::GRAPH_FAILED);
    ExpectSingleErrorReport("EZ1014", {"shape inference execution failed"});
}
