/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <gtest/gtest.h>
#include "base/err_msg.h"
#include "base/context_builder/op_infer_datatype_context_builder.h"
#include "experiment/metadef/common/util/error_manager/error_manager.h"
#include "../../../op_graph/inplace_apply_ftrl_proto.h"

namespace ops {
ge::graphStatus InferDataTypeForInplaceApplyFtrl(gert::InferDataTypeContext* context);
}

namespace {
class InplaceApplyFtrlInferDtype : public testing::Test {
protected:
    static void SetUpTestSuite()
    {
        // Graph UT does not load the nnopbase catalog that defines EZ1014.
        constexpr char errorCodes[] = R"json({"error_info_list":[
            {"errClass":"Nnopbase Errors","errTitle":"Execution_Error",
             "ErrCode":"EZ1014",
             "ErrMessage":"Failed to execute operator %s. Reason: %s.",
             "Arglist":"opName, reason",
             "suggestion":{"Possible Cause":"N/A","Solution":"N/A"}}
        ]})json";
        ASSERT_EQ(error_message::RegisterFormatErrorMessage(errorCodes, sizeof(errorCodes) - 1), 0);
    }
};

auto BuildDtypeContext(ge::DataType dtype, size_t outputs = 3, int changedInput = -1)
{
    gert::OpInferDataTypeContextBuilder builder;
    builder.OpName("InplaceApplyFtrl")
        .OpType("InplaceApplyFtrl")
        .IOInstanceNum(std::vector<uint32_t>(8, 1), std::vector<uint32_t>(outputs, 1));
    for (size_t i = 0; i < 8; ++i) {
        builder.InputTensorDesc(i, static_cast<int>(i) == changedInput ? ge::DT_FLOAT16 : dtype, ge::FORMAT_ND,
                                ge::FORMAT_ND);
    }
    for (size_t i = 0; i < outputs; ++i) {
        builder.OutputTensorDesc(i, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    return builder.Build();
}

void ExpectDtypeError(gert::InferDataTypeContext* context, const char* code)
{
    auto& manager = ErrorManager::GetInstance();
    ASSERT_EQ(manager.Init(), 0);
    manager.GetRawErrorMessages();
    EXPECT_EQ(ops::InferDataTypeForInplaceApplyFtrl(context), ge::GRAPH_FAILED);
    const auto errors = manager.GetRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, code);
}
} // namespace

TEST_F(InplaceApplyFtrlInferDtype, MissingOutputSlotsFailOnce)
{
    for (size_t outputs : {1U, 2U}) {
        SCOPED_TRACE(outputs);
        auto holder = BuildDtypeContext(ge::DT_FLOAT, outputs);
        ASSERT_NE(holder.GetContext(), nullptr);
        ExpectDtypeError(holder.GetContext(), "EZ1014");
    }
}

TEST_F(InplaceApplyFtrlInferDtype, RejectsNullContext) { ExpectDtypeError(nullptr, "EZ1014"); }

TEST_F(InplaceApplyFtrlInferDtype, RejectsUnsupportedAndMixedDtypesOnce)
{
    for (auto dtype : {ge::DT_INT32, ge::DT_DOUBLE, ge::DT_UNDEFINED}) {
        auto holder = BuildDtypeContext(dtype);
        ExpectDtypeError(holder.GetContext(), "EZ0019");
    }
    for (int input = 0; input < 8; ++input) {
        SCOPED_TRACE(input);
        auto holder = BuildDtypeContext(ge::DT_FLOAT, 3, input);
        ExpectDtypeError(holder.GetContext(), "EZ0019");
    }
}

TEST_F(InplaceApplyFtrlInferDtype, SetsAllThreeOutputs)
{
    for (auto dtype : {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}) {
        auto holder = BuildDtypeContext(dtype);
        auto* context = holder.GetContext();
        ASSERT_NE(context, nullptr);
        ASSERT_EQ(ops::InferDataTypeForInplaceApplyFtrl(context), ge::GRAPH_SUCCESS);
        for (size_t i = 0; i < 3; ++i) {
            EXPECT_EQ(context->GetOutputDataType(i), dtype);
        }
    }
}
