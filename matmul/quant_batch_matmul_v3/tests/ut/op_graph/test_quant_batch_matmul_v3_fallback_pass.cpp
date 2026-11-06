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
 * \file test_quant_batch_matmul_v3_fallback_pass.cpp
 * \brief Fallback dispatch and argument forwarding tests.
 */

#include <array>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "base/context_builder/op_kernel_run_context_builder.h"
#include "op_fallback.h"

namespace {
struct FallbackCall {
    std::string api;
    gert::OpExecuteContext* context = nullptr;
    std::vector<const gert::Tensor*> tensors;
    bool transA = false;
    bool transB = false;
    int64_t groupSize = -1;
    int count = 0;
    ge::graphStatus status = ge::GRAPH_SUCCESS;
} call;

ge::graphStatus CaptureCall(const char* api, gert::OpExecuteContext* context, const gert::Tensor* x1,
                            const gert::Tensor* x2, const gert::Tensor* scale, const gert::Tensor* offset,
                            const gert::Tensor* pertoken, const gert::Tensor* bias, bool transA, bool transB,
                            const gert::Tensor* output)
{
    call.api = api;
    call.context = context;
    call.tensors = {x1, x2, scale, offset, pertoken, bias, output};
    call.transA = transA;
    call.transB = transB;
    ++call.count;
    return call.status;
}

ge::graphStatus CaptureCall(const char* api, gert::OpExecuteContext* context, const gert::Tensor* x1,
                            const gert::Tensor* x2, const gert::Tensor* pertoken, const gert::Tensor* scale,
                            const gert::Tensor* yScale, const gert::Tensor* x1Offset, const gert::Tensor* x2Offset,
                            const gert::Tensor* yOffset, const gert::Tensor* bias, bool transA, bool transB,
                            int64_t groupSize, const gert::Tensor* output)
{
    call.api = api;
    call.context = context;
    call.tensors = {x1, x2, pertoken, scale, yScale, x1Offset, x2Offset, yOffset, bias, output};
    call.transA = transA;
    call.transB = transB;
    call.groupSize = groupSize;
    ++call.count;
    return call.status;
}

struct IgnoreFallbackRegistration {
    template <typename Function>
    bool OpExecuteFunc(Function) const
    {
        return true;
    }
};
} // namespace

// Exercise the real adapter while replacing only the ACLNN call boundary. Suppress the included
// translation unit's registration so its stubbed callback cannot replace the production registry entry.
#pragma push_macro("CANN_OPS_OPB_SYN_EXEC_ACLNN")
#pragma push_macro("IMPL_OP")
#undef CANN_OPS_OPB_SYN_EXEC_ACLNN
#undef IMPL_OP
#define CANN_OPS_OPB_SYN_EXEC_ACLNN(ctx, api, ...) CaptureCall(#api, ctx, __VA_ARGS__)
#define IMPL_OP(opType) \
    static const bool kQbmmFallbackTestRegistration = IgnoreFallbackRegistration {}
#include "../../../op_graph/quant_batch_matmul_v3_fallback.cpp"
#pragma pop_macro("IMPL_OP")
#pragma pop_macro("CANN_OPS_OPB_SYN_EXEC_ACLNN")

namespace {
gert::Tensor MakeFallbackTensor(std::initializer_list<int64_t> dims, ge::DataType dtype)
{
    gert::StorageFormat format;
    format.SetOriginFormat(ge::FORMAT_ND);
    format.SetStorageFormat(ge::FORMAT_ND);
    return gert::Tensor(gert::StorageShape(dims, dims), format, gert::kOnDeviceHbm, dtype, nullptr);
}

class QuantBatchMatmulV3Fallback : public testing::Test {
protected:
    void SetUp() override { call = FallbackCall{}; }

    ge::graphStatus Run(std::array<gert::Tensor*, 6> inputs, gert::Tensor* output, bool transA = false,
                        bool transB = false, int64_t groupSize = 0, size_t attrCount = 4, bool missingAttrs = false)
    {
        gert::OpKernelContextBuilder builder;
        builder.OpType("QuantBatchMatmulV3").OpName("qbmm_fallback_ut");
        std::vector<uint32_t> instances{1, 1, 1};
        std::vector<void*> values;
        for (size_t i = 3; i < inputs.size(); ++i) {
            instances.push_back(inputs[i] == nullptr ? 0U : 1U);
        }
        builder.IOInstanceNum(instances, {1});
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (instances[i] == 0) {
                continue;
            }
            builder.InputTensorDesc(values.size(), inputs[i] == nullptr ? ge::DT_INT8 : inputs[i]->GetDataType(),
                                    ge::FORMAT_ND, ge::FORMAT_ND);
            values.push_back(inputs[i]);
        }
        builder.OutputTensorDesc(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND);
        values.push_back(output);
        values.resize(values.size() + static_cast<size_t>(gert::OpExecuteInputExtendIndex::kNum), nullptr);
        builder.Inputs(values).Outputs({nullptr});
        if (attrCount > 0) {
            builder.AppendAttr(static_cast<int64_t>(ge::DT_BF16));
        }
        if (attrCount > 1) {
            builder.AppendAttr(transA);
        }
        if (attrCount > 2) {
            builder.AppendAttr(transB);
        }
        if (attrCount > 3) {
            builder.AppendAttr(groupSize);
        }
        auto holder = builder.Build();
        auto* context = reinterpret_cast<gert::OpExecuteContext*>(holder.GetContext());
        if (missingAttrs) {
            holder.GetContext()->GetContext()->compute_node_info = nullptr;
        }
        const auto status = fallback::QuantBatchMatmulV3ExecuteFunc(context);
        if (call.count != 0) {
            EXPECT_EQ(call.context, context);
        }
        return status;
    }

    gert::Tensor x1 = MakeFallbackTensor({32, 64}, ge::DT_INT8);
    gert::Tensor x2 = MakeFallbackTensor({64, 48}, ge::DT_INT8);
    gert::Tensor scale = MakeFallbackTensor({48}, ge::DT_FLOAT);
    gert::Tensor offset = MakeFallbackTensor({48}, ge::DT_FLOAT);
    gert::Tensor bias = MakeFallbackTensor({48}, ge::DT_FLOAT);
    gert::Tensor pertoken = MakeFallbackTensor({32}, ge::DT_FLOAT);
    gert::Tensor output = MakeFallbackTensor({32, 48}, ge::DT_BF16);
};

TEST_F(QuantBatchMatmulV3Fallback, RejectsNullContextAndRequiredTensorsBeforeDispatch)
{
    EXPECT_EQ(fallback::QuantBatchMatmulV3ExecuteFunc(nullptr), ge::GRAPH_FAILED);
    for (size_t missing = 0; missing < 3; ++missing) {
        SCOPED_TRACE(missing);
        std::array<gert::Tensor*, 6> inputs{&x1, &x2, &scale, nullptr, nullptr, nullptr};
        inputs[missing] = nullptr;
        EXPECT_EQ(Run(inputs, &output), ge::GRAPH_FAILED);
    }
    EXPECT_EQ(Run({&x1, &x2, &scale, nullptr, nullptr, nullptr}, nullptr), ge::GRAPH_FAILED);
    EXPECT_EQ(Run({&x1, &x2, &scale, nullptr, nullptr, nullptr}, &output, false, false, 0, 0, true), ge::GRAPH_FAILED);
    EXPECT_EQ(call.count, 0);
}

TEST_F(QuantBatchMatmulV3Fallback, V4PreservesOptionalInputPositionsAndTransposeFlags)
{
    for (unsigned mask = 0; mask < 8; ++mask) {
        SCOPED_TRACE(mask);
        auto* optionalOffset = (mask & 1) ? &offset : nullptr;
        auto* optionalBias = (mask & 2) ? &bias : nullptr;
        auto* optionalPertoken = (mask & 4) ? &pertoken : nullptr;
        for (bool transA : {false, true}) {
            for (bool transB : {false, true}) {
                call = FallbackCall{};
                ASSERT_EQ(
                    Run({&x1, &x2, &scale, optionalOffset, optionalBias, optionalPertoken}, &output, transA, transB),
                    ge::GRAPH_SUCCESS);
                EXPECT_EQ(call.api, "aclnnQuantMatmulV4");
                EXPECT_EQ(call.count, 1);
                EXPECT_EQ(call.transA, transA);
                EXPECT_EQ(call.transB, transB);
                EXPECT_EQ(call.tensors, (std::vector<const gert::Tensor*>{&x1, &x2, &scale, optionalOffset,
                                                                          optionalPertoken, optionalBias, &output}));
            }
        }
    }
}

TEST_F(QuantBatchMatmulV3Fallback, V5ForLowBitInputsPreservesGroupSizeAndNullSlots)
{
    for (auto dtype : {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_HIFLOAT8, ge::DT_FLOAT4_E2M1}) {
        SCOPED_TRACE(static_cast<int>(dtype));
        x1.SetDataType(dtype);
        x2.SetDataType(dtype);
        call = FallbackCall{};
        constexpr int64_t groupSize = (1LL << 32) | (1LL << 16) | 32;
        ASSERT_EQ(Run({&x1, &x2, &scale, &offset, &bias, &pertoken}, &output, true, true, groupSize),
                  ge::GRAPH_SUCCESS);
        EXPECT_EQ(call.api, "aclnnQuantMatmulV5");
        EXPECT_EQ(call.count, 1);
        EXPECT_TRUE(call.transA);
        EXPECT_TRUE(call.transB);
        EXPECT_EQ(call.groupSize, groupSize);
        EXPECT_EQ(call.tensors, (std::vector<const gert::Tensor*>{&x1, &x2, &pertoken, &scale, nullptr, nullptr,
                                                                  &offset, nullptr, &bias, &output}));
    }
}

TEST_F(QuantBatchMatmulV3Fallback, GroupedInt8RequiresBothScaleRanksAndFloatBias)
{
    scale = MakeFallbackTensor({1, 48}, ge::DT_FLOAT);
    pertoken = MakeFallbackTensor({32, 1}, ge::DT_FLOAT);
    for (auto* optionalBias : {static_cast<gert::Tensor*>(nullptr), &bias}) {
        call = FallbackCall{};
        ASSERT_EQ(Run({&x1, &x2, &scale, nullptr, optionalBias, &pertoken}, &output), ge::GRAPH_SUCCESS);
        EXPECT_EQ(call.api, "aclnnQuantMatmulV5");
    }
    bias.SetDataType(ge::DT_BF16);
    ASSERT_EQ(Run({&x1, &x2, &scale, nullptr, &bias, &pertoken}, &output), ge::GRAPH_SUCCESS);
    EXPECT_EQ(call.api, "aclnnQuantMatmulV4");
    scale = MakeFallbackTensor({48}, ge::DT_FLOAT);
    ASSERT_EQ(Run({&x1, &x2, &scale, nullptr, nullptr, &pertoken}, &output), ge::GRAPH_SUCCESS);
    EXPECT_EQ(call.api, "aclnnQuantMatmulV4");
    scale = MakeFallbackTensor({1, 48}, ge::DT_FLOAT);
    pertoken = MakeFallbackTensor({32}, ge::DT_FLOAT);
    ASSERT_EQ(Run({&x1, &x2, &scale, nullptr, nullptr, &pertoken}, &output), ge::GRAPH_SUCCESS);
    EXPECT_EQ(call.api, "aclnnQuantMatmulV4");
}

TEST_F(QuantBatchMatmulV3Fallback, MissingTrailingAttributesUseDefaults)
{
    for (bool lowBit : {false, true}) {
        x1.SetDataType(lowBit ? ge::DT_FLOAT8_E4M3FN : ge::DT_INT8);
        call = FallbackCall{};
        ASSERT_EQ(Run({&x1, &x2, &scale, nullptr, nullptr, nullptr}, &output, true, true, 123, 1), ge::GRAPH_SUCCESS);
        EXPECT_FALSE(call.transA);
        EXPECT_FALSE(call.transB);
        EXPECT_EQ(call.api, lowBit ? "aclnnQuantMatmulV5" : "aclnnQuantMatmulV4");
        if (lowBit) {
            EXPECT_EQ(call.groupSize, 0);
        }
    }
}

TEST_F(QuantBatchMatmulV3Fallback, PropagatesV4AndV5ExecutionFailures)
{
    for (auto dtype : {ge::DT_INT8, ge::DT_FLOAT8_E4M3FN}) {
        x1.SetDataType(dtype);
        call = FallbackCall{};
        call.status = ge::GRAPH_PARAM_INVALID;
        EXPECT_EQ(Run({&x1, &x2, &scale, nullptr, nullptr, nullptr}, &output), ge::GRAPH_FAILED);
        EXPECT_EQ(call.count, 1);
    }
}
} // namespace
