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
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "exe_graph/runtime/compute_node_info.h"
#include "ge/ge_api_types.h"
#include "graph/types.h"

// KernelRunContext and AsyncAnyValue are C structs defined in the global namespace
extern "C" {
#include "exe_graph/runtime/kernel_run_context.h"
}

// Include the real op_fallback.h first so its include guard is set; then replace the
// synchronous aclnn dispatch macro with a recording stub. When the fallback source below
// includes op_fallback.h again the guard prevents redefinition, so the stub stays active
// and the dispatch logic can be verified deterministically.
#include "op_fallback.h"
#undef CANN_OPS_OPB_SYN_EXEC_ACLNN

namespace qbmmv4_ut {
// Records every parameter the fallback forwards to aclnnQuantMatmulV5.
struct ExecAclnnRecord {
    int32_t callCount = 0;
    std::string apiName;
    const gert::Tensor* x1 = nullptr;
    const gert::Tensor* x2 = nullptr;
    const gert::Tensor* x1Scale = nullptr;
    const gert::Tensor* x2Scale = nullptr;
    const gert::Tensor* yScale = nullptr;
    const gert::Tensor* x1Offset = nullptr;
    const gert::Tensor* x2Offset = nullptr;
    const gert::Tensor* yOffset = nullptr;
    const gert::Tensor* bias = nullptr;
    bool transposeX1 = false;
    bool transposeX2 = false;
    int64_t groupSize = 0;
    const gert::Tensor* output = nullptr;
    ge::graphStatus retToReturn = ge::GRAPH_SUCCESS;

    void Reset()
    {
        callCount = 0;
        apiName.clear();
        x1 = nullptr;
        x2 = nullptr;
        x1Scale = nullptr;
        x2Scale = nullptr;
        yScale = nullptr;
        x1Offset = nullptr;
        x2Offset = nullptr;
        yOffset = nullptr;
        bias = nullptr;
        transposeX1 = false;
        transposeX2 = false;
        groupSize = 0;
        output = nullptr;
        retToReturn = ge::GRAPH_SUCCESS;
    }
};

inline ExecAclnnRecord g_execRecord;

inline ge::graphStatus StubExecAclnn(gert::OpExecuteContext* ctx, const char* apiName, const gert::Tensor* x1,
                                     const gert::Tensor* x2, const gert::Tensor* x1Scale, const gert::Tensor* x2Scale,
                                     const gert::Tensor* yScale, const gert::Tensor* x1Offset,
                                     const gert::Tensor* x2Offset, const gert::Tensor* yOffset,
                                     const gert::Tensor* bias, const bool transposeX1, const bool transposeX2,
                                     const int64_t groupSize, const gert::Tensor* output)
{
    (void)ctx;
    auto& record = g_execRecord;
    ++record.callCount;
    record.apiName = apiName;
    record.x1 = x1;
    record.x2 = x2;
    record.x1Scale = x1Scale;
    record.x2Scale = x2Scale;
    record.yScale = yScale;
    record.x1Offset = x1Offset;
    record.x2Offset = x2Offset;
    record.yOffset = yOffset;
    record.bias = bias;
    record.transposeX1 = transposeX1;
    record.transposeX2 = transposeX2;
    record.groupSize = groupSize;
    record.output = output;
    return record.retToReturn;
}
} // namespace qbmmv4_ut

// UT stub for the synchronous aclnn execution: records the arguments and returns a
// configurable status instead of loading the real opapi library.
#define CANN_OPS_OPB_SYN_EXEC_ACLNN(ctx, aclnnApi, ...) qbmmv4_ut::StubExecAclnn((ctx), #aclnnApi, ##__VA_ARGS__)

// Include the fallback source to make the static function accessible
#include "../../../op_graph/quant_batch_matmul_v4_fallback.cpp"

// Build ComputeNodeInfo (attrs included) through the public OpKernelContextBuilder API, so the
// test never hand-rolls gert::RuntimeAttrs' private memory layout and no longer redefines the
// library symbol gert::RuntimeAttrs::GetPointerByIndex (which relied on
// --allow-multiple-definition, an ODR violation).
#include "compute_node_info_faker.h"

using namespace fallback;
using namespace ge;

namespace {

// IR definition of QuantBatchMatmulV4 (proto): 10 inputs (x1, x2, bias, x1_scale, x2_scale,
// y_scale, x1_offset, x2_offset, y_offset, x2_table) and 1 output (y).
constexpr size_t kIrInputs = 10;
constexpr size_t kIrBiasIdx = 2;
constexpr size_t kIrX1ScaleIdx = 3;
constexpr size_t kIrX2ScaleIdx = 4;
constexpr size_t kIrYScaleIdx = 5;
// Attr indices in the proto order: dtype(0), compute_type(1), transpose_x1(2), transpose_x2(3),
// group_size(4).
constexpr size_t kAttrNum = 5;

gert::Tensor MakeTensor(std::initializer_list<int64_t> dims, DataType dtype, Format format = FORMAT_ND)
{
    gert::StorageShape storageShape{dims, dims};
    gert::StorageFormat storageFormat;
    storageFormat.SetOriginFormat(format);
    storageFormat.SetStorageFormat(format);
    void* dummyAddr = reinterpret_cast<void*>(0xDEADBEEF);
    return gert::Tensor(storageShape, storageFormat, gert::kOnDeviceHbm, dtype, dummyAddr);
}

// Which optional inputs are instantiated on the compute node.
struct OptionalInputs {
    bool bias = false;
    bool x1Scale = false;
    bool x2Scale = false;
    bool yScale = false;
};

// Build a ComputeNodeInfo for QuantBatchMatmulV4 (10 inputs, 1 output, 5 attrs) through the
// public builder API. The five proto attrs (dtype, compute_type, transpose_x1, transpose_x2,
// group_size) are appended in proto order; attrCount truncates the tail so the dropped indices
// resolve to null with the real library semantics (missing-attr default branches).
struct ComputeNodeInfoHelper {
    gert::ComputeNodeInfoFaker faker;
    const void* nodeInfo = nullptr;
    size_t actualInputs = 0;

    ComputeNodeInfoHelper(const bool transposeX1, const bool transposeX2, const int64_t groupSize,
                          const OptionalInputs& optionals, const size_t attrCount = kAttrNum,
                          const char* nodeType = "QuantBatchMatmulV4")
    {
        std::vector<uint32_t> instanceNums(kIrInputs, 0);
        instanceNums[0] = 1; // x1
        instanceNums[1] = 1; // x2
        if (optionals.bias) {
            instanceNums[kIrBiasIdx] = 1;
        }
        if (optionals.x1Scale) {
            instanceNums[kIrX1ScaleIdx] = 1;
        }
        if (optionals.x2Scale) {
            instanceNums[kIrX2ScaleIdx] = 1;
        }
        if (optionals.yScale) {
            instanceNums[kIrYScaleIdx] = 1;
        }
        for (const uint32_t num : instanceNums) {
            actualInputs += num;
        }

        faker.NodeType(nodeType).IrInstanceNum(instanceNums, {1});
        for (size_t i = 0; i < kIrInputs; ++i) {
            if (instanceNums[i] != 0) {
                faker.InputTd(i, DT_INT8, FORMAT_ND, FORMAT_ND);
            }
        }
        faker.OutputTd(0, DT_BF16, FORMAT_ND, FORMAT_ND);
        if (attrCount >= 1) {
            faker.AppendAttr(static_cast<int64_t>(1)); // dtype (unused by the fallback)
        }
        if (attrCount >= 2) {
            faker.AppendAttr(static_cast<int64_t>(-1)); // compute_type (unused by the fallback)
        }
        if (attrCount >= 3) {
            faker.AppendAttr(transposeX1); // transpose_x1
        }
        if (attrCount >= 4) {
            faker.AppendAttr(transposeX2); // transpose_x2
        }
        if (attrCount >= 5) {
            faker.AppendAttr(groupSize); // group_size
        }
        nodeInfo = faker.Build();
    }

    void* AsVoidPtr() { return const_cast<void*>(nodeInfo); }
};

// Constructs a mock OpExecuteContext on a KernelRunContext layout:
//   values = [actual inputs..., actual outputs...,
//             extend inputs (kAllocate/kStream/kExecuteOption/kExecuteFunc/kFwkData),
//             extend outputs (kBlockMemory)]
struct MockExecContext {
    std::vector<uint8_t> buffer;
    std::vector<gert::Chain*> chains;
    gert::OpExecuteContext* ctx = nullptr;

    MockExecContext(std::vector<gert::Tensor*> inputs, std::vector<gert::Tensor*> outputs, void* computeNodeInfo)
    {
        const size_t numInputs = inputs.size();
        const size_t numOutputs = outputs.size();
        constexpr size_t kExtendInputs = static_cast<size_t>(gert::OpExecuteInputExtendIndex::kNum);
        constexpr size_t kExtendOutputs = static_cast<size_t>(gert::OpExecuteOutputIndex::kNum);
        const size_t totalValues = numInputs + numOutputs + kExtendInputs + kExtendOutputs;

        const size_t ctxSize = sizeof(KernelRunContext) + (totalValues > 1 ? (totalValues - 1) * sizeof(void*) : 0);
        buffer.resize(ctxSize);
        std::memset(buffer.data(), 0, ctxSize);

        auto krc = reinterpret_cast<KernelRunContext*>(buffer.data());
        krc->input_size = numInputs + numOutputs + kExtendInputs;
        krc->output_size = kExtendOutputs;
        krc->compute_node_info = computeNodeInfo;
        krc->kernel_extend_info = nullptr;
        krc->output_start = nullptr;

        for (size_t i = 0; i < numInputs; ++i) {
            krc->values[i] = MakeChainOrNull(inputs[i]);
        }
        for (size_t i = 0; i < numOutputs; ++i) {
            krc->values[numInputs + i] = MakeChainOrNull(outputs[i]);
        }
        // Extend inputs/outputs stay null: the stubbed dispatch macro never touches them.

        ctx = reinterpret_cast<gert::OpExecuteContext*>(buffer.data());
    }

    ~MockExecContext()
    {
        for (auto* chain : chains) {
            chain->Set(nullptr, nullptr);
            delete chain;
        }
    }

private:
    AsyncAnyValue* MakeChainOrNull(gert::Tensor* tensor)
    {
        if (tensor == nullptr) {
            return nullptr;
        }
        auto* chain = new gert::Chain();
        chains.push_back(chain);
        chain->Set(tensor, nullptr);
        return reinterpret_cast<AsyncAnyValue*>(chain);
    }
};

// Bundles the node info and the context so both share the same instantiation decisions.
struct FallbackEnv {
    ComputeNodeInfoHelper cnih;
    gert::Tensor x1;
    gert::Tensor x2;
    gert::Tensor bias;
    gert::Tensor x1Scale;
    gert::Tensor x2Scale;
    gert::Tensor yScale;
    gert::Tensor output;
    std::unique_ptr<MockExecContext> ctx;

    FallbackEnv(const bool transposeX1, const bool transposeX2, const int64_t groupSize,
                const OptionalInputs& optionals, const size_t attrCount = kAttrNum)
        : cnih(transposeX1, transposeX2, groupSize, optionals, attrCount),
          x1(MakeTensor({16, 128}, DT_INT8)),
          x2(MakeTensor({256, 128}, DT_INT4)),
          bias(MakeTensor({1, 256}, DT_FLOAT16)),
          x1Scale(MakeTensor({16, 8}, DT_FLOAT16)),
          x2Scale(MakeTensor({256, 8}, DT_BF16)),
          yScale(MakeTensor({1, 256}, DT_UINT64)),
          output(MakeTensor({16, 256}, DT_BF16))
    {
        // Slot order must match ComputeNodeInfoHelper: x1, x2, then instantiated optionals in
        // IR order (bias, x1_scale, x2_scale, y_scale), then the output.
        std::vector<gert::Tensor*> inputs = {&x1, &x2};
        if (optionals.bias) {
            inputs.push_back(&bias);
        }
        if (optionals.x1Scale) {
            inputs.push_back(&x1Scale);
        }
        if (optionals.x2Scale) {
            inputs.push_back(&x2Scale);
        }
        if (optionals.yScale) {
            inputs.push_back(&yScale);
        }
        ctx = std::make_unique<MockExecContext>(inputs, std::vector<gert::Tensor*>{&output}, cnih.AsVoidPtr());
    }
};

} // namespace

class QuantBatchMatmulV4FallbackTest : public ::testing::Test {
protected:
    void SetUp() override { qbmmv4_ut::g_execRecord.Reset(); }
    void TearDown() override {}
};

// Error-handling branches
TEST_F(QuantBatchMatmulV4FallbackTest, NullContext)
{
    EXPECT_EQ(QuantBatchMatmulV4ExecuteFunc(nullptr), GRAPH_FAILED);
    EXPECT_EQ(qbmmv4_ut::g_execRecord.callCount, 0);
}

TEST_F(QuantBatchMatmulV4FallbackTest, X1IsNull)
{
    FallbackEnv env(false, true, 64, OptionalInputs{});
    // Drop x1 from the context values.
    auto& krc = *reinterpret_cast<KernelRunContext*>(env.ctx->buffer.data());
    krc.values[0] = nullptr;
    EXPECT_EQ(QuantBatchMatmulV4ExecuteFunc(env.ctx->ctx), GRAPH_FAILED);
    EXPECT_EQ(qbmmv4_ut::g_execRecord.callCount, 0);
}

TEST_F(QuantBatchMatmulV4FallbackTest, X2IsNull)
{
    FallbackEnv env(false, true, 64, OptionalInputs{});
    auto& krc = *reinterpret_cast<KernelRunContext*>(env.ctx->buffer.data());
    krc.values[1] = nullptr;
    EXPECT_EQ(QuantBatchMatmulV4ExecuteFunc(env.ctx->ctx), GRAPH_FAILED);
    EXPECT_EQ(qbmmv4_ut::g_execRecord.callCount, 0);
}

TEST_F(QuantBatchMatmulV4FallbackTest, OutputIsNull)
{
    FallbackEnv env(false, true, 64, OptionalInputs{});
    // Output sits right after the actual inputs.
    auto& krc = *reinterpret_cast<KernelRunContext*>(env.ctx->buffer.data());
    krc.values[env.cnih.actualInputs] = nullptr;
    EXPECT_EQ(QuantBatchMatmulV4ExecuteFunc(env.ctx->ctx), GRAPH_FAILED);
    EXPECT_EQ(qbmmv4_ut::g_execRecord.callCount, 0);
}

TEST_F(QuantBatchMatmulV4FallbackTest, AttrsIsNull)
{
    // No ComputeNodeInfo: GetAttrs() returns null and the func fails before dispatch.
    gert::Tensor x1 = MakeTensor({16, 128}, DT_INT8);
    gert::Tensor x2 = MakeTensor({256, 128}, DT_INT4);
    gert::Tensor output = MakeTensor({16, 256}, DT_BF16);
    MockExecContext mc({&x1, &x2}, {&output}, nullptr);
    EXPECT_EQ(QuantBatchMatmulV4ExecuteFunc(mc.ctx), GRAPH_FAILED);
    EXPECT_EQ(qbmmv4_ut::g_execRecord.callCount, 0);
}

TEST_F(QuantBatchMatmulV4FallbackTest, AclnnExecuteFailed)
{
    FallbackEnv env(false, true, 64, OptionalInputs{});
    qbmmv4_ut::g_execRecord.retToReturn = GRAPH_FAILED;
    EXPECT_EQ(QuantBatchMatmulV4ExecuteFunc(env.ctx->ctx), GRAPH_FAILED);
    EXPECT_EQ(qbmmv4_ut::g_execRecord.callCount, 1);
}

// Dispatch: only the required inputs are present, all optional tensors resolve to null.
TEST_F(QuantBatchMatmulV4FallbackTest, ExecWithoutOptionalInputs)
{
    FallbackEnv env(false, true, 64, OptionalInputs{});
    EXPECT_EQ(QuantBatchMatmulV4ExecuteFunc(env.ctx->ctx), GRAPH_SUCCESS);
    const auto& record = qbmmv4_ut::g_execRecord;
    EXPECT_EQ(record.callCount, 1);
    EXPECT_EQ(record.apiName, "aclnnQuantMatmulV5");
    EXPECT_EQ(record.x1, &env.x1);
    EXPECT_EQ(record.x2, &env.x2);
    EXPECT_EQ(record.output, &env.output);
    EXPECT_EQ(record.bias, nullptr);
    EXPECT_EQ(record.x1Scale, nullptr);
    EXPECT_EQ(record.x2Scale, nullptr);
    EXPECT_EQ(record.yScale, nullptr);
    EXPECT_EQ(record.x1Offset, nullptr);
    EXPECT_EQ(record.x2Offset, nullptr);
    EXPECT_EQ(record.yOffset, nullptr);
    EXPECT_FALSE(record.transposeX1);
    EXPECT_TRUE(record.transposeX2);
    EXPECT_EQ(record.groupSize, 64);
}

// Dispatch: every optional input is instantiated and forwarded.
TEST_F(QuantBatchMatmulV4FallbackTest, ExecWithAllOptionalInputs)
{
    OptionalInputs optionals;
    optionals.bias = true;
    optionals.x1Scale = true;
    optionals.x2Scale = true;
    optionals.yScale = true;
    FallbackEnv env(true, true, 128, optionals);
    EXPECT_EQ(QuantBatchMatmulV4ExecuteFunc(env.ctx->ctx), GRAPH_SUCCESS);
    const auto& record = qbmmv4_ut::g_execRecord;
    EXPECT_EQ(record.callCount, 1);
    EXPECT_EQ(record.x1, &env.x1);
    EXPECT_EQ(record.x2, &env.x2);
    EXPECT_EQ(record.bias, &env.bias);
    EXPECT_EQ(record.x1Scale, &env.x1Scale);
    EXPECT_EQ(record.x2Scale, &env.x2Scale);
    EXPECT_EQ(record.yScale, &env.yScale);
    EXPECT_EQ(record.output, &env.output);
    EXPECT_TRUE(record.transposeX1);
    EXPECT_TRUE(record.transposeX2);
    EXPECT_EQ(record.groupSize, 128);
}

// Dispatch: a subset of the optional inputs is instantiated, the rest stay null.
TEST_F(QuantBatchMatmulV4FallbackTest, ExecWithPartialOptionalInputs)
{
    OptionalInputs optionals;
    optionals.bias = true;
    optionals.x2Scale = true;
    FallbackEnv env(false, true, 32, optionals);
    EXPECT_EQ(QuantBatchMatmulV4ExecuteFunc(env.ctx->ctx), GRAPH_SUCCESS);
    const auto& record = qbmmv4_ut::g_execRecord;
    EXPECT_EQ(record.callCount, 1);
    EXPECT_EQ(record.bias, &env.bias);
    EXPECT_EQ(record.x2Scale, &env.x2Scale);
    EXPECT_EQ(record.x1Scale, nullptr);
    EXPECT_EQ(record.yScale, nullptr);
}

// Attr defaults: transpose_x2/group_size missing (attrs truncated after transpose_x1) ->
// transpose_x1 is forwarded (true), transpose_x2 falls back to false, group_size to 32.
TEST_F(QuantBatchMatmulV4FallbackTest, TransposeX2AndGroupSizeAttrNullUseDefaults)
{
    FallbackEnv env(true, true, 64, OptionalInputs{}, 3);
    EXPECT_EQ(QuantBatchMatmulV4ExecuteFunc(env.ctx->ctx), GRAPH_SUCCESS);
    const auto& record = qbmmv4_ut::g_execRecord;
    EXPECT_EQ(record.callCount, 1);
    EXPECT_TRUE(record.transposeX1);
    EXPECT_FALSE(record.transposeX2);
    EXPECT_EQ(record.groupSize, 32);
}

// Attr default: group_size pointer null -> default 32 is forwarded.
TEST_F(QuantBatchMatmulV4FallbackTest, GroupSizeAttrNullUsesDefault)
{
    FallbackEnv env(false, true, 128, OptionalInputs{}, 4);
    EXPECT_EQ(QuantBatchMatmulV4ExecuteFunc(env.ctx->ctx), GRAPH_SUCCESS);
    const auto& record = qbmmv4_ut::g_execRecord;
    EXPECT_EQ(record.callCount, 1);
    EXPECT_EQ(record.groupSize, 32);
}

// Attr defaults: all quant-related attrs missing -> false/false/32.
TEST_F(QuantBatchMatmulV4FallbackTest, AllQuantAttrsNullUseDefaults)
{
    FallbackEnv env(true, true, 128, OptionalInputs{}, 2);
    EXPECT_EQ(QuantBatchMatmulV4ExecuteFunc(env.ctx->ctx), GRAPH_SUCCESS);
    const auto& record = qbmmv4_ut::g_execRecord;
    EXPECT_EQ(record.callCount, 1);
    EXPECT_FALSE(record.transposeX1);
    EXPECT_FALSE(record.transposeX2);
    EXPECT_EQ(record.groupSize, 32);
}

// Attr values: explicit transpose/group_size combinations are forwarded unchanged.
TEST_F(QuantBatchMatmulV4FallbackTest, TransposeAndGroupSizeValuesForwarded)
{
    FallbackEnv env(false, true, 64, OptionalInputs{});
    EXPECT_EQ(QuantBatchMatmulV4ExecuteFunc(env.ctx->ctx), GRAPH_SUCCESS);
    EXPECT_EQ(qbmmv4_ut::g_execRecord.transposeX1, false);
    EXPECT_EQ(qbmmv4_ut::g_execRecord.transposeX2, true);
    EXPECT_EQ(qbmmv4_ut::g_execRecord.groupSize, 64);

    qbmmv4_ut::g_execRecord.Reset();
    FallbackEnv env2(true, false, 16, OptionalInputs{});
    EXPECT_EQ(QuantBatchMatmulV4ExecuteFunc(env2.ctx->ctx), GRAPH_SUCCESS);
    EXPECT_EQ(qbmmv4_ut::g_execRecord.transposeX1, true);
    EXPECT_EQ(qbmmv4_ut::g_execRecord.transposeX2, false);
    EXPECT_EQ(qbmmv4_ut::g_execRecord.groupSize, 16);
}
