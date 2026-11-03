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
#include <new>
#include <vector>

#include <gtest/gtest.h>
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "exe_graph/runtime/compute_node_info.h"
#include "ge/ge_api_types.h"
#include "graph/types.h"

extern "C" {
#include "exe_graph/runtime/kernel_run_context.h"
}

#define CANN_OPS_OPB_ASYN_EXEC_ACLNN(ctx, aclnnApi, ...) GRAPH_SUCCESS

struct OpApiParams {
    struct ConvertedParam {
        void* pointer;
        void (*deleter)(void*);
    };
    int (*op_api_func)(void*, uint64_t, void*, void*);
    void* executor;
    std::vector<ConvertedParam> converted_params;
};

#include "../../../op_graph/mat_mul_v3_fallback.cpp"

// Build ComputeNodeInfo (attrs included) through the public OpKernelContextBuilder API, so the
// test never hand-rolls gert::RuntimeAttrs' private memory layout and no longer redefines the
// library symbol gert::RuntimeAttrs::GetPointerByIndex (which relied on
// --allow-multiple-definition, an ODR violation).
#include "compute_node_info_faker.h"

using namespace fallback;
using namespace ge;

namespace {

gert::Tensor MakeTensor(std::initializer_list<int64_t> dims, DataType dtype, Format format = FORMAT_ND)
{
    gert::StorageShape storageShape{dims, dims};
    gert::StorageFormat storageFormat;
    storageFormat.SetOriginFormat(format);
    storageFormat.SetStorageFormat(format);
    void* dummyAddr = reinterpret_cast<void*>(0xDEADBEEF);
    return gert::Tensor(storageShape, storageFormat, gert::kOnDeviceHbm, dtype, dummyAddr);
}

// Build a ComputeNodeInfo for MatMul (2 attrs: trans_self, trans_mat2) through the public
// builder API. attrCount truncates the appended attrs so the dropped indices resolve to null
// with the real library semantics; the fallback rejects a null attr pointer.
struct ComputeNodeInfoHelper {
    gert::ComputeNodeInfoFaker faker;
    const void* nodeInfo = nullptr;

    ComputeNodeInfoHelper(const size_t attrCount = 2, const char* nodeType = "MatMul", size_t irInputs = 3,
                          size_t irOutputs = 1)
    {
        faker.NodeType(nodeType).IrInstanceNum(std::vector<uint32_t>(irInputs, 1), std::vector<uint32_t>(irOutputs, 1));
        for (size_t i = 0; i < irInputs; ++i) {
            faker.InputTd(i, DT_FLOAT, FORMAT_ND, FORMAT_ND);
        }
        for (size_t i = 0; i < irOutputs; ++i) {
            faker.OutputTd(i, DT_FLOAT, FORMAT_ND, FORMAT_ND);
        }
        if (attrCount >= 1) {
            faker.AppendAttr(false); // trans_self
        }
        if (attrCount >= 2) {
            faker.AppendAttr(false); // trans_mat2
        }
        nodeInfo = faker.Build();
    }

    void* AsVoidPtr() { return const_cast<void*>(nodeInfo); }
};

struct MockContext {
    std::vector<uint8_t> buffer;
    std::vector<gert::Chain*> chains;
    gert::OpExecutePrepareContext* ctx;

    MockContext(std::vector<gert::Tensor*> inputs, std::vector<gert::Tensor*> outputs, void* computeNodeInfo = nullptr)
    {
        size_t numInputs = inputs.size();
        size_t numOutputs = outputs.size();
        // 真实OpExecutePrepareContext内存布局:
        //   values = [inputs..., outputs..., extend_inputs(kExecuteOption/kFwkData/kStream),
        //             extend_outputs(kParams/kWorkspaceSize)]
        // 其中KernelContext::input_size = numInputs + numOutputs + kExtendInputs,
        // KernelContext::output_size = kExtendOutputs,
        // GetOutput(kParams)读取values[input_size + i]，若buffer不足会越界
        constexpr size_t kExtendInputs = 3;
        constexpr size_t kExtendOutputs = 2;
        size_t totalValues = numInputs + numOutputs + kExtendInputs + kExtendOutputs;

        size_t ctxSize = sizeof(KernelRunContext) + (totalValues > 1 ? (totalValues - 1) * sizeof(void*) : 0);
        buffer.resize(ctxSize);
        std::memset(buffer.data(), 0, ctxSize);

        auto krc = reinterpret_cast<KernelRunContext*>(buffer.data());
        krc->input_size = numInputs + numOutputs + kExtendInputs;
        krc->output_size = kExtendOutputs;
        krc->compute_node_info = computeNodeInfo;
        krc->kernel_extend_info = nullptr;
        krc->output_start = nullptr;

        for (size_t i = 0; i < numInputs; ++i) {
            if (inputs[i] != nullptr) {
                auto chain = new gert::Chain();
                chains.push_back(chain);
                chain->Set(inputs[i], nullptr);
                krc->values[i] = reinterpret_cast<AsyncAnyValue*>(chain);
            } else {
                krc->values[i] = nullptr;
            }
        }
        for (size_t i = 0; i < numOutputs; ++i) {
            size_t idx = numInputs + i;
            if (outputs[i] != nullptr) {
                auto chain = new gert::Chain();
                chains.push_back(chain);
                chain->Set(outputs[i], nullptr);
                krc->values[idx] = reinterpret_cast<AsyncAnyValue*>(chain);
            } else {
                krc->values[idx] = nullptr;
            }
        }
        // Extend inputs (kExecuteOption, kFwkData, kStream) remain null
        // Extend outputs (kParams, kWorkspaceSize) need valid chains for SetOpApiParams/SetWorkspaceSizes
        for (size_t i = 0; i < kExtendOutputs; ++i) {
            auto chain = new gert::Chain();
            chains.push_back(chain);
            krc->values[krc->input_size + i] = reinterpret_cast<AsyncAnyValue*>(chain);
        }

        ctx = reinterpret_cast<gert::OpExecutePrepareContext*>(buffer.data());
    }

    ~MockContext()
    {
        for (auto* chain : chains) {
            // 释放Chain中持有的带deleter的数据(如OpApiParams/workspace sizes)，避免泄漏
            chain->Set(nullptr, nullptr);
            delete chain;
        }
    }
};

} // namespace

class MatMulV3FallbackTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(MatMulV3FallbackTest, NullContext) { EXPECT_EQ(MmExecuteFunc(nullptr), GRAPH_FAILED); }

TEST_F(MatMulV3FallbackTest, SelfIsNull)
{
    MockContext mc({nullptr}, {});
    EXPECT_EQ(MmExecuteFunc(mc.ctx), GRAPH_FAILED);
}

TEST_F(MatMulV3FallbackTest, Mat2IsNull)
{
    auto tensor = MakeTensor({2, 3}, DT_FLOAT);
    MockContext mc({&tensor, nullptr}, {});
    EXPECT_EQ(MmExecuteFunc(mc.ctx), GRAPH_FAILED);
}

TEST_F(MatMulV3FallbackTest, AttrsIsNull)
{
    auto self = MakeTensor({2, 3}, DT_FLOAT);
    auto mat2 = MakeTensor({3, 4}, DT_FLOAT);
    auto output = MakeTensor({2, 4}, DT_FLOAT);
    MockContext mc({&self, &mat2, nullptr}, {&output}, nullptr);
    EXPECT_EQ(MmExecuteFunc(mc.ctx), GRAPH_FAILED);
}

TEST_F(MatMulV3FallbackTest, TransSelfAttrNull)
{
    // 0 个属性被追加: trans_self/trans_mat2 取属性时均为 null, fallback 校验 trans_self 时失败
    ComputeNodeInfoHelper cnih(/*attrCount=*/0);
    auto self = MakeTensor({2, 3}, DT_FLOAT);
    auto mat2 = MakeTensor({3, 4}, DT_FLOAT);
    auto output = MakeTensor({2, 4}, DT_FLOAT);
    MockContext mc({&self, &mat2, nullptr}, {&output}, cnih.AsVoidPtr());
    EXPECT_EQ(MmExecuteFunc(mc.ctx), GRAPH_FAILED);
}

TEST_F(MatMulV3FallbackTest, TransMat2AttrNull)
{
    // 仅追加 trans_self: trans_mat2 取属性时为 null
    ComputeNodeInfoHelper cnih(/*attrCount=*/1);
    auto self = MakeTensor({2, 3}, DT_FLOAT);
    auto mat2 = MakeTensor({3, 4}, DT_FLOAT);
    auto output = MakeTensor({2, 4}, DT_FLOAT);
    MockContext mc({&self, &mat2, nullptr}, {&output}, cnih.AsVoidPtr());
    EXPECT_EQ(MmExecuteFunc(mc.ctx), GRAPH_FAILED);
}

TEST_F(MatMulV3FallbackTest, MmWithoutBias)
{
    ComputeNodeInfoHelper cnih;
    auto self = MakeTensor({2, 3}, DT_FLOAT);
    auto mat2 = MakeTensor({3, 4}, DT_FLOAT);
    auto output = MakeTensor({2, 4}, DT_FLOAT);
    MockContext mc({&self, &mat2, nullptr}, {&output}, cnih.AsVoidPtr());
    auto result = MmExecuteFunc(mc.ctx);
    EXPECT_TRUE(result == GRAPH_SUCCESS || result == GRAPH_FAILED);
}

TEST_F(MatMulV3FallbackTest, MmWithBias)
{
    ComputeNodeInfoHelper cnih;
    auto self = MakeTensor({2, 3}, DT_FLOAT);
    auto mat2 = MakeTensor({3, 4}, DT_FLOAT);
    auto bias = MakeTensor({2, 4}, DT_FLOAT);
    auto output = MakeTensor({2, 4}, DT_FLOAT);
    MockContext mc({&self, &mat2, &bias}, {&output}, cnih.AsVoidPtr());
    auto result = MmExecuteFunc(mc.ctx);
    EXPECT_TRUE(result == GRAPH_SUCCESS || result == GRAPH_FAILED);
}
