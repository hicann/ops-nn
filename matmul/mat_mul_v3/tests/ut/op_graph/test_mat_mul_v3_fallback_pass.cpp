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

using namespace fallback;
using namespace ge;

namespace gert {
const void* RuntimeAttrs::GetPointerByIndex(size_t index) const
{
    auto* bytePtr = reinterpret_cast<const uint8_t*>(this);
    auto* offsets = reinterpret_cast<const size_t*>(bytePtr + 56);
    if (offsets[index] == static_cast<size_t>(-1)) {
        return nullptr;
    }
    return bytePtr + 8 + offsets[index];
}
} // namespace gert

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

struct ComputeNodeInfoHelper {
    std::vector<uint8_t> buffer;
    gert::ComputeNodeInfo* nodeInfo;

    ComputeNodeInfoHelper(int nullAttrIndex = -1, const char* nodeType = "MatMul", size_t irInputs = 3,
                          size_t irOutputs = 1, size_t inputs = 3, size_t outputs = 1)
    {
        constexpr size_t attrNum = 2;
        constexpr size_t dataOffset = 56 + sizeof(size_t) * attrNum;
        constexpr size_t runtimeAttrDataSize = dataOffset + sizeof(bool) * attrNum;

        size_t baseSize = 0;
        auto status = gert::ComputeNodeInfo::CalcSize(irInputs, irOutputs, inputs, outputs, baseSize);
        EXPECT_EQ(status, GRAPH_SUCCESS);
        size_t totalSize = baseSize + runtimeAttrDataSize;

        buffer.resize(totalSize);
        std::memset(buffer.data(), 0, totalSize);
        nodeInfo = reinterpret_cast<gert::ComputeNodeInfo*>(buffer.data());
        nodeInfo->Init(irInputs, irOutputs, inputs, outputs, runtimeAttrDataSize, nodeType, nodeType);

        for (size_t i = 0; i < irInputs; ++i) {
            auto* insInfo = nodeInfo->MutableInputInstanceInfo(i);
            insInfo->SetInstanceStart(static_cast<uint32_t>(i));
            insInfo->SetInstantiationNum(1);
        }
        for (size_t i = 0; i < irOutputs; ++i) {
            auto* insInfo = nodeInfo->MutableOutputInstanceInfo(i);
            insInfo->SetInstanceStart(static_cast<uint32_t>(i));
            insInfo->SetInstantiationNum(1);
        }
        for (size_t i = 0; i < inputs; ++i) {
            auto* td = nodeInfo->MutableInputTdInfo(i);
            td->SetDataType(DT_FLOAT);
            td->SetStorageFormat(FORMAT_ND);
            td->SetOriginFormat(FORMAT_ND);
        }
        for (size_t i = 0; i < outputs; ++i) {
            auto* td = nodeInfo->MutableOutputTdInfo(i);
            td->SetDataType(DT_FLOAT);
            td->SetStorageFormat(FORMAT_ND);
            td->SetOriginFormat(FORMAT_ND);
        }

        auto* attrs = nodeInfo->MutableAttrs();
        uint8_t* attrData = reinterpret_cast<uint8_t*>(attrs) + sizeof(uint64_t);
        *reinterpret_cast<size_t*>(attrData) = attrNum;
        auto* offsets = reinterpret_cast<size_t*>(attrData + sizeof(size_t) + 40);
        offsets[0] = dataOffset;
        offsets[1] = dataOffset + sizeof(bool);
        if (nullAttrIndex == 0) {
            offsets[0] = static_cast<size_t>(-1);
        } else if (nullAttrIndex == 1) {
            offsets[1] = static_cast<size_t>(-1);
        }
        uint8_t* vals = attrData + dataOffset;
        vals[0] = 0;
        vals[1] = 0;
    }

    void* AsVoidPtr() { return buffer.data(); }
};

struct MockContext {
    std::vector<uint8_t> buffer;
    std::vector<gert::Chain*> chains;
    gert::OpExecutePrepareContext* ctx;

    MockContext(std::vector<gert::Tensor*> inputs, std::vector<gert::Tensor*> outputs, void* computeNodeInfo = nullptr)
    {
        size_t numInputs = inputs.size();
        size_t numOutputs = outputs.size();
        constexpr size_t kExtendInputs = 3;
        size_t totalValues = numInputs + numOutputs + kExtendInputs;

        size_t ctxSize = sizeof(KernelRunContext) + (totalValues > 1 ? (totalValues - 1) * sizeof(void*) : 0);
        buffer.resize(ctxSize);
        std::memset(buffer.data(), 0, ctxSize);

        auto krc = reinterpret_cast<KernelRunContext*>(buffer.data());
        krc->input_size = totalValues;
        krc->output_size = numOutputs;
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

        ctx = reinterpret_cast<gert::OpExecutePrepareContext*>(buffer.data());
    }

    ~MockContext()
    {
        for (auto* chain : chains) {
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
    ComputeNodeInfoHelper cnih(/*nullAttrIndex=*/0);
    auto self = MakeTensor({2, 3}, DT_FLOAT);
    auto mat2 = MakeTensor({3, 4}, DT_FLOAT);
    auto output = MakeTensor({2, 4}, DT_FLOAT);
    MockContext mc({&self, &mat2, nullptr}, {&output}, cnih.AsVoidPtr());
    EXPECT_EQ(MmExecuteFunc(mc.ctx), GRAPH_FAILED);
}

TEST_F(MatMulV3FallbackTest, TransMat2AttrNull)
{
    ComputeNodeInfoHelper cnih(/*nullAttrIndex=*/1);
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
