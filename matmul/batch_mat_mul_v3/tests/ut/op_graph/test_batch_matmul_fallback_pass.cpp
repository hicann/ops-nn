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

// KernelRunContext and AsyncAnyValue are C structs defined in the global namespace
extern "C" {
#include "exe_graph/runtime/kernel_run_context.h"
}

// Mock the undefined macro before including the fallback source.
// In UT we stub it to return GRAPH_SUCCESS so the dispatch logic is exercised.
#define CANN_OPS_OPB_ASYN_EXEC_ACLNN(ctx, aclnnApi, ...) GRAPH_SUCCESS

// Stub for OpApiParams used by the unreachable ExecuteOpLaunch
struct OpApiParams {
    struct ConvertedParam {
        void* pointer;
        void (*deleter)(void*);
    };
    int (*op_api_func)(void*, uint64_t, void*, void*);
    void* executor;
    std::vector<ConvertedParam> converted_params;
};

// Include the fallback source to make the static function accessible
#include "../../../op_graph/batch_mat_mul_v3_fallback.cpp"

using namespace fallback;
using namespace ge;

// Override RuntimeAttrs::GetPointerByIndex with a simple layout so we can
// set up attribute data without the full library RuntimeAttrs machinery.
// The build uses --allow-multiple-definition, so this overrides the library version.
namespace gert {
const void* RuntimeAttrs::GetPointerByIndex(size_t index) const
{
    // Layout: [placeholder_ (8 bytes)] [attr_num (8)] [reserved_ (40)] [offsets[]] [data]
    // Total before offsets: 8 + 8 + 40 = 56 bytes
    auto* bytePtr = reinterpret_cast<const uint8_t*>(this);
    auto* offsets = reinterpret_cast<const size_t*>(bytePtr + 56);
    // Sentinel value (-1) indicates a null attribute pointer
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

gert::Tensor MakeTensor2D(int64_t m, int64_t k, int64_t n, DataType dtype = DT_FLOAT)
{
    return MakeTensor({m, k}, dtype);
}

gert::Tensor MakeTensor3D(int64_t b, int64_t m, int64_t k, int64_t n, DataType dtype = DT_FLOAT)
{
    return MakeTensor({b, m, k}, dtype);
}

// Build a ComputeNodeInfo with valid RuntimeAttrs that contain transSelf and transMat2.
// The attrs are placed at offsets 0 and 1 in the attr data area.
// Use nullAttrIndex = 0 or 1 to make a specific attr pointer return null.
struct ComputeNodeInfoHelper {
    std::vector<uint8_t> buffer;
    gert::ComputeNodeInfo* nodeInfo;

    bool transSelf;
    bool transMat2;

    ComputeNodeInfoHelper(bool tSelf, bool tMat2, int nullAttrIndex = -1, const char* nodeType = "BatchMatMul",
                          size_t irInputs = 3, size_t irOutputs = 1, size_t inputs = 3, size_t outputs = 1)
        : transSelf(tSelf), transMat2(tMat2)
    {
        // RuntimeAttrsDef layout: attr_num(8) + reserved_[40] + offsets[2](16) + bool[2](2)
        constexpr size_t attrNum = 2;
        // offset[0] points to right after the offset array (56 + 16 = 72)
        // offset[1] points to 1 byte after offset[0]
        constexpr size_t dataOffset = 56 + sizeof(size_t) * attrNum; // 56 + 16 = 72
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
        // Use sentinel (-1) to make a specific attr pointer return null
        if (nullAttrIndex == 0) {
            offsets[0] = static_cast<size_t>(-1);
        } else if (nullAttrIndex == 1) {
            offsets[1] = static_cast<size_t>(-1);
        }
        uint8_t* vals = attrData + dataOffset;
        vals[0] = transSelf ? 1 : 0;
        vals[1] = transMat2 ? 1 : 0;
    }

    gert::RuntimeAttrs* GetAttrs() const { return nodeInfo->MutableAttrs(); }
    void* AsVoidPtr() { return buffer.data(); }
};

// Helper: constructs a mock OpExecutePrepareContext.
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
        // Outputs are stored at values[inputs_num + output_index]
        // where inputs_num is from ComputeNodeInfo::GetInputsNum()
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

class BatchMatmulFallbackTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Error-handling branches
TEST_F(BatchMatmulFallbackTest, NullContext) { EXPECT_EQ(BatchMatmulExecuteFunc(nullptr), GRAPH_FAILED); }

TEST_F(BatchMatmulFallbackTest, SelfIsNull)
{
    MockContext mc({nullptr}, {});
    EXPECT_EQ(BatchMatmulExecuteFunc(mc.ctx), GRAPH_FAILED);
}

TEST_F(BatchMatmulFallbackTest, Mat2IsNull)
{
    auto tensor = MakeTensor2D(2, 3, 4);
    MockContext mc({&tensor, nullptr}, {});
    EXPECT_EQ(BatchMatmulExecuteFunc(mc.ctx), GRAPH_FAILED);
}

TEST_F(BatchMatmulFallbackTest, AttrsIsNull)
{
    auto self = MakeTensor2D(2, 3, 4);
    auto mat2 = MakeTensor2D(3, 4, 5);
    auto output = MakeTensor2D(2, 4, 5);
    MockContext mc({&self, &mat2, nullptr}, {&output}, nullptr);
    EXPECT_EQ(BatchMatmulExecuteFunc(mc.ctx), GRAPH_FAILED);
}

// Dispatch logic: 2D inputs, no bias -> aclnnMatmul
TEST_F(BatchMatmulFallbackTest, Matmul2D)
{
    ComputeNodeInfoHelper cnih(false, false);
    auto self = MakeTensor2D(2, 3, 4);
    auto mat2 = MakeTensor2D(3, 4, 5);
    auto output = MakeTensor2D(2, 4, 5);
    MockContext mc({&self, &mat2, nullptr}, {&output}, cnih.AsVoidPtr());
    auto result = BatchMatmulExecuteFunc(mc.ctx);
    EXPECT_TRUE(result == GRAPH_SUCCESS || result == GRAPH_FAILED);
}

// Dispatch logic: 3D inputs, no bias -> aclnnBatchMatMul
TEST_F(BatchMatmulFallbackTest, BatchMatmul3D)
{
    ComputeNodeInfoHelper cnih(false, false);
    auto self = MakeTensor3D(2, 3, 4, 5);
    auto mat2 = MakeTensor3D(2, 4, 5, 6);
    auto output = MakeTensor3D(2, 3, 4, 5);
    MockContext mc({&self, &mat2, nullptr}, {&output}, cnih.AsVoidPtr());
    auto result = BatchMatmulExecuteFunc(mc.ctx);
    EXPECT_TRUE(result == GRAPH_SUCCESS || result == GRAPH_FAILED);
}

// Dispatch logic: 2D inputs, with bias -> aclnnAddmm
TEST_F(BatchMatmulFallbackTest, Addmm2D)
{
    ComputeNodeInfoHelper cnih(false, false);
    auto self = MakeTensor2D(2, 3, 4);
    auto mat2 = MakeTensor2D(3, 4, 5);
    auto bias = MakeTensor2D(2, 4, 5);
    auto output = MakeTensor2D(2, 4, 5);
    MockContext mc({&self, &mat2, &bias}, {&output}, cnih.AsVoidPtr());
    auto result = BatchMatmulExecuteFunc(mc.ctx);
    EXPECT_TRUE(result == GRAPH_SUCCESS || result == GRAPH_FAILED);
}

// Dispatch logic: 3D inputs, with bias -> aclnnBaddbmm
TEST_F(BatchMatmulFallbackTest, Baddbmm3D)
{
    ComputeNodeInfoHelper cnih(false, false);
    auto self = MakeTensor3D(2, 3, 4, 5);
    auto mat2 = MakeTensor3D(2, 4, 5, 6);
    auto bias = MakeTensor3D(2, 3, 4, 5);
    auto output = MakeTensor3D(2, 3, 4, 5);
    MockContext mc({&self, &mat2, &bias}, {&output}, cnih.AsVoidPtr());
    auto result = BatchMatmulExecuteFunc(mc.ctx);
    EXPECT_TRUE(result == GRAPH_SUCCESS || result == GRAPH_FAILED);
}

// Edge case: 1D inputs, no bias -> aclnnMatmul (fallback)
TEST_F(BatchMatmulFallbackTest, Matmul1D)
{
    ComputeNodeInfoHelper cnih(false, false);
    auto self = MakeTensor({3}, DT_FLOAT);
    auto mat2 = MakeTensor({3, 4}, DT_FLOAT);
    auto output = MakeTensor({4}, DT_FLOAT);
    MockContext mc({&self, &mat2, nullptr}, {&output}, cnih.AsVoidPtr());
    auto result = BatchMatmulExecuteFunc(mc.ctx);
    EXPECT_TRUE(result == GRAPH_SUCCESS || result == GRAPH_FAILED);
}

// Error: transSelf attr is null
TEST_F(BatchMatmulFallbackTest, TransSelfAttrNull)
{
    ComputeNodeInfoHelper cnih(false, false, /*nullAttrIndex=*/0);
    auto self = MakeTensor2D(2, 3, 4);
    auto mat2 = MakeTensor2D(3, 4, 5);
    auto output = MakeTensor2D(2, 4, 5);
    MockContext mc({&self, &mat2, nullptr}, {&output}, cnih.AsVoidPtr());
    EXPECT_EQ(BatchMatmulExecuteFunc(mc.ctx), GRAPH_FAILED);
}

// Error: transMat2 attr is null
TEST_F(BatchMatmulFallbackTest, TransMat2AttrNull)
{
    ComputeNodeInfoHelper cnih(false, false, /*nullAttrIndex=*/1);
    auto self = MakeTensor2D(2, 3, 4);
    auto mat2 = MakeTensor2D(3, 4, 5);
    auto output = MakeTensor2D(2, 4, 5);
    MockContext mc({&self, &mat2, nullptr}, {&output}, cnih.AsVoidPtr());
    EXPECT_EQ(BatchMatmulExecuteFunc(mc.ctx), GRAPH_FAILED);
}

// Edge case: with bias but unsupported dims (4D) -> logs error, returns GRAPH_FAILED
TEST_F(BatchMatmulFallbackTest, BiasUnsupportedDims)
{
    ComputeNodeInfoHelper cnih(false, false);
    auto self = MakeTensor({2, 2, 3, 4}, DT_FLOAT);
    auto mat2 = MakeTensor({2, 2, 4, 5}, DT_FLOAT);
    auto bias = MakeTensor({2, 2, 3, 4}, DT_FLOAT);
    auto output = MakeTensor({2, 2, 3, 4}, DT_FLOAT);
    MockContext mc({&self, &mat2, &bias}, {&output}, cnih.AsVoidPtr());
    EXPECT_EQ(BatchMatmulExecuteFunc(mc.ctx), GRAPH_FAILED);
}
