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

// 先包含真实 op_fallback.h 建立 include guard, 使被测源文件中的 #include "op_fallback.h" 变为空操作,
// 再重定义 GET_OP_API_FUNC 与 CANN_OPS_OPB_SYN_EXEC_ACLNN 为桩实现(后定义的宏生效),
// 从而在单测中确定性隔离 dlopen(libopapi.so) 与真实 aclnn 调用:
// 1. aclCreateTensor/aclDestroyTensor 由本文件提供的假实现替代, 支持按调用序号注入失败;
// 2. CANN_OPS_OPB_SYN_EXEC_ACLNN 返回可配置的桩结果, 覆盖执行成功/失败两条返回路径。
#include "op_fallback.h"

namespace {

// 第 N 次 aclCreateTensor 假实现返回 nullptr(-1 表示不注入失败), 用于触发 ConvertMmType 的失败分支
int g_createTensorCallCount = 0;
int g_failAtCreateCall = -1;
// CANN_OPS_OPB_SYN_EXEC_ACLNN 桩返回值, 用于触发 ExecuteFunc 中 apiRet != GRAPH_SUCCESS 分支
ge::graphStatus g_stubExecRet = ge::GRAPH_SUCCESS;

aclTensor* FakeAclCreateTensor(const int64_t* viewDims, uint64_t viewDimsNum, aclDataType dataType,
                               const int64_t* stride, int64_t offset, aclFormat format, const int64_t* storageDims,
                               uint64_t storageDimsNum, void* tensorData)
{
    (void)viewDims;
    (void)viewDimsNum;
    (void)dataType;
    (void)stride;
    (void)offset;
    (void)format;
    (void)storageDims;
    (void)storageDimsNum;
    (void)tensorData;
    const int call = g_createTensorCallCount++;
    if (call == g_failAtCreateCall) {
        return nullptr;
    }
    return reinterpret_cast<aclTensor*>(new int64_t(call));
}

int FakeAclDestroyTensor(const aclTensor* tensor)
{
    delete reinterpret_cast<int64_t*>(const_cast<aclTensor*>(tensor));
    return 0;
}

void* GetFakeOpApiFunc(const char* apiName)
{
    if (std::strcmp(apiName, "aclCreateTensor") == 0) {
        return reinterpret_cast<void*>(&FakeAclCreateTensor);
    }
    if (std::strcmp(apiName, "aclDestroyTensor") == 0) {
        return reinterpret_cast<void*>(&FakeAclDestroyTensor);
    }
    return nullptr;
}

} // namespace

#define GET_OP_API_FUNC(apiName) reinterpret_cast<_##apiName>(GetFakeOpApiFunc(#apiName))
#define CANN_OPS_OPB_SYN_EXEC_ACLNN(ctx, aclnnApi, ...) g_stubExecRet

// Include the fallback source to make the static function accessible
#include "../../../op_graph/weight_quant_batch_matmul_v2_fallback.cpp"

// Build ComputeNodeInfo (attrs included) through the public OpKernelContextBuilder API, so the
// test never hand-rolls gert::RuntimeAttrs' private memory layout and no longer redefines the
// library symbol gert::RuntimeAttrs::GetPointerByIndex (which relied on
// --allow-multiple-definition, an ODR violation).
#include "compute_node_info_faker.h"

using namespace fallback;
using namespace ge;

namespace {

// WeightQuantBatchMatmulV2 IR: 7 inputs(x/weight/antiquant_scale 必选, 后 4 个可选), 1 output, 5 attrs
constexpr size_t WQBMM_ATTR_NUM = 5;

gert::Tensor MakeTensor(std::initializer_list<int64_t> dims, DataType dtype, Format format = FORMAT_ND)
{
    gert::StorageShape storageShape{dims, dims};
    gert::StorageFormat storageFormat;
    storageFormat.SetOriginFormat(format);
    storageFormat.SetStorageFormat(format);
    void* dummyAddr = reinterpret_cast<void*>(0xDEADBEEF);
    return gert::Tensor(storageShape, storageFormat, gert::kOnDeviceHbm, dtype, dummyAddr);
}

// 属性规格: 五个属性按 IR 顺序通过公开构建器 AppendAttr 追加;
// attrCount 截断尾部, 未追加到的属性 index 取属性时按真实库语义返回 nullptr(走默认值分支)
struct AttrsSpec {
    bool transposeX = false;
    bool transposeWeight = false;
    int64_t groupSize = 0;
    int64_t dtype = -1;
    int64_t innerPrecise = 0;
    size_t attrCount = WQBMM_ATTR_NUM;
};

// Build a ComputeNodeInfo for WeightQuantBatchMatmulV2 (7 inputs, 1 output, 5 attrs) through
// the public builder API. The five proto attrs (transpose_x, transpose_weight,
// antiquant_group_size, dtype, inner_precise) are appended in proto order; attrs.attrCount
// truncates the tail so the dropped indices resolve to null with the real library semantics.
struct ComputeNodeInfoHelper {
    gert::ComputeNodeInfoFaker faker;
    const void* nodeInfo = nullptr;

    // presentInputs: 每个 IR 输入是否实例化(可选输入未实例化时 GetOptionalInputTensor 返回 nullptr)
    ComputeNodeInfoHelper(const std::vector<bool>& presentInputs, const AttrsSpec& attrs,
                          const char* nodeType = "WeightQuantBatchMatmulV2")
    {
        std::vector<uint32_t> instanceNums;
        instanceNums.reserve(presentInputs.size());
        for (bool present : presentInputs) {
            instanceNums.push_back(present ? 1U : 0U);
        }

        faker.NodeType(nodeType).IrInstanceNum(instanceNums, {1});
        for (size_t i = 0; i < presentInputs.size(); ++i) {
            if (presentInputs[i]) {
                faker.InputTd(i, DT_FLOAT16, FORMAT_ND, FORMAT_ND);
            }
        }
        faker.OutputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND);
        if (attrs.attrCount >= 1) {
            faker.AppendAttr(attrs.transposeX); // transpose_x
        }
        if (attrs.attrCount >= 2) {
            faker.AppendAttr(attrs.transposeWeight); // transpose_weight
        }
        if (attrs.attrCount >= 3) {
            faker.AppendAttr(attrs.groupSize); // antiquant_group_size
        }
        if (attrs.attrCount >= 4) {
            faker.AppendAttr(attrs.dtype); // dtype
        }
        if (attrs.attrCount >= 5) {
            faker.AppendAttr(attrs.innerPrecise); // inner_precise
        }
        nodeInfo = faker.Build();
    }

    void* AsVoidPtr() { return const_cast<void*>(nodeInfo); }
};

// Mock gert::OpExecuteContext 底层的 KernelRunContext 内存布局:
//   values = [inputs..., outputs..., extend inputs(kAllocate/kStream/kExecuteOption/kExecuteFunc/kFwkData),
//             extend outputs(kBlockMemory)]
// 其中 input_size = inputs + outputs + 5, output_size = 1
struct MockContext {
    std::vector<uint8_t> buffer;
    std::vector<gert::Chain*> chains;
    gert::OpExecuteContext* ctx;

    MockContext(std::vector<gert::Tensor*> inputs, std::vector<gert::Tensor*> outputs, void* computeNodeInfo = nullptr)
    {
        const size_t numInputs = inputs.size();
        const size_t numOutputs = outputs.size();
        constexpr size_t kExtendInputs = 5;
        constexpr size_t kExtendOutputs = 1;
        const size_t inputSize = numInputs + numOutputs + kExtendInputs;
        const size_t outputSize = kExtendOutputs;
        const size_t totalValues = inputSize + outputSize;

        size_t ctxSize = sizeof(KernelRunContext) + (totalValues > 1 ? (totalValues - 1) * sizeof(void*) : 0);
        buffer.resize(ctxSize);
        std::memset(buffer.data(), 0, ctxSize);

        auto krc = reinterpret_cast<KernelRunContext*>(buffer.data());
        krc->input_size = inputSize;
        krc->output_size = outputSize;
        krc->compute_node_info = computeNodeInfo;
        krc->kernel_extend_info = nullptr;
        krc->output_start = nullptr;

        for (size_t i = 0; i < numInputs; ++i) {
            SetTensorValue(krc, i, inputs[i]);
        }
        for (size_t i = 0; i < numOutputs; ++i) {
            SetTensorValue(krc, numInputs + i, outputs[i]);
        }
        // kAllocate 槽位提供空数据 chain(防御库内 FreeWorkspace 读取空指针), 其余扩展槽位保持为空
        auto* allocateChain = new gert::Chain();
        chains.push_back(allocateChain);
        allocateChain->Set(nullptr, nullptr);
        krc->values[numInputs + numOutputs] = reinterpret_cast<AsyncAnyValue*>(allocateChain);

        ctx = reinterpret_cast<gert::OpExecuteContext*>(buffer.data());
    }

    ~MockContext()
    {
        for (auto* chain : chains) {
            chain->Set(nullptr, nullptr);
            delete chain;
        }
    }

private:
    void SetTensorValue(KernelRunContext* krc, size_t index, gert::Tensor* tensor)
    {
        if (tensor != nullptr) {
            auto* chain = new gert::Chain();
            chains.push_back(chain);
            chain->Set(tensor, nullptr);
            krc->values[index] = reinterpret_cast<AsyncAnyValue*>(chain);
        } else {
            krc->values[index] = nullptr;
        }
    }
};

} // namespace

class WeightQuantBmmV2FallbackTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        g_createTensorCallCount = 0;
        g_failAtCreateCall = -1;
        g_stubExecRet = GRAPH_SUCCESS;
    }

    void TearDown() override {}

    // 典型输入集合(m=2,k=3,n=4): x(m,k) fp16 / weight(k,n) int8 / antiquant_scale(n,) fp16 / y(m,n) fp16,
    // 形状可通过构造参数定制(transpose_x=true 时 x 为 (k,m), transpose_weight=true 时 weight 为 (n,k))
    struct FullTensors {
        gert::Tensor x;
        gert::Tensor weight;
        gert::Tensor antiquantScale;
        gert::Tensor antiquantOffset;
        gert::Tensor quantScale;
        gert::Tensor quantOffset;
        gert::Tensor bias;
        gert::Tensor y;

        explicit FullTensors(std::initializer_list<int64_t> xDims = {2, 3},
                             std::initializer_list<int64_t> weightDims = {3, 4},
                             std::initializer_list<int64_t> scaleDims = {4})
            : x(MakeTensor(xDims, DT_FLOAT16)),
              weight(MakeTensor(weightDims, DT_INT8)),
              antiquantScale(MakeTensor(scaleDims, DT_FLOAT16)),
              antiquantOffset(MakeTensor({4}, DT_FLOAT16)),
              quantScale(MakeTensor({1, 4}, DT_FLOAT)),
              quantOffset(MakeTensor({1, 4}, DT_FLOAT)),
              bias(MakeTensor({4}, DT_FLOAT16)),
              y(MakeTensor({2, 4}, DT_FLOAT16))
        {}

        std::vector<gert::Tensor*> AllInputs()
        {
            return {&x, &weight, &antiquantScale, &antiquantOffset, &quantScale, &quantOffset, &bias};
        }

        std::vector<bool> AllPresent() const { return {true, true, true, true, true, true, true}; }
    };

    graphStatus RunExecute(const AttrsSpec& attrs, const std::vector<bool>& present,
                           const std::vector<gert::Tensor*>& inputs, gert::Tensor* output)
    {
        ComputeNodeInfoHelper cnih(present, attrs);
        MockContext ctx(inputs, std::vector<gert::Tensor*>{output}, cnih.AsVoidPtr());
        return WeightQuantBmmV2ExecuteFunc(ctx.ctx);
    }
};

TEST_F(WeightQuantBmmV2FallbackTest, NullContext) { EXPECT_EQ(WeightQuantBmmV2ExecuteFunc(nullptr), GRAPH_FAILED); }

TEST_F(WeightQuantBmmV2FallbackTest, AclTensorDeleterNullptr)
{
    // unique_ptr 析构时空指针不会触发 deleter, 此处直接验证 deleter 自身的空指针安全
    AclTensorDeleter deleter;
    deleter(nullptr);
}

TEST_F(WeightQuantBmmV2FallbackTest, XIsNull)
{
    FullTensors t;
    auto inputs = t.AllInputs();
    inputs[0] = nullptr;
    EXPECT_EQ(RunExecute(AttrsSpec{}, t.AllPresent(), inputs, &t.y), GRAPH_FAILED);
}

TEST_F(WeightQuantBmmV2FallbackTest, WeightIsNull)
{
    FullTensors t;
    auto inputs = t.AllInputs();
    inputs[1] = nullptr;
    EXPECT_EQ(RunExecute(AttrsSpec{}, t.AllPresent(), inputs, &t.y), GRAPH_FAILED);
}

TEST_F(WeightQuantBmmV2FallbackTest, AntiquantScaleIsNull)
{
    FullTensors t;
    auto inputs = t.AllInputs();
    inputs[2] = nullptr;
    EXPECT_EQ(RunExecute(AttrsSpec{}, t.AllPresent(), inputs, &t.y), GRAPH_FAILED);
}

TEST_F(WeightQuantBmmV2FallbackTest, YIsNull)
{
    FullTensors t;
    EXPECT_EQ(RunExecute(AttrsSpec{}, t.AllPresent(), t.AllInputs(), nullptr), GRAPH_FAILED);
}

TEST_F(WeightQuantBmmV2FallbackTest, AttrsIsNull)
{
    FullTensors t;
    MockContext ctx(t.AllInputs(), {&t.y}, nullptr);
    EXPECT_EQ(WeightQuantBmmV2ExecuteFunc(ctx.ctx), GRAPH_FAILED);
}

TEST_F(WeightQuantBmmV2FallbackTest, ConvertXFail)
{
    FullTensors t;
    g_failAtCreateCall = 0; // 第 1 次 aclCreateTensor(x 转换)失败
    EXPECT_EQ(RunExecute(AttrsSpec{}, t.AllPresent(), t.AllInputs(), &t.y), GRAPH_FAILED);
}

TEST_F(WeightQuantBmmV2FallbackTest, ConvertWeightFail)
{
    FullTensors t;
    g_failAtCreateCall = 1; // 第 2 次 aclCreateTensor(weight 转换)失败
    EXPECT_EQ(RunExecute(AttrsSpec{}, t.AllPresent(), t.AllInputs(), &t.y), GRAPH_FAILED);
}

TEST_F(WeightQuantBmmV2FallbackTest, ConvertAntiquantScaleFail)
{
    // antiquant_scale 使用 2D, 保证走桩化的 aclCreateTensor 调用(1D 会进入头文件内的 ConvertType)
    FullTensors t({2, 3}, {3, 4}, {3, 4});
    g_failAtCreateCall = 2; // 第 3 次 aclCreateTensor(antiquant_scale 转换)失败
    EXPECT_EQ(RunExecute(AttrsSpec{}, t.AllPresent(), t.AllInputs(), &t.y), GRAPH_FAILED);
}

TEST_F(WeightQuantBmmV2FallbackTest, AclnnV2ExecFail)
{
    FullTensors t;
    g_stubExecRet = GRAPH_FAILED; // innerPrecise=0 走 V2 分支, apiRet 失败
    EXPECT_EQ(RunExecute(AttrsSpec{}, t.AllPresent(), t.AllInputs(), &t.y), GRAPH_FAILED);
}

TEST_F(WeightQuantBmmV2FallbackTest, AclnnV3ExecFail)
{
    FullTensors t;
    AttrsSpec attrs;
    attrs.innerPrecise = 1; // 走 V3 分支
    g_stubExecRet = GRAPH_FAILED;
    EXPECT_EQ(RunExecute(attrs, t.AllPresent(), t.AllInputs(), &t.y), GRAPH_FAILED);
}

TEST_F(WeightQuantBmmV2FallbackTest, ExecV2WithAllOptionalInputs)
{
    FullTensors t; // antiquant_scale 为 1D, 覆盖 ConvertMmType 的 ConvertType 分支
    EXPECT_EQ(RunExecute(AttrsSpec{}, t.AllPresent(), t.AllInputs(), &t.y), GRAPH_SUCCESS);
}

TEST_F(WeightQuantBmmV2FallbackTest, ExecV3WithAllOptionalInputs)
{
    FullTensors t;
    AttrsSpec attrs;
    attrs.innerPrecise = 1;
    attrs.groupSize = 32;
    EXPECT_EQ(RunExecute(attrs, t.AllPresent(), t.AllInputs(), &t.y), GRAPH_SUCCESS);
}

TEST_F(WeightQuantBmmV2FallbackTest, ExecInnerPreciseOtherValue)
{
    FullTensors t;
    AttrsSpec attrs;
    attrs.innerPrecise = 2; // 非 0 非 1, 走 else 分支(V2)
    EXPECT_EQ(RunExecute(attrs, t.AllPresent(), t.AllInputs(), &t.y), GRAPH_SUCCESS);
}

TEST_F(WeightQuantBmmV2FallbackTest, ExecOptionalInputsAllAbsent)
{
    FullTensors t;
    // 仅 3 个必选输入实例化, 4 个可选输入未实例化, 覆盖可选输入为空路径
    const std::vector<bool> present = {true, true, true, false, false, false, false};
    const std::vector<gert::Tensor*> inputs = {&t.x, &t.weight, &t.antiquantScale};
    EXPECT_EQ(RunExecute(AttrsSpec{}, present, inputs, &t.y), GRAPH_SUCCESS);
}

TEST_F(WeightQuantBmmV2FallbackTest, ExecMixedOptionalInputs)
{
    FullTensors t;
    // antiquant_offset 存在, 其余可选输入不存在
    const std::vector<bool> present = {true, true, true, true, false, false, false};
    const std::vector<gert::Tensor*> inputs = {&t.x, &t.weight, &t.antiquantScale, &t.antiquantOffset};
    EXPECT_EQ(RunExecute(AttrsSpec{}, present, inputs, &t.y), GRAPH_SUCCESS);
}

TEST_F(WeightQuantBmmV2FallbackTest, TransposeXTrue)
{
    FullTensors t({3, 2}); // transpose_x=true 时 x 形状为 (k, m)
    AttrsSpec attrs;
    attrs.transposeX = true;
    EXPECT_EQ(RunExecute(attrs, t.AllPresent(), t.AllInputs(), &t.y), GRAPH_SUCCESS);
}

TEST_F(WeightQuantBmmV2FallbackTest, TransposeWeightTrue)
{
    FullTensors t({2, 3}, {4, 3}, {4, 3}); // weight 为 (n,k), 2D scale 覆盖转置交换分支
    AttrsSpec attrs;
    attrs.transposeWeight = true;
    EXPECT_EQ(RunExecute(attrs, t.AllPresent(), t.AllInputs(), &t.y), GRAPH_SUCCESS);
}

TEST_F(WeightQuantBmmV2FallbackTest, TransposeBothTrue)
{
    FullTensors t({3, 2}, {4, 3}, {4, 3});
    AttrsSpec attrs;
    attrs.transposeX = true;
    attrs.transposeWeight = true;
    attrs.innerPrecise = 1;
    EXPECT_EQ(RunExecute(attrs, t.AllPresent(), t.AllInputs(), &t.y), GRAPH_SUCCESS);
}

TEST_F(WeightQuantBmmV2FallbackTest, Batched3DTensors)
{
    // 3D 批量输入覆盖多维 stride 计算循环
    gert::Tensor x = MakeTensor({2, 2, 3}, DT_FLOAT16);
    gert::Tensor weight = MakeTensor({2, 3, 4}, DT_INT8);
    gert::Tensor antiquantScale = MakeTensor({2, 4}, DT_FLOAT16);
    gert::Tensor y = MakeTensor({2, 2, 4}, DT_FLOAT16);
    AttrsSpec attrs;
    attrs.innerPrecise = 1;
    const std::vector<bool> present = {true, true, true, false, false, false, false};
    const std::vector<gert::Tensor*> inputs = {&x, &weight, &antiquantScale};
    EXPECT_EQ(RunExecute(attrs, present, inputs, &y), GRAPH_SUCCESS);
}

// 属性从 transpose_x 起全部缺失(真实语义为尾部截断): transpose_x/transpose_weight 走默认 false,
// antiquant_group_size 走默认 0, inner_precise 走默认 0(V2 分支)
TEST_F(WeightQuantBmmV2FallbackTest, AttrsFromTransposeXNull)
{
    FullTensors t;
    AttrsSpec attrs;
    attrs.attrCount = 0;
    EXPECT_EQ(RunExecute(attrs, t.AllPresent(), t.AllInputs(), &t.y), GRAPH_SUCCESS);
}

// transpose_x 之后(含 transpose_weight)的属性缺失: transpose_weight 走默认 false
TEST_F(WeightQuantBmmV2FallbackTest, AttrsFromTransposeWeightNull)
{
    FullTensors t;
    AttrsSpec attrs;
    attrs.attrCount = 1;
    EXPECT_EQ(RunExecute(attrs, t.AllPresent(), t.AllInputs(), &t.y), GRAPH_SUCCESS);
}

// antiquant_group_size 及其后属性缺失: antiquant_group_size 走默认 0
TEST_F(WeightQuantBmmV2FallbackTest, AttrsFromAntiquantGroupSizeNull)
{
    FullTensors t;
    AttrsSpec attrs;
    attrs.attrCount = 2;
    EXPECT_EQ(RunExecute(attrs, t.AllPresent(), t.AllInputs(), &t.y), GRAPH_SUCCESS);
}

// dtype 及其后属性缺失(fallback 不读 dtype): inner_precise 走默认 0, 走 V2 分支
TEST_F(WeightQuantBmmV2FallbackTest, AttrsFromDtypeNull)
{
    FullTensors t;
    AttrsSpec attrs;
    attrs.attrCount = 3;
    EXPECT_EQ(RunExecute(attrs, t.AllPresent(), t.AllInputs(), &t.y), GRAPH_SUCCESS);
}

// inner_precise 缺失: 走默认 0, 走 V2 分支
TEST_F(WeightQuantBmmV2FallbackTest, AttrsFromInnerPreciseNull)
{
    FullTensors t;
    AttrsSpec attrs;
    attrs.attrCount = 4;
    EXPECT_EQ(RunExecute(attrs, t.AllPresent(), t.AllInputs(), &t.y), GRAPH_SUCCESS);
}
