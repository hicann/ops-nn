/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <vector>
#include <memory>

#include "log/log.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tensor.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"

namespace {
// 构造一个 1-D int32 const tensor，shape=(N,)，内容 = axes
std::unique_ptr<uint8_t[]> MakeAxesConstTensor(const std::vector<int32_t>& axes)
{
    size_t totalSize = 0;
    auto holder = gert::Tensor::CreateFollowing(static_cast<int64_t>(axes.size()), ge::DT_INT32, totalSize);
    auto* t = reinterpret_cast<gert::Tensor*>(holder.get());
    t->MutableStorageShape().SetDimNum(1);
    t->MutableStorageShape().SetDim(0, static_cast<int64_t>(axes.size()));
    t->MutableOriginShape().SetDimNum(1);
    t->MutableOriginShape().SetDim(0, static_cast<int64_t>(axes.size()));
    auto* dataPtr = reinterpret_cast<int32_t*>(holder.get() + sizeof(gert::Tensor));
    for (size_t i = 0; i < axes.size(); ++i) {
        dataPtr[i] = axes[i];
    }
    return holder;
}
} // namespace

class EuclideanNormInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "EuclideanNormInfershape SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "EuclideanNormInfershape TearDown" << std::endl; }
};

TEST_F(EuclideanNormInfershape, infershape_keepdims_false_reduce_last)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EuclideanNorm")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    // x shape = (4, 64)，axes = [1] → 输出 = (4,)
    gert::StorageShape xShape = {{4, 64}, {4, 64}};
    gert::StorageShape outShape;
    auto axesHolder = MakeAxesConstTensor({1});
    auto* axesTensor = reinterpret_cast<gert::Tensor*>(axesHolder.get());

    bool keepDims = false;
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("EuclideanNorm")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes(std::vector<void*>{static_cast<void*>(&xShape), static_cast<void*>(axesTensor)})
                      .OutputShapes({&outShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"keep_dims", Ops::NN::AnyValue::CreateFrom<bool>(keepDims)}})
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto* out = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDimNum(), 1U);
    EXPECT_EQ(out->GetDim(0), 4);
}

TEST_F(EuclideanNormInfershape, infershape_keepdims_true_reduce_last)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EuclideanNorm")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    // x shape = (4, 64)，axes = [-1]（归一化为 1），keep_dims=true → 输出 = (4, 1)
    gert::StorageShape xShape = {{4, 64}, {4, 64}};
    gert::StorageShape outShape;
    auto axesHolder = MakeAxesConstTensor({-1});
    auto* axesTensor = reinterpret_cast<gert::Tensor*>(axesHolder.get());

    bool keepDims = true;
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("EuclideanNorm")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes(std::vector<void*>{static_cast<void*>(&xShape), static_cast<void*>(axesTensor)})
                      .OutputShapes({&outShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"keep_dims", Ops::NN::AnyValue::CreateFrom<bool>(keepDims)}})
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto* out = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDimNum(), 2U);
    EXPECT_EQ(out->GetDim(0), 4);
    EXPECT_EQ(out->GetDim(1), 1);
}

TEST_F(EuclideanNormInfershape, infershape_invalid_duplicate_axes)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EuclideanNorm")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{4, 64}, {4, 64}};
    gert::StorageShape outShape;
    // axes = [1, 1] → 重复，应当失败
    auto axesHolder = MakeAxesConstTensor({1, 1});
    auto* axesTensor = reinterpret_cast<gert::Tensor*>(axesHolder.get());

    bool keepDims = false;
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("EuclideanNorm")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes(std::vector<void*>{static_cast<void*>(&xShape), static_cast<void*>(axesTensor)})
                      .OutputShapes({&outShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"keep_dims", Ops::NN::AnyValue::CreateFrom<bool>(keepDims)}})
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}
TEST_F(EuclideanNormInfershape, infershape_scalar_keepdims_true_stays_scalar)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EuclideanNorm")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    gert::StorageShape xShape = {{}, {}};
    gert::StorageShape outShape;
    auto axesHolder = MakeAxesConstTensor({});
    auto* axesTensor = reinterpret_cast<gert::Tensor*>(axesHolder.get());

    bool keepDims = true;
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("EuclideanNorm")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes(std::vector<void*>{static_cast<void*>(&xShape), static_cast<void*>(axesTensor)})
                      .OutputShapes({&outShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"keep_dims", Ops::NN::AnyValue::CreateFrom<bool>(keepDims)}})
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto* out = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDimNum(), 0U);
}

TEST_F(EuclideanNormInfershape, infershape_empty_axes_identity)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EuclideanNorm")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    // axes=[]（空张量）→ 无归约轴：y = x = (4, 64)，keep_dims 不生效
    gert::StorageShape xShape = {{4, 64}, {4, 64}};
    gert::StorageShape outShape;
    auto axesHolder = MakeAxesConstTensor({});
    auto* axesTensor = reinterpret_cast<gert::Tensor*>(axesHolder.get());

    bool keepDims = true;
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("EuclideanNorm")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes(std::vector<void*>{static_cast<void*>(&xShape), static_cast<void*>(axesTensor)})
                      .OutputShapes({&outShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"keep_dims", Ops::NN::AnyValue::CreateFrom<bool>(keepDims)}})
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto* out = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDimNum(), 2U);
    EXPECT_EQ(out->GetDim(0), 4);
    EXPECT_EQ(out->GetDim(1), 64);
}

TEST_F(EuclideanNormInfershape, infershape_empty_axes_zero_dim_identity)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EuclideanNorm")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    // axes=[] + x 含 0 维（issue #25 场景 c_s_021）→ y = x = (1,0,1,1,1,1)，0 元素输出
    // 修复前误判 full reduce → y 标量，与 0 字节 golden 输出错配导致 kernel 越界写
    gert::StorageShape xShape = {{1, 0, 1, 1, 1, 1}, {1, 0, 1, 1, 1, 1}};
    gert::StorageShape outShape;
    auto axesHolder = MakeAxesConstTensor({});
    auto* axesTensor = reinterpret_cast<gert::Tensor*>(axesHolder.get());

    bool keepDims = false;
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("EuclideanNorm")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes(std::vector<void*>{static_cast<void*>(&xShape), static_cast<void*>(axesTensor)})
                      .OutputShapes({&outShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"keep_dims", Ops::NN::AnyValue::CreateFrom<bool>(keepDims)}})
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto* out = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDimNum(), 6U);
    EXPECT_EQ(out->GetDim(0), 1);
    EXPECT_EQ(out->GetDim(1), 0);
    EXPECT_EQ(out->GetDim(2), 1);
    EXPECT_EQ(out->GetDim(3), 1);
    EXPECT_EQ(out->GetDim(4), 1);
    EXPECT_EQ(out->GetDim(5), 1);
}

// ────────────────────────────────────────────────────────────────────────────
// 用例 7（FR-09 看护）：pattern 轴数上限两门一致化（MAX_PATTERN_RANK=9）
//   rank-10 交替归约（预处理后 pattern=10 > 9）→ infershape 拒绝（修复前放行，两门不一致）；
//   rank-9 交替归约（pattern=9 达上限）→ 放行；
//   rank-10 全归约（合轴后 pattern=2）→ 放行（限制对象是交替段数，非原始 rank）。
// ────────────────────────────────────────────────────────────────────────────
TEST_F(EuclideanNormInfershape, infershape_rank10_alternating_rejected)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EuclideanNorm")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    const std::vector<int64_t> shape10 = {2, 3, 2, 3, 2, 3, 2, 3, 2, 3};
    const std::vector<int32_t> axesAlt10 = {1, 3, 5, 7, 9};
    {
        gert::StorageShape xShape = {{}, {}};
        xShape.MutableStorageShape().SetDimNum(shape10.size());
        xShape.MutableOriginShape().SetDimNum(shape10.size());
        for (size_t i = 0; i < shape10.size(); ++i) {
            xShape.MutableStorageShape().SetDim(i, shape10[i]);
            xShape.MutableOriginShape().SetDim(i, shape10[i]);
        }
        gert::StorageShape outShape;
        auto axesHolder = MakeAxesConstTensor(axesAlt10);
        auto* axesTensor = reinterpret_cast<gert::Tensor*>(axesHolder.get());
        auto holder = gert::InferShapeContextFaker()
                          .SetOpType("EuclideanNorm")
                          .NodeIoNum(2, 1)
                          .IrInstanceNum({1, 1})
                          .InputShapes(std::vector<void*>{static_cast<void*>(&xShape), static_cast<void*>(axesTensor)})
                          .OutputShapes({&outShape})
                          .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeAttrs({{"keep_dims", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                          .Build();
        EXPECT_NE(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    }
    {
        // rank-9 交替（pattern=9 达上限）→ 放行
        const std::vector<int64_t> shape9 = {2, 3, 2, 3, 2, 3, 2, 3, 2};
        const std::vector<int32_t> axesAlt9 = {1, 3, 5, 7};
        gert::StorageShape xShape = {{}, {}};
        xShape.MutableStorageShape().SetDimNum(shape9.size());
        xShape.MutableOriginShape().SetDimNum(shape9.size());
        for (size_t i = 0; i < shape9.size(); ++i) {
            xShape.MutableStorageShape().SetDim(i, shape9[i]);
            xShape.MutableOriginShape().SetDim(i, shape9[i]);
        }
        gert::StorageShape outShape;
        auto axesHolder = MakeAxesConstTensor(axesAlt9);
        auto* axesTensor = reinterpret_cast<gert::Tensor*>(axesHolder.get());
        auto holder = gert::InferShapeContextFaker()
                          .SetOpType("EuclideanNorm")
                          .NodeIoNum(2, 1)
                          .IrInstanceNum({1, 1})
                          .InputShapes(std::vector<void*>{static_cast<void*>(&xShape), static_cast<void*>(axesTensor)})
                          .OutputShapes({&outShape})
                          .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeAttrs({{"keep_dims", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                          .Build();
        EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    }
    {
        // rank-10 全归约（合轴后 pattern=2）→ 放行
        const std::vector<int64_t> shape10b = {2, 3, 2, 3, 2, 3, 2, 3, 2, 3};
        const std::vector<int32_t> axesFull10 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        gert::StorageShape xShape = {{}, {}};
        xShape.MutableStorageShape().SetDimNum(shape10b.size());
        xShape.MutableOriginShape().SetDimNum(shape10b.size());
        for (size_t i = 0; i < shape10b.size(); ++i) {
            xShape.MutableStorageShape().SetDim(i, shape10b[i]);
            xShape.MutableOriginShape().SetDim(i, shape10b[i]);
        }
        gert::StorageShape outShape;
        auto axesHolder = MakeAxesConstTensor(axesFull10);
        auto* axesTensor = reinterpret_cast<gert::Tensor*>(axesHolder.get());
        auto holder = gert::InferShapeContextFaker()
                          .SetOpType("EuclideanNorm")
                          .NodeIoNum(2, 1)
                          .IrInstanceNum({1, 1})
                          .InputShapes(std::vector<void*>{static_cast<void*>(&xShape), static_cast<void*>(axesTensor)})
                          .OutputShapes({&outShape})
                          .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeAttrs({{"keep_dims", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                          .Build();
        EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// 用例 8（FR-09 看护，T13）：CountPatternAxesAfterPreprocess 全 1 退化两子分支
//   x=(1,1) axes=[]：① 全 1 轴去空后保留 A=1 占位（DropSizeOneAxes 尽删后的占位子分支）；
//   ② 纯 A 全退化 [1] 末尾补 R=1（无 R 轴增广的 [1] 特例子分支）→ pattern=[A=1,R=1]
//   （axisNum=2 在 [kMinAxisNum, MAX_PATTERN_RANK] 域内放行）→ y = (1,1) 恒等。
// ────────────────────────────────────────────────────────────────────────────
TEST_F(EuclideanNormInfershape, infershape_all_one_empty_axes_pattern_count)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EuclideanNorm")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    // x shape = (1,1)（全 1），axes = []（空张量）→ 无归约轴：y = x = (1,1)
    gert::StorageShape xShape = {{1, 1}, {1, 1}};
    gert::StorageShape outShape;
    auto axesHolder = MakeAxesConstTensor({});
    auto* axesTensor = reinterpret_cast<gert::Tensor*>(axesHolder.get());

    bool keepDims = true;
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("EuclideanNorm")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes(std::vector<void*>{static_cast<void*>(&xShape), static_cast<void*>(axesTensor)})
                      .OutputShapes({&outShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"keep_dims", Ops::NN::AnyValue::CreateFrom<bool>(keepDims)}})
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto* out = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDimNum(), 2U);
    EXPECT_EQ(out->GetDim(0), 1);
    EXPECT_EQ(out->GetDim(1), 1);
}

// ────────────────────────────────────────────────────────────────────────────
// 用例 9（C04 看护，检视修复轮）：空 tensor 高交替段数与 tiling 同判
//   x=(0,2,0,2,0,2,0,2,0,2)+axes={1,3,5,7,9}（A 轴全 0 → tiling EMPTY_A 早退路径，
//   pattern 轴数门不适用）：修复前 FR-09 门无条件生效（去 1/合轴后 pattern=10>9）
//   误拒（可用性回退——修前该输入端到端正确）；修复后镜像 tiling 空 tensor 旁路
//   跳过轴数门 → GRAPH_SUCCESS，y=(0,0,0,0,0)（归约掉全部值 2 的轴，保留 5 根
//   0 长 A 轴 → 0 元素输出，与 tiling 同判）。
// ────────────────────────────────────────────────────────────────────────────
TEST_F(EuclideanNormInfershape, infershape_empty_tensor_high_alternation_accepted)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EuclideanNorm")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    const std::vector<int64_t> xDims = {0, 2, 0, 2, 0, 2, 0, 2, 0, 2};
    const std::vector<int32_t> axesAlt = {1, 3, 5, 7, 9};
    gert::StorageShape xShape = {{}, {}};
    xShape.MutableStorageShape().SetDimNum(xDims.size());
    xShape.MutableOriginShape().SetDimNum(xDims.size());
    for (size_t i = 0; i < xDims.size(); ++i) {
        xShape.MutableStorageShape().SetDim(i, xDims[i]);
        xShape.MutableOriginShape().SetDim(i, xDims[i]);
    }
    gert::StorageShape outShape;
    auto axesHolder = MakeAxesConstTensor(axesAlt);
    auto* axesTensor = reinterpret_cast<gert::Tensor*>(axesHolder.get());
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("EuclideanNorm")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes(std::vector<void*>{static_cast<void*>(&xShape), static_cast<void*>(axesTensor)})
                      .OutputShapes({&outShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"keep_dims", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto* out = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDimNum(), 5U);
    EXPECT_EQ(out->GetDim(0), 0);
    EXPECT_EQ(out->GetDim(1), 0);
    EXPECT_EQ(out->GetDim(2), 0);
    EXPECT_EQ(out->GetDim(3), 0);
    EXPECT_EQ(out->GetDim(4), 0);
}

// ────────────────────────────────────────────────────────────────────────────
// 用例 10（C04 看护，检视修复轮）：-1 未知维跳过轴数门（实际段数运行时定）
//   x=(-1,2,-1,2,-1,2,-1,2,-1,2)+axes={1,3,5,7,9}（含未知维）：编译期按占位计数
//   pattern=10>9，但实际段数由运行时形状决定、由 tiling 运行时门兜底——
//   修复后跳过轴数门 → GRAPH_SUCCESS，y=(-1,-1,-1,-1,-1)（归约掉全部值 2 的
//   已知轴，保留 5 根未知维透传）。
// ────────────────────────────────────────────────────────────────────────────
TEST_F(EuclideanNormInfershape, infershape_unknown_dim_high_alternation_skips_gate)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EuclideanNorm")->infer_shape;
    ASSERT_NE(inferShapeFunc, nullptr);

    const std::vector<int64_t> xDims = {-1, 2, -1, 2, -1, 2, -1, 2, -1, 2};
    const std::vector<int32_t> axesAlt = {1, 3, 5, 7, 9};
    gert::StorageShape xShape = {{}, {}};
    xShape.MutableStorageShape().SetDimNum(xDims.size());
    xShape.MutableOriginShape().SetDimNum(xDims.size());
    for (size_t i = 0; i < xDims.size(); ++i) {
        xShape.MutableStorageShape().SetDim(i, xDims[i]);
        xShape.MutableOriginShape().SetDim(i, xDims[i]);
    }
    gert::StorageShape outShape;
    auto axesHolder = MakeAxesConstTensor(axesAlt);
    auto* axesTensor = reinterpret_cast<gert::Tensor*>(axesHolder.get());
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("EuclideanNorm")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes(std::vector<void*>{static_cast<void*>(&xShape), static_cast<void*>(axesTensor)})
                      .OutputShapes({&outShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"keep_dims", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .Build();

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto* out = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_NE(out, nullptr);
    EXPECT_EQ(out->GetDimNum(), 5U);
    EXPECT_EQ(out->GetDim(0), -1);
    EXPECT_EQ(out->GetDim(1), -1);
    EXPECT_EQ(out->GetDim(2), -1);
    EXPECT_EQ(out->GetDim(3), -1);
    EXPECT_EQ(out->GetDim(4), -1);
}
