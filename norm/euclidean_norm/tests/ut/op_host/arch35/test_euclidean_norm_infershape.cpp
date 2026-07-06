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
