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
#include <iostream>
#include <utility>
#include <vector>
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ge;

namespace {
constexpr int64_t K_DESC_SIZE = 128; // 与 op_kernel/arch35/update_tensor_desc_tiling_data.h 中 kDescSize 一致
constexpr int64_t K_MAX_RANK = 124;  // 与 op_kernel/arch35/update_tensor_desc_tiling_data.h 中 kMaxRank 一致

// 输出 shape 是 faker 内部出参存储（OutputShapes() 为兼容空实现），
// 推导结果必须从 context->GetOutputShape(0) 读取
struct InferShapeResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    std::vector<int64_t> outputDims;
};

InferShapeResult RunInferShapeWithAttr(const std::vector<int64_t>& shapeAttr,
                                       const std::vector<int64_t>& xDims = {2, 3})
{
    InferShapeResult result;
    gert::Shape dummyInputShape; // x 为占位输入，shape 不参与推导
    for (const int64_t dim : xDims) {
        dummyInputShape.AppendDim(dim);
    }
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&dummyInputShape})
                      .NodeAttrs({{"shape", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(shapeAttr)}})
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    result.status = gert::OpImplRegistry::GetInstance().GetOpImpl("UpdateTensorDesc")->infer_shape(context);
    if (result.status == ge::GRAPH_SUCCESS) {
        const gert::Shape* yShape = context->GetOutputShape(0);
        if (yShape != nullptr) {
            for (size_t i = 0; i < yShape->GetDimNum(); i++) {
                result.outputDims.push_back(yShape->GetDim(i));
            }
        }
    }
    return result;
}
} // namespace

class UpdateTensorDescInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "UpdateTensorDescInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "UpdateTensorDescInfershape TearDown" << std::endl; }
};

TEST_F(UpdateTensorDescInfershape, update_tensor_desc_infershape_test1)
{
    // 常规 rank3：y.shape = attr shape
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("UpdateTensorDesc"), nullptr);
    const auto r = RunInferShapeWithAttr({4, 8, 4});
    EXPECT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.outputDims, std::vector<int64_t>({4, 8, 4}));
}

TEST_F(UpdateTensorDescInfershape, update_tensor_desc_infershape_test2)
{
    // rank1 最小 numel 场景（numel = 128）
    const auto r = RunInferShapeWithAttr({128});
    EXPECT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.outputDims, std::vector<int64_t>({128}));
}

TEST_F(UpdateTensorDescInfershape, update_tensor_desc_infershape_test3)
{
    // rank5 场景
    const auto r = RunInferShapeWithAttr({2, 2, 2, 2, 8});
    EXPECT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.outputDims, std::vector<int64_t>({2, 2, 2, 2, 8}));
}

TEST_F(UpdateTensorDescInfershape, update_tensor_desc_infershape_test4)
{
    // 非法场景：rank(attr shape) = 0
    const auto r = RunInferShapeWithAttr({});
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}

TEST_F(UpdateTensorDescInfershape, update_tensor_desc_infershape_test5)
{
    // 非法场景：numel(attr shape) < 128
    const auto r = RunInferShapeWithAttr({4, 4});
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}

TEST_F(UpdateTensorDescInfershape, update_tensor_desc_infershape_test6)
{
    // 非法场景：存在负维度
    const auto r = RunInferShapeWithAttr({4, -8, 4});
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}

TEST_F(UpdateTensorDescInfershape, update_tensor_desc_infershape_test7)
{
    // 非法场景：rank(attr shape) > kMaxRank
    std::vector<int64_t> shapeAttr(static_cast<size_t>(K_MAX_RANK) + 1, 2);
    const auto r = RunInferShapeWithAttr(shapeAttr);
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}

TEST_F(UpdateTensorDescInfershape, update_tensor_desc_infershape_test8)
{
    // 动态 shape：x desc 含 -1（未知维，秩已知），x 为占位输入不影响推导，y 仍由 attr 决定
    const auto r = RunInferShapeWithAttr({4, 8, 4}, {-1, -1});
    EXPECT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.outputDims, std::vector<int64_t>({4, 8, 4}));
}

TEST_F(UpdateTensorDescInfershape, update_tensor_desc_infershape_test9)
{
    // 动态 shape：x desc 为 [-2]（未知秩），x 为占位输入不影响推导，y 仍由 attr 决定
    const auto r = RunInferShapeWithAttr({4, 8, 4}, {-2});
    EXPECT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.outputDims, std::vector<int64_t>({4, 8, 4}));
}

TEST_F(UpdateTensorDescInfershape, update_tensor_desc_infershape_test10)
{
    // 非法场景：attr shape 含 -1（未知维）——attr 为具体值契约（非负），fail-fast 拒绝
    const auto r = RunInferShapeWithAttr({-1, 128});
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}

TEST_F(UpdateTensorDescInfershape, update_tensor_desc_infershape_test11)
{
    // 非法场景：attr shape 为 [-2]（未知秩）——同上 fail-fast 拒绝
    const auto r = RunInferShapeWithAttr({-2});
    EXPECT_EQ(r.status, ge::GRAPH_FAILED);
}

TEST_F(UpdateTensorDescInfershape, update_tensor_desc_inferdatatype_test1)
{
    // y 的 dtype 恒为 DT_INT64（与 x 的 dtype 无关）
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("UpdateTensorDesc"), nullptr);
    auto inferDataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("UpdateTensorDesc")->infer_datatype;
    ASSERT_NE(inferDataTypeFunc, nullptr);

    ge::DataType inputXRef = ge::DT_FLOAT16;
    ge::DataType outputYRef = ge::DT_INT64;
    auto holder = gert::InferDataTypeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"shape",
                                   Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{4, 8, 4})}})
                      .InputDataTypes({&inputXRef})
                      .OutputDataTypes({&outputYRef})
                      .Build();
    auto context = holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(inferDataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT64);
}
