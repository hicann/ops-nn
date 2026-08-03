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
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_kernel/nll_loss_grad_tiling_data.h"

namespace NllLossGradUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "NllLossGrad";

struct NllLossGradTestParam {
    std::string caseName;
    std::initializer_list<int64_t> xShape;
    ge::DataType xDtype;
    ge::Format xFormat;
    std::initializer_list<int64_t> y_gradShape;
    ge::DataType y_gradDtype;
    ge::Format y_gradFormat;
    std::initializer_list<int64_t> targetShape;
    ge::DataType targetDtype;
    ge::Format targetFormat;
    std::initializer_list<int64_t> weightShape;
    ge::DataType weightDtype;
    ge::Format weightFormat;
    std::initializer_list<int64_t> total_weightShape;
    ge::DataType total_weightDtype;
    ge::Format total_weightFormat;
    std::initializer_list<int64_t> x_gradShape;
    ge::DataType x_gradDtype;
    ge::Format x_gradFormat;
    std::string socVersion;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t maxAIVNum;
    uint64_t ubSize;
    uint64_t tilingDataMaxSize;
    std::string reduction;
    int64_t ignoreIndex;
};

static NllLossGradTestParam testCases[] = {
    // 9 种 dtype 组合，NormalWeight，验证 schMode 编码
    // float32/int32 -> key 0
    {"f32_i32",
     {4, 7},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {1},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {4},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {7},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {1},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {4, 7},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     "Ascend910B",
     ge::GRAPH_SUCCESS,
     0UL,
     EMPTY_EXPECT_TILING_DATA,
     {0},
     64,
     262144,
     4096,
     "mean",
     -100},
    // bf16/int32 -> key 1
    {"bf16_i32",
     {5, 9},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {5},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {5},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {9},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {1},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {5, 9},
     ge::DT_BF16,
     ge::FORMAT_ND,
     "Ascend910B",
     ge::GRAPH_SUCCESS,
     1UL,
     EMPTY_EXPECT_TILING_DATA,
     {0},
     64,
     262144,
     4096,
     "none",
     -100},
    // float32/int64 -> key 2
    {"f32_i64",
     {6, 11},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {1},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {6},
     ge::DT_INT64,
     ge::FORMAT_ND,
     {11},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {1},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {6, 11},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     "Ascend910B",
     ge::GRAPH_SUCCESS,
     2UL,
     EMPTY_EXPECT_TILING_DATA,
     {0},
     64,
     262144,
     4096,
     "sum",
     -100},
    // bf16/int64 -> key 3
    {"bf16_i64",
     {3, 13},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {1},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {3},
     ge::DT_INT64,
     ge::FORMAT_ND,
     {13},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {1},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {3, 13},
     ge::DT_BF16,
     ge::FORMAT_ND,
     "Ascend910B",
     ge::GRAPH_SUCCESS,
     3UL,
     EMPTY_EXPECT_TILING_DATA,
     {0},
     64,
     262144,
     4096,
     "mean",
     -100},
    // float32/uint8 -> key 4
    {"f32_u8",
     {8, 50},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {8},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {8},
     ge::DT_UINT8,
     ge::FORMAT_ND,
     {50},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {1},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {8, 50},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     "Ascend910B",
     ge::GRAPH_SUCCESS,
     4UL,
     EMPTY_EXPECT_TILING_DATA,
     {0},
     64,
     262144,
     4096,
     "none",
     -100},
    // bf16/uint8 -> key 5
    {"bf16_u8",
     {4, 30},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {1},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {4},
     ge::DT_UINT8,
     ge::FORMAT_ND,
     {30},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {1},
     ge::DT_BF16,
     ge::FORMAT_ND,
     {4, 30},
     ge::DT_BF16,
     ge::FORMAT_ND,
     "Ascend910B",
     ge::GRAPH_SUCCESS,
     5UL,
     EMPTY_EXPECT_TILING_DATA,
     {0},
     64,
     262144,
     4096,
     "mean",
     -100},
    // float16/int32 -> key 6
    {"f16_i32",
     {7, 13},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {1},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {7},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {13},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {1},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {7, 13},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     "Ascend910B",
     ge::GRAPH_SUCCESS,
     6UL,
     EMPTY_EXPECT_TILING_DATA,
     {0},
     64,
     262144,
     4096,
     "sum",
     -100},
    // float16/int64 -> key 7
    {"f16_i64",
     {32, 1},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {32},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {32},
     ge::DT_INT64,
     ge::FORMAT_ND,
     {1},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {1},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {32, 1},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     "Ascend910B",
     ge::GRAPH_SUCCESS,
     7UL,
     EMPTY_EXPECT_TILING_DATA,
     {0},
     64,
     262144,
     4096,
     "none",
     -100},
    // float16/uint8 -> key 8
    {"f16_u8",
     {3, 20},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {1},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {3},
     ge::DT_UINT8,
     ge::FORMAT_ND,
     {20},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {1},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     {3, 20},
     ge::DT_FLOAT16,
     ge::FORMAT_ND,
     "Ascend910B",
     ge::GRAPH_SUCCESS,
     8UL,
     EMPTY_EXPECT_TILING_DATA,
     {0},
     64,
     262144,
     4096,
     "mean",
     -100},
    // fp32 large C, BigWeight -> key 0 (float32/int32)
    {"f32_bigweight",
     {8, 200000},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {1},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {8},
     ge::DT_INT32,
     ge::FORMAT_ND,
     {200000},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {1},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     {8, 200000},
     ge::DT_FLOAT,
     ge::FORMAT_ND,
     "Ascend910B",
     ge::GRAPH_SUCCESS,
     0UL,
     EMPTY_EXPECT_TILING_DATA,
     {0},
     64,
     262144,
     4096,
     "mean",
     -100},
};

class NllLossGradTilingTest : public testing::TestWithParam<NllLossGradTestParam> {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

struct NllLossGradCompileInfo {
} compileInfo;

static void TestOneParamCase(const NllLossGradTestParam& param)
{
    gert::StorageShape xShape = {param.xShape, param.xShape};
    gert::StorageShape y_gradShape = {param.y_gradShape, param.y_gradShape};
    gert::StorageShape targetShape = {param.targetShape, param.targetShape};
    gert::StorageShape weightShape = {param.weightShape, param.weightShape};
    gert::StorageShape total_weightShape = {param.total_weightShape, param.total_weightShape};
    gert::StorageShape x_gradShape = {param.x_gradShape, param.x_gradShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{xShape, param.xDtype, param.xFormat},
         {y_gradShape, param.y_gradDtype, param.y_gradFormat},
         {targetShape, param.targetDtype, param.targetFormat},
         {weightShape, param.weightDtype, param.weightFormat},
         {total_weightShape, param.total_weightDtype, param.total_weightFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{x_gradShape, param.x_gradDtype, param.x_gradFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;
    attrs_.push_back(
        gert::TilingContextPara::OpAttr("reduction", Ops::NN::AnyValue::CreateFrom<std::string>(param.reduction)));
    attrs_.push_back(
        gert::TilingContextPara::OpAttr("ignore_index", Ops::NN::AnyValue::CreateFrom<int64_t>(param.ignoreIndex)));
    gert::TilingContextPara tilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_, &compileInfo,
                                              param.maxAIVNum, param.ubSize, param.tilingDataMaxSize);
    ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey, param.expectTilingData,
                    param.expectWorkspaces);
}

TEST_P(NllLossGradTilingTest, tiling_test)
{
    const NllLossGradTestParam& param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(NllLossGradTilingTests, NllLossGradTilingTest, testing::ValuesIn(testCases));

} // namespace NllLossGradUT
