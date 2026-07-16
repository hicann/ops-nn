/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cmath>
#include <gtest/gtest.h>
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "../../../op_kernel/mse_loss_tiling_data.h"

namespace MseLossUT {
using namespace ge;
using namespace gert;

static const std::string OP_NAME = "MseLoss";

struct MseLossTestParam {
    std::string caseName;
    std::initializer_list<int64_t> predictShape;
    ge::DataType dtype;
    std::initializer_list<int64_t> yShape;
    std::string reductionAttr;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::vector<int64_t> expectWorkspaces;
    uint64_t maxAIVNum;
    uint64_t ubSize;
    int64_t expectTotalNum;
    int64_t expectBlockFactor;
    int64_t expectUbFactor;
    int64_t expectReduction;
    int64_t expectBlockNum;
    int64_t expectWorkspaceFloatsPerCore;
};

static constexpr int64_t ASCENDC_TOOLS_WORKSPACE = 16 * 1024 * 1024;

static MseLossTestParam testCases[] = {
    {"mean_fp16_small",
     {3, 5},
     ge::DT_FLOAT16,
     {1},
     "mean",
     ge::GRAPH_SUCCESS,
     0UL,
     {0},
     64,
     262144,
     15,
     15,
     64,
     2,
     1,
     8},
    {"mean_fp16_unaligned_multicore",
     {1025},
     ge::DT_FLOAT16,
     {1},
     "mean",
     ge::GRAPH_SUCCESS,
     0UL,
     {0},
     64,
     262144,
     1025,
     1025,
     1088,
     2,
     1,
     8},
    {"sum_fp32_large",
     {4096},
     ge::DT_FLOAT,
     {1},
     "sum",
     ge::GRAPH_SUCCESS,
     1UL,
     {0},
     64,
     262144,
     4096,
     4096,
     4096,
     1,
     1,
     8},
    {"none_bf16_large",
     {4096},
     ge::DT_BF16,
     {4096},
     "none",
     ge::GRAPH_SUCCESS,
     2UL,
     {0},
     64,
     262144,
     4096,
     64,
     4096,
     0,
     64,
     8},
    {"mean_fp32_empty", {0}, ge::DT_FLOAT, {1}, "mean", ge::GRAPH_SUCCESS, 1UL, {0}, 64, 262144, 0, 1, 64, 2, 1, 8},
    {"invalid_shape", {2, 3}, ge::DT_FLOAT16, {1}, "none", ge::GRAPH_FAILED, 0UL, {}, 64, 262144, 0, 0, 0, 0, 0, 0},
    {"invalid_ub_size", {65}, ge::DT_FLOAT16, {1}, "mean", ge::GRAPH_FAILED, 0UL, {}, 64, 16384, 0, 0, 0, 0, 0, 0},
};

class MseLossTilingTest : public testing::TestWithParam<MseLossTestParam> {};

struct MseLossCompileInfo {
} compileInfo;

static gert::TilingContextPara BuildContextParam(const MseLossTestParam& param)
{
    gert::StorageShape predictShape = {param.predictShape, param.predictShape};
    gert::StorageShape labelShape = {param.predictShape, param.predictShape};
    gert::StorageShape yShape = {param.yShape, param.yShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc(
        {{predictShape, param.dtype, ge::FORMAT_ND}, {labelShape, param.dtype, ge::FORMAT_ND}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc({{yShape, param.dtype, ge::FORMAT_ND}});
    std::vector<gert::TilingContextPara::OpAttr> attrs;
    attrs.push_back(
        gert::TilingContextPara::OpAttr("reduction", Ops::NN::AnyValue::CreateFrom<std::string>(param.reductionAttr)));
    return gert::TilingContextPara(OP_NAME, inputTensorDesc, outputTensorDesc, attrs, &compileInfo, param.maxAIVNum,
                                   param.ubSize, 4096);
}

TEST_P(MseLossTilingTest, tiling_test)
{
    const MseLossTestParam& param = GetParam();
    auto tilingContextPara = BuildContextParam(param);
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(tilingContextPara, tilingInfo);

    if (param.status == ge::GRAPH_FAILED) {
        EXPECT_FALSE(ok);
        return;
    }

    ASSERT_TRUE(ok);
    EXPECT_EQ(static_cast<uint64_t>(tilingInfo.tilingKey), param.expectTilingKey);
    ASSERT_EQ(tilingInfo.workspaceSizes, param.expectWorkspaces);
    EXPECT_EQ(tilingInfo.blockNum, static_cast<size_t>(param.expectBlockNum));
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(MseLossTilingData));

    auto* tilingData = reinterpret_cast<MseLossTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->totalNum, param.expectTotalNum);
    EXPECT_EQ(tilingData->blockFactor, param.expectBlockFactor);
    EXPECT_EQ(tilingData->ubFactor, param.expectUbFactor);
    EXPECT_EQ(tilingData->reduction, param.expectReduction);
    EXPECT_EQ(tilingData->blockNum, param.expectBlockNum);
    EXPECT_EQ(tilingData->workspaceFloatsPerCore, param.expectWorkspaceFloatsPerCore);
    if (param.expectReduction == 2 && param.expectTotalNum == 0) {
        EXPECT_TRUE(std::isnan(tilingData->meanScale));
    } else if (param.expectReduction == 2) {
        EXPECT_NEAR(tilingData->meanScale, 1.0f / static_cast<float>(param.expectTotalNum), 1e-7f);
    }
}

INSTANTIATE_TEST_SUITE_P(MseLossTilingTests, MseLossTilingTest, testing::ValuesIn(testCases));

} // namespace MseLossUT
