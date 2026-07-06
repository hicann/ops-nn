/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Pei Haobo<@xiaopei-1>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

#include "tiling_case_executor.h"
#include "experimental/pooling/average_pooling_grad/op_kernel/average_pooling_grad_tiling_data.h"

using namespace ge;

namespace {
constexpr uint64_t TILING_KEY_TILED = 1U;
constexpr uint64_t TILING_KEY_NO_OVERLAP = 2U;

struct AveragePoolingGradCompileInfo {
};

struct TilingCaseParam {
    ge::DataType dtype = ge::DT_FLOAT;
    std::array<int64_t, 4> origInputShape = {2, 1, 4, 6};
    std::vector<int64_t> gradOutputShape = {2, 1, 4, 6};
    std::vector<int64_t> gradInputShape = {2, 1, 4, 6};
    int64_t kernelH = 1;
    int64_t kernelW = 1;
    int64_t strideH = 1;
    int64_t strideW = 1;
    int64_t padTop = 0;
    int64_t padBottom = 0;
    int64_t padLeft = 0;
    int64_t padRight = 0;
    bool ceilMode = false;
    bool countIncludePad = true;
    int64_t divisorOverride = 0;
    uint64_t expectTilingKey = TILING_KEY_NO_OVERLAP;
    uint32_t expectTotalElements = 48;
    uint32_t expectTileTaskNum = 2;
};

static gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::Shape shape;
    shape.SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape[i] = dims[i];
    }
    gert::StorageShape storageShape;
    storageShape.MutableOriginShape() = shape;
    storageShape.MutableStorageShape() = shape;
    return storageShape;
}

static gert::TilingContextPara BuildTilingContext(const TilingCaseParam& param,
    AveragePoolingGradCompileInfo& compileInfo)
{
    const std::vector<int64_t> origInputShapeTensorShape = {4};
    auto origInputStorageShape = MakeStorageShape(origInputShapeTensorShape);
    auto gradOutputStorageShape = MakeStorageShape(param.gradOutputShape);
    auto gradInputStorageShape = MakeStorageShape(param.gradInputShape);
    return gert::TilingContextPara(
        "AveragePoolingGrad",
        {
            {origInputStorageShape, ge::DT_INT64, ge::FORMAT_ND, true,
                const_cast<int64_t*>(param.origInputShape.data())},
            {gradOutputStorageShape, param.dtype, ge::FORMAT_ND},
        },
        {
            {gradInputStorageShape, param.dtype, ge::FORMAT_ND},
        },
        {
            {"kernel_h", Ops::NN::AnyValue::CreateFrom<int64_t>(param.kernelH)},
            {"kernel_w", Ops::NN::AnyValue::CreateFrom<int64_t>(param.kernelW)},
            {"stride_h", Ops::NN::AnyValue::CreateFrom<int64_t>(param.strideH)},
            {"stride_w", Ops::NN::AnyValue::CreateFrom<int64_t>(param.strideW)},
            {"pad_top", Ops::NN::AnyValue::CreateFrom<int64_t>(param.padTop)},
            {"pad_bottom", Ops::NN::AnyValue::CreateFrom<int64_t>(param.padBottom)},
            {"pad_left", Ops::NN::AnyValue::CreateFrom<int64_t>(param.padLeft)},
            {"pad_right", Ops::NN::AnyValue::CreateFrom<int64_t>(param.padRight)},
            {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(param.ceilMode)},
            {"count_include_pad", Ops::NN::AnyValue::CreateFrom<bool>(param.countIncludePad)},
            {"divisor_override", Ops::NN::AnyValue::CreateFrom<int64_t>(param.divisorOverride)},
        },
        &compileInfo,
        64,
        262144,
        4096);
}

static void RunSuccessCase(const TilingCaseParam& param)
{
    AveragePoolingGradCompileInfo compileInfo;
    auto tilingContextPara = BuildTilingContext(param, compileInfo);
    TilingInfo tilingInfo;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, param.expectTilingKey);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
    EXPECT_EQ(tilingInfo.workspaceSizes[0], 0U);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(AveragePoolingGradTilingData));

    const auto* tilingData = reinterpret_cast<const AveragePoolingGradTilingData*>(tilingInfo.tilingData.get());
    EXPECT_EQ(tilingData->n, static_cast<uint32_t>(param.origInputShape[0]));
    EXPECT_EQ(tilingData->c, static_cast<uint32_t>(param.origInputShape[1]));
    EXPECT_EQ(tilingData->hIn, static_cast<uint32_t>(param.origInputShape[2]));
    EXPECT_EQ(tilingData->wIn, static_cast<uint32_t>(param.origInputShape[3]));
    EXPECT_EQ(tilingData->kernelH, static_cast<uint32_t>(param.kernelH));
    EXPECT_EQ(tilingData->kernelW, static_cast<uint32_t>(param.kernelW));
    EXPECT_EQ(tilingData->strideH, static_cast<uint32_t>(param.strideH));
    EXPECT_EQ(tilingData->strideW, static_cast<uint32_t>(param.strideW));
    EXPECT_EQ(tilingData->padTop, static_cast<uint32_t>(param.padTop));
    EXPECT_EQ(tilingData->padLeft, static_cast<uint32_t>(param.padLeft));
    EXPECT_EQ(tilingData->countIncludePad, param.countIncludePad ? 1U : 0U);
    EXPECT_EQ(tilingData->divisorOverride, static_cast<int32_t>(param.divisorOverride));
    EXPECT_EQ(tilingData->totalElements, param.expectTotalElements);
    EXPECT_EQ(tilingData->tileTaskNum, param.expectTileTaskNum);
    EXPECT_GT(tilingData->gradOutCacheDataNum, 0U);
}
} // namespace

class AveragePoolingGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AveragePoolingGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AveragePoolingGradTiling TearDown" << std::endl;
    }
};

TEST_F(AveragePoolingGradTiling, average_pooling_grad_float32_no_overlap_success)
{
    RunSuccessCase({});
}

TEST_F(AveragePoolingGradTiling, average_pooling_grad_float16_no_overlap_success)
{
    TilingCaseParam param;
    param.dtype = ge::DT_FLOAT16;
    RunSuccessCase(param);
}

TEST_F(AveragePoolingGradTiling, average_pooling_grad_overlap_success)
{
    TilingCaseParam param;
    param.origInputShape = {1, 1, 3, 3};
    param.gradOutputShape = {1, 1, 2, 2};
    param.gradInputShape = {1, 1, 3, 3};
    param.kernelH = 2;
    param.kernelW = 2;
    param.strideH = 1;
    param.strideW = 1;
    param.expectTilingKey = TILING_KEY_TILED;
    param.expectTotalElements = 9;
    param.expectTileTaskNum = 1;
    RunSuccessCase(param);
}

TEST_F(AveragePoolingGradTiling, average_pooling_grad_bfloat16_failed)
{
    TilingCaseParam param;
    param.dtype = ge::DT_BF16;
    AveragePoolingGradCompileInfo compileInfo;
    auto tilingContextPara = BuildTilingContext(param, compileInfo);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}
