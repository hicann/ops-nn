/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file test_quantize_tiling.cpp
 * \brief Quantize host tiling UT: tiling key + folded fields (per-tensor / per-channel / empty / zp).
 */
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "../../../../../../tests/ut/common/any_value.h"

namespace {
constexpr uint64_t KEY_PER_CHANNEL = 0;
constexpr uint64_t KEY_PER_TENSOR = 1;

// Layout mirror of optiling::QuantizeCompileInfo (op_host/quantize_tiling.h).
struct QuantizeCompileInfoUt {
    int64_t coreNum = 0;
    uint64_t ubSize = 0;
};

// Layout mirror of optiling::QuantizeTilingData (op_host/quantize_tiling.h).
struct QuantizeTilingDataUt {
    uint32_t numCore;
    uint32_t hasZeroPoint;
    int64_t channelNum;
    int64_t rowLen;
    int64_t totalRows;
    int64_t totalElems;
    int64_t blockFactor;
    int64_t blockTailFactor;
    int64_t baseLen;
};

struct TilingResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    uint64_t key = 0;
    uint32_t blockDim = 0;
    QuantizeTilingDataUt data{};
};

TilingResult RunQuantizeTiling(gert::StorageShape x, gert::StorageShape scales, bool hasZp, gert::StorageShape zp,
                               ge::DataType xDt, ge::DataType yDt, int64_t axis, int64_t coreNum, uint64_t ubSize)
{
    TilingResult result;
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("Quantize");
    EXPECT_NE(opImpl, nullptr);
    if (opImpl == nullptr) {
        return result;
    }
    auto tilingFunc = opImpl->tiling;
    EXPECT_NE(tilingFunc, nullptr);
    if (tilingFunc == nullptr) {
        return result;
    }

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    QuantizeCompileInfoUt compileInfo;
    compileInfo.coreNum = coreNum;
    compileInfo.ubSize = ubSize;

    gert::StorageShape yShape = x;
    gert::StorageShape* zpPtr = hasZp ? &zp : nullptr;

    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());

    auto holder = gert::TilingContextFaker()
                      .SetOpType("Quantize")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x, &scales, zpPtr})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, xDt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, yDt, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<std::string>("torch.qint8")},
                                  {"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(axis)}})
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();

    auto context = holder.GetContext<gert::TilingContext>();
    result.status = tilingFunc(context);
    if (result.status == ge::GRAPH_SUCCESS) {
        result.key = context->GetTilingKey();
        result.blockDim = context->GetBlockDim();
        std::memcpy(&result.data, context->GetRawTilingData()->GetData(), sizeof(QuantizeTilingDataUt));
    }
    return result;
}
} // namespace

class QuantizeTilingTest : public testing::Test {};

TEST_F(QuantizeTilingTest, per_tensor_2d_no_zp_key1)
{
    gert::StorageShape x = {{8, 16}, {8, 16}};
    gert::StorageShape scales = {{1}, {1}};
    gert::StorageShape zp = {{1}, {1}};
    auto r = RunQuantizeTiling(x, scales, false, zp, ge::DT_FLOAT, ge::DT_INT8, 1, 8, 196608);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.key, KEY_PER_TENSOR);
    EXPECT_EQ(r.data.hasZeroPoint, 0U);
    EXPECT_EQ(r.data.channelNum, 1);
    EXPECT_EQ(r.data.rowLen, 1);
    EXPECT_EQ(r.data.totalRows, 1);
    EXPECT_EQ(r.data.totalElems, 128);
    EXPECT_GE(r.data.numCore, 1U);
    EXPECT_LE(r.data.numCore, 8U);
    EXPECT_GE(r.data.blockFactor, 1);
    EXPECT_GE(r.data.blockFactor * static_cast<int64_t>(r.data.numCore), r.data.totalElems);
    EXPECT_GE(r.data.blockTailFactor, 1);
    EXPECT_LE(r.data.blockTailFactor, r.data.blockFactor);
    EXPECT_GE(r.data.baseLen, 1);
}

TEST_F(QuantizeTilingTest, per_tensor_1d_with_zp_key1)
{
    gert::StorageShape x = {{32}, {32}};
    gert::StorageShape scales = {{1}, {1}};
    gert::StorageShape zp = {{1}, {1}};
    auto r = RunQuantizeTiling(x, scales, true, zp, ge::DT_FLOAT, ge::DT_INT8, 1, 8, 196608);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.key, KEY_PER_TENSOR);
    EXPECT_EQ(r.data.hasZeroPoint, 1U);
    EXPECT_EQ(r.data.channelNum, 1);
    EXPECT_EQ(r.data.totalElems, 32);
}

TEST_F(QuantizeTilingTest, per_channel_3d_axis1_key0)
{
    gert::StorageShape x = {{2, 4, 8}, {2, 4, 8}};
    gert::StorageShape scales = {{4}, {4}};
    gert::StorageShape zp = {{4}, {4}};
    auto r = RunQuantizeTiling(x, scales, true, zp, ge::DT_FLOAT16, ge::DT_INT8, 1, 8, 196608);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.key, KEY_PER_CHANNEL);
    EXPECT_EQ(r.data.hasZeroPoint, 1U);
    EXPECT_EQ(r.data.channelNum, 4);
    EXPECT_EQ(r.data.rowLen, 8);
    EXPECT_EQ(r.data.totalRows, 8);
    EXPECT_EQ(r.data.totalElems, 64);
    EXPECT_GE(r.data.numCore, 1U);
    EXPECT_GE(r.data.blockFactor, 1);
    EXPECT_GE(r.data.blockFactor * static_cast<int64_t>(r.data.numCore), r.data.totalRows);
    EXPECT_GE(r.data.blockTailFactor, 1);
    EXPECT_LE(r.data.blockTailFactor, r.data.blockFactor);
}

TEST_F(QuantizeTilingTest, per_channel_2d_axis_neg1_rowlen1_key0)
{
    gert::StorageShape x = {{3, 5}, {3, 5}};
    gert::StorageShape scales = {{5}, {5}};
    gert::StorageShape zp = {{5}, {5}};
    auto r = RunQuantizeTiling(x, scales, false, zp, ge::DT_FLOAT, ge::DT_INT8, -1, 8, 196608);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.key, KEY_PER_CHANNEL);
    EXPECT_EQ(r.data.channelNum, 5);
    EXPECT_EQ(r.data.rowLen, 1);
    EXPECT_EQ(r.data.totalRows, 15);
    EXPECT_EQ(r.data.totalElems, 15);
}

TEST_F(QuantizeTilingTest, empty_tensor_noop_key1)
{
    gert::StorageShape x = {{0, 4}, {0, 4}};
    gert::StorageShape scales = {{1}, {1}};
    gert::StorageShape zp = {{1}, {1}};
    auto r = RunQuantizeTiling(x, scales, false, zp, ge::DT_FLOAT, ge::DT_INT8, 1, 8, 196608);
    ASSERT_EQ(r.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(r.key, KEY_PER_TENSOR);
    EXPECT_EQ(r.data.totalElems, 0);
    EXPECT_EQ(r.data.numCore, 1U);
    EXPECT_EQ(r.data.blockFactor, 0);
    EXPECT_EQ(r.data.blockTailFactor, 0);
}
