/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_huber_loss_tiling.cpp
 * \brief Tiling unit tests, at the registered entry point.
 *
 * Covers what only the framework mediates: which attribute slot each value
 * comes from, the shape and dtype checks, the published tiling key, and the
 * workspace request. The tiling arithmetic itself is pure (see
 * op_host/huber_loss_tiling_calc.h).
 */
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <vector>
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"
#include "../../../op_kernel/huber_loss_tiling_data.h"

namespace {

constexpr uint64_t kCoreNum = 40;
constexpr uint64_t kUbSize = 196608;

struct HuberLossCompileInfo {
} g_compileInfo;

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    shape.MutableOriginShape().SetDimNum(0);
    shape.MutableStorageShape().SetDimNum(0);
    for (const int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

struct Case {
    std::vector<int64_t> inputShape;
    std::vector<int64_t> targetShape;
    std::vector<int64_t> outShape;
    ge::DataType inputDtype = ge::DT_FLOAT;
    ge::DataType targetDtype = ge::DT_FLOAT;
    ge::DataType outDtype = ge::DT_FLOAT;
    int64_t reduction = HUBER_LOSS_REDUCE_NONE;
    float delta = 1.0f;
    uint64_t coreNum = kCoreNum;
    uint64_t ubSize = kUbSize;
};

gert::TilingContextPara BuildContext(const Case& c, gert::StorageShape& in, gert::StorageShape& tgt,
                                     gert::StorageShape& out)
{
    in = MakeStorageShape(c.inputShape);
    tgt = MakeStorageShape(c.targetShape);
    out = MakeStorageShape(c.outShape);
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {{in, c.inputDtype, ge::FORMAT_ND},
                                                                      {tgt, c.targetDtype, ge::FORMAT_ND}};
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {{out, c.outDtype, ge::FORMAT_ND}};
    // Attribute ORDER is the contract: reduction is index 0, delta is index 1,
    // matching the OpDef. Tiling reads GetInt(0) and GetFloat(1) positionally.
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"reduction", Ops::NN::AnyValue::CreateFrom<int64_t>(c.reduction)},
        {"delta", Ops::NN::AnyValue::CreateFrom<float>(c.delta)}};
    return gert::TilingContextPara("HuberLoss", inputs, outputs, attrs, &g_compileInfo, c.coreNum, c.ubSize, 4096);
}

bool RunTiling(const Case& c, TilingInfo& info)
{
    gert::StorageShape in;
    gert::StorageShape tgt;
    gert::StorageShape out;
    auto context = BuildContext(c, in, tgt, out);
    return ExecuteTiling(context, info);
}

const HuberLossTilingData* Data(const TilingInfo& info)
{
    return reinterpret_cast<const HuberLossTilingData*>(info.tilingData.get());
}

uint64_t Numel(const std::vector<int64_t>& shape)
{
    uint64_t n = 1;
    for (const int64_t d : shape) {
        n *= static_cast<uint64_t>(d);
    }
    return n;
}

// ===========================================================================
// The output tensor is validated in tiling, not infershape: on the aclnn path
// the caller supplies the output directly and infershape never sees it. These
// cases pin the rejection of out shapes and dtypes that mismatch reduction.
// ===========================================================================

TEST(HuberLossTilingTest, RejectsOutputSmallerThanInputForNone)
{
    TilingInfo info;
    EXPECT_FALSE(RunTiling({{64}, {64}, {32}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, HUBER_LOSS_REDUCE_NONE}, info));
}

TEST(HuberLossTilingTest, RejectsScalarOutputForNone)
{
    TilingInfo info;
    EXPECT_FALSE(RunTiling({{64}, {64}, {}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, HUBER_LOSS_REDUCE_NONE}, info));
}

TEST(HuberLossTilingTest, RejectsFullShapedOutputForReduce)
{
    TilingInfo info;
    EXPECT_FALSE(RunTiling({{64}, {64}, {64}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, HUBER_LOSS_REDUCE_MEAN}, info));
    EXPECT_FALSE(RunTiling({{64}, {64}, {64}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, HUBER_LOSS_REDUCE_SUM}, info));
}

TEST(HuberLossTilingTest, RejectsMismatchedOutputDtype)
{
    TilingInfo info;
    EXPECT_FALSE(
        RunTiling({{64}, {64}, {64}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, HUBER_LOSS_REDUCE_NONE}, info));
}

TEST(HuberLossTilingTest, AcceptsScalarOutputAtRankZeroAndRankOne)
{
    // Rank 0 is what infershape produces and what the scalar contract calls
    // for. Rank 1 {1} is accepted too: it is equally safe and callers commonly
    // spell a scalar that way.
    TilingInfo info;
    EXPECT_TRUE(RunTiling({{64}, {64}, {}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, HUBER_LOSS_REDUCE_MEAN}, info));
    EXPECT_TRUE(RunTiling({{64}, {64}, {1}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, HUBER_LOSS_REDUCE_MEAN}, info));
}

// ===========================================================================
// Attribute wiring. Both values are addressed positionally, so a swap between
// them is a silent corruption that the arithmetic tests cannot see -- they are
// handed reduction and delta as function arguments.
// ===========================================================================

TEST(HuberLossTilingTest, ReadsBothAttributesFromTheirOwnSlots)
{
    TilingInfo info;
    // sum with a non-default delta: if the two slots were swapped, either the
    // schedule mode or the delta would come out wrong.
    ASSERT_TRUE(
        RunTiling({{1024}, {1024}, {}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, HUBER_LOSS_REDUCE_SUM, 0.5f}, info));
    ASSERT_EQ(info.tilingDataSize, sizeof(HuberLossTilingData));
    EXPECT_EQ(info.tilingKey, HUBER_LOSS_SCH_MODE_REDUCE);
    EXPECT_FLOAT_EQ(Data(info)->delta, 0.5f);
    EXPECT_EQ(Data(info)->reduction, HUBER_LOSS_REDUCE_SUM);
}

TEST(HuberLossTilingTest, DivisorEncodesTheReductionMode)
{
    TilingInfo info;
    // mean divides by the element count, sum divides by one. The kernel has no
    // branch on the mode in its epilogue; it divides unconditionally, so this
    // field is the only thing separating the two.
    ASSERT_TRUE(
        RunTiling({{1000}, {1000}, {}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, HUBER_LOSS_REDUCE_MEAN}, info));
    EXPECT_FLOAT_EQ(Data(info)->divisor, 1000.0f);

    ASSERT_TRUE(RunTiling({{1000}, {1000}, {}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, HUBER_LOSS_REDUCE_SUM}, info));
    EXPECT_FLOAT_EQ(Data(info)->divisor, 1.0f);
}

// ===========================================================================
// Tiling key and workspace: the two things the kernel build and the runtime
// read out of tiling, neither of which the arithmetic tests publish.
// ===========================================================================

TEST(HuberLossTilingTest, TilingKeySelectsTheScheduleMode)
{
    TilingInfo info;
    ASSERT_TRUE(
        RunTiling({{1024}, {1024}, {1024}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, HUBER_LOSS_REDUCE_NONE}, info));
    EXPECT_EQ(info.tilingKey, HUBER_LOSS_SCH_MODE_NONE);

    ASSERT_TRUE(
        RunTiling({{1024}, {1024}, {}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, HUBER_LOSS_REDUCE_MEAN}, info));
    EXPECT_EQ(info.tilingKey, HUBER_LOSS_SCH_MODE_REDUCE);

    ASSERT_TRUE(RunTiling({{1024}, {1024}, {}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, HUBER_LOSS_REDUCE_SUM}, info));
    EXPECT_EQ(info.tilingKey, HUBER_LOSS_SCH_MODE_REDUCE);
}

TEST(HuberLossTilingTest, ReduceAsksForMoreWorkspaceThanNone)
{
    // The operator's own workspace holds the cross-core reduction slots and
    // the sync buffer, and is requested only for the reduced modes. The
    // returned size also carries the framework's reserved system workspace,
    // added unconditionally, so this compares the two modes against each
    // other rather than against an absolute number.
    //
    // SetScheduleMode(BATCH_MODE) cannot be read back through TilingInfo, so
    // what is pinned here is the predicate that guards the call.
    TilingInfo noneInfo;
    TilingInfo reduceInfo;
    ASSERT_TRUE(RunTiling(
        {{100000}, {100000}, {100000}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, HUBER_LOSS_REDUCE_NONE}, noneInfo));
    ASSERT_TRUE(RunTiling({{100000}, {100000}, {}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, HUBER_LOSS_REDUCE_MEAN},
                          reduceInfo));
    ASSERT_EQ(noneInfo.workspaceSizes.size(), 1U);
    ASSERT_EQ(reduceInfo.workspaceSizes.size(), 1U);
    EXPECT_GT(reduceInfo.workspaceSizes[0], noneInfo.workspaceSizes[0]);
    // The slot region sits after the sync buffer, so a reduced plan must place
    // it at a non-zero offset; none never allocates one.
    EXPECT_GT(Data(reduceInfo)->slotRegionOffset, 0U);
    EXPECT_EQ(Data(noneInfo)->slotRegionOffset, 0U);
}

// ===========================================================================
// Core split. The arithmetic tests already sweep this; the point here is that
// the split published through the framework is the same one, i.e. that nothing
// is lost between CalcTiling and the tiling data the kernel receives.
// ===========================================================================

TEST(HuberLossTilingTest, PublishedSplitConservesEveryElement)
{
    const std::vector<std::vector<int64_t>> shapes = {{1}, {31}, {1024}, {32769}, {99991}, {4, 1024, 1024}};
    for (const auto& shape : shapes) {
        TilingInfo info;
        ASSERT_TRUE(
            RunTiling({shape, shape, shape, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, HUBER_LOSS_REDUCE_NONE}, info))
            << "numel=" << Numel(shape);
        const auto* d = Data(info);
        EXPECT_EQ(d->totalNumel, Numel(shape));
        EXPECT_EQ(info.blockNum, d->usedCoreNum);
        EXPECT_GT(d->tileDataNum, 0U);
        EXPECT_LE(d->usedCoreNum, kCoreNum);
        const uint64_t covered = static_cast<uint64_t>(d->frontCoreNum) * d->coreDataNum +
                                 static_cast<uint64_t>(d->usedCoreNum - d->frontCoreNum) * d->tailCoreDataNum;
        EXPECT_EQ(covered, Numel(shape)) << "numel=" << Numel(shape);
    }
}

TEST(HuberLossTilingTest, EmptyTensorIsLegalAndStillSizesBuffers)
{
    TilingInfo info;
    // An empty tensor is handled through the divisor alone: mean over zero
    // elements is 0/0, the NaN PyTorch produces. tileDataNum must still be
    // positive because the buffers are sized from it.
    ASSERT_TRUE(RunTiling({{0}, {0}, {0}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, HUBER_LOSS_REDUCE_NONE}, info));
    EXPECT_EQ(Data(info)->totalNumel, 0U);
    EXPECT_GT(Data(info)->tileDataNum, 0U);

    ASSERT_TRUE(RunTiling({{0}, {0}, {}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, HUBER_LOSS_REDUCE_MEAN}, info));
    EXPECT_FLOAT_EQ(Data(info)->divisor, 0.0f);
}

TEST(HuberLossTilingTest, AcceptsEverySupportedDtype)
{
    for (const ge::DataType dtype : {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}) {
        TilingInfo info;
        ASSERT_TRUE(RunTiling({{4096}, {4096}, {4096}, dtype, dtype, dtype, HUBER_LOSS_REDUCE_NONE}, info))
            << "dtype=" << static_cast<int>(dtype);
        EXPECT_GT(Data(info)->tileDataNum, 0U);
    }
}

// ===========================================================================
// Rejections.
// ===========================================================================

TEST(HuberLossTilingTest, RejectsMismatchedInputShapes)
{
    TilingInfo info;
    EXPECT_FALSE(
        RunTiling({{2, 3}, {2, 4}, {2, 3}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, HUBER_LOSS_REDUCE_NONE}, info));
}

TEST(HuberLossTilingTest, RejectsMismatchedInputDtypes)
{
    TilingInfo info;
    EXPECT_FALSE(RunTiling({{8}, {8}, {8}, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT, HUBER_LOSS_REDUCE_NONE}, info));
}

TEST(HuberLossTilingTest, RejectsOutOfRangeReduction)
{
    TilingInfo info;
    EXPECT_FALSE(RunTiling({{8}, {8}, {8}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, 3}, info));
    EXPECT_FALSE(RunTiling({{8}, {8}, {8}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, -1}, info));
}

TEST(HuberLossTilingTest, RejectsNonPositiveOrNaNDelta)
{
    TilingInfo info;
    const Case base = {{8}, {8}, {8}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, HUBER_LOSS_REDUCE_NONE};
    Case zero = base;
    zero.delta = 0.0f;
    Case negative = base;
    negative.delta = -1.0f;
    Case nan = base;
    nan.delta = std::nanf("");
    EXPECT_FALSE(RunTiling(zero, info));
    EXPECT_FALSE(RunTiling(negative, info));
    EXPECT_FALSE(RunTiling(nan, info));
}

TEST(HuberLossTilingTest, AcceptsInfiniteDelta)
{
    // delta is constrained to > 0 and nothing more. At +inf the formula
    // degenerates to 0.5e^2, which is a legal request, not an error.
    TilingInfo info;
    Case c = {{1024}, {1024}, {1024}, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, HUBER_LOSS_REDUCE_NONE};
    c.delta = std::numeric_limits<float>::infinity();
    EXPECT_TRUE(RunTiling(c, info));
}

// The UB budget is consulted: too little space is refused, and what fits is
// sized from the space actually available. 1024 and 8192 sit below one
// ACC_LEN granule for this shape and are refused; smaller values are clamped
// by the platform layer before reaching the operator's own arithmetic, so
// they cannot exercise this path.
TEST(HuberLossTilingTest, RejectsUbTooSmallForOneTile)
{
    TilingInfo info;
    Case c = {{1024 * 1024}, {1024 * 1024}, {1024 * 1024},         ge::DT_FLOAT,
              ge::DT_FLOAT,  ge::DT_FLOAT,  HUBER_LOSS_REDUCE_NONE};
    c.ubSize = 8192;
    EXPECT_FALSE(RunTiling(c, info));
    c.ubSize = 1024;
    EXPECT_FALSE(RunTiling(c, info));
}

TEST(HuberLossTilingTest, TileSizeFollowsAvailableUb)
{
    TilingInfo big;
    TilingInfo small;
    Case c = {{1024 * 1024}, {1024 * 1024}, {1024 * 1024},         ge::DT_FLOAT,
              ge::DT_FLOAT,  ge::DT_FLOAT,  HUBER_LOSS_REDUCE_NONE};
    c.ubSize = 196608;
    ASSERT_TRUE(RunTiling(c, big));
    c.ubSize = 32768;
    ASSERT_TRUE(RunTiling(c, small));
    EXPECT_LT(Data(small)->tileDataNum, Data(big)->tileDataNum)
        << "tile at 32768 B = " << Data(small)->tileDataNum << ", tile at 196608 B = " << Data(big)->tileDataNum;
}

} // namespace
