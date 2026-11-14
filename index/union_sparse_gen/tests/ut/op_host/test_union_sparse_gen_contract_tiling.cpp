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
#include <limits>
#include "../../../op_host/union_sparse_gen_contract.h"
#include "../../../op_kernel/arch22/union_sparse_gen_membership_plan.h"
#include "../../../op_kernel/union_sparse_gen_tiling_data.h"

namespace {
using namespace UnionSparseGenContract;
TEST(UnionSparseGenContract, PrefixAndFixedBatchGeometry)
{
    int64_t batch = -1;
    EXPECT_TRUE(BatchSize(32, 3, -1, batch));
    EXPECT_EQ(batch, 2);
    EXPECT_TRUE(BatchSize(32, -1, 2, batch));
    EXPECT_EQ(batch, 2);
    EXPECT_TRUE(BatchSize(0, -1, 0, batch));
    EXPECT_FALSE(BatchSize(32, -1, 0, batch));
    EXPECT_FALSE(BatchSize(31, -1, 2, batch));
    EXPECT_FALSE(BatchSize(32, -1, -1, batch));
    EXPECT_TRUE(SupportsMask(1));
    EXPECT_TRUE(SupportsMask(0));
    const int64_t shape[] = {1, 128};
    EXPECT_TRUE(SupportsBlockShape(shape, 2));
    EXPECT_FALSE(SupportsBlockShape(nullptr, 2));
    EXPECT_FALSE(SupportsBlockShape(shape, 1));
    EXPECT_FALSE(SupportsBlockShape(shape, 3));
    EXPECT_FALSE(SupportsMask(2));
    EXPECT_TRUE(SupportsSpec(1, 128, 8, 1, 4, true, 0));
    EXPECT_FALSE(SupportsSpec(1, 8, 8, 1, 4, true, 0));
    EXPECT_FALSE(SupportsSpec(1, 128, 8, 1, 4, false, 0));
    EXPECT_TRUE(Attributes(4));
    EXPECT_FALSE(Attributes(0));
}

TEST(UnionSparseGenContract, MetadataBudgetIncludesOnlyConsumedWindows)
{
    EXPECT_EQ(UnionSparseGenMetadataBytes(true, false, false), 2112);
    EXPECT_EQ(UnionSparseGenMetadataBytes(true, true, true), 2080);
    EXPECT_EQ(UnionSparseGenMetadataBytes(true, true, false), 2624);
    EXPECT_EQ(UnionSparseGenMetadataBytes(true, false, true), 1568);
    EXPECT_EQ(UnionSparseGenMetadataBytes(false, true, true), 1024);
    EXPECT_EQ(UnionSparseGenMetadataBytes(false, true, false), 1568);
    EXPECT_EQ(UnionSparseGenMetadataBudgetBytes(true, false, false), 2176);
    EXPECT_EQ(UnionSparseGenMetadataBudgetBytes(true, true, false), 2624);
    EXPECT_EQ(UnionSparseGenMetadataBudgetBytes(false, true, true), 2176);
    // Measured device reports 196352B, not the rounded 192KiB illustration.
    // The largest aligned resident K must still fit with only usedQ present.
    constexpr int64_t fixed = 16384 + 128 + 8192 + 65536 + 1024;
    const int64_t algorithm = (196352 - fixed - UnionSparseGenMetadataBudgetBytes(true, true, false)) / 32 * 32;
    EXPECT_EQ(algorithm, 102464);
    EXPECT_GE(algorithm, 8 * 664 * 16 + UnionSparseGenA2Plan::VectorScratchBytes(256));
}

TEST(UnionSparseGenContract, LengthModesAndMaxima)
{
    int64_t batch = -1;
    EXPECT_TRUE(LengthGeometry(32, 3, 2, -1, -1, batch));
    EXPECT_EQ(batch, 2);
    EXPECT_TRUE(LengthGeometry(32, -1, 2, 16, -1, batch));
    EXPECT_FALSE(LengthGeometry(32, -1, 2, -1, -1, batch));
    EXPECT_FALSE(LengthGeometry(32, -1, 2, 0, -1, batch));
    EXPECT_FALSE(LengthGeometry(32, -1, 2, 8, -1, batch));
    EXPECT_FALSE(LengthGeometry(32, 3, 2, -2, -1, batch));
    EXPECT_FALSE(LengthGeometry(32, 3, 2, -1, -2, batch));
    EXPECT_TRUE(LengthGeometry(0, -1, 0, 1, -1, batch));
}

TEST(UnionSparseGenContract, DeliveryCapacityRejectsVirtualTail)
{
    int64_t capacity = 0;
    ASSERT_TRUE(Capacity(16, 8, 1024, 0, capacity));
    EXPECT_EQ(capacity, 128);
    EXPECT_FALSE(Capacity(16, 8, 1024, 1, capacity));
    ASSERT_TRUE(Capacity(512, 8, 33, 0, capacity));
    EXPECT_EQ(capacity, 33);
}

TEST(UnionSparseGenContract, CapacityRejectsInvalidAndOverflowingBounds)
{
    int64_t capacity = 0;
    const int64_t maximum = std::numeric_limits<int64_t>::max();
    EXPECT_FALSE(Capacity(maximum, 8, 128, 1, capacity));
    EXPECT_FALSE(Capacity(maximum / 8, 8, 128, 1, capacity));
    EXPECT_TRUE(Capacity(maximum / 8, 8, 128, 0, capacity));
    EXPECT_FALSE(Capacity(maximum / 8 + 1, 8, 128, 0, capacity));
    EXPECT_FALSE(Capacity(16, 8, int64_t(std::numeric_limits<int32_t>::max()) + 1, 0, capacity));
    EXPECT_FALSE(Capacity(0, 8, 128, 0, capacity));
    EXPECT_FALSE(Capacity(16, 4, 128, 0, capacity));
    EXPECT_FALSE(Capacity(16, 8, 128, 2, capacity));
}

TEST(UnionSparseGenContract, ByteExtentsAreCheckedBeforeMultiplication)
{
    const int64_t maximum = std::numeric_limits<int64_t>::max();
    EXPECT_TRUE(OutputFits(2, 0, 128, 128));
    EXPECT_TRUE(InputFits(2, 45, 513));
    EXPECT_FALSE(InputFits(2, maximum / 4, 2));
    EXPECT_TRUE(OutputFits(1, maximum / 4, 1, 1));
    EXPECT_FALSE(OutputFits(1, maximum / 4 + 1, 1, 1));
    EXPECT_FALSE(OutputFits(1, 1, maximum / 4, 128));
    EXPECT_FALSE(OutputFits(1, -1, 128, 4));
}
TEST(UnionSparseGenA2Plan, FullSortBudgetIncludesRoundedRowsAndScratch)
{
    using namespace UnionSparseGenA2Plan;
    // Tail changes K512 from 512 to544 padded records per query.
    EXPECT_EQ(MembershipSortStride(512, 0), 512);
    EXPECT_EQ(MembershipSortStride(512, 1), 544);
    constexpr int64_t exact = 8 * 544 * 16 + kMembershipSortScratchBytes;
    EXPECT_TRUE(MembershipSortFits(512, 8, 1, exact));
    EXPECT_FALSE(MembershipSortFits(512, 8, 1, exact - 1));
    EXPECT_TRUE(MembershipSortFits(513, 8, 0, exact));
}

TEST(UnionSparseGenA2Plan, SortRepeatLimitIsCheckedBeforeExtremeStrideArithmetic)
{
    using namespace UnionSparseGenA2Plan;
    const int64_t maximum = std::numeric_limits<int64_t>::max();
    EXPECT_FALSE(MembershipSortFits(maximum, 8, 1, maximum));
    EXPECT_FALSE(MembershipSortFits(maximum / 8, 8, 0, maximum));
    EXPECT_FALSE(MembershipSortFits(1024, 8, 1, maximum));
    EXPECT_FALSE(MembershipSortFits(224, 32, 1, maximum));
    EXPECT_TRUE(MembershipSortFits(224, 32, 0, maximum));
    EXPECT_TRUE(MembershipSortFits(480, 16, 0, maximum));
    EXPECT_FALSE(MembershipSortFits(480, 16, 1, maximum));
    EXPECT_FALSE(MembershipSortFits(512, 0, 0, maximum));
}
} // namespace
