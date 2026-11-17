/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <gtest/gtest.h>

#include "../../../../unique/op_api/aclnn_unique.h"
#include "../../../../unique/op_api/aclnn_unique2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/op_executor.h"
#include "opdev/platform.h"

namespace {
constexpr int64_t ELEMENTS = 8;
constexpr size_t PRIVATE_OUTPUTS = 3;
// Three output shape records, each storing rank plus up to eight dimensions.
constexpr int64_t SHAPE_STORAGE_ELEMENTS = 27;

enum class Entry { Unique, Unique2 };
enum class Route { Aicore, Aicpu };

class AclnnUniqueSortingRouteTest : public testing::Test {};

void ExpectRoute(const aclTensor* self, aclTensor* values, Entry entry, Route route)
{
    auto inverse = TensorDesc({0}, ACL_INT64, ACL_FORMAT_ND).ToAclType();
    auto counts = TensorDesc({0}, ACL_INT64, ACL_FORMAT_ND).ToAclType();
    const auto inverseShape = inverse->GetViewShape();
    const auto countsShape = counts->GetViewShape();
    uint64_t workspaceSize = 0;
    aclOpExecutor* rawExecutor = nullptr;
    aclnnStatus status;
    if (entry == Entry::Unique2) {
        status = aclnnUnique2GetWorkspaceSize(self, true, false, false, values, inverse.get(), counts.get(),
                                              &workspaceSize, &rawExecutor);
    } else {
        status = aclnnUniqueGetWorkspaceSize(self, true, false, values, inverse.get(), &workspaceSize, &rawExecutor);
    }
    std::unique_ptr<aclOpExecutor> executor(rawExecutor);
    ASSERT_EQ(status, ACLNN_SUCCESS);
    ASSERT_NE(executor.get(), nullptr);
    EXPECT_EQ(inverse->GetViewShape(), inverseShape);
    EXPECT_EQ(counts->GetViewShape(), countsShape);

    // Both launcher stubs return SUCCESS. Private output allocation distinguishes the actual route:
    // values-only AICore borrows private inverse/count placeholders and shape storage; AICPU does not.
    const auto& allocated = executor->allocatedTensorList_;
    if (route == Route::Aicpu) {
        EXPECT_TRUE(allocated.empty());
        return;
    }
    ASSERT_EQ(allocated.size(), PRIVATE_OUTPUTS);
    for (const auto* tensor : allocated) {
        EXPECT_EQ(tensor->GetDataType(), op::DataType::DT_INT64);
        EXPECT_EQ(tensor->GetViewShape().GetDimNum(), 1U);
    }
    EXPECT_EQ(allocated[0]->GetViewShape().GetDim(0), 1);
    EXPECT_EQ(allocated[1]->GetViewShape().GetDim(0), 1);
    EXPECT_EQ(allocated[PRIVATE_OUTPUTS - 1]->GetViewShape().GetDim(0), SHAPE_STORAGE_ELEMENTS);
}

void ExpectBothEntries(const aclTensor* self, aclTensor* values, Route route)
{
    ExpectRoute(self, values, Entry::Unique, route);
    ExpectRoute(self, values, Entry::Unique2, route);
}

TEST_F(AclnnUniqueSortingRouteTest, ValuesOnlyUsesPrivateOutputsForBothEntries)
{
    op::SocVersionManager soc(op::SocVersion::ASCEND950);
    for (const auto dtype : {ACL_FLOAT, ACL_FLOAT16, ACL_BF16}) {
        SCOPED_TRACE(dtype);
        auto self = TensorDesc({ELEMENTS}, dtype, ACL_FORMAT_ND).ToAclType();
        auto values = TensorDesc({ELEMENTS}, dtype, ACL_FORMAT_ND).ToAclType();
        ExpectBothEntries(self.get(), values.get(), Route::Aicore);
    }
}

TEST_F(AclnnUniqueSortingRouteTest, NonRegbaseUsesExistingAicpuFallback)
{
    op::SocVersionManager soc(op::SocVersion::ASCEND910B);
    auto self = TensorDesc({ELEMENTS}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
    auto values = TensorDesc({ELEMENTS}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
    ExpectBothEntries(self.get(), values.get(), Route::Aicpu);
}

TEST_F(AclnnUniqueSortingRouteTest, DtypesOutsideValuesOnlySetUseExistingAicpuFallback)
{
    op::SocVersionManager soc(op::SocVersion::ASCEND950);
    for (const auto dtype : {ACL_BOOL, ACL_DOUBLE}) {
        SCOPED_TRACE(dtype);
        auto self = TensorDesc({ELEMENTS}, dtype, ACL_FORMAT_ND).ToAclType();
        auto values = TensorDesc({ELEMENTS}, dtype, ACL_FORMAT_ND).ToAclType();
        ExpectBothEntries(self.get(), values.get(), Route::Aicpu);
    }
}

TEST_F(AclnnUniqueSortingRouteTest, AdjacentKnownStorageRemainsEligibleInEitherDirection)
{
    op::SocVersionManager soc(op::SocVersion::ASCEND950);
    std::array<float, ELEMENTS * 2> storage{};
    auto self = TensorDesc({ELEMENTS}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
    auto values = TensorDesc({ELEMENTS}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
    self->SetStorageAddr(storage.data());
    values->SetStorageAddr(storage.data() + ELEMENTS);
    ExpectBothEntries(self.get(), values.get(), Route::Aicore);
    self->SetStorageAddr(storage.data() + ELEMENTS);
    values->SetStorageAddr(storage.data());
    ExpectBothEntries(self.get(), values.get(), Route::Aicore);
}

TEST_F(AclnnUniqueSortingRouteTest, ExecutorOwnedUnresolvedStorageRemainsEligible)
{
    op::SocVersionManager soc(op::SocVersion::ASCEND950);
    std::array<float, ELEMENTS> storage{};
    auto self = TensorDesc({ELEMENTS}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
    auto values = TensorDesc({ELEMENTS}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
    self->SetStorageAddr(storage.data());
    ExpectBothEntries(self.get(), values.get(), Route::Aicore);
    self->SetStorageAddr(nullptr);
    values->SetStorageAddr(storage.data());
    ExpectBothEntries(self.get(), values.get(), Route::Aicore);
}
} // namespace
