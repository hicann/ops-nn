/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <dlfcn.h>
#include <memory>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"
#include "../../../op_api/aclnn_scatter_nd_update.h"
#include "aclnn_kernels/contiguous.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/op_executor.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"

namespace {
struct WritebackState {
    const aclTensor* original = nullptr;
    const aclTensor* contiguous = nullptr;
    const aclTensor* source = nullptr;
    const aclTensor* destination = nullptr;
    size_t contiguousCalls = 0;
    size_t viewCopyCalls = 0;
};

thread_local WritebackState* scatterNdWritebackState = nullptr;
} // namespace

// Intercept calls only while this fixture is active. Other tests must reach the
// original implementation, including libopapi_math.so when preloaded by CI.
namespace l0op {
const aclTensor* Contiguous(const aclTensor* x, aclOpExecutor* executor)
{
    auto* state = scatterNdWritebackState;
    if (state == nullptr) {
        using Function = const aclTensor* (*)(const aclTensor*, aclOpExecutor*);
        static const auto original = reinterpret_cast<Function>(
            dlsym(RTLD_NEXT, "_ZN4l0op10ContiguousEPK9aclTensorP13aclOpExecutor"));
        EXPECT_NE(original, nullptr);
        return original == nullptr ? nullptr : original(x, executor);
    }
    if (x == state->original) {
        ++state->contiguousCalls;
        if (!op::IsContiguous(x)) {
            state->contiguous = executor->AllocTensor(x->GetViewShape(), x->GetDataType());
            return state->contiguous;
        }
    }
    return x;
}

const aclTensor* ViewCopy(const aclTensor* x, const aclTensor* y, aclOpExecutor* executor)
{
    auto* state = scatterNdWritebackState;
    if (state == nullptr) {
        using Function = const aclTensor* (*)(const aclTensor*, const aclTensor*, aclOpExecutor*);
        static const auto original = reinterpret_cast<Function>(
            dlsym(RTLD_NEXT, "_ZN4l0op8ViewCopyEPK9aclTensorS2_P13aclOpExecutor"));
        EXPECT_NE(original, nullptr);
        return original == nullptr ? nullptr : original(x, y, executor);
    }
    ++state->viewCopyCalls;
    state->source = x;
    state->destination = y;
    return x;
}
} // namespace l0op

namespace {
struct WritebackLayout {
    std::vector<int64_t> strides;
    bool needsWriteback;
    bool callsContiguous;
};

using WritebackParam = std::tuple<op::SocVersion, WritebackLayout, int64_t, aclDataType>;

class ScatterNdUpdateWritebackTest : public testing::TestWithParam<WritebackParam> {
protected:
    void SetUp() override
    {
        previousState_ = scatterNdWritebackState;
        scatterNdWritebackState = &state_;
    }

    void TearDown() override
    {
        executor_.reset();
        scatterNdWritebackState = previousState_;
    }

    WritebackState state_;
    WritebackState* previousState_ = nullptr;
    std::unique_ptr<aclOpExecutor> executor_;
};

TEST_P(ScatterNdUpdateWritebackTest, PreservesOriginalView)
{
    const auto& [soc, layout, offset, indexType] = GetParam();
    op::SocVersionManager platform(soc);
    const auto expectedArch = soc == op::SocVersion::ASCEND950 ? NpuArch::DAV_3510 : NpuArch::DAV_2201;
    ASSERT_EQ(op::GetCurrentPlatformInfo().GetCurNpuArch(), expectedArch);

    auto var = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND, layout.strides, offset, {32}).ToAclType();
    auto indices = TensorDesc({1, 1}, indexType, ACL_FORMAT_ND).ToAclType();
    auto updates = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
    ASSERT_NE(var.get(), nullptr);
    ASSERT_NE(indices.get(), nullptr);
    ASSERT_NE(updates.get(), nullptr);
    state_.original = var.get();
    const auto originalShape = var->GetViewShape();
    const auto originalStrides = var->GetViewStrides();
    const auto originalStorageShape = var->GetStorageShape();

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto status = aclnnScatterNdUpdateGetWorkspaceSize(var.get(), indices.get(), updates.get(), &workspaceSize,
                                                             &executor);
    executor_.reset(executor);
    ASSERT_EQ(status, ACLNN_SUCCESS);
    ASSERT_NE(executor, nullptr);

    EXPECT_EQ(state_.contiguousCalls, layout.callsContiguous ? 1U : 0U);
    if (layout.needsWriteback) {
        ASSERT_NE(state_.contiguous, nullptr);
        ASSERT_NE(state_.contiguous, var.get());
        EXPECT_TRUE(op::IsContiguous(state_.contiguous));
        EXPECT_EQ(state_.viewCopyCalls, 1U);
        EXPECT_EQ(state_.source, state_.contiguous);
        EXPECT_EQ(state_.destination, var.get());
    } else {
        EXPECT_EQ(state_.viewCopyCalls, 0U);
        EXPECT_EQ(state_.contiguous, nullptr);
    }
    EXPECT_EQ(var->GetViewShape(), originalShape);
    EXPECT_EQ(var->GetViewStrides(), originalStrides);
    EXPECT_EQ(var->GetStorageShape(), originalStorageShape);
    EXPECT_EQ(var->GetViewOffset(), offset);
}

INSTANTIATE_TEST_SUITE_P(
    PlatformsAndLayouts, ScatterNdUpdateWritebackTest,
    testing::Combine(testing::Values(op::SocVersion::ASCEND910B, op::SocVersion::ASCEND910_93,
                                     op::SocVersion::ASCEND950),
                     testing::Values(WritebackLayout{{6, 2}, true, true}, WritebackLayout{{1, 2}, true, true},
                                     WritebackLayout{{3, 1}, false, true}, WritebackLayout{{6, 1}, false, false}),
                     testing::Values(int64_t{0}, int64_t{3}), testing::Values(ACL_INT32, ACL_INT64)));
} // namespace
