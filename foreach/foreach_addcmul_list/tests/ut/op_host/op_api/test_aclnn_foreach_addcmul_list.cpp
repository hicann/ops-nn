/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <array>
#include <float.h>
#include "gtest/gtest.h"

#include "../../../../op_host/op_api/aclnn_foreach_addcmul_list.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_foreach_addcmul_list_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "foreach_addcmul_list_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "foreach_addcmul_list_test TearDown" << endl; }
};

// 统一入口：只走第一段接口（workspace 计算 + executor 组装），与仓内其它 l2_* 用例口径一致。
static aclnnStatus CallGetWorkspaceSize(const aclTensorList* x1, const aclTensorList* x2, const aclTensorList* x3,
                                        const aclTensor* scalars, const aclTensorList* out, uint64_t* workspaceSize)
{
    aclOpExecutor* executor = nullptr;
    auto ret = aclnnForeachAddcmulListGetWorkspaceSize(x1, x2, x3, scalars, out, workspaceSize, &executor);
    if (executor != nullptr) {
        delete executor;
    }
    return ret;
}

// 改动点 2f/2h：输出非连续（transpose 布局）时应先取连续临时缓冲计算再 ViewCopy 回写，
// 接口必须成功（修复前 autogen 无此步，且无 UT 可覆盖该分支）。
TEST_F(l2_foreach_addcmul_list_test, ascend910B2_addcmul_list_test_non_contiguous_out_transposed)
{
    auto x1 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x2 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x3 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalars = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});

    uint64_t workspaceSize = 0;
    auto ret = CallGetWorkspaceSize(DescToAclContainer(TensorListDesc({x1})), DescToAclContainer(TensorListDesc({x2})),
                                    DescToAclContainer(TensorListDesc({x3})), DescToAclContainer(scalars),
                                    DescToAclContainer(TensorListDesc({out})), &workspaceSize);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

// 改动点 2f/2h：输入输出同时非连续（不同 stride/storage），三段连续化 + 回写组合路径。
TEST_F(l2_foreach_addcmul_list_test, ascend910B2_addcmul_list_test_non_contiguous_input_and_out)
{
    auto x1 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).ValueRange(-1, 1);
    auto x2 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).ValueRange(-1, 1);
    auto x3 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).ValueRange(-1, 1);
    auto scalars = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {9, 2}, 1, {2, 9});

    uint64_t workspaceSize = 0;
    auto ret = CallGetWorkspaceSize(DescToAclContainer(TensorListDesc({x1})), DescToAclContainer(TensorListDesc({x2})),
                                    DescToAclContainer(TensorListDesc({x3})), DescToAclContainer(scalars),
                                    DescToAclContainer(TensorListDesc({out})), &workspaceSize);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

// 改动点 2f/2h：out 的 view 布局本身连续但 storage 大于 view（共享大 buffer 的切片），
// 走 storage_shape != view_shape 判定分支：必须先 Contiguous 再 ViewCopy，不能直接按连续写。
TEST_F(l2_foreach_addcmul_list_test, ascend910B2_addcmul_list_test_out_storage_slice)
{
    auto x1 = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x2 = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x3 = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalars = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND, {6, 1}, 0, {2, 6});

    uint64_t workspaceSize = 0;
    auto ret = CallGetWorkspaceSize(DescToAclContainer(TensorListDesc({x1})), DescToAclContainer(TensorListDesc({x2})),
                                    DescToAclContainer(TensorListDesc({x3})), DescToAclContainer(scalars),
                                    DescToAclContainer(TensorListDesc({out})), &workspaceSize);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

// 改动点 2g：scalars 为共享 storage 的切片视图（view(1,)/storage(2,)/offset 1），
// 必须规整为独立连续张量，否则 kernel 按下标平坦读取会取错标量。
TEST_F(l2_foreach_addcmul_list_test, ascend910B2_addcmul_list_test_scalars_storage_slice)
{
    auto x1 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x2 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x3 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalars = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND, {1}, 1, {2}).ValueRange(-1, 1);
    auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    uint64_t workspaceSize = 0;
    auto ret = CallGetWorkspaceSize(DescToAclContainer(TensorListDesc({x1})), DescToAclContainer(TensorListDesc({x2})),
                                    DescToAclContainer(TensorListDesc({x3})), DescToAclContainer(scalars),
                                    DescToAclContainer(TensorListDesc({out})), &workspaceSize);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

// 改动点 2f/2h：单次调用内多个 tensor 的输出布局混杂（连续 / transpose / storage 切片），
// 回写必须逐项按索引对齐，不能错位。
TEST_F(l2_foreach_addcmul_list_test, ascend910B2_addcmul_list_test_multi_tensor_mixed_out_layouts)
{
    auto x0 = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x1 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x2 = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto z0 = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto z1 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto z2 = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto w0 = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto w1 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto w2 = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalars = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out0 = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out1 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
    auto out2 = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND, {6, 1}, 0, {2, 6});

    uint64_t workspaceSize = 0;
    auto ret = CallGetWorkspaceSize(DescToAclContainer(TensorListDesc({x0, x1, x2})),
                                    DescToAclContainer(TensorListDesc({z0, z1, z2})),
                                    DescToAclContainer(TensorListDesc({w0, w1, w2})), DescToAclContainer(scalars),
                                    DescToAclContainer(TensorListDesc({out0, out1, out2})), &workspaceSize);
    EXPECT_EQ(ret, ACL_SUCCESS);
}

// 改动点 2j：空 TensorList 短路返回成功且 workspaceSize 为 0（手写 L2 的可观测行为，
// autogen 路径会走到 tiling 报错，故此用例同时是"手写实现生效"的行为断言）。
TEST_F(l2_foreach_addcmul_list_test, ascend910B2_addcmul_list_test_empty_tensor_list)
{
    aclTensorList* x1List = aclCreateTensorList(nullptr, 0);
    aclTensorList* x2List = aclCreateTensorList(nullptr, 0);
    aclTensorList* x3List = aclCreateTensorList(nullptr, 0);
    aclTensorList* outList = aclCreateTensorList(nullptr, 0);
    auto scalars = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    ASSERT_NE(x1List, nullptr);
    ASSERT_NE(x2List, nullptr);
    ASSERT_NE(x3List, nullptr);
    ASSERT_NE(outList, nullptr);

    // 空 TensorList（四列表长度一致且均为 0）为合法 no-op：返回成功与零 workspace。
    uint64_t workspaceSize = 1;
    auto ret = CallGetWorkspaceSize(x1List, x2List, x3List, DescToAclContainer(scalars), outList, &workspaceSize);
    EXPECT_EQ(ret, ACL_SUCCESS);
    EXPECT_EQ(workspaceSize, 0U);
    aclDestroyTensorList(x1List);
    aclDestroyTensorList(x2List);
    aclDestroyTensorList(x3List);
    aclDestroyTensorList(outList);
}

// 反向：x1 为空指针。
TEST_F(l2_foreach_addcmul_list_test, ascend910B2_addcmul_list_test_null_x1_rejected)
{
    auto x = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalars = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    uint64_t workspaceSize = 0;
    auto ret = CallGetWorkspaceSize(nullptr, DescToAclContainer(TensorListDesc({x})),
                                    DescToAclContainer(TensorListDesc({x})), DescToAclContainer(scalars),
                                    DescToAclContainer(TensorListDesc({x})), &workspaceSize);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
}

// 反向：out 为空指针。
TEST_F(l2_foreach_addcmul_list_test, ascend910B2_addcmul_list_test_null_out_rejected)
{
    auto x = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalars = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    uint64_t workspaceSize = 0;
    auto ret = CallGetWorkspaceSize(DescToAclContainer(TensorListDesc({x})), DescToAclContainer(TensorListDesc({x})),
                                    DescToAclContainer(TensorListDesc({x})), DescToAclContainer(scalars), nullptr,
                                    &workspaceSize);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
}

// 反向：scalars 为空指针。
TEST_F(l2_foreach_addcmul_list_test, ascend910B2_addcmul_list_test_null_scalars_rejected)
{
    auto x = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    uint64_t workspaceSize = 0;
    auto ret = CallGetWorkspaceSize(DescToAclContainer(TensorListDesc({x})), DescToAclContainer(TensorListDesc({x})),
                                    DescToAclContainer(TensorListDesc({x})), nullptr,
                                    DescToAclContainer(TensorListDesc({x})), &workspaceSize);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
}

// 反向：list 内元素为空指针，必须在解引用前被 CheckTensorListNotNull 拦下。
TEST_F(l2_foreach_addcmul_list_test, ascend910B2_addcmul_list_test_null_entry_in_list_rejected)
{
    const aclTensor* nullEntry[] = {nullptr};
    aclTensorList* x1List = aclCreateTensorList(nullEntry, 1);
    auto x = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalars = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    ASSERT_NE(x1List, nullptr);

    uint64_t workspaceSize = 0;
    auto ret = CallGetWorkspaceSize(x1List, DescToAclContainer(TensorListDesc({x})),
                                    DescToAclContainer(TensorListDesc({x})), DescToAclContainer(scalars),
                                    DescToAclContainer(TensorListDesc({x})), &workspaceSize);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
    aclDestroyTensorList(x1List);
}

// 反向：out list 长度与 x1 不一致。
TEST_F(l2_foreach_addcmul_list_test, ascend910B2_addcmul_list_test_out_list_too_short)
{
    auto x = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalars = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    uint64_t workspaceSize = 0;
    auto ret = CallGetWorkspaceSize(DescToAclContainer(TensorListDesc({x, x})),
                                    DescToAclContainer(TensorListDesc({x, x})),
                                    DescToAclContainer(TensorListDesc({x, x})), DescToAclContainer(scalars),
                                    DescToAclContainer(TensorListDesc({out})), &workspaceSize);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

// 反向：out 的 view shape 与 x1 不一致。
TEST_F(l2_foreach_addcmul_list_test, ascend910B2_addcmul_list_test_out_shape_mismatch_rejected)
{
    // out 的 shape size 需 >= x1（与资料口径一致）：此处 out 元素数 4 < x 的 8，应被拒绝。
    auto x = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({1, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalars = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    uint64_t workspaceSize = 0;
    auto ret = CallGetWorkspaceSize(DescToAclContainer(TensorListDesc({x})), DescToAclContainer(TensorListDesc({x})),
                                    DescToAclContainer(TensorListDesc({x})), DescToAclContainer(scalars),
                                    DescToAclContainer(TensorListDesc({out})), &workspaceSize);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

// 反向：out 的 shape size 大于 x1 时拒绝。kernel 只写 x1 的元素数，而回写按 out 的完整形状
// 进行；若放行且 out 为非连续视图，未写入的尾部会被未初始化数据覆盖。
TEST_F(l2_foreach_addcmul_list_test, ascend910B2_addcmul_list_test_out_shape_larger_rejected)
{
    auto x = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalars = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    uint64_t workspaceSize = 0;
    auto ret = CallGetWorkspaceSize(DescToAclContainer(TensorListDesc({x})), DescToAclContainer(TensorListDesc({x})),
                                    DescToAclContainer(TensorListDesc({x})), DescToAclContainer(scalars),
                                    DescToAclContainer(TensorListDesc({out})), &workspaceSize);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

// 反向：x2 的 TensorList 成员数少于 x1（长度不一致须在 L2 前置拒绝，否则按下标解引用越界）。
TEST_F(l2_foreach_addcmul_list_test, ascend910B2_addcmul_list_test_x2_list_shorter_rejected)
{
    auto x = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalars = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    uint64_t workspaceSize = 0;
    auto ret = CallGetWorkspaceSize(DescToAclContainer(TensorListDesc({x, x})), DescToAclContainer(TensorListDesc({x})),
                                    DescToAclContainer(TensorListDesc({x, x})), DescToAclContainer(scalars),
                                    DescToAclContainer(TensorListDesc({x, x})), &workspaceSize);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

// 反向：out 的 TensorList 成员数多于 x1。
TEST_F(l2_foreach_addcmul_list_test, ascend910B2_addcmul_list_test_out_list_longer_rejected)
{
    auto x = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalars = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    uint64_t workspaceSize = 0;
    auto ret = CallGetWorkspaceSize(DescToAclContainer(TensorListDesc({x})), DescToAclContainer(TensorListDesc({x})),
                                    DescToAclContainer(TensorListDesc({x})), DescToAclContainer(scalars),
                                    DescToAclContainer(TensorListDesc({x, x})), &workspaceSize);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

// 反向：workspaceSize 为空指针（L2_DFX_PHASE_1 内部 CheckPhase1Params 拦截，返回 161001）。
TEST_F(l2_foreach_addcmul_list_test, ascend910B2_addcmul_list_test_null_workspace_size_rejected)
{
    auto x = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalars = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ret = aclnnForeachAddcmulListGetWorkspaceSize(
        DescToAclContainer(TensorListDesc({x})), DescToAclContainer(TensorListDesc({x})),
        DescToAclContainer(TensorListDesc({x})), DescToAclContainer(scalars), DescToAclContainer(TensorListDesc({x})),
        nullptr, nullptr);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
}

// 反向：executor 为空指针。
TEST_F(l2_foreach_addcmul_list_test, ascend910B2_addcmul_list_test_null_executor_rejected)
{
    auto x = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalars = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    uint64_t workspaceSize = 0;
    auto ret = aclnnForeachAddcmulListGetWorkspaceSize(
        DescToAclContainer(TensorListDesc({x})), DescToAclContainer(TensorListDesc({x})),
        DescToAclContainer(TensorListDesc({x})), DescToAclContainer(scalars), DescToAclContainer(TensorListDesc({x})),
        &workspaceSize, nullptr);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
}

// 反向：out 的 dtype 与 x1 不一致。
TEST_F(l2_foreach_addcmul_list_test, ascend910B2_addcmul_list_test_out_dtype_mismatch_rejected)
{
    auto x = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 4}, ACL_BF16, ACL_FORMAT_ND);
    auto scalars = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    uint64_t workspaceSize = 0;
    auto ret = CallGetWorkspaceSize(DescToAclContainer(TensorListDesc({x})), DescToAclContainer(TensorListDesc({x})),
                                    DescToAclContainer(TensorListDesc({x})), DescToAclContainer(scalars),
                                    DescToAclContainer(TensorListDesc({out})), &workspaceSize);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

// 反向：不在支持列表内的 dtype（double）。
TEST_F(l2_foreach_addcmul_list_test, ascend910B2_addcmul_list_test_unsupported_dtype_rejected)
{
    auto x = TensorDesc({2, 4}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto scalars = TensorDesc({1}, ACL_DOUBLE, ACL_FORMAT_ND);
    uint64_t workspaceSize = 0;
    auto ret = CallGetWorkspaceSize(DescToAclContainer(TensorListDesc({x})), DescToAclContainer(TensorListDesc({x})),
                                    DescToAclContainer(TensorListDesc({x})), DescToAclContainer(scalars),
                                    DescToAclContainer(TensorListDesc({x})), &workspaceSize);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

// 反向：scalars 的 dtype 与 x1 不一致。
TEST_F(l2_foreach_addcmul_list_test, ascend910B2_addcmul_list_test_scalars_dtype_mismatch_rejected)
{
    auto x = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalars = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);
    uint64_t workspaceSize = 0;
    auto ret = CallGetWorkspaceSize(DescToAclContainer(TensorListDesc({x})), DescToAclContainer(TensorListDesc({x})),
                                    DescToAclContainer(TensorListDesc({x})), DescToAclContainer(scalars),
                                    DescToAclContainer(TensorListDesc({x})), &workspaceSize);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

// 反向：scalars 元素个数与 x1 的 tensor 个数不一致（2 个 tensor 配 1 个标量）。
TEST_F(l2_foreach_addcmul_list_test, ascend910B2_addcmul_list_test_scalars_count_mismatch_rejected)
{
    auto x = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalars = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    uint64_t workspaceSize = 0;
    auto ret = CallGetWorkspaceSize(DescToAclContainer(TensorListDesc({x, x})),
                                    DescToAclContainer(TensorListDesc({x, x})),
                                    DescToAclContainer(TensorListDesc({x, x})), DescToAclContainer(scalars),
                                    DescToAclContainer(TensorListDesc({x, x})), &workspaceSize);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}
