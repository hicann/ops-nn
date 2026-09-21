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
 * \file test_aclnn_cla_gate_backward.cpp
 * \brief
 */

#include <array>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "opdev/op_log.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "../../../op_api/aclnn_cla_gate_backward.h"

using namespace std;

class l2_cla_gate_backward_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "cla_gate_backward_test SetUp" << endl; }
    static void TearDownTestCase() { cout << "cla_gate_backward_test TearDown" << endl; }
};

namespace {
// 标准 TND 用例：T=8, N=64, D=256。
const vector<int64_t> kTndDims = {8, 64, 256};
const vector<int64_t> kLogitsDims = {8, 64};

TensorDesc AttnDesc(aclDataType dtype = ACL_BF16, aclFormat format = ACL_FORMAT_ND)
{
    return TensorDesc(kTndDims, dtype, format).ValueRange(-1, 1);
}

TensorDesc LogitsDesc(aclDataType dtype = ACL_BF16, aclFormat format = ACL_FORMAT_ND)
{
    return TensorDesc(kLogitsDims, dtype, format).ValueRange(-1, 1);
}

// 封装一次 GetWorkspaceSize：输入输出均由 TensorDesc 构造，允许单点覆盖 shape/dtype/format。
aclnnStatus RunClaGateBackward(const TensorDesc& gradMerged, const TensorDesc& globalAttn, const TensorDesc& localAttn,
                               const TensorDesc& globalGateLogits, const TensorDesc& localGateLogits,
                               const char* inputAttnLayout, const TensorDesc& gradGlobalAttnOut,
                               const TensorDesc& gradLocalAttnOut, const TensorDesc& gradGlobalGateLogitsOut,
                               const TensorDesc& gradLocalGateLogitsOut)
{
    auto ut = OP_API_UT(aclnnClaGateBackward,
                        INPUT(gradMerged, globalAttn, localAttn, globalGateLogits, localGateLogits, inputAttnLayout),
                        OUTPUT(gradGlobalAttnOut, gradLocalAttnOut, gradGlobalGateLogitsOut, gradLocalGateLogitsOut));
    uint64_t workspaceSize = 0;
    return ut.TestGetWorkspaceSize(&workspaceSize);
}
} // namespace

// ---------------- 正常路径 ----------------

TEST_F(l2_cla_gate_backward_test, ascend950_bf16_tnd_normal)
{
    const char* layout = "TND";
    EXPECT_EQ(RunClaGateBackward(AttnDesc(), AttnDesc(), AttnDesc(), LogitsDesc(), LogitsDesc(), layout, AttnDesc(),
                                 AttnDesc(), LogitsDesc(), LogitsDesc()),
              ACLNN_SUCCESS);
}

TEST_F(l2_cla_gate_backward_test, ascend950_fp16_tnd_normal)
{
    const char* layout = "TND";
    EXPECT_EQ(RunClaGateBackward(AttnDesc(ACL_FLOAT16), AttnDesc(ACL_FLOAT16), AttnDesc(ACL_FLOAT16),
                                 LogitsDesc(ACL_FLOAT16), LogitsDesc(ACL_FLOAT16), layout, AttnDesc(ACL_FLOAT16),
                                 AttnDesc(ACL_FLOAT16), LogitsDesc(ACL_FLOAT16), LogitsDesc(ACL_FLOAT16)),
              ACLNN_SUCCESS);
}

// inputAttnLayout == nullptr 时按默认 "TND" 处理。
TEST_F(l2_cla_gate_backward_test, ascend950_null_layout_default_tnd)
{
    EXPECT_EQ(RunClaGateBackward(AttnDesc(), AttnDesc(), AttnDesc(), LogitsDesc(), LogitsDesc(), nullptr, AttnDesc(),
                                 AttnDesc(), LogitsDesc(), LogitsDesc()),
              ACLNN_SUCCESS);
}

// 非连续输入：走 Contiguous 分支。
TEST_F(l2_cla_gate_backward_test, ascend950_not_contiguous_normal)
{
    const char* layout = "TND";
    TensorDesc gradMerged = TensorDesc(kTndDims, ACL_BF16, ACL_FORMAT_ND, {16384, 512, 1}, 0, {8, 64, 512})
                                .ValueRange(-1, 1);
    EXPECT_EQ(RunClaGateBackward(gradMerged, AttnDesc(), AttnDesc(), LogitsDesc(), LogitsDesc(), layout, AttnDesc(),
                                 AttnDesc(), LogitsDesc(), LogitsDesc()),
              ACLNN_SUCCESS);
}

// FRACTAL_NZ 仅告警不拦截。
TEST_F(l2_cla_gate_backward_test, ascend950_format_nz_warn_only)
{
    const char* layout = "TND";
    EXPECT_EQ(RunClaGateBackward(AttnDesc(ACL_BF16, ACL_FORMAT_FRACTAL_NZ), AttnDesc(), AttnDesc(), LogitsDesc(),
                                 LogitsDesc(), layout, AttnDesc(), AttnDesc(), LogitsDesc(), LogitsDesc()),
              ACLNN_SUCCESS);
}

// ---------------- CheckNotNull：9 路输入 / 输出空指针 ----------------

TEST_F(l2_cla_gate_backward_test, ascend950_exception_null_grad_merged)
{
    const char* layout = "TND";
    auto ut = OP_API_UT(aclnnClaGateBackward,
                        INPUT((aclTensor*)nullptr, AttnDesc(), AttnDesc(), LogitsDesc(), LogitsDesc(), layout),
                        OUTPUT(AttnDesc(), AttnDesc(), LogitsDesc(), LogitsDesc()));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_cla_gate_backward_test, ascend950_exception_null_global_attn)
{
    const char* layout = "TND";
    auto ut = OP_API_UT(aclnnClaGateBackward,
                        INPUT(AttnDesc(), (aclTensor*)nullptr, AttnDesc(), LogitsDesc(), LogitsDesc(), layout),
                        OUTPUT(AttnDesc(), AttnDesc(), LogitsDesc(), LogitsDesc()));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_cla_gate_backward_test, ascend950_exception_null_local_attn)
{
    const char* layout = "TND";
    auto ut = OP_API_UT(aclnnClaGateBackward,
                        INPUT(AttnDesc(), AttnDesc(), (aclTensor*)nullptr, LogitsDesc(), LogitsDesc(), layout),
                        OUTPUT(AttnDesc(), AttnDesc(), LogitsDesc(), LogitsDesc()));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_cla_gate_backward_test, ascend950_exception_null_global_gate_logits)
{
    const char* layout = "TND";
    auto ut = OP_API_UT(aclnnClaGateBackward,
                        INPUT(AttnDesc(), AttnDesc(), AttnDesc(), (aclTensor*)nullptr, LogitsDesc(), layout),
                        OUTPUT(AttnDesc(), AttnDesc(), LogitsDesc(), LogitsDesc()));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_cla_gate_backward_test, ascend950_exception_null_local_gate_logits)
{
    const char* layout = "TND";
    auto ut = OP_API_UT(aclnnClaGateBackward,
                        INPUT(AttnDesc(), AttnDesc(), AttnDesc(), LogitsDesc(), (aclTensor*)nullptr, layout),
                        OUTPUT(AttnDesc(), AttnDesc(), LogitsDesc(), LogitsDesc()));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_cla_gate_backward_test, ascend950_exception_null_grad_global_attn_out)
{
    const char* layout = "TND";
    auto ut = OP_API_UT(aclnnClaGateBackward,
                        INPUT(AttnDesc(), AttnDesc(), AttnDesc(), LogitsDesc(), LogitsDesc(), layout),
                        OUTPUT((aclTensor*)nullptr, AttnDesc(), LogitsDesc(), LogitsDesc()));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_cla_gate_backward_test, ascend950_exception_null_grad_local_attn_out)
{
    const char* layout = "TND";
    auto ut = OP_API_UT(aclnnClaGateBackward,
                        INPUT(AttnDesc(), AttnDesc(), AttnDesc(), LogitsDesc(), LogitsDesc(), layout),
                        OUTPUT(AttnDesc(), (aclTensor*)nullptr, LogitsDesc(), LogitsDesc()));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_cla_gate_backward_test, ascend950_exception_null_grad_global_gate_logits_out)
{
    const char* layout = "TND";
    auto ut = OP_API_UT(aclnnClaGateBackward,
                        INPUT(AttnDesc(), AttnDesc(), AttnDesc(), LogitsDesc(), LogitsDesc(), layout),
                        OUTPUT(AttnDesc(), AttnDesc(), (aclTensor*)nullptr, LogitsDesc()));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_cla_gate_backward_test, ascend950_exception_null_grad_local_gate_logits_out)
{
    const char* layout = "TND";
    auto ut = OP_API_UT(aclnnClaGateBackward,
                        INPUT(AttnDesc(), AttnDesc(), AttnDesc(), LogitsDesc(), LogitsDesc(), layout),
                        OUTPUT(AttnDesc(), AttnDesc(), LogitsDesc(), (aclTensor*)nullptr));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

// ---------------- CheckDtypeValid ----------------

TEST_F(l2_cla_gate_backward_test, ascend950_exception_grad_merged_dtype_unsupported)
{
    const char* layout = "TND";
    EXPECT_EQ(RunClaGateBackward(AttnDesc(ACL_FLOAT), AttnDesc(ACL_FLOAT), AttnDesc(ACL_FLOAT), LogitsDesc(ACL_FLOAT),
                                 LogitsDesc(ACL_FLOAT), layout, AttnDesc(ACL_FLOAT), AttnDesc(ACL_FLOAT),
                                 LogitsDesc(ACL_FLOAT), LogitsDesc(ACL_FLOAT)),
              ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_cla_gate_backward_test, ascend950_exception_global_attn_dtype_unsupported)
{
    const char* layout = "TND";
    EXPECT_EQ(RunClaGateBackward(AttnDesc(), AttnDesc(ACL_FLOAT), AttnDesc(), LogitsDesc(), LogitsDesc(), layout,
                                 AttnDesc(), AttnDesc(), LogitsDesc(), LogitsDesc()),
              ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_cla_gate_backward_test, ascend950_exception_local_attn_dtype_unsupported)
{
    const char* layout = "TND";
    EXPECT_EQ(RunClaGateBackward(AttnDesc(), AttnDesc(), AttnDesc(ACL_FLOAT), LogitsDesc(), LogitsDesc(), layout,
                                 AttnDesc(), AttnDesc(), LogitsDesc(), LogitsDesc()),
              ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_cla_gate_backward_test, ascend950_exception_global_gate_logits_dtype_unsupported)
{
    const char* layout = "TND";
    EXPECT_EQ(RunClaGateBackward(AttnDesc(), AttnDesc(), AttnDesc(), LogitsDesc(ACL_FLOAT), LogitsDesc(), layout,
                                 AttnDesc(), AttnDesc(), LogitsDesc(), LogitsDesc()),
              ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_cla_gate_backward_test, ascend950_exception_local_gate_logits_dtype_unsupported)
{
    const char* layout = "TND";
    EXPECT_EQ(RunClaGateBackward(AttnDesc(), AttnDesc(), AttnDesc(), LogitsDesc(), LogitsDesc(ACL_FLOAT), layout,
                                 AttnDesc(), AttnDesc(), LogitsDesc(), LogitsDesc()),
              ACLNN_ERR_PARAM_INVALID);
}

// 输入 dtype 均受支持但彼此不一致。
TEST_F(l2_cla_gate_backward_test, ascend950_exception_input_dtype_mismatch)
{
    const char* layout = "TND";
    EXPECT_EQ(RunClaGateBackward(AttnDesc(), AttnDesc(ACL_FLOAT16), AttnDesc(), LogitsDesc(), LogitsDesc(), layout,
                                 AttnDesc(), AttnDesc(), LogitsDesc(), LogitsDesc()),
              ACLNN_ERR_PARAM_INVALID);
}

// 输出 dtype 与输入不一致。
TEST_F(l2_cla_gate_backward_test, ascend950_exception_output_dtype_mismatch)
{
    const char* layout = "TND";
    EXPECT_EQ(RunClaGateBackward(AttnDesc(), AttnDesc(), AttnDesc(), LogitsDesc(), LogitsDesc(), layout, AttnDesc(),
                                 AttnDesc(ACL_FLOAT16), LogitsDesc(), LogitsDesc()),
              ACLNN_ERR_PARAM_INVALID);
}

// ---------------- CheckMaxDimension ----------------

TEST_F(l2_cla_gate_backward_test, ascend950_exception_grad_merged_rank_too_high)
{
    const char* layout = "TND";
    TensorDesc gradTooHigh = TensorDesc({1, 1, 1, 1, 1, 1, 1, 64, 256}, ACL_BF16, ACL_FORMAT_ND);
    EXPECT_EQ(RunClaGateBackward(gradTooHigh, AttnDesc(), AttnDesc(), LogitsDesc(), LogitsDesc(), layout, AttnDesc(),
                                 AttnDesc(), LogitsDesc(), LogitsDesc()),
              ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_cla_gate_backward_test, ascend950_exception_global_attn_rank_too_high)
{
    const char* layout = "TND";
    TensorDesc attnTooHigh = TensorDesc({1, 1, 1, 1, 1, 1, 8, 64, 256}, ACL_BF16, ACL_FORMAT_ND);
    EXPECT_EQ(RunClaGateBackward(AttnDesc(), attnTooHigh, AttnDesc(), LogitsDesc(), LogitsDesc(), layout, AttnDesc(),
                                 AttnDesc(), LogitsDesc(), LogitsDesc()),
              ACLNN_ERR_PARAM_INVALID);
}

// ---------------- CheckShapeValid ----------------

// gradMerged 必须为 3D。
TEST_F(l2_cla_gate_backward_test, ascend950_exception_grad_merged_not_3d)
{
    const char* layout = "TND";
    TensorDesc grad2d = TensorDesc({8, 16384}, ACL_BF16, ACL_FORMAT_ND);
    EXPECT_EQ(RunClaGateBackward(grad2d, AttnDesc(), AttnDesc(), LogitsDesc(), LogitsDesc(), layout, AttnDesc(),
                                 AttnDesc(), LogitsDesc(), LogitsDesc()),
              ACLNN_ERR_PARAM_INVALID);
}

// globalAttn 与 gradMerged shape 不一致。
TEST_F(l2_cla_gate_backward_test, ascend950_exception_global_attn_shape_mismatch)
{
    const char* layout = "TND";
    TensorDesc globalAttn = TensorDesc({8, 64, 128}, ACL_BF16, ACL_FORMAT_ND);
    EXPECT_EQ(RunClaGateBackward(AttnDesc(), globalAttn, AttnDesc(), LogitsDesc(), LogitsDesc(), layout, AttnDesc(),
                                 AttnDesc(), LogitsDesc(), LogitsDesc()),
              ACLNN_ERR_PARAM_INVALID);
}

// localAttn 与 gradMerged shape 不一致。
TEST_F(l2_cla_gate_backward_test, ascend950_exception_local_attn_shape_mismatch)
{
    const char* layout = "TND";
    TensorDesc localAttn = TensorDesc({8, 32, 256}, ACL_BF16, ACL_FORMAT_ND);
    EXPECT_EQ(RunClaGateBackward(AttnDesc(), AttnDesc(), localAttn, LogitsDesc(), LogitsDesc(), layout, AttnDesc(),
                                 AttnDesc(), LogitsDesc(), LogitsDesc()),
              ACLNN_ERR_PARAM_INVALID);
}

// gate logits 必须为 2D。
TEST_F(l2_cla_gate_backward_test, ascend950_exception_global_gate_logits_not_2d)
{
    const char* layout = "TND";
    TensorDesc logits3d = TensorDesc({8, 64, 1}, ACL_BF16, ACL_FORMAT_ND);
    EXPECT_EQ(RunClaGateBackward(AttnDesc(), AttnDesc(), AttnDesc(), logits3d, logits3d, layout, AttnDesc(), AttnDesc(),
                                 logits3d, logits3d),
              ACLNN_ERR_PARAM_INVALID);
}

// globalGateLogits 与 localGateLogits shape 不一致。
TEST_F(l2_cla_gate_backward_test, ascend950_exception_gate_logits_shape_mismatch)
{
    const char* layout = "TND";
    TensorDesc localLogits = TensorDesc({8, 32}, ACL_BF16, ACL_FORMAT_ND);
    EXPECT_EQ(RunClaGateBackward(AttnDesc(), AttnDesc(), AttnDesc(), LogitsDesc(), localLogits, layout, AttnDesc(),
                                 AttnDesc(), LogitsDesc(), localLogits),
              ACLNN_ERR_PARAM_INVALID);
}

// gradGlobalAttnOut shape 与 gradMerged 不一致。
TEST_F(l2_cla_gate_backward_test, ascend950_exception_grad_global_attn_out_shape_mismatch)
{
    const char* layout = "TND";
    TensorDesc badOut = TensorDesc({8, 64, 128}, ACL_BF16, ACL_FORMAT_ND);
    EXPECT_EQ(RunClaGateBackward(AttnDesc(), AttnDesc(), AttnDesc(), LogitsDesc(), LogitsDesc(), layout, badOut,
                                 AttnDesc(), LogitsDesc(), LogitsDesc()),
              ACLNN_ERR_PARAM_INVALID);
}

// gradLocalAttnOut shape 与 gradMerged 不一致。
TEST_F(l2_cla_gate_backward_test, ascend950_exception_grad_local_attn_out_shape_mismatch)
{
    const char* layout = "TND";
    TensorDesc badOut = TensorDesc({8, 64, 128}, ACL_BF16, ACL_FORMAT_ND);
    EXPECT_EQ(RunClaGateBackward(AttnDesc(), AttnDesc(), AttnDesc(), LogitsDesc(), LogitsDesc(), layout, AttnDesc(),
                                 badOut, LogitsDesc(), LogitsDesc()),
              ACLNN_ERR_PARAM_INVALID);
}

// gradGlobalGateLogitsOut shape 与 globalGateLogits 不一致。
TEST_F(l2_cla_gate_backward_test, ascend950_exception_grad_global_gate_logits_out_shape_mismatch)
{
    const char* layout = "TND";
    TensorDesc badOut = TensorDesc({8, 32}, ACL_BF16, ACL_FORMAT_ND);
    EXPECT_EQ(RunClaGateBackward(AttnDesc(), AttnDesc(), AttnDesc(), LogitsDesc(), LogitsDesc(), layout, AttnDesc(),
                                 AttnDesc(), badOut, LogitsDesc()),
              ACLNN_ERR_PARAM_INVALID);
}

// gradLocalGateLogitsOut shape 与 localGateLogits 不一致。
TEST_F(l2_cla_gate_backward_test, ascend950_exception_grad_local_gate_logits_out_shape_mismatch)
{
    const char* layout = "TND";
    TensorDesc badOut = TensorDesc({8, 32}, ACL_BF16, ACL_FORMAT_ND);
    EXPECT_EQ(RunClaGateBackward(AttnDesc(), AttnDesc(), AttnDesc(), LogitsDesc(), LogitsDesc(), layout, AttnDesc(),
                                 AttnDesc(), LogitsDesc(), badOut),
              ACLNN_ERR_PARAM_INVALID);
}

// ---------------- 第二段接口 ----------------

TEST_F(l2_cla_gate_backward_test, ascend950_phase2_null_executor)
{
    aclnnStatus aclRet = aclnnClaGateBackward(nullptr, 0, nullptr, nullptr);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}
