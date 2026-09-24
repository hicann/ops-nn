/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * aclnnThnnFusedGruCell L2 UT：
 *   单步 GRU 门控融合计算。输入 inputGates/hiddenGates (B,3H)、hx (B,H)
 *   及可选 inputBias/hiddenBias (3H,)，输出 hy (B,H) 与 storage (B,5H)。
 *   算子仅支持 RegBase 平台（如 Ascend950），正例经 SocVersionManager
 *   mock 为 ASCEND950；负例覆盖空指针 / dtype / shape 契约。
 */

#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "opdev/op_log.h"
#include "../../../op_api/aclnn_thnn_fused_gru_cell.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_thnn_fused_gru_cell_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "l2_thnn_fused_gru_cell_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "l2_thnn_fused_gru_cell_test TearDown" << std::endl; }
};

// ==================== 正例 ====================

// 正例1: fp32，双 bias 在位
TEST_F(l2_thnn_fused_gru_cell_test, ascend950_normal_float_with_bias)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t B = 3;
    int64_t H = 5;
    vector<int64_t> gatesShape = {B, 3 * H};
    vector<int64_t> commonShape = {B, H};
    vector<int64_t> biasShape = {3 * H};
    vector<int64_t> storageShape = {B, 5 * H};

    auto inputGates = TensorDesc(gatesShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gatesShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hx = TensorDesc(commonShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputBias = TensorDesc(biasShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenBias = TensorDesc(biasShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hyOut = TensorDesc(commonShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storageOut = TensorDesc(storageShape, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedGruCell, INPUT(inputGates, hiddenGates, hx, inputBias, hiddenBias),
                        OUTPUT(hyOut, storageOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正例2: fp16，双 bias 缺省（nullptr ≡ 全零 bias）
TEST_F(l2_thnn_fused_gru_cell_test, ascend950_normal_float16_no_bias)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t B = 4;
    int64_t H = 8;
    vector<int64_t> gatesShape = {B, 3 * H};
    vector<int64_t> commonShape = {B, H};
    vector<int64_t> storageShape = {B, 5 * H};

    auto inputGates = TensorDesc(gatesShape, ACL_FLOAT16, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gatesShape, ACL_FLOAT16, ACL_FORMAT_ND);
    auto hx = TensorDesc(commonShape, ACL_FLOAT16, ACL_FORMAT_ND);
    auto hyOut = TensorDesc(commonShape, ACL_FLOAT16, ACL_FORMAT_ND);
    auto storageOut = TensorDesc(storageShape, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedGruCell, INPUT(inputGates, hiddenGates, hx, nullptr, nullptr),
                        OUTPUT(hyOut, storageOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正例3: bf16，仅 inputBias 在位（bias 非对称组合）
TEST_F(l2_thnn_fused_gru_cell_test, ascend950_normal_bfloat16_input_bias_only)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t B = 2;
    int64_t H = 16;
    vector<int64_t> gatesShape = {B, 3 * H};
    vector<int64_t> commonShape = {B, H};
    vector<int64_t> biasShape = {3 * H};
    vector<int64_t> storageShape = {B, 5 * H};

    auto inputGates = TensorDesc(gatesShape, ACL_BF16, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gatesShape, ACL_BF16, ACL_FORMAT_ND);
    auto hx = TensorDesc(commonShape, ACL_BF16, ACL_FORMAT_ND);
    auto inputBias = TensorDesc(biasShape, ACL_BF16, ACL_FORMAT_ND);
    auto hyOut = TensorDesc(commonShape, ACL_BF16, ACL_FORMAT_ND);
    auto storageOut = TensorDesc(storageShape, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedGruCell, INPUT(inputGates, hiddenGates, hx, inputBias, nullptr),
                        OUTPUT(hyOut, storageOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正例4: fp32 较大 shape（多核切分形态）
TEST_F(l2_thnn_fused_gru_cell_test, ascend950_normal_float_large_shape)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t B = 64;
    int64_t H = 256;
    vector<int64_t> gatesShape = {B, 3 * H};
    vector<int64_t> commonShape = {B, H};
    vector<int64_t> biasShape = {3 * H};
    vector<int64_t> storageShape = {B, 5 * H};

    auto inputGates = TensorDesc(gatesShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gatesShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hx = TensorDesc(commonShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputBias = TensorDesc(biasShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenBias = TensorDesc(biasShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hyOut = TensorDesc(commonShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storageOut = TensorDesc(storageShape, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedGruCell, INPUT(inputGates, hiddenGates, hx, inputBias, hiddenBias),
                        OUTPUT(hyOut, storageOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正例5: 空 Tensor（B == 0）合法，直接返回空输出不下发 kernel
TEST_F(l2_thnn_fused_gru_cell_test, ascend950_normal_empty_batch)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t B = 0;
    int64_t H = 8;
    vector<int64_t> gatesShape = {B, 3 * H};
    vector<int64_t> commonShape = {B, H};
    vector<int64_t> storageShape = {B, 5 * H};

    auto inputGates = TensorDesc(gatesShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gatesShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hx = TensorDesc(commonShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hyOut = TensorDesc(commonShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storageOut = TensorDesc(storageShape, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedGruCell, INPUT(inputGates, hiddenGates, hx, nullptr, nullptr),
                        OUTPUT(hyOut, storageOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_EQ(workspaceSize, 0UL);
}

// ==================== 负例 ====================

// 负例1: 非 RegBase 平台（默认 Ascend910B）应被平台校验拦截
TEST_F(l2_thnn_fused_gru_cell_test, ascend910b_not_regbase_rejected)
{
    int64_t B = 3;
    int64_t H = 5;
    vector<int64_t> gatesShape = {B, 3 * H};
    vector<int64_t> commonShape = {B, H};
    vector<int64_t> storageShape = {B, 5 * H};

    auto inputGates = TensorDesc(gatesShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gatesShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hx = TensorDesc(commonShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hyOut = TensorDesc(commonShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storageOut = TensorDesc(storageShape, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedGruCell, INPUT(inputGates, hiddenGates, hx, nullptr, nullptr),
                        OUTPUT(hyOut, storageOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 负例2: 空 inputGates 指针，期望 ACLNN_ERR_PARAM_NULLPTR
TEST_F(l2_thnn_fused_gru_cell_test, ascend950_null_input_gates)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t B = 3;
    int64_t H = 5;
    vector<int64_t> gatesShape = {B, 3 * H};
    vector<int64_t> commonShape = {B, H};
    vector<int64_t> storageShape = {B, 5 * H};

    auto hiddenGates = TensorDesc(gatesShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hx = TensorDesc(commonShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hyOut = TensorDesc(commonShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storageOut = TensorDesc(storageShape, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedGruCell, INPUT(nullptr, hiddenGates, hx, nullptr, nullptr),
                        OUTPUT(hyOut, storageOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 负例3: 空 hx 指针，期望 ACLNN_ERR_PARAM_NULLPTR
TEST_F(l2_thnn_fused_gru_cell_test, ascend950_null_hx)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t B = 3;
    int64_t H = 5;
    vector<int64_t> gatesShape = {B, 3 * H};
    vector<int64_t> commonShape = {B, H};
    vector<int64_t> storageShape = {B, 5 * H};

    auto inputGates = TensorDesc(gatesShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gatesShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hyOut = TensorDesc(commonShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storageOut = TensorDesc(storageShape, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedGruCell, INPUT(inputGates, hiddenGates, nullptr, nullptr, nullptr),
                        OUTPUT(hyOut, storageOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 负例4: 空 hyOut 指针，期望 ACLNN_ERR_PARAM_NULLPTR
TEST_F(l2_thnn_fused_gru_cell_test, ascend950_null_hy_out)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t B = 3;
    int64_t H = 5;
    vector<int64_t> gatesShape = {B, 3 * H};
    vector<int64_t> commonShape = {B, H};
    vector<int64_t> storageShape = {B, 5 * H};

    auto inputGates = TensorDesc(gatesShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gatesShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hx = TensorDesc(commonShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storageOut = TensorDesc(storageShape, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedGruCell, INPUT(inputGates, hiddenGates, hx, nullptr, nullptr),
                        OUTPUT(nullptr, storageOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 负例5: 空 storageOut 指针，期望 ACLNN_ERR_PARAM_NULLPTR
TEST_F(l2_thnn_fused_gru_cell_test, ascend950_null_storage_out)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t B = 3;
    int64_t H = 5;
    vector<int64_t> gatesShape = {B, 3 * H};
    vector<int64_t> commonShape = {B, H};

    auto inputGates = TensorDesc(gatesShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gatesShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hx = TensorDesc(commonShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hyOut = TensorDesc(commonShape, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedGruCell, INPUT(inputGates, hiddenGates, hx, nullptr, nullptr),
                        OUTPUT(hyOut, nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 负例6: 不支持的 dtype（INT32），期望 ACLNN_ERR_PARAM_INVALID
TEST_F(l2_thnn_fused_gru_cell_test, ascend950_dtype_int32_not_support)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t B = 3;
    int64_t H = 5;
    vector<int64_t> gatesShape = {B, 3 * H};
    vector<int64_t> commonShape = {B, H};
    vector<int64_t> storageShape = {B, 5 * H};

    auto inputGates = TensorDesc(gatesShape, ACL_INT32, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gatesShape, ACL_INT32, ACL_FORMAT_ND);
    auto hx = TensorDesc(commonShape, ACL_INT32, ACL_FORMAT_ND);
    auto hyOut = TensorDesc(commonShape, ACL_INT32, ACL_FORMAT_ND);
    auto storageOut = TensorDesc(storageShape, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedGruCell, INPUT(inputGates, hiddenGates, hx, nullptr, nullptr),
                        OUTPUT(hyOut, storageOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 负例7: hiddenGates 与 inputGates dtype 不一致，期望 ACLNN_ERR_PARAM_INVALID
TEST_F(l2_thnn_fused_gru_cell_test, ascend950_dtype_mismatch_hidden_gates)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t B = 3;
    int64_t H = 5;
    vector<int64_t> gatesShape = {B, 3 * H};
    vector<int64_t> commonShape = {B, H};
    vector<int64_t> storageShape = {B, 5 * H};

    auto inputGates = TensorDesc(gatesShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gatesShape, ACL_FLOAT16, ACL_FORMAT_ND);
    auto hx = TensorDesc(commonShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hyOut = TensorDesc(commonShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storageOut = TensorDesc(storageShape, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedGruCell, INPUT(inputGates, hiddenGates, hx, nullptr, nullptr),
                        OUTPUT(hyOut, storageOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 负例8: hyOut dtype 与输入不一致，期望 ACLNN_ERR_PARAM_INVALID
TEST_F(l2_thnn_fused_gru_cell_test, ascend950_dtype_mismatch_hy_out)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t B = 3;
    int64_t H = 5;
    vector<int64_t> gatesShape = {B, 3 * H};
    vector<int64_t> commonShape = {B, H};
    vector<int64_t> storageShape = {B, 5 * H};

    auto inputGates = TensorDesc(gatesShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gatesShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hx = TensorDesc(commonShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hyOut = TensorDesc(commonShape, ACL_FLOAT16, ACL_FORMAT_ND);
    auto storageOut = TensorDesc(storageShape, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedGruCell, INPUT(inputGates, hiddenGates, hx, nullptr, nullptr),
                        OUTPUT(hyOut, storageOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 负例9: inputGates 维度不足（1 维），期望 ACLNN_ERR_PARAM_INVALID
TEST_F(l2_thnn_fused_gru_cell_test, ascend950_input_gates_dim_invalid)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t B = 3;
    int64_t H = 5;
    vector<int64_t> gatesShape = {B * 3 * H}; // 期望 2 维 (B, 3H)，构造 1 维触发维度校验
    vector<int64_t> commonShape = {B, H};
    vector<int64_t> storageShape = {B, 5 * H};

    auto inputGates = TensorDesc(gatesShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gatesShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hx = TensorDesc(commonShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hyOut = TensorDesc(commonShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storageOut = TensorDesc(storageShape, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedGruCell, INPUT(inputGates, hiddenGates, hx, nullptr, nullptr),
                        OUTPUT(hyOut, storageOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 负例10: inputGates 列数 != 3H，期望 ACLNN_ERR_PARAM_INVALID
TEST_F(l2_thnn_fused_gru_cell_test, ascend950_input_gates_cols_not_3h)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t B = 3;
    int64_t H = 5;
    vector<int64_t> gatesShape = {B, 4 * H}; // 期望 (B, 3H)
    vector<int64_t> commonShape = {B, H};
    vector<int64_t> storageShape = {B, 5 * H};

    auto inputGates = TensorDesc(gatesShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gatesShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hx = TensorDesc(commonShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hyOut = TensorDesc(commonShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storageOut = TensorDesc(storageShape, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedGruCell, INPUT(inputGates, hiddenGates, hx, nullptr, nullptr),
                        OUTPUT(hyOut, storageOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 负例11: hiddenGates shape != inputGates shape，期望 ACLNN_ERR_PARAM_INVALID
TEST_F(l2_thnn_fused_gru_cell_test, ascend950_hidden_gates_shape_mismatch)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t B = 3;
    int64_t H = 5;
    vector<int64_t> gatesShape = {B, 3 * H};
    vector<int64_t> commonShape = {B, H};
    vector<int64_t> storageShape = {B, 5 * H};

    auto inputGates = TensorDesc(gatesShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc({B + 1, 3 * H}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hx = TensorDesc(commonShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hyOut = TensorDesc(commonShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storageOut = TensorDesc(storageShape, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedGruCell, INPUT(inputGates, hiddenGates, hx, nullptr, nullptr),
                        OUTPUT(hyOut, storageOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 负例12: hyOut shape != (B, H)，期望 ACLNN_ERR_PARAM_INVALID
TEST_F(l2_thnn_fused_gru_cell_test, ascend950_hy_out_shape_mismatch)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t B = 3;
    int64_t H = 5;
    vector<int64_t> gatesShape = {B, 3 * H};
    vector<int64_t> commonShape = {B, H};
    vector<int64_t> storageShape = {B, 5 * H};

    auto inputGates = TensorDesc(gatesShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gatesShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hx = TensorDesc(commonShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hyOut = TensorDesc({B, 2 * H}, ACL_FLOAT, ACL_FORMAT_ND); // 期望 (B, H)
    auto storageOut = TensorDesc(storageShape, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedGruCell, INPUT(inputGates, hiddenGates, hx, nullptr, nullptr),
                        OUTPUT(hyOut, storageOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 负例13: storageOut shape != (B, 5H)，期望 ACLNN_ERR_PARAM_INVALID
TEST_F(l2_thnn_fused_gru_cell_test, ascend950_storage_out_shape_mismatch)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t B = 3;
    int64_t H = 5;
    vector<int64_t> gatesShape = {B, 3 * H};
    vector<int64_t> commonShape = {B, H};

    auto inputGates = TensorDesc(gatesShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gatesShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hx = TensorDesc(commonShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hyOut = TensorDesc(commonShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storageOut = TensorDesc({B, 4 * H}, ACL_FLOAT, ACL_FORMAT_ND); // 期望 (B, 5H)

    auto ut = OP_API_UT(aclnnThnnFusedGruCell, INPUT(inputGates, hiddenGates, hx, nullptr, nullptr),
                        OUTPUT(hyOut, storageOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 负例14: bias numel != 3H，期望 ACLNN_ERR_PARAM_INVALID
TEST_F(l2_thnn_fused_gru_cell_test, ascend950_bias_numel_mismatch)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t B = 3;
    int64_t H = 5;
    vector<int64_t> gatesShape = {B, 3 * H};
    vector<int64_t> commonShape = {B, H};
    vector<int64_t> storageShape = {B, 5 * H};

    auto inputGates = TensorDesc(gatesShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gatesShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hx = TensorDesc(commonShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputBias = TensorDesc({3 * H + 1}, ACL_FLOAT, ACL_FORMAT_ND); // 期望 numel == 3H
    auto hyOut = TensorDesc(commonShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storageOut = TensorDesc(storageShape, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedGruCell, INPUT(inputGates, hiddenGates, hx, inputBias, nullptr),
                        OUTPUT(hyOut, storageOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 负例15: bias 维度错误（2 维），期望 ACLNN_ERR_PARAM_INVALID
TEST_F(l2_thnn_fused_gru_cell_test, ascend950_bias_dim_invalid)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND950);
    int64_t B = 3;
    int64_t H = 5;
    vector<int64_t> gatesShape = {B, 3 * H};
    vector<int64_t> commonShape = {B, H};
    vector<int64_t> storageShape = {B, 5 * H};

    auto inputGates = TensorDesc(gatesShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gatesShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hx = TensorDesc(commonShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputBias = TensorDesc({1, 3 * H}, ACL_FLOAT, ACL_FORMAT_ND); // 期望 rank 1
    auto hyOut = TensorDesc(commonShape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storageOut = TensorDesc(storageShape, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnThnnFusedGruCell, INPUT(inputGates, hiddenGates, hx, inputBias, nullptr),
                        OUTPUT(hyOut, storageOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
