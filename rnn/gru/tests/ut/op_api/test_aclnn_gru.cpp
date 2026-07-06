/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "op_host/op_api/aclnn_gru.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class GruApiTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ==================== 正例 ====================

// 正例1: 单层单向FP16训练模式，无偏置，无hx，batch_first=false
TEST_F(GruApiTest, case_single_layer_unidirectional_fp16_train_no_bias)
{
    int64_t T = 2;
    int64_t B = 3;
    int64_t I = 4;
    int64_t H = 5;
    int64_t gateNum = 3;
    int64_t numLayers = 1;
    int64_t dScale = 1;
    int64_t ldScale = numLayers * dScale;

    auto input = TensorDesc({T, B, I}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto wi = TensorDesc({gateNum * H, I}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto wh = TensorDesc({gateNum * H, H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto params = TensorListDesc({wi, wh});

    auto output = TensorDesc({T, B, dScale * H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto hy = TensorDesc({ldScale, B, H}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto rOut = TensorListDesc(ldScale, TensorDesc({T, B, H}, ACL_FLOAT16, ACL_FORMAT_ND));
    auto zOut = TensorListDesc(ldScale, TensorDesc({T, B, H}, ACL_FLOAT16, ACL_FORMAT_ND));
    auto nOut = TensorListDesc(ldScale, TensorDesc({T, B, H}, ACL_FLOAT16, ACL_FORMAT_ND));
    auto hnOut = TensorListDesc(ldScale, TensorDesc({T, B, H}, ACL_FLOAT16, ACL_FORMAT_ND));
    auto hOut = TensorListDesc(ldScale, TensorDesc({T, B, H}, ACL_FLOAT16, ACL_FORMAT_ND));

    auto ut = OP_API_UT(aclnnGRU, INPUT(input, params, nullptr, nullptr),
                        OUTPUT(output, hy, rOut, zOut, nOut, hnOut, hOut), false, numLayers, 0.0, true, false, false);

    uint64_t workspaceSize = 0;
    aclnnStatus ret = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
}

// 正例2: 单层单向FP32训练模式，有偏置，有hx
TEST_F(GruApiTest, case_single_layer_unidirectional_fp32_train_with_bias_hx)
{
    int64_t T = 3;
    int64_t B = 2;
    int64_t I = 8;
    int64_t H = 16;
    int64_t gateNum = 3;
    int64_t numLayers = 1;
    int64_t dScale = 1;
    int64_t ldScale = numLayers * dScale;

    auto input = TensorDesc({T, B, I}, ACL_FLOAT, ACL_FORMAT_ND);
    auto wi = TensorDesc({gateNum * H, I}, ACL_FLOAT, ACL_FORMAT_ND);
    auto wh = TensorDesc({gateNum * H, H}, ACL_FLOAT, ACL_FORMAT_ND);
    auto bi = TensorDesc({gateNum * H}, ACL_FLOAT, ACL_FORMAT_ND);
    auto bh = TensorDesc({gateNum * H}, ACL_FLOAT, ACL_FORMAT_ND);
    auto params = TensorListDesc({wi, wh, bi, bh});

    auto hx = TensorDesc({ldScale, B, H}, ACL_FLOAT, ACL_FORMAT_ND);

    auto output = TensorDesc({T, B, dScale * H}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hy = TensorDesc({ldScale, B, H}, ACL_FLOAT, ACL_FORMAT_ND);

    auto rOut = TensorListDesc(ldScale, TensorDesc({T, B, H}, ACL_FLOAT, ACL_FORMAT_ND));
    auto zOut = TensorListDesc(ldScale, TensorDesc({T, B, H}, ACL_FLOAT, ACL_FORMAT_ND));
    auto nOut = TensorListDesc(ldScale, TensorDesc({T, B, H}, ACL_FLOAT, ACL_FORMAT_ND));
    auto hnOut = TensorListDesc(ldScale, TensorDesc({T, B, H}, ACL_FLOAT, ACL_FORMAT_ND));
    auto hOut = TensorListDesc(ldScale, TensorDesc({T, B, H}, ACL_FLOAT, ACL_FORMAT_ND));

    auto ut = OP_API_UT(aclnnGRU, INPUT(input, params, hx, nullptr), OUTPUT(output, hy, rOut, zOut, nOut, hnOut, hOut),
                        true, numLayers, 0.0, true, false, false);

    uint64_t workspaceSize = 0;
    aclnnStatus ret = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
}

// 正例3: 单层单向FP16推理模式，无偏置，无hx
TEST_F(GruApiTest, case_single_layer_unidirectional_fp16_inference)
{
    int64_t T = 2;
    int64_t B = 3;
    int64_t I = 4;
    int64_t H = 5;
    int64_t gateNum = 3;
    int64_t numLayers = 1;
    int64_t dScale = 1;
    int64_t ldScale = numLayers * dScale;

    auto input = TensorDesc({T, B, I}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto wi = TensorDesc({gateNum * H, I}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto wh = TensorDesc({gateNum * H, H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto params = TensorListDesc({wi, wh});

    auto output = TensorDesc({T, B, dScale * H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto hy = TensorDesc({ldScale, B, H}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGRU, INPUT(input, params, nullptr, nullptr),
                        OUTPUT(output, hy, nullptr, nullptr, nullptr, nullptr, nullptr), false, numLayers, 0.0, false,
                        false, false);

    uint64_t workspaceSize = 0;
    aclnnStatus ret = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
}

// 正例4: 双层双向FP16训练模式，有偏置，有hx，batch_first=true
TEST_F(GruApiTest, case_double_layer_bidirectional_fp16_train_batch_first)
{
    int64_t T = 4;
    int64_t B = 3;
    int64_t I = 8;
    int64_t H = 16;
    int64_t gateNum = 3;
    int64_t numLayers = 2;
    int64_t dScale = 2;
    int64_t ldScale = numLayers * dScale;
    int64_t bScale = 2;

    auto input = TensorDesc({B, T, I}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto wi = TensorDesc({gateNum * H, I}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto wh = TensorDesc({gateNum * H, H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto bi = TensorDesc({gateNum * H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto bh = TensorDesc({gateNum * H}, ACL_FLOAT16, ACL_FORMAT_ND);

    // 2层双向，每层4个参数(wi, wh, bi, bh)，共 2*2*4 = 8 个参数
    auto params = TensorListDesc({
        wi, wh, bi, bh,                                                    // L0-F
        wi, wh, bi, bh,                                                    // L0-B
        TensorDesc({gateNum * H, dScale * H}, ACL_FLOAT16, ACL_FORMAT_ND), // wi L1-F
        TensorDesc({gateNum * H, H}, ACL_FLOAT16, ACL_FORMAT_ND),          // wh L1-F
        TensorDesc({gateNum * H}, ACL_FLOAT16, ACL_FORMAT_ND),             // bi L1-F
        TensorDesc({gateNum * H}, ACL_FLOAT16, ACL_FORMAT_ND),             // bh L1-F
        TensorDesc({gateNum * H, dScale * H}, ACL_FLOAT16, ACL_FORMAT_ND), // wi L1-B
        TensorDesc({gateNum * H, H}, ACL_FLOAT16, ACL_FORMAT_ND),          // wh L1-B
        TensorDesc({gateNum * H}, ACL_FLOAT16, ACL_FORMAT_ND),             // bi L1-B
        TensorDesc({gateNum * H}, ACL_FLOAT16, ACL_FORMAT_ND),             // bh L1-B
    });

    auto hx = TensorDesc({ldScale, B, H}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto output = TensorDesc({B, T, dScale * H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto hy = TensorDesc({ldScale, B, H}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto rOut = TensorListDesc(ldScale, TensorDesc({T, B, H}, ACL_FLOAT16, ACL_FORMAT_ND));
    auto zOut = TensorListDesc(ldScale, TensorDesc({T, B, H}, ACL_FLOAT16, ACL_FORMAT_ND));
    auto nOut = TensorListDesc(ldScale, TensorDesc({T, B, H}, ACL_FLOAT16, ACL_FORMAT_ND));
    auto hnOut = TensorListDesc(ldScale, TensorDesc({T, B, H}, ACL_FLOAT16, ACL_FORMAT_ND));
    auto hOut = TensorListDesc(ldScale, TensorDesc({T, B, H}, ACL_FLOAT16, ACL_FORMAT_ND));

    auto ut = OP_API_UT(aclnnGRU, INPUT(input, params, hx, nullptr), OUTPUT(output, hy, rOut, zOut, nOut, hnOut, hOut),
                        true, numLayers, 0.0, true, true, true);

    uint64_t workspaceSize = 0;
    aclnnStatus ret = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
}

// 正例5: 单层单向FP32推理模式，有偏置，无hx
TEST_F(GruApiTest, case_single_layer_unidirectional_fp32_inference_with_bias)
{
    int64_t T = 2;
    int64_t B = 4;
    int64_t I = 6;
    int64_t H = 12;
    int64_t gateNum = 3;
    int64_t numLayers = 1;
    int64_t dScale = 1;
    int64_t ldScale = numLayers * dScale;

    auto input = TensorDesc({T, B, I}, ACL_FLOAT, ACL_FORMAT_ND);
    auto wi = TensorDesc({gateNum * H, I}, ACL_FLOAT, ACL_FORMAT_ND);
    auto wh = TensorDesc({gateNum * H, H}, ACL_FLOAT, ACL_FORMAT_ND);
    auto bi = TensorDesc({gateNum * H}, ACL_FLOAT, ACL_FORMAT_ND);
    auto bh = TensorDesc({gateNum * H}, ACL_FLOAT, ACL_FORMAT_ND);
    auto params = TensorListDesc({wi, wh, bi, bh});

    auto output = TensorDesc({T, B, dScale * H}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hy = TensorDesc({ldScale, B, H}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGRU, INPUT(input, params, nullptr, nullptr),
                        OUTPUT(output, hy, nullptr, nullptr, nullptr, nullptr, nullptr), true, numLayers, 0.0, false,
                        false, false);

    uint64_t workspaceSize = 0;
    aclnnStatus ret = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
}

// 正例6: 单层单向FP16训练模式，无偏置，有hx
TEST_F(GruApiTest, case_single_layer_unidirectional_fp16_train_with_hx_no_bias)
{
    int64_t T = 3;
    int64_t B = 2;
    int64_t I = 8;
    int64_t H = 12;
    int64_t gateNum = 3;
    int64_t numLayers = 1;
    int64_t dScale = 1;
    int64_t ldScale = numLayers * dScale;

    auto input = TensorDesc({T, B, I}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto wi = TensorDesc({gateNum * H, I}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto wh = TensorDesc({gateNum * H, H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto params = TensorListDesc({wi, wh});

    auto hx = TensorDesc({ldScale, B, H}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto output = TensorDesc({T, B, dScale * H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto hy = TensorDesc({ldScale, B, H}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto rOut = TensorListDesc(ldScale, TensorDesc({T, B, H}, ACL_FLOAT16, ACL_FORMAT_ND));
    auto zOut = TensorListDesc(ldScale, TensorDesc({T, B, H}, ACL_FLOAT16, ACL_FORMAT_ND));
    auto nOut = TensorListDesc(ldScale, TensorDesc({T, B, H}, ACL_FLOAT16, ACL_FORMAT_ND));
    auto hnOut = TensorListDesc(ldScale, TensorDesc({T, B, H}, ACL_FLOAT16, ACL_FORMAT_ND));
    auto hOut = TensorListDesc(ldScale, TensorDesc({T, B, H}, ACL_FLOAT16, ACL_FORMAT_ND));

    auto ut = OP_API_UT(aclnnGRU, INPUT(input, params, hx, nullptr), OUTPUT(output, hy, rOut, zOut, nOut, hnOut, hOut),
                        false, numLayers, 0.0, true, false, false);

    uint64_t workspaceSize = 0;
    aclnnStatus ret = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
}

// ==================== 负例 ====================

// 负例1: null input，期望 ACLNN_ERR_PARAM_NULLPTR
TEST_F(GruApiTest, case_null_input)
{
    int64_t T = 2;
    int64_t B = 3;
    int64_t I = 4;
    int64_t H = 5;
    int64_t gateNum = 3;
    int64_t numLayers = 1;
    int64_t dScale = 1;
    int64_t ldScale = numLayers * dScale;

    auto wi = TensorDesc({gateNum * H, I}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto wh = TensorDesc({gateNum * H, H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto params = TensorListDesc({wi, wh});

    auto output = TensorDesc({T, B, dScale * H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto hy = TensorDesc({ldScale, B, H}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGRU, INPUT(nullptr, params, nullptr, nullptr),
                        OUTPUT(output, hy, nullptr, nullptr, nullptr, nullptr, nullptr), false, numLayers, 0.0, false,
                        false, false);

    uint64_t workspaceSize = 0;
    aclnnStatus ret = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
}

// 负例2: null params，期望 ACLNN_ERR_PARAM_NULLPTR
TEST_F(GruApiTest, case_null_params)
{
    int64_t T = 2;
    int64_t B = 3;
    int64_t I = 4;
    int64_t H = 5;
    int64_t numLayers = 1;
    int64_t dScale = 1;
    int64_t ldScale = numLayers * dScale;

    auto input = TensorDesc({T, B, I}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto output = TensorDesc({T, B, dScale * H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto hy = TensorDesc({ldScale, B, H}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGRU, INPUT(input, nullptr, nullptr, nullptr),
                        OUTPUT(output, hy, nullptr, nullptr, nullptr, nullptr, nullptr), false, numLayers, 0.0, false,
                        false, false);

    uint64_t workspaceSize = 0;
    aclnnStatus ret = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
}

// 负例3: null output，期望 ACLNN_ERR_PARAM_NULLPTR
TEST_F(GruApiTest, case_null_output)
{
    int64_t T = 2;
    int64_t B = 3;
    int64_t I = 4;
    int64_t H = 5;
    int64_t gateNum = 3;
    int64_t numLayers = 1;
    int64_t ldScale = numLayers;

    auto input = TensorDesc({T, B, I}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto wi = TensorDesc({gateNum * H, I}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto wh = TensorDesc({gateNum * H, H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto params = TensorListDesc({wi, wh});

    auto hy = TensorDesc({ldScale, B, H}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGRU, INPUT(input, params, nullptr, nullptr),
                        OUTPUT(nullptr, hy, nullptr, nullptr, nullptr, nullptr, nullptr), false, numLayers, 0.0, false,
                        false, false);

    uint64_t workspaceSize = 0;
    aclnnStatus ret = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
}

// 负例4: null hy，期望 ACLNN_ERR_PARAM_NULLPTR
TEST_F(GruApiTest, case_null_hy)
{
    int64_t T = 2;
    int64_t B = 3;
    int64_t I = 4;
    int64_t H = 5;
    int64_t gateNum = 3;
    int64_t numLayers = 1;
    int64_t dScale = 1;

    auto input = TensorDesc({T, B, I}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto wi = TensorDesc({gateNum * H, I}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto wh = TensorDesc({gateNum * H, H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto params = TensorListDesc({wi, wh});

    auto output = TensorDesc({T, B, dScale * H}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGRU, INPUT(input, params, nullptr, nullptr),
                        OUTPUT(output, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr), false, numLayers, 0.0,
                        false, false, false);

    uint64_t workspaceSize = 0;
    aclnnStatus ret = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
}

// 负例5: 训练模式下rOut为空，期望 ACLNN_ERR_PARAM_NULLPTR
TEST_F(GruApiTest, case_train_null_rout)
{
    int64_t T = 2;
    int64_t B = 3;
    int64_t I = 4;
    int64_t H = 5;
    int64_t gateNum = 3;
    int64_t numLayers = 1;
    int64_t dScale = 1;
    int64_t ldScale = numLayers * dScale;

    auto input = TensorDesc({T, B, I}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto wi = TensorDesc({gateNum * H, I}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto wh = TensorDesc({gateNum * H, H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto params = TensorListDesc({wi, wh});

    auto output = TensorDesc({T, B, dScale * H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto hy = TensorDesc({ldScale, B, H}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGRU, INPUT(input, params, nullptr, nullptr),
                        OUTPUT(output, hy, nullptr, nullptr, nullptr, nullptr, nullptr), false, numLayers, 0.0, true,
                        false, false);

    uint64_t workspaceSize = 0;
    aclnnStatus ret = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
}

// 负例6: 输入维度错误（4D输入），期望 ACLNN_ERR_PARAM_INVALID
TEST_F(GruApiTest, case_invalid_input_dim)
{
    int64_t T = 2;
    int64_t B = 3;
    int64_t I = 4;
    int64_t H = 5;
    int64_t gateNum = 3;
    int64_t numLayers = 1;
    int64_t dScale = 1;
    int64_t ldScale = numLayers * dScale;

    auto input = TensorDesc({T, B, I, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto wi = TensorDesc({gateNum * H, I}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto wh = TensorDesc({gateNum * H, H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto params = TensorListDesc({wi, wh});

    auto output = TensorDesc({T, B, dScale * H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto hy = TensorDesc({ldScale, B, H}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGRU, INPUT(input, params, nullptr, nullptr),
                        OUTPUT(output, hy, nullptr, nullptr, nullptr, nullptr, nullptr), false, numLayers, 0.0, false,
                        false, false);

    uint64_t workspaceSize = 0;
    aclnnStatus ret = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

// 负例7: params数量不匹配，期望 ACLNN_ERR_PARAM_INVALID
TEST_F(GruApiTest, case_params_count_mismatch)
{
    int64_t T = 2;
    int64_t B = 3;
    int64_t I = 4;
    int64_t H = 5;
    int64_t gateNum = 3;
    int64_t numLayers = 1;
    int64_t dScale = 1;
    int64_t ldScale = numLayers * dScale;

    auto input = TensorDesc({T, B, I}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto wi = TensorDesc({gateNum * H, I}, ACL_FLOAT16, ACL_FORMAT_ND);
    // 只给1个参数，但需要2个(wi, wh)
    auto params = TensorListDesc({wi});

    auto output = TensorDesc({T, B, dScale * H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto hy = TensorDesc({ldScale, B, H}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGRU, INPUT(input, params, nullptr, nullptr),
                        OUTPUT(output, hy, nullptr, nullptr, nullptr, nullptr, nullptr), false, numLayers, 0.0, false,
                        false, false);

    uint64_t workspaceSize = 0;
    aclnnStatus ret = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}

// 负例8: 输出shape不匹配，期望 ACLNN_ERR_PARAM_INVALID
TEST_F(GruApiTest, case_output_shape_mismatch)
{
    int64_t T = 2;
    int64_t B = 3;
    int64_t I = 4;
    int64_t H = 5;
    int64_t gateNum = 3;
    int64_t numLayers = 1;
    int64_t dScale = 1;
    int64_t ldScale = numLayers * dScale;

    auto input = TensorDesc({T, B, I}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto wi = TensorDesc({gateNum * H, I}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto wh = TensorDesc({gateNum * H, H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto params = TensorListDesc({wi, wh});

    // output shape should be (T, B, dScale*H), but use wrong shape
    auto output = TensorDesc({T, B, H + 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto hy = TensorDesc({ldScale, B, H}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGRU, INPUT(input, params, nullptr, nullptr),
                        OUTPUT(output, hy, nullptr, nullptr, nullptr, nullptr, nullptr), false, numLayers, 0.0, false,
                        false, false);

    uint64_t workspaceSize = 0;
    aclnnStatus ret = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_INVALID);
}