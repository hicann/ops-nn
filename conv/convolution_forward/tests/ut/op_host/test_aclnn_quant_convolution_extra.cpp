/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_quant_convolution_extra.cpp
 * \brief aclnnQuantConvolution 参数校验分支补测：round_mode/offset_x/groups 的各错误分支、
 *        pad 尺寸分支(2/4/6维)、2D量化卷积路径、dtype/scale 校验分支。
 *        与现有 test_aclnn_quant_convolution.cpp 互补，后者只覆盖常规合法组合。
 *        ND专属分支(round_mode/offset_x校验、DAV dtype矩阵、2D路径)放在 quant_convolution_nd_test，
 *        其用 stub 的 SocVersionManager 切到 ASCEND950(DAV_3510)使 IsSocSupportND 为真。
 */

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "../../../op_host/op_api/aclnn_quant_convolution.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

#include "opdev/platform.h"

using namespace op;
using namespace std;

namespace {
constexpr int64_t kGroups1 = 1;
constexpr int32_t kOffsetX0 = 0;

// 超长字符串(>1024)用于round_mode长度校验分支
const char* MakeLongRoundMode()
{
    static std::string longStr(1200, 'a');
    return longStr.c_str();
}
} // namespace

class quant_convolution_extra_test : public testing::Test {
public:
    void SetUp() {}

    void TearDown() {}
};

// ND架构平台覆写fixture：UT的opdev平台是stub(tests/ut/op_api/stub/opdev/platform.cpp)，
// 默认DAV_2201使IsSocSupportND为false，round_mode/offset_x校验、DAV dtype矩阵、2D量化卷积等分支不可达。
// 用stub提供的RAII工具SocVersionManager切到ASCEND950(DAV_3510)，析构自动恢复。
class quant_convolution_nd_test : public testing::Test {
public:
    void SetUp() override { socVersionManager_ = std::make_unique<op::SocVersionManager>(SocVersion::ASCEND950); }

    void TearDown() override { socVersionManager_.reset(); }

private:
    std::unique_ptr<op::SocVersionManager> socVersionManager_;
};

// round_mode：HIFLOAT8输出时必须为"round"，合法取值走通主路径
TEST_F(quant_convolution_nd_test, hif8_output_round_mode_valid)
{
    vector<int64_t> inp = {1, 2, 16, 16, 16};
    vector<int64_t> weight = {2, 2, 3, 3, 3};
    vector<int64_t> bias = {2};
    vector<int64_t> scale = {2};
    vector<int64_t> output = {1, 2, 16, 16, 16};
    auto stridesDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto paddingDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto dilationDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto outputPaddingDesc = IntArrayDesc(std::vector<int64_t>{0, 0, 0});

    auto inpDesc = TensorDesc(inp, ACL_HIFLOAT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weightDesc = TensorDesc(weight, ACL_HIFLOAT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto biasDesc = TensorDesc(bias, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scaleDesc = TensorDesc(scale, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto outputDesc = TensorDesc(output, ACL_HIFLOAT8, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnQuantConvolution,
                        INPUT(inpDesc, weightDesc, biasDesc, scaleDesc, nullptr, stridesDesc, paddingDesc, dilationDesc,
                              false, outputPaddingDesc, kGroups1, kOffsetX0, "round"),
                        OUTPUT(outputDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// round_mode：HIFLOAT8输出时取"rint"报参数错误
TEST_F(quant_convolution_nd_test, hif8_output_round_mode_wrong)
{
    vector<int64_t> inp = {1, 2, 16, 16, 16};
    vector<int64_t> weight = {2, 2, 3, 3, 3};
    vector<int64_t> bias = {2};
    vector<int64_t> scale = {2};
    vector<int64_t> output = {1, 2, 16, 16, 16};
    auto stridesDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto paddingDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto dilationDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto outputPaddingDesc = IntArrayDesc(std::vector<int64_t>{0, 0, 0});

    auto inpDesc = TensorDesc(inp, ACL_HIFLOAT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weightDesc = TensorDesc(weight, ACL_HIFLOAT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto biasDesc = TensorDesc(bias, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scaleDesc = TensorDesc(scale, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto outputDesc = TensorDesc(output, ACL_HIFLOAT8, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnQuantConvolution,
                        INPUT(inpDesc, weightDesc, biasDesc, scaleDesc, nullptr, stridesDesc, paddingDesc, dilationDesc,
                              false, outputPaddingDesc, kGroups1, kOffsetX0, "rint"),
                        OUTPUT(outputDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// round_mode：HIFLOAT8输出时为空指针，校验器统一返回参数错误
TEST_F(quant_convolution_nd_test, hif8_output_round_mode_nullptr)
{
    vector<int64_t> inp = {1, 2, 16, 16, 16};
    vector<int64_t> weight = {2, 2, 3, 3, 3};
    vector<int64_t> bias = {2};
    vector<int64_t> scale = {2};
    vector<int64_t> output = {1, 2, 16, 16, 16};
    auto stridesDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto paddingDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto dilationDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto outputPaddingDesc = IntArrayDesc(std::vector<int64_t>{0, 0, 0});

    auto inpDesc = TensorDesc(inp, ACL_HIFLOAT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weightDesc = TensorDesc(weight, ACL_HIFLOAT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto biasDesc = TensorDesc(bias, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scaleDesc = TensorDesc(scale, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto outputDesc = TensorDesc(output, ACL_HIFLOAT8, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnQuantConvolution,
                        INPUT(inpDesc, weightDesc, biasDesc, scaleDesc, nullptr, stridesDesc, paddingDesc, dilationDesc,
                              false, outputPaddingDesc, kGroups1, kOffsetX0, nullptr),
                        OUTPUT(outputDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// round_mode：非量化输出(bf16)时传合法字符串，走告警+字符串校验成功路径
TEST_F(quant_convolution_nd_test, bf16_output_valid_string_round_mode)
{
    vector<int64_t> inp = {1, 2, 16, 16, 16};
    vector<int64_t> weight = {2, 2, 3, 3, 3};
    vector<int64_t> bias = {2};
    vector<int64_t> scale = {2};
    vector<int64_t> output = {1, 2, 16, 16, 16};
    auto stridesDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto paddingDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto dilationDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto outputPaddingDesc = IntArrayDesc(std::vector<int64_t>{0, 0, 0});

    auto inpDesc = TensorDesc(inp, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weightDesc = TensorDesc(weight, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto biasDesc = TensorDesc(bias, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scaleDesc = TensorDesc(scale, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto outputDesc = TensorDesc(output, ACL_BF16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnQuantConvolution,
                        INPUT(inpDesc, weightDesc, biasDesc, scaleDesc, nullptr, stridesDesc, paddingDesc, dilationDesc,
                              false, outputPaddingDesc, kGroups1, kOffsetX0, "rint"),
                        OUTPUT(outputDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// round_mode：非法字符('-')报参数错误
TEST_F(quant_convolution_nd_test, bf16_output_round_mode_invalid_char)
{
    vector<int64_t> inp = {1, 2, 16, 16, 16};
    vector<int64_t> weight = {2, 2, 3, 3, 3};
    vector<int64_t> bias = {2};
    vector<int64_t> scale = {2};
    vector<int64_t> output = {1, 2, 16, 16, 16};
    auto stridesDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto paddingDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto dilationDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto outputPaddingDesc = IntArrayDesc(std::vector<int64_t>{0, 0, 0});

    auto inpDesc = TensorDesc(inp, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weightDesc = TensorDesc(weight, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto biasDesc = TensorDesc(bias, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scaleDesc = TensorDesc(scale, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto outputDesc = TensorDesc(output, ACL_BF16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnQuantConvolution,
                        INPUT(inpDesc, weightDesc, biasDesc, scaleDesc, nullptr, stridesDesc, paddingDesc, dilationDesc,
                              false, outputPaddingDesc, kGroups1, kOffsetX0, "bad-str"),
                        OUTPUT(outputDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// round_mode：超长字符串(>1024)报参数错误
TEST_F(quant_convolution_nd_test, bf16_output_round_mode_too_long)
{
    vector<int64_t> inp = {1, 2, 16, 16, 16};
    vector<int64_t> weight = {2, 2, 3, 3, 3};
    vector<int64_t> bias = {2};
    vector<int64_t> scale = {2};
    vector<int64_t> output = {1, 2, 16, 16, 16};
    auto stridesDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto paddingDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto dilationDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto outputPaddingDesc = IntArrayDesc(std::vector<int64_t>{0, 0, 0});

    auto inpDesc = TensorDesc(inp, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weightDesc = TensorDesc(weight, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto biasDesc = TensorDesc(bias, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scaleDesc = TensorDesc(scale, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto outputDesc = TensorDesc(output, ACL_BF16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnQuantConvolution,
                        INPUT(inpDesc, weightDesc, biasDesc, scaleDesc, nullptr, stridesDesc, paddingDesc, dilationDesc,
                              false, outputPaddingDesc, kGroups1, kOffsetX0, MakeLongRoundMode()),
                        OUTPUT(outputDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// offset_x：x为HIFLOAT8时offset_x必须为0
TEST_F(quant_convolution_nd_test, hif8_input_offsetx_not_zero)
{
    vector<int64_t> inp = {1, 2, 16, 16, 16};
    vector<int64_t> weight = {2, 2, 3, 3, 3};
    vector<int64_t> bias = {2};
    vector<int64_t> scale = {2};
    vector<int64_t> output = {1, 2, 16, 16, 16};
    auto stridesDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto paddingDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto dilationDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto outputPaddingDesc = IntArrayDesc(std::vector<int64_t>{0, 0, 0});

    auto inpDesc = TensorDesc(inp, ACL_HIFLOAT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weightDesc = TensorDesc(weight, ACL_HIFLOAT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto biasDesc = TensorDesc(bias, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scaleDesc = TensorDesc(scale, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto outputDesc = TensorDesc(output, ACL_FLOAT, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnQuantConvolution,
                        INPUT(inpDesc, weightDesc, biasDesc, scaleDesc, nullptr, stridesDesc, paddingDesc, dilationDesc,
                              false, outputPaddingDesc, kGroups1, 1, nullptr),
                        OUTPUT(outputDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// offset_x：超出上限127
TEST_F(quant_convolution_nd_test, offsetx_over_max)
{
    vector<int64_t> inp = {1, 2, 16, 16, 16};
    vector<int64_t> weight = {2, 2, 3, 3, 3};
    vector<int64_t> bias = {2};
    vector<int64_t> scale = {2};
    vector<int64_t> output = {1, 2, 16, 16, 16};
    auto stridesDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto paddingDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto dilationDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto outputPaddingDesc = IntArrayDesc(std::vector<int64_t>{0, 0, 0});

    auto inpDesc = TensorDesc(inp, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weightDesc = TensorDesc(weight, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto biasDesc = TensorDesc(bias, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scaleDesc = TensorDesc(scale, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto outputDesc = TensorDesc(output, ACL_BF16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnQuantConvolution,
                        INPUT(inpDesc, weightDesc, biasDesc, scaleDesc, nullptr, stridesDesc, paddingDesc, dilationDesc,
                              false, outputPaddingDesc, kGroups1, 128, nullptr),
                        OUTPUT(outputDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// offset_x：低于下限-128
TEST_F(quant_convolution_nd_test, offsetx_below_min)
{
    vector<int64_t> inp = {1, 2, 16, 16, 16};
    vector<int64_t> weight = {2, 2, 3, 3, 3};
    vector<int64_t> bias = {2};
    vector<int64_t> scale = {2};
    vector<int64_t> output = {1, 2, 16, 16, 16};
    auto stridesDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto paddingDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto dilationDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto outputPaddingDesc = IntArrayDesc(std::vector<int64_t>{0, 0, 0});

    auto inpDesc = TensorDesc(inp, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weightDesc = TensorDesc(weight, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto biasDesc = TensorDesc(bias, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scaleDesc = TensorDesc(scale, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto outputDesc = TensorDesc(output, ACL_BF16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnQuantConvolution,
                        INPUT(inpDesc, weightDesc, biasDesc, scaleDesc, nullptr, stridesDesc, paddingDesc, dilationDesc,
                              false, outputPaddingDesc, kGroups1, -129, nullptr),
                        OUTPUT(outputDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 3D pad为6维：ND路径按6维透传
TEST_F(quant_convolution_nd_test, quant3d_pad6_success)
{
    vector<int64_t> inp = {1, 2, 16, 16, 16};
    vector<int64_t> weight = {2, 2, 3, 3, 3};
    vector<int64_t> bias = {2};
    vector<int64_t> scale = {2};
    vector<int64_t> output = {1, 2, 16, 16, 16};
    auto stridesDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto paddingDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1, 1, 1, 1});
    auto dilationDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto outputPaddingDesc = IntArrayDesc(std::vector<int64_t>{0, 0, 0});

    auto inpDesc = TensorDesc(inp, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weightDesc = TensorDesc(weight, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto biasDesc = TensorDesc(bias, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scaleDesc = TensorDesc(scale, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto outputDesc = TensorDesc(output, ACL_BF16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnQuantConvolution,
                        INPUT(inpDesc, weightDesc, biasDesc, scaleDesc, nullptr, stridesDesc, paddingDesc, dilationDesc,
                              false, outputPaddingDesc, kGroups1, kOffsetX0, nullptr),
                        OUTPUT(outputDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 2D量化卷积：4维输入pad为4维(scale按ExtendConv2D的infershape要求用INT64)
TEST_F(quant_convolution_nd_test, quant2d_pad4_success)
{
    vector<int64_t> inp = {1, 4, 16, 16};
    vector<int64_t> weight = {8, 4, 3, 3};
    vector<int64_t> bias = {8};
    vector<int64_t> scale = {8};
    vector<int64_t> output = {1, 8, 16, 16};
    auto stridesDesc = IntArrayDesc(std::vector<int64_t>{1, 1});
    auto paddingDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1, 1});
    auto dilationDesc = IntArrayDesc(std::vector<int64_t>{1, 1});
    auto outputPaddingDesc = IntArrayDesc(std::vector<int64_t>{0, 0});

    auto inpDesc = TensorDesc(inp, ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weightDesc = TensorDesc(weight, ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto biasDesc = TensorDesc(bias, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scaleDesc = TensorDesc(scale, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto outputDesc = TensorDesc(output, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnQuantConvolution,
                        INPUT(inpDesc, weightDesc, biasDesc, scaleDesc, nullptr, stridesDesc, paddingDesc, dilationDesc,
                              false, outputPaddingDesc, kGroups1, kOffsetX0, nullptr),
                        OUTPUT(outputDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 2D量化卷积：pad为2维，走pad扩成4维分支
TEST_F(quant_convolution_nd_test, quant2d_pad2_success)
{
    vector<int64_t> inp = {1, 4, 16, 16};
    vector<int64_t> weight = {8, 4, 3, 3};
    vector<int64_t> bias = {8};
    vector<int64_t> scale = {8};
    vector<int64_t> output = {1, 8, 16, 16};
    auto stridesDesc = IntArrayDesc(std::vector<int64_t>{1, 1});
    auto paddingDesc = IntArrayDesc(std::vector<int64_t>{1, 1});
    auto dilationDesc = IntArrayDesc(std::vector<int64_t>{1, 1});
    auto outputPaddingDesc = IntArrayDesc(std::vector<int64_t>{0, 0});

    auto inpDesc = TensorDesc(inp, ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weightDesc = TensorDesc(weight, ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto biasDesc = TensorDesc(bias, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scaleDesc = TensorDesc(scale, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto outputDesc = TensorDesc(output, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnQuantConvolution,
                        INPUT(inpDesc, weightDesc, biasDesc, scaleDesc, nullptr, stridesDesc, paddingDesc, dilationDesc,
                              false, outputPaddingDesc, kGroups1, kOffsetX0, nullptr),
                        OUTPUT(outputDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 2D量化卷积：stride维度不足，L0校验失败
TEST_F(quant_convolution_nd_test, quant2d_stride_too_small_fail)
{
    vector<int64_t> inp = {1, 4, 16, 16};
    vector<int64_t> weight = {8, 4, 3, 3};
    vector<int64_t> bias = {8};
    vector<int64_t> scale = {8};
    vector<int64_t> output = {1, 8, 16, 16};
    auto stridesDesc = IntArrayDesc(std::vector<int64_t>{1});
    auto paddingDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1, 1});
    auto dilationDesc = IntArrayDesc(std::vector<int64_t>{1, 1});
    auto outputPaddingDesc = IntArrayDesc(std::vector<int64_t>{0, 0});

    auto inpDesc = TensorDesc(inp, ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto weightDesc = TensorDesc(weight, ACL_INT8, ACL_FORMAT_NCHW).ValueRange(0, 2);
    auto biasDesc = TensorDesc(bias, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scaleDesc = TensorDesc(scale, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto outputDesc = TensorDesc(output, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnQuantConvolution,
                        INPUT(inpDesc, weightDesc, biasDesc, scaleDesc, nullptr, stridesDesc, paddingDesc, dilationDesc,
                              false, outputPaddingDesc, kGroups1, kOffsetX0, nullptr),
                        OUTPUT(outputDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// scale：ND soc下不支持INT32，仅INT64/UINT64/FLOAT
TEST_F(quant_convolution_nd_test, scale_dtype_int32_fail)
{
    vector<int64_t> inp = {1, 2, 16, 16, 16};
    vector<int64_t> weight = {2, 2, 3, 3, 3};
    vector<int64_t> bias = {2};
    vector<int64_t> scale = {2};
    vector<int64_t> output = {1, 2, 16, 16, 16};
    auto stridesDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto paddingDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto dilationDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto outputPaddingDesc = IntArrayDesc(std::vector<int64_t>{0, 0, 0});

    auto inpDesc = TensorDesc(inp, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weightDesc = TensorDesc(weight, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto biasDesc = TensorDesc(bias, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scaleDesc = TensorDesc(scale, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 2);
    auto outputDesc = TensorDesc(output, ACL_BF16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnQuantConvolution,
                        INPUT(inpDesc, weightDesc, biasDesc, scaleDesc, nullptr, stridesDesc, paddingDesc, dilationDesc,
                              false, outputPaddingDesc, kGroups1, kOffsetX0, nullptr),
                        OUTPUT(outputDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// scale：ND soc下支持INT64
TEST_F(quant_convolution_nd_test, scale_dtype_int64_success)
{
    vector<int64_t> inp = {1, 2, 16, 16, 16};
    vector<int64_t> weight = {2, 2, 3, 3, 3};
    vector<int64_t> bias = {2};
    vector<int64_t> scale = {2};
    vector<int64_t> output = {1, 2, 16, 16, 16};
    auto stridesDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto paddingDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto dilationDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto outputPaddingDesc = IntArrayDesc(std::vector<int64_t>{0, 0, 0});

    auto inpDesc = TensorDesc(inp, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weightDesc = TensorDesc(weight, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto biasDesc = TensorDesc(bias, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scaleDesc = TensorDesc(scale, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto outputDesc = TensorDesc(output, ACL_BF16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnQuantConvolution,
                        INPUT(inpDesc, weightDesc, biasDesc, scaleDesc, nullptr, stridesDesc, paddingDesc, dilationDesc,
                              false, outputPaddingDesc, kGroups1, kOffsetX0, nullptr),
                        OUTPUT(outputDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// transposed=true：走权重转置校验分支
TEST_F(quant_convolution_nd_test, quant3d_transposed)
{
    vector<int64_t> inp = {1, 2, 16, 16, 16};
    vector<int64_t> weight = {2, 2, 3, 3, 3};
    vector<int64_t> bias = {2};
    vector<int64_t> scale = {2};
    vector<int64_t> output = {1, 2, 17, 17, 17};
    auto stridesDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto paddingDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto dilationDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto outputPaddingDesc = IntArrayDesc(std::vector<int64_t>{0, 0, 0});

    auto inpDesc = TensorDesc(inp, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weightDesc = TensorDesc(weight, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto biasDesc = TensorDesc(bias, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scaleDesc = TensorDesc(scale, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto outputDesc = TensorDesc(output, ACL_BF16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnQuantConvolution,
                        INPUT(inpDesc, weightDesc, biasDesc, scaleDesc, nullptr, stridesDesc, paddingDesc, dilationDesc,
                              true, outputPaddingDesc, kGroups1, kOffsetX0, nullptr),
                        OUTPUT(outputDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_NE(aclRet, ACLNN_ERR_INNER);
}

// groups：取0报错(默认平台即覆盖)
TEST_F(quant_convolution_extra_test, groups_zero_fail)
{
    vector<int64_t> inp = {1, 2, 16, 16, 16};
    vector<int64_t> weight = {2, 2, 3, 3, 3};
    vector<int64_t> bias = {2};
    vector<int64_t> scale = {2};
    vector<int64_t> output = {1, 2, 16, 16, 16};
    auto stridesDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto paddingDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto dilationDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto outputPaddingDesc = IntArrayDesc(std::vector<int64_t>{0, 0, 0});

    auto inpDesc = TensorDesc(inp, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weightDesc = TensorDesc(weight, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto biasDesc = TensorDesc(bias, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scaleDesc = TensorDesc(scale, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto outputDesc = TensorDesc(output, ACL_BF16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnQuantConvolution,
                        INPUT(inpDesc, weightDesc, biasDesc, scaleDesc, nullptr, stridesDesc, paddingDesc, dilationDesc,
                              false, outputPaddingDesc, 0, kOffsetX0, nullptr),
                        OUTPUT(outputDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// groups：filter的C*groups不等于x的C
TEST_F(quant_convolution_extra_test, groups_cin_mismatch)
{
    vector<int64_t> inp = {1, 5, 16, 16, 16};
    vector<int64_t> weight = {4, 2, 3, 3, 3};
    vector<int64_t> bias = {4};
    vector<int64_t> scale = {4};
    vector<int64_t> output = {1, 4, 16, 16, 16};
    auto stridesDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto paddingDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto dilationDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto outputPaddingDesc = IntArrayDesc(std::vector<int64_t>{0, 0, 0});

    auto inpDesc = TensorDesc(inp, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weightDesc = TensorDesc(weight, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto biasDesc = TensorDesc(bias, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scaleDesc = TensorDesc(scale, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto outputDesc = TensorDesc(output, ACL_BF16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnQuantConvolution,
                        INPUT(inpDesc, weightDesc, biasDesc, scaleDesc, nullptr, stridesDesc, paddingDesc, dilationDesc,
                              false, outputPaddingDesc, 2, kOffsetX0, nullptr),
                        OUTPUT(outputDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// x的N维为0：shape校验失败
TEST_F(quant_convolution_extra_test, x_dim_n_zero_fail)
{
    vector<int64_t> inp = {0, 2, 16, 16, 16};
    vector<int64_t> weight = {2, 2, 3, 3, 3};
    vector<int64_t> bias = {2};
    vector<int64_t> scale = {2};
    vector<int64_t> output = {0, 2, 16, 16, 16};
    auto stridesDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto paddingDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto dilationDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto outputPaddingDesc = IntArrayDesc(std::vector<int64_t>{0, 0, 0});

    auto inpDesc = TensorDesc(inp, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weightDesc = TensorDesc(weight, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto biasDesc = TensorDesc(bias, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scaleDesc = TensorDesc(scale, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto outputDesc = TensorDesc(output, ACL_BF16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnQuantConvolution,
                        INPUT(inpDesc, weightDesc, biasDesc, scaleDesc, nullptr, stridesDesc, paddingDesc, dilationDesc,
                              false, outputPaddingDesc, kGroups1, kOffsetX0, nullptr),
                        OUTPUT(outputDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// filter的H维为0：shape校验失败
TEST_F(quant_convolution_extra_test, filter_dim_h_zero_fail)
{
    vector<int64_t> inp = {1, 2, 16, 16, 16};
    vector<int64_t> weight = {2, 2, 3, 0, 3};
    vector<int64_t> bias = {2};
    vector<int64_t> scale = {2};
    vector<int64_t> output = {1, 2, 16, 16, 16};
    auto stridesDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto paddingDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto dilationDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto outputPaddingDesc = IntArrayDesc(std::vector<int64_t>{0, 0, 0});

    auto inpDesc = TensorDesc(inp, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weightDesc = TensorDesc(weight, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto biasDesc = TensorDesc(bias, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scaleDesc = TensorDesc(scale, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto outputDesc = TensorDesc(output, ACL_BF16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnQuantConvolution,
                        INPUT(inpDesc, weightDesc, biasDesc, scaleDesc, nullptr, stridesDesc, paddingDesc, dilationDesc,
                              false, outputPaddingDesc, kGroups1, kOffsetX0, nullptr),
                        OUTPUT(outputDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// dtype：x不支持INT32
TEST_F(quant_convolution_extra_test, x_dtype_int32_fail)
{
    vector<int64_t> inp = {1, 2, 16, 16, 16};
    vector<int64_t> weight = {2, 2, 3, 3, 3};
    vector<int64_t> bias = {2};
    vector<int64_t> scale = {2};
    vector<int64_t> output = {1, 2, 16, 16, 16};
    auto stridesDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto paddingDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto dilationDesc = IntArrayDesc(std::vector<int64_t>{1, 1, 1});
    auto outputPaddingDesc = IntArrayDesc(std::vector<int64_t>{0, 0, 0});

    auto inpDesc = TensorDesc(inp, ACL_INT32, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto weightDesc = TensorDesc(weight, ACL_INT8, ACL_FORMAT_NCDHW).ValueRange(0, 2);
    auto biasDesc = TensorDesc(bias, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto scaleDesc = TensorDesc(scale, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    auto outputDesc = TensorDesc(output, ACL_BF16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);
    auto ut = OP_API_UT(aclnnQuantConvolution,
                        INPUT(inpDesc, weightDesc, biasDesc, scaleDesc, nullptr, stridesDesc, paddingDesc, dilationDesc,
                              false, outputPaddingDesc, kGroups1, kOffsetX0, nullptr),
                        OUTPUT(outputDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
