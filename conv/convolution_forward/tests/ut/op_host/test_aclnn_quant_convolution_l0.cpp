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
 * \file test_quant_convolution_l0.cpp
 * \brief quant_convolution.cpp 的 L0 算子（ExtendConv2dNCHW / QuantConv3dNCDHW）UT。
 *        这两个 L0 导出接口在本仓内无调用方（供外仓融合使用），历史覆盖率为0；
 *        用例直接构造 aclOpExecutor 调用，覆盖 L0Inner 的参数校验分支、
 *        ConstructQuantConvNewPad 的各 pad 尺寸分支与 INFER_SHAPE/加执行列表的主路径。
 */

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
// 框架头需先于源文件包含(其operator<<重载解析依赖可见顺序)；l0op接口未从libophost_nn_opapi.so导出，
// UT直接include源文件获得符号(同matmul fallback UT做法)
#include "../../../op_host/op_api/quant_convolution.cpp"

using namespace op;
using namespace std;

namespace {
// 独立构造的执行器：L0接口在真实运行时由aclnn流程持有executor，UT直接构造后传入
struct L0TestCtx {
    aclOpExecutor executor;

    aclIntArray* NewIntArray(const std::vector<int64_t>& values)
    {
        return executor.AllocIntArray(values.data(), values.size());
    }
};

// 2D量化卷积输入：x/filter为int8，NCHW
std::unique_ptr<aclTensor, void (*)(aclTensor*)> MakeQuant2DTensor(const std::vector<int64_t>& dims,
                                                                   aclDataType dtype = ACL_INT8)
{
    return TensorDesc(dims, dtype, ACL_FORMAT_NCHW).ToAclType();
}

// 3D量化卷积输入：NCDHW
std::unique_ptr<aclTensor, void (*)(aclTensor*)> MakeQuant3DTensor(const std::vector<int64_t>& dims,
                                                                   aclDataType dtype = ACL_INT8)
{
    return TensorDesc(dims, dtype, ACL_FORMAT_NCDHW).ToAclType();
}
} // namespace

class QuantConvolutionL0Test : public testing::Test {
public:
    void SetUp() {}
    void TearDown() {}
};

// ExtendConv2d主路径：4维pad，校验通过后走INFER_SHAPE与加执行列表
TEST_F(QuantConvolutionL0Test, ExtendConv2dNCHW_pad4_success)
{
    L0TestCtx ctx;
    auto input = MakeQuant2DTensor({1, 4, 8, 8});
    auto weight = MakeQuant2DTensor({8, 4, 3, 3});
    auto scale = MakeQuant2DTensor({8}, ACL_INT64);
    auto bias = MakeQuant2DTensor({8}, ACL_INT32);
    auto* stride = ctx.NewIntArray({1, 1});
    auto* padding = ctx.NewIntArray({1, 1, 1, 1});
    auto* dilation = ctx.NewIntArray({1, 1});
    const aclTensor* out = l0op::ExtendConv2dNCHW(input.get(), weight.get(), scale.get(), bias.get(),
                                                  DataType::DT_FLOAT16, stride, padding, dilation, 1, 0, "rint",
                                                  &ctx.executor);
    EXPECT_NE(out, nullptr);
}

// 2维pad：ConstructQuantConvNewPad按(padH,padH,padW,padW)扩成4维
TEST_F(QuantConvolutionL0Test, ExtendConv2dNCHW_pad2_success)
{
    L0TestCtx ctx;
    auto input = MakeQuant2DTensor({1, 4, 8, 8});
    auto weight = MakeQuant2DTensor({8, 4, 3, 3});
    auto scale = MakeQuant2DTensor({8}, ACL_INT64);
    auto bias = MakeQuant2DTensor({8}, ACL_INT32);
    auto* stride = ctx.NewIntArray({1, 1});
    auto* padding = ctx.NewIntArray({1, 1});
    auto* dilation = ctx.NewIntArray({1, 1});
    const aclTensor* out = l0op::ExtendConv2dNCHW(input.get(), weight.get(), scale.get(), bias.get(),
                                                  DataType::DT_FLOAT16, stride, padding, dilation, 1, 0, "rint",
                                                  &ctx.executor);
    EXPECT_NE(out, nullptr);
}

// 非法pad尺寸(5)：ConstructQuantConvNewPad落入else分支，返回的pad尺寸不为4，L0报错
TEST_F(QuantConvolutionL0Test, ExtendConv2dNCHW_invalid_pad_fail)
{
    L0TestCtx ctx;
    auto input = MakeQuant2DTensor({1, 4, 8, 8});
    auto weight = MakeQuant2DTensor({8, 4, 3, 3});
    auto scale = MakeQuant2DTensor({8}, ACL_INT64);
    auto bias = MakeQuant2DTensor({8}, ACL_INT32);
    auto* stride = ctx.NewIntArray({1, 1});
    auto* padding = ctx.NewIntArray({1, 1, 1, 1, 1});
    auto* dilation = ctx.NewIntArray({1, 1});
    const aclTensor* out = l0op::ExtendConv2dNCHW(input.get(), weight.get(), scale.get(), bias.get(),
                                                  DataType::DT_FLOAT16, stride, padding, dilation, 1, 0, "rint",
                                                  &ctx.executor);
    EXPECT_EQ(out, nullptr);
}

// stride维度小于2：L0入口校验失败
TEST_F(QuantConvolutionL0Test, ExtendConv2dNCHW_stride_too_small_fail)
{
    L0TestCtx ctx;
    auto input = MakeQuant2DTensor({1, 4, 8, 8});
    auto weight = MakeQuant2DTensor({8, 4, 3, 3});
    auto scale = MakeQuant2DTensor({8}, ACL_INT64);
    auto bias = MakeQuant2DTensor({8}, ACL_INT32);
    auto* stride = ctx.NewIntArray({1});
    auto* padding = ctx.NewIntArray({1, 1, 1, 1});
    auto* dilation = ctx.NewIntArray({1, 1});
    const aclTensor* out = l0op::ExtendConv2dNCHW(input.get(), weight.get(), scale.get(), bias.get(),
                                                  DataType::DT_FLOAT16, stride, padding, dilation, 1, 0, "rint",
                                                  &ctx.executor);
    EXPECT_EQ(out, nullptr);
}

// dilation维度小于2：L0入口校验失败
TEST_F(QuantConvolutionL0Test, ExtendConv2dNCHW_dilation_too_small_fail)
{
    L0TestCtx ctx;
    auto input = MakeQuant2DTensor({1, 4, 8, 8});
    auto weight = MakeQuant2DTensor({8, 4, 3, 3});
    auto scale = MakeQuant2DTensor({8}, ACL_INT64);
    auto bias = MakeQuant2DTensor({8}, ACL_INT32);
    auto* stride = ctx.NewIntArray({1, 1});
    auto* padding = ctx.NewIntArray({1, 1, 1, 1});
    auto* dilation = ctx.NewIntArray({1});
    const aclTensor* out = l0op::ExtendConv2dNCHW(input.get(), weight.get(), scale.get(), bias.get(),
                                                  DataType::DT_FLOAT16, stride, padding, dilation, 1, 0, "rint",
                                                  &ctx.executor);
    EXPECT_EQ(out, nullptr);
}

// QuantConv3d主路径：6维pad
TEST_F(QuantConvolutionL0Test, QuantConv3dNCDHW_pad6_success)
{
    L0TestCtx ctx;
    auto input = MakeQuant3DTensor({1, 4, 8, 8, 8});
    auto weight = MakeQuant3DTensor({8, 4, 3, 3, 3});
    auto scale = MakeQuant3DTensor({8}, ACL_FLOAT);
    auto bias = MakeQuant3DTensor({8}, ACL_FLOAT);
    auto* stride = ctx.NewIntArray({1, 1, 1});
    auto* padding = ctx.NewIntArray({1, 1, 1, 1, 1, 1});
    auto* dilation = ctx.NewIntArray({1, 1, 1});
    const aclTensor* out = l0op::QuantConv3dNCDHW(input.get(), weight.get(), scale.get(), bias.get(),
                                                  DataType::DT_FLOAT16, stride, padding, dilation, 1, 0, "rint",
                                                  &ctx.executor);
    EXPECT_NE(out, nullptr);
}

// 3维pad：ConstructQuantConvNewPad无匹配分支(尺寸需恰为6)落入else，pad尺寸校验失败
TEST_F(QuantConvolutionL0Test, QuantConv3dNCDHW_pad3_fail)
{
    L0TestCtx ctx;
    auto input = MakeQuant3DTensor({1, 4, 8, 8, 8});
    auto weight = MakeQuant3DTensor({8, 4, 3, 3, 3});
    auto scale = MakeQuant3DTensor({8}, ACL_FLOAT);
    auto bias = MakeQuant3DTensor({8}, ACL_FLOAT);
    auto* stride = ctx.NewIntArray({1, 1, 1});
    auto* padding = ctx.NewIntArray({1, 1, 1});
    auto* dilation = ctx.NewIntArray({1, 1, 1});
    const aclTensor* out = l0op::QuantConv3dNCDHW(input.get(), weight.get(), scale.get(), bias.get(),
                                                  DataType::DT_FLOAT16, stride, padding, dilation, 1, 0, "rint",
                                                  &ctx.executor);
    EXPECT_EQ(out, nullptr);
}

// 非法pad尺寸(4)：非6维，L0报错
TEST_F(QuantConvolutionL0Test, QuantConv3dNCDHW_invalid_pad_fail)
{
    L0TestCtx ctx;
    auto input = MakeQuant3DTensor({1, 4, 8, 8, 8});
    auto weight = MakeQuant3DTensor({8, 4, 3, 3, 3});
    auto scale = MakeQuant3DTensor({8}, ACL_FLOAT);
    auto bias = MakeQuant3DTensor({8}, ACL_FLOAT);
    auto* stride = ctx.NewIntArray({1, 1, 1});
    auto* padding = ctx.NewIntArray({1, 1, 1, 1});
    auto* dilation = ctx.NewIntArray({1, 1, 1});
    const aclTensor* out = l0op::QuantConv3dNCDHW(input.get(), weight.get(), scale.get(), bias.get(),
                                                  DataType::DT_FLOAT16, stride, padding, dilation, 1, 0, "rint",
                                                  &ctx.executor);
    EXPECT_EQ(out, nullptr);
}

// 3D stride维度小于3：L0入口校验失败
TEST_F(QuantConvolutionL0Test, QuantConv3dNCDHW_stride_too_small_fail)
{
    L0TestCtx ctx;
    auto input = MakeQuant3DTensor({1, 4, 8, 8, 8});
    auto weight = MakeQuant3DTensor({8, 4, 3, 3, 3});
    auto scale = MakeQuant3DTensor({8}, ACL_FLOAT);
    auto bias = MakeQuant3DTensor({8}, ACL_FLOAT);
    auto* stride = ctx.NewIntArray({1, 1});
    auto* padding = ctx.NewIntArray({1, 1, 1, 1, 1, 1});
    auto* dilation = ctx.NewIntArray({1, 1, 1});
    const aclTensor* out = l0op::QuantConv3dNCDHW(input.get(), weight.get(), scale.get(), bias.get(),
                                                  DataType::DT_FLOAT16, stride, padding, dilation, 1, 0, "rint",
                                                  &ctx.executor);
    EXPECT_EQ(out, nullptr);
}

// 3D dilation维度小于3：L0入口校验失败
TEST_F(QuantConvolutionL0Test, QuantConv3dNCDHW_dilation_too_small_fail)
{
    L0TestCtx ctx;
    auto input = MakeQuant3DTensor({1, 4, 8, 8, 8});
    auto weight = MakeQuant3DTensor({8, 4, 3, 3, 3});
    auto scale = MakeQuant3DTensor({8}, ACL_FLOAT);
    auto bias = MakeQuant3DTensor({8}, ACL_FLOAT);
    auto* stride = ctx.NewIntArray({1, 1, 1});
    auto* padding = ctx.NewIntArray({1, 1, 1, 1, 1, 1});
    auto* dilation = ctx.NewIntArray({1, 1});
    const aclTensor* out = l0op::QuantConv3dNCDHW(input.get(), weight.get(), scale.get(), bias.get(),
                                                  DataType::DT_FLOAT16, stride, padding, dilation, 1, 0, "rint",
                                                  &ctx.executor);
    EXPECT_EQ(out, nullptr);
}
