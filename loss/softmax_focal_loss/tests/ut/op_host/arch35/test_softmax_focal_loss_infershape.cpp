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
 * \file test_softmax_focal_loss_infershape.cpp
 * \brief infershape 的 reduction 语义 UT。
 *
 * infershape: none -> pred shape, mean/sum -> 0 维标量。
 * 实际执行仍只支持 none, 由 tiling 的 CheckReduction 拦截, 不进入 kernel。
 */
#include <gtest/gtest.h>
#include <iostream>
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "../../../../op_graph/softmax_focal_loss_proto.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "platform/platform_info.h"

namespace {
constexpr int64_t N_DIM = 32;
constexpr int64_t C_DIM = 128;

gert::OpImplRegisterV2::InferShapeKernelFunc GetSflInferShape()
{
    auto impl = gert::OpImplRegistry::GetInstance().GetOpImpl("SoftmaxFocalLoss");
    return impl == nullptr ? nullptr : impl->infer_shape;
}

struct SflOutShape {
    size_t dimNum = 0;
    int64_t dim0 = 0;
    int64_t dim1 = 0;
};

// 按 reduction 取值跑一次 infershape; 返回 GRAPH_SUCCESS/GRAPH_FAILED
ge::graphStatus RunInfer(const char* reduction, SflOutShape* outShape = nullptr)
{
    auto func = GetSflInferShape();
    if (func == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::StorageShape predShape = {{N_DIM, C_DIM}, {N_DIM, C_DIM}};
    gert::StorageShape targetShape = {{N_DIM, C_DIM}, {N_DIM, C_DIM}};
    gert::StorageShape weightShape = {{N_DIM, C_DIM}, {N_DIM, C_DIM}};
    gert::StorageShape yShape = {{N_DIM, C_DIM}, {N_DIM, C_DIM}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&predShape, &targetShape, &weightShape})
                      .OutputShapes({&yShape})
                      .NodeAttrs({
                          {"gamma", Ops::NN::AnyValue::CreateFrom<float>(2.0f)},
                          {"alpha", Ops::NN::AnyValue::CreateFrom<float>(0.25f)},
                          {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>(reduction)},
                      })
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    auto context = holder.GetContext<gert::InferShapeContext>();
    auto ret = func(context);
    if (ret == ge::GRAPH_SUCCESS && outShape != nullptr) {
        const gert::Shape* shape = context->GetOutputShape(0);
        outShape->dimNum = shape->GetDimNum();
        if (outShape->dimNum >= 2) {
            outShape->dim0 = shape->GetDim(0);
            outShape->dim1 = shape->GetDim(1);
        }
    }
    return ret;
}
} // namespace

// Ascend950: "none" 输出 pred shape, mean/sum 输出 0 维标量。
class SoftmaxFocalLossInferShapeA5 : public testing::Test {
protected:
    void SetUp() override
    {
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend950";
        optiCompilationInfo.soc_version = "Ascend950";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }
};

TEST_F(SoftmaxFocalLossInferShapeA5, reduction_none_pass_and_keep_pred_shape)
{
    SflOutShape outShape;
    ASSERT_EQ(RunInfer("none", &outShape), ge::GRAPH_SUCCESS);
    EXPECT_EQ(outShape.dimNum, 2u);
    EXPECT_EQ(outShape.dim0, N_DIM);
    EXPECT_EQ(outShape.dim1, C_DIM);
}

TEST_F(SoftmaxFocalLossInferShapeA5, reduction_mean_produces_scalar_shape)
{
    SflOutShape outShape;
    ASSERT_EQ(RunInfer("mean", &outShape), ge::GRAPH_SUCCESS);
    EXPECT_EQ(outShape.dimNum, 0u);
}

TEST_F(SoftmaxFocalLossInferShapeA5, reduction_sum_produces_scalar_shape)
{
    SflOutShape outShape;
    ASSERT_EQ(RunInfer("sum", &outShape), ge::GRAPH_SUCCESS);
    EXPECT_EQ(outShape.dimNum, 0u);
}

// 大小写不敏感: "MEAN" 与 "mean" 同语义
TEST_F(SoftmaxFocalLossInferShapeA5, reduction_uppercase_mean_produces_scalar_shape)
{
    SflOutShape outShape;
    ASSERT_EQ(RunInfer("MEAN", &outShape), ge::GRAPH_SUCCESS);
    EXPECT_EQ(outShape.dimNum, 0u);
}

// 其它 soc: 与 Ascend950 保持同一 infershape 契约。
class SoftmaxFocalLossInferShapeNonA5 : public testing::Test {
protected:
    void SetUp() override
    {
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 48;
        platformInfo.str_info.short_soc_version = "Ascend910B";
        optiCompilationInfo.soc_version = "Ascend910B";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }
};

TEST_F(SoftmaxFocalLossInferShapeNonA5, reduction_mean_produces_scalar_shape)
{
    SflOutShape outShape;
    ASSERT_EQ(RunInfer("mean", &outShape), ge::GRAPH_SUCCESS);
    EXPECT_EQ(outShape.dimNum, 0u);
}
