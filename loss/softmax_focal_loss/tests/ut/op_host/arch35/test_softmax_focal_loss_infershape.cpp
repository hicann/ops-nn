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
 * 当前只支持 reduction="none", 故 infershape 在 Ascend950 上对 mean/sum 显式拒收。
 * 该判定按 soc 门控 —— 非 Ascend950 保持原行为(不读属性、y 与 pred 同形), 本文件两侧都测。
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

// 按 reduction 取值跑一次 infershape; 返回 GRAPH_SUCCESS/GRAPH_FAILED
ge::graphStatus RunInfer(const char* reduction, int64_t* outDim0 = nullptr, int64_t* outDim1 = nullptr)
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
    if (ret == ge::GRAPH_SUCCESS && outDim0 != nullptr && outDim1 != nullptr) {
        *outDim0 = context->GetOutputShape(0)->GetDim(0);
        *outDim1 = context->GetOutputShape(0)->GetDim(1);
    }
    return ret;
}
} // namespace

// Ascend950: 读 reduction, "none" 通过且 y 与 pred 同形, mean/sum 显式拒收。
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
    int64_t d0 = 0;
    int64_t d1 = 0;
    ASSERT_EQ(RunInfer("none", &d0, &d1), ge::GRAPH_SUCCESS);
    EXPECT_EQ(d0, N_DIM);
    EXPECT_EQ(d1, C_DIM);
}

TEST_F(SoftmaxFocalLossInferShapeA5, reduction_mean_rejected_not_implemented)
{
    EXPECT_EQ(RunInfer("mean"), ge::GRAPH_FAILED);
}

TEST_F(SoftmaxFocalLossInferShapeA5, reduction_sum_rejected_not_implemented)
{
    EXPECT_EQ(RunInfer("sum"), ge::GRAPH_FAILED);
}

// 大小写不敏感: "MEAN" 同样应拒(与 tiling 的 CheckReduction 口径一致)
TEST_F(SoftmaxFocalLossInferShapeA5, reduction_uppercase_mean_rejected)
{
    EXPECT_EQ(RunInfer("MEAN"), ge::GRAPH_FAILED);
}

// 其它 soc: 保持原行为, 不读属性, mean 也按同形推导通过。
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

TEST_F(SoftmaxFocalLossInferShapeNonA5, reduction_mean_keeps_legacy_behavior)
{
    int64_t d0 = 0;
    int64_t d1 = 0;
    ASSERT_EQ(RunInfer("mean", &d0, &d1), ge::GRAPH_SUCCESS);
    EXPECT_EQ(d0, N_DIM);
    EXPECT_EQ(d1, C_DIM);
}
