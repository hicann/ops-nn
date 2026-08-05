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
 * \file test_mse_loss_v2_infershape.cpp
 * \brief MSELossV2 的 -2 UNKNOWN_RANK infershape 用例（arch35 交付新增）。
 *
 * 为何单独一份放在 arch35/ 而不是加进根目录那份：
 *   MSELossV2 是双代算子（def 注册 ascend910b/ascend910_93/ascend310p/ascend950），
 *   根目录的 test_mse_loss_v2_infershape.cpp 是 A2 也在跑的既有 UT。按 arch35 移植铁律
 *   「不是我们这次开发的文件尽量不动，只加我们的 arch35 新文件」，本次交付新增的用例
 *   隔离到 arch35/，避免扩张 A2 的测试面、也避免在 A2 门禁上阻塞与 A2 无关的交付。
 *
 * 覆盖：infershape 对 unknown-rank 走整体拷贝透传（reduction=none），
 *       sum/mean 恒为标量（rank 0），与输入 rank 是否已知无关。红线 R4 点名的必验项。
 */
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "ut_op_common.h"
#include "../../../../../../tests/ut/common/any_value.h"

using namespace std;

namespace {
std::vector<int64_t> ToVector(const gert::Shape& shape)
{
    std::vector<int64_t> dims(shape.GetDimNum(), 0);
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        dims[i] = shape.GetDim(i);
    }
    return dims;
}
} // namespace

class MSELossV2Arch35Infershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MSELossV2 arch35 Infershape SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "MSELossV2 arch35 Infershape TearDown" << std::endl; }

    static gert::KernelRunContextHolder Build(const char* reduction, gert::StorageShape& in, gert::StorageShape& out)
    {
        return gert::InferShapeContextFaker()
            .SetOpType("MSELossV2")
            .NodeIoNum(2, 1)
            .IrInstanceNum({1, 1})
            .InputShapes({&in, &in})
            .OutputShapes({&out})
            .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs({{"reduction", Ops::NN::AnyValue::CreateFrom<string>(reduction)}})
            .Build();
    }
};

// reduction=none：unknown-rank 原样透传 {-2}
TEST_F(MSELossV2Arch35Infershape, unknown_rank_none_passthrough)
{
    gert::StorageShape in = {{-2}, {-2}};
    gert::StorageShape out = {{-2}, {-2}};
    auto holder = Build("none", in, out);

    auto context = holder.GetContext<gert::InferShapeContext>();
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MSELossV2")->infer_shape;
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

    std::vector<int64_t> expected = {-2};
    EXPECT_EQ(ToVector(*context->GetOutputShape(0)), expected);
}

// reduction=mean：规约输出恒为标量，与输入 rank 已不已知无关
TEST_F(MSELossV2Arch35Infershape, unknown_rank_mean_is_scalar)
{
    gert::StorageShape in = {{-2}, {-2}};
    gert::StorageShape out = {{-2}, {-2}};
    auto holder = Build("mean", in, out);

    auto context = holder.GetContext<gert::InferShapeContext>();
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MSELossV2")->infer_shape;
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

    EXPECT_EQ(context->GetOutputShape(0)->GetDimNum(), 0U);
}
