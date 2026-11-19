/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_test_util.h" // NOLINT
#include "ut_op_common.h"
#include "platform/platform_info.h"

class MseLossGradV2 : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MseLossGradV2 Proto Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "MseLossGradV2 Proto Test TearDown" << std::endl; }
};

std::vector<int64_t> ToVectorForMseLossGradV2(const gert::Shape& shape)
{
    size_t shape_size = shape.GetDimNum();
    std::vector<int64_t> shape_vec(shape_size, 0);

    for (size_t i = 0; i < shape_size; i++) {
        shape_vec[i] = shape.GetDim(i);
    }
    return shape_vec;
}

TEST_F(MseLossGradV2, MseLossGradV2_infershape_none_fp16_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MseLossGradV2"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MseLossGradV2")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT16;
        ge::DataType output_ref = ge::DT_FLOAT16;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(3)
                                  .NodeIoNum(3, 1)
                                  .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &input_ref, &input_ref})
                                  .OutputDataTypes({&output_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetInputDataType(1), input_ref);
        EXPECT_EQ(context->GetInputDataType(2), input_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
    }
}

TEST_F(MseLossGradV2, MseLossGradV2_infershape_none_fp16_case_1)
{
    gert::StorageShape predictShape = {{32, 64}, {32, 64}};
    gert::StorageShape labelShape = {{32, 64}, {32, 64}};
    gert::StorageShape doutShape = {{32, 64}, {32, 64}};
    gert::StorageShape yShape = {{32, 64}, {32, 64}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("MseLossGradV2")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&predictShape, &labelShape, &doutShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("mean")}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MseLossGradV2")->infer_shape;
    ge::graphStatus ret = infer_shape_func(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    std::vector<int64_t> expectedyGradShape = {32, 64};
    auto yGradShape = context->GetOutputShape(0);
    EXPECT_EQ(ToVectorForMseLossGradV2(*yGradShape), expectedyGradShape);
}

TEST_F(MseLossGradV2, MseLossGradV2_infershape_none_bf16_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MseLossGradV2"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MseLossGradV2")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_BF16;
        ge::DataType output_ref = ge::DT_BF16;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(3)
                                  .NodeIoNum(3, 1)
                                  .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &input_ref, &input_ref})
                                  .OutputDataTypes({&output_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetInputDataType(1), input_ref);
        EXPECT_EQ(context->GetInputDataType(2), input_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
    }
}

TEST_F(MseLossGradV2, MseLossGradV2_infershape_none_bp16_case_1)
{
    gert::StorageShape predictShape = {{32, 64}, {32, 64}};
    gert::StorageShape labelShape = {{32, 64}, {32, 64}};
    gert::StorageShape doutShape = {{32, 64}, {32, 64}};
    gert::StorageShape yShape = {{32, 64}, {32, 64}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("MseLossGradV2")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&predictShape, &labelShape, &doutShape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("mean")}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MseLossGradV2")->infer_shape;
    ge::graphStatus ret = infer_shape_func(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    std::vector<int64_t> expectedyGradShape = {32, 64};
    auto yGradShape = context->GetOutputShape(0);
    EXPECT_EQ(ToVectorForMseLossGradV2(*yGradShape), expectedyGradShape);
}

// 负向场景1：必选输入 dout 的实例数置 0（IrInstanceNum 第 3 个输入实例数为 0），
// context 的 input_size（输入实例数之和）变为 2，GetInputShape(INPUT_DOUT_IDX) 按下标越界
// 规则返回空指针，首个触达的整改检查点为 InferShapeForMseLossGrad 中 param_name="doutShape"
// （EZ0037 首报），前序 predictShape/labelShape 检查点因实例数保留而正常通过（见断言）。
// 注意：InferShapeContextFaker 的单参数 IrInstanceNum 是空操作，且 NodeIoNum(IonNum) 与
// IrInstanceNum(IOInstanceNum) 互斥，必须仅用双参数 IrInstanceNum 控制实例数
TEST_F(MseLossGradV2, mse_loss_grad_v2_infershape_null_input_shape_001)
{
    gert::StorageShape predictShape = {{32, 64}, {32, 64}};
    gert::StorageShape labelShape = {{32, 64}, {32, 64}};

    // 第 3 个输入 dout 实例数置 0（InputShapes 第 3 位 nullptr 占位，faker 不创建该输入链），
    // 保留 1 个输出实例，使输出链存在
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("MseLossGradV2")
                      .IrInstanceNum({1, 1, 0}, {1})
                      .InputShapes({&predictShape, &labelShape, nullptr})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("mean")}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context->GetInputShape(0), nullptr);
    ASSERT_NE(context->GetInputShape(1), nullptr);
    ASSERT_EQ(context->GetInputShape(2), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MseLossGradV2")->infer_shape;
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_FAILED);
}

// 负向场景2：输出 y 的实例数置 0（IrInstanceNum 输出实例数为 0），
// context 的 output_size 变为 0，GetOutputShape(OUTPUT_LOSS_GRAD_IDX) 按下标越界规则返回空指针，
// 对应整改检查点 param_name="lossGradShape"（InferShapeForMseLossGrad 末个检查点，基线与
// Regbase 分支检查点顺序一致），前序输入 shape 检查点因实例数保留而正常通过，整改后应上报
// EZ0037 首错并返回 ge::GRAPH_FAILED
TEST_F(MseLossGradV2, mse_loss_grad_v2_infershape_null_output_shape_002)
{
    gert::StorageShape predictShape = {{32, 64}, {32, 64}};
    gert::StorageShape labelShape = {{32, 64}, {32, 64}};
    gert::StorageShape doutShape = {{32, 64}, {32, 64}};

    // 输出实例数置 0 使 output_size 为 0（不调用 NodeOutputTd，避免为输出创建链），
    // 输入实例数保持 3 个且 shape 全部合法
    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("MseLossGradV2")
                      .IrInstanceNum({1, 1, 1}, {0})
                      .InputShapes({&predictShape, &labelShape, &doutShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("mean")}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context->GetInputShape(2), nullptr);
    ASSERT_EQ(context->GetOutputShape(0), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MseLossGradV2")->infer_shape;
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_FAILED);
}

// 负向场景3（基线分支）：predict 为 unknown rank（-2），基线分支 IsUnknownRank 检查点命中，
// 输出 dim0 置 -2 并返回失败
TEST_F(MseLossGradV2, mse_loss_grad_v2_infershape_unknown_rank_003)
{
    gert::StorageShape predictShape = {{-2}, {-2}};
    gert::StorageShape labelShape = {{32, 64}, {32, 64}};
    gert::StorageShape doutShape = {{32, 64}, {32, 64}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("MseLossGradV2")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&predictShape, &labelShape, &doutShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("mean")}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context->GetInputShape(0), nullptr);
    ASSERT_NE(context->GetOutputShape(0), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MseLossGradV2")->infer_shape;
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_FAILED);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(0), -2);
}

// ascend950(regbase) 分支专用套件：SetUpTestCase 向 fe::PlatformInfoManager 单例注入
// Ascend950 平台信息使 IsRegBaseSoc() 返回 true（参照 npu_get_float_status 先例），
// TearDownTestCase 擦除注入并复位 OptionalInfo，避免进程级单例污染后续套件
// （仓内仅本算子读取该单例的 PlatformInfo 重载，注入面可控）。
// 声明 shape 契约校验分支（declaredRank > 0 的 rank/dim 冲突上报）不可构造：
// InferShapeContextFaker 的 OutputShapes 为空实现，输出 shape 槽位由 builder 默认构造
// （rank=0），declaredRank 恒为 0，契约校验被跳过，故不构造对应负向用例。
class MseLossGradV2Regbase : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend950";
        optiCompilationInfo.soc_version = "Ascend950";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
        std::cout << "MseLossGradV2Regbase SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        fe::PlatformInfoManager::Instance().platform_info_map_.erase("Ascend950");
        fe::OptionalInfo emptyOptiCompilationInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(emptyOptiCompilationInfo);
        std::cout << "MseLossGradV2Regbase TearDown" << std::endl;
    }
};

// 场景4（regbase 正向）：y.shape = broadcast(dout, predict, label)，
// predict{32,64} x label{64} 广播为 {32,64}，再与 dout{32,1} 广播仍为 {32,64}
TEST_F(MseLossGradV2Regbase, mse_loss_grad_v2_regbase_broadcast_success_004)
{
    gert::StorageShape predictShape = {{32, 64}, {32, 64}};
    gert::StorageShape labelShape = {{64}, {64}};
    gert::StorageShape doutShape = {{32, 1}, {32, 1}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("MseLossGradV2")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&predictShape, &labelShape, &doutShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("mean")}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MseLossGradV2")->infer_shape;
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

    std::vector<int64_t> expectedyGradShape = {32, 64};
    auto yGradShape = context->GetOutputShape(0);
    EXPECT_EQ(ToVectorForMseLossGradV2(*yGradShape), expectedyGradShape);
}

// 负向场景5（regbase）：predict 为 unknown rank（-2），regbase 分支 IsUnknownRank 命中，
// 输出 dim0 置 -2 并返回失败
TEST_F(MseLossGradV2Regbase, mse_loss_grad_v2_regbase_unknown_rank_005)
{
    gert::StorageShape predictShape = {{-2}, {-2}};
    gert::StorageShape labelShape = {{32, 64}, {32, 64}};
    gert::StorageShape doutShape = {{32, 64}, {32, 64}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("MseLossGradV2")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&predictShape, &labelShape, &doutShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("mean")}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MseLossGradV2")->infer_shape;
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_FAILED);
    EXPECT_EQ(context->GetOutputShape(0)->GetDim(0), -2);
}

// 负向场景6（regbase）：reduction 取非法值，IsLegalReduction 校验失败并返回失败
TEST_F(MseLossGradV2Regbase, mse_loss_grad_v2_regbase_invalid_reduction_006)
{
    gert::StorageShape predictShape = {{32, 64}, {32, 64}};
    gert::StorageShape labelShape = {{32, 64}, {32, 64}};
    gert::StorageShape doutShape = {{32, 64}, {32, 64}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("MseLossGradV2")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&predictShape, &labelShape, &doutShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("invalid")}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MseLossGradV2")->infer_shape;
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_FAILED);
}

// 负向场景7（regbase）：predict 与 label 具体维冲突（32 vs 33），首次 BroadcastShapes 失败并返回失败
TEST_F(MseLossGradV2Regbase, mse_loss_grad_v2_regbase_broadcast_conflict_007)
{
    gert::StorageShape predictShape = {{32, 64}, {32, 64}};
    gert::StorageShape labelShape = {{33, 64}, {33, 64}};
    gert::StorageShape doutShape = {{32, 64}, {32, 64}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("MseLossGradV2")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&predictShape, &labelShape, &doutShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("mean")}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MseLossGradV2")->infer_shape;
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_FAILED);
}

// 负向场景8（regbase）：predict/label 广播结果与 dout 具体维冲突（32 vs 33），
// 第二次 BroadcastShapes 失败并返回失败（predict rank-1 同时覆盖 da==1 取 db 的分支）
TEST_F(MseLossGradV2Regbase, mse_loss_grad_v2_regbase_dout_conflict_008)
{
    gert::StorageShape predictShape = {{64}, {64}};
    gert::StorageShape labelShape = {{32, 64}, {32, 64}};
    gert::StorageShape doutShape = {{33, 64}, {33, 64}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("MseLossGradV2")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&predictShape, &labelShape, &doutShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("mean")}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MseLossGradV2")->infer_shape;
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_FAILED);
}

// 场景9（regbase）：未知维(-1)按通配处理，两方均为 -1 时取该值，广播结果保留 -1
TEST_F(MseLossGradV2Regbase, mse_loss_grad_v2_regbase_negative_dim_wildcard_009)
{
    gert::StorageShape predictShape = {{-1, 64}, {-1, 64}};
    gert::StorageShape labelShape = {{-1, 64}, {-1, 64}};
    gert::StorageShape doutShape = {{32, 64}, {32, 64}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("MseLossGradV2")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&predictShape, &labelShape, &doutShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("mean")}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MseLossGradV2")->infer_shape;
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

    std::vector<int64_t> expectedyGradShape = {-1, 64};
    auto yGradShape = context->GetOutputShape(0);
    EXPECT_EQ(ToVectorForMseLossGradV2(*yGradShape), expectedyGradShape);
}

// 负向场景10（regbase）：必选输入 dout 的实例个数置 0，regbase 分支 doutShape 判空命中并返回失败
TEST_F(MseLossGradV2Regbase, mse_loss_grad_v2_regbase_null_input_shape_010)
{
    gert::StorageShape predictShape = {{32, 64}, {32, 64}};
    gert::StorageShape labelShape = {{32, 64}, {32, 64}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("MseLossGradV2")
                      .IrInstanceNum({1, 1, 0}, {1})
                      .InputShapes({&predictShape, &labelShape, nullptr})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("mean")}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context->GetInputShape(0), nullptr);
    ASSERT_NE(context->GetInputShape(1), nullptr);
    ASSERT_EQ(context->GetInputShape(2), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MseLossGradV2")->infer_shape;
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_FAILED);
}

// 负向场景11（regbase）：输出 y 的实例个数置 0，regbase 分支 lossGradShape 判空命中并返回失败
TEST_F(MseLossGradV2Regbase, mse_loss_grad_v2_regbase_null_output_shape_011)
{
    gert::StorageShape predictShape = {{32, 64}, {32, 64}};
    gert::StorageShape labelShape = {{32, 64}, {32, 64}};
    gert::StorageShape doutShape = {{32, 64}, {32, 64}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("MseLossGradV2")
                      .IrInstanceNum({1, 1, 1}, {0})
                      .InputShapes({&predictShape, &labelShape, &doutShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("mean")}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    ASSERT_NE(context->GetInputShape(2), nullptr);
    ASSERT_EQ(context->GetOutputShape(0), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MseLossGradV2")->infer_shape;
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_FAILED);
}

// ascend960dt(A6) 分支专用套件：IsRegBaseSoc() 除 Ascend950 外还需匹配 Ascend960DT 的
// short_soc_version，否则 960dt 在 GE 图模式下落入基线分支(输出 = predict shape)，
// 广播用例输出形状推导错误。注入方式与 MseLossGradV2Regbase 一致。
class MseLossGradV2Regbase960dt : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend960DT";
        optiCompilationInfo.soc_version = "Ascend960DT";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend960DT"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
        std::cout << "MseLossGradV2Regbase960dt SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        fe::PlatformInfoManager::Instance().platform_info_map_.erase("Ascend960DT");
        fe::OptionalInfo emptyOptiCompilationInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(emptyOptiCompilationInfo);
        std::cout << "MseLossGradV2Regbase960dt TearDown" << std::endl;
    }
};

// 场景12（960dt regbase 正向）：y.shape = broadcast(dout, predict, label)，
// predict{1,4} x label{3,4} 广播为 {3,4}，再与 dout{3,4} 广播仍为 {3,4}
TEST_F(MseLossGradV2Regbase960dt, mse_loss_grad_v2_regbase_960dt_broadcast_success_012)
{
    gert::StorageShape predictShape = {{1, 4}, {1, 4}};
    gert::StorageShape labelShape = {{3, 4}, {3, 4}};
    gert::StorageShape doutShape = {{3, 4}, {3, 4}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("MseLossGradV2")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&predictShape, &labelShape, &doutShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("mean")}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MseLossGradV2")->infer_shape;
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

    std::vector<int64_t> expectedyGradShape = {3, 4};
    auto yGradShape = context->GetOutputShape(0);
    EXPECT_EQ(ToVectorForMseLossGradV2(*yGradShape), expectedyGradShape);
}

// 负向场景13（960dt regbase）：reduction 取非法值，regbase 分支 IsLegalReduction 校验失败
TEST_F(MseLossGradV2Regbase960dt, mse_loss_grad_v2_regbase_960dt_invalid_reduction_013)
{
    gert::StorageShape predictShape = {{32, 64}, {32, 64}};
    gert::StorageShape labelShape = {{32, 64}, {32, 64}};
    gert::StorageShape doutShape = {{32, 64}, {32, 64}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("MseLossGradV2")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&predictShape, &labelShape, &doutShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("invalid")}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MseLossGradV2")->infer_shape;
    EXPECT_EQ(infer_shape_func(context), ge::GRAPH_FAILED);
}
