/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "../../../op_graph/inplace_apply_adagrad_da_proto.h"
// op_host UT 二进制不链接 op_graph 模块, InferDataType 的注册位于
// op_graph/inplace_apply_adagrad_da_graph_infer.cpp; 在此处引入源文件以使
// GetOpImpl("InplaceApplyAdagradDA")->infer_datatype 在 UT 中可用
// (生产交付中 infershape/graph_infer/optiling 以多注册器方式合并同一 op 类型, 语义一致)。
#include "../../../op_graph/inplace_apply_adagrad_da_graph_infer.cpp"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"

class InplaceApplyAdagradDAInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "InplaceApplyAdagradDA InferShape Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "InplaceApplyAdagradDA InferShape Test TearDown" << std::endl; }

    static void InitPlatform()
    {
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend950";
        optiCompilationInfo.soc_version = "Ascend950";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    // 构建 InferShape 上下文: inputs 为全部输入 shape(数量决定节点 IO 数), outputs 为 3 个输出 shape。
    static gert::KernelRunContextHolder MakeHolder(const std::vector<gert::Shape*>& inputs,
                                                   const std::vector<gert::Shape*>& outputs)
    {
        gert::InferShapeContextFaker faker;
        faker.NodeIoNum(inputs.size(), outputs.size()).InputShapes(inputs).OutputShapes(outputs);
        for (size_t i = 0; i < inputs.size(); ++i) {
            faker.NodeInputTd(i, (i == 7) ? ge::DT_INT32 : ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
        }
        for (size_t i = 0; i < outputs.size(); ++i) {
            faker.NodeOutputTd(i, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
        }
        return faker.Build();
    }

    // 标准合法输入: var 为 shapes[0], 输入 1-3 与 var 同 shape, 输入 4-7 为 [1] 标量。
    static std::vector<gert::Shape*> StandardInputs(gert::Shape* varShape, gert::Shape* scalarShape,
                                                    gert::Shape* overrideShape = nullptr, size_t overrideIndex = 0)
    {
        std::vector<gert::Shape*> inputs;
        for (size_t i = 0; i < 8; ++i) {
            if (i == overrideIndex && overrideShape != nullptr) {
                inputs.push_back(overrideShape);
            } else if (i >= 4) {
                inputs.push_back(scalarShape);
            } else {
                inputs.push_back(varShape);
            }
        }
        return inputs;
    }
};

// ---------- 正向用例(合法输入必须成功, 输出与 var 同 shape) ----------

TEST_F(InplaceApplyAdagradDAInferShapeTest, infershape_1d_fp32_test)
{
    InitPlatform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceApplyAdagradDA")->infer_shape;

    gert::Shape var_shape = {128};
    gert::Shape scalar_shape = {1};
    gert::Shape out_shape_0 = {};
    gert::Shape out_shape_1 = {};
    gert::Shape out_shape_2 = {};

    auto holder = MakeHolder(StandardInputs(&var_shape, &scalar_shape), {&out_shape_0, &out_shape_1, &out_shape_2});
    auto* context = holder.GetContext<gert::InferShapeContext>();

    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(*context->GetOutputShape(0), var_shape);
    EXPECT_EQ(*context->GetOutputShape(1), var_shape);
    EXPECT_EQ(*context->GetOutputShape(2), var_shape);
}

TEST_F(InplaceApplyAdagradDAInferShapeTest, infershape_2d_fp16_test)
{
    InitPlatform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceApplyAdagradDA")->infer_shape;

    gert::Shape var_shape = {16, 32};
    gert::Shape scalar_shape = {1};
    gert::Shape out_shape_0 = {};
    gert::Shape out_shape_1 = {};
    gert::Shape out_shape_2 = {};

    auto holder = MakeHolder(StandardInputs(&var_shape, &scalar_shape), {&out_shape_0, &out_shape_1, &out_shape_2});
    auto* context = holder.GetContext<gert::InferShapeContext>();

    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(*context->GetOutputShape(0), var_shape);
}

TEST_F(InplaceApplyAdagradDAInferShapeTest, infershape_3d_fp16_test)
{
    InitPlatform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceApplyAdagradDA")->infer_shape;

    gert::Shape var_shape = {4, 8, 16};
    gert::Shape scalar_shape = {1};
    gert::Shape out_shape_0 = {};
    gert::Shape out_shape_1 = {};
    gert::Shape out_shape_2 = {};

    auto holder = MakeHolder(StandardInputs(&var_shape, &scalar_shape), {&out_shape_0, &out_shape_1, &out_shape_2});
    auto* context = holder.GetContext<gert::InferShapeContext>();

    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(*context->GetOutputShape(0), var_shape);
}

TEST_F(InplaceApplyAdagradDAInferShapeTest, infershape_scalar_test)
{
    InitPlatform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceApplyAdagradDA")->infer_shape;

    // rank-0 标量 var 为公开契约合法输入 (tiling 侧 EnsureNotScalar 归一化为 {1})
    gert::Shape var_shape = {};
    gert::Shape scalar_shape = {1};
    gert::Shape out_shape_0 = {};
    gert::Shape out_shape_1 = {};
    gert::Shape out_shape_2 = {};

    auto holder = MakeHolder(StandardInputs(&var_shape, &scalar_shape), {&out_shape_0, &out_shape_1, &out_shape_2});
    auto* context = holder.GetContext<gert::InferShapeContext>();

    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(*context->GetOutputShape(0), var_shape);
}

TEST_F(InplaceApplyAdagradDAInferShapeTest, infershape_high_rank_test)
{
    InitPlatform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceApplyAdagradDA")->infer_shape;

    // rank-5 在 FORMAT_ND 公开契约 rank 0..8 之内
    gert::Shape var_shape = {2, 3, 4, 5, 6};
    gert::Shape scalar_shape = {1};
    gert::Shape out_shape_0 = {};
    gert::Shape out_shape_1 = {};
    gert::Shape out_shape_2 = {};

    auto holder = MakeHolder(StandardInputs(&var_shape, &scalar_shape), {&out_shape_0, &out_shape_1, &out_shape_2});
    auto* context = holder.GetContext<gert::InferShapeContext>();

    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(*context->GetOutputShape(0), var_shape);
}

TEST_F(InplaceApplyAdagradDAInferShapeTest, infershape_empty_tensor_test)
{
    InitPlatform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceApplyAdagradDA")->infer_shape;

    // 0 元素张量为合法输入 (tiling 侧提前成功返回)
    gert::Shape var_shape = {0};
    gert::Shape scalar_shape = {1};
    gert::Shape out_shape_0 = {};
    gert::Shape out_shape_1 = {};
    gert::Shape out_shape_2 = {};

    auto holder = MakeHolder(StandardInputs(&var_shape, &scalar_shape), {&out_shape_0, &out_shape_1, &out_shape_2});
    auto* context = holder.GetContext<gert::InferShapeContext>();

    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(*context->GetOutputShape(0), var_shape);
}

TEST_F(InplaceApplyAdagradDAInferShapeTest, infershape_dynamic_tensor_inputs_test)
{
    InitPlatform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceApplyAdagradDA")->infer_shape;

    // 动态 shape 图(如 GEIR dynamic 模式): 张量输入含 -1 未知维, 图构造期无法判定关系,
    // Host 门禁放行, 输出透传 var shape, 关系校验由 tiling 以绑定后 shape 把关
    gert::Shape var_shape = {2, 3};
    gert::Shape dynamic_shape = {-1, -1};
    gert::Shape scalar_shape = {1};
    gert::Shape out_shape_0 = {};
    gert::Shape out_shape_1 = {};
    gert::Shape out_shape_2 = {};

    auto holder = MakeHolder(StandardInputs(&var_shape, &scalar_shape, &dynamic_shape, 1),
                             {&out_shape_0, &out_shape_1, &out_shape_2});
    auto* context = holder.GetContext<gert::InferShapeContext>();

    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(*context->GetOutputShape(0), var_shape);
    EXPECT_EQ(*context->GetOutputShape(2), var_shape);
}

TEST_F(InplaceApplyAdagradDAInferShapeTest, infershape_all_dynamic_test)
{
    InitPlatform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceApplyAdagradDA")->infer_shape;

    // var 与张量输入均为动态 shape: 输出透传 var 的动态 shape
    gert::Shape var_shape = {-1, -1};
    gert::Shape scalar_shape = {1};
    gert::Shape out_shape_0 = {};
    gert::Shape out_shape_1 = {};
    gert::Shape out_shape_2 = {};

    auto holder = MakeHolder(StandardInputs(&var_shape, &scalar_shape), {&out_shape_0, &out_shape_1, &out_shape_2});
    auto* context = holder.GetContext<gert::InferShapeContext>();

    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(*context->GetOutputShape(0), var_shape);
}

TEST_F(InplaceApplyAdagradDAInferShapeTest, infershape_dynamic_scalar_test)
{
    InitPlatform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceApplyAdagradDA")->infer_shape;

    // 标量输入含未知维 [-1]: 无法在图构造期判定 [1], 放行并留给 tiling
    gert::Shape var_shape = {128};
    gert::Shape scalar_shape = {1};
    gert::Shape dynamic_scalar = {-1};
    gert::Shape out_shape_0 = {};
    gert::Shape out_shape_1 = {};
    gert::Shape out_shape_2 = {};

    auto holder = MakeHolder(StandardInputs(&var_shape, &scalar_shape, &dynamic_scalar, 4),
                             {&out_shape_0, &out_shape_1, &out_shape_2});
    auto* context = holder.GetContext<gert::InferShapeContext>();

    ASSERT_EQ(inferShapeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(*context->GetOutputShape(0), var_shape);
}

// ---------- 负向用例: rank 超限 ----------

TEST_F(InplaceApplyAdagradDAInferShapeTest, infershape_rank9_var_reject_test)
{
    InitPlatform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceApplyAdagradDA")->infer_shape;

    // FORMAT_ND 公开契约 rank 0..8, rank-9 必须在 Host 推理门禁独立拒绝
    gert::Shape var_shape = {2, 3, 4, 5, 6, 7, 8, 9, 10};
    gert::Shape scalar_shape = {1};
    gert::Shape out_shape_0 = {};
    gert::Shape out_shape_1 = {};
    gert::Shape out_shape_2 = {};

    auto holder = MakeHolder(StandardInputs(&var_shape, &scalar_shape), {&out_shape_0, &out_shape_1, &out_shape_2});

    EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// ---------- 负向用例: 张量输入 shape 关系违约 ----------

TEST_F(InplaceApplyAdagradDAInferShapeTest, infershape_tensor_mismatch_reject_test)
{
    InitPlatform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceApplyAdagradDA")->infer_shape;

    gert::Shape var_shape = {128};
    gert::Shape wrong_shape = {128, 2};
    gert::Shape scalar_shape = {1};
    gert::Shape out_shape_0 = {};
    gert::Shape out_shape_1 = {};
    gert::Shape out_shape_2 = {};

    // gradient_accumulator(1) / gradient_squared_accumulator(2) / grad(3) 与 var 不同 shape
    for (size_t k = 1; k <= 3; ++k) {
        auto holder = MakeHolder(StandardInputs(&var_shape, &scalar_shape, &wrong_shape, k),
                                 {&out_shape_0, &out_shape_1, &out_shape_2});
        EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED)
            << "input index " << k;
    }
}

// ---------- 负向用例: 标量输入非 [1] ----------

TEST_F(InplaceApplyAdagradDAInferShapeTest, infershape_scalar_nonscalar_reject_test)
{
    InitPlatform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceApplyAdagradDA")->infer_shape;

    gert::Shape var_shape = {128};
    gert::Shape scalar_shape = {1};
    gert::Shape wrong_shape = {2};
    gert::Shape wrong_rank0 = {};
    gert::Shape out_shape_0 = {};
    gert::Shape out_shape_1 = {};
    gert::Shape out_shape_2 = {};

    // lr(4) / l1(5) / l2(6) / global_step(7) 必须为 [1]
    for (size_t k = 4; k <= 7; ++k) {
        auto holder = MakeHolder(StandardInputs(&var_shape, &scalar_shape, &wrong_shape, k),
                                 {&out_shape_0, &out_shape_1, &out_shape_2});
        EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED)
            << "input index " << k;
        auto holderRank0 = MakeHolder(StandardInputs(&var_shape, &scalar_shape, &wrong_rank0, k),
                                      {&out_shape_0, &out_shape_1, &out_shape_2});
        EXPECT_EQ(inferShapeFunc(holderRank0.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED)
            << "input index " << k << " rank-0";
    }
}

// ---------- 负向用例: 必选输入缺失 ----------

TEST_F(InplaceApplyAdagradDAInferShapeTest, infershape_missing_input_reject_test)
{
    InitPlatform();
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceApplyAdagradDA")->infer_shape;

    gert::Shape var_shape = {128};
    gert::Shape scalar_shape = {1};
    gert::Shape out_shape_0 = {};
    gert::Shape out_shape_1 = {};
    gert::Shape out_shape_2 = {};

    // global_step 缺失: 7 输入节点, GetInputShape(7) 返回空指针
    {
        std::vector<gert::Shape*> sevenInputs = {&var_shape,    &var_shape,    &var_shape,   &var_shape,
                                                 &scalar_shape, &scalar_shape, &scalar_shape};
        auto holder = MakeHolder(sevenInputs, {&out_shape_0, &out_shape_1, &out_shape_2});
        EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
    }

    // gradient_accumulator 缺失: 输入集少一项, 尾部索引出现空指针
    {
        std::vector<gert::Shape*> sevenInputs = {&var_shape,    &var_shape,    &var_shape,   &scalar_shape,
                                                 &scalar_shape, &scalar_shape, &scalar_shape};
        auto holder = MakeHolder(sevenInputs, {&out_shape_0, &out_shape_1, &out_shape_2});
        EXPECT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
    }
}

class InplaceApplyAdagradDAInferDataTypeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "InplaceApplyAdagradDA InferDataType Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "InplaceApplyAdagradDA InferDataType Test TearDown" << std::endl; }

    static void InitPlatform()
    {
        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend950";
        optiCompilationInfo.soc_version = "Ascend950";
        fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    // 输入索引（var=0, gradient_accumulator=1, gradient_squared_accumulator=2, grad=3,
    //            lr=4, l1=5, l2=6, global_step=7）
    static constexpr size_t kInputCount = 8;
    static constexpr size_t kGlobalStepIndex = 7;

    // 构建 8 输入 3 输出的 InferDataType 上下文; dtypes 为 8 个输入 dtype。
    static gert::KernelRunContextHolder MakeHolder(const std::vector<ge::DataType>& dtypes,
                                                   std::vector<ge::DataType>& outputs)
    {
        std::vector<ge::DataType> inputs = dtypes;
        gert::InferDataTypeContextFaker faker;
        faker.NodeIoNum(inputs.size(), outputs.size())
            .InputDataTypes(MakeTypePtrs(inputs))
            .OutputDataTypes(MakeTypePtrs(outputs));
        for (size_t i = 0; i < inputs.size(); ++i) {
            faker.NodeInputTd(i, inputs[i], ge::FORMAT_ND, ge::FORMAT_ND);
        }
        for (size_t i = 0; i < outputs.size(); ++i) {
            faker.NodeOutputTd(i, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
        }
        return faker.Build();
    }

    static std::vector<void*> MakeTypePtrs(std::vector<ge::DataType>& dtypes)
    {
        std::vector<void*> ptrs;
        for (auto& dtype : dtypes) {
            ptrs.push_back(&dtype);
        }
        return ptrs;
    }

    // 标准合法 dtype 组合: 张量输入全为 varDtype, global_step 为 gsDtype。
    static std::vector<ge::DataType> StandardDtypes(ge::DataType varDtype, ge::DataType gsDtype,
                                                    ge::DataType overrideDtype = ge::DT_UNDEFINED,
                                                    size_t overrideIndex = 0)
    {
        std::vector<ge::DataType> dtypes(kInputCount, varDtype);
        dtypes[kGlobalStepIndex] = gsDtype;
        if (overrideIndex < kInputCount && overrideDtype != ge::DT_UNDEFINED) {
            dtypes[overrideIndex] = overrideDtype;
        }
        return dtypes;
    }
};

// ---------- 正向用例(合法 dtype 必须成功, 输出与 var 同 dtype) ----------

TEST_F(InplaceApplyAdagradDAInferDataTypeTest, inferdatatype_fp32_int32_test)
{
    InitPlatform();
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceApplyAdagradDA")->infer_datatype;

    std::vector<ge::DataType> outputs = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
    auto holder = MakeHolder(StandardDtypes(ge::DT_FLOAT, ge::DT_INT32), outputs);

    ASSERT_EQ(dataTypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    auto* context = holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_FLOAT);
}

TEST_F(InplaceApplyAdagradDAInferDataTypeTest, inferdatatype_fp16_int64_test)
{
    InitPlatform();
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceApplyAdagradDA")->infer_datatype;

    std::vector<ge::DataType> outputs = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16};
    auto holder = MakeHolder(StandardDtypes(ge::DT_FLOAT16, ge::DT_INT64), outputs);

    ASSERT_EQ(dataTypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_SUCCESS);
    auto* context = holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

// ---------- 负向用例: var dtype 不支持 ----------

TEST_F(InplaceApplyAdagradDAInferDataTypeTest, inferdatatype_var_unsupported_reject_test)
{
    InitPlatform();
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceApplyAdagradDA")->infer_datatype;

    std::vector<ge::DataType> outputs = {ge::DT_DOUBLE, ge::DT_DOUBLE, ge::DT_DOUBLE};
    auto holder = MakeHolder(StandardDtypes(ge::DT_DOUBLE, ge::DT_INT32), outputs);

    EXPECT_EQ(dataTypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_FAILED);
}

// ---------- 负向用例: 输入 1..6 dtype 不支持 / 与 var 不匹配 ----------

TEST_F(InplaceApplyAdagradDAInferDataTypeTest, inferdatatype_input_unsupported_reject_test)
{
    InitPlatform();
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceApplyAdagradDA")->infer_datatype;

    std::vector<ge::DataType> outputs = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
    // gradient_accumulator(1) ... l2(6) 逐一置为 DT_DOUBLE(不在 {FLOAT, FLOAT16} 支持域)
    for (size_t k = 1; k <= 6; ++k) {
        auto holder = MakeHolder(StandardDtypes(ge::DT_FLOAT, ge::DT_INT32, ge::DT_DOUBLE, k), outputs);
        EXPECT_EQ(dataTypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_FAILED)
            << "input index " << k;
    }
}

TEST_F(InplaceApplyAdagradDAInferDataTypeTest, inferdatatype_input_mismatch_reject_test)
{
    InitPlatform();
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceApplyAdagradDA")->infer_datatype;

    std::vector<ge::DataType> outputs = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
    // var=FLOAT, 输入 1..6 逐一置为 FLOAT16(dtype 与 var 不一致)
    for (size_t k = 1; k <= 6; ++k) {
        auto holder = MakeHolder(StandardDtypes(ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT16, k), outputs);
        EXPECT_EQ(dataTypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_FAILED)
            << "input index " << k;
    }
}

// ---------- 负向用例: global_step dtype 不支持 ----------

TEST_F(InplaceApplyAdagradDAInferDataTypeTest, inferdatatype_global_step_unsupported_reject_test)
{
    InitPlatform();
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceApplyAdagradDA")->infer_datatype;

    std::vector<ge::DataType> outputs = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
    auto holder = MakeHolder(StandardDtypes(ge::DT_FLOAT, ge::DT_FLOAT), outputs);

    EXPECT_EQ(dataTypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_FAILED);
}

// ---------- 负向用例: global_step 缺失(越界返回 DT_UNDEFINED) ----------

TEST_F(InplaceApplyAdagradDAInferDataTypeTest, inferdatatype_missing_input_reject_test)
{
    InitPlatform();
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceApplyAdagradDA")->infer_datatype;

    // 7 输入节点: GetInputDataType(7) 返回 DT_UNDEFINED, 必须被 dtype 域校验拒绝
    std::vector<ge::DataType> inputs = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT,
                                        ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
    std::vector<ge::DataType> outputs = {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT};
    gert::InferDataTypeContextFaker faker;
    faker.NodeIoNum(7, outputs.size())
        .InputDataTypes({&inputs[0], &inputs[1], &inputs[2], &inputs[3], &inputs[4], &inputs[5], &inputs[6]});
    faker.OutputDataTypes({&outputs[0], &outputs[1], &outputs[2]});
    for (size_t i = 0; i < 7; ++i) {
        faker.NodeInputTd(i, inputs[i], ge::FORMAT_ND, ge::FORMAT_ND);
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        faker.NodeOutputTd(i, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    auto holder = faker.Build();

    EXPECT_EQ(dataTypeFunc(holder.GetContext<gert::InferDataTypeContext>()), ge::GRAPH_FAILED);
}
