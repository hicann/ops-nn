/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"
#include "ut_op_util.h"
#include "../../../../op_kernel/arch35/foreach_erf_tiling_data.h"
#include "../../../../op_kernel/arch35/foreach_erf_tiling_key.h"

using namespace ut_util;

class ForeachErfTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ForeachErfTilingTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "ForeachErfTilingTest TearDown" << std::endl; }
};

static void DoCase(std::initializer_list<int64_t> xShape, ge::DataType dt, uint64_t expectKey)
{
    fe::PlatFormInfos pf;
    std::map<std::string, std::string> soc, ai, intr;
    GetPlatFormInfos(R"({"hardware_info":{"UB_SIZE":253952,"CORE_NUM":64}})", soc, ai, intr);
    pf.Init();

    struct ForeachErfCompileInfo {
    } ci;
    auto opType = std::string("ForeachErf");
    auto tilingFn = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFn = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    ASSERT_NE(tilingFn, nullptr);
    ASSERT_NE(tilingParseFn, nullptr);

    std::string ciStr = R"({"device_id":null})";
    auto kh = gert::KernelRunContextFaker()
                  .KernelIONum(2, 1)
                  .Inputs({const_cast<char*>(ciStr.c_str()), reinterpret_cast<void*>(&pf)})
                  .Outputs({&ci})
                  .Build();
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init();
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc);
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", ai);
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intr);
    ASSERT_EQ(tilingParseFn(kh.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    gert::StorageShape xS = {xShape, xShape};
    gert::StorageShape yS = {xShape, xShape};
    auto param = gert::TilingData::CreateCap(4096);
    auto wsh = gert::ContinuousVector::Create<size_t>(4096);
    auto th = gert::TilingContextFaker()
                  .NodeIoNum(1, 1)
                  .IrInstanceNum({1})
                  .InputShapes({&xS})
                  .OutputShapes({&yS})
                  .CompileInfo(&ci)
                  .PlatformInfo(reinterpret_cast<char*>(&pf))
                  .NodeInputTd(0, dt, ge::FORMAT_ND, ge::FORMAT_ND)
                  .NodeOutputTd(0, dt, ge::FORMAT_ND, ge::FORMAT_ND)
                  .TilingData(param.get())
                  .Workspace(reinterpret_cast<gert::ContinuousVector*>(wsh.get()))
                  .Build();
    auto* ctx = th.GetContext<gert::TilingContext>();
    ctx->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc);
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreSpec", ai);
    ctx->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intr);
    EXPECT_EQ(tilingFn(ctx), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ctx->GetTilingKey(), expectKey);
}

TEST_F(ForeachErfTilingTest, foreach_erf_float32)
{
    DoCase({4, 4}, ge::DT_FLOAT, GET_TPL_TILING_KEY(FOREACH_ERF_TPL_SCH_MODE_FLOAT));
}

TEST_F(ForeachErfTilingTest, foreach_erf_float16)
{
    DoCase({4, 4}, ge::DT_FLOAT16, GET_TPL_TILING_KEY(FOREACH_ERF_TPL_SCH_MODE_FLOAT16));
}

TEST_F(ForeachErfTilingTest, foreach_erf_bfloat16)
{
    DoCase({4, 4}, ge::DT_BF16, GET_TPL_TILING_KEY(FOREACH_ERF_TPL_SCH_MODE_BF16));
}

// ===== 多实例/校验分支用例: 覆盖逐成员 dtype 循环/y dtype/y 容量校验、
// 列表长度上限、全空张量退化、不支持 dtype 与元素总数溢出拒绝路径 =====

static gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    auto setDims = [&dims](gert::Shape& target) {
        target.SetDimNum(dims.size());
        for (size_t i = 0; i < dims.size(); i++) {
            target.SetDim(i, dims[i]);
        }
    };
    setDims(shape.MutableStorageShape());
    setDims(shape.MutableOriginShape());
    return shape;
}

struct ListCaseExpectation {
    ge::graphStatus status = ge::GRAPH_FAILED;
    uint64_t tilingKey = 0;
    int32_t tensorCount = 0;
    int64_t totalElements = -1;
    int32_t needCoreNum = -1;
    std::vector<int64_t> tensorElements;
};

// 多成员单输入列表通用用例: xShapes/yShapes 逐成员, xDtypes.size()==1 表示整表统一 dtype
// (走 NodeInputTd), 否则逐实例 dtype(走 InputTensors 注入); outInstanceNum 为输出列表实例数;
// outDtype 为输出 TensorDesc 声明 dtype(缺省 FLOAT, 与 x 一致时同寻常路径)
static void DoListCase(const std::vector<std::vector<int64_t>>& xShapes,
                       const std::vector<std::vector<int64_t>>& yShapes, const std::vector<ge::DataType>& xDtypes,
                       uint32_t outInstanceNum, const ListCaseExpectation& expect, ge::DataType outDtype = ge::DT_FLOAT)
{
    const size_t inNum = xShapes.size();
    ASSERT_EQ(yShapes.size(), static_cast<size_t>(outInstanceNum));
    fe::PlatFormInfos pf;
    std::map<std::string, std::string> soc, ai, intr;
    GetPlatFormInfos(R"({"hardware_info":{"UB_SIZE":253952,"CORE_NUM":64}})", soc, ai, intr);
    pf.Init();

    struct ForeachErfCompileInfo {
    } ci;
    auto opType = std::string("ForeachErf");
    auto tilingFn = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFn = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    ASSERT_NE(tilingFn, nullptr);
    ASSERT_NE(tilingParseFn, nullptr);

    std::string ciStr = R"({"device_id":null})";
    auto kh = gert::KernelRunContextFaker()
                  .KernelIONum(2, 1)
                  .Inputs({const_cast<char*>(ciStr.c_str()), reinterpret_cast<void*>(&pf)})
                  .Outputs({&ci})
                  .Build();
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init();
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc);
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", ai);
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kh.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intr);
    ASSERT_EQ(tilingParseFn(kh.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    std::vector<gert::StorageShape> xS;
    std::vector<gert::StorageShape> yS;
    for (const auto& dims : xShapes) {
        xS.push_back(MakeStorageShape(dims));
    }
    for (const auto& dims : yShapes) {
        yS.push_back(MakeStorageShape(dims));
    }
    std::vector<gert::StorageShape*> xPtrs;
    std::vector<gert::StorageShape*> yPtrs;
    for (auto& shape : xS) {
        xPtrs.push_back(&shape);
    }
    for (auto& shape : yS) {
        yPtrs.push_back(&shape);
    }

    auto param = gert::TilingData::CreateCap(4096);
    auto wsh = gert::ContinuousVector::Create<size_t>(4096);
    gert::TilingContextFaker th;
    th.NodeIoNum(1, 1)
        .IrInstanceNum({static_cast<uint32_t>(inNum)}, {outInstanceNum})
        .InputShapes(xPtrs)
        .OutputShapes(yPtrs)
        .CompileInfo(&ci)
        .PlatformInfo(reinterpret_cast<char*>(&pf))
        .NodeOutputTd(0, outDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .TilingData(param.get())
        .Workspace(reinterpret_cast<gert::ContinuousVector*>(wsh.get()));
    if (xDtypes.size() == 1U) {
        th.NodeInputTd(0, xDtypes[0], ge::FORMAT_ND, ge::FORMAT_ND);
    } else {
        // NodeInputTd 对同一输入的全部实例统一设置 dtype, 逐实例 dtype 须经 InputTensors 预注入
        // (预注入后 Build 内部的 InputTensors 装配被跳过, 语义等价)
        std::vector<gert::Tensor> tensors(inNum);
        std::vector<gert::Tensor*> tensorPtrs;
        for (size_t i = 0; i < inNum; i++) {
            tensors[i].SetDataType(xDtypes[i]);
            tensors[i].SetOriginFormat(ge::FORMAT_ND);
            tensors[i].SetStorageFormat(ge::FORMAT_ND);
            tensors[i].MutableStorageShape() = xS[i].MutableStorageShape();
            tensors[i].MutableOriginShape() = xS[i].MutableOriginShape();
            tensorPtrs.push_back(&tensors[i]);
        }
        th.InputTensors(tensorPtrs);
    }
    auto holder = th.Build();
    auto* ctx = holder.GetContext<gert::TilingContext>();
    ctx->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc);
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreSpec", ai);
    ctx->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    ctx->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intr);

    EXPECT_EQ(tilingFn(ctx), expect.status);
    if (expect.status == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(ctx->GetTilingKey(), expect.tilingKey);
        const auto* tiling = ctx->GetTilingData<ForeachErfTilingData>();
        ASSERT_NE(tiling, nullptr);
        EXPECT_EQ(tiling->tensorCount, expect.tensorCount);
        if (expect.totalElements >= 0) {
            EXPECT_EQ(tiling->totalElements, expect.totalElements);
        }
        if (expect.needCoreNum >= 0) {
            EXPECT_EQ(tiling->needCoreNum, expect.needCoreNum);
        }
        for (size_t i = 0; i < expect.tensorElements.size(); i++) {
            EXPECT_EQ(tiling->tensorElements[i], expect.tensorElements[i]);
        }
    }
}

// 多成员同 dtype 正向: 覆盖逐成员 dtype 循环与 y 容量检查/多核算核(CeilDiv>coreNum 钳位)
TEST_F(ForeachErfTilingTest, foreach_erf_multi_tensor_success)
{
    ListCaseExpectation exp;
    exp.status = ge::GRAPH_SUCCESS;
    exp.tilingKey = GET_TPL_TILING_KEY(FOREACH_ERF_TPL_SCH_MODE_FLOAT);
    exp.tensorCount = 2;
    exp.totalElements = 65536 + 8; // CeilDiv(65544,1024)=65 > coreNum=64 → min 钳位 64
    exp.needCoreNum = 64;
    exp.tensorElements = {65536, 8};
    DoListCase({{65536}, {8}}, {{65536}, {8}}, {ge::DT_FLOAT}, 2, exp);
}

// 列表内 dtype 不一致拒绝(逐实例 dtype 注入: 成员0 FP32/成员1 FP16)
TEST_F(ForeachErfTilingTest, foreach_erf_dtype_mismatch_in_list_reject)
{
    ListCaseExpectation exp;
    exp.status = ge::GRAPH_FAILED;
    DoListCase({{4, 4}, {8}}, {{4, 4}, {8}}, {ge::DT_FLOAT, ge::DT_FLOAT16}, 2, exp);
}

// 输出列表实例数与输入不一致拒绝(输出实例数校验, 纵深防御)
TEST_F(ForeachErfTilingTest, foreach_erf_out_instance_mismatch_reject)
{
    ListCaseExpectation exp;
    exp.status = ge::GRAPH_FAILED;
    DoListCase({{4, 4}, {8}}, {{4, 4}}, {ge::DT_FLOAT}, 1, exp);
}

// 深 shape 成员放行(ND/elewise 逐元素族不设算子级 shape 限制, shape 自由度由框架层承载):
// x/y 双侧同为 9 轴形状直调 tiling 应成功, 任意 shape 支持面的正向验证
TEST_F(ForeachErfTilingTest, foreach_erf_deep_shape_member_success)
{
    ListCaseExpectation exp;
    exp.status = ge::GRAPH_SUCCESS;
    exp.tilingKey = GET_TPL_TILING_KEY(FOREACH_ERF_TPL_SCH_MODE_FLOAT);
    exp.tensorCount = 1;
    exp.totalElements = 512;
    exp.needCoreNum = 1;
    exp.tensorElements = {512};
    DoListCase({{2, 2, 2, 2, 2, 2, 2, 2, 2}}, {{2, 2, 2, 2, 2, 2, 2, 2, 2}}, {ge::DT_FLOAT}, 1, exp);
}

// y 成员容量小于对应 x 成员拒绝(kernel 按 x 推导写入量的越界写守卫)
TEST_F(ForeachErfTilingTest, foreach_erf_out_capacity_less_reject)
{
    ListCaseExpectation exp;
    exp.status = ge::GRAPH_FAILED;
    DoListCase({{16}, {8}}, {{16}, {4}}, {ge::DT_FLOAT}, 2, exp);
}

// y 容量充足(大于 x)时放行不拒(tiling 门仅拦截容量不足的越界写方向), 正对照
TEST_F(ForeachErfTilingTest, foreach_erf_out_capacity_greater_success)
{
    ListCaseExpectation exp;
    exp.status = ge::GRAPH_SUCCESS;
    exp.tilingKey = GET_TPL_TILING_KEY(FOREACH_ERF_TPL_SCH_MODE_FLOAT);
    exp.tensorCount = 1;
    exp.totalElements = 6;
    exp.needCoreNum = 1;
    exp.tensorElements = {6};
    DoListCase({{2, 3}}, {{4, 3}}, {ge::DT_FLOAT}, 1, exp);
}

// 列表长度超过 256 上限拒绝(与 arch22 共享 tiling 容量 MAX_TENSOR_CONT 对齐)
TEST_F(ForeachErfTilingTest, foreach_erf_tensor_num_exceed_reject)
{
    ListCaseExpectation exp;
    exp.status = ge::GRAPH_FAILED;
    const std::vector<std::vector<int64_t>> shapes(257, {1});
    DoListCase(shapes, shapes, {ge::DT_FLOAT}, 257, exp);
}

// 全空张量列表: totalElements==0 退化单核成功(空 Tensor 支持契约)
TEST_F(ForeachErfTilingTest, foreach_erf_empty_tensors_success)
{
    ListCaseExpectation exp;
    exp.status = ge::GRAPH_SUCCESS;
    exp.tilingKey = GET_TPL_TILING_KEY(FOREACH_ERF_TPL_SCH_MODE_FLOAT);
    exp.tensorCount = 2;
    exp.totalElements = 0;
    exp.needCoreNum = 1;
    exp.tensorElements = {0, 0};
    DoListCase({{0}, {0}}, {{0}, {0}}, {ge::DT_FLOAT}, 2, exp);
}

// 不支持的 dtype(INT32)拒绝: tiling key 仅映射 FP32/FP16/BF16
TEST_F(ForeachErfTilingTest, foreach_erf_unsupported_dtype_reject)
{
    ListCaseExpectation exp;
    exp.status = ge::GRAPH_FAILED;
    DoListCase({{4, 4}}, {{4, 4}}, {ge::DT_INT32}, 1, exp);
}

// y 成员 dtype 与 x 不一致拒绝: kernel 以 x 的 dtype 实例化模板并按该宽度寻址 y 缓冲,
// y 为较窄 dtype 时元素数容量门无法拦截字节级越界写, 须独立校验
TEST_F(ForeachErfTilingTest, foreach_erf_out_dtype_mismatch_reject)
{
    ListCaseExpectation exp;
    exp.status = ge::GRAPH_FAILED;
    DoListCase({{16}}, {{16}}, {ge::DT_FLOAT}, 1, exp, ge::DT_FLOAT16);
}

// 单负维(动态 shape 标记)致元素总数为负拒绝
TEST_F(ForeachErfTilingTest, foreach_erf_negative_numel_reject)
{
    ListCaseExpectation exp;
    exp.status = ge::GRAPH_FAILED;
    DoListCase({{-1, 5}}, {{-1, 5}}, {ge::DT_FLOAT}, 1, exp);
}

// 列表元素总数 int64 累加溢出拒绝(2^62 + 2^62 > INT64_MAX)
TEST_F(ForeachErfTilingTest, foreach_erf_total_elements_overflow_reject)
{
    ListCaseExpectation exp;
    exp.status = ge::GRAPH_FAILED;
    DoListCase({{4611686018427387904}, {4611686018427387904}}, {{4611686018427387904}, {4611686018427387904}},
               {ge::DT_FLOAT}, 2, exp);
}
