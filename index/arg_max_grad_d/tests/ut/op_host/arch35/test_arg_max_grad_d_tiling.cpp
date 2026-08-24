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
 * \file test_arg_max_grad_d_tiling.cpp
 * \brief ArgMaxGradD arch35 tiling UT(需 --soc=ascend950, tiling 受 COMPUTE_UNIT 门控)
 */
#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <vector>

#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class ArgMaxGradDTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ArgMaxGradDTiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "ArgMaxGradDTiling TearDown" << std::endl; }
};

static void InitPlatForm(fe::PlatFormInfos& platFormInfo, map<string, string>& socInfos,
                         map<string, string>& aicoreSpec, map<string, string>& intrinsics)
{
    string hardwareInfo = R"({
        "hardware_info": {"UB_SIZE": 253952, "CORE_NUM": 64}
                          })";
    GetPlatFormInfos(hardwareInfo.c_str(), socInfos, aicoreSpec, intrinsics);
    platFormInfo.Init();
}

// xyz2ShapeOv 用于负向用例: 覆盖 xyz2 的 shape, 缺省表示与 xyz1 一致
// gert::StorageShape 没有接 std::vector 的构造, 逐维 Append
static gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (int64_t d : dims) {
        shape.MutableStorageShape().AppendDim(d);
        shape.MutableOriginShape().AppendDim(d);
    }
    return shape;
}

// 负向用例用 override 参数覆盖对应输入的 shape/dtype, 缺省表示按正常语义推导
struct ArgMaxGradDCase {
    std::vector<int64_t> varShape;
    int64_t dimension = 0;
    ge::DataType varDtype = ge::DT_FLOAT;
    ge::DataType updatesDtype = ge::DT_UNDEFINED; // 缺省=跟随 var
    ge::DataType indicesDtype = ge::DT_INT32;
    ge::DataType assistDtype = ge::DT_INT32;
    std::vector<int64_t> assistShapeOv{};  // 缺省=与 var 同形
    std::vector<int64_t> updatesShapeOv{}; // 缺省=与 indices 同形
};

// 一次 tiling 调用要用到的五个 shape, 打包传递以免调用函数过长
struct ArgMaxGradDShapes {
    gert::StorageShape var;
    gert::StorageShape idx;
    gert::StorageShape upd;
    gert::StorageShape assist;
    gert::StorageShape y;
};

// indices/updates 把 dimension 轴 squeeze 掉(rank(var) == rank(updates) + 1); rank(var)==1 时为 {1}
static std::vector<int64_t> SqueezeDimAxis(const std::vector<int64_t>& varShape, int64_t dimension)
{
    int64_t rank = static_cast<int64_t>(varShape.size());
    int64_t dim = dimension < 0 ? dimension + rank : dimension;
    std::vector<int64_t> idxDims;
    if (rank <= 1 || dim < 0 || dim >= rank) {
        idxDims.push_back(1);
        return idxDims;
    }
    for (int64_t i = 0; i < rank; ++i) {
        if (i != dim) {
            idxDims.push_back(varShape[i]);
        }
    }
    return idxDims;
}

template <typename TilingParseFunc, typename CompileInfoT>
static void RunTilingParse(TilingParseFunc tilingParseFunc, fe::PlatFormInfos& platFormInfo,
                           map<string, string>& socInfos, map<string, string>& aicoreSpec,
                           map<string, string>& intrinsics, CompileInfoT& compileInfo)
{
    string compileInfoStr = R"({"device_id": null})";
    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(compileInfoStr.c_str()), reinterpret_cast<void*>(&platFormInfo)})
                            .Outputs({&compileInfo})
                            .Build();
    auto* parseCtx = kernelHolder.template GetContext<gert::TilingParseContext>();
    EXPECT_TRUE(parseCtx->GetPlatformInfo()->Init());
    parseCtx->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    parseCtx->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    parseCtx->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    parseCtx->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    EXPECT_EQ(tilingParseFunc(kernelHolder.template GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
}

// 构图与执行拆出来, 避免单函数超过 CodeCheck 的 50 行阈值
template <typename CompileInfoT>
static auto MakeTilingHolder(const ArgMaxGradDCase& c, ArgMaxGradDShapes& shapes, ge::DataType updDtype,
                             CompileInfoT& compileInfo, fe::PlatFormInfos& platFormInfo, uint8_t* param,
                             gert::ContinuousVector* wsSize)
{
    return gert::TilingContextFaker()
        .SetOpType("ArgMaxGradD")
        .NodeIoNum(4, 1)
        .IrInstanceNum({1, 1, 1, 1})
        .InputShapes({&shapes.var, &shapes.idx, &shapes.upd, &shapes.assist})
        .OutputShapes({&shapes.y})
        .CompileInfo(&compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
        .NodeInputTd(0, c.varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, c.indicesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, updDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, c.assistDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, c.varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({{"dimension", Ops::NN::AnyValue::CreateFrom<int64_t>(c.dimension)}})
        .TilingData(param)
        .Workspace(wsSize)
        .Build();
}

template <typename TilingFunc, typename HolderT>
static ge::graphStatus RunTiling(TilingFunc tilingFunc, HolderT& holder, map<string, string>& socInfos,
                                 map<string, string>& aicoreSpec, map<string, string>& intrinsics, uint64_t& tilingKey,
                                 uint32_t& blockDim)
{
    gert::TilingContext* tilingContext = holder.template GetContext<gert::TilingContext>();
    EXPECT_NE(tilingContext, nullptr);
    EXPECT_NE(tilingContext->GetPlatformInfo(), nullptr);
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    auto ret = tilingFunc(tilingContext);
    if (ret == ge::GRAPH_SUCCESS) {
        tilingKey = tilingContext->GetTilingKey();
        blockDim = tilingContext->GetBlockDim();
    }
    return ret;
}

static ge::graphStatus DoArgMaxGradDTilingCase(const ArgMaxGradDCase& c, uint64_t& tilingKey, uint32_t& blockDim)
{
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics);

    struct ArgMaxGradDCompileInfo {};
    ArgMaxGradDCompileInfo compileInfo;

    std::string opType("ArgMaxGradD");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str());
    EXPECT_NE(opImpl, nullptr);
    if (opImpl == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto tilingFunc = opImpl->tiling;
    auto tilingParseFunc = opImpl->tiling_parse;

    RunTilingParse(tilingParseFunc, platFormInfo, socInfos, aicoreSpec, intrinsics, compileInfo);

    std::vector<int64_t> idxDims = SqueezeDimAxis(c.varShape, c.dimension);
    std::vector<int64_t> assistDims = c.assistShapeOv.empty() ? c.varShape : c.assistShapeOv;
    std::vector<int64_t> updDims = c.updatesShapeOv.empty() ? idxDims : c.updatesShapeOv;

    ArgMaxGradDShapes shapes{MakeShape(c.varShape), MakeShape(idxDims), MakeShape(updDims), MakeShape(assistDims),
                             MakeShape(c.varShape)};

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    EXPECT_NE(param, nullptr);

    ge::DataType updDtype = c.updatesDtype == ge::DT_UNDEFINED ? c.varDtype : c.updatesDtype;
    auto holder = MakeTilingHolder(c, shapes, updDtype, compileInfo, platFormInfo, param.get(), wsSize);
    return RunTiling(tilingFunc, holder, socInfos, aicoreSpec, intrinsics, tilingKey, blockDim);
}

TEST_F(ArgMaxGradDTiling, test_tiling_fp32_inner_multi)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoArgMaxGradDTilingCase({{4, 16, 8}, 1, ge::DT_FLOAT}, tilingKey, blockDim), ge::GRAPH_SUCCESS);
    // 切核按输出元素总数(4*16*8=512)且边界 32B 对齐: fp32 一块 8 个元素, 每核 8 个 → 64 核全用上
    EXPECT_EQ(blockDim, 64U);
}

// outer 恰好等于核数: 满核且无尾核
TEST_F(ArgMaxGradDTiling, test_tiling_full_core)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoArgMaxGradDTilingCase({{64, 16, 8}, 1, ge::DT_FLOAT}, tilingKey, blockDim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(blockDim, 64U);
}

// outer 超过核数且不整除: 满核带尾核
TEST_F(ArgMaxGradDTiling, test_tiling_tail_core)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoArgMaxGradDTilingCase({{129, 16, 8}, 1, ge::DT_FLOAT}, tilingKey, blockDim), ge::GRAPH_SUCCESS);
    EXPECT_LE(blockDim, 64U);
    EXPECT_GT(blockDim, 0U);
}

TEST_F(ArgMaxGradDTiling, test_tiling_fp16_inner_multi)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoArgMaxGradDTilingCase({{4, 16, 8}, 1, ge::DT_FLOAT16}, tilingKey, blockDim), ge::GRAPH_SUCCESS);
}

TEST_F(ArgMaxGradDTiling, test_tiling_int32_inner_multi)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoArgMaxGradDTilingCase({{4, 16, 8}, 1, ge::DT_INT32}, tilingKey, blockDim), ge::GRAPH_SUCCESS);
}

TEST_F(ArgMaxGradDTiling, test_tiling_int8_inner_multi)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoArgMaxGradDTilingCase({{4, 16, 8}, 1, ge::DT_INT8}, tilingKey, blockDim), ge::GRAPH_SUCCESS);
}

// inner==1(dimension 指向最后一维)与 inner>1 必须落到不同的 TilingKey
TEST_F(ArgMaxGradDTiling, test_tiling_key_differs_by_inner)
{
    uint64_t keyMulti = 0;
    uint64_t keyOne = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoArgMaxGradDTilingCase({{4, 16, 8}, 1, ge::DT_FLOAT}, keyMulti, blockDim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(DoArgMaxGradDTilingCase({{4, 128}, 1, ge::DT_FLOAT}, keyOne, blockDim), ge::GRAPH_SUCCESS);
    EXPECT_NE(keyMulti, keyOne);
}

// dtype 不进 TilingKey: 同一形态下四种 dtype 的 key 必须相同
TEST_F(ArgMaxGradDTiling, test_tiling_key_independent_of_dtype)
{
    uint64_t k32 = 0;
    uint64_t k16 = 0;
    uint64_t ki32 = 0;
    uint64_t ki8 = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoArgMaxGradDTilingCase({{4, 16, 8}, 1, ge::DT_FLOAT}, k32, blockDim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(DoArgMaxGradDTilingCase({{4, 16, 8}, 1, ge::DT_FLOAT16}, k16, blockDim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(DoArgMaxGradDTilingCase({{4, 16, 8}, 1, ge::DT_INT32}, ki32, blockDim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(DoArgMaxGradDTilingCase({{4, 16, 8}, 1, ge::DT_INT8}, ki8, blockDim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(k32, k16);
    EXPECT_EQ(k32, ki32);
    EXPECT_EQ(k32, ki8);
}

TEST_F(ArgMaxGradDTiling, test_tiling_negative_dimension)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    // dimension=-2 归一化后为 1, 与显式传 1 等价
    uint64_t keyPos = 0;
    EXPECT_EQ(DoArgMaxGradDTilingCase({{4, 16, 8}, -2, ge::DT_FLOAT}, tilingKey, blockDim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(DoArgMaxGradDTilingCase({{4, 16, 8}, 1, ge::DT_FLOAT}, keyPos, blockDim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingKey, keyPos);
}

TEST_F(ArgMaxGradDTiling, test_tiling_rank1)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoArgMaxGradDTilingCase({{64}, 0, ge::DT_FLOAT}, tilingKey, blockDim), ge::GRAPH_SUCCESS);
}

TEST_F(ArgMaxGradDTiling, test_tiling_big_axis_multi_chunk)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoArgMaxGradDTilingCase({{1, 4, 40000}, 1, ge::DT_FLOAT}, tilingKey, blockDim), ge::GRAPH_SUCCESS);
}

TEST_F(ArgMaxGradDTiling, test_tiling_empty_tensor)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoArgMaxGradDTilingCase({{4, 0, 8}, 1, ge::DT_FLOAT}, tilingKey, blockDim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(blockDim, 1U); // 空 tensor 不进核, 但 blockDim 必须合法
}

TEST_F(ArgMaxGradDTiling, test_tiling_invalid_var_dtype)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoArgMaxGradDTilingCase({{4, 16, 8}, 1, ge::DT_UINT8}, tilingKey, blockDim), ge::GRAPH_FAILED);
}

TEST_F(ArgMaxGradDTiling, test_tiling_updates_dtype_mismatch)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoArgMaxGradDTilingCase({{4, 16, 8}, 1, ge::DT_FLOAT, ge::DT_FLOAT16}, tilingKey, blockDim),
              ge::GRAPH_FAILED);
}

TEST_F(ArgMaxGradDTiling, test_tiling_indices_dtype_invalid)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(
        DoArgMaxGradDTilingCase({{4, 16, 8}, 1, ge::DT_FLOAT, ge::DT_UNDEFINED, ge::DT_INT64}, tilingKey, blockDim),
        ge::GRAPH_FAILED);
}

TEST_F(ArgMaxGradDTiling, test_tiling_assist_shape_mismatch)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    ArgMaxGradDCase c{{4, 16, 8}, 1, ge::DT_FLOAT};
    c.assistShapeOv = {4, 16, 4};
    EXPECT_EQ(DoArgMaxGradDTilingCase(c, tilingKey, blockDim), ge::GRAPH_FAILED);
}

TEST_F(ArgMaxGradDTiling, test_tiling_updates_shape_mismatch)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    ArgMaxGradDCase c{{4, 16, 8}, 1, ge::DT_FLOAT};
    c.updatesShapeOv = {4, 4};
    EXPECT_EQ(DoArgMaxGradDTilingCase(c, tilingKey, blockDim), ge::GRAPH_FAILED);
}

TEST_F(ArgMaxGradDTiling, test_tiling_dimension_out_of_range)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoArgMaxGradDTilingCase({{4, 16, 8}, 3, ge::DT_FLOAT}, tilingKey, blockDim), ge::GRAPH_FAILED);
    EXPECT_EQ(DoArgMaxGradDTilingCase({{4, 16, 8}, -4, ge::DT_FLOAT}, tilingKey, blockDim), ge::GRAPH_FAILED);
}
