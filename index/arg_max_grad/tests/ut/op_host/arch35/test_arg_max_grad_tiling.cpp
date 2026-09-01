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
 * \file test_arg_max_grad_tiling.cpp
 * \brief ArgMaxGrad arch35 tiling UT(需 --soc=ascend950, tiling 受 COMPUTE_UNIT 门控)
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
#include "../../../../op_kernel/arch35/arg_max_grad_tiling_data.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class ArgMaxGradTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ArgMaxGradTiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "ArgMaxGradTiling TearDown" << std::endl; }
};

static void InitPlatForm(fe::PlatFormInfos& platFormInfo, map<string, string>& socInfos,
                         map<string, string>& aicoreSpec, map<string, string>& intrinsics, int64_t ubSize = 253952,
                         int64_t coreNum = 64)
{
    // 平台参数可覆盖: 用于验证 tiling 对异常平台信息的拒收(见 test_tiling_platform_* 三例)
    string hardwareInfo = R"({
        "hardware_info": {"UB_SIZE": )" +
                          std::to_string(ubSize) + R"(, "CORE_NUM": )" + std::to_string(coreNum) + R"(}
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
struct ArgMaxGradCase {
    std::vector<int64_t> varShape;
    int64_t dimension = 0;
    ge::DataType varDtype = ge::DT_FLOAT;
    ge::DataType updatesDtype = ge::DT_UNDEFINED; // 缺省=跟随 var
    ge::DataType indicesDtype = ge::DT_INT32;
    std::vector<int64_t> updatesShapeOv{}; // 缺省=与 indices 同形
};

// 一次 tiling 调用要用到的四个 shape, 打包传递以免调用函数过长
struct ArgMaxGradShapes {
    gert::StorageShape var;
    gert::StorageShape idx;
    gert::StorageShape upd;
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
static auto MakeTilingHolder(const ArgMaxGradCase& c, ArgMaxGradShapes& shapes, ge::DataType updDtype,
                             CompileInfoT& compileInfo, fe::PlatFormInfos& platFormInfo, uint8_t* param,
                             gert::ContinuousVector* wsSize)
{
    return gert::TilingContextFaker()
        .SetOpType("ArgMaxGrad")
        .NodeIoNum(3, 1)
        .IrInstanceNum({1, 1, 1})
        .InputShapes({&shapes.var, &shapes.idx, &shapes.upd})
        .OutputShapes({&shapes.y})
        .CompileInfo(&compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
        .NodeInputTd(0, c.varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, c.indicesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, updDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, c.varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({{"dimension", Ops::NN::AnyValue::CreateFrom<int64_t>(c.dimension)}})
        .TilingData(param)
        .Workspace(wsSize)
        .Build();
}

template <typename TilingFunc, typename HolderT>
static ge::graphStatus RunTiling(TilingFunc tilingFunc, HolderT& holder, map<string, string>& socInfos,
                                 map<string, string>& aicoreSpec, map<string, string>& intrinsics, uint64_t& tilingKey,
                                 uint32_t& blockDim, ArgMaxGradArch35TilingData* tdOut)
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
        auto raw = tilingContext->GetRawTilingData();
        if (tdOut != nullptr && raw != nullptr && raw->GetDataSize() >= sizeof(ArgMaxGradArch35TilingData)) {
            *tdOut = *reinterpret_cast<const ArgMaxGradArch35TilingData*>(raw->GetData());
        }
    }
    return ret;
}

static ge::graphStatus DoArgMaxGradTilingCase(const ArgMaxGradCase& c, uint64_t& tilingKey, uint32_t& blockDim,
                                              ArgMaxGradArch35TilingData* tdOut = nullptr, int64_t ubSize = 253952,
                                              int64_t coreNum = 64)
{
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics, ubSize, coreNum);

    struct ArgMaxGradCompileInfo {};
    ArgMaxGradCompileInfo compileInfo;

    std::string opType("ArgMaxGrad");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str());
    EXPECT_NE(opImpl, nullptr);
    if (opImpl == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto tilingFunc = opImpl->tiling;
    auto tilingParseFunc = opImpl->tiling_parse;

    RunTilingParse(tilingParseFunc, platFormInfo, socInfos, aicoreSpec, intrinsics, compileInfo);

    std::vector<int64_t> idxDims = SqueezeDimAxis(c.varShape, c.dimension);
    std::vector<int64_t> updDims = c.updatesShapeOv.empty() ? idxDims : c.updatesShapeOv;

    ArgMaxGradShapes shapes{MakeShape(c.varShape), MakeShape(idxDims), MakeShape(updDims), MakeShape(c.varShape)};

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    EXPECT_NE(param, nullptr);

    ge::DataType updDtype = c.updatesDtype == ge::DT_UNDEFINED ? c.varDtype : c.updatesDtype;
    auto holder = MakeTilingHolder(c, shapes, updDtype, compileInfo, platFormInfo, param.get(), wsSize);
    return RunTiling(tilingFunc, holder, socInfos, aicoreSpec, intrinsics, tilingKey, blockDim, tdOut);
}

TEST_F(ArgMaxGradTiling, test_tiling_fp32_inner_multi)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoArgMaxGradTilingCase({{4, 16, 8}, 1, ge::DT_FLOAT}, tilingKey, blockDim), ge::GRAPH_SUCCESS);
    // 切核按输出元素总数(4*16*8=512)且边界 32B 对齐: fp32 一块 8 个元素, 每核 8 个 → 64 核全用上
    EXPECT_EQ(blockDim, 64U);
}

// outer 恰好等于核数: 满核且无尾核
TEST_F(ArgMaxGradTiling, test_tiling_full_core)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoArgMaxGradTilingCase({{64, 16, 8}, 1, ge::DT_FLOAT}, tilingKey, blockDim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(blockDim, 64U);
}

// outer 超过核数且不整除: 满核带尾核
TEST_F(ArgMaxGradTiling, test_tiling_tail_core)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoArgMaxGradTilingCase({{129, 16, 8}, 1, ge::DT_FLOAT}, tilingKey, blockDim), ge::GRAPH_SUCCESS);
    EXPECT_LE(blockDim, 64U);
    EXPECT_GT(blockDim, 0U);
}

TEST_F(ArgMaxGradTiling, test_tiling_fp16_inner_multi)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoArgMaxGradTilingCase({{4, 16, 8}, 1, ge::DT_FLOAT16}, tilingKey, blockDim), ge::GRAPH_SUCCESS);
}

TEST_F(ArgMaxGradTiling, test_tiling_int32_inner_multi)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoArgMaxGradTilingCase({{4, 16, 8}, 1, ge::DT_INT32}, tilingKey, blockDim), ge::GRAPH_SUCCESS);
}

TEST_F(ArgMaxGradTiling, test_tiling_int8_inner_multi)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoArgMaxGradTilingCase({{4, 16, 8}, 1, ge::DT_INT8}, tilingKey, blockDim), ge::GRAPH_SUCCESS);
}

// inner==1(dimension 指向最后一维)与 inner>1 必须落到不同的 TilingKey
TEST_F(ArgMaxGradTiling, test_tiling_key_differs_by_inner)
{
    uint64_t keyMulti = 0;
    uint64_t keyOne = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoArgMaxGradTilingCase({{4, 16, 8}, 1, ge::DT_FLOAT}, keyMulti, blockDim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(DoArgMaxGradTilingCase({{4, 128}, 1, ge::DT_FLOAT}, keyOne, blockDim), ge::GRAPH_SUCCESS);
    EXPECT_NE(keyMulti, keyOne);
}

// dtype 不进 TilingKey: 同一形态下四种 dtype 的 key 必须相同
TEST_F(ArgMaxGradTiling, test_tiling_key_independent_of_dtype)
{
    uint64_t k32 = 0;
    uint64_t k16 = 0;
    uint64_t ki32 = 0;
    uint64_t ki8 = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoArgMaxGradTilingCase({{4, 16, 8}, 1, ge::DT_FLOAT}, k32, blockDim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(DoArgMaxGradTilingCase({{4, 16, 8}, 1, ge::DT_FLOAT16}, k16, blockDim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(DoArgMaxGradTilingCase({{4, 16, 8}, 1, ge::DT_INT32}, ki32, blockDim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(DoArgMaxGradTilingCase({{4, 16, 8}, 1, ge::DT_INT8}, ki8, blockDim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(k32, k16);
    EXPECT_EQ(k32, ki32);
    EXPECT_EQ(k32, ki8);
}

TEST_F(ArgMaxGradTiling, test_tiling_negative_dimension)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    // dimension=-2 归一化后为 1, 与显式传 1 等价
    uint64_t keyPos = 0;
    EXPECT_EQ(DoArgMaxGradTilingCase({{4, 16, 8}, -2, ge::DT_FLOAT}, tilingKey, blockDim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(DoArgMaxGradTilingCase({{4, 16, 8}, 1, ge::DT_FLOAT}, keyPos, blockDim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tilingKey, keyPos);
}

TEST_F(ArgMaxGradTiling, test_tiling_rank1)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoArgMaxGradTilingCase({{64}, 0, ge::DT_FLOAT}, tilingKey, blockDim), ge::GRAPH_SUCCESS);
}

TEST_F(ArgMaxGradTiling, test_tiling_big_axis_multi_chunk)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoArgMaxGradTilingCase({{1, 4, 40000}, 1, ge::DT_FLOAT}, tilingKey, blockDim), ge::GRAPH_SUCCESS);
}

TEST_F(ArgMaxGradTiling, test_tiling_empty_tensor)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoArgMaxGradTilingCase({{4, 0, 8}, 1, ge::DT_FLOAT}, tilingKey, blockDim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(blockDim, 1U); // 空 tensor 不进核, 但 blockDim 必须合法
}

TEST_F(ArgMaxGradTiling, test_tiling_invalid_var_dtype)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoArgMaxGradTilingCase({{4, 16, 8}, 1, ge::DT_UINT8}, tilingKey, blockDim), ge::GRAPH_FAILED);
}

TEST_F(ArgMaxGradTiling, test_tiling_updates_dtype_mismatch)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoArgMaxGradTilingCase({{4, 16, 8}, 1, ge::DT_FLOAT, ge::DT_FLOAT16}, tilingKey, blockDim),
              ge::GRAPH_FAILED);
}

TEST_F(ArgMaxGradTiling, test_tiling_indices_dtype_invalid)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(
        DoArgMaxGradTilingCase({{4, 16, 8}, 1, ge::DT_FLOAT, ge::DT_UNDEFINED, ge::DT_INT64}, tilingKey, blockDim),
        ge::GRAPH_FAILED);
}

TEST_F(ArgMaxGradTiling, test_tiling_updates_shape_mismatch)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    ArgMaxGradCase c{{4, 16, 8}, 1, ge::DT_FLOAT};
    c.updatesShapeOv = {4, 4};
    EXPECT_EQ(DoArgMaxGradTilingCase(c, tilingKey, blockDim), ge::GRAPH_FAILED);
}

TEST_F(ArgMaxGradTiling, test_tiling_dimension_out_of_range)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoArgMaxGradTilingCase({{4, 16, 8}, 3, ge::DT_FLOAT}, tilingKey, blockDim), ge::GRAPH_FAILED);
    EXPECT_EQ(DoArgMaxGradTilingCase({{4, 16, 8}, -4, ge::DT_FLOAT}, tilingKey, blockDim), ge::GRAPH_FAILED);
}

// ── 布局归一 (outer, D, inner) 的值断言 ──────────────────────────────────────────
// 这三条对应 A2 assist_int32_help 的三个分支(dimension 落在最后一维 / 倒数第二维 / 更靠前),
// 它们决定内核里 k 的生成方式。只断言返回码是看不出布局算错的, 必须比值。
TEST_F(ArgMaxGradTiling, test_tiling_layout_dim_is_last_axis)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    ArgMaxGradArch35TilingData td{};
    EXPECT_EQ(DoArgMaxGradTilingCase({{2, 3, 4}, 2, ge::DT_FLOAT}, tilingKey, blockDim, &td), ge::GRAPH_SUCCESS);
    EXPECT_EQ(td.outer, 6);
    EXPECT_EQ(td.dimSize, 4);
    EXPECT_EQ(td.inner, 1);
    EXPECT_EQ(td.totalElems, 24);
}

TEST_F(ArgMaxGradTiling, test_tiling_layout_dim_is_second_last_axis)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    ArgMaxGradArch35TilingData td{};
    EXPECT_EQ(DoArgMaxGradTilingCase({{2, 3, 4}, 1, ge::DT_FLOAT}, tilingKey, blockDim, &td), ge::GRAPH_SUCCESS);
    EXPECT_EQ(td.outer, 2);
    EXPECT_EQ(td.dimSize, 3);
    EXPECT_EQ(td.inner, 4);
    EXPECT_EQ(td.totalElems, 24);
}

TEST_F(ArgMaxGradTiling, test_tiling_layout_dim_is_earlier_axis)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    ArgMaxGradArch35TilingData td{};
    EXPECT_EQ(DoArgMaxGradTilingCase({{2, 3, 4, 5}, 0, ge::DT_FLOAT}, tilingKey, blockDim, &td), ge::GRAPH_SUCCESS);
    EXPECT_EQ(td.outer, 1);
    EXPECT_EQ(td.dimSize, 2);
    EXPECT_EQ(td.inner, 60);
    EXPECT_EQ(td.totalElems, 120);
}

// 负数 dimension 必须与等价的正数下标得到完全相同的布局(而不只是"都成功")
TEST_F(ArgMaxGradTiling, test_tiling_negative_dimension_layout_equivalent)
{
    uint64_t keyNeg = 0;
    uint64_t keyPos = 0;
    uint32_t blockDim = 0;
    ArgMaxGradArch35TilingData tdNeg{};
    ArgMaxGradArch35TilingData tdPos{};
    EXPECT_EQ(DoArgMaxGradTilingCase({{2, 3, 4}, -2, ge::DT_FLOAT}, keyNeg, blockDim, &tdNeg), ge::GRAPH_SUCCESS);
    EXPECT_EQ(DoArgMaxGradTilingCase({{2, 3, 4}, 1, ge::DT_FLOAT}, keyPos, blockDim, &tdPos), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tdNeg.outer, tdPos.outer);
    EXPECT_EQ(tdNeg.dimSize, tdPos.dimSize);
    EXPECT_EQ(tdNeg.inner, tdPos.inner);
    EXPECT_EQ(keyNeg, keyPos);
}

// rank 1~8 全区间: 每一档都要能出 tiling, 且 totalElems 等于形状连乘
TEST_F(ArgMaxGradTiling, test_tiling_rank_full_range)
{
    for (int64_t rank = 1; rank <= 8; ++rank) {
        std::vector<int64_t> shape(static_cast<size_t>(rank), 2);
        int64_t expect = 1;
        for (auto d : shape) {
            expect *= d;
        }
        uint64_t tilingKey = 0;
        uint32_t blockDim = 0;
        ArgMaxGradArch35TilingData td{};
        ArgMaxGradCase c{shape, rank - 1, ge::DT_FLOAT};
        EXPECT_EQ(DoArgMaxGradTilingCase(c, tilingKey, blockDim, &td), ge::GRAPH_SUCCESS) << "rank=" << rank;
        EXPECT_EQ(td.totalElems, expect) << "rank=" << rank;
        EXPECT_EQ(td.dimSize, shape[static_cast<size_t>(rank - 1)]) << "rank=" << rank;
        EXPECT_EQ(td.inner, 1) << "rank=" << rank;
    }
}

// 多核切分: elemsPerCore 必须 32B 对齐(跨核不共享搬运块), 且能覆盖全部元素
// inner==1 且 D 很小、outer 很大: 每段能装 colsPerChunk/D 个 outer, indices/updates 各要
// 一个与之等长的标量缓冲, 但 bytesPerPoint 对 inner==1 把这两块记成 0 —— 段长按不含它们的
// 口径算满 UB 后再分配, 总量必然超。这类形状在支持面之内, 不该被 tiling 拒收。
TEST_F(ArgMaxGradTiling, test_tiling_inner_one_small_axis_large_outer)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    ArgMaxGradArch35TilingData td{};
    for (int64_t d : {2, 4, 8}) {
        EXPECT_EQ(DoArgMaxGradTilingCase({{1000000, d, 1}, 1, ge::DT_FLOAT}, tilingKey, blockDim, &td),
                  ge::GRAPH_SUCCESS)
            << "d=" << d;
    }
}

// 平台信息异常时必须干净拒收(而不是拿非法值继续算)。这三条对应 tiling 里三处拒收分支:
//   核数为 0 / UB 不大于 SIMD-SIMT dcache 预留 / UB 扣掉预留后装不下一个向量整宽的元素
// rank 为 0(标量 var)必须拒收: 被选轴不存在, 语义不成立
TEST_F(ArgMaxGradTiling, test_tiling_rank0_rejected)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoArgMaxGradTilingCase({{}, 0, ge::DT_FLOAT}, tilingKey, blockDim), ge::GRAPH_FAILED);
}

TEST_F(ArgMaxGradTiling, test_tiling_platform_zero_core_rejected)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoArgMaxGradTilingCase({{4, 16, 8}, 1, ge::DT_FLOAT}, tilingKey, blockDim, nullptr, 253952, 0),
              ge::GRAPH_FAILED);
}

TEST_F(ArgMaxGradTiling, test_tiling_platform_ub_not_above_dcache_rejected)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    // 32 KB 恰好等于 SIMD/SIMT 共用 dcache 的预留量, 扣完一个字节都不剩
    EXPECT_EQ(DoArgMaxGradTilingCase({{4, 16, 8}, 1, ge::DT_FLOAT}, tilingKey, blockDim, nullptr, 32 * 1024, 64),
              ge::GRAPH_FAILED);
}

TEST_F(ArgMaxGradTiling, test_tiling_platform_ub_too_small_for_one_vector_rejected)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    // 比 dcache 多 512 B: 过得了 Init 的门槛, 但一个向量整宽(64 个 fp32 元素)的各路 buffer 放不下
    EXPECT_EQ(DoArgMaxGradTilingCase({{4, 16, 8}, 1, ge::DT_FLOAT}, tilingKey, blockDim, nullptr, 32 * 1024 + 512, 64),
              ge::GRAPH_FAILED);
}

TEST_F(ArgMaxGradTiling, test_tiling_core_split_is_block_aligned)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    ArgMaxGradArch35TilingData td{};
    EXPECT_EQ(DoArgMaxGradTilingCase({{64, 128, 32}, 1, ge::DT_FLOAT}, tilingKey, blockDim, &td), ge::GRAPH_SUCCESS);
    EXPECT_GT(td.elemsPerCore, 0);
    EXPECT_EQ(td.elemsPerCore % (32 / static_cast<int64_t>(sizeof(float))), 0);
    EXPECT_GE(td.elemsPerCore * static_cast<int64_t>(blockDim), td.totalElems);
    EXPECT_GT(td.colsPerChunk, 0);
}
