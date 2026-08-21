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
 * \file test_chamfer_distance_tiling.cpp
 * \brief ChamferDistance arch35 tiling UT(需 --soc=ascend950, tiling 受 COMPUTE_UNIT 门控)
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

class ChamferDistanceTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ChamferDistanceTiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "ChamferDistanceTiling TearDown" << std::endl; }
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
static ge::graphStatus DoChamferDistanceTilingCase(std::initializer_list<int64_t> shape, ge::DataType xyz1Dtype,
                                                   ge::DataType xyz2Dtype, uint64_t& tilingKey, uint32_t& blockDim,
                                                   std::initializer_list<int64_t> xyz2ShapeOv = {})
{
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics);

    struct ChamferDistanceCompileInfo {};
    ChamferDistanceCompileInfo compileInfo;

    std::string opType("ChamferDistance");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str());
    EXPECT_NE(opImpl, nullptr);
    if (opImpl == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto tilingFunc = opImpl->tiling;
    auto tilingParseFunc = opImpl->tiling_parse;

    string compileInfoStr = R"({"device_id": null})";
    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(compileInfoStr.c_str()), reinterpret_cast<void*>(&platFormInfo)})
                            .Outputs({&compileInfo})
                            .Build();
    EXPECT_TRUE(kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                           intrinsics);
    EXPECT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    gert::StorageShape xyz1Shape = {shape, shape};
    gert::StorageShape xyz2Shape = xyz2ShapeOv.size() > 0 ? gert::StorageShape{xyz2ShapeOv, xyz2ShapeOv} :
                                                            gert::StorageShape{shape, shape};
    // 输出 (B, N) = 取 xyz 的第 1、2 维
    std::vector<int64_t> dims(shape);
    std::vector<int64_t> outDims;
    if (dims.size() == 3) {
        outDims = {dims[1], dims[2]};
    } else {
        outDims = dims;
    }
    gert::StorageShape distShape;
    for (int64_t d : outDims) {
        distShape.MutableStorageShape().AppendDim(d);
        distShape.MutableOriginShape().AppendDim(d);
    }
    gert::StorageShape idxShape = distShape;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspaceSizeHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHolder.get());
    EXPECT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType("ChamferDistance")
                      .NodeIoNum(2, 4)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xyz1Shape, &xyz2Shape})
                      .OutputShapes({&distShape, &distShape, &idxShape, &idxShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, xyz1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, xyz2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, xyz1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, xyz1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tilingContext = holder.GetContext<gert::TilingContext>();
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

TEST_F(ChamferDistanceTiling, test_tiling_fp32_basic)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoChamferDistanceTilingCase({2, 4, 128}, ge::DT_FLOAT, ge::DT_FLOAT, tilingKey, blockDim),
              ge::GRAPH_SUCCESS);
    // 512 个查询点分 64 核 → 每核 8 个
    EXPECT_EQ(blockDim, 64U);
}

TEST_F(ChamferDistanceTiling, test_tiling_fp16_basic)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoChamferDistanceTilingCase({2, 4, 128}, ge::DT_FLOAT16, ge::DT_FLOAT16, tilingKey, blockDim),
              ge::GRAPH_SUCCESS);
}

TEST_F(ChamferDistanceTiling, test_tiling_bf16_basic)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoChamferDistanceTilingCase({2, 4, 128}, ge::DT_BF16, ge::DT_BF16, tilingKey, blockDim),
              ge::GRAPH_SUCCESS);
}

// dtype 不进 TilingKey(由 def 的 dtype profile 驱动内核实例化), 三个 dtype 都应落到同一个 key 0
TEST_F(ChamferDistanceTiling, test_tiling_key_is_zero_for_all_dtypes)
{
    uint64_t keyFp32 = 1;
    uint64_t keyFp16 = 1;
    uint64_t keyBf16 = 1;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoChamferDistanceTilingCase({2, 4, 128}, ge::DT_FLOAT, ge::DT_FLOAT, keyFp32, blockDim),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(DoChamferDistanceTilingCase({2, 4, 128}, ge::DT_FLOAT16, ge::DT_FLOAT16, keyFp16, blockDim),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(DoChamferDistanceTilingCase({2, 4, 128}, ge::DT_BF16, ge::DT_BF16, keyBf16, blockDim), ge::GRAPH_SUCCESS);
    EXPECT_EQ(keyFp32, 0U);
    EXPECT_EQ(keyFp16, 0U);
    EXPECT_EQ(keyBf16, 0U);
}

TEST_F(ChamferDistanceTiling, test_tiling_single_task_less_than_core)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoChamferDistanceTilingCase({2, 1, 8}, ge::DT_FLOAT, ge::DT_FLOAT, tilingKey, blockDim),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(blockDim, 8U);
}

TEST_F(ChamferDistanceTiling, test_tiling_big_n_multi_chunk)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoChamferDistanceTilingCase({2, 1, 40000}, ge::DT_FLOAT, ge::DT_FLOAT, tilingKey, blockDim),
              ge::GRAPH_SUCCESS);
}

TEST_F(ChamferDistanceTiling, test_tiling_empty_n)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoChamferDistanceTilingCase({2, 4, 0}, ge::DT_FLOAT, ge::DT_FLOAT, tilingKey, blockDim),
              ge::GRAPH_SUCCESS);
    EXPECT_EQ(blockDim, 1U);
}

// ---- 负向用例: 非法输入必须被 tiling 拒收 ----

TEST_F(ChamferDistanceTiling, test_tiling_invalid_dtype_int32)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoChamferDistanceTilingCase({2, 4, 128}, ge::DT_INT32, ge::DT_INT32, tilingKey, blockDim),
              ge::GRAPH_FAILED);
}

TEST_F(ChamferDistanceTiling, test_tiling_dtype_mismatch)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoChamferDistanceTilingCase({2, 4, 128}, ge::DT_FLOAT, ge::DT_FLOAT16, tilingKey, blockDim),
              ge::GRAPH_FAILED);
}

TEST_F(ChamferDistanceTiling, test_tiling_invalid_rank)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoChamferDistanceTilingCase({4, 128}, ge::DT_FLOAT, ge::DT_FLOAT, tilingKey, blockDim), ge::GRAPH_FAILED);
}

TEST_F(ChamferDistanceTiling, test_tiling_first_dim_not_two)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoChamferDistanceTilingCase({3, 4, 128}, ge::DT_FLOAT, ge::DT_FLOAT, tilingKey, blockDim),
              ge::GRAPH_FAILED);
}

TEST_F(ChamferDistanceTiling, test_tiling_shape_mismatch)
{
    uint64_t tilingKey = 0;
    uint32_t blockDim = 0;
    EXPECT_EQ(DoChamferDistanceTilingCase({2, 4, 128}, ge::DT_FLOAT, ge::DT_FLOAT, tilingKey, blockDim, {2, 4, 64}),
              ge::GRAPH_FAILED);
}
