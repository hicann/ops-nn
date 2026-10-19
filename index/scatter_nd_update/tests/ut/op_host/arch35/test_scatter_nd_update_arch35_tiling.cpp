/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_scatter_nd_update_arch35_tiling.cpp
 * \brief ScatterNdUpdate arch35 tiling ut，重点覆盖 960DT hifloat4 的 int8 单位换算
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>

#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"
#include "../../../../op_host/arch35/scatter_nd_update_tiling_regbase.h"

class ScatterNdUpdateArch35TilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ScatterNdUpdateArch35TilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ScatterNdUpdateArch35TilingTest TearDown" << std::endl; }
};

namespace {
struct ScatterTilingResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    ScatterNdUpdateRegBaseTilingData tilingData{};
};

// 构造 var/indices/updates 均连续的 tiling 上下文并执行 tiling；
// npuArch 非空时通过 version res 模拟指定平台（"9201" 表示 960DT）
static ScatterTilingResult RunScatterNdUpdateTiling(const std::vector<int64_t>& varShape, ge::DataType varDtype,
                                                    const std::vector<int64_t>& indicesShape, ge::DataType indicesDtype,
                                                    const std::vector<int64_t>& updatesShape,
                                                    const std::string& npuArch = "")
{
    ScatterTilingResult result;
    std::string compileInfoString = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 40}
    })";
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> socVersionInfos = {{"Short_SoC_version", "Ascend960DT"}};
    if (!npuArch.empty()) {
        socVersionInfos["NpuArch"] = npuArch;
    }
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    optiling::ScatterNdUpdateCompileInfo compileInfo;

    std::string opType("ScatterNdUpdate");
    if (gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()) == nullptr) {
        std::cout << "ScatterNdUpdate op impl not registered" << std::endl;
        return result;
    }
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    auto parseHolder = gert::KernelRunContextFaker()
                           .KernelIONum(3, 1)
                           .Inputs({const_cast<char*>("{}"), reinterpret_cast<void*>(&platformInfo)})
                           .Outputs({&compileInfo})
                           .Build();
    auto parseContext = parseHolder.GetContext<gert::TilingParseContext>();
    parseContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    parseContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    parseContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    parseContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    parseContext->GetPlatformInfo()->SetPlatformRes("version", socVersionInfos);
    if (tilingParseFunc(parseHolder.GetContext<gert::KernelContext>()) != ge::GRAPH_SUCCESS) {
        std::cout << "ScatterNdUpdate tiling parse failed" << std::endl;
        return result;
    }

    auto MakeShape = [](const std::vector<int64_t>& dims) {
        gert::StorageShape shape;
        for (auto d : dims) {
            shape.MutableOriginShape().AppendDim(d);
            shape.MutableStorageShape().AppendDim(d);
        }
        return shape;
    };
    gert::StorageShape varStorage = MakeShape(varShape);
    gert::StorageShape indicesStorage = MakeShape(indicesShape);
    gert::StorageShape updatesStorage = MakeShape(updatesShape);

    gert::Tensor varTensor(varStorage, {ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()}, gert::kFollowing,
                           varDtype, nullptr);
    gert::Tensor indicesTensor(indicesStorage, {ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()}, gert::kFollowing,
                               indicesDtype, nullptr);
    gert::Tensor updatesTensor(updatesStorage, {ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()}, gert::kFollowing,
                               varDtype, nullptr);

    auto tilingDataHolder = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());

    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1}, {1})
                      .InputShapes({&varStorage, &indicesStorage, &updatesStorage})
                      .InputTensors({&varTensor, &indicesTensor, &updatesTensor})
                      .OutputShapes({&varStorage})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, indicesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"use_locking", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(tilingDataHolder.get())
                      .Workspace(wsSize)
                      .Build();
    auto tilingContext = holder.GetContext<gert::TilingContext>();
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", socVersionInfos);

    result.status = tilingFunc(tilingContext);
    if (result.status != ge::GRAPH_SUCCESS) {
        return result;
    }
    auto* tilingData = reinterpret_cast<const ScatterNdUpdateRegBaseTilingData*>(
        tilingContext->GetRawTilingData()->GetData());
    result.tilingData = *tilingData;
    return result;
}
} // namespace

// 960DT hif4 单索引轴场景：afterAxis/outputStorageShapeSize/strideList 均换算为 int8 单位
TEST_F(ScatterNdUpdateArch35TilingTest, Hif4ContiguousRank1UnitsConvertedToInt8)
{
    auto result = RunScatterNdUpdateTiling({4, 8}, ge::DT_HIFLOAT4, {3, 1}, ge::DT_INT32, {3, 8}, "9201");
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    // var[4,8] hif4：32 个 hif4 元素 = 16 个 int8；afterAxis 8 hif4 -> 4 int8；stride[0] 8 hif4 -> 4 int8
    EXPECT_EQ(result.tilingData.afterAxis, 4);
    EXPECT_EQ(result.tilingData.sliceSize, 4UL);
    EXPECT_EQ(result.tilingData.outputStorageShapeSize, 16UL);
    EXPECT_EQ(result.tilingData.strideList[0], 4UL);
    EXPECT_EQ(result.tilingData.strideList[1], 0UL); // 尾轴 stride 1/2 取整为 0，kernel 不使用该维
    EXPECT_EQ(result.tilingData.varInAxis, 4);       // 行数不随 dtype 单位换算变化
    EXPECT_EQ(result.tilingData.indicesAxis, 3);
    EXPECT_EQ(result.tilingData.IsContiguous, 1);
}

// 960DT hif4 多索引轴场景：rank=2 时索引轴 stride 同步折半，flat 偏移与 yGm_[int8] 单位一致
TEST_F(ScatterNdUpdateArch35TilingTest, Hif4ContiguousRank2UnitsConvertedToInt8)
{
    auto result = RunScatterNdUpdateTiling({4, 6, 8}, ge::DT_HIFLOAT4, {2, 3, 2}, ge::DT_INT64, {2, 3, 8}, "9201");
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    // stride 原始 hif4 单位为 [48, 8, 1]，折半后 [24, 4, 0]；var 共 192 hif4 -> 96 int8
    EXPECT_EQ(result.tilingData.afterAxis, 4);
    EXPECT_EQ(result.tilingData.outputStorageShapeSize, 96UL);
    EXPECT_EQ(result.tilingData.strideList[0], 24UL);
    EXPECT_EQ(result.tilingData.strideList[1], 4UL);
    EXPECT_EQ(result.tilingData.strideList[2], 0UL);
    EXPECT_EQ(result.tilingData.varInAxis, 24); // 4 * 6 行
    EXPECT_EQ(result.tilingData.indicesAxis, 6);
}

// 对照场景：同 shape 的 fp16 输入不做任何折半，验证非 hif4 路径不受影响
TEST_F(ScatterNdUpdateArch35TilingTest, Float16ContiguousUnitsUnchanged)
{
    auto result = RunScatterNdUpdateTiling({4, 8}, ge::DT_FLOAT16, {3, 1}, ge::DT_INT32, {3, 8}, "9201");
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingData.afterAxis, 8);
    EXPECT_EQ(result.tilingData.sliceSize, 8UL);
    EXPECT_EQ(result.tilingData.outputStorageShapeSize, 32UL);
    EXPECT_EQ(result.tilingData.strideList[0], 8UL);
    EXPECT_EQ(result.tilingData.strideList[1], 1UL);
    EXPECT_EQ(result.tilingData.varInAxis, 4);
}

// 960DT hif4 每行元素数为奇数时直接报错，不进行换算
TEST_F(ScatterNdUpdateArch35TilingTest, Hif4OddAfterAxisRejected)
{
    auto result = RunScatterNdUpdateTiling({4, 7}, ge::DT_HIFLOAT4, {3, 1}, ge::DT_INT32, {3, 7}, "9201");
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}
