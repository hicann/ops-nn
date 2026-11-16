/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <vector>

#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tensor.h"
#include "test_cube_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"
#include "../../../../op_kernel/arch35/euclidean_norm_tiling_data.h"

using namespace ut_util;
using namespace std;
using namespace ge;

namespace {

constexpr char kCompileInfo[] = R"({
    "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                    "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                    "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                    "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                    "CORE_NUM": 24}
})";

// 构造 1-D int32 const tensor，shape=(N,)，内容 = axes
std::unique_ptr<uint8_t[]> MakeAxesConstTensor(const std::vector<int32_t>& axes)
{
    size_t totalSize = 0;
    auto holder = gert::Tensor::CreateFollowing(static_cast<int64_t>(axes.size()), ge::DT_INT32, totalSize);
    auto* t = reinterpret_cast<gert::Tensor*>(holder.get());
    t->MutableStorageShape().SetDimNum(1);
    t->MutableStorageShape().SetDim(0, static_cast<int64_t>(axes.size()));
    t->MutableOriginShape().SetDimNum(1);
    t->MutableOriginShape().SetDim(0, static_cast<int64_t>(axes.size()));
    auto* dataPtr = reinterpret_cast<int32_t*>(holder.get() + sizeof(gert::Tensor));
    for (size_t i = 0; i < axes.size(); ++i) {
        dataPtr[i] = axes[i];
    }
    return holder;
}

struct EuclideanNormCompileInfoForTest {};

// 共用 setup：构造 platform_info / kernel context faker
void SetupPlatform(fe::PlatFormInfos& platformInfo, map<string, string>& socInfos, map<string, string>& aicoreSpec,
                   map<string, string>& intrinsics)
{
    GetPlatFormInfos(kCompileInfo, socInfos, aicoreSpec, intrinsics);
    platformInfo.Init();
}

void SetupKernelHolder(fe::PlatFormInfos& platformInfo, EuclideanNormCompileInfoForTest& compileInfo,
                       const map<string, string>& socInfos, const map<string, string>& aicoreSpec,
                       const map<string, string>& intrinsics, gert::KernelRunContextHolder& kernelHolder)
{
    std::string compileInfoStr(kCompileInfo);
    kernelHolder = gert::KernelRunContextFaker()
                       .KernelIONum(2, 1)
                       .Inputs({const_cast<char*>(compileInfoStr.c_str()), reinterpret_cast<void*>(&platformInfo)})
                       .Outputs({&compileInfo})
                       .Build();
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "SoCInfo", const_cast<map<string, string>&>(socInfos));
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreSpec", const_cast<map<string, string>&>(aicoreSpec));
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", const_cast<map<string, string>&>(intrinsics));
}

// FR-01/03/04 看护用例共用：轻量 tiling 调用（shape 元数据，零大内存分配），返回状态码 + TilingData 摘要
struct UtTilingResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    EuclideanNormTilingData td;
};
UtTilingResult RunTilingSimple(const std::vector<int64_t>& xDims, const std::vector<int32_t>& axes, ge::DataType dtype,
                               bool keepDims, const std::vector<int64_t>& yDims)
{
    UtTilingResult result;
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    EuclideanNormCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{}, {}};
    xShape.MutableStorageShape().SetDimNum(xDims.size());
    xShape.MutableOriginShape().SetDimNum(xDims.size());
    for (size_t i = 0; i < xDims.size(); ++i) {
        xShape.MutableStorageShape().SetDim(i, xDims[i]);
        xShape.MutableOriginShape().SetDim(i, xDims[i]);
    }
    gert::StorageShape yShape = {{}, {}};
    yShape.MutableStorageShape().SetDimNum(yDims.size());
    yShape.MutableOriginShape().SetDimNum(yDims.size());
    for (size_t i = 0; i < yDims.size(); ++i) {
        yShape.MutableStorageShape().SetDim(i, yDims[i]);
        yShape.MutableOriginShape().SetDim(i, yDims[i]);
    }

    auto axesHolder = MakeAxesConstTensor(axes);
    gert::StorageShape axesShape = {{static_cast<int64_t>(axes.size())}, {static_cast<int64_t>(axes.size())}};
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> consts;
    consts.emplace_back(1U, std::move(axesHolder));

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EuclideanNorm")->tiling;
    if (tilingFunc == nullptr) {
        return result;
    }
    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    auto holder = gert::TilingContextFaker()
                      .SetOpType("EuclideanNorm")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &axesShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"keep_dims", Ops::NN::AnyValue::CreateFrom<bool>(keepDims)}})
                      .ConstInput(consts)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    auto* tilingContext = holder.GetContext<gert::TilingContext>();
    result.status = tilingFunc(tilingContext);
    auto* rawTilingData = tilingContext->GetRawTilingData();
    if (rawTilingData != nullptr) {
        result.td = *reinterpret_cast<const EuclideanNormTilingData*>(rawTilingData->GetData());
    }
    return result;
}

// kernel cacheCount 公式镜像（euclidean_norm.h :493-495 / :412-420 FindNearestPower2+CalLog2）
static int64_t UtKernelCacheCount(int64_t rLoopCntTotal)
{
    if (rLoopCntTotal <= 2) {
        return 1;
    }
    int64_t num = rLoopCntTotal - 1;
    int64_t pow = 0;
    while (num > 1) {
        num >>= 1;
        ++pow;
    }
    int64_t v = static_cast<int64_t>(1) << pow;
    int64_t lg = 0;
    while (v > 1) {
        v >>= 1;
        ++lg;
    }
    return lg + 1;
}

constexpr int64_t kUtUbSize = 196608; // 与 kCompileInfo UB_SIZE 一致

} // namespace

class EuclideanNormTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "EuclideanNormTiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "EuclideanNormTiling TearDown" << std::endl; }
};

// ────────────────────────────────────────────────────────────────────────────
// 用例 1：fp32 输入，shape=(4, 64)，axes=[1]，keep_dims=false → tiling success
// ────────────────────────────────────────────────────────────────────────────
TEST_F(EuclideanNormTiling, tiling_fp32_2d_reduce_last)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    EuclideanNormCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{4, 64}, {4, 64}};
    gert::StorageShape yShape = {{4}, {4}};

    auto axesHolder = MakeAxesConstTensor({1});
    gert::StorageShape axesShape = {{1}, {1}};
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> consts;
    consts.emplace_back(1U, std::move(axesHolder));

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EuclideanNorm")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());
    ASSERT_NE(param, nullptr);

    bool keepDims = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("EuclideanNorm")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &axesShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"keep_dims", Ops::NN::AnyValue::CreateFrom<bool>(keepDims)}})
                      .ConstInput(consts)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    auto* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext, nullptr);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
}

// ────────────────────────────────────────────────────────────────────────────
// 用例 2：fp16 输入，shape=(8, 1024)，axes=[1]，keep_dims=false
// ────────────────────────────────────────────────────────────────────────────
TEST_F(EuclideanNormTiling, tiling_fp16_2d_reduce_last)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    EuclideanNormCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{8, 1024}, {8, 1024}};
    gert::StorageShape yShape = {{8}, {8}};

    auto axesHolder = MakeAxesConstTensor({1});
    gert::StorageShape axesShape = {{1}, {1}};
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> consts;
    consts.emplace_back(1U, std::move(axesHolder));

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EuclideanNorm")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool keepDims = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("EuclideanNorm")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &axesShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"keep_dims", Ops::NN::AnyValue::CreateFrom<bool>(keepDims)}})
                      .ConstInput(consts)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    auto* tilingContext = holder.GetContext<gert::TilingContext>();
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);
}

// ────────────────────────────────────────────────────────────────────────────
// 用例 3：int32 输入，shape=(16, 32)，axes=[0] → tail R 路径 (axisNum=2)
// ────────────────────────────────────────────────────────────────────────────
TEST_F(EuclideanNormTiling, tiling_int32_2d_reduce_first)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    EuclideanNormCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    // reduce axis = 0：经合轴 / 补 leading A 后 pattern 仍是 AR（前置 A=1）
    gert::StorageShape xShape = {{16, 32}, {16, 32}};
    gert::StorageShape yShape = {{32}, {32}};

    auto axesHolder = MakeAxesConstTensor({0});
    gert::StorageShape axesShape = {{1}, {1}};
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> consts;
    consts.emplace_back(1U, std::move(axesHolder));

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EuclideanNorm")->tiling;
    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool keepDims = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("EuclideanNorm")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &axesShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"keep_dims", Ops::NN::AnyValue::CreateFrom<bool>(keepDims)}})
                      .ConstInput(consts)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    EXPECT_EQ(tilingFunc(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}

// ────────────────────────────────────────────────────────────────────────────
// 用例 5（v2 专属）：fp32 (4, 4, 8, 8) reduce axes=[1, 3] → ARAR pattern (tail R)
//   预期 climb：rSplit 从 LastR=3 climb 出去到 1，UB 内 R bundle 含 2 根 R 轴；
//   aSplit 可能也 climb。这是 v1 简化下做不到的多轴 bundle。
// ────────────────────────────────────────────────────────────────────────────
TEST_F(EuclideanNormTiling, tiling_v2_climb_arar_fp32)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    EuclideanNormCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    // shape (4, 4, 8, 8)，axes=[1, 3] → ARAR pattern：A=4, R=4, A=8, R=8 (4 axes，tail R)
    gert::StorageShape xShape = {{4, 4, 8, 8}, {4, 4, 8, 8}};
    gert::StorageShape yShape = {{4, 8}, {4, 8}};

    auto axesHolder = MakeAxesConstTensor({1, 3});
    gert::StorageShape axesShape = {{2}, {2}};
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> consts;
    consts.emplace_back(1U, std::move(axesHolder));

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EuclideanNorm")->tiling;
    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool keepDims = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("EuclideanNorm")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &axesShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"keep_dims", Ops::NN::AnyValue::CreateFrom<bool>(keepDims)}})
                      .ConstInput(consts)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    EXPECT_EQ(tilingFunc(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}

// ────────────────────────────────────────────────────────────────────────────
// 用例 6（v2 专属）：fp16 (2, 3, 4, 5, 6) reduce axes=[1, 3] → ARARA pattern (tail A)
//   预期 climb：aSplit 从 LastA=4 climb 出去；UB 内 A bundle 含多根 A 轴。
// ────────────────────────────────────────────────────────────────────────────
TEST_F(EuclideanNormTiling, tiling_v2_climb_arara_fp16)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    EuclideanNormCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}};
    gert::StorageShape yShape = {{2, 4, 6}, {2, 4, 6}};

    auto axesHolder = MakeAxesConstTensor({1, 3});
    gert::StorageShape axesShape = {{2}, {2}};
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> consts;
    consts.emplace_back(1U, std::move(axesHolder));

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EuclideanNorm")->tiling;
    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool keepDims = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("EuclideanNorm")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &axesShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"keep_dims", Ops::NN::AnyValue::CreateFrom<bool>(keepDims)}})
                      .ConstInput(consts)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    EXPECT_EQ(tilingFunc(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}

// ────────────────────────────────────────────────────────────────────────────
// 用例 4：bf16 输入，shape=(2,4,8)，axes=[2] → ARA pattern (tail A)
// ────────────────────────────────────────────────────────────────────────────
TEST_F(EuclideanNormTiling, tiling_bf16_3d_reduce_middle)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    EuclideanNormCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    // reduce axis = 1 → 合轴后变 A=2, R=4, A=8 = ARA tail A
    gert::StorageShape xShape = {{2, 4, 8}, {2, 4, 8}};
    gert::StorageShape yShape = {{2, 8}, {2, 8}};

    auto axesHolder = MakeAxesConstTensor({1});
    gert::StorageShape axesShape = {{1}, {1}};
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> consts;
    consts.emplace_back(1U, std::move(axesHolder));

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EuclideanNorm")->tiling;
    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool keepDims = false;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("EuclideanNorm")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &axesShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"keep_dims", Ops::NN::AnyValue::CreateFrom<bool>(keepDims)}})
                      .ConstInput(consts)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    EXPECT_EQ(tilingFunc(holder.GetContext<gert::TilingContext>()), ge::GRAPH_SUCCESS);
}

// ────────────────────────────────────────────────────────────────────────────
// 用例 7：axes=[]（空张量）→ 无归约轴（恒等 shape，y=x）
//   shape=(4, 64) fp32 axes=[] keep_dims=true → 全 A pattern → PadRIfPureA 前置 [A=1, R=1]
//   → ARA（tail-A），R=1 归约产出逐元素 sqrt(x²)=|x|，输出 shape 保持 (4, 64)
// ────────────────────────────────────────────────────────────────────────────
TEST_F(EuclideanNormTiling, tiling_fp32_2d_empty_axes_no_reduce)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    EuclideanNormCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{4, 64}, {4, 64}};
    gert::StorageShape yShape = {{4, 64}, {4, 64}}; // 无归约轴：y = x

    auto axesHolder = MakeAxesConstTensor({}); // 空 axes
    // axes=[]需用1-D且dim=0的shape表达，标量shape(0-D)的GetShapeSize为1
    gert::StorageShape axesShape = {{0}, {0}};
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> consts;
    consts.emplace_back(1U, std::move(axesHolder));

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EuclideanNorm")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    bool keepDims = true;
    auto holder = gert::TilingContextFaker()
                      .SetOpType("EuclideanNorm")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &axesShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"keep_dims", Ops::NN::AnyValue::CreateFrom<bool>(keepDims)}})
                      .ConstInput(consts)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    auto* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext, nullptr);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);

    auto* rawTilingData = tilingContext->GetRawTilingData();
    ASSERT_NE(rawTilingData, nullptr);
    const auto* td = reinterpret_cast<const EuclideanNormTilingData*>(rawTilingData->GetData());
    // 全 A 合轴 A(256) → PadRIfPureA 前置 [A=1, R=1] → ARA tail-A
    EXPECT_EQ(td->axisNum, 3);
    EXPECT_EQ(td->axisShape[0], 1);
    EXPECT_EQ(td->axisShape[1], 1);
    EXPECT_EQ(td->axisShape[2], 256);
    EXPECT_EQ(td->rUbFactor, 1); // R=1 逐元素归约
}

// ────────────────────────────────────────────────────────────────────────────
// 用例 8：axes=[] + 含 0 维的 x（issue #25 复刻，c_s_021）→ EMPTY_A 全核早退
//   x=(1,0,1,1,1,1) fp16 axes=[]：0 维落在 A 轴 → EMPTY_A，usedCoreNum=0、零写，
//   修复前误展开全归约 → EMPTY_R → 向 0 字节输出写 2 字节（越界写）
// ────────────────────────────────────────────────────────────────────────────
TEST_F(EuclideanNormTiling, tiling_fp16_zero_dim_empty_axes_empty_a)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    EuclideanNormCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{1, 0, 1, 1, 1, 1}, {1, 0, 1, 1, 1, 1}};
    gert::StorageShape yShape = {{1, 0, 1, 1, 1, 1}, {1, 0, 1, 1, 1, 1}}; // 恒等：0 元素输出

    auto axesHolder = MakeAxesConstTensor({});
    gert::StorageShape axesShape = {{0}, {0}};
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> consts;
    consts.emplace_back(1U, std::move(axesHolder));

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EuclideanNorm")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    auto holder = gert::TilingContextFaker()
                      .SetOpType("EuclideanNorm")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &axesShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .ConstInput(consts)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    auto* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext, nullptr);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);

    auto* rawTilingData = tilingContext->GetRawTilingData();
    ASSERT_NE(rawTilingData, nullptr);
    const auto* td = reinterpret_cast<const EuclideanNormTilingData*>(rawTilingData->GetData());
    EXPECT_EQ(td->usedCoreNum, 0); // EMPTY_A 全核早退，kernel 零写
    EXPECT_EQ(td->axisShape[0], 0);
}

// ────────────────────────────────────────────────────────────────────────────
// 用例 9：axes=[] + 全 1 shape → DropSizeOneAxes 全退化 → [A=1] + 补 R=1 → AR（tail-R）
//   x=(1, 1) fp16 axes=[]：输出保持 (1, 1)
// ────────────────────────────────────────────────────────────────────────────
TEST_F(EuclideanNormTiling, tiling_fp16_all_one_empty_axes_ar)
{
    fe::PlatFormInfos platformInfo;
    map<string, string> socInfos, aicoreSpec, intrinsics;
    SetupPlatform(platformInfo, socInfos, aicoreSpec, intrinsics);
    EuclideanNormCompileInfoForTest compileInfo;
    gert::KernelRunContextHolder kernelHolder;
    SetupKernelHolder(platformInfo, compileInfo, socInfos, aicoreSpec, intrinsics, kernelHolder);

    gert::StorageShape xShape = {{1, 1}, {1, 1}};
    gert::StorageShape yShape = {{1, 1}, {1, 1}};

    auto axesHolder = MakeAxesConstTensor({});
    gert::StorageShape axesShape = {{0}, {0}};
    std::vector<std::pair<size_t, std::unique_ptr<uint8_t[]>>> consts;
    consts.emplace_back(1U, std::move(axesHolder));

    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EuclideanNorm")->tiling;
    ASSERT_NE(tilingFunc, nullptr);

    auto param = gert::TilingData::CreateCap(4096);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto* wsSize = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());

    auto holder = gert::TilingContextFaker()
                      .SetOpType("EuclideanNorm")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &axesShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .ConstInput(consts)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();
    auto* tilingContext = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tilingContext, nullptr);
    EXPECT_EQ(tilingFunc(tilingContext), ge::GRAPH_SUCCESS);

    auto* rawTilingData = tilingContext->GetRawTilingData();
    ASSERT_NE(rawTilingData, nullptr);
    const auto* td = reinterpret_cast<const EuclideanNormTilingData*>(rawTilingData->GetData());
    EXPECT_EQ(td->axisNum, 2); // A=1, R=1
    EXPECT_EQ(td->axisShape[0], 1);
    EXPECT_EQ(td->axisShape[1], 1);
}

// ────────────────────────────────────────────────────────────────────────────
// 用例 10（FR-01 看护）：fp32 tail-R 对齐窗口 R=11257（T08 probe 修前超限实例）
//   修复前：R ∈ (16·⌊rIMax/16⌋, rIMax) 窗口内 rUbFactorAlign=CeilAlign 后越过预算，
//   2×pre+2×tmp+cache+post = 196640 > UB=196608（超 32B）；
//   修复后（预算对齐口径 + 四缓冲总量终检）：四缓冲总和 ≤ UB。
// ────────────────────────────────────────────────────────────────────────────
TEST_F(EuclideanNormTiling, tiling_fp32_tail_r_aligned_window_ub_budget)
{
    auto result = RunTilingSimple({8, 11257}, {1}, ge::DT_FLOAT, false, {8});
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    const int64_t sum4 = 2 * result.td.preReduceUbSize + 2 * result.td.tmpBufUbSize + result.td.cacheBufUbSize +
                         result.td.postReduceUbSize;
    EXPECT_LE(sum4, kUtUbSize); // 修复前实测 196640 > 196608
}

// ────────────────────────────────────────────────────────────────────────────
// 用例 11（FR-01 看护）：fp16 tail-R 对齐窗口 R=15009（T08 probe 修前超限实例，超 96B）
// ────────────────────────────────────────────────────────────────────────────
TEST_F(EuclideanNormTiling, tiling_fp16_tail_r_aligned_window_ub_budget)
{
    auto result = RunTilingSimple({8, 15009}, {1}, ge::DT_FLOAT16, false, {8});
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    const int64_t sum4 = 2 * result.td.preReduceUbSize + 2 * result.td.tmpBufUbSize + result.td.cacheBufUbSize +
                         result.td.postReduceUbSize;
    EXPECT_LE(sum4, kUtUbSize); // 修复前实测 196704 > 196608
}

// ────────────────────────────────────────────────────────────────────────────
// 用例 12（FR-04 看护）：0-D 标量 x 的 axes 必须为空（对齐 infershape 门）
//   0-D + axes=[0]/[-1] → tiling 拒绝（修复前放行，两门不一致）；
//   0-D + axes=[] → 正常（规格化 [1] 后 PadRIfPureA 兜底 AR）。
// ────────────────────────────────────────────────────────────────────────────
TEST_F(EuclideanNormTiling, tiling_0d_x_nonempty_axes_rejected)
{
    EXPECT_NE(RunTilingSimple({}, {0}, ge::DT_FLOAT, false, {}).status, ge::GRAPH_SUCCESS);
    EXPECT_NE(RunTilingSimple({}, {-1}, ge::DT_FLOAT, false, {}).status, ge::GRAPH_SUCCESS);
    auto ok = RunTilingSimple({}, {}, ge::DT_FLOAT, true, {});
    EXPECT_EQ(ok.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(ok.td.axisNum, 2); // 规格化 [1] → A=1，PadRIfPureA 补 R=1 → AR
}

// ────────────────────────────────────────────────────────────────────────────
// 用例 13（FR-03 看护）：大 R 元数据下 cacheBuf 一致性约束
//   3-D (2, 7e12, 56) axes=[1]（ARA tail-A，纯 shape 元数据、零大内存分配）：
//   修复前 laneN=112、rLoopCntTotal≈7e10 → rootEnd=(cc-1)×levelStride+laneN > 4096（树根越界写域）；
//   修复后（EnforceCacheBufConstraint 回退切分）rootEnd ≤ 4096。
// ────────────────────────────────────────────────────────────────────────────
TEST_F(EuclideanNormTiling, tiling_fp32_cachabuf_big_r_constraint)
{
    auto result = RunTilingSimple({2, 7000000000000LL, 56}, {1}, ge::DT_FLOAT, false, {2, 56});
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    const int64_t laneN = result.td.aUbFactorAlign * result.td.innerAProdAlign;
    const int64_t levelStride = (laneN + 7) / 8 * 8; // CeilAlign(laneN, kBlockF32=8)
    const int64_t cacheCount = UtKernelCacheCount(result.td.rLoopCntTotal);
    const int64_t rootEnd = (cacheCount - 1) * levelStride + laneN;
    EXPECT_LE(rootEnd, 4096); // cacheBuf 16KB / 4B
}

// ────────────────────────────────────────────────────────────────────────────
// 用例 14（FR-03 看护，T13）：aUbFactor=1 后的吐轴回退分支（EnforceCacheBufConstraint）
//   fp16 (2, R, 120) axes=[1]（ARA tail-A，纯 shape 元数据、零大内存分配）：
//   A 侧 climb 吸入 tail A=120（ipa=CeilAlign(120,16)=128 达 atmp=128）后 aUbFactor=1、
//   aSplit=0（外层 A）、laneN=128；rLoopCntTotal≈5.5e11 → cacheCount=39 →
//   rootEnd=39×128=4992 > 4096。回退轮：aUbFactor 已 =1 → 吐轴分支——aSplitAxisIdx
//   改 lastA、innerAProd[Align] 重置 1、aUbFactor=axisShape[lastA]/2=60，随后 tail-A
//   对齐同步 aUbFactorAlign=CeilAlign(60,16)=64；下一轮 rootEnd=39×64=2496 ≤ 4096 收敛。
// ────────────────────────────────────────────────────────────────────────────
TEST_F(EuclideanNormTiling, tiling_fp16_cachabuf_spit_axis_fallback)
{
    auto result = RunTilingSimple({2, 64321360424896LL, 120}, {1}, ge::DT_FLOAT16, false, {2, 120});
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    // 吐轴分支直接证据：aSplitAxisIdx 从外层 A（0）移到 lastA（2）——减半路径不改 aSplit；
    // aUbFactor=60=axisShape[lastA]/2（吐轴半根；初始值 1 经减半路径恒为 1，不可能产生 60）
    EXPECT_EQ(result.td.aSplitAxisIdx, 2);
    EXPECT_EQ(result.td.aUbFactor, 60);
    EXPECT_EQ(result.td.aUbFactorAlign, 64); // tail-A 对齐同步 CeilAlign(60, bsAElem=16)
    EXPECT_EQ(result.td.innerAProdAlign, 1); // 吐轴重置 inner bundle
    const int64_t laneN = result.td.aUbFactorAlign * result.td.innerAProdAlign;
    const int64_t levelStride = (laneN + 7) / 8 * 8; // CeilAlign(laneN, kBlockF32=8)
    const int64_t cacheCount = UtKernelCacheCount(result.td.rLoopCntTotal);
    const int64_t rootEnd = (cacheCount - 1) * levelStride + laneN;
    EXPECT_LE(rootEnd, 4096); // 吐轴回退后树根上界收敛
    const int64_t sum4 = 2 * result.td.preReduceUbSize + 2 * result.td.tmpBufUbSize + result.td.cacheBufUbSize +
                         result.td.postReduceUbSize;
    EXPECT_LE(sum4, kUtUbSize); // 回退重算后四缓冲总量仍守预算
}
