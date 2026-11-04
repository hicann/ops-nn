/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <iostream>
#include <vector>
#include <fstream>
#include <gtest/gtest.h>

#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "graph/types.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "../../../../op_host/arch35/relu6_grad_tiling_arch35.h"

using namespace ut_util;
using namespace std;
using namespace ge;

namespace {
const std::string OP_TYPE = "Relu6Grad";
constexpr std::array<ge::Format, 3> ND_FORMATS = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
const std::string COMPILE_INFO_JSON = R"({
      "hardware_info": {
        "BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64
      }
    })";
} // namespace

class Relu6GradTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Relu6GradTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "Relu6GradTilingTest TearDown" << std::endl; }
};

// Run a single tiling case. Both inputs share the same dtype; output dtype
// equals input dtype. Returns the tiling function status so callers can
// assert either SUCCESS or FAILED depending on what they're testing.
static ge::graphStatus RunTiling(ge::DataType gradDtype, ge::DataType featDtype, ge::DataType outDtype,
                                 gert::StorageShape& gShape, gert::StorageShape& fShape, gert::StorageShape& outShape,
                                 const std::array<ge::Format, 3>& storageFormats = ND_FORMATS,
                                 const std::array<ge::Format, 3>& originFormats = ND_FORMATS,
                                 uint64_t* outTilingKey = nullptr)
{
    std::map<std::string, std::string> soc_infos;
    std::map<std::string, std::string> aicore_spec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    GetPlatFormInfos(COMPILE_INFO_JSON.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    Ops::Base::BroadcastCompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 262144;

    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(COMPILE_INFO_JSON.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    EXPECT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    EXPECT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    EXPECT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(OP_TYPE)
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&gShape, &fShape})
                      .OutputShapes({&outShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, gradDtype, originFormats[0], storageFormats[0])
                      .NodeInputTd(1, featDtype, originFormats[1], storageFormats[1])
                      .NodeOutputTd(0, outDtype, originFormats[2], storageFormats[2])
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    auto context = holder.GetContext<gert::TilingContext>();
    auto ret = tiling_func(context);
    if (ret == ge::GRAPH_SUCCESS && outTilingKey != nullptr) {
        *outTilingKey = context->GetTilingKey();
    }
    return ret;
}

TEST_F(Relu6GradTilingTest, same_shape_fp32)
{
    gert::StorageShape s = {{16, 16}, {16, 16}};
    EXPECT_EQ(RunTiling(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, s, s, s), ge::GRAPH_SUCCESS);
}

TEST_F(Relu6GradTilingTest, same_shape_fp16)
{
    gert::StorageShape s = {{16, 16}, {16, 16}};
    EXPECT_EQ(RunTiling(ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, s, s, s), ge::GRAPH_SUCCESS);
}

TEST_F(Relu6GradTilingTest, same_shape_bf16)
{
    gert::StorageShape s = {{16, 16}, {16, 16}};
    EXPECT_EQ(RunTiling(ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, s, s, s), ge::GRAPH_SUCCESS);
}

TEST_F(Relu6GradTilingTest, rejects_private_storage_format_on_each_port)
{
    gert::StorageShape s = {{1, 2, 2, 2}, {1, 2, 2, 2}};
    const std::array<const char*, 3> portNames = {"gradients", "features", "backprops"};
    const auto nzWithSubformat = static_cast<ge::Format>(ge::GetFormatFromSub(ge::FORMAT_FRACTAL_NZ, 1));
    for (auto format : {ge::FORMAT_FRACTAL_NZ, ge::FORMAT_NC1HWC0, ge::FORMAT_FRACTAL_Z, nzWithSubformat}) {
        for (size_t port = 0; port < portNames.size(); ++port) {
            SCOPED_TRACE(testing::Message() << "port=" << portNames[port] << ", storage_format=" << format);
            auto storageFormats = ND_FORMATS;
            storageFormats[port] = format;
            EXPECT_EQ(RunTiling(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, s, s, s, storageFormats), ge::GRAPH_FAILED);
        }
    }
}

TEST_F(Relu6GradTilingTest, accepts_public_storage_format_on_each_port)
{
    gert::StorageShape s = {{1, 2, 2, 2}, {1, 2, 2, 2}};
    const std::array<const char*, 3> portNames = {"gradients", "features", "backprops"};
    uint64_t ndTilingKey = 0;
    ASSERT_EQ(RunTiling(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, s, s, s, ND_FORMATS, ND_FORMATS, &ndTilingKey),
              ge::GRAPH_SUCCESS);
    for (auto format : {ge::FORMAT_NCHW, ge::FORMAT_NHWC}) {
        for (size_t port = 0; port < portNames.size(); ++port) {
            SCOPED_TRACE(testing::Message() << "port=" << portNames[port] << ", storage_format=" << format);
            auto storageFormats = ND_FORMATS;
            auto originFormats = ND_FORMATS;
            storageFormats[port] = format;
            originFormats[port] = format;
            uint64_t tilingKey = 0;
            auto ret = RunTiling(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, s, s, s, storageFormats, originFormats,
                                 &tilingKey);
            EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
            if (ret == ge::GRAPH_SUCCESS) {
                EXPECT_EQ(tilingKey, ndTilingKey);
            }
        }
    }
}

TEST_F(Relu6GradTilingTest, accepts_nd_storage_with_nchw_origin)
{
    gert::StorageShape s = {{1, 2, 2, 2}, {1, 2, 2, 2}};
    const std::array<ge::Format, 3> originFormats = {ge::FORMAT_NCHW, ge::FORMAT_NCHW, ge::FORMAT_NCHW};
    EXPECT_EQ(RunTiling(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, s, s, s, ND_FORMATS, originFormats),
              ge::GRAPH_SUCCESS);
}

TEST_F(Relu6GradTilingTest, axis_broadcast_fp32)
{
    gert::StorageShape g = {{16, 1}, {16, 1}};
    gert::StorageShape f = {{1, 16}, {1, 16}};
    gert::StorageShape o = {{16, 16}, {16, 16}};
    EXPECT_EQ(RunTiling(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, g, f, o), ge::GRAPH_SUCCESS);
}

TEST_F(Relu6GradTilingTest, cross_rank_broadcast_fp16)
{
    gert::StorageShape g = {{4}, {4}};
    gert::StorageShape f = {{3, 4}, {3, 4}};
    gert::StorageShape o = {{3, 4}, {3, 4}};
    EXPECT_EQ(RunTiling(ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, g, f, o), ge::GRAPH_SUCCESS);
}

// Per-axis (outer-product) broadcast [16,1] x [1,16] -> [16,16], fp16 / bf16.
TEST_F(Relu6GradTilingTest, axis_broadcast_fp16)
{
    gert::StorageShape g = {{16, 1}, {16, 1}};
    gert::StorageShape f = {{1, 16}, {1, 16}};
    gert::StorageShape o = {{16, 16}, {16, 16}};
    EXPECT_EQ(RunTiling(ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, g, f, o), ge::GRAPH_SUCCESS);
}

TEST_F(Relu6GradTilingTest, axis_broadcast_bf16)
{
    gert::StorageShape g = {{16, 1}, {16, 1}};
    gert::StorageShape f = {{1, 16}, {1, 16}};
    gert::StorageShape o = {{16, 16}, {16, 16}};
    EXPECT_EQ(RunTiling(ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, g, f, o), ge::GRAPH_SUCCESS);
}

// Single-axis broadcast: gradients carries the size-1 dim (trailing / leading).
TEST_F(Relu6GradTilingTest, trailing_dim_broadcast_fp32)
{
    gert::StorageShape g = {{3, 1}, {3, 1}};
    gert::StorageShape f = {{3, 4}, {3, 4}};
    gert::StorageShape o = {{3, 4}, {3, 4}};
    EXPECT_EQ(RunTiling(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, g, f, o), ge::GRAPH_SUCCESS);
}

TEST_F(Relu6GradTilingTest, leading_dim_broadcast_fp16)
{
    gert::StorageShape g = {{1, 4}, {1, 4}};
    gert::StorageShape f = {{3, 4}, {3, 4}};
    gert::StorageShape o = {{3, 4}, {3, 4}};
    EXPECT_EQ(RunTiling(ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, g, f, o), ge::GRAPH_SUCCESS);
}

// Scalar operand ([1]) broadcast against a full tensor, both directions.
TEST_F(Relu6GradTilingTest, scalar_grad_broadcast_fp32)
{
    gert::StorageShape g = {{1}, {1}};
    gert::StorageShape f = {{16, 16}, {16, 16}};
    gert::StorageShape o = {{16, 16}, {16, 16}};
    EXPECT_EQ(RunTiling(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, g, f, o), ge::GRAPH_SUCCESS);
}

TEST_F(Relu6GradTilingTest, scalar_feat_broadcast_bf16)
{
    gert::StorageShape g = {{16, 16}, {16, 16}};
    gert::StorageShape f = {{1}, {1}};
    gert::StorageShape o = {{16, 16}, {16, 16}};
    EXPECT_EQ(RunTiling(ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, g, f, o), ge::GRAPH_SUCCESS);
}

// 3D per-axis mutual broadcast [2,1,4] x [1,3,1] -> [2,3,4], fp32.
TEST_F(Relu6GradTilingTest, mutual_broadcast_3d_fp32)
{
    gert::StorageShape g = {{2, 1, 4}, {2, 1, 4}};
    gert::StorageShape f = {{1, 3, 1}, {1, 3, 1}};
    gert::StorageShape o = {{2, 3, 4}, {2, 3, 4}};
    EXPECT_EQ(RunTiling(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, g, f, o), ge::GRAPH_SUCCESS);
}

// Cross-rank broadcast into 3D: gradients=[4], features=[2,3,4] -> [2,3,4], bf16.
TEST_F(Relu6GradTilingTest, cross_rank_3d_broadcast_bf16)
{
    gert::StorageShape g = {{4}, {4}};
    gert::StorageShape f = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape o = {{2, 3, 4}, {2, 3, 4}};
    EXPECT_EQ(RunTiling(ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, g, f, o), ge::GRAPH_SUCCESS);
}

// Reject case: gradients fp16 vs features fp32.
TEST_F(Relu6GradTilingTest, mix_dtype_reject)
{
    gert::StorageShape s = {{16, 16}, {16, 16}};
    EXPECT_EQ(RunTiling(ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16, s, s, s), ge::GRAPH_FAILED);
}

// Reject case: int32 not in supported dtype list.
TEST_F(Relu6GradTilingTest, int32_reject)
{
    gert::StorageShape s = {{16, 16}, {16, 16}};
    EXPECT_EQ(RunTiling(ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, s, s, s), ge::GRAPH_FAILED);
}

// 8 dims is the maximum supported rank.
TEST_F(Relu6GradTilingTest, max_dim_num_fp32)
{
    gert::StorageShape s = {{2, 1, 1, 1, 1, 1, 1, 2}, {2, 1, 1, 1, 1, 1, 1, 2}};
    EXPECT_EQ(RunTiling(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, s, s, s), ge::GRAPH_SUCCESS);
}

// Reject case: gradients has 9 dims. The other operand is a scalar, so the
// only thing wrong with this case is the rank of gradients.
TEST_F(Relu6GradTilingTest, grad_over_max_dim_num_reject)
{
    gert::StorageShape g = {{2, 1, 1, 1, 1, 1, 1, 1, 2}, {2, 1, 1, 1, 1, 1, 1, 1, 2}};
    gert::StorageShape f = {{1}, {1}};
    EXPECT_EQ(RunTiling(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, g, f, g), ge::GRAPH_FAILED);
}

// Reject case: features has 9 dims.
TEST_F(Relu6GradTilingTest, feat_over_max_dim_num_reject)
{
    gert::StorageShape g = {{1}, {1}};
    gert::StorageShape f = {{2, 1, 1, 1, 1, 1, 1, 1, 2}, {2, 1, 1, 1, 1, 1, 1, 1, 2}};
    EXPECT_EQ(RunTiling(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, g, f, f), ge::GRAPH_FAILED);
}
