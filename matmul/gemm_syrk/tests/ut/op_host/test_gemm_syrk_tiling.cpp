/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "../../../op_host/op_tiling/arch35/gemm_syrk_compile_info.h"
#include "../../../op_kernel/arch35/gemm_syrk_tiling_data.h"
#include "../../../op_kernel/arch35/gemm_syrk_tiling_key.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"

using namespace std;
using ge::DT_BF16;
using ge::DT_FLOAT16;

namespace {

struct GemmSyrkTilingTestParam {
    string case_name;
    string compile_info;
    ge::Format a_format{ge::FORMAT_ND};
    ge::Format c_format{ge::FORMAT_ND};
    ge::DataType a_dtype{ge::DT_FLOAT16};
    ge::DataType c_dtype{ge::DT_FLOAT16};
    std::initializer_list<int64_t> a_shape;
    std::initializer_list<int64_t> c_shape;
    float alpha{1.0F};
    float beta{1.0F};
    bool transpose_x{false};
    string fill_mode{"full"};
    uint32_t expect_m{0};
    uint32_t expect_n{0};
    uint32_t expect_k{0};
    uint32_t expect_batch{1};
    uint32_t expect_base_block{0}; // 0: do not check
    ge::graphStatus expect_status{ge::GRAPH_SUCCESS};
};

class GemmSyrkTilingRuntime : public testing::TestWithParam<GemmSyrkTilingTestParam> {
    virtual void SetUp() {}
};

static string get_map_string(const map<string, string>& m, const string& key)
{
    auto it = m.find(key);
    return it == m.end() ? "" : it->second;
}

void TestOneParamCase(const GemmSyrkTilingTestParam& param)
{
    gert::StorageShape a_shape = {param.a_shape, param.a_shape};
    gert::StorageShape c_shape = {param.c_shape, param.c_shape};
    std::vector<gert::StorageShape> output_shapes(1, {param.c_shape, param.c_shape});
    std::vector<void*> output_shapes_ref(1);
    output_shapes_ref[0] = &output_shapes[0];

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::GemmSyrkCompileInfo compile_info;
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(param.compile_info.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(param.compile_info.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("GemmSyrk"), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GemmSyrk")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GemmSyrk")->tiling_parse;
    auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("GemmSyrk")->gen_simplifiedkey;
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    EXPECT_GT(compile_info.aicNum, 0UL);
    if (param.expect_status == ge::GRAPH_SUCCESS) {
        // The 1:2 MIX invariant only holds for the capable platform; the
        // bad-ratio negative case deliberately violates it.
        EXPECT_EQ(compile_info.aivNum, compile_info.aicNum * 2UL);
    }

    auto tiling_data = gert::TilingData::CreateCap(2048);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    gert::KernelRunContextHolder holder;
    holder = gert::TilingContextFaker()
                 .SetOpType("GemmSyrk")
                 .NodeIoNum(2, 1)
                 .IrInstanceNum({1, 1})
                 .InputShapes({&a_shape, &c_shape})
                 .OutputShapes(output_shapes_ref)
                 .NodeAttrs({{"alpha", Ops::NN::AnyValue::CreateFrom<float>(param.alpha)},
                             {"beta", Ops::NN::AnyValue::CreateFrom<float>(param.beta)},
                             {"transpose_x", Ops::NN::AnyValue::CreateFrom<bool>(param.transpose_x)},
                             {"fill_mode", Ops::NN::AnyValue::CreateFrom<std::string>(param.fill_mode)}})
                 .NodeInputTd(0, param.a_dtype, param.a_format, param.a_format)
                 .NodeInputTd(1, param.c_dtype, param.c_format, param.c_format)
                 .NodeOutputTd(0, param.c_dtype, param.c_format, param.c_format)
                 .CompileInfo(&compile_info)
                 .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                 .TilingData(tiling_data.get())
                 .Workspace(ws_size)
                 .Build();

    auto tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    auto ret = tiling_func(tiling_context);
    EXPECT_EQ(ret, param.expect_status) << param.case_name;
    if (param.expect_status != ge::GRAPH_SUCCESS) {
        return;
    }

    ge::char_t simplifiedKey[100] = {0};
    // Without an explicit GenSimplifiedKey registration the runtime uses the
    // default key derivation; only invoke the hook when it is registered.
    if (gen_simplifiedkey_func != nullptr) {
        ASSERT_EQ(gen_simplifiedkey_func(tiling_context, simplifiedKey), ge::GRAPH_SUCCESS);
    }
    uint64_t tiling_key = tiling_context->GetTilingKey();
    uint32_t block_dim = tiling_context->GetBlockDim();
    cout << "===== " << param.case_name << ": tiling_key " << tiling_key << ", block_dim " << block_dim << std::endl;

    EXPECT_EQ(tiling_key,
              GET_TPL_TILING_KEY(SYRK_KERNEL_BASIC, param.transpose_x ? SYRK_TRANS_TRUE : SYRK_TRANS_FALSE));

    auto raw_tiling_data = tiling_context->GetRawTilingData();
    ASSERT_GE(raw_tiling_data->GetDataSize(), sizeof(GemmSyrkTilingData));
    const auto* syrkTilingData = reinterpret_cast<const GemmSyrkTilingData*>(raw_tiling_data->GetData());
    cout << "===== " << param.case_name << " tiling detail: baseBlock " << syrkTilingData->baseBlock << ", baseK "
         << syrkTilingData->baseK << ", kL1 " << syrkTilingData->kL1 << ", usedCoreNum " << syrkTilingData->usedCoreNum
         << std::endl;
    EXPECT_EQ(syrkTilingData->m, param.expect_m);
    EXPECT_EQ(syrkTilingData->n, param.expect_n);
    EXPECT_EQ(syrkTilingData->k, param.expect_k);
    EXPECT_EQ(syrkTilingData->batch, param.expect_batch);
    EXPECT_FLOAT_EQ(syrkTilingData->alpha, param.alpha);
    EXPECT_FLOAT_EQ(syrkTilingData->beta, param.beta);

    // Symmetric square contract: one 16-aligned block (baseM == baseN ==
    // mL1 == nL1), bounded by floor16(sqrt(L0C/2/4)) = 176 on Ascend 950
    // (single L0C half-slot accumulator).
    ASSERT_GT(syrkTilingData->baseBlock, 0U);
    EXPECT_EQ(syrkTilingData->baseBlock % 16U, 0U);
    EXPECT_LE(syrkTilingData->baseBlock, 176U);
    if (param.expect_base_block != 0U) {
        EXPECT_EQ(syrkTilingData->baseBlock, param.expect_base_block) << param.case_name;
    }
    EXPECT_GT(syrkTilingData->baseK, 0U);
    EXPECT_EQ(syrkTilingData->baseK % 16U, 0U);
    EXPECT_GE(syrkTilingData->kL1, syrkTilingData->baseK);
    EXPECT_GT(syrkTilingData->usedCoreNum, 0U);
    EXPECT_LE(syrkTilingData->usedCoreNum, 32U);
    EXPECT_EQ(block_dim, syrkTilingData->usedCoreNum);
}

static const string kAscend950CompileInfo = R"({
    "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown",
    "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false,
    "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true,
    "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288,
    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144,
    "CORE_NUM": 32, "socVersion": "Ascend950",
    "core_type_list": "CubeCore,VectorCore",
    "cube_core_cnt": 32, "vector_core_cnt": 64}
})";

// Ascend950 with non-1:2 aic:aiv ratio (cube=32, vector=48): IsCapable fails.
static const string kAscend950CompileInfoBadRatio = R"({
    "hardware_info": {"BT_SIZE": 4096, "load3d_constraints": "unknown",
    "Intrinsic_fix_pipe_l0c2out": true, "Intrinsic_data_move_l12ub": false,
    "Intrinsic_data_move_l0c2ub": false, "Intrinsic_data_move_out2l1_nd2nz": true,
    "UB_SIZE": 253952, "L2_SIZE": 134217728, "L1_SIZE": 524288,
    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144,
    "CORE_NUM": 32, "socVersion": "Ascend950",
    "core_type_list": "CubeCore,VectorCore",
    "cube_core_cnt": 32, "vector_core_cnt": 48}
})";

static const GemmSyrkTilingTestParam general_cases_params[] = {
    // 2D fp16, m=256: block sizing comes from the BMM ASW basic strategy + syrk clamps.
    {"GemmSyrk_950_basic_2d",
     kAscend950CompileInfo,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     DT_FLOAT16,
     DT_FLOAT16,
     {256, 512},
     {256, 256},
     3.0F,
     2.0F,
     false,
     "full",
     256,
     256,
     512,
     1,
     ge::GRAPH_SUCCESS},
    // 3D fp16, batch=2, m=128.
    {"GemmSyrk_950_basic_3d",
     kAscend950CompileInfo,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     DT_FLOAT16,
     DT_FLOAT16,
     {2, 128, 64},
     {2, 128, 128},
     1.0F,
     1.0F,
     false,
     "full",
     128,
     128,
     64,
     2,
     ge::GRAPH_SUCCESS},
    // bf16 tail, m=17.
    {"GemmSyrk_950_bf16_tail",
     kAscend950CompileInfo,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     DT_BF16,
     DT_BF16,
     {17, 10},
     {17, 17},
     3.687209F,
     2.067589F,
     false,
     "full",
     17,
     17,
     10,
     1,
     ge::GRAPH_SUCCESS},
    // transpose_x, 2D: a stored (k, m) = (512, 256).
    {"GemmSyrk_950_trans_2d",
     kAscend950CompileInfo,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     DT_FLOAT16,
     DT_FLOAT16,
     {512, 256},
     {256, 256},
     3.0F,
     2.0F,
     true,
     "full",
     256,
     256,
     512,
     1,
     ge::GRAPH_SUCCESS},
    // transpose_x, 3D: a stored (batch, k, m) = (2, 64, 128).
    {"GemmSyrk_950_trans_3d",
     kAscend950CompileInfo,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     DT_FLOAT16,
     DT_FLOAT16,
     {2, 64, 128},
     {2, 128, 128},
     1.0F,
     1.0F,
     true,
     "full",
     128,
     128,
     64,
     2,
     ge::GRAPH_SUCCESS},
    // Skinny tall-k: m=40 < one 48 block, k=63300. The syrk traffic-optimal
    // layout is the single diagonal slot (nB == 1, every a row fetched once);
    // the wave cap must not split it into a 2x2 grid (2x GM->L1 traffic plus
    // thin 8-row tail fetches).
    {"GemmSyrk_950_skinny_40_63300",
     kAscend950CompileInfo,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     DT_FLOAT16,
     DT_FLOAT16,
     {40, 63300},
     {40, 40},
     1.0F,
     1.0F,
     false,
     "full",
     40,
     40,
     63300,
     1,
     ge::GRAPH_SUCCESS},
    // Perf regression shapes from the on-board sweep (see stc HANDOFF):
    // block sizing must stay wave-adaptive for these mid/large m.
    {"GemmSyrk_950_perf_512_4096",
     kAscend950CompileInfo,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     DT_FLOAT16,
     DT_FLOAT16,
     {512, 4096},
     {512, 512},
     1.0F,
     1.0F,
     false,
     "full",
     512,
     512,
     4096,
     1,
     80,
     ge::GRAPH_SUCCESS},
    {"GemmSyrk_950_perf_768_8192",
     kAscend950CompileInfo,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     DT_FLOAT16,
     DT_FLOAT16,
     {768, 8192},
     {768, 768},
     1.0F,
     1.0F,
     false,
     "full",
     768,
     768,
     8192,
     1,
     ge::GRAPH_SUCCESS},
    {"GemmSyrk_950_perf_1024_16384",
     kAscend950CompileInfo,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     DT_FLOAT16,
     DT_FLOAT16,
     {1024, 16384},
     {1024, 1024},
     1.0F,
     1.0F,
     false,
     "full",
     1024,
     1024,
     16384,
     1,
     ge::GRAPH_SUCCESS},
    {"GemmSyrk_950_perf_1024_4096",
     kAscend950CompileInfo,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     DT_FLOAT16,
     DT_FLOAT16,
     {1024, 4096},
     {1024, 1024},
     1.0F,
     1.0F,
     false,
     "full",
     1024,
     1024,
     4096,
     1,
     160,
     ge::GRAPH_SUCCESS},
    // Reject: c is not square.
    {"GemmSyrk_950_reject_non_square_c",
     kAscend950CompileInfo,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     DT_FLOAT16,
     DT_FLOAT16,
     {32, 64},
     {32, 16},
     1.0F,
     1.0F,
     false,
     "full",
     0,
     0,
     0,
     1,
     0,
     ge::GRAPH_FAILED},
    // Reject: batch-axis mismatch (no in-place broadcast).
    {"GemmSyrk_950_reject_batch_mismatch",
     kAscend950CompileInfo,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     DT_FLOAT16,
     DT_FLOAT16,
     {2, 32, 64},
     {3, 32, 32},
     1.0F,
     1.0F,
     false,
     "full",
     0,
     0,
     0,
     1,
     0,
     ge::GRAPH_FAILED},
    // Reject: k == 0 (aclnn routes this to an elementwise scaling instead).
    {"GemmSyrk_950_reject_k_zero",
     kAscend950CompileInfo,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     DT_FLOAT16,
     DT_FLOAT16,
     {32, 0},
     {32, 32},
     1.0F,
     1.0F,
     false,
     "full",
     0,
     0,
     0,
     1,
     0,
     ge::GRAPH_FAILED},
    // Reject: dtype mismatch between a and c.
    {"GemmSyrk_950_reject_dtype_mismatch",
     kAscend950CompileInfo,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     DT_FLOAT16,
     DT_BF16,
     {32, 64},
     {32, 32},
     1.0F,
     1.0F,
     false,
     "full",
     0,
     0,
     0,
     1,
     0,
     ge::GRAPH_FAILED},
    // Reject: transpose_x with no matching m axis on a's last dim.
    {"GemmSyrk_950_reject_trans_m_mismatch",
     kAscend950CompileInfo,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     DT_FLOAT16,
     DT_FLOAT16,
     {64, 32},
     {64, 64},
     1.0F,
     1.0F,
     true,
     "full",
     0,
     0,
     0,
     1,
     0,
     ge::GRAPH_FAILED},
    // Reject: fill_mode "up" is declared but not implemented.
    {"GemmSyrk_950_reject_fill_mode_up",
     kAscend950CompileInfo,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     DT_FLOAT16,
     DT_FLOAT16,
     {256, 512},
     {256, 256},
     1.0F,
     1.0F,
     false,
     "up",
     0,
     0,
     0,
     1,
     0,
     ge::GRAPH_FAILED},
    // Reject: aicNum:aivNum != 1:2 (cube=32, vector=48): the basic tiling's
    // IsCapable fails and the strategy has no further priority.
    {"GemmSyrk_950_reject_aiv_not_double_aic",
     kAscend950CompileInfoBadRatio,
     ge::FORMAT_ND,
     ge::FORMAT_ND,
     DT_FLOAT16,
     DT_FLOAT16,
     {32, 64},
     {32, 32},
     1.0F,
     1.0F,
     false,
     "full",
     0,
     0,
     0,
     1,
     0,
     ge::GRAPH_FAILED},
};

INSTANTIATE_TEST_SUITE_P(GemmSyrkTilingSuite, GemmSyrkTilingRuntime, testing::ValuesIn(general_cases_params));

TEST_P(GemmSyrkTilingRuntime, general_cases) { TestOneParamCase(GetParam()); }
} // namespace
