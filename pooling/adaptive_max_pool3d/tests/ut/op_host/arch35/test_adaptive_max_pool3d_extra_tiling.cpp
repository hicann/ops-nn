/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_adaptive_max_pool3d_extra_tiling.cpp
 * \brief AdaptiveMaxPool3d arch35 tiling 覆盖增强: adaptive_pool3d_common 侧 para(0)/big_kernel(1)/
 *        simt(2) 模板选中链路与 AdaptivePool3dBaseTiling(GetAndCheckIndicesDtype/GetShapeAttrsInfo) 未覆盖分支。
 *
 * 与既有 test_adaptive_max_pool3d_tiling.cpp 的差异: 既有用例未设置平台 "version" 信息, common 基类
 * GetShapeAttrsInfo(adaptive_pool3d_tiling.cpp 检查 npuArch==DAV_3510)返回 GRAPH_PARAM_INVALID,
 * 0/1/2 号模板全部 fallthrough 到算子自有的 3/4 号模板; 本文件通过 SetPlatformRes("version",
 * {Short_SoC_version:Ascend950, NpuArch:3510}) 使 common 模板真正参与选择, 覆盖:
 *   - para: IsCapable(kprod<128/DHW<INT32_MAX/NC>=vfLen/2/UB)/BinarySearch/SearchOuter 各分支/PostTiling
 *   - big_kernel: IsCapable 三档阈值([128,256)/[256,1024)/[1024,+inf) 含两档拒绝)/DoBlockTiling 双分支
 *   - simt: GetAndCheckIndicesDtype 全部分支/四档 GetTilingKey/MIN|MAX_THREAD_NUM/PostTiling
 *   - base: 4D 输入/output_size attr 缺失/空 tensor/CalKernelSizeOneDimMax 快捷分支
 *
 * 说明: 平台 UB=245760/CORE_NUM=64; vfLen=256/dtypeSize(fp16:128, fp32:64); alignNum=32/dtypeSize。
 * para 的 SearchOuterSingle 以 blockFactor 是否增大决定回退, 而 blockFactor 在首次进入
 * SearchOuterSingle 前从未计算(初值 0), 故第一个可缩减的 factor 总会被还原, 预期值按此推演。
 *
 * tiling key 位编码(adaptive_pool3d_tiling_struct.h ASCENDC_TPL_ARGS_DECL):
 *   TEMPLATE_MODE(2bit<<0) | DYTPE_MODE(3bit<<2) | MULTI_MODE(1bit<<5) | FORMAT_MODE(1bit<<6)
 *   para=(0,0,0,0)=0; big=(1,0,0,0)=1; simt=(2,{1,2,3,4},0,0)={6,10,14,18}
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"
#include "pooling/adaptive_pool3d_common/op_host/arch35/adaptive_pool3d_tiling.h"

using namespace std;
using namespace ge;

namespace optiling {
struct AdaptiveMaxPool3dCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSizePlatForm = 0;
};
} // namespace optiling

class AdaptiveMaxPool3dExtraTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AdaptiveMaxPool3dExtraTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AdaptiveMaxPool3dExtraTiling TearDown" << std::endl; }
};

/*
 * 通用执行器: 构造 Ascend950(NpuArch=3510, UB=245760, 64 核) 平台的 TilingContext 并执行 tiling_func。
 * 双输出: y(跟随输入 dtype) + indices(INT32/INT64/非法值); attrs 由用例自行构造(output_size 必带,
 * indices_dtype 可选, 缺省时 GetAndCheckIndicesDtype 走 attrDtypePtr==nullptr 默认 INT32 分支)。
 * expectedStatus: 期望返回值; expectedKey/expectedBlockDim: 期望 tiling key 与 block dim(-1 表示不断言)。
 */
static void ExecuteAdaptiveMaxPool3dExtraCase(gert::StorageShape& xShape, gert::StorageShape& yShape,
                                              ge::DataType dataType, ge::DataType indicesDataType,
                                              const std::vector<std::pair<std::string, Ops::NN::AnyValue>>& attrs,
                                              ge::graphStatus expectedStatus, uint64_t expectedKey = 0,
                                              int64_t expectedBlockDim = -1)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    std::map<std::string, std::string> npu_arch_infos = {{"NpuArch", "3510"}};
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    optiling::AdaptiveMaxPool3dCompileInfo compile_info;

    string op_type("AdaptiveMaxPool3d");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto tiling_data = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(tiling_data, nullptr);

    gert::StorageShape indices_shape = yShape;
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &indices_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dataType, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(0, dataType, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(1, indicesDataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(attrs)
                      .TilingData(tiling_data.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", npu_arch_infos);

    ASSERT_EQ(tiling_func(tiling_context), expectedStatus);
    if (expectedStatus == ge::GRAPH_SUCCESS) {
        ASSERT_EQ(tiling_context->GetTilingKey(), expectedKey);
    }
    if (expectedBlockDim >= 0) {
        ASSERT_EQ(static_cast<int64_t>(tiling_context->GetBlockDim()), expectedBlockDim);
    }
}

// 便捷构造: output_size + indices_dtype(缺省不带时 hasIndicesDtypeAttr=false)
static std::vector<std::pair<std::string, Ops::NN::AnyValue>> MakeAttrs(std::vector<int64_t> outputSize,
                                                                        bool hasIndicesDtypeAttr = true,
                                                                        int64_t indicesDtypeAttr = 3)
{
    std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrs = {
        {"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(outputSize)}};
    if (hasIndicesDtypeAttr) {
        attrs.push_back({"indices_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(indicesDtypeAttr)});
    }
    return attrs;
}

// ===================== Para 模板(优先级 0) 正向选中 =====================

// para fp16: kD=kH=2/kW=4(kprod=16<128), NC=64>=vfLen/2, IsCapable 的 UB 检查通过(occupy=114688);
// InitUbFactor(do=ho=wo=2) 后 occupy=458752>245760 -> BinarySearch(doFactor) 收缩到 1 后满足;
// SearchOuter: doFactor=1 跳过, hoFactor=2 是首个进入的 SearchOuterSingle(blockFactor 0->1 判增,
// 还原 ho=2), woFactor 2->1 正常收缩, 最终 totalOuter=4/useCoreNum=4; key=0, blockDim=4
TEST_F(AdaptiveMaxPool3dExtraTiling, adaptive_max_pool3d_para_fp16_binary_search)
{
    gert::StorageShape xShape = {{1, 64, 4, 4, 8}, {1, 64, 4, 4, 8}};
    gert::StorageShape yShape = {{1, 64, 2, 2, 2}, {1, 64, 2, 2, 2}};
    ExecuteAdaptiveMaxPool3dExtraCase(xShape, yShape, ge::DT_FLOAT16, ge::DT_INT32, MakeAttrs({2, 2, 2}),
                                      ge::GRAPH_SUCCESS, 0, 4);
}

// para fp32: vfLen=64, NC=32>=32; SearchUbFactor 首个判断直接返回(初始 occupy=163840<245760,
// BinarySearch 未进入); SearchOuter: doFactor=2 首个进入被还原, ho/wo 各收缩到 1,
// 最终 totalOuter=4/useCoreNum=4; key=0, blockDim=4
TEST_F(AdaptiveMaxPool3dExtraTiling, adaptive_max_pool3d_para_fp32_ub_fit_without_search)
{
    gert::StorageShape xShape = {{1, 32, 4, 4, 8}, {1, 32, 4, 4, 8}};
    gert::StorageShape yShape = {{1, 32, 2, 2, 2}, {1, 32, 2, 2, 2}};
    ExecuteAdaptiveMaxPool3dExtraCase(xShape, yShape, ge::DT_FLOAT, ge::DT_INT32, MakeAttrs({2, 2, 2}),
                                      ge::GRAPH_SUCCESS, 0, 4);
}

// para fp16: kD=kH=kW=2(kprod=8<128), NC=64; BinarySearch(doFactor=4) 两步均超 UB 收缩到 1(覆盖
// right=mid-1 回退), BinarySearch(hoFactor=4) 中 mid=2 命中(bestSplit 前进到 2, 覆盖 left=mid+1),
// SearchOuterSingle(woFactor=4) 连续 3 轮收缩直到 initFactor=1 退出; 最终 totalOuter=32/useCoreNum=32;
// key=0, blockDim=32
TEST_F(AdaptiveMaxPool3dExtraTiling, adaptive_max_pool3d_para_fp16_multi_step_search)
{
    gert::StorageShape xShape = {{1, 64, 8, 8, 8}, {1, 64, 8, 8, 8}};
    gert::StorageShape yShape = {{1, 64, 4, 4, 4}, {1, 64, 4, 4, 4}};
    ExecuteAdaptiveMaxPool3dExtraCase(xShape, yShape, ge::DT_FLOAT16, ge::DT_INT32, MakeAttrs({4, 4, 4}),
                                      ge::GRAPH_SUCCESS, 0, 32);
}

// para fp16 不带 indices_dtype attr: GetAndCheckIndicesDtype 中 GetAttrPointer<int>(1) 越界返回
// nullptr, 走 attrDtype=DTYPE_INT32 默认分支(与 def 文件 indices_dtype 缺省值一致); key=0, blockDim=4
TEST_F(AdaptiveMaxPool3dExtraTiling, adaptive_max_pool3d_para_fp16_indices_dtype_attr_default)
{
    gert::StorageShape xShape = {{1, 64, 4, 4, 8}, {1, 64, 4, 4, 8}};
    gert::StorageShape yShape = {{1, 64, 2, 2, 2}, {1, 64, 2, 2, 2}};
    ExecuteAdaptiveMaxPool3dExtraCase(xShape, yShape, ge::DT_FLOAT16, ge::DT_INT32, MakeAttrs({2, 2, 2}, false),
                                      ge::GRAPH_SUCCESS, 0, 4);
}

// para fp16 4D 输入: base GetShapeAttrsInfo 走 DIM_NUM_FOUR 分支(nIn=1, cIn=dim0, d/h/w=dim1..3),
// 等价于 {1,64,4,4,8}; key=0, blockDim=4
TEST_F(AdaptiveMaxPool3dExtraTiling, adaptive_max_pool3d_para_fp16_4d_input)
{
    gert::StorageShape xShape = {{64, 4, 4, 8}, {64, 4, 4, 8}};
    gert::StorageShape yShape = {{64, 2, 2, 2}, {64, 2, 2, 2}};
    ExecuteAdaptiveMaxPool3dExtraCase(xShape, yShape, ge::DT_FLOAT16, ge::DT_INT32, MakeAttrs({2, 2, 2}),
                                      ge::GRAPH_SUCCESS, 0, 4);
}

// para fp16: NC=4096(ncOuter=32), dOut=1/hOut=2/wOut=1, UB 一次通过无需收缩;
// SearchOuter 中 hoFactor=2 作为首个进入的 SearchOuterSingle, 其内部唯一一次 CalUbBlockFactor
// (ho=1) 得 totalOuter=64/useCoreNum=64 后还原退出, 紧接着的 useCoreNum==coreNum 判断命中
// SearchOuter 的提前返回分支; 最终 CalUbBlockFactor 回到 ho=2 得 useCoreNum=32; key=0, blockDim=32
TEST_F(AdaptiveMaxPool3dExtraTiling, adaptive_max_pool3d_para_search_outer_early_return)
{
    gert::StorageShape xShape = {{1, 4096, 2, 2, 1}, {1, 4096, 2, 2, 1}};
    gert::StorageShape yShape = {{1, 4096, 1, 2, 1}, {1, 4096, 1, 2, 1}};
    ExecuteAdaptiveMaxPool3dExtraCase(xShape, yShape, ge::DT_FLOAT16, ge::DT_INT32, MakeAttrs({1, 2, 1}),
                                      ge::GRAPH_SUCCESS, 0, 32);
}

// para fp16: NC=2816(ncOuter=22), dOut=1/hOut=2/wOut=3; SearchOuterSingle(woFactor=3) 第 2 轮
// wo=1 时 totalOuter 44->66 使 blockFactor 1->2 增大, 命中回退还原分支(break 且 woFactor 还原为 2);
// 最终 totalOuter=44/useCoreNum=44; key=0, blockDim=44
TEST_F(AdaptiveMaxPool3dExtraTiling, adaptive_max_pool3d_para_search_outer_block_factor_increase_break)
{
    gert::StorageShape xShape = {{1, 2816, 2, 2, 3}, {1, 2816, 2, 2, 3}};
    gert::StorageShape yShape = {{1, 2816, 1, 2, 3}, {1, 2816, 1, 2, 3}};
    ExecuteAdaptiveMaxPool3dExtraCase(xShape, yShape, ge::DT_FLOAT16, ge::DT_INT32, MakeAttrs({1, 2, 3}),
                                      ge::GRAPH_SUCCESS, 0, 44);
}

// para fp16: NC=2048(ncOuter=16), dOut=1/hOut=2/wOut=4; SearchOuterSingle(woFactor=4) 逐轮收缩
// (wo=3/2 时 totalOuter=32 继续循环, wo=1 时 totalOuter=64 且 blockFactor 保持 1), while 条件因
// useCoreNum==coreNum 退出(非 initFactor 退出); 最终 totalOuter=64/useCoreNum=64; key=0, blockDim=64
TEST_F(AdaptiveMaxPool3dExtraTiling, adaptive_max_pool3d_para_search_outer_exit_by_core_num)
{
    gert::StorageShape xShape = {{1, 2048, 2, 2, 4}, {1, 2048, 2, 2, 4}};
    gert::StorageShape yShape = {{1, 2048, 1, 2, 4}, {1, 2048, 1, 2, 4}};
    ExecuteAdaptiveMaxPool3dExtraCase(xShape, yShape, ge::DT_FLOAT16, ge::DT_INT32, MakeAttrs({1, 2, 4}),
                                      ge::GRAPH_SUCCESS, 0, 64);
}

// ===================== BigKernel 模板(优先级 1) 正向选中 =====================

// big_kernel fp16: kD=kH=kW=32(kprod=32768>=1024, 第三档无输出规模限制), para 因 kprod>=128 拒绝;
// totalIdx=1<coreNum -> blockFactor=0 分支(coreNums=totalIdx=1), fp16 maxCount 走 B2 上限 32640;
// key=(MODE_1,DTYPE_0,0,0)=1, blockDim=coreNums=1
TEST_F(AdaptiveMaxPool3dExtraTiling, adaptive_max_pool3d_big_kernel_fp16_single_output)
{
    gert::StorageShape xShape = {{1, 1, 32, 32, 32}, {1, 1, 32, 32, 32}};
    gert::StorageShape yShape = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    ExecuteAdaptiveMaxPool3dExtraCase(xShape, yShape, ge::DT_FLOAT16, ge::DT_INT32, MakeAttrs({1, 1, 1}),
                                      ge::GRAPH_SUCCESS, 1, 1);
}

// big_kernel fp32: 同上形状, xDtype=DT_FLOAT 时 maxCount=defaultMaxSize/4 不再受 B2 上限(29696);
// key=1, blockDim=1
TEST_F(AdaptiveMaxPool3dExtraTiling, adaptive_max_pool3d_big_kernel_fp32_single_output)
{
    gert::StorageShape xShape = {{1, 1, 32, 32, 32}, {1, 1, 32, 32, 32}};
    gert::StorageShape yShape = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    ExecuteAdaptiveMaxPool3dExtraCase(xShape, yShape, ge::DT_FLOAT, ge::DT_INT32, MakeAttrs({1, 1, 1}),
                                      ge::GRAPH_SUCCESS, 1, 1);
}

// big_kernel 第一档: kD=kH=kW=6(kprod=216 属 [128,256)) 且 totalOut=1<5120 -> 命中;
// batchCount=32640/216=151; key=1, blockDim=1
TEST_F(AdaptiveMaxPool3dExtraTiling, adaptive_max_pool3d_big_kernel_band_128_to_256)
{
    gert::StorageShape xShape = {{1, 1, 6, 6, 6}, {1, 1, 6, 6, 6}};
    gert::StorageShape yShape = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    ExecuteAdaptiveMaxPool3dExtraCase(xShape, yShape, ge::DT_FLOAT16, ge::DT_INT32, MakeAttrs({1, 1, 1}),
                                      ge::GRAPH_SUCCESS, 1, 1);
}

// big_kernel 第二档多核: kD=kH=kW=8(kprod=512 属 [256,1024)) 且 totalOut=2*40=80<10240;
// totalIdx=80>=coreNum=64 -> blockFactor=1>0 分支(coreNums=64, blockTail=16); key=1, blockDim=64
TEST_F(AdaptiveMaxPool3dExtraTiling, adaptive_max_pool3d_big_kernel_band_256_to_1024_multi_core)
{
    gert::StorageShape xShape = {{2, 40, 8, 8, 8}, {2, 40, 8, 8, 8}};
    gert::StorageShape yShape = {{2, 40, 1, 1, 1}, {2, 40, 1, 1, 1}};
    ExecuteAdaptiveMaxPool3dExtraCase(xShape, yShape, ge::DT_FLOAT16, ge::DT_INT32, MakeAttrs({1, 1, 1}),
                                      ge::GRAPH_SUCCESS, 1, 64);
}

// ===================== Simt 模板(优先级 2, 兜底) 正向选中 =====================

// simt fp16: kprod=8<128 且 NC=1<vfLen/2 -> para/big 均拒绝后落到 simt(2);
// kW=2<=32 -> threadNum=MAX_THREAD_NUM=1024, outputDataCount=64 -> blockNum=1;
// indexNeedNum=512/divNeedNum=64 均 <INT32_MAX -> key=(MODE_2,INT32_UINT32,0,0)=6, blockDim=1
TEST_F(AdaptiveMaxPool3dExtraTiling, adaptive_max_pool3d_simt_fp16_max_thread)
{
    gert::StorageShape xShape = {{1, 1, 8, 8, 8}, {1, 1, 8, 8, 8}};
    gert::StorageShape yShape = {{1, 1, 4, 4, 4}, {1, 1, 4, 4, 4}};
    ExecuteAdaptiveMaxPool3dExtraCase(xShape, yShape, ge::DT_FLOAT16, ge::DT_INT32, MakeAttrs({4, 4, 4}),
                                      ge::GRAPH_SUCCESS, 6, 1);
}

// simt MIN_THREAD_NUM 分支: kW(64,1)=64>32 -> threadNum=512; NC=63<64 使 para 拒绝,
// dOut=9>dIn=1 时 kD=1 故 kprod=64<128 使 big 拒绝; outputDataCount=63*9=567 -> blockNum=ceil(567/512)=2;
// key=6, blockDim=2
TEST_F(AdaptiveMaxPool3dExtraTiling, adaptive_max_pool3d_simt_min_thread)
{
    gert::StorageShape xShape = {{1, 63, 1, 1, 64}, {1, 63, 1, 1, 64}};
    gert::StorageShape yShape = {{1, 63, 9, 1, 1}, {1, 63, 9, 1, 1}};
    ExecuteAdaptiveMaxPool3dExtraCase(xShape, yShape, ge::DT_FLOAT16, ge::DT_INT32, MakeAttrs({9, 1, 1}),
                                      ge::GRAPH_SUCCESS, 6, 2);
}

// simt INT64_UINT32 档: dIn=hIn=wIn=1296, dOut=hOut=wOut=324(窗口恰为 4, kprod=64<128);
// D*H*W=1296^3=2176782336>INT32_MAX -> para 的 isIndexSizeMeet 拒绝, 且 indices 必须为 INT64
// (覆盖 GetAndCheckIndicesDtype 的 INT32_MAX 强制 INT64 校验通过路径);
// indexNeedNum=2176782336>INT32_MAX 而 divNeedNum=34012224<INT32_MAX -> key=10;
// blockNum=ceil(34012224/1024)=33215 截断到 coreNum=64; blockDim=64
TEST_F(AdaptiveMaxPool3dExtraTiling, adaptive_max_pool3d_simt_int64_index_uint32_div)
{
    gert::StorageShape xShape = {{1, 1, 1296, 1296, 1296}, {1, 1, 1296, 1296, 1296}};
    gert::StorageShape yShape = {{1, 1, 324, 324, 324}, {1, 1, 324, 324, 324}};
    ExecuteAdaptiveMaxPool3dExtraCase(xShape, yShape, ge::DT_FLOAT16, ge::DT_INT64, MakeAttrs({324, 324, 324}, true, 9),
                                      ge::GRAPH_SUCCESS, 10, 64);
}

// simt INT32_UINT64 档: dOut=40000>10000 -> CalKernelSizeOneDimMax 走快捷分支(覆盖 base 85-87 行),
// kD=(60000+39999)/40000+1=3, kprod=3<128; NC=1 使 para 拒绝;
// indexNeedNum=60000<=INT32_MAX 而 divNeedNum=dIn*dOut=2400000000>INT32_MAX -> key=14;
// blockNum=ceil(40000/1024)=40; blockDim=40
TEST_F(AdaptiveMaxPool3dExtraTiling, adaptive_max_pool3d_simt_int32_index_uint64_div)
{
    gert::StorageShape xShape = {{1, 1, 60000, 1, 1}, {1, 1, 60000, 1, 1}};
    gert::StorageShape yShape = {{1, 1, 40000, 1, 1}, {1, 1, 40000, 1, 1}};
    ExecuteAdaptiveMaxPool3dExtraCase(xShape, yShape, ge::DT_FLOAT16, ge::DT_INT32, MakeAttrs({40000, 1, 1}),
                                      ge::GRAPH_SUCCESS, 14, 40);
}

// simt INT64_UINT64 档: 输出尺寸等于输入(恒等池化, kD=kH=kW=1, kprod=1<128), D*H*W=1291^3=
// 2151685171>INT32_MAX -> indices 必须 INT64; indexNeedNum 与 divNeedNum(=outputDataCount)均>
// INT32_MAX -> key=18; blockNum 截断到 64; blockDim=64
TEST_F(AdaptiveMaxPool3dExtraTiling, adaptive_max_pool3d_simt_int64_index_uint64_div)
{
    gert::StorageShape xShape = {{1, 1, 1291, 1291, 1291}, {1, 1, 1291, 1291, 1291}};
    gert::StorageShape yShape = {{1, 1, 1291, 1291, 1291}, {1, 1, 1291, 1291, 1291}};
    ExecuteAdaptiveMaxPool3dExtraCase(xShape, yShape, ge::DT_FLOAT16, ge::DT_INT64,
                                      MakeAttrs({1291, 1291, 1291}, true, 9), ge::GRAPH_SUCCESS, 18, 64);
}

// ===================== BigKernel 阈值拒绝后 fallthrough 到 Simt =====================

// big_kernel 第一档拒绝: kprod=216 属 [128,256) 但 totalOut=8*80*2*2*2=5120 不满足 <5120,
// para 亦因 kprod>=128 拒绝 -> simt; outputDataCount=5120 -> blockNum=ceil(5120/1024)=5;
// key=6, blockDim=5
TEST_F(AdaptiveMaxPool3dExtraTiling, adaptive_max_pool3d_big_kernel_band1_reject_to_simt)
{
    gert::StorageShape xShape = {{8, 80, 12, 12, 12}, {8, 80, 12, 12, 12}};
    gert::StorageShape yShape = {{8, 80, 2, 2, 2}, {8, 80, 2, 2, 2}};
    ExecuteAdaptiveMaxPool3dExtraCase(xShape, yShape, ge::DT_FLOAT16, ge::DT_INT32, MakeAttrs({2, 2, 2}),
                                      ge::GRAPH_SUCCESS, 6, 5);
}

// big_kernel 第二档拒绝: kprod=512 属 [256,1024) 但 totalOut=10241 不满足 <10240,
// para 因 kprod>=128 拒绝 -> simt; outputDataCount=10241 -> blockNum=ceil(10241/1024)=11;
// key=6, blockDim=11
TEST_F(AdaptiveMaxPool3dExtraTiling, adaptive_max_pool3d_big_kernel_band2_reject_to_simt)
{
    gert::StorageShape xShape = {{1, 10241, 8, 8, 8}, {1, 10241, 8, 8, 8}};
    gert::StorageShape yShape = {{1, 10241, 1, 1, 1}, {1, 10241, 1, 1, 1}};
    ExecuteAdaptiveMaxPool3dExtraCase(xShape, yShape, ge::DT_FLOAT16, ge::DT_INT32, MakeAttrs({1, 1, 1}),
                                      ge::GRAPH_SUCCESS, 6, 11);
}

// ===================== GetAndCheckIndicesDtype 错误分支 =====================

// simt 侧 indices 输出 dtype 非法(DT_FLOAT): kprod=8/NC=1 使 para/big 先拒绝,
// simt DoOpTiling 的 GetAndCheckIndicesDtype 要求 INT32/INT64 -> GRAPH_FAILED
TEST_F(AdaptiveMaxPool3dExtraTiling, adaptive_max_pool3d_simt_invalid_indices_dtype)
{
    gert::StorageShape xShape = {{1, 1, 8, 8, 8}, {1, 1, 8, 8, 8}};
    gert::StorageShape yShape = {{1, 1, 4, 4, 4}, {1, 1, 4, 4, 4}};
    ExecuteAdaptiveMaxPool3dExtraCase(xShape, yShape, ge::DT_FLOAT16, ge::DT_FLOAT, MakeAttrs({4, 4, 4}),
                                      ge::GRAPH_FAILED);
}

// big_kernel 侧 indices 输出 dtype 非法(DT_FLOAT): kprod=32768>=1024 使 big IsCapable 通过,
// DoOpTiling 的 GetAndCheckIndicesDtype 失败 -> GRAPH_FAILED(不再 fallthrough 到 3/4 号模板)
TEST_F(AdaptiveMaxPool3dExtraTiling, adaptive_max_pool3d_big_kernel_invalid_indices_dtype)
{
    gert::StorageShape xShape = {{1, 1, 32, 32, 32}, {1, 1, 32, 32, 32}};
    gert::StorageShape yShape = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    ExecuteAdaptiveMaxPool3dExtraCase(xShape, yShape, ge::DT_FLOAT16, ge::DT_FLOAT, MakeAttrs({1, 1, 1}),
                                      ge::GRAPH_FAILED);
}

// D*H*W=1300^3=2197000000>INT32_MAX 但 indices=DT_INT32: para 因 isIndexSizeMeet 拒绝,
// big IsCapable 通过(kprod>=1024), GetAndCheckIndicesDtype 强制 INT64 校验失败 -> GRAPH_FAILED
TEST_F(AdaptiveMaxPool3dExtraTiling, adaptive_max_pool3d_dhw_exceed_int32_indices_int32)
{
    gert::StorageShape xShape = {{1, 1, 1300, 1300, 1300}, {1, 1, 1300, 1300, 1300}};
    gert::StorageShape yShape = {{1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}};
    ExecuteAdaptiveMaxPool3dExtraCase(xShape, yShape, ge::DT_FLOAT16, ge::DT_INT32, MakeAttrs({1, 1, 1}),
                                      ge::GRAPH_FAILED);
}

// attr indices_dtype 非法值(5, 合法集合为 3=INT32/9=INT64): para IsCapable 通过,
// DoOpTiling 的 GetAndCheckIndicesDtype attr 校验失败 -> GRAPH_FAILED
TEST_F(AdaptiveMaxPool3dExtraTiling, adaptive_max_pool3d_para_invalid_indices_dtype_attr)
{
    gert::StorageShape xShape = {{1, 64, 4, 4, 8}, {1, 64, 4, 4, 8}};
    gert::StorageShape yShape = {{1, 64, 2, 2, 2}, {1, 64, 2, 2, 2}};
    ExecuteAdaptiveMaxPool3dExtraCase(xShape, yShape, ge::DT_FLOAT16, ge::DT_INT32, MakeAttrs({2, 2, 2}, true, 5),
                                      ge::GRAPH_FAILED);
}

// ===================== Base GetShapeAttrsInfo 未覆盖错误分支 =====================

// 缺失 output_size attr(GetAttrPointer(0) 返回 nullptr): base GetShapeAttrsInfo 的
// OP_CHECK_NULL(outputSizePtr) 失败 -> GRAPH_FAILED
TEST_F(AdaptiveMaxPool3dExtraTiling, adaptive_max_pool3d_missing_output_size_attr)
{
    gert::StorageShape xShape = {{1, 64, 4, 4, 8}, {1, 64, 4, 4, 8}};
    gert::StorageShape yShape = {{1, 64, 2, 2, 2}, {1, 64, 2, 2, 2}};
    ExecuteAdaptiveMaxPool3dExtraCase(xShape, yShape, ge::DT_FLOAT16, ge::DT_INT32,
                                      std::vector<std::pair<std::string, Ops::NN::AnyValue>>(), ge::GRAPH_FAILED);
}

// 空 tensor(nIn=0): base GetShapeAttrsInfo 的逐维 >=1 校验失败 -> GRAPH_FAILED
TEST_F(AdaptiveMaxPool3dExtraTiling, adaptive_max_pool3d_empty_tensor_zero_dim)
{
    gert::StorageShape xShape = {{0, 64, 4, 4, 8}, {0, 64, 4, 4, 8}};
    gert::StorageShape yShape = {{0, 64, 2, 2, 2}, {0, 64, 2, 2, 2}};
    ExecuteAdaptiveMaxPool3dExtraCase(xShape, yShape, ge::DT_FLOAT16, ge::DT_INT32, MakeAttrs({2, 2, 2}),
                                      ge::GRAPH_FAILED);
}
