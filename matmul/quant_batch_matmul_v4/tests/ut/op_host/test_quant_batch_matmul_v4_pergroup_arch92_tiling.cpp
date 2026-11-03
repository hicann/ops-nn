/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// quant_batch_matmul_v4_pergroup_arch92_tiling.cpp 的 UT。
// 说明：被测类头文件 quant_batch_matmul_v4_pergroup_arch92_tiling.h 的依赖链中，
// quant_batch_matmul_v4_polling_tiling.h 使用了仅算子侧编译目标可解析的 include（依赖 -I matmul），
// UT 编译目标无法包含该头文件链，因此本用例不直接 include 被测头文件，而是：
//   1) 通过 TilingRegistry 以 arch92 优先级直接调度被测 tiling 模板（类的定义与实现均在 tiling obj 中）；
//   2) 辅以一条走算子注册入口（opImpl->tiling）的集成用例，覆盖 npuArch=DAV_9201 到 arch92 的路由。
// 断言语义：RunArch92Tiling 返回 SUCCESS 即 arch92 tiling 成功；分析失败与 IsCapable 拒绝均表现为 GRAPH_FAILED。

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "ut_string_utils.h"

#include "op_host/tiling_templates_registry.h"
#include "../../../op_host/op_tiling/quant_batch_matmul_v4_compile_info.h"
#include "../../../op_kernel/arch35/quant_batch_matmul_v4_tiling_key.h"
#include "common/inc/op_host/op_const_def_nn.h"

using namespace ut_str;

namespace {
// 与 op_tiling/quant_batch_matmul_v4_tiling_registry.cpp 中的 PERGROUP_ARCH92_PRIORITY 保持一致
constexpr int32_t QBMMV4_PERGROUP_ARCH92_PRIORITY = 8;
constexpr uint64_t
    QBMMV4_ARCH92_WORKSPACE_SIZE = 16777216UL; // 16 * 1024 * 1024，见 arch35/quant_batch_matmul_v4_tiling.h

// arch92 pergroup tiling 用例配置：默认值为合法 S8S4(m=256, k=256, n=128, group=128) 场景
struct Arch92UtCfg {
    int64_t m = 256;
    int64_t k = 256;
    int64_t n = 128;
    int64_t groupSize = 128;
    bool transA = false;
    bool transB = false;

    ge::DataType x1Dtype = ge::DT_INT8;
    ge::DataType x2Dtype = ge::DT_INT4;
    ge::DataType cDtype = ge::DT_FLOAT16;
    ge::DataType x1ScaleDtype = ge::DT_FLOAT;
    ge::DataType x2ScaleDtype = ge::DT_UINT64;
    ge::DataType yScaleDtype = ge::DT_UINT64;

    bool hasX1Scale = true;
    bool hasX2Scale = true;
    bool hasYScale = false;

    std::vector<int64_t> x1Dims;       // 为空时取 {m, k}
    std::vector<int64_t> x2Dims;       // 为空时按 transB 取 NZ 布局
    std::vector<int64_t> x2OriginDims; // 为空时与 storage 一致
    std::vector<int64_t> x1ScaleDims;  // 为空时取 {m, 1}
    std::vector<int64_t> x2ScaleDims;  // 为空时取 {k / groupSize, n}
    std::vector<int64_t> yScaleDims;   // 为空时取 {1, n}

    ge::Format x1Format = ge::FORMAT_ND;
    ge::Format x2Format = ge::FORMAT_FRACTAL_NZ;

    bool withGroupSizeAttr = true;

    uint32_t aicNum = 32;
    uint64_t ubSize = 253952;
    uint64_t l1Size = 524288;
    uint64_t l0aSize = 65536;
    uint64_t l0bSize = 65536;
    uint64_t l0cSize = 262144;

    bool withWorkspace = true;
    uint64_t tilingDataCap = 4096;
};

// 构造 Fp8Fp4(T_G) 场景默认配置：x1 FP8 + x2 FP4 + x2Scale BF16 + yScale UINT64 + group 32
Arch92UtCfg MakeFp8Fp4Cfg()
{
    Arch92UtCfg cfg;
    cfg.x1Dtype = ge::DT_FLOAT8_E4M3FN;
    cfg.x2Dtype = ge::DT_FLOAT4_E2M1;
    cfg.cDtype = ge::DT_BF16;
    cfg.x2ScaleDtype = ge::DT_BF16;
    cfg.hasX1Scale = false;
    cfg.hasYScale = true;
    cfg.groupSize = 32;
    cfg.m = 128;
    return cfg;
}

// 持有 faker 上下文，保证用例生命周期内指针有效
class Arch92UtEnv {
public:
    explicit Arch92UtEnv(Arch92UtCfg cfg) : cfg_(std::move(cfg)) { SetUp(); }

    // 只调度 arch92 模板：结果无歧义（不会串到 REG_BASE）
    ge::graphStatus RunArch92Tiling()
    {
        return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(
            Ctx(), std::vector<int32_t>{QBMMV4_PERGROUP_ARCH92_PRIORITY});
    }

    // 走算子注册入口：TilingFunc 依 compileInfo.npuArch=DAV_9201 路由到 arch92 优先级
    ge::graphStatus RunOpEntryTiling()
    {
        auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("QuantBatchMatmulV4");
        if (opImpl == nullptr || opImpl->tiling == nullptr) {
            return ge::GRAPH_FAILED;
        }
        return opImpl->tiling(Ctx());
    }

    gert::TilingContext* Ctx() { return holder_.GetContext<gert::TilingContext>(); }

    // tiling 成功后，从 raw tiling data 解出的落盘结果
    const qbmmv4_tiling::QuantBatchMatmulV4PerGroupTilingDataParams* Td()
    {
        return reinterpret_cast<const qbmmv4_tiling::QuantBatchMatmulV4PerGroupTilingDataParams*>(
            Ctx()->GetRawTilingData()->GetData());
    }

private:
    void SetUp()
    {
        const auto& cfg = cfg_;
        gert::StorageShape x1Shape = MakeStorageShape(cfg.x1Dims.empty() ? std::vector<int64_t>{cfg.m, cfg.k} :
                                                                           cfg.x1Dims);
        std::vector<int64_t> x2Default = cfg.transB ? std::vector<int64_t>{2, 1, cfg.n, cfg.k / 2} :
                                                      std::vector<int64_t>{1, cfg.k / 64, 64, cfg.n};
        gert::StorageShape x2Shape = MakeStorageShape(cfg.x2Dims.empty() ? x2Default : cfg.x2Dims);
        if (!cfg.x2OriginDims.empty()) {
            x2Shape.MutableOriginShape() = ToShape(cfg.x2OriginDims);
        }
        gert::StorageShape x1ScaleShape = MakeStorageShape(cfg.x1ScaleDims.empty() ? std::vector<int64_t>{cfg.m, 1} :
                                                                                     cfg.x1ScaleDims);
        gert::StorageShape x2ScaleShape = MakeStorageShape(
            cfg.x2ScaleDims.empty() ? std::vector<int64_t>{cfg.k / cfg.groupSize, cfg.n} : cfg.x2ScaleDims);
        gert::StorageShape yScaleShape = MakeStorageShape(cfg.yScaleDims.empty() ? std::vector<int64_t>{1, cfg.n} :
                                                                                   cfg.yScaleDims);
        gert::StorageShape outputShape = MakeStorageShape({cfg.m, cfg.n});

        platformInfo_.Init();
        compileInfo_ = {};
        compileInfo_.aicNum = cfg.aicNum;
        compileInfo_.ubSize = cfg.ubSize;
        compileInfo_.l1Size = cfg.l1Size;
        compileInfo_.l0aSize = cfg.l0aSize;
        compileInfo_.l0bSize = cfg.l0bSize;
        compileInfo_.l0cSize = cfg.l0cSize;
        // TilingFunc 按 npuArch 路由：DAV_9201 -> {arch92, reg_base}，arch92 优先
        compileInfo_.socVersion = platform_ascendc::SocVersion::ASCEND950;
        compileInfo_.npuArch = NpuArch::DAV_9201;
        compileInfo_.supportMmadS8S4 = false;

        rawTilingData_ = gert::TilingData::CreateCap(cfg.tilingDataCap);
        if (cfg.withWorkspace) {
            workspaceBuf_ = gert::ContinuousVector::Create<size_t>(64);
        }

        gert::TilingContextFaker faker;
        faker.SetOpType("QuantBatchMatmulV4");
        faker.NodeIoNum(10, 1);
        faker.IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
        std::vector<void*> inputShapes(10, nullptr);
        inputShapes[0] = &x1Shape;
        inputShapes[1] = &x2Shape;
        inputShapes[3] = cfg.hasX1Scale ? &x1ScaleShape : nullptr;
        inputShapes[4] = cfg.hasX2Scale ? &x2ScaleShape : nullptr;
        inputShapes[5] = cfg.hasYScale ? &yScaleShape : nullptr;
        faker.InputShapes(inputShapes);
        faker.NodeInputTd(0, cfg.x1Dtype, cfg.x1Format, cfg.x1Format);
        faker.NodeInputTd(1, cfg.x2Dtype, cfg.x2Format, cfg.x2Format);
        faker.NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
        faker.NodeInputTd(3, cfg.x1ScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND);
        faker.NodeInputTd(4, cfg.x2ScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND);
        faker.NodeInputTd(5, cfg.yScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND);
        faker.NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
        faker.NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
        faker.NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
        faker.NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
        faker.NodeOutputTd(0, cfg.cDtype, ge::FORMAT_ND, ge::FORMAT_ND);
        faker.OutputShapes({&outputShape});
        std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrs = {
            {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(cfg.groupSize)},
            {"compute_type", Ops::NN::AnyValue::CreateFrom<bool>(false)},
            {"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(cfg.transA)},
            {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(cfg.transB)}};
        if (cfg.withGroupSizeAttr) {
            attrs.push_back({"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(cfg.groupSize)});
        }
        faker.NodeAttrs(attrs);
        faker.CompileInfo(&compileInfo_);
        faker.PlatformInfo(reinterpret_cast<char*>(&platformInfo_));
        faker.TilingData(rawTilingData_.get());
        if (cfg.withWorkspace) {
            faker.Workspace(reinterpret_cast<gert::ContinuousVector*>(workspaceBuf_.get()));
        }
        holder_ = faker.Build();
    }

    Arch92UtCfg cfg_;
    fe::PlatFormInfos platformInfo_;
    optiling::QuantBatchMatmulV4CompileInfo compileInfo_ = {};
    std::unique_ptr<uint8_t[]> rawTilingData_;
    std::unique_ptr<uint8_t[]> workspaceBuf_;
    gert::KernelRunContextHolder holder_;
};
} // namespace

// ---------------- arch92 tiling 全流程：S8S4 / vector antiquant / Fp8Fp4 / MX ----------------

TEST(QuantBatchMatmulV4Arch92FlowTest, S8S4BasicSuccess)
{
    Arch92UtEnv env(Arch92UtCfg{});
    ASSERT_EQ(env.RunArch92Tiling(), ge::GRAPH_SUCCESS);
    auto* ctx = env.Ctx();
    EXPECT_EQ(ctx->GetTilingKey(), GET_TPL_TILING_KEY(QBMMV4_NOT_TRANS, QBMMV4_PER_GROUP, QBMMV4_OPTION_ATTR_NONE,
                                                      QBMMV4_IS_WEIGHT_NZ, QBMMV4_BASIS));
    EXPECT_EQ(ctx->GetBlockDim(), 2U); // ceil(256/128) * ceil(128/128)
    EXPECT_EQ(ctx->GetWorkspaceSizes(1)[0], QBMMV4_ARCH92_WORKSPACE_SIZE);
    EXPECT_EQ(ctx->GetRawTilingData()->GetDataSize(),
              sizeof(qbmmv4_tiling::QuantBatchMatmulV4PerGroupTilingDataParams));
    const auto* td = env.Td();
    EXPECT_EQ(td->cubeBlockDimM, 1);
    EXPECT_EQ(td->cubeBlockDimN, 2);
    EXPECT_EQ(td->matmulTiling.M, 256);
    EXPECT_EQ(td->matmulTiling.N, 128);
    EXPECT_EQ(td->matmulTiling.Ka, 256);
    EXPECT_EQ(td->matmulTiling.Kb, 256);
    EXPECT_EQ(td->matmulTiling.baseM, 128);
    EXPECT_EQ(td->matmulTiling.baseN, 128);
    EXPECT_EQ(td->matmulTiling.baseK, 128); // S8S4: baseK = groupSize
    EXPECT_EQ(td->matmulTiling.stepM, 1);
    EXPECT_EQ(td->matmulTiling.stepN, 1);
    EXPECT_EQ(td->matmulTiling.stepKa, 4);
    EXPECT_EQ(td->matmulTiling.stepKb, 4);
    EXPECT_EQ(td->matmulTiling.depthA1, 8);
    EXPECT_EQ(td->matmulTiling.depthB1, 8);
    EXPECT_EQ(td->matmulTiling.dbL0A, 2);
    EXPECT_EQ(td->matmulTiling.dbL0B, 2);
    EXPECT_EQ(td->matmulTiling.dbL0C, 1);
    EXPECT_EQ(td->matmulTiling.singleCoreM, 128);
    EXPECT_EQ(td->matmulTiling.singleCoreN, 128);
    EXPECT_EQ(td->matmulTiling.isBias, 0);
    EXPECT_EQ(td->groupSize, 0U); // groupSize 字段仅在 Fp8Fp4 分支写入
}

TEST(QuantBatchMatmulV4Arch92FlowTest, S8S4VectorAntiquantSuccess)
{
    Arch92UtCfg cfg; // x2Scale 为 FLOAT 时走 vector antiquant 轮询求解分支
    cfg.x2ScaleDtype = ge::DT_FLOAT;
    cfg.cDtype = ge::DT_BF16;
    Arch92UtEnv env(cfg);
    ASSERT_EQ(env.RunArch92Tiling(), ge::GRAPH_SUCCESS);
    auto* ctx = env.Ctx();
    EXPECT_EQ(ctx->GetTilingKey(), GET_TPL_TILING_KEY(QBMMV4_NOT_TRANS, QBMMV4_PER_GROUP, QBMMV4_OPTION_ATTR_NONE,
                                                      QBMMV4_IS_WEIGHT_NZ, QBMMV4_ANTIQUANT_VECTOR));
    EXPECT_GT(ctx->GetBlockDim(), 0U);
    const auto* td = env.Td();
    EXPECT_EQ(td->matmulTiling.M, 256);
    EXPECT_EQ(td->matmulTiling.N, 128);
    EXPECT_EQ(td->matmulTiling.Ka, 256);
    EXPECT_EQ(td->matmulTiling.Kb, 256);
    EXPECT_EQ(td->matmulTiling.baseK, 128); // min(128, groupSize)
    EXPECT_EQ(td->matmulTiling.stepKa, 4);
    EXPECT_EQ(td->matmulTiling.stepKb, 4);
    EXPECT_EQ(td->matmulTiling.depthA1, 8);
    EXPECT_EQ(td->matmulTiling.depthB1, 8);
    EXPECT_EQ(td->matmulTiling.dbL0A, 2);
    EXPECT_EQ(td->matmulTiling.dbL0B, 2);
    EXPECT_EQ(td->matmulTiling.dbL0C, 2);
    EXPECT_GT(td->matmulTiling.baseM, 0);
    EXPECT_GT(td->matmulTiling.baseN, 0);
    EXPECT_EQ(td->matmulTiling.usedCoreNum, static_cast<int32_t>(ctx->GetBlockDim()));
    EXPECT_EQ(static_cast<uint32_t>(td->cubeBlockDimM) * static_cast<uint32_t>(td->cubeBlockDimN), ctx->GetBlockDim());
    EXPECT_EQ(td->enablePreAlloc, 1); // kSize=256 不满足关闭 prealloc 条件
    EXPECT_EQ(ctx->GetWorkspaceSizes(1)[0], QBMMV4_ARCH92_WORKSPACE_SIZE);
}

TEST(QuantBatchMatmulV4Arch92FlowTest, Fp8Fp4BasicSuccess)
{
    Arch92UtCfg cfg = MakeFp8Fp4Cfg();
    cfg.ubSize = 524288;  // 放宽 UB 使 baseK 能扩展到 kSize
    cfg.l1Size = 1048576; // 放宽 L1 使 baseK 能扩展到 kSize
    Arch92UtEnv env(cfg);
    ASSERT_EQ(env.RunArch92Tiling(), ge::GRAPH_SUCCESS);
    auto* ctx = env.Ctx();
    EXPECT_EQ(ctx->GetTilingKey(), GET_TPL_TILING_KEY(QBMMV4_NOT_TRANS, QBMMV4_TCG, QBMMV4_OPTION_ATTR_NONE,
                                                      QBMMV4_IS_WEIGHT_NZ, QBMMV4_BASIS));
    EXPECT_EQ(ctx->GetBlockDim(), 1U);
    const auto* td = env.Td();
    EXPECT_EQ(td->cubeBlockDimM, 1);
    EXPECT_EQ(td->cubeBlockDimN, 1);
    EXPECT_EQ(td->groupSize, 32U);
    EXPECT_EQ(td->matmulTiling.M, 128);
    EXPECT_EQ(td->matmulTiling.N, 128);
    EXPECT_EQ(td->matmulTiling.Ka, 256);
    EXPECT_EQ(td->matmulTiling.Kb, 256);
    EXPECT_EQ(td->matmulTiling.baseM, 128);
    EXPECT_EQ(td->matmulTiling.baseN, 128);
    EXPECT_EQ(td->matmulTiling.baseK, 256); // L1/L0A/L0B/UB 均不限制，baseK 扩到 kSize
    EXPECT_EQ(td->matmulTiling.dbL0C, 1);
    EXPECT_EQ(ctx->GetWorkspaceSizes(1)[0], QBMMV4_ARCH92_WORKSPACE_SIZE);
}

TEST(QuantBatchMatmulV4Arch92FlowTest, Fp8Fp4BaseKCappedByMaxBaseKSize)
{
    Arch92UtCfg cfg = MakeFp8Fp4Cfg();
    cfg.k = 2048;
    cfg.x2Dims = {1, 32, 64, 128};
    cfg.ubSize = 2097152;
    cfg.l1Size = 3145728;
    cfg.l0aSize = 262144;
    cfg.l0bSize = 131072;
    Arch92UtEnv env(cfg);
    ASSERT_EQ(env.RunArch92Tiling(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(env.Td()->matmulTiling.baseK, 1024); // MAX_BASE_K_SIZE 截断
    EXPECT_EQ(env.Td()->matmulTiling.Ka, 2048);
    EXPECT_EQ(env.Td()->matmulTiling.Kb, 2048);
}

// GetFp8Fp4BaseK 四个资源上限 break 分支 + 首轮即 break 分支，均通过完整 tiling 流程覆盖

TEST(QuantBatchMatmulV4Arch92FlowTest, Fp8Fp4BaseKBreakByL1Size)
{
    Arch92UtCfg cfg = MakeFp8Fp4Cfg();
    cfg.k = 1024;
    cfg.x2Dims = {1, 16, 64, 128};
    cfg.l1Size = 150000; // K=64 通过(133120)，K=96 溢出(198656)
    Arch92UtEnv env(cfg);
    ASSERT_EQ(env.RunArch92Tiling(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(env.Td()->matmulTiling.baseK, 64);
}

TEST(QuantBatchMatmulV4Arch92FlowTest, Fp8Fp4BaseKBreakByL0aSize)
{
    Arch92UtCfg cfg = MakeFp8Fp4Cfg();
    cfg.k = 1024;
    cfg.x2Dims = {1, 16, 64, 128};
    cfg.l1Size = 1048576;
    cfg.l0aSize = 20000; // K=64 通过(16384)，K=96 溢出(24576)
    Arch92UtEnv env(cfg);
    ASSERT_EQ(env.RunArch92Tiling(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(env.Td()->matmulTiling.baseK, 64);
}

TEST(QuantBatchMatmulV4Arch92FlowTest, Fp8Fp4BaseKBreakByL0bSize)
{
    Arch92UtCfg cfg = MakeFp8Fp4Cfg();
    cfg.k = 1024;
    cfg.x2Dims = {1, 16, 64, 128};
    cfg.l1Size = 1048576;
    cfg.l0bSize = 10000; // K=64 通过(8192)，K=96 溢出(12288)
    Arch92UtEnv env(cfg);
    ASSERT_EQ(env.RunArch92Tiling(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(env.Td()->matmulTiling.baseK, 64);
}

TEST(QuantBatchMatmulV4Arch92FlowTest, Fp8Fp4BaseKBreakByUbSize)
{
    Arch92UtCfg cfg = MakeFp8Fp4Cfg();
    cfg.k = 1024;
    cfg.x2Dims = {1, 16, 64, 128};
    cfg.l1Size = 1048576;
    cfg.ubSize = 120000; // K=64 通过(102432)，K=96 溢出(153632)
    Arch92UtEnv env(cfg);
    ASSERT_EQ(env.RunArch92Tiling(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(env.Td()->matmulTiling.baseK, 64);
}

TEST(QuantBatchMatmulV4Arch92FlowTest, Fp8Fp4BaseKBreakAtFirstIteration)
{
    Arch92UtCfg cfg = MakeFp8Fp4Cfg();
    cfg.k = 1024;
    cfg.x2Dims = {1, 16, 64, 128};
    cfg.l1Size = 65536; // 首轮 K=32 即溢出(67520)，baseK 保持 groupSize
    Arch92UtEnv env(cfg);
    ASSERT_EQ(env.RunArch92Tiling(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(env.Td()->matmulTiling.baseK, 32);
    EXPECT_EQ(env.Td()->groupSize, 32U);
}

TEST(QuantBatchMatmulV4Arch92FlowTest, Fp8Fp4YScaleOneDimSuccess)
{
    Arch92UtCfg cfg = MakeFp8Fp4Cfg();
    cfg.ubSize = 524288;
    cfg.l1Size = 1048576;
    cfg.yScaleDims = {128}; // [n] 一维 yScale 合法
    Arch92UtEnv env(cfg);
    ASSERT_EQ(env.RunArch92Tiling(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(env.Ctx()->GetTilingKey(), GET_TPL_TILING_KEY(QBMMV4_NOT_TRANS, QBMMV4_TCG, QBMMV4_OPTION_ATTR_NONE,
                                                            QBMMV4_IS_WEIGHT_NZ, QBMMV4_BASIS));
}

// MX(x1Scale E8M0)场景：AnalyzeContext 将 isFp8Fp4_ 置 false，IsCapable 走
// "!isS8S4_ && !isFp8Fp4_" 分支返回不支持；同时覆盖 AnalyzeInputs 的 8 字节对齐路径
TEST(QuantBatchMatmulV4Arch92FlowTest, MxScaleNotCapable)
{
    Arch92UtCfg cfg = MakeFp8Fp4Cfg();
    cfg.hasX1Scale = true;
    cfg.x1ScaleDtype = ge::DT_FLOAT8_E8M0;
    cfg.k = 200; // 200 % 8 == 0，但 200 % 64 != 0，验证走 MX 8 字节对齐
    cfg.x1Dims = {256, 200};
    cfg.x2Dims = {1, 25, 8, 128};
    cfg.x2ScaleDims = {6, 128};
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED); // IsCapable 拒绝 -> PARAM_INVALID -> 无可用模板
}

TEST(QuantBatchMatmulV4Arch92FlowTest, MxScaleWithTransBUsesOriginShape)
{
    Arch92UtCfg cfg = MakeFp8Fp4Cfg();
    cfg.hasX1Scale = true;
    cfg.x1ScaleDtype = ge::DT_FLOAT8_E8M0;
    cfg.transB = true; // MX + transB：kSize 取 x1Outer，nSize 取 x2 原始 shape dim0
    cfg.x2Dims = {2, 1, 128, 128};
    cfg.x2OriginDims = {128, 256};
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92FlowTest, Hif4MxScaleNotCapable)
{
    Arch92UtCfg cfg = MakeFp8Fp4Cfg();
    cfg.x2Dtype = Ops::NN::DT_HIFLOAT4; // HIF4 也命中 isMxFp8MxFp4OrHif4，走 8 字节对齐
    cfg.hasX1Scale = true;
    cfg.x1ScaleDtype = ge::DT_FLOAT8_E8M0;
    cfg.k = 200;
    cfg.x1Dims = {256, 200};
    cfg.x2Dims = {1, 25, 8, 128};
    cfg.x2ScaleDims = {6, 128};
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

// ---------------- IsCapable 分支覆盖（拒绝场景经注册入口表现为 GRAPH_FAILED） ----------------

TEST(QuantBatchMatmulV4Arch92IsCapableTest, RejectInvalidADtype)
{
    Arch92UtCfg cfg;
    cfg.x1Dtype = ge::DT_FLOAT16;
    cfg.hasX1Scale = false;
    cfg.x2ScaleDtype = ge::DT_BF16;
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92IsCapableTest, RejectInt8WithNonInt4Weight)
{
    Arch92UtCfg cfg;
    cfg.x2Dtype = ge::DT_INT8;
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92IsCapableTest, RejectFp8WithNonFp4Weight)
{
    Arch92UtCfg cfg = MakeFp8Fp4Cfg();
    cfg.x2Dtype = ge::DT_FLOAT8_E4M3FN;
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92IsCapableTest, S8S4RejectX1ScaleDtype)
{
    Arch92UtCfg cfg;
    cfg.x1ScaleDtype = ge::DT_FLOAT16;
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92IsCapableTest, S8S4RejectX2ScaleDtype)
{
    Arch92UtCfg cfg;
    cfg.x2ScaleDtype = ge::DT_BF16;
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92IsCapableTest, S8S4RejectCDtype)
{
    Arch92UtCfg cfg;
    cfg.cDtype = ge::DT_INT8;
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92IsCapableTest, S8S4RejectUnsupportedGroupSize)
{
    Arch92UtCfg cfg;
    cfg.groupSize = 32; // A8W4 仅支持 64/128
    cfg.x2ScaleDims = {8, 128};
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92IsCapableTest, S8S4RejectX1ScaleOuterDim)
{
    Arch92UtCfg cfg;
    cfg.x1ScaleDims = {320, 1}; // outer != mSize(256)
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92IsCapableTest, S8S4RejectX1ScaleInnerDim)
{
    Arch92UtCfg cfg;
    cfg.x1ScaleDims = {256, 2}; // inner != 1
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92IsCapableTest, Fp8Fp4RejectX2ScaleDtype)
{
    Arch92UtCfg cfg = MakeFp8Fp4Cfg();
    cfg.x2ScaleDtype = ge::DT_FLOAT;
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92IsCapableTest, Fp8Fp4RejectCDtype)
{
    Arch92UtCfg cfg = MakeFp8Fp4Cfg();
    cfg.cDtype = ge::DT_FLOAT16;
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92IsCapableTest, Fp8Fp4RejectGroupSize)
{
    Arch92UtCfg cfg = MakeFp8Fp4Cfg();
    cfg.groupSize = 64; // Fp8Fp4 仅支持 32
    cfg.x2ScaleDims = {4, 128};
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92IsCapableTest, RejectKNotDivisibleByGroup)
{
    Arch92UtCfg cfg;
    cfg.k = 192; // 192 % 128 != 0
    cfg.x1Dims = {256, 192};
    cfg.x2Dims = {1, 3, 64, 128};
    cfg.x2ScaleDims = {1, 128};
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92IsCapableTest, RejectTransB)
{
    Arch92UtCfg cfg;
    cfg.transB = true;
    cfg.x2Dims = {2, 1, 128, 128}; // K1=2*K0=128 = kSize(256)
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92IsCapableTest, RejectX2ScaleOuterDim)
{
    Arch92UtCfg cfg;
    cfg.x2ScaleDims = {3, 128}; // 3 * 128 != kSize(256)
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92IsCapableTest, RejectX2ScaleInnerDim)
{
    Arch92UtCfg cfg;
    cfg.x2ScaleDims = {2, 64}; // inner != nSize(128)
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92IsCapableTest, AcceptS8S4)
{
    Arch92UtEnv env(Arch92UtCfg{});
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(env.Ctx()->GetTilingKey(), GET_TPL_TILING_KEY(QBMMV4_NOT_TRANS, QBMMV4_PER_GROUP, QBMMV4_OPTION_ATTR_NONE,
                                                            QBMMV4_IS_WEIGHT_NZ, QBMMV4_BASIS));
}

TEST(QuantBatchMatmulV4Arch92IsCapableTest, AcceptFp8Fp4)
{
    Arch92UtEnv env(MakeFp8Fp4Cfg());
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(env.Ctx()->GetTilingKey(), GET_TPL_TILING_KEY(QBMMV4_NOT_TRANS, QBMMV4_TCG, QBMMV4_OPTION_ATTR_NONE,
                                                            QBMMV4_IS_WEIGHT_NZ, QBMMV4_BASIS));
}

// ---------------- GetShapeAttrsInfo 失败分支（分析失败直接返回 GRAPH_FAILED） ----------------

TEST(QuantBatchMatmulV4Arch92ShapeGuardTest, FailWhenX2ScaleAbsent)
{
    Arch92UtCfg cfg;
    cfg.hasX2Scale = false;
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92ShapeGuardTest, FailWhenTransATrue)
{
    Arch92UtCfg cfg;
    cfg.transA = true;
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92ShapeGuardTest, FailWhenX1Not2D)
{
    Arch92UtCfg cfg;
    cfg.x1Dims = {2, 256, 256};
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92ShapeGuardTest, FailWhenX2Not4D)
{
    Arch92UtCfg cfg;
    cfg.x2Dims = {256, 128};
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92ShapeGuardTest, FailWhenGroupSizeAttrMissing)
{
    Arch92UtCfg cfg;
    cfg.withGroupSizeAttr = false;
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92ShapeGuardTest, FailWhenFp8Fp4X1ScaleNotNull)
{
    Arch92UtCfg cfg = MakeFp8Fp4Cfg();
    cfg.hasX1Scale = true; // 非 MX 量化时 x1Scale 必须为空
    cfg.x1ScaleDtype = ge::DT_FLOAT;
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92ShapeGuardTest, FailWhenS8S4X1ScaleAbsent)
{
    Arch92UtCfg cfg;
    cfg.hasX1Scale = false;
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92ShapeGuardTest, FailWhenX1FormatNotNd)
{
    Arch92UtCfg cfg;
    cfg.x1Format = ge::FORMAT_NCHW;
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92ShapeGuardTest, FailWhenX2FormatNotFractalNz)
{
    Arch92UtCfg cfg;
    cfg.x2Format = ge::FORMAT_ND;
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92ShapeGuardTest, FailWhenKNotAlignedTo64)
{
    Arch92UtCfg cfg;
    cfg.k = 200; // 200 % 64 != 0
    cfg.x1Dims = {256, 200};
    cfg.x2Dims = {1, 25, 8, 128};
    cfg.x2ScaleDims = {1, 128};
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92ShapeGuardTest, FailWhenNNotAlignedTo64)
{
    Arch92UtCfg cfg;
    cfg.n = 8; // 8 % 64 != 0
    cfg.x2Dims = {1, 4, 64, 8};
    cfg.x2ScaleDims = {2, 8};
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92ShapeGuardTest, FailWhenKDimMismatchNoTransB)
{
    Arch92UtCfg cfg;
    cfg.x2Dims = {1, 2, 64, 128}; // kSize = 128 != x1Outer(256)
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92ShapeGuardTest, FailWhenKDimMismatchTransB)
{
    Arch92UtCfg cfg;
    cfg.transB = true;
    cfg.x2Dims = {1, 1, 128, 64}; // kSize = 1 * 64 = 64 != x1Outer(256)
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92ShapeGuardTest, FailWhenYScaleAbsent)
{
    Arch92UtCfg cfg = MakeFp8Fp4Cfg();
    cfg.hasYScale = false;
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92ShapeGuardTest, FailWhenYScaleDtypeInvalid)
{
    Arch92UtCfg cfg = MakeFp8Fp4Cfg();
    cfg.yScaleDtype = ge::DT_FLOAT;
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92ShapeGuardTest, FailWhenYScaleShapeSizeZero)
{
    Arch92UtCfg cfg = MakeFp8Fp4Cfg();
    cfg.yScaleDims = {1, 0};
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92ShapeGuardTest, FailWhenYScaleThreeDims)
{
    Arch92UtCfg cfg = MakeFp8Fp4Cfg();
    cfg.yScaleDims = {1, 1, 128};
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92ShapeGuardTest, FailWhenYScaleTwoDimsWithLeadingNotOne)
{
    Arch92UtCfg cfg = MakeFp8Fp4Cfg();
    cfg.yScaleDims = {2, 128};
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92ShapeGuardTest, FailWhenX1ScaleNot2D)
{
    Arch92UtCfg cfg;
    cfg.x1ScaleDims = {256};
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92ShapeGuardTest, FailWhenX2ScaleNot2D)
{
    Arch92UtCfg cfg;
    cfg.x2ScaleDims = {128};
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92ShapeGuardTest, FailWhenX2ScaleNot2DFp8Fp4)
{
    Arch92UtCfg cfg = MakeFp8Fp4Cfg();
    cfg.x2ScaleDims = {128};
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

// ---------------- 其它分支 ----------------

TEST(QuantBatchMatmulV4Arch92InternalTest, DoOpTilingFailsWhenTilingDataCapacitySmall)
{
    Arch92UtCfg cfg;
    cfg.tilingDataCap = 16; // 小于 sizeof(QuantBatchMatmulV4PerGroupTilingDataParams)
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

TEST(QuantBatchMatmulV4Arch92InternalTest, GetWorkspaceSizeFailsWithoutWorkspace)
{
    Arch92UtCfg cfg;
    cfg.withWorkspace = false;
    Arch92UtEnv env(cfg);
    EXPECT_EQ(env.RunArch92Tiling(), ge::GRAPH_FAILED);
}

// 集成用例：走算子注册入口，验证 TilingFunc 依 compileInfo.npuArch=DAV_9201 路由到 arch92 模板
TEST(QuantBatchMatmulV4Arch92InternalTest, RouteToArch92ViaOpEntry)
{
    Arch92UtEnv env(Arch92UtCfg{});
    ASSERT_EQ(env.RunOpEntryTiling(), ge::GRAPH_SUCCESS);
    auto* ctx = env.Ctx();
    EXPECT_EQ(ctx->GetTilingKey(), GET_TPL_TILING_KEY(QBMMV4_NOT_TRANS, QBMMV4_PER_GROUP, QBMMV4_OPTION_ATTR_NONE,
                                                      QBMMV4_IS_WEIGHT_NZ, QBMMV4_BASIS));
    EXPECT_EQ(ctx->GetBlockDim(), 2U);
    EXPECT_EQ(ctx->GetWorkspaceSizes(1)[0], QBMMV4_ARCH92_WORKSPACE_SIZE);
    EXPECT_EQ(env.Td()->matmulTiling.baseK, 128);
}
