/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * \file test_non_zero_tiling.cpp
 * \brief NonZero arch22(DAV_2201/ascend910b,ascend910_93) host tiling UT
 *
 * 覆盖核心路径：
 *  - 形状：rank1/rank2/rank4/rank8 的 numel 归一化与 shape 下发（stride 由 kernel 现场按 shape 计算，不再下发）
 *  - 切核：numel<8 → 单核（numPerCore8=0, tailNum=numel）；numel=1000 → 64 核（numPerCore8=8, tailNum=496）
 *  - dtype：BOOL/INT8/UINT8 → xElemSize=1；FLOAT16 → xElemSize=2；BF16 → xElemSize=-2（符号位标记）；FLOAT →
 * xElemSize=4
 *  - transpose 属性 0/1 → tilingData.transpose
 *  - workspace：AlignUp(realCoreNum*8*4, 64) + 系统 workspace；BlockDim=realCoreNum
 *  - 失败分支：coreNum<=0、ubSize<=0、rank==0（空 shape）、rank>8、transpose 属性缺失
 */

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "base/err_mgr.h"
#include "ut_op_util.h"
#include "test_cube_util.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "index/non_zero/op_host/arch22/non_zero_tiling.h"

using namespace ge;
using namespace ut_util;

namespace {
constexpr int64_t DEFAULT_CORE_NUM = 64;
constexpr int64_t DEFAULT_UB_SIZE = 253952; // 248 KB
constexpr int64_t MAX_DIM_NUM = 8;
constexpr int64_t ELEM_BYTE = 4;        // kernel 侧 count 缓冲按 int32 计数
constexpr int64_t WORKSPACE_ALIGN = 64; // 用户 workspace 按 64B 对齐
constexpr size_t TILING_DATA_CAP = 4096;
constexpr size_t WORKSPACE_SLOT_CAP = 64 * 8;
const char* const OP_TYPE = "NonZero";

int64_t AlignUp(int64_t value, int64_t align) { return ((value + align - 1) / align) * align; }

// NonZeroTilingData 由 TILING_DATA_FIELD_DEF(_ARR) 生成：字段均为 int64_t 且按声明顺序连续排布
// （data_size_ 始终为 8 的倍数，无对齐填充）。kernel 侧 GET_TILING_DATA 生成的同名结构体即按此布局
// 以公有成员直接读取，因此这里用同布局的 POD 镜像解析原始 tiling buffer，并以 GetDataSize() 做布局校验。
struct NonZeroTilingLayout {
    int64_t numel;
    int64_t rank;
    int64_t realCoreNum;
    int64_t numPerCore8;
    int64_t tailNum;
    int64_t transpose;
    int64_t xElemSize;
    int64_t shape[MAX_DIM_NUM];
};

static_assert(sizeof(NonZeroTilingLayout) == (7 + static_cast<int64_t>(MAX_DIM_NUM)) * sizeof(int64_t),
              "NonZero arch22 tiling data must be 15 continuous int64_t fields (<=128B JIT tiling budget)");

// StorageShape 仅支持 initializer_list 构造，这里从动态 shape 向量构建（origin == storage，ND 连续布局）
gert::StorageShape BuildStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape ss;
    ss.MutableOriginShape().SetDimNum(dims.size());
    ss.MutableStorageShape().SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        ss.MutableOriginShape().SetDim(i, dims[i]);
        ss.MutableStorageShape().SetDim(i, dims[i]);
    }
    return ss;
}

void InitPlatForm(fe::PlatFormInfos& platformInfo, std::map<std::string, std::string>& socInfos,
                  std::map<std::string, std::string>& aicoreSpec, std::map<std::string, std::string>& intrinsics)
{
    std::string compileInfoString = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    GetPlatFormInfos(compileInfoString.c_str(), socInfos, aicoreSpec, intrinsics);
    platformInfo.Init();
}

// 构造 tiling 上下文并执行已注册的 arch22 host tiling 函数。
// 说明：gtest 的 ASSERT_* 在构造函数中会展开为 "return 值" 导致编译失败，故构造期不做断言，
//       tiling 执行结果统一由用例内的 ASSERT/EXPECT 校验。
class TilingRunner {
public:
    TilingRunner(const std::vector<int64_t>& dims, ge::DataType dtype, bool transpose,
                 int64_t coreNum = DEFAULT_CORE_NUM, int64_t ubSize = DEFAULT_UB_SIZE, bool withTransposeAttr = true,
                 ge::DataType outputDtype = ge::DT_INT64)
    {
        compileInfo_.coreNum = coreNum;
        compileInfo_.ubSize = ubSize;

        gert::StorageShape xShape = BuildStorageShape(dims);
        gert::StorageShape yShape({}, {}); // 输出 shape 依赖计算，此处仅占位

        InitPlatForm(platformInfo_, socInfos_, aicoreSpec_, intrinsics_);

        param_ = gert::TilingData::CreateCap(TILING_DATA_CAP);
        workspaceHolder_ = gert::ContinuousVector::Create<size_t>(WORKSPACE_SLOT_CAP);
        auto* workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder_.get());

        std::vector<std::pair<std::string, Ops::NN::AnyValue>> attrs;
        if (withTransposeAttr) {
            attrs.emplace_back("transpose", Ops::NN::AnyValue::CreateFrom<bool>(transpose));
        }

        holder_ = gert::TilingContextFaker()
                      .SetOpType(OP_TYPE)
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo_)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo_))
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(attrs)
                      .TilingData(param_.get())
                      .Workspace(workspace)
                      .Build();
        context_ = holder_.GetContext<gert::TilingContext>();
        if (context_ == nullptr || context_->GetPlatformInfo() == nullptr) {
            return;
        }
        context_->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos_);
        context_->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec_);
        context_->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics_);
        context_->GetPlatformInfo()->SetCoreNumByCoreType("AICore");

        // 构造函数中不可使用 gtest 断言宏，tilingFunc 存在性由用例侧校验
        auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(OP_TYPE);
        if (opImpl != nullptr) {
            tilingFunc_ = opImpl->tiling;
        }
        if (tilingFunc_ != nullptr) {
            status_ = tilingFunc_(context_);
        }
    }

    ge::graphStatus Status() const { return status_; }
    gert::TilingContext* Context() const { return context_; }
    int32_t BlockDim() const { return context_->GetBlockDim(); }
    size_t WorkspaceSize() const { return context_->GetWorkspaceSizes(1)[0]; }

    // 原始 tiling buffer 内字段解析（调用前需保证 tiling 成功）
    const NonZeroTilingLayout* Data() const
    {
        return reinterpret_cast<const NonZeroTilingLayout*>(context_->GetRawTilingData()->GetData());
    }
    size_t DataSize() const { return context_->GetRawTilingData()->GetDataSize(); }

private:
    fe::PlatFormInfos platformInfo_;
    std::map<std::string, std::string> socInfos_;
    std::map<std::string, std::string> aicoreSpec_;
    std::map<std::string, std::string> intrinsics_;
    optiling::NonZeroCompileInfo compileInfo_;
    std::unique_ptr<uint8_t[]> param_;
    std::unique_ptr<uint8_t[]> workspaceHolder_;
    gert::KernelRunContextHolder holder_;
    gert::TilingContext* context_ = nullptr;
    std::function<ge::graphStatus(gert::TilingContext*)> tilingFunc_;
    ge::graphStatus status_ = ge::GRAPH_FAILED;
};
void ExpectSingleError(const char* code, std::initializer_list<const char*> expectedParts)
{
    const auto errors = error_message::GetErrMgrRawErrorMessages();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_STREQ(errors[0].error_id.get(), code);
    ASSERT_NE(errors[0].error_message.get(), nullptr);
    const std::string message(errors[0].error_message.get());
    for (const char* part : expectedParts) {
        EXPECT_NE(message.find(part), std::string::npos) << message;
    }
    if (std::string(code) == "EZ1008") {
        for (const char* detail : {"compile info", "AIV core count", "UB size", "transpose"}) {
            EXPECT_EQ(message.find(detail), std::string::npos) << message;
        }
    }
}
} // namespace

class NonZeroTilingArch22Test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "NonZeroTilingArch22Test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "NonZeroTilingArch22Test TearDown" << std::endl; }

    // tiling 成功后统一的布局/上下文前置校验
    static void ExpectTilingOk(const TilingRunner& r)
    {
        EXPECT_EQ(r.Status(), ge::GRAPH_SUCCESS);
        ASSERT_NE(r.Context(), nullptr);
        ASSERT_NE(r.Context()->GetRawTilingData(), nullptr);
        // 字段顺序/数量与镜像布局一致（GetDataSize 由真实 TilingData 类计算）
        EXPECT_EQ(r.DataSize(), sizeof(NonZeroTilingLayout));
    }
};

// ---- rank1 正常路径：numel=1000 均分到 64 核 ---------------------------------------------------
TEST_F(NonZeroTilingArch22Test, NonZeroTilingArch22_rank1_numel1000_fp32)
{
    TilingRunner r({1000}, ge::DT_FLOAT, /*transpose=*/false);
    ExpectTilingOk(r);

    const auto* td = r.Data();
    EXPECT_EQ(td->numel, 1000);
    EXPECT_EQ(td->rank, 1);
    EXPECT_EQ(td->realCoreNum, 64); // min(64, max(1000/8, 1)) = 64
    EXPECT_EQ(td->numPerCore8, 8);  // (1000 / (8*64)) * 8 = 8
    EXPECT_EQ(td->tailNum, 496);    // 1000 - 63*8 = 496（尾核）
    EXPECT_EQ(td->transpose, 0);
    EXPECT_EQ(td->xElemSize, 4); // DT_FLOAT
    EXPECT_EQ(td->shape[0], 1000);
    for (int64_t d = 1; d < MAX_DIM_NUM; ++d) {
        EXPECT_EQ(td->shape[d], 1);
    }
    EXPECT_EQ(r.BlockDim(), 64);
    // workspace = 系统 workspace + AlignUp(realCoreNum*8*4, 64)（系统部分依赖平台配置，此处只校验下界）
    EXPECT_GE(r.WorkspaceSize(), static_cast<size_t>(AlignUp(64 * 8 * ELEM_BYTE, WORKSPACE_ALIGN)));
}

// ---- numel < 8：收敛为单核（numPerCore8=0, tailNum=numel） ---------------------------------------
TEST_F(NonZeroTilingArch22Test, NonZeroTilingArch22_numel_lt8_single_core)
{
    TilingRunner r({5}, ge::DT_FLOAT, /*transpose=*/false);
    ExpectTilingOk(r);

    const auto* td = r.Data();
    EXPECT_EQ(td->numel, 5);
    EXPECT_EQ(td->rank, 1);
    EXPECT_EQ(td->realCoreNum, 1); // max(5/8, 1) = 1
    EXPECT_EQ(td->numPerCore8, 0);
    EXPECT_EQ(td->tailNum, 5);
    EXPECT_EQ(r.BlockDim(), 1);
    EXPECT_GE(r.WorkspaceSize(), static_cast<size_t>(AlignUp(1 * 8 * ELEM_BYTE, WORKSPACE_ALIGN)));
}

// ---- numel 恰为 8：仍收敛为单核（numel/8 == 1） -------------------------------------------------
TEST_F(NonZeroTilingArch22Test, NonZeroTilingArch22_numel_eq8_single_core)
{
    TilingRunner r({8}, ge::DT_FLOAT, /*transpose=*/false);
    ExpectTilingOk(r);

    const auto* td = r.Data();
    EXPECT_EQ(td->numel, 8);
    EXPECT_EQ(td->realCoreNum, 1);
    EXPECT_EQ(td->numPerCore8, 0);
    EXPECT_EQ(td->tailNum, 8);
    EXPECT_EQ(r.BlockDim(), 1);
}

// ---- rank2：numel=6 → 单核；stride 反解 shape[1] ----------------------------------------------
TEST_F(NonZeroTilingArch22Test, NonZeroTilingArch22_rank2_numel_lt8_stride)
{
    TilingRunner r({2, 3}, ge::DT_FLOAT, /*transpose=*/false);
    ExpectTilingOk(r);

    const auto* td = r.Data();
    EXPECT_EQ(td->numel, 6);
    EXPECT_EQ(td->rank, 2);
    EXPECT_EQ(td->realCoreNum, 1);
    EXPECT_EQ(td->numPerCore8, 0);
    EXPECT_EQ(td->tailNum, 6);
    EXPECT_EQ(td->shape[0], 2);
    EXPECT_EQ(td->shape[1], 3);
    for (int64_t d = 2; d < MAX_DIM_NUM; ++d) {
        EXPECT_EQ(td->shape[d], 1);
    }
}

// ---- rank4：numel=120 → 15 核，前 14 核各 8，尾核 8 -------------------------------------------
TEST_F(NonZeroTilingArch22Test, NonZeroTilingArch22_rank4_numel120_stride)
{
    TilingRunner r({2, 3, 4, 5}, ge::DT_FLOAT, /*transpose=*/false);
    ExpectTilingOk(r);

    const auto* td = r.Data();
    EXPECT_EQ(td->numel, 120);
    EXPECT_EQ(td->rank, 4);
    EXPECT_EQ(td->realCoreNum, 15); // min(64, 120/8) = 15
    EXPECT_EQ(td->numPerCore8, 8);  // (120 / (8*15)) * 8 = 8
    EXPECT_EQ(td->tailNum, 8);      // 120 - 14*8 = 8
    EXPECT_EQ(td->shape[0], 2);
    EXPECT_EQ(td->shape[1], 3);
    EXPECT_EQ(td->shape[2], 4);
    EXPECT_EQ(td->shape[3], 5);
    EXPECT_EQ(r.BlockDim(), 15);
    EXPECT_GE(r.WorkspaceSize(), static_cast<size_t>(AlignUp(15 * 8 * ELEM_BYTE, WORKSPACE_ALIGN)));
}

// ---- workspace：用户部分随 realCoreNum 变化（系统 workspace 相同，故差值即用户部分差值）----------
TEST_F(NonZeroTilingArch22Test, NonZeroTilingArch22_workspace_scales_with_real_core_num)
{
    TilingRunner many({1000}, ge::DT_FLOAT, /*transpose=*/false); // realCoreNum=64
    TilingRunner single({5}, ge::DT_FLOAT, /*transpose=*/false);  // realCoreNum=1
    ExpectTilingOk(many);
    ExpectTilingOk(single);

    const int64_t manyUser = AlignUp(64 * 8 * ELEM_BYTE, WORKSPACE_ALIGN);
    const int64_t singleUser = AlignUp(1 * 8 * ELEM_BYTE, WORKSPACE_ALIGN);
    EXPECT_EQ(many.WorkspaceSize() - single.WorkspaceSize(), static_cast<size_t>(manyUser - singleUser));
}

// ---- rank8：多维 stride 全量反解 ---------------------------------------------------------------
TEST_F(NonZeroTilingArch22Test, NonZeroTilingArch22_rank8_stride_full)
{
    const std::vector<int64_t> dims = {2, 3, 4, 5, 1, 2, 3, 4};
    TilingRunner r(dims, ge::DT_FLOAT, /*transpose=*/false);
    ExpectTilingOk(r);

    const auto* td = r.Data();
    EXPECT_EQ(td->numel, 2880);
    EXPECT_EQ(td->rank, 8);
    EXPECT_EQ(td->realCoreNum, 64);
    EXPECT_EQ(td->numPerCore8, 40); // (2880 / (8*64)) * 8 = 40
    EXPECT_EQ(td->tailNum, 360);    // 2880 - 63*40 = 360
    for (int64_t d = 0; d < MAX_DIM_NUM; ++d) {
        EXPECT_EQ(td->shape[d], dims[d]);
    }
}

// ---- dtype 分支：xElemSize（bf16 以 -2 编码） ----------------------------------------------------
TEST_F(NonZeroTilingArch22Test, NonZeroTilingArch22_dtype_bool)
{
    TilingRunner r({64}, ge::DT_BOOL, /*transpose=*/false);
    ExpectTilingOk(r);
    const auto* td = r.Data();
    EXPECT_EQ(td->xElemSize, 1);
    EXPECT_EQ(td->numel, 64);
    EXPECT_EQ(td->realCoreNum, 8); // min(64, 64/8) = 8
    EXPECT_EQ(td->numPerCore8, 8);
    EXPECT_EQ(td->tailNum, 8);
}

TEST_F(NonZeroTilingArch22Test, NonZeroTilingArch22_dtype_int8)
{
    TilingRunner r({64}, ge::DT_INT8, /*transpose=*/false);
    ExpectTilingOk(r);
    const auto* td = r.Data();
    EXPECT_EQ(td->xElemSize, 1);
}

TEST_F(NonZeroTilingArch22Test, NonZeroTilingArch22_dtype_uint8)
{
    TilingRunner r({64}, ge::DT_UINT8, /*transpose=*/false);
    ExpectTilingOk(r);
    const auto* td = r.Data();
    EXPECT_EQ(td->xElemSize, 1);
}

TEST_F(NonZeroTilingArch22Test, NonZeroTilingArch22_dtype_fp16)
{
    TilingRunner r({64}, ge::DT_FLOAT16, /*transpose=*/false);
    ExpectTilingOk(r);
    const auto* td = r.Data();
    EXPECT_EQ(td->xElemSize, 2);
}

TEST_F(NonZeroTilingArch22Test, NonZeroTilingArch22_dtype_bf16)
{
    TilingRunner r({64}, ge::DT_BF16, /*transpose=*/false);
    ExpectTilingOk(r);
    const auto* td = r.Data();
    EXPECT_EQ(td->xElemSize, -2); // bf16 以符号位编码，kernel 取绝对值得字节数
}

TEST_F(NonZeroTilingArch22Test, NonZeroTilingArch22_dtype_fp32)
{
    TilingRunner r({64}, ge::DT_FLOAT, /*transpose=*/false);
    ExpectTilingOk(r);
    const auto* td = r.Data();
    EXPECT_EQ(td->xElemSize, 4);
}

// ---- transpose 属性 0/1 ------------------------------------------------------------------------
TEST_F(NonZeroTilingArch22Test, NonZeroTilingArch22_transpose_false)
{
    TilingRunner r({1000}, ge::DT_FLOAT, /*transpose=*/false);
    ExpectTilingOk(r);
    EXPECT_EQ(r.Data()->transpose, 0);
}

TEST_F(NonZeroTilingArch22Test, NonZeroTilingArch22_transpose_true)
{
    TilingRunner r({1000}, ge::DT_FLOAT, /*transpose=*/true);
    ExpectTilingOk(r);
    const auto* td = r.Data();
    EXPECT_EQ(td->transpose, 1);
    // transpose 只影响输出布局，切核参数不应变化
    EXPECT_EQ(td->numel, 1000);
    EXPECT_EQ(td->realCoreNum, 64);
    EXPECT_EQ(td->numPerCore8, 8);
}

// ---- 失败分支：coreNum <= 0 --------------------------------------------------------------------
TEST_F(NonZeroTilingArch22Test, NonZeroTilingArch22_core_num_zero_failed)
{
    (void)error_message::GetErrMgrErrorMessage();
    TilingRunner r({1000}, ge::DT_FLOAT, /*transpose=*/false, /*coreNum=*/0);
    EXPECT_EQ(r.Status(), ge::GRAPH_FAILED);
    ExpectSingleError("EZ1008", {"NonZero", "tiling execution failed", "Preserve the execution logs"});
}

// ---- 失败分支：ubSize <= 0 --------------------------------------------------------------------
TEST_F(NonZeroTilingArch22Test, NonZeroTilingArch22_ub_size_zero_failed)
{
    (void)error_message::GetErrMgrErrorMessage();
    TilingRunner r({1000}, ge::DT_FLOAT, /*transpose=*/false, DEFAULT_CORE_NUM, /*ubSize=*/0);
    EXPECT_EQ(r.Status(), ge::GRAPH_FAILED);
    ExpectSingleError("EZ1008", {"NonZero", "tiling execution failed", "Preserve the execution logs"});
}

// ---- 失败分支：rank == 0（空 shape） -----------------------------------------------------------
TEST_F(NonZeroTilingArch22Test, NonZeroTilingArch22_rank_zero_failed)
{
    (void)error_message::GetErrMgrErrorMessage();
    TilingRunner r({}, ge::DT_FLOAT, /*transpose=*/false);
    EXPECT_EQ(r.Status(), ge::GRAPH_FAILED);
    ExpectSingleError("EZ0012", {"Parameter x", "shape dim 0", "rank must be in [1, 8]"});
}

// ---- 失败分支：rank > 8 ------------------------------------------------------------------------
TEST_F(NonZeroTilingArch22Test, NonZeroTilingArch22_rank_exceed_max_failed)
{
    (void)error_message::GetErrMgrErrorMessage();
    TilingRunner r({2, 2, 2, 2, 2, 2, 2, 2, 2}, ge::DT_FLOAT, /*transpose=*/false);
    EXPECT_EQ(r.Status(), ge::GRAPH_FAILED);
    ExpectSingleError("EZ0012", {"Parameter x", "shape dim 9", "rank must be in [1, 8]"});
}

// ---- 失败分支：transpose 属性缺失 --------------------------------------------------------------
TEST_F(NonZeroTilingArch22Test, NonZeroTilingArch22_transpose_attr_missing_failed)
{
    (void)error_message::GetErrMgrErrorMessage();
    TilingRunner r({1000}, ge::DT_FLOAT, /*transpose=*/false, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE,
                   /*withTransposeAttr=*/false);
    EXPECT_EQ(r.Status(), ge::GRAPH_FAILED);
    ExpectSingleError("EZ1008", {"NonZero", "tiling execution failed", "Preserve the execution logs"});
}

TEST_F(NonZeroTilingArch22Test, NonZeroTilingArch22_invalid_output_dtype_failed)
{
    (void)error_message::GetErrMgrErrorMessage();
    TilingRunner r({1000}, ge::DT_FLOAT, false, DEFAULT_CORE_NUM, DEFAULT_UB_SIZE, true, ge::DT_FLOAT);
    EXPECT_EQ(r.Status(), ge::GRAPH_FAILED);
    ExpectSingleError("EZ0020", {"Parameter y", "FLOAT", "set the dtype attribute to int32 or int64"});
}
