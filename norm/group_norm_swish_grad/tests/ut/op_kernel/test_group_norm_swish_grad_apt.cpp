/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unistd.h>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "group_norm_swish_grad_tiling_def.h"
#include "data_utils.h"

using namespace std;

extern "C" __global__ __aicore__ void group_norm_swish_grad(GM_ADDR dy, GM_ADDR mean, GM_ADDR rstd, GM_ADDR x,
                                                            GM_ADDR gamma, GM_ADDR beta, GM_ADDR dx, GM_ADDR dgamma,
                                                            GM_ADDR dbeta, GM_ADDR workspace, GM_ADDR tiling);

class group_norm_swish_grad_apt_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "group_norm_swish_grad_apt_test SetUp\n" << endl; }

    static void TearDownTestCase() { cout << "group_norm_swish_grad_apt_test TearDown\n" << endl; }
};

namespace {
constexpr uint32_t GROUP_FIT_UB_KEY = 0;
constexpr uint32_t CHANNEL_FIT_UB_KEY = 1;
constexpr uint32_t CHANNEL_SPLIT_UB_KEY = 3;
constexpr size_t DEFAULT_WORKSPACE_BYTES = 4096 * 16;

struct KernelCaseConfig {
    uint32_t kernelTilingKey = 0;
    uint32_t stage1TilingKey = GROUP_FIT_UB_KEY;
    uint32_t n = 1;
    uint32_t c = 8;
    uint32_t hxw = 64;
    uint32_t g = 2;
    uint32_t blockDim = 8;
    uint32_t taskNumPerCore = 1;
    uint32_t taskNumPerTailCore = 1;
    uint32_t tailCore = 8;
    uint32_t mode1UbCapCNum = 1;
    uint32_t mode1UbIterCNum = 1;
    uint32_t mode1UbTailCNum = 1;
    uint32_t mode2UbCapacityEle = 0;
    uint32_t mode2UbIterationNum = 0;
    uint32_t mode2UbTailNum = 0;
    uint32_t workSpaceSize = 0;
    uint32_t stage2CoreUsed = 0;
    uint32_t castEleNum = 64;
    uint32_t tailCastNum = 0;
    uint32_t coreBatchParts = 1;
    uint32_t coreBatchPartsTailRepeat = 0;
    uint32_t repeatTime4Stage2 = 0;
    uint32_t dgammaIsRequire = 1;
    uint32_t dbetaIsRequire = 1;
    float swishScale = 1.0f;
    string dataType = "float";
};

static void FillTilingData(GroupNormSwishGradTilingData* tilingData, const KernelCaseConfig& config)
{
    tilingData->Tiling_key = config.stage1TilingKey;
    tilingData->N = config.n;
    tilingData->C = config.c;
    tilingData->HXW = config.hxw;
    tilingData->G = config.g;
    tilingData->NXG = config.n * config.g;
    tilingData->C_G = config.c / config.g;
    tilingData->task_num_per_core = config.taskNumPerCore;
    tilingData->task_num_per_tail_core = config.taskNumPerTailCore;
    tilingData->tail_core = config.tailCore;
    tilingData->mode1_ub_cap_C_num = config.mode1UbCapCNum;
    tilingData->mode1_ub_iter_C_num = config.mode1UbIterCNum;
    tilingData->mode1_ub_tail_C_num = config.mode1UbTailCNum;
    tilingData->mode2_ub_capacity_ele = config.mode2UbCapacityEle;
    tilingData->mode2_ub_iteration_num = config.mode2UbIterationNum;
    tilingData->mode2_ub_tail_num = config.mode2UbTailNum;
    tilingData->workSpaceSize = config.workSpaceSize;
    tilingData->stage2CoreUsed = config.stage2CoreUsed;
    tilingData->castEleNum = config.castEleNum;
    tilingData->tailCastNum = config.tailCastNum;
    tilingData->coreBatchParts = config.coreBatchParts;
    tilingData->coreBatchPartsTailRepeat = config.coreBatchPartsTailRepeat;
    tilingData->repeatTime4Stage2 = config.repeatTime4Stage2;
    tilingData->dgamma_is_require = config.dgammaIsRequire;
    tilingData->dbeta_is_require = config.dbetaIsRequire;
    tilingData->swish_scale = config.swishScale;
}

static void PrepareInputData(const KernelCaseConfig& config)
{
    system("cp -r "
           "../../../../norm/group_norm_swish_grad/tests/ut/op_kernel/group_norm_swish_grad_data ./");
    system("chmod -R 755 ./group_norm_swish_grad_data/");
    system("cd ./group_norm_swish_grad_data/ && rm -rf ./*bin");
    string command = "cd ./group_norm_swish_grad_data/ && python3 gen_data.py " + to_string(config.n) + " " +
                     to_string(config.c) + " " + to_string(config.hxw) + " " + to_string(config.g) + " " +
                     config.dataType;
    system(command.c_str());
}

static void RunKernelCase(const KernelCaseConfig& config)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    PrepareInputData(config);

    const size_t allEleNum = static_cast<size_t>(config.n) * config.c * config.hxw;
    const size_t nxg = static_cast<size_t>(config.n) * config.g;
    size_t dySize = allEleNum * sizeof(float);
    size_t meanSize = nxg * sizeof(float);
    size_t rstdSize = meanSize;
    size_t xSize = dySize;
    size_t gammaSize = static_cast<size_t>(config.c) * sizeof(float);
    size_t betaSize = gammaSize;
    size_t dxSize = dySize;
    size_t dgammaSize = gammaSize;
    size_t dbetaSize = betaSize;
    const size_t minWorkspaceSize = static_cast<size_t>(config.workSpaceSize) * 2 * sizeof(float) +
                                    DEFAULT_WORKSPACE_BYTES;
    const size_t workspaceSize = (minWorkspaceSize > DEFAULT_WORKSPACE_BYTES) ? minWorkspaceSize :
                                                                                DEFAULT_WORKSPACE_BYTES;

    uint8_t* dy = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(dySize));
    uint8_t* mean = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(meanSize));
    uint8_t* rstd = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(rstdSize));
    uint8_t* x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(xSize));
    uint8_t* gamma = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(gammaSize));
    uint8_t* beta = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(betaSize));
    uint8_t* dx = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(dxSize));
    uint8_t* dgamma = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(dgammaSize));
    uint8_t* dbeta = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(dbetaSize));
    uint8_t* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(workspaceSize));
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(GroupNormSwishGradTilingData)));

    char* currentDir = get_current_dir_name();
    string path(currentDir);
    ReadFile(path + "/group_norm_swish_grad_data/input_dy.bin", dySize, dy, dySize);
    ReadFile(path + "/group_norm_swish_grad_data/input_mean.bin", meanSize, mean, meanSize);
    ReadFile(path + "/group_norm_swish_grad_data/input_rstd.bin", rstdSize, rstd, rstdSize);
    ReadFile(path + "/group_norm_swish_grad_data/input_x.bin", xSize, x, xSize);
    ReadFile(path + "/group_norm_swish_grad_data/input_gamma.bin", gammaSize, gamma, gammaSize);
    ReadFile(path + "/group_norm_swish_grad_data/input_beta.bin", betaSize, beta, betaSize);

    auto* tilingData = reinterpret_cast<GroupNormSwishGradTilingData*>(tiling);
    FillTilingData(tilingData, config);
    ICPU_SET_TILING_KEY(config.kernelTilingKey);
    ICPU_RUN_KF(group_norm_swish_grad, config.blockDim, dy, mean, rstd, x, gamma, beta, dx, dgamma, dbeta, workspace,
                reinterpret_cast<uint8_t*>(tilingData));

    AscendC::GmFree(dy);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(currentDir);
}
} // namespace

// CMake defines __CCE_AICORE__=350 for the 950 arch35 UT target; keep this case out of other UT builds.
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 350)
TEST_F(group_norm_swish_grad_apt_test, test_case_group_fit_unalign_fp32)
{
    KernelCaseConfig config;
    config.kernelTilingKey = 0;
    config.stage1TilingKey = GROUP_FIT_UB_KEY;
    config.n = 1;
    config.c = 8;
    config.hxw = 65;
    config.g = 2;
    config.blockDim = 2;
    config.tailCore = 2;
    config.mode1UbCapCNum = 4;
    config.mode1UbTailCNum = 4;
    RunKernelCase(config);
}

TEST_F(group_norm_swish_grad_apt_test, test_case_channel_fit_fp16_dbeta_only)
{
    KernelCaseConfig config;
    config.kernelTilingKey = 1;
    config.stage1TilingKey = CHANNEL_FIT_UB_KEY;
    config.n = 1;
    config.c = 16;
    config.hxw = 256;
    config.g = 4;
    config.blockDim = 4;
    config.tailCore = 4;
    config.mode1UbIterCNum = 4;
    config.mode1UbTailCNum = 1;
    config.dgammaIsRequire = 0;
    config.dbetaIsRequire = 1;
    config.dataType = "float16";
    RunKernelCase(config);
}

TEST_F(group_norm_swish_grad_apt_test, test_case_stage2_fp32_deterministic_kahan_tail)
{
    KernelCaseConfig config;
    config.kernelTilingKey = 10;
    config.stage1TilingKey = GROUP_FIT_UB_KEY;
    config.n = 5;
    config.c = 96;
    config.hxw = 16;
    config.g = 4;
    config.blockDim = 8;
    config.taskNumPerCore = 3;
    config.taskNumPerTailCore = 2;
    config.tailCore = 4;
    config.mode1UbCapCNum = 24;
    config.mode1UbTailCNum = 24;
    config.workSpaceSize = 288;
    config.stage2CoreUsed = 2;
    config.castEleNum = 64;
    config.tailCastNum = 32;
    config.coreBatchParts = 2;
    config.repeatTime4Stage2 = 2;
    RunKernelCase(config);
}

TEST_F(group_norm_swish_grad_apt_test, test_case_stage2_fp16_deterministic_reduce_cast)
{
    KernelCaseConfig config;
    config.kernelTilingKey = 11;
    config.stage1TilingKey = GROUP_FIT_UB_KEY;
    config.n = 4;
    config.c = 32;
    config.hxw = 64;
    config.g = 4;
    config.blockDim = 8;
    config.taskNumPerCore = 2;
    config.taskNumPerTailCore = 2;
    config.tailCore = 8;
    config.mode1UbCapCNum = 8;
    config.mode1UbTailCNum = 8;
    config.workSpaceSize = 64;
    config.stage2CoreUsed = 1;
    config.castEleNum = 64;
    config.tailCastNum = 32;
    config.coreBatchParts = 2;
    config.repeatTime4Stage2 = 1;
    config.dataType = "float16";
    RunKernelCase(config);
}

TEST_F(group_norm_swish_grad_apt_test, test_case_channel_split_bf16_deterministic)
{
    KernelCaseConfig config;
    config.kernelTilingKey = 12;
    config.stage1TilingKey = CHANNEL_SPLIT_UB_KEY;
    config.n = 2;
    config.c = 16;
    config.hxw = 512;
    config.g = 4;
    config.blockDim = 8;
    config.taskNumPerCore = 1;
    config.taskNumPerTailCore = 1;
    config.tailCore = 8;
    config.mode2UbCapacityEle = 256;
    config.mode2UbIterationNum = 2;
    config.mode2UbTailNum = 256;
    config.workSpaceSize = 16;
    config.stage2CoreUsed = 1;
    config.castEleNum = 64;
    config.tailCastNum = 16;
    config.coreBatchParts = 1;
    config.repeatTime4Stage2 = 1;
    config.dataType = "float16";
    RunKernelCase(config);
}
#endif
