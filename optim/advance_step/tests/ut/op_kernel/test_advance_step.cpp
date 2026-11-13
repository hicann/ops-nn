/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cstring>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "test_advance_step_tiling_def.h"

using namespace std;

extern "C" __global__ __aicore__ void advance_step(GM_ADDR input_tokens, GM_ADDR sampled_token_ids,
                                                   GM_ADDR input_positions, GM_ADDR seq_lens, GM_ADDR slot_mapping,
                                                   GM_ADDR block_tables, GM_ADDR spec_token, GM_ADDR accepted_num,
                                                   GM_ADDR input_tokens_out, GM_ADDR input_positions_out,
                                                   GM_ADDR seq_lens_out, GM_ADDR slot_mapping_out, GM_ADDR workspace,
                                                   GM_ADDR tiling);
class advance_step_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "advance_step SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "advance_step TearDown\n" << endl; }
};

TEST_F(advance_step_test, test_advance_step_int_0)
{
    system("cp -rf "
           "../../../../optim/advance_step/tests/ut/op_kernel/advance_step_data ./");

    system("chmod -R 755 ./advance_step_data/");
    // aiv_num 与下方 needCoreNum 一致（ascend910B1 为 40，ascend950pr_9599 为 56）
    system("cd ./advance_step_data/ && python3 gen_data.py '(16, 1)' '(8, 1)' 'int64' 40");
    size_t M = 16;
    size_t N = 8;
    size_t D = N / 2;

    size_t inputTokenSize = M * sizeof(int64_t);
    size_t sampledTokenSize = N * sizeof(int64_t);

    uint8_t* input_tokens = (uint8_t*)AscendC::GmAlloc(inputTokenSize);
    uint8_t* sampled_token_ids = (uint8_t*)AscendC::GmAlloc(sampledTokenSize);
    uint8_t* input_positions = (uint8_t*)AscendC::GmAlloc(inputTokenSize);
    uint8_t* seq_lens = (uint8_t*)AscendC::GmAlloc(inputTokenSize);
    uint8_t* slot_mapping = (uint8_t*)AscendC::GmAlloc(inputTokenSize);
    uint8_t* block_tables = (uint8_t*)AscendC::GmAlloc(inputTokenSize);

    uint64_t tilingKey = 1;
    uint32_t blockDim = 1;
    size_t workspaceFileSize = 16781184;
    int64_t needCoreNum = 40;
    int64_t blockTablesStride = 1;
    int64_t numSeqs = 16;
    int64_t numQueries = 8;
    int64_t blockSize = 8;
    size_t tilingDataSize = sizeof(AdvanceStepTilingDataTest);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    std::string fileName1 = "./advance_step_data/input_tokens_input_advance_step.bin";
    std::string fileName2 = "./advance_step_data/sampled_token_ids_input_advance_step.bin";
    std::string fileName3 = "./advance_step_data/input_positions_input_advance_step.bin";
    std::string fileName4 = "./advance_step_data/seq_lens_input_advance_step.bin";
    std::string fileName5 = "./advance_step_data/slot_mapping_input_advance_step.bin";
    std::string fileName6 = "./advance_step_data/block_tables_input_advance_step.bin";

    ReadFile(fileName1, inputTokenSize, input_tokens, inputTokenSize);
    ReadFile(fileName2, sampledTokenSize, sampled_token_ids, sampledTokenSize);
    ReadFile(fileName3, inputTokenSize, input_positions, inputTokenSize);
    ReadFile(fileName4, inputTokenSize, seq_lens, inputTokenSize);
    ReadFile(fileName5, inputTokenSize, slot_mapping, inputTokenSize);
    ReadFile(fileName6, inputTokenSize, block_tables, inputTokenSize);

    AdvanceStepTilingDataTest* tilingDatafromBin = reinterpret_cast<AdvanceStepTilingDataTest*>(tiling);
    tilingDatafromBin->needCoreNum = needCoreNum;
    tilingDatafromBin->blockTablesStride = blockTablesStride;
    tilingDatafromBin->numSeqs = numSeqs;
    tilingDatafromBin->numQueries = numQueries;
    tilingDatafromBin->blockSize = blockSize;
    // 单核执行（blockDim=1）：每核行数 = numQueries，核 0 覆盖全部 query 行
    tilingDatafromBin->perCoreSeqs = numQueries;

    ICPU_SET_TILING_KEY(tilingKey);
    // CPU 仿真默认 MIX_MODE 会为混合核组派生多个子进程重复执行 kernel（blockDim=1 时为 3 次），
    // 导致 inplace 累加型输出与单次执行的 golden 不一致；AIV_MODE 下按 blockDim 1:1 派生
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(advance_step, blockDim, input_tokens, sampled_token_ids, input_positions, seq_lens, slot_mapping,
                block_tables, nullptr, nullptr, input_tokens, input_positions, seq_lens, slot_mapping, workspace,
                (uint8_t*)tilingDatafromBin);
    fileName1 = "./advance_step_data/1_output_advance_step.bin";
    fileName2 = "./advance_step_data/2_output_advance_step.bin";
    fileName3 = "./advance_step_data/3_output_advance_step.bin";
    fileName4 = "./advance_step_data/4_output_advance_step.bin";
    WriteFile(fileName1, input_tokens, inputTokenSize);
    WriteFile(fileName2, input_positions, inputTokenSize);
    WriteFile(fileName3, seq_lens, inputTokenSize);
    WriteFile(fileName4, slot_mapping, inputTokenSize);

    AscendC::GmFree((void*)input_tokens);
    AscendC::GmFree((void*)sampled_token_ids);
    AscendC::GmFree((void*)input_positions);
    AscendC::GmFree((void*)seq_lens);
    AscendC::GmFree((void*)slot_mapping);
    AscendC::GmFree((void*)block_tables);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    EXPECT_EQ(system("cd ./advance_step_data/ && python3 compare_data.py 'int64'"), 0);
}

// V1 路径契约外值域（垃圾数据）防御见证：seqLens 含越上界/负值/压界位置时，
// 查表下标越出 blockTables 张量包络则不读 GM、查表值取 0（与 V2 A5 防护语义一致）；
// 对应 issue「AdvanceStep kernel 内存越界」的修复验证
TEST_F(advance_step_test, test_advance_step_int_1)
{
    system("cp -rf "
           "../../../../optim/advance_step/tests/ut/op_kernel/advance_step_data ./");
    system("chmod -R 755 ./advance_step_data/");
    // aiv_num 与下方 needCoreNum 一致（ascend910B1 为 40，ascend950pr_9599 为 56）
    system("cd ./advance_step_data/ && python3 gen_data.py v1oob 40");
    size_t M = 16;
    size_t N = 8;

    size_t inputTokenSize = M * sizeof(int64_t);
    size_t sampledTokenSize = N * sizeof(int64_t);

    uint8_t* input_tokens = (uint8_t*)AscendC::GmAlloc(inputTokenSize);
    uint8_t* sampled_token_ids = (uint8_t*)AscendC::GmAlloc(sampledTokenSize);
    uint8_t* input_positions = (uint8_t*)AscendC::GmAlloc(inputTokenSize);
    uint8_t* seq_lens = (uint8_t*)AscendC::GmAlloc(inputTokenSize);
    uint8_t* slot_mapping = (uint8_t*)AscendC::GmAlloc(inputTokenSize);
    uint8_t* block_tables = (uint8_t*)AscendC::GmAlloc(inputTokenSize);

    uint64_t tilingKey = 1;
    uint32_t blockDim = 1;
    size_t workspaceFileSize = 16781184;
    int64_t needCoreNum = 40;
    int64_t blockTablesStride = 1;
    int64_t numSeqs = 16;
    int64_t numQueries = 8;
    int64_t blockSize = 8;
    size_t tilingDataSize = sizeof(AdvanceStepTilingDataTest);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    std::string fileName1 = "./advance_step_data/oob_input_tokens_input_advance_step.bin";
    std::string fileName2 = "./advance_step_data/oob_sampled_token_ids_input_advance_step.bin";
    std::string fileName3 = "./advance_step_data/oob_input_positions_input_advance_step.bin";
    std::string fileName4 = "./advance_step_data/oob_seq_lens_input_advance_step.bin";
    std::string fileName5 = "./advance_step_data/oob_slot_mapping_input_advance_step.bin";
    std::string fileName6 = "./advance_step_data/oob_block_tables_input_advance_step.bin";

    ReadFile(fileName1, inputTokenSize, input_tokens, inputTokenSize);
    ReadFile(fileName2, sampledTokenSize, sampled_token_ids, sampledTokenSize);
    ReadFile(fileName3, inputTokenSize, input_positions, inputTokenSize);
    ReadFile(fileName4, inputTokenSize, seq_lens, inputTokenSize);
    ReadFile(fileName5, inputTokenSize, slot_mapping, inputTokenSize);
    ReadFile(fileName6, inputTokenSize, block_tables, inputTokenSize);

    AdvanceStepTilingDataTest* tilingDatafromBin = reinterpret_cast<AdvanceStepTilingDataTest*>(tiling);
    tilingDatafromBin->needCoreNum = needCoreNum;
    tilingDatafromBin->blockTablesStride = blockTablesStride;
    tilingDatafromBin->numSeqs = numSeqs;
    tilingDatafromBin->numQueries = numQueries;
    tilingDatafromBin->blockSize = blockSize;
    // 单核执行（blockDim=1）：每核行数 = numQueries，核 0 覆盖全部 query 行
    tilingDatafromBin->perCoreSeqs = numQueries;

    ICPU_SET_TILING_KEY(tilingKey);
    // CPU 仿真默认 MIX_MODE 会为混合核组派生多个子进程重复执行 kernel
    // （blockDim=1 时为 3 次），导致 inplace 累加型输出与单次执行的 golden 不一致；
    // AIV_MODE 下按 blockDim 1:1 派生
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(advance_step, blockDim, input_tokens, sampled_token_ids, input_positions, seq_lens, slot_mapping,
                block_tables, nullptr, nullptr, input_tokens, input_positions, seq_lens, slot_mapping, workspace,
                (uint8_t*)tilingDatafromBin);
    fileName1 = "./advance_step_data/oob_1_output_advance_step.bin";
    fileName2 = "./advance_step_data/oob_2_output_advance_step.bin";
    fileName3 = "./advance_step_data/oob_3_output_advance_step.bin";
    fileName4 = "./advance_step_data/oob_4_output_advance_step.bin";
    WriteFile(fileName1, input_tokens, inputTokenSize);
    WriteFile(fileName2, input_positions, inputTokenSize);
    WriteFile(fileName3, seq_lens, inputTokenSize);
    WriteFile(fileName4, slot_mapping, inputTokenSize);

    AscendC::GmFree((void*)input_tokens);
    AscendC::GmFree((void*)sampled_token_ids);
    AscendC::GmFree((void*)input_positions);
    AscendC::GmFree((void*)seq_lens);
    AscendC::GmFree((void*)slot_mapping);
    AscendC::GmFree((void*)block_tables);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    EXPECT_EQ(system("cd ./advance_step_data/ && python3 compare_data.py 'int64' oob_"), 0);
}

// V1（legacy）路径多核切分：blockDim=8、每核 1 行（perCoreSeqs=1，与真实 tiling 对
// numQueries<aivNum 的切分一致），数据与 golden 与单核用例完全相同——见证分区不改变输出
TEST_F(advance_step_test, test_advance_step_int_4)
{
    system("cp -rf "
           "../../../../optim/advance_step/tests/ut/op_kernel/advance_step_data ./");

    system("chmod -R 755 ./advance_step_data/");
    // aiv_num 与下方 needCoreNum 一致（ascend910B1 为 40，ascend950pr_9599 为 56）
    system("cd ./advance_step_data/ && python3 gen_data.py '(16, 1)' '(8, 1)' 'int64' 40");
    size_t M = 16;
    size_t N = 8;

    size_t inputTokenSize = M * sizeof(int64_t);
    size_t sampledTokenSize = N * sizeof(int64_t);

    uint8_t* input_tokens = (uint8_t*)AscendC::GmAlloc(inputTokenSize);
    uint8_t* sampled_token_ids = (uint8_t*)AscendC::GmAlloc(sampledTokenSize);
    uint8_t* input_positions = (uint8_t*)AscendC::GmAlloc(inputTokenSize);
    uint8_t* seq_lens = (uint8_t*)AscendC::GmAlloc(inputTokenSize);
    uint8_t* slot_mapping = (uint8_t*)AscendC::GmAlloc(inputTokenSize);
    uint8_t* block_tables = (uint8_t*)AscendC::GmAlloc(inputTokenSize);

    uint64_t tilingKey = 1;
    uint32_t blockDim = 8;
    size_t workspaceFileSize = 16781184;
    int64_t needCoreNum = 40;
    int64_t blockTablesStride = 1;
    int64_t numSeqs = 16;
    int64_t numQueries = 8;
    int64_t blockSize = 8;
    size_t tilingDataSize = sizeof(AdvanceStepTilingDataTest);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    std::string fileName1 = "./advance_step_data/input_tokens_input_advance_step.bin";
    std::string fileName2 = "./advance_step_data/sampled_token_ids_input_advance_step.bin";
    std::string fileName3 = "./advance_step_data/input_positions_input_advance_step.bin";
    std::string fileName4 = "./advance_step_data/seq_lens_input_advance_step.bin";
    std::string fileName5 = "./advance_step_data/slot_mapping_input_advance_step.bin";
    std::string fileName6 = "./advance_step_data/block_tables_input_advance_step.bin";

    ReadFile(fileName1, inputTokenSize, input_tokens, inputTokenSize);
    ReadFile(fileName2, sampledTokenSize, sampled_token_ids, sampledTokenSize);
    ReadFile(fileName3, inputTokenSize, input_positions, inputTokenSize);
    ReadFile(fileName4, inputTokenSize, seq_lens, inputTokenSize);
    ReadFile(fileName5, inputTokenSize, slot_mapping, inputTokenSize);
    ReadFile(fileName6, inputTokenSize, block_tables, inputTokenSize);

    AdvanceStepTilingDataTest* tilingDatafromBin = reinterpret_cast<AdvanceStepTilingDataTest*>(tiling);
    tilingDatafromBin->needCoreNum = needCoreNum;
    tilingDatafromBin->blockTablesStride = blockTablesStride;
    tilingDatafromBin->numSeqs = numSeqs;
    tilingDatafromBin->numQueries = numQueries;
    tilingDatafromBin->blockSize = blockSize;
    // 多核执行（blockDim=8）：每核行数 = CeilDiv(8, 40) = 1，核 b 处理行 b
    tilingDatafromBin->perCoreSeqs = 1;

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(advance_step, blockDim, input_tokens, sampled_token_ids, input_positions, seq_lens, slot_mapping,
                block_tables, nullptr, nullptr, input_tokens, input_positions, seq_lens, slot_mapping, workspace,
                (uint8_t*)tilingDatafromBin);
    fileName1 = "./advance_step_data/1_output_advance_step.bin";
    fileName2 = "./advance_step_data/2_output_advance_step.bin";
    fileName3 = "./advance_step_data/3_output_advance_step.bin";
    fileName4 = "./advance_step_data/4_output_advance_step.bin";
    WriteFile(fileName1, input_tokens, inputTokenSize);
    WriteFile(fileName2, input_positions, inputTokenSize);
    WriteFile(fileName3, seq_lens, inputTokenSize);
    WriteFile(fileName4, slot_mapping, inputTokenSize);

    AscendC::GmFree((void*)input_tokens);
    AscendC::GmFree((void*)sampled_token_ids);
    AscendC::GmFree((void*)input_positions);
    AscendC::GmFree((void*)seq_lens);
    AscendC::GmFree((void*)slot_mapping);
    AscendC::GmFree((void*)block_tables);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    EXPECT_EQ(system("cd ./advance_step_data/ && python3 compare_data.py 'int64'"), 0);
}

static void RunV2Case(bool vectorGather)
{
    // V2（投机采样）路径：参数与 gen_data.py v2 模式保持一致
    size_t numSeqs = 16;
    size_t specNum = 2;
    const size_t tokenEachReqs = specNum + 1;
    const size_t blockTablesStride = 1000;
    const int64_t blockSize = 128;

    system("cp -rf "
           "../../../../optim/advance_step/tests/ut/op_kernel/advance_step_data ./");
    system("chmod -R 755 ./advance_step_data/");
    system("cd ./advance_step_data/ && python3 gen_data.py v2");

    size_t stateSize = numSeqs * tokenEachReqs * sizeof(int64_t);
    size_t sampledTokenSize = numSeqs * tokenEachReqs * sizeof(int64_t);
    size_t specTokenSize = numSeqs * specNum * sizeof(int64_t);
    size_t acceptedNumSize = numSeqs * sizeof(int64_t);
    size_t blockTableSize = numSeqs * blockTablesStride * sizeof(int64_t);

    uint8_t* input_tokens = (uint8_t*)AscendC::GmAlloc(stateSize);
    uint8_t* sampled_token_ids = (uint8_t*)AscendC::GmAlloc(sampledTokenSize);
    uint8_t* input_positions = (uint8_t*)AscendC::GmAlloc(stateSize);
    uint8_t* seq_lens = (uint8_t*)AscendC::GmAlloc(stateSize);
    uint8_t* slot_mapping = (uint8_t*)AscendC::GmAlloc(stateSize);
    uint8_t* block_tables = (uint8_t*)AscendC::GmAlloc(blockTableSize);
    uint8_t* accepted_num = (uint8_t*)AscendC::GmAlloc(acceptedNumSize);
    uint8_t* spec_token = (uint8_t*)AscendC::GmAlloc(specTokenSize);

    uint64_t tilingKey = 2;
    uint32_t blockDim = 16;
    size_t workspaceFileSize = 16781184;
    size_t tilingDataSize = sizeof(AdvanceStepTilingDataTest);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    std::string fileName1 = "./advance_step_data/v2_input_tokens_input_advance_step.bin";
    std::string fileName2 = "./advance_step_data/v2_sampled_token_ids_input_advance_step.bin";
    std::string fileName3 = "./advance_step_data/v2_input_positions_input_advance_step.bin";
    std::string fileName4 = "./advance_step_data/v2_seq_lens_input_advance_step.bin";
    std::string fileName5 = "./advance_step_data/v2_slot_mapping_input_advance_step.bin";
    std::string fileName6 = "./advance_step_data/v2_block_tables_input_advance_step.bin";
    std::string fileName7 = "./advance_step_data/v2_spec_token_input_advance_step.bin";
    std::string fileName8 = "./advance_step_data/v2_accepted_num_input_advance_step.bin";
    ReadFile(fileName1, stateSize, input_tokens, stateSize);
    ReadFile(fileName2, sampledTokenSize, sampled_token_ids, sampledTokenSize);
    ReadFile(fileName3, stateSize, input_positions, stateSize);
    ReadFile(fileName4, stateSize, seq_lens, stateSize);
    ReadFile(fileName5, stateSize, slot_mapping, stateSize);
    ReadFile(fileName6, blockTableSize, block_tables, blockTableSize);
    ReadFile(fileName7, specTokenSize, spec_token, specTokenSize);
    ReadFile(fileName8, acceptedNumSize, accepted_num, acceptedNumSize);

    AdvanceStepTilingDataTest* tilingDatafromBin = reinterpret_cast<AdvanceStepTilingDataTest*>(tiling);
    tilingDatafromBin->needCoreNum = 16;
    tilingDatafromBin->blockTablesStride = blockTablesStride;
    tilingDatafromBin->numSeqs = numSeqs;
    tilingDatafromBin->numQueries = numSeqs;
    tilingDatafromBin->blockSize = blockSize;
    tilingDatafromBin->perCoreSeqs = 1;
    tilingDatafromBin->lastCoreSeqs = 1;
    tilingDatafromBin->perLoopMaxSeqs = 1;
    tilingDatafromBin->specNum = specNum;
    // vectorGather=true 走分块搬运+向量 gather；false 走标量防护回退（两条路径均需覆盖）
    tilingDatafromBin->vectorGather = vectorGather ? 1 : 0;

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(advance_step, blockDim, input_tokens, sampled_token_ids, input_positions, seq_lens, slot_mapping,
                block_tables, spec_token, accepted_num, input_tokens, input_positions, seq_lens, slot_mapping,
                workspace, (uint8_t*)tilingDatafromBin);

    fileName1 = "./advance_step_data/v2_1_output_advance_step.bin";
    fileName2 = "./advance_step_data/v2_2_output_advance_step.bin";
    fileName3 = "./advance_step_data/v2_3_output_advance_step.bin";
    fileName4 = "./advance_step_data/v2_4_output_advance_step.bin";
    WriteFile(fileName1, input_tokens, stateSize);
    WriteFile(fileName2, input_positions, stateSize);
    WriteFile(fileName3, seq_lens, stateSize);
    WriteFile(fileName4, slot_mapping, stateSize);

    AscendC::GmFree((void*)input_tokens);
    AscendC::GmFree((void*)sampled_token_ids);
    AscendC::GmFree((void*)input_positions);
    AscendC::GmFree((void*)seq_lens);
    AscendC::GmFree((void*)slot_mapping);
    AscendC::GmFree((void*)block_tables);
    AscendC::GmFree((void*)accepted_num);
    AscendC::GmFree((void*)spec_token);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    EXPECT_EQ(system("cd ./advance_step_data/ && python3 compare_data.py 'int64' v2_"), 0);
}

// V2（投机采样）路径：分块搬运 + 向量 gather（tiling vectorGather=1）
TEST_F(advance_step_test, test_advance_step_int_2) { RunV2Case(true); }

// V2（投机采样）路径：标量防护回退（vectorGather=0，超大 stride 场景路径）
TEST_F(advance_step_test, test_advance_step_int_3) { RunV2Case(false); }

// V2（投机采样）specNum=1（tokenEachReqs=2）路径：覆盖 ComputeInputTokens 的 te==2
// 向量化哨兵检索快速路径（lastToken 三分支：行首 -1 / 行内 -1 / 无 -1）与逐行标量化调用
// 向量化后的行展开/flat 计算；参数与 gen_data.py v2spec1 模式保持一致
static void RunV2Spec1Case(bool vectorGather)
{
    size_t numSeqs = 16;
    size_t specNum = 1;
    const size_t tokenEachReqs = specNum + 1;
    const size_t blockTablesStride = 1000;
    const int64_t blockSize = 128;

    system("cp -rf "
           "../../../../optim/advance_step/tests/ut/op_kernel/advance_step_data ./");
    system("chmod -R 755 ./advance_step_data/");
    system("cd ./advance_step_data/ && python3 gen_data.py v2spec1");

    size_t stateSize = numSeqs * tokenEachReqs * sizeof(int64_t);
    size_t sampledTokenSize = numSeqs * tokenEachReqs * sizeof(int64_t);
    size_t specTokenSize = numSeqs * specNum * sizeof(int64_t);
    size_t acceptedNumSize = numSeqs * sizeof(int64_t);
    size_t blockTableSize = numSeqs * blockTablesStride * sizeof(int64_t);

    uint8_t* input_tokens = (uint8_t*)AscendC::GmAlloc(stateSize);
    uint8_t* sampled_token_ids = (uint8_t*)AscendC::GmAlloc(sampledTokenSize);
    uint8_t* input_positions = (uint8_t*)AscendC::GmAlloc(stateSize);
    uint8_t* seq_lens = (uint8_t*)AscendC::GmAlloc(stateSize);
    uint8_t* slot_mapping = (uint8_t*)AscendC::GmAlloc(stateSize);
    uint8_t* block_tables = (uint8_t*)AscendC::GmAlloc(blockTableSize);
    uint8_t* accepted_num = (uint8_t*)AscendC::GmAlloc(acceptedNumSize);
    uint8_t* spec_token = (uint8_t*)AscendC::GmAlloc(specTokenSize);

    uint64_t tilingKey = 2;
    uint32_t blockDim = 16;
    size_t workspaceFileSize = 16781184;
    size_t tilingDataSize = sizeof(AdvanceStepTilingDataTest);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    std::string fileName1 = "./advance_step_data/v2_input_tokens_input_advance_step.bin";
    std::string fileName2 = "./advance_step_data/v2_sampled_token_ids_input_advance_step.bin";
    std::string fileName3 = "./advance_step_data/v2_input_positions_input_advance_step.bin";
    std::string fileName4 = "./advance_step_data/v2_seq_lens_input_advance_step.bin";
    std::string fileName5 = "./advance_step_data/v2_slot_mapping_input_advance_step.bin";
    std::string fileName6 = "./advance_step_data/v2_block_tables_input_advance_step.bin";
    std::string fileName7 = "./advance_step_data/v2_spec_token_input_advance_step.bin";
    std::string fileName8 = "./advance_step_data/v2_accepted_num_input_advance_step.bin";
    ReadFile(fileName1, stateSize, input_tokens, stateSize);
    ReadFile(fileName2, sampledTokenSize, sampled_token_ids, sampledTokenSize);
    ReadFile(fileName3, stateSize, input_positions, stateSize);
    ReadFile(fileName4, stateSize, seq_lens, stateSize);
    ReadFile(fileName5, stateSize, slot_mapping, stateSize);
    ReadFile(fileName6, blockTableSize, block_tables, blockTableSize);
    ReadFile(fileName7, specTokenSize, spec_token, specTokenSize);
    ReadFile(fileName8, acceptedNumSize, accepted_num, acceptedNumSize);

    AdvanceStepTilingDataTest* tilingDatafromBin = reinterpret_cast<AdvanceStepTilingDataTest*>(tiling);
    tilingDatafromBin->needCoreNum = 16;
    tilingDatafromBin->blockTablesStride = blockTablesStride;
    tilingDatafromBin->numSeqs = numSeqs;
    tilingDatafromBin->numQueries = numSeqs;
    tilingDatafromBin->blockSize = blockSize;
    tilingDatafromBin->perCoreSeqs = 1;
    tilingDatafromBin->lastCoreSeqs = 1;
    tilingDatafromBin->perLoopMaxSeqs = 1;
    tilingDatafromBin->specNum = specNum;
    tilingDatafromBin->vectorGather = vectorGather ? 1 : 0;

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(advance_step, blockDim, input_tokens, sampled_token_ids, input_positions, seq_lens, slot_mapping,
                block_tables, spec_token, accepted_num, input_tokens, input_positions, seq_lens, slot_mapping,
                workspace, (uint8_t*)tilingDatafromBin);

    fileName1 = "./advance_step_data/v2_1_output_advance_step.bin";
    fileName2 = "./advance_step_data/v2_2_output_advance_step.bin";
    fileName3 = "./advance_step_data/v2_3_output_advance_step.bin";
    fileName4 = "./advance_step_data/v2_4_output_advance_step.bin";
    WriteFile(fileName1, input_tokens, stateSize);
    WriteFile(fileName2, input_positions, stateSize);
    WriteFile(fileName3, seq_lens, stateSize);
    WriteFile(fileName4, slot_mapping, stateSize);

    AscendC::GmFree((void*)input_tokens);
    AscendC::GmFree((void*)sampled_token_ids);
    AscendC::GmFree((void*)input_positions);
    AscendC::GmFree((void*)seq_lens);
    AscendC::GmFree((void*)slot_mapping);
    AscendC::GmFree((void*)block_tables);
    AscendC::GmFree((void*)accepted_num);
    AscendC::GmFree((void*)spec_token);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
    EXPECT_EQ(system("cd ./advance_step_data/ && python3 compare_data.py 'int64' v2_"), 0);
}

TEST_F(advance_step_test, test_advance_step_int_5) { RunV2Spec1Case(true); }
TEST_F(advance_step_test, test_advance_step_int_6) { RunV2Spec1Case(false); }

// V1 宽表（blockTablesStride>1）查表公式见证（issue #6573）：物理块号按平坦下标
// stride*i + pos/blockSize 读取，兼容 CPU 仿真单核执行（perCoreSeqs=numQueries）
TEST_F(advance_step_test, test_advance_step_wide_block_tables)
{
    constexpr int64_t kNumSeqs = 3;
    constexpr int64_t kNumQueries = 2;
    constexpr int64_t kBlockSize = 4;
    constexpr int64_t kTableStride = 2;
    constexpr int64_t kNeedCoreNum = 40;
    const size_t seqSize = kNumSeqs * sizeof(int64_t);
    const size_t tableSize = kNumSeqs * kTableStride * sizeof(int64_t);

    uint8_t* input_tokens = (uint8_t*)AscendC::GmAlloc(seqSize);
    uint8_t* sampled_token_ids = (uint8_t*)AscendC::GmAlloc(kNumQueries * sizeof(int64_t));
    uint8_t* input_positions = (uint8_t*)AscendC::GmAlloc(seqSize);
    uint8_t* seq_lens = (uint8_t*)AscendC::GmAlloc(seqSize);
    uint8_t* slot_mapping = (uint8_t*)AscendC::GmAlloc(seqSize);
    uint8_t* block_tables = (uint8_t*)AscendC::GmAlloc(tableSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16781184);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(AdvanceStepTilingDataTest));

    const int64_t inputTokensHost[kNumSeqs] = {0};
    const int64_t sampledHost[kNumQueries] = {5, 6};
    const int64_t positionsHost[kNumSeqs] = {0};
    const int64_t seqLensHost[kNumSeqs] = {0, 4, 0};
    const int64_t slotHost[kNumSeqs] = {-1, -1, -1};
    const int64_t tablesHost[kNumSeqs * kTableStride] = {7, 99, 11, 42, 21, 77};
    memcpy(input_tokens, inputTokensHost, seqSize);
    memcpy(sampled_token_ids, sampledHost, kNumQueries * sizeof(int64_t));
    memcpy(input_positions, positionsHost, seqSize);
    memcpy(seq_lens, seqLensHost, seqSize);
    memcpy(slot_mapping, slotHost, seqSize);
    memcpy(block_tables, tablesHost, tableSize);

    AdvanceStepTilingDataTest* tilingData = reinterpret_cast<AdvanceStepTilingDataTest*>(tiling);
    tilingData->needCoreNum = kNeedCoreNum;
    tilingData->blockTablesStride = kTableStride;
    tilingData->numSeqs = kNumSeqs;
    tilingData->numQueries = kNumQueries;
    tilingData->blockSize = kBlockSize;
    // CPU 仿真单核（blockIdx=0）覆盖全部 query 行：perCoreSeqs 按单核语义取 numQueries
    tilingData->perCoreSeqs = kNumQueries;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(advance_step, 1, input_tokens, sampled_token_ids, input_positions, seq_lens, slot_mapping, block_tables,
                nullptr, nullptr, input_tokens, input_positions, seq_lens, slot_mapping, workspace,
                (uint8_t*)tilingData);

    const int64_t expectedSlot[kNumSeqs] = {28, 168, -1};
    EXPECT_EQ(0, memcmp(slot_mapping, expectedSlot, seqSize));

    AscendC::GmFree((void*)input_tokens);
    AscendC::GmFree((void*)sampled_token_ids);
    AscendC::GmFree((void*)input_positions);
    AscendC::GmFree((void*)seq_lens);
    AscendC::GmFree((void*)slot_mapping);
    AscendC::GmFree((void*)block_tables);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);
}

static void RunLegacyQueryCase(int64_t kNumQueries)
{
    SCOPED_TRACE(kNumQueries);
    const int64_t kNumSeqs = kNumQueries + 85;
    constexpr int64_t kBlockSize = 16;
    constexpr int64_t kTableStride = 8;
    constexpr int64_t kNeedCoreNum = 40;
    const size_t seqSize = kNumSeqs * sizeof(int64_t);
    const size_t querySize = kNumQueries * sizeof(int64_t);
    const size_t tableSize = kNumSeqs * kTableStride * sizeof(int64_t);
    uint8_t* input_tokens = (uint8_t*)AscendC::GmAlloc(seqSize);
    uint8_t* sampled_token_ids = (uint8_t*)AscendC::GmAlloc(querySize);
    uint8_t* input_positions = (uint8_t*)AscendC::GmAlloc(seqSize);
    uint8_t* seq_lens = (uint8_t*)AscendC::GmAlloc(seqSize);
    uint8_t* slot_mapping = (uint8_t*)AscendC::GmAlloc(seqSize);
    uint8_t* block_tables = (uint8_t*)AscendC::GmAlloc(tableSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16781184);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(AdvanceStepTilingDataTest));
    std::vector<int64_t> tokens(kNumSeqs, -7), sampled(kNumQueries), positions(kNumSeqs, -7), lengths(kNumSeqs),
        slots(kNumSeqs, -9), tables(kNumSeqs * kTableStride);
    for (int64_t i = 0; i < kNumSeqs; ++i) {
        lengths[i] = 3 + i % 60;
    }
    for (int64_t i = 0; i < kNumQueries; ++i) {
        sampled[i] = 7000 + i;
    }
    for (int64_t i = 0; i < kNumSeqs * kTableStride; ++i) {
        tables[i] = 1000 + i;
    }
    memcpy(input_tokens, tokens.data(), seqSize);
    memcpy(sampled_token_ids, sampled.data(), querySize);
    memcpy(input_positions, positions.data(), seqSize);
    memcpy(seq_lens, lengths.data(), seqSize);
    memcpy(slot_mapping, slots.data(), seqSize);
    memcpy(block_tables, tables.data(), tableSize);
    auto* tilingData = reinterpret_cast<AdvanceStepTilingDataTest*>(tiling);
    tilingData->needCoreNum = kNeedCoreNum;
    tilingData->blockTablesStride = kTableStride;
    tilingData->numSeqs = kNumSeqs;
    tilingData->numQueries = kNumQueries;
    tilingData->blockSize = kBlockSize;
    // CPU 仿真单核（blockIdx=0）覆盖全部 query 行：perCoreSeqs 按单核语义取 numQueries
    tilingData->perCoreSeqs = kNumQueries;
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(advance_step, 1, input_tokens, sampled_token_ids, input_positions, seq_lens, slot_mapping, block_tables,
                nullptr, nullptr, input_tokens, input_positions, seq_lens, slot_mapping, workspace,
                (uint8_t*)tilingData);
    for (int64_t i = 0; i < kNumQueries; ++i) {
        EXPECT_EQ(reinterpret_cast<int64_t*>(input_tokens)[i], sampled[i]);
        EXPECT_EQ(reinterpret_cast<int64_t*>(seq_lens)[i], lengths[i] + 1);
        EXPECT_EQ(reinterpret_cast<int64_t*>(input_positions)[i], lengths[i]);
        int64_t offset = lengths[i];
        EXPECT_EQ(reinterpret_cast<int64_t*>(slot_mapping)[i],
                  tables[i * kTableStride + offset / kBlockSize] * kBlockSize + offset % kBlockSize);
    }
    for (int64_t i = kNumQueries; i < kNumSeqs; ++i) {
        EXPECT_EQ(reinterpret_cast<int64_t*>(input_tokens)[i], 0);
        EXPECT_EQ(reinterpret_cast<int64_t*>(input_positions)[i], 0);
        EXPECT_EQ(reinterpret_cast<int64_t*>(slot_mapping)[i], -1);
        EXPECT_EQ(reinterpret_cast<int64_t*>(seq_lens)[i], lengths[i]);
    }
    AscendC::GmFree(input_tokens);
    AscendC::GmFree(sampled_token_ids);
    AscendC::GmFree(input_positions);
    AscendC::GmFree(seq_lens);
    AscendC::GmFree(slot_mapping);
    AscendC::GmFree(block_tables);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(advance_step_test, test_advance_step_all_queries_and_padding)
{
    for (int64_t numQueries : {1, 2, 39, 40, 41, 80, 100, 101, 257}) {
        RunLegacyQueryCase(numQueries);
    }
}
