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
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "test_advance_step_tiling_def.h"

using namespace std;

extern "C" __global__ __aicore__ void advance_step(
    GM_ADDR input_tokens, GM_ADDR sampled_token_ids, GM_ADDR input_positions, GM_ADDR seq_lens, GM_ADDR slot_mapping,
    GM_ADDR block_tables, GM_ADDR spec_token, GM_ADDR accepted_num, GM_ADDR workspace, GM_ADDR tiling);
class advance_step_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "advance_step SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "advance_step TearDown\n" << endl;
    }
};

TEST_F(advance_step_test, test_advance_step_int_0)
{
    system(
        "cp -rf "
        "../../../../optim/advance_step/tests/ut/op_kernel/advance_step_data ./");
        
    system("chmod -R 755 ./advance_step_data/");
    system("cd ./advance_step_data/ && python3 gen_data.py '(16, 1)' '(8, 1)' 'int64'");
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

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(
        advance_step, blockDim, input_tokens, sampled_token_ids, input_positions, seq_lens, slot_mapping, block_tables,
         nullptr, nullptr, workspace, (uint8_t*)tilingDatafromBin);
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
    system("cd ./advance_step_data/ && python3 compare_data.py 'int64'");
}