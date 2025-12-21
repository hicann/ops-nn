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
#include <gtest/gtest.h>

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "fused_cross_entropy_loss_with_max_sum_tiling_def.h"
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void fused_cross_entropy_loss_with_max_sum(GM_ADDR logits_max, GM_ADDR sum_exp_logits,
                                GM_ADDR predicted_logits, GM_ADDR input, GM_ADDR weight,GM_ADDR vocab_parallel_logits, 
                                GM_ADDR loss, GM_ADDR softmax_logits, GM_ADDR workspace, GM_ADDR tiling);

class fused_cross_entropy_loss_with_max_sum_test : public testing::Test {
    protected:

    static void SetUpTestCase() {
        cout << "fused_cross_entropy_loss_with_max_sum_test SetUp\n" << endl;
    }
    static void TearDownTestCase() {
        cout << "fused_cross_entropy_loss_with_max_sum_test TearDown\n" << endl;
    }
};

TEST_F(fused_cross_entropy_loss_with_max_sum_test, params_fp32) 
{
    size_t vocab_parallel_logits_ = 1024 * 4096 * sizeof(float);
    size_t logits_max_ = 1024 * sizeof(float);
    size_t sum_exp_logits_ = 1024 * sizeof(float);
    size_t predicted_logits_ = 1024 * sizeof(float);
    size_t input_ = 1024 * sizeof(float);
    size_t weight_ = 1024 * sizeof(float);
    size_t softmax_logits_ = 1024 *  4096 * sizeof(float);
    size_t loss_ = 1024 *sizeof(float);
    size_t tiling_ = sizeof(FusedCrossEntropyLossWithMaxSumTilingData);

    uint8_t *vocab_parallel_logits = (uint8_t *)AscendC::GmAlloc(vocab_parallel_logits_);
    uint8_t *logits_max = (uint8_t *)AscendC::GmAlloc(logits_max_);
    uint8_t *sum_exp_logits = (uint8_t *)AscendC::GmAlloc(sum_exp_logits_);
    uint8_t *predicted_logits = (uint8_t *)AscendC::GmAlloc(predicted_logits_);
    uint8_t *input = (uint8_t *)AscendC::GmAlloc(input_);
    uint8_t *weight = (uint8_t *)AscendC::GmAlloc(weight_);
    uint8_t *softmax_logits = (uint8_t *)AscendC::GmAlloc(softmax_logits_);
    uint8_t *loss = (uint8_t *)AscendC::GmAlloc(loss_);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024);

    memset(workspace, 0, 16 * 1024 * 1024);

    system("cp -r ../../../../loss/fused_cross_entropy_loss_with_max_sum/tests/ut/op_kernel/gen_data ./");
    system("chmod -R 755 ./gen_data/");
    system("cd ./gen_data/ && rm -rf ./*bin");
    system("cd ./gen_data/ && python3 gen_data.py 1024 4096 1024");

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/gen_data/vocab_parallel_logits.bin", vocab_parallel_logits_, vocab_parallel_logits, vocab_parallel_logits_);
    ReadFile(path + "/gen_data/logits_max.bin", logits_max_, logits_max, logits_max_);
    ReadFile(path + "/gen_data/sum_exp_logits.bin", sum_exp_logits_, sum_exp_logits, sum_exp_logits_);
    ReadFile(path + "/gen_data/predicted_logits.bin", predicted_logits_, predicted_logits, predicted_logits_);
    ReadFile(path + "/gen_data/input.bin", input_, input, input_);
    ReadFile(path + "/gen_data/weight.bin", weight_, weight, weight_);
    ReadFile(path + "/gen_data/tiling.bin", tiling_, tiling, tiling_);

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(fused_cross_entropy_loss_with_max_sum, 48, logits_max, sum_exp_logits, predicted_logits, input, weight, vocab_parallel_logits, loss, softmax_logits,  workspace, tiling);

    AscendC::GmFree(vocab_parallel_logits);
    AscendC::GmFree(logits_max);
    AscendC::GmFree(sum_exp_logits);
    AscendC::GmFree(predicted_logits);
    AscendC::GmFree(input);
    AscendC::GmFree(weight);
    AscendC::GmFree(softmax_logits);
    AscendC::GmFree(loss);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(fused_cross_entropy_loss_with_max_sum_test, params_bf16) 
{
    size_t vocab_parallel_logits_ = 1024 * 4096 * sizeof(half);
    size_t logits_max_ = 1024 * sizeof(float);
    size_t sum_exp_logits_ = 1024 * sizeof(float);
    size_t predicted_logits_ = 1024 * sizeof(float);
    size_t input_ = 1024 * sizeof(float);
    size_t weight_ = 1024 * sizeof(float);
    size_t softmax_logits_ = 1024 *  4096 * sizeof(float);
    size_t loss_ = 1024 *sizeof(float);
    size_t tiling_ = sizeof(FusedCrossEntropyLossWithMaxSumTilingData);

    uint8_t *vocab_parallel_logits = (uint8_t *)AscendC::GmAlloc(vocab_parallel_logits_);
    uint8_t *logits_max = (uint8_t *)AscendC::GmAlloc(logits_max_);
    uint8_t *sum_exp_logits = (uint8_t *)AscendC::GmAlloc(sum_exp_logits_);
    uint8_t *predicted_logits = (uint8_t *)AscendC::GmAlloc(predicted_logits_);
    uint8_t *input = (uint8_t *)AscendC::GmAlloc(input_);
    uint8_t *weight = (uint8_t *)AscendC::GmAlloc(weight_);
    uint8_t *softmax_logits = (uint8_t *)AscendC::GmAlloc(softmax_logits_);
    uint8_t *loss = (uint8_t *)AscendC::GmAlloc(loss_);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024);

    memset(workspace, 0, 16 * 1024 * 1024);

    system("cp -r ../../../../loss/fused_cross_entropy_loss_with_max_sum/tests/ut/op_kernel/gen_data ./");
    system("chmod -R 755 ./gen_data/");
    system("cd ./gen_data/ && rm -rf ./*bin");
    system("cd ./gen_data/ && python3 gen_data_bf16.py 1024 4096 1024");

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/gen_data/vocab_parallel_logits.bin", vocab_parallel_logits_, vocab_parallel_logits, vocab_parallel_logits_);
    ReadFile(path + "/gen_data/logits_max.bin", logits_max_, logits_max, logits_max_);
    ReadFile(path + "/gen_data/sum_exp_logits.bin", sum_exp_logits_, sum_exp_logits, sum_exp_logits_);
    ReadFile(path + "/gen_data/predicted_logits.bin", predicted_logits_, predicted_logits, predicted_logits_);
    ReadFile(path + "/gen_data/input.bin", input_, input, input_);
    ReadFile(path + "/gen_data/weight.bin", weight_, weight, weight_);
    ReadFile(path + "/gen_data/tiling.bin", tiling_, tiling, tiling_);

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(fused_cross_entropy_loss_with_max_sum, 48, logits_max, sum_exp_logits, predicted_logits, input, weight, vocab_parallel_logits, loss, softmax_logits,  workspace, tiling);

    AscendC::GmFree(vocab_parallel_logits);
    AscendC::GmFree(logits_max);
    AscendC::GmFree(sum_exp_logits);
    AscendC::GmFree(predicted_logits);
    AscendC::GmFree(input);
    AscendC::GmFree(weight);
    AscendC::GmFree(softmax_logits);
    AscendC::GmFree(loss);
    AscendC::GmFree(tiling);
    free(path_);
}



TEST_F(fused_cross_entropy_loss_with_max_sum_test, params_fp16) 
{
    size_t vocab_parallel_logits_ = 1024 * 4096 * sizeof(half);
    size_t logits_max_ = 1024 * sizeof(float);
    size_t sum_exp_logits_ = 1024 * sizeof(float);
    size_t predicted_logits_ = 1024 * sizeof(float);
    size_t input_ = 1024 * sizeof(float);
    size_t weight_ = 1024 * sizeof(float);
    size_t softmax_logits_ = 1024 *  4096 * sizeof(float);
    size_t loss_ = 1024 *sizeof(float);
    size_t tiling_ = sizeof(FusedCrossEntropyLossWithMaxSumTilingData);

    uint8_t *vocab_parallel_logits = (uint8_t *)AscendC::GmAlloc(vocab_parallel_logits_);
    uint8_t *logits_max = (uint8_t *)AscendC::GmAlloc(logits_max_);
    uint8_t *sum_exp_logits = (uint8_t *)AscendC::GmAlloc(sum_exp_logits_);
    uint8_t *predicted_logits = (uint8_t *)AscendC::GmAlloc(predicted_logits_);
    uint8_t *input = (uint8_t *)AscendC::GmAlloc(input_);
    uint8_t *weight = (uint8_t *)AscendC::GmAlloc(weight_);
    uint8_t *softmax_logits = (uint8_t *)AscendC::GmAlloc(softmax_logits_);
    uint8_t *loss = (uint8_t *)AscendC::GmAlloc(loss_);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024);

    memset(workspace, 0, 16 * 1024 * 1024);

    system("cp -r ../../../../loss/fused_cross_entropy_loss_with_max_sum/tests/ut/op_kernel/gen_data ./");
    system("chmod -R 755 ./gen_data/");
    system("cd ./gen_data/ && rm -rf ./*bin");
    system("cd ./gen_data/ && python3 gen_data_fp16.py 1024 4096 1024");

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/gen_data/vocab_parallel_logits.bin", vocab_parallel_logits_, vocab_parallel_logits, vocab_parallel_logits_);
    ReadFile(path + "/gen_data/logits_max.bin", logits_max_, logits_max, logits_max_);
    ReadFile(path + "/gen_data/sum_exp_logits.bin", sum_exp_logits_, sum_exp_logits, sum_exp_logits_);
    ReadFile(path + "/gen_data/predicted_logits.bin", predicted_logits_, predicted_logits, predicted_logits_);
    ReadFile(path + "/gen_data/input.bin", input_, input, input_);
    ReadFile(path + "/gen_data/weight.bin", weight_, weight, weight_);
    ReadFile(path + "/gen_data/tiling.bin", tiling_, tiling, tiling_);

    ICPU_SET_TILING_KEY(2);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(fused_cross_entropy_loss_with_max_sum, 48, logits_max, sum_exp_logits, predicted_logits, input, weight, vocab_parallel_logits, loss, softmax_logits,  workspace, tiling);

    AscendC::GmFree(vocab_parallel_logits);
    AscendC::GmFree(logits_max);
    AscendC::GmFree(sum_exp_logits);
    AscendC::GmFree(predicted_logits);
    AscendC::GmFree(input);
    AscendC::GmFree(weight);
    AscendC::GmFree(softmax_logits);
    AscendC::GmFree(loss);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(fused_cross_entropy_loss_with_max_sum_test, params_nullptr) 
{
    size_t vocab_parallel_logits_ = 1024 * 4096 * sizeof(half);
    size_t logits_max_ = 1024 * sizeof(float);
    size_t sum_exp_logits_ = 1024 * sizeof(float);
    size_t predicted_logits_ = 1024 * sizeof(float);
    size_t input_ = 1024 * sizeof(float);
    size_t weight_ = 1024 * sizeof(float);
    size_t softmax_logits_ = 1024 *  4096 * sizeof(float);
    size_t loss_ = 1024 *sizeof(float);
    size_t tiling_ = sizeof(FusedCrossEntropyLossWithMaxSumTilingData);

    uint8_t *vocab_parallel_logits = (uint8_t *)AscendC::GmAlloc(vocab_parallel_logits_);
    uint8_t *logits_max = (uint8_t *)AscendC::GmAlloc(logits_max_);
    uint8_t *sum_exp_logits = (uint8_t *)AscendC::GmAlloc(sum_exp_logits_);
    uint8_t *predicted_logits = (uint8_t *)AscendC::GmAlloc(predicted_logits_);
    uint8_t *input = (uint8_t *)AscendC::GmAlloc(input_);
    uint8_t *weight = (uint8_t *)AscendC::GmAlloc(weight_);
    uint8_t *softmax_logits = (uint8_t *)AscendC::GmAlloc(softmax_logits_);
    uint8_t *loss = (uint8_t *)AscendC::GmAlloc(loss_);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024);

    memset(workspace, 0, 16 * 1024 * 1024);

    system("cp -r ../../../../loss/fused_cross_entropy_loss_with_max_sum/tests/ut/op_kernel/gen_data ./");
    system("chmod -R 755 ./gen_data/");
    system("cd ./gen_data/ && rm -rf ./*bin");
    system("cd ./gen_data/ && python3 gen_data_fp16.py 1024 4096 1024");

    char *path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/gen_data/vocab_parallel_logits.bin", vocab_parallel_logits_, vocab_parallel_logits, vocab_parallel_logits_);
    ReadFile(path + "/gen_data/logits_max.bin", logits_max_, logits_max, logits_max_);
    ReadFile(path + "/gen_data/sum_exp_logits.bin", sum_exp_logits_, sum_exp_logits, sum_exp_logits_);
    ReadFile(path + "/gen_data/predicted_logits.bin", predicted_logits_, predicted_logits, predicted_logits_);
    ReadFile(path + "/gen_data/input.bin", input_, input, input_);
    ReadFile(path + "/gen_data/weight.bin", weight_, weight, weight_);
    ReadFile(path + "/gen_data/tiling.bin", tiling_, tiling, tiling_);

    ICPU_SET_TILING_KEY(3);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(fused_cross_entropy_loss_with_max_sum, 0, logits_max, sum_exp_logits, predicted_logits, input, weight, vocab_parallel_logits, loss, softmax_logits,  workspace, tiling);

    AscendC::GmFree(vocab_parallel_logits);
    AscendC::GmFree(logits_max);
    AscendC::GmFree(sum_exp_logits);
    AscendC::GmFree(predicted_logits);
    AscendC::GmFree(input);
    AscendC::GmFree(weight);
    AscendC::GmFree(softmax_logits);
    AscendC::GmFree(loss);
    AscendC::GmFree(tiling);
    free(path_);
}