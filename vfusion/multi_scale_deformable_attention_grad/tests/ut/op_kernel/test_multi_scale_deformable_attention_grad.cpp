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
#include "gtest/gtest.h"
#include "multi_scale_deformable_attention_grad_tiling_def.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void multi_scale_deformable_attention_grad(GM_ADDR value_gm, GM_ADDR spatial_shapes_gm,
                                                                            GM_ADDR level_start_index_gm, GM_ADDR sampling_loc_gm,
                                                                            GM_ADDR attn_weight_gm, GM_ADDR grad_output_gm,
                                                                            GM_ADDR grad_value_gm, GM_ADDR grad_sampling_loc_gm,
                                                                            GM_ADDR grad_attn_weight_gm, GM_ADDR workspace,
                                                                            GM_ADDR tiling_data);

class multi_scale_deformable_attention_grad_test : public testing::Test {
    protected:
    static void SetUpTestCase() {
        cout << "multi_scale_deformable_attention_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase() {
        cout << "multi_scale_deformable_attention_grad_test TearDown\n" << endl;
    }
};

TEST_F(multi_scale_deformable_attention_grad_test, test_case_fp32) {
    uint64_t batch_size = 1;
    uint64_t spatial_size = 24;
    uint64_t num_heads = 8;
    uint64_t channels = 8;
    uint64_t num_levels = 1;
    uint64_t num_query = 100;
    uint64_t num_point = 2;

    // inputs
    size_t value_size = batch_size * spatial_size * num_heads * channels * sizeof(float);
    size_t value_spatial_shape_size = num_levels * 2 * sizeof(int32_t);
    size_t level_start_index_size = num_levels * sizeof(int32_t);
    size_t sample_loc_size = batch_size * num_query * num_heads * num_levels * num_point * 2 * sizeof(float);
    size_t attention_weight_size = batch_size * num_query * num_heads * num_levels * num_point * sizeof(float);
    size_t grad_output_size = batch_size * num_query * num_heads * channels * sizeof(float);
    size_t grad_value_size = batch_size * spatial_size * num_heads * channels * sizeof(float);
    size_t grad_sample_loc_size = sample_loc_size;
    size_t grad_attn_weight_size = attention_weight_size;
    size_t tiling_data_size = sizeof(MultiScaleDeformableAttentionGradTilingData);

    uint8_t* value = (uint8_t*)AscendC::GmAlloc(value_size);
    uint8_t* value_spatial_shape = (uint8_t*)AscendC::GmAlloc(value_spatial_shape_size);
    uint8_t* level_start_index = (uint8_t*)AscendC::GmAlloc(level_start_index_size);
    uint8_t* sample_loc = (uint8_t*)AscendC::GmAlloc(sample_loc_size);
    uint8_t* attention_weight = (uint8_t*)AscendC::GmAlloc(attention_weight_size);
    uint8_t* grad_output = (uint8_t*)AscendC::GmAlloc(grad_output_size);
    uint8_t* grad_value = (uint8_t*)AscendC::GmAlloc(grad_value_size);
    uint8_t* grad_sample_loc = (uint8_t*)AscendC::GmAlloc(grad_sample_loc_size);
    uint8_t* grad_attn_weight = (uint8_t*)AscendC::GmAlloc(grad_attn_weight_size);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size + 16);
    uint32_t blockDim = 48;

    memset(workspace, 0, 16 * 1024 * 1024);

    system("cp -rf ../../../../vfusion/multi_scale_deformable_attention_grad/tests/ut/op_kernel/multi_scale_deformable_attention_grad_data/ ./");
    system("chmod -R 755 ./multi_scale_deformable_attention_grad_data/");
    system("cd ./multi_scale_deformable_attention_grad_data/ && rm -rf ./*bin");
    system("cd ./multi_scale_deformable_attention_grad_data/ && python3 gen_data.py 1 8 8 1 2 100");
    system("cd ./multi_scale_deformable_attention_grad_data/ && python3 gen_tiling.py 1 8 8 1 2 100");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/multi_scale_deformable_attention_grad_data/value.bin", value_size, value, value_size);
    ReadFile(path + "/multi_scale_deformable_attention_grad_data/value_spatial_shape.bin", value_spatial_shape_size, value_spatial_shape, value_spatial_shape_size);
    ReadFile(path + "/multi_scale_deformable_attention_grad_data/level_start_index.bin", level_start_index_size, level_start_index, level_start_index_size);
    ReadFile(path + "/multi_scale_deformable_attention_grad_data/sample_loc.bin", sample_loc_size, sample_loc, sample_loc_size);
    ReadFile(path + "/multi_scale_deformable_attention_grad_data/attention_weight.bin", attention_weight_size, attention_weight, attention_weight_size);
    ReadFile(path + "/multi_scale_deformable_attention_grad_data/grad_output.bin", grad_output_size, grad_output, grad_output_size);
    ReadFile(path + "/multi_scale_deformable_attention_grad_data/grad_value.bin", grad_value_size, grad_value, grad_value_size);
    ReadFile(path + "/multi_scale_deformable_attention_grad_data/grad_sample_loc.bin", grad_sample_loc_size, grad_sample_loc, grad_sample_loc_size);
    ReadFile(path + "/multi_scale_deformable_attention_grad_data/grad_attn_weight.bin", grad_attn_weight_size, grad_attn_weight, grad_attn_weight_size);
    ReadFile(path + "/multi_scale_deformable_attention_grad_data/tiling.bin", tiling_data_size, tiling, tiling_data_size + 16);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(multi_scale_deformable_attention_grad, blockDim, value, value_spatial_shape, level_start_index,
        sample_loc, attention_weight, grad_output, grad_value, grad_sample_loc, grad_attn_weight, workspace, tiling);

    AscendC::GmFree(value);
    AscendC::GmFree(value_spatial_shape);
    AscendC::GmFree(level_start_index);
    AscendC::GmFree(sample_loc);
    AscendC::GmFree(attention_weight);
    AscendC::GmFree(grad_output);
    AscendC::GmFree(grad_value);
    AscendC::GmFree(grad_sample_loc);
    AscendC::GmFree(grad_attn_weight);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    free(path_);
}