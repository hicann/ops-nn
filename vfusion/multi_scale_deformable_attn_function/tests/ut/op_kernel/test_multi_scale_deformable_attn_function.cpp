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
#include "multi_scale_deformable_attn_function_tiling_def.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void multi_scale_deformable_attn_function(GM_ADDR value_gm, GM_ADDR spatial_shapes_gm,
                                                                            GM_ADDR level_start_index_gm, GM_ADDR sampling_loc_gm,
                                                                            GM_ADDR attn_weight_gm, GM_ADDR output_gm,
                                                                            GM_ADDR workspace, GM_ADDR tiling_data);

class multi_scale_deformable_attn_function_test : public testing::Test {
    protected:
    static void SetUpTestCase() {
        cout << "multi_scale_deformable_attn_function_test SetUp\n" << endl;
    }
    static void TearDownTestCase() {
        cout << "multi_scale_deformable_attn_function_test TearDown\n" << endl;
    }
};

TEST_F(multi_scale_deformable_attn_function_test, msda_test_high_perf) {
    uint64_t batch_size = 1;
    uint64_t spatial_size = 24;
    uint64_t num_heads = 8;
    uint64_t channels = 64;
    uint64_t num_levels = 1;
    uint64_t num_query = 100;
    uint64_t num_point = 2;

    // inputs
    size_t value_size = batch_size * spatial_size * num_heads * channels * sizeof(float);
    size_t value_spatial_shape_size = num_levels * 2 * sizeof(int32_t);
    size_t level_start_index_size = num_levels * sizeof(int32_t);
    size_t sample_loc_size = batch_size * num_query * num_heads * num_levels * num_point * 2 * sizeof(float);
    size_t attention_weight_size = batch_size * num_query * num_heads * num_levels * num_point * sizeof(float);
    size_t output_size = batch_size * num_query * num_heads * channels * sizeof(float);
    size_t tiling_data_size = sizeof(MultiScaleDeformableAttnFunctionTilingData);

    uint8_t* value = (uint8_t*)AscendC::GmAlloc(value_size + 32);
    uint8_t* value_spatial_shape = (uint8_t*)AscendC::GmAlloc(value_spatial_shape_size + 32);
    uint8_t* level_start_index = (uint8_t*)AscendC::GmAlloc(level_start_index_size + 32);
    uint8_t* sample_loc = (uint8_t*)AscendC::GmAlloc(sample_loc_size + 32);
    uint8_t* attention_weight = (uint8_t*)AscendC::GmAlloc(attention_weight_size + 32);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(output_size + 32);

    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(1024 * 16 * 1024);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tiling_data_size + 32);
    uint32_t blockDim = 48;

    memset(workspace, 0, 16 * 1024 * 1024);
    system("cp -rf ../../../../vfusion/multi_scale_deformable_attn_function/tests/ut/op_kernel/multi_scale_deformable_attn_function_data/ ./");
    system("chmod -R 755 ./multi_scale_deformable_attn_function_data/");
    system("cd ./multi_scale_deformable_attn_function_data/ && rm -rf ./*bin");
    // bs, num_heads, channels, num_levels, num_points, num_queries
    system("cd ./multi_scale_deformable_attn_function_data/ && python3 gen_data.py 1 8 64 1 2 100");
    system("cd ./multi_scale_deformable_attn_function_data/ && python3 gen_tiling.py 1 8 64 1 2 100");

    char * path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/multi_scale_deformable_attn_function_data/value.bin", value_size, value, value_size);
    ReadFile(path + "/multi_scale_deformable_attn_function_data/value_spatial_shape.bin", value_spatial_shape_size, value_spatial_shape, value_spatial_shape_size);
    ReadFile(path + "/multi_scale_deformable_attn_function_data/level_start_index.bin", level_start_index_size, level_start_index, level_start_index_size);
    ReadFile(path + "/multi_scale_deformable_attn_function_data/sample_loc.bin", sample_loc_size, sample_loc, sample_loc_size);
    ReadFile(path + "/multi_scale_deformable_attn_function_data/attention_weight.bin", attention_weight_size, attention_weight, attention_weight_size);
    ReadFile(path + "/multi_scale_deformable_attn_function_data/output.bin", output_size, output, output_size);
    std::cout<<"tiling_data_size:"<<tiling_data_size<<std::endl;
    ReadFile(path + "/multi_scale_deformable_attn_function_data/tiling.bin", tiling_data_size, tiling, tiling_data_size);

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    std::cout<<"ICPU_RUN_KF start:"<<tiling<<std::endl;
    ICPU_RUN_KF(multi_scale_deformable_attn_function, blockDim, value, value_spatial_shape, level_start_index,
        sample_loc, attention_weight, output, workspace, tiling);
    std::cout<<"ICPU_RUN_KF Finsh:"<<tiling<<std::endl;

    AscendC::GmFree(value);
    AscendC::GmFree(value_spatial_shape);
    AscendC::GmFree(level_start_index);
    AscendC::GmFree(sample_loc);
    AscendC::GmFree(attention_weight);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    free(path_);
}
