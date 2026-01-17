/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "convolutionbackward.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "runtime/context.h"

using namespace op;

namespace l0op {
const int64_t CONV3D_DIM = 5;
const int64_t PAD_DIM_4 = 4;
const int64_t DIM_0 = 0;
const int64_t DIM_1 = 1;
const int64_t DIM_2 = 2;
const int64_t DIM_3 = 3;
const size_t CONV3D_BACKPROP_WHITE_LIST_CASE_SIZE = 26;
const int64_t D_DIM_NCDHW_INDEX = 2;
const int64_t H_DIM_NCDHW_INDEX = 3;
const int64_t W_DIM_NCDHW_INDEX = 4;
constexpr int64_t C1_DIM_NDC1HWC0_INDEX = 2;
constexpr int64_t C0_DIM_NDC1HWC0_INDEX = 5;
constexpr uint32_t BASIC_BLOCK_SIZE_128 = 128;
constexpr int32_t KERNEL_HW_4 = 4;
constexpr int32_t KERNEL_HW_9 = 9;
constexpr int32_t KERNEL_HW_16 = 16;
constexpr int32_t FORMAT_6HD_DIM = 6;
constexpr uint64_t DETERMINISTIC_FILTER_V2_STRIDE_MAX_VALUE = 7;
constexpr uint64_t DETERMINISTIC_PADDING_COEF = 2;
constexpr uint64_t DETERMINISTIC_CHANNEL_16 = 16;

const vector<vector<int64_t>> CONV2D_BACKPROP_INPUT_CAST_WHITE_LIST =
{
  {
    DataType::DT_BF16,  // input data type
    7, 4096, 1, 1,        // input shape
    64, 1024, 1, 2,        // filter shape
    7, 64, 1, 12,        // outBackprop shape
    1, 5,                  // stride
    0, 50,                  // padding
    1, 45,                  // dilation
    4                      // groups
  },
};

const vector<vector<int64_t>> CONV2D_BACKPROP_INPUT_V2_WHITE_LIST =
{
  {
    DataType::DT_FLOAT,  // input data type
    4, 320, 80, 80,        // input shape
    320, 320, 3, 3,        // filter shape
    4, 320, 80, 80,        // outBackprop shape
    1, 1,                  // stride
    1, 1,                  // padding
    1, 1,                  // dilation
    1                      // groups
  },
  {
    DataType::DT_FLOAT,  // input data type
    3, 2, 3, 3,        // input shape
    2, 2, 1, 1,        // filter shape
    3, 2, 3, 3,        // outBackprop shape
    1, 1,                  // stride
    0, 0,                  // padding
    1, 1,                  // dilation
    1                      // groups
  },
  {
    DataType::DT_FLOAT,  // input data type
    1, 4, 1, 8,        // input shape
    2, 4, 1, 4,        // filter shape
    1, 2, 1, 5,        // outBackprop shape
    1, 1,                  // stride
    0, 0,                  // padding
    1, 1,                  // dilation
    1                      // groups
  },
  {
    DataType::DT_FLOAT,  // input data type
    1, 4, 1, 9,        // input shape
    2, 4, 1, 4,        // filter shape
    1, 2, 1, 8,        // outBackprop shape
    1, 1,                  // stride
    1, 1,                  // padding
    1, 1,                  // dilation
    1                      // groups
  },
  {
    DataType::DT_FLOAT,  // input data type
    4, 4, 4, 4,        // input shape
    8, 4, 3, 3,        // filter shape
    4, 8, 1, 1,        // outBackprop shape
    3, 3,                  // stride
    0, 0,                  // padding
    1, 1,                  // dilation
    1                      // groups
  }
};

const vector<vector<int64_t>> CONV2D_BACKPROP_FILTER_V3_WHITE_LIST =
{
  {
    DataType::DT_FLOAT16,  // input data type
    1, 640, 104, 152,      // input shape
    640, 640, 3, 3,        // filter shape
    1, 640, 104, 152,      // outBackprop shape
    1, 1,                  // stride
    1, 1,                  // padding
    1, 1,                  // dilation
    1                      // groups
  },
  {
    DataType::DT_FLOAT,  // input data type
    4, 320, 80, 80,        // input shape
    320, 320, 3, 3,        // filter shape
    4, 320, 80, 80,        // outBackprop shape
    1, 1,                  // stride
    1, 1,                  // padding
    1, 1,                  // dilation
    1                      // groups
  }
};

const vector<vector<int64_t>> CONV3D_BACKPROP_INPUT_V2_WHITE_LIST =
{
  // ID_2
  {
    DataType::DT_BF16,  // input data type
    1, 256, 62, 66, 66, // input shape
    128, 256, 3, 3, 3,  // filter shape
    1, 128, 60, 64, 64, // outBackprop shape
    1, 1, 1,            // stide
    0, 0, 0,            // padding
    1, 1, 1,            // dilation
    1                   // groups
  },
  // ID_4
  {
    DataType::DT_BF16,
    1, 256, 60, 64, 64,
    4, 256, 1, 1, 1,
    1, 4, 60, 64, 64,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1
  },
  // ID_6
  {
    DataType::DT_BF16,
    1, 256, 122, 130, 130,
    256, 256, 4, 4, 4,
    1, 256, 60, 64, 64,
    2, 2, 2,
    0, 0, 0,
    1, 1, 1,
    1
  },
  {
    DataType::DT_BF16,   // input data type
    4, 3, 17, 229, 229,  // input shape
    64, 3, 7, 7, 7,      // filter shape
    4, 64, 6, 112, 112,  // outBackprop shape
    2, 2, 2,             // stide
    0, 0, 0,             // padding
    1, 1, 1,             // dilation
    1                    // groups
  },
  // 15
  {
    DataType::DT_BF16,
    1, 4, 5, 32, 32,
    4, 4, 1, 1, 1,
    1, 4, 5, 32, 32,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1
  },
  // 16
  {
    DataType::DT_BF16,  // input data type
    1, 4, 7, 32, 32,    // input shape
    512, 4, 3, 3, 3,    // filter shape
    1, 512, 5, 32, 32,  // outBackprop shape
    1, 1, 1,            // stide
    0, 1, 1,            // padding
    1, 1, 1,            // dilation
    1                   // groups
  },
  // 17
  {
    DataType::DT_BF16,
    1, 512, 5, 32, 32,
    512, 512, 1, 1, 1,
    1, 512, 5, 32, 32,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1
  },
  // 18
  {
    DataType::DT_BF16,
    1, 512, 5, 65, 65,
    512, 512, 1, 3, 3,
    1, 512, 5, 32, 32,
    1, 2, 2,
    0, 0, 0,
    1, 1, 1,
    1
  },
  //19
  {
    DataType::DT_BF16,
    1, 512, 7, 32, 32,
    512, 512, 3, 3, 3,
    1, 512, 5, 32, 32,
    1, 1, 1,
    0, 1, 1,
    1, 1, 1,
    1
  },
  // 20
  {
    DataType::DT_BF16,
    1, 256, 5, 64, 64,
    512, 256, 1, 1, 1,
    1, 512, 5, 64, 64,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1
  },
  // 21
  {
    DataType::DT_BF16,
    1, 256, 7, 64, 64,
    512, 256, 3, 3, 3,
    1, 512, 5, 64, 64,
    1, 1, 1,
    0, 1, 1,
    1, 1, 1,
    1
  },
  // 22
  {
    DataType::DT_BF16,
    1, 512, 5, 64, 64,
    512, 512, 1, 3, 3,
    1, 512, 5, 64, 64,
    1, 1, 1,
    0, 1, 1,
    1, 1, 1,
    1
  },
  // 23
  {
    DataType::DT_BF16,
    1, 512, 7, 64, 64,
    512, 512, 3, 3, 3,
    1, 512, 5, 64, 64,
    1, 1, 1,
    0, 1, 1,
    1, 1, 1,
    1
  },
  // 25
  {
    DataType::DT_BF16,
    1, 512, 11, 64, 64,
    512, 512, 3, 3, 3,
    1, 512, 9, 64, 64,
    1, 1, 1,
    0, 1, 1,
    1, 1, 1,
    1
  },
  // 26
  {
    DataType::DT_BF16,
    1, 512, 7, 32, 32,
    8, 512, 3, 3, 3,
    1, 8, 5, 32, 32,
    1, 1, 1,
    0, 1, 1,
    1, 1, 1,
    1
  },
  // 27
  {
    DataType::DT_BF16,  // input data type
    1, 8, 5, 32, 32,    // input shape
    8, 8, 1, 1, 1,      // filter shape
    1, 8, 5, 32, 32,    // outBackprop shape
    1, 1, 1,            // stide
    0, 0, 0,            // padding
    1, 1, 1,            // dilation
    1                   // groups
  },
  {
    DataType::DT_BF16,
    2, 512, 19, 128, 128,
    256, 512, 3, 3, 3,
    2, 256, 17, 128, 128,
    1, 1, 1,
    0, 1, 1,
    1, 1, 1,
    1
  },
  {
    DataType::DT_BF16,
    2, 4, 5, 32, 32,
    4, 4, 1, 1, 1,
    2, 4, 5, 32, 32,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1
  },
  {
    DataType::DT_BF16,  // input data type
    2, 4, 7, 32, 32,    // input shape
    512, 4, 3, 3, 3,    // filter shape
    2, 512, 5, 32, 32,  // outBackprop shape
    1, 1, 1,            // stide
    0, 1, 1,            // padding
    1, 1, 1,            // dilation
    1                   // groups
  },
  {
    DataType::DT_BF16,
    2, 512, 5, 32, 32,
    512, 512, 1, 1, 1,
    2, 512, 5, 32, 32,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1
  },
  {
    DataType::DT_BF16,
    2, 512, 5, 65, 65,
    512, 512, 1, 3, 3,
    2, 512, 5, 32, 32,
    1, 2, 2,
    0, 0, 0,
    1, 1, 1,
    1
  },
  {
    DataType::DT_BF16,
    2, 512, 7, 32, 32,
    512, 512, 3, 3, 3,
    2, 512, 5, 32, 32,
    1, 1, 1,
    0, 1, 1,
    1, 1, 1,
    1
  },
  {
    DataType::DT_BF16,
    2, 256, 5, 64, 64,
    512, 256, 1, 1, 1,
    2, 512, 5, 64, 64,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1
  },
  {
    DataType::DT_BF16,
    2, 256, 7, 64, 64,
    512, 256, 3, 3, 3,
    2, 512, 5, 64, 64,
    1, 1, 1,
    0, 1, 1,
    1, 1, 1,
    1
  },
  {
    DataType::DT_BF16,
    2, 512, 5, 64, 64,
    512, 512, 1, 3, 3,
    2, 512, 5, 64, 64,
    1, 1, 1,
    0, 1, 1,
    1, 1, 1,
    1
  },
  {
    DataType::DT_BF16,
    2, 512, 7, 64, 64,
    512, 512, 3, 3, 3,
    2, 512, 5, 64, 64,
    1, 1, 1,
    0, 1, 1,
    1, 1, 1,
    1
  },
  {
    DataType::DT_BF16,
    2, 512, 11, 64, 64,
    512, 512, 3, 3, 3,
    2, 512, 9, 64, 64,
    1, 1, 1,
    0, 1, 1,
    1, 1, 1,
    1
  },
  {
    DataType::DT_BF16,
    2, 512, 7, 32, 32,
    8, 512, 3, 3, 3,
    2, 8, 5, 32, 32,
    1, 1, 1,
    0, 1, 1,
    1, 1, 1,
    1
  },
  {
    DataType::DT_BF16,  // input data type
    2, 8, 5, 32, 32,    // input shape
    8, 8, 1, 1, 1,      // filter shape
    2, 8, 5, 32, 32,    // outBackprop shape
    1, 1, 1,            // stide
    0, 0, 0,            // padding
    1, 1, 1,            // dilation
    1                   // groups
  },
  {
    DataType::DT_BF16,
    16, 128, 20, 64, 64,
    128, 128, 1, 1, 1,
    16, 128, 20, 64, 64,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    16, 128, 22, 66, 66,
    128, 128, 3, 3, 3,
    16, 128, 20, 64, 64,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    16, 256, 20, 32, 32,
    256, 256, 1, 1, 1,
    16, 256, 20, 32, 32,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    16, 682, 20, 32, 32,
    256, 682, 1, 1, 1,
    16, 256, 20, 32, 32,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    16, 256, 20, 32, 32,
    1364, 256, 1, 1, 1,
    16, 1364, 20, 32, 32,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    16, 512, 20, 16, 16,
    512, 512, 1, 1, 1,
    16, 512, 20, 16, 16,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    16, 1365, 20, 16, 16,
    512, 1365, 1, 1, 1,
    16, 512, 20, 16, 16,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    16, 512, 20, 16, 16,
    2730, 512, 1, 1, 1,
    16, 2730, 20, 16, 16,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    16, 512, 10, 16, 16,
    512, 512, 1, 1, 1,
    16, 512, 10, 16, 16,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    16, 512, 12, 18, 18,
    512, 512, 3, 3, 3,
    16, 512, 10, 16, 16,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    16, 512, 5, 16, 16,
    512, 512, 1, 1, 1,
    16, 512, 5, 16, 16,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    16, 512, 7, 18, 18,
    512, 512, 3, 3, 3,
    16, 512, 5, 16, 16,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    16, 1365, 5, 16, 16,
    512, 1365, 1, 1, 1,
    16, 512, 5, 16, 16,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    16, 512, 5, 16, 16,
    2730, 512, 1, 1, 1,
    16, 2730, 5, 16, 16,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    16, 1365,20, 16, 16,
    512, 1365, 1, 1, 1,
    16, 512, 20, 16, 16,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    16, 682, 20, 32, 32,
    256, 682, 1, 1, 1,
    16, 256, 20, 32, 32,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    16, 256, 22, 34, 34,
    256, 256, 3, 3, 3,
    16, 256, 20, 32, 32,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    16, 512, 22, 18, 18,
    512, 512, 3, 3, 3,
    16, 512, 20, 16, 16,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    8, 64, 22, 130, 130,
    3, 64, 3, 3, 3,
    8, 3, 20, 128, 128,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    8, 128, 20, 64, 64,
    128, 128, 1, 1, 1,
    8, 128, 20, 64, 64,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    8, 128, 22, 66, 66,
    128, 128, 3, 3, 3,
    8, 128, 20, 64, 64,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    8, 256, 20, 32, 32,
    256, 256, 1, 1, 1,
    8, 256, 20, 32, 32,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    8, 682, 20, 32, 32,
    256, 682, 1, 1, 1,
    8, 256, 20, 32, 32,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    8, 256, 20, 32, 32,
    1364, 256, 1, 1, 1,
    8, 1364, 20, 32, 32,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    8, 512, 20, 16, 16,
    512, 512, 1, 1, 1,
    8, 512, 20, 16, 16,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    8, 1365, 20, 16, 16,
    512, 1365, 1, 1, 1,
    8, 512, 20, 16, 16,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    8, 512, 20, 16, 16,
    2730, 512, 1, 1, 1,
    8, 2730, 20, 16, 16,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    8, 512, 10, 16, 16,
    512, 512, 1, 1, 1,
    8, 512, 10, 16, 16,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    8, 512, 12, 18, 18,
    512, 512, 3, 3, 3,
    8, 512, 10, 16, 16,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    8, 512, 5, 16, 16,
    512, 512, 1, 1, 1,
    8, 512, 5, 16, 16,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    8, 512, 7, 18, 18,
    512, 512, 3, 3, 3,
    8, 512, 5, 16, 16,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    8, 1365, 5, 16, 16,
    512, 1365, 1, 1, 1,
    8, 512, 5, 16, 16,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    8, 512, 5, 16, 16,
    2730, 512, 1, 1, 1,
    8, 2730, 5, 16, 16,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    1, 240, 6, 62, 62,
    120, 240, 3, 3, 3,
    1, 120, 4, 60, 60,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    1, 240, 4, 60, 60,
    4, 240, 1, 1, 1,
    1, 4, 4, 60, 60,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    1, 240, 6, 62, 62,
    240, 240, 3, 3, 3,
    1, 240, 4, 60, 60,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    1, 128, 4, 60, 60,
    256, 128, 1, 1, 1,
    1, 256, 4, 60, 60,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    1, 256, 6, 62, 62,
    128, 256, 3, 3, 3,
    1, 128, 4, 60, 60,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    1, 4, 4, 60, 60,
    256, 4, 1, 1, 1,
    1, 256, 4, 60, 60,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    1, 256, 4, 60, 60,
    4, 256, 1, 1, 1,
    1, 4, 4, 60, 60,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    1, 256, 6, 62, 62,
    256, 256, 3, 3, 3,
    1, 256, 4, 60, 60,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    1, 128, 60, 64, 64,
    256, 128, 1, 1, 1,
    1, 256, 60, 64, 64,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    1, 256, 62, 66, 66,
    128, 256, 3, 3, 3,
    1, 128, 60, 64, 64,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    1, 4, 60, 64, 64,
    256, 4, 1, 1, 1,
    1, 256, 60, 64, 64,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    1, 256, 60, 64, 64,
    4, 256, 1, 1, 1,
    1, 4, 60, 64, 64,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    1, 256, 62, 66, 66,
    256, 256, 3, 3, 3,
    1, 256, 60, 64, 64,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1,
  },
  {
    DataType::DT_BF16,
    1, 256, 122, 130, 130,
    256, 256, 4, 4, 4,
    1, 256, 60, 64, 64,
    2, 2, 2,
    0, 0, 0,
    1, 1, 1,
    1,
  },
};

const vector<vector<int64_t>> CONV3D_BACKPROP_FILTER_V2_WHITE_LIST =
{
  {
    DataType::DT_BF16,
    1, 128, 19, 256, 256,
    3, 128, 3, 3, 3,
    1, 3, 17, 256, 256,
    1, 1, 1,
    0, 1, 1,
    1, 1, 1,
    1
  },
  {
    DataType::DT_BF16,
    1, 512, 7, 32, 32,
    8, 512, 3, 3, 3,
    1, 8, 5, 32, 32,
    1, 1, 1,
    0, 1, 1,
    1, 1, 1,
    1
  },
  {
    DataType::DT_BF16,
    2, 128, 19, 256, 256,
    3, 128, 3, 3, 3,
    2, 3, 17, 256, 256,
    1, 1, 1,
    0, 1, 1,
    1, 1, 1,
    1
  },
  {
    DataType::DT_BF16,
    16, 64, 22, 130, 130,
    3, 64, 3, 3, 3,
    16, 3, 20, 128, 128,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1
  },
  {
    DataType::DT_BF16,
    16, 64, 22, 130, 130,
    64, 64, 3, 3, 3,
    16, 64, 20, 128, 128,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1
  },
  {
    DataType::DT_BF16,
    16, 3, 26, 134, 134,
    64, 3, 7, 7, 7,
    16, 64, 20, 128, 128,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1
  },
};

const vector<vector<int64_t>> CONV3D_BACKPROP_FILTER_V2_TRANSDATA_MERGE_WHITE_LIST =
{
  {
    DataType::DT_BF16,
    1, 8, 5, 32, 32,
    8, 8, 1, 1, 1,
    1, 8, 5, 32, 32,
    1, 1, 1,
    0, 0, 0,
    1, 1, 1,
    1
  }
};

static void AddAclIntArrayToCaseInfo(const aclIntArray &seg, vector<int64_t> &caseInfo)
{
  size_t len = seg.Size();
  for (size_t i = 0; i < len; i++) {
    caseInfo.push_back(seg[i]);
  }
}

static void AddTensorShapeToCaseInfo(const aclTensor &seg, vector<int64_t> &caseInfo)
{
  auto segShape = seg.GetViewShape();
  size_t dimNum = segShape.GetDimNum();
  for (size_t i=0; i < dimNum; i++) {
    caseInfo.push_back(segShape.GetDim(i));
  }
}

static void ConstructCaseInfo(const ConvBackpropParams &params, vector<int64_t> &caseInfo)
{
  caseInfo.reserve(CONV3D_BACKPROP_WHITE_LIST_CASE_SIZE);
  auto inputDataType = params.input->GetDataType();
  caseInfo.push_back(static_cast<int64_t>(inputDataType));
  AddTensorShapeToCaseInfo(*(params.input), caseInfo);
  AddTensorShapeToCaseInfo(*(params.weight), caseInfo);
  AddTensorShapeToCaseInfo(*(params.outBackprop), caseInfo);
  AddAclIntArrayToCaseInfo(*(params.stride), caseInfo);
  AddAclIntArrayToCaseInfo(*(params.padding), caseInfo);
  AddAclIntArrayToCaseInfo(*(params.dilation), caseInfo);
  caseInfo.push_back(params.groups);
}

static bool IsConv3DV2WhiteListCase(const vector<int64_t> &caseInfo, const vector<vector<int64_t>> &whiteList)
{
  if (caseInfo[0] != DataType::DT_BF16 && caseInfo[0] != DataType::DT_FLOAT16) {
    return false;
  }
  vector<int64_t> caseInfoShape(caseInfo);
  caseInfoShape.erase(caseInfoShape.begin());
  for (auto it = whiteList.begin(); it != whiteList.end(); ++it) {
    vector<int64_t> itShape(*it);
    itShape.erase(itShape.begin());
    if (caseInfoShape == itShape) {
      return true;
    }
  }
  return false;
}

static bool CheckV2Dilation(const ConvBackpropParams &params)
{
  const aclIntArray &dilation = *(params.dilation);
  for (size_t i = 0; i < dilation.Size(); i++) {
    if (dilation[i] > 1) {
      return true;
    }
  }
  return false;
}

static bool IsConv2DV2WhiteListCase(const vector<int64_t> &caseInfo, const vector<vector<int64_t>> &whiteList)
{
  for (auto it = whiteList.begin(); it != whiteList.end(); ++it) {
    if (*it == caseInfo) {
      return true;
    }
  }
  return false;
}

static bool CheckV2Stride(const ConvBackpropParams &params)
{
  const aclIntArray &stride = *(params.stride);
  for (size_t i = 0; i < stride.Size(); i++) {
    if (stride[i] > 2) { // 2: stride threshold for v2
      OP_LOGD("Conv3ddx v2 not support stride > 2");
      return false;
    }
  }
  return true;
}

static bool CheckOutputPadding(const ConvBackpropParams &params)
{
  auto outputShape = params.outBackprop->GetViewShape();
  auto kernelShape = params.weight->GetViewShape();
  auto inputShape = params.input->GetViewShape();
  const aclIntArray &dilation = *(params.dilation);
  const aclIntArray &padding = *(params.padding);
  const aclIntArray &stride = *(params.stride);

  for (size_t i = D_DIM_NCDHW_INDEX; i < kernelShape.GetDimNum(); i++) {
    auto sizeOut = outputShape.GetDim(i);
    auto sizeIn = inputShape.GetDim(i);
    auto kernelSize = kernelShape.GetDim(i);
    auto strideLocal = stride[i - D_DIM_NCDHW_INDEX];
    auto paddingLocal = padding[i - D_DIM_NCDHW_INDEX];
    auto dilationLocal = dilation[i - D_DIM_NCDHW_INDEX];
    auto outputPadding = sizeIn - ((sizeOut - 1) * strideLocal + (kernelSize - 1) * dilationLocal - 2 * paddingLocal + 1);
    if (outputPadding > 0) {
      OP_LOGD("outputPadding is: %ld which is greater than 0. Routing to V2", outputPadding);
      return true;
    }
  }
  return false;
}

static bool CheckV2Pad(const ConvBackpropParams &params)
{
  const aclIntArray &padding = *(params.padding);
  for (size_t i = 0; i < padding.Size(); i++) {
    if (padding[i] > 2) { // 2: padding threshold for v2
      OP_LOGD("Conv3ddx v2 not support pad > 2");
      return false;
    }
  }
  return true;
}

static bool CheckV2Kernel(const ConvBackpropParams &params)
{
  const aclTensor &kernel = *(params.weight);
  auto kernelShape = kernel.GetViewShape();
  for (size_t i = D_DIM_NCDHW_INDEX; i < kernelShape.GetDimNum(); i++) {
    if (kernelShape.GetDim(i) > 4) { // 4: kernel threshold for v2
      OP_LOGD("Conv3ddx v2 not support kernel > 4");
      return false;
    }
  }
  return true;
}

static bool CheckV1Functionality()
{
  SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
  if (socVersion == SocVersion::ASCEND910_95) {
    return false;
  }

  return true;
}

static bool CheckV2Functionality(const ConvBackpropParams &params)
{
  if (params.input->GetOriginalFormat() != op::Format::FORMAT_NCDHW) {
    OP_LOGD("Conv3d transpose v2 not support format except FORMAT_NCDHW");
    return false;
  }

  const aclIntArray &stride = *(params.stride);
  const aclIntArray &dilation = *(params.dilation);
  auto kernelShape = params.weight->GetViewShape();
  if (stride[0] > kernelShape[D_DIM_NCDHW_INDEX]) {
    OP_LOGD("Conv3ddx v2 not support stride_d > kernel_d");
    return false;
  }

  const aclIntArray &padding = *(params.padding);
  for (size_t i = 0; i < padding.Size(); i++) {
    if (((i + D_DIM_NCDHW_INDEX) >= kernelShape.GetDimNum()) || (padding[i] >= kernelShape[i + D_DIM_NCDHW_INDEX])) {
      OP_LOGD("Conv3ddx v2 not support pad >= kernel");
      return false;
    }
  }

  // 参考CUBE算子规格梳理
  SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
  if (socVersion == SocVersion::ASCEND910B) {
    auto outputShape = params.outBackprop->GetViewShape();
    // fmapH = min((2+(kernel_h-1)*dilation_h), ((Ho-1)*stride_h+1))
    int64_t fmapH = (outputShape[H_DIM_NCDHW_INDEX] - 1) * stride[1] + 1;
    // 2为待装载数据头尾存在跨行的情况
    fmapH = min(fmapH, 2 + (kernelShape[H_DIM_NCDHW_INDEX] - 1) * dilation[DIM_1]);
    int64_t l1SizeLimit = fmapH * outputShape[W_DIM_NCDHW_INDEX] * stride[DIM_2];
    if (l1SizeLimit > 4096) { // 4096: l1SizeLimit
      OP_LOGD("Conv3ddx v2 not support because L1 size limit is not meet");
      return false;
    }
  }

  return true;
}

static bool CheckV2Perf(const ConvBackpropParams &params)
{
  // 网络泛化case规格，测试结果 V2平均性能好于TBE
  if (!CheckV2Stride(params) || !CheckV2Pad(params) || !CheckV2Kernel(params)) {
    return false;
  }

  // 对网络泛化case中，V1性能劣化的场景进行拦截(din不能分核，分核集中在M方向，M又比较小，导致用不满核或者拖尾严重)
  // 从劣化的shape中分析出来的条件为: din不能分核 && (m<5500 || dout>85)
  auto inputShape = params.input->GetViewShape();
  int64_t m = inputShape[H_DIM_NCDHW_INDEX] * inputShape[W_DIM_NCDHW_INDEX];
  const int64_t mThreshold = 5500;
  const int64_t dinThreshold = 85;
  int64_t din = inputShape[D_DIM_NCDHW_INDEX];
  if (m < mThreshold || din > dinThreshold) {
    OP_LOGD("Conv3ddx v2 not support because shapes are not suitable");
    return false;
  }

  return true;
}

static bool IsConv3DBackpropInputUseV2(const ConvBackpropParams &params)
{
  SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
  if (socVersion == SocVersion::ASCEND910_95) {
    return true;
  }
  // 1. 以下检查为V1功能上不支持的场景或规格，走V2
  if (!CheckV1Functionality()) {
    return true;
  }

  // 2. 以下检查为V2功能上不支持的场景或规格，走TBE
  if (!CheckV2Functionality(params)) {
    return false;
  }

  // 3. OutputPadding > 0 与 dilation > 1的规格，走V2
  if (CheckOutputPadding(params) || CheckV2Dilation(params)) {
    return true;
  }

  // 4. V1V2都支持的场景，根据性能指标决定通路，默认走V2
  return CheckV2Perf(params);
}

bool IsConv2DBackpropInputTo3DCase(const ConvBackpropParams &params) {
  vector<int64_t> caseInfo;
  ConstructCaseInfo(params, caseInfo);
  SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
  if (socVersion == SocVersion::ASCEND910_95) {
    return true;
  }
  return IsConv2DV2WhiteListCase(caseInfo, CONV2D_BACKPROP_INPUT_V2_WHITE_LIST);
}

bool IsConv2DBackpropInputToCastCase(const ConvBackpropParams &params) {
 	   vector<int64_t> caseInfo;
 	   ConstructCaseInfo(params, caseInfo);
 	   SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
 	   if (socVersion == SocVersion::ASCEND910_95) {
 	     return false;
 	   }
 	   return IsConv2DV2WhiteListCase(caseInfo, CONV2D_BACKPROP_INPUT_CAST_WHITE_LIST);
}

bool IsConv2DBpFilterTo3Dcase(const ConvBackpropParams &params) {
  vector<int64_t> caseInfo;
  ConstructCaseInfo(params, caseInfo);
  return IsConv2DV2WhiteListCase(caseInfo, CONV2D_BACKPROP_FILTER_V3_WHITE_LIST);
}

bool IsConv3DBackpropInputV2(const ConvBackpropParams &params) {
  // 按是否支持TBE划分
  if (params.groups > 1 || params.input->GetDataType() == DataType::DT_FLOAT) {
    return true;
  }
  vector<int64_t> caseInfo;
  ConstructCaseInfo(params, caseInfo);
  return IsConv3DV2WhiteListCase(caseInfo, CONV3D_BACKPROP_INPUT_V2_WHITE_LIST) || IsConv3DBackpropInputUseV2(params);
}

static const aclTensor *GetOutputSize(const aclTensor *tensor, aclOpExecutor *executor) {
  auto tensorOriginalShape = op::ToShapeVector(tensor->GetOriginalShape());
  auto tensorSizeArry = executor->AllocIntArray(tensorOriginalShape.data(), tensorOriginalShape.size());
  OP_CHECK(tensorSizeArry != nullptr, OP_LOGD("tensorOriginalShape alloc failed."), return nullptr);
  auto outputSize = executor->ConvertToTensor(tensorSizeArry, DataType::DT_INT32);
  op::Format inputFormat = tensorOriginalShape.size() == CONV3D_DIM ? Format::FORMAT_NCDHW : Format::FORMAT_NCHW;
  const_cast<aclTensor *>(outputSize)->SetStorageFormat(inputFormat);
  const_cast<aclTensor *>(outputSize)->SetViewFormat(inputFormat);
  const_cast<aclTensor *>(outputSize)->SetOriginalFormat(inputFormat);

  return outputSize;
}

OP_TYPE_REGISTER(Conv2DBackpropInput);
static aclnnStatus Conv2DBackpropInputWithFlag(const aclTensor *input, const aclTensor *weight,
                                              const aclTensor *outBackprop, const aclIntArray *stride,
                                              const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                              int64_t useHf32Flag, aclTensor *&output, aclOpExecutor *executor) {
  L0_DFX(Conv2DBackpropInputWithFlag, input, weight, outBackprop, stride, padding, dilation, groups, useHf32Flag);
  FVector<int64_t> newStrides{1, 1, (*stride)[0], (*stride)[1]};
  FVector<int64_t> newDilation{1, 1, (*dilation)[0], (*dilation)[1]};
  FVector<int64_t> newPad{(*padding)[0], (*padding)[0], (*padding)[1], (*padding)[1]};
  // padding为四维, [up, down, left, right]
  if (padding->Size() == PAD_DIM_4) {
    newPad = {(*padding)[0], (*padding)[1], (*padding)[2], (*padding)[3]};
  }
  auto stride4 = executor->AllocIntArray(newStrides.data(), 4);
  OP_CHECK(stride4 != nullptr, OP_LOGD("newStrides alloc failed."), return ACLNN_ERR_INNER_INFERSHAPE_ERROR);
  auto dilation4 = executor->AllocIntArray(newDilation.data(), 4);
  OP_CHECK(dilation4 != nullptr, OP_LOGD("newDilation alloc failed."), return ACLNN_ERR_INNER_INFERSHAPE_ERROR);
  auto pad4 = executor->AllocIntArray(newPad.data(), 4);
  OP_CHECK(pad4 != nullptr, OP_LOGD("newPad alloc failed."), return ACLNN_ERR_INNER_INFERSHAPE_ERROR);
  const char *dataFormat = "NCHW";
  const char *paddingString = "";

  auto inputSize = GetOutputSize(input, executor);
  OP_LOGD("inputSize is: %s", inputSize->ToString().GetString());
  auto ret = INFER_SHAPE(Conv2DBackpropInput, OP_INPUT(inputSize, weight, outBackprop), OP_OUTPUT(output),
                         OP_ATTR(stride4, pad4, dilation4, groups, dataFormat, paddingString, useHf32Flag));
  if (ret != ACLNN_SUCCESS) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Conv2DBackpropInput InferShape failed.");
    return ACLNN_ERR_INNER_INFERSHAPE_ERROR;
  }
  ret = ADD_TO_LAUNCHER_LIST_AICORE(Conv2DBackpropInput, OP_INPUT(inputSize, weight, outBackprop),
    OP_OUTPUT(output), OP_ATTR(stride4, pad4, dilation4, groups, dataFormat, paddingString, useHf32Flag));
  OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return ACLNN_ERR_INNER_NULLPTR,
                                      "Conv2DBackpropInput ADD_TO_LAUNCHER_LIST_AICORE failed.");
  return ACLNN_SUCCESS;
}
// fp16输入fp16输出
const aclTensor *Conv2DBackpropInputFp162Fp16(const aclTensor *input, const aclTensor *weight,
                                              const aclTensor *outBackprop, const aclIntArray *stride,
                                              const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                              aclOpExecutor *executor) {
  L0_DFX(Conv2DBackpropInputFp162Fp16, input, weight, outBackprop, stride, padding, dilation, groups);
  auto output = executor->AllocTensor(op::DataType::DT_FLOAT16, input->GetStorageFormat(), op::Format::FORMAT_NCHW);
  int64_t useHf32Flag = 0x0;
  aclnnStatus ret = Conv2DBackpropInputWithFlag(input, weight, outBackprop, stride, padding, dilation, groups,
    useHf32Flag, output, executor);
  OP_CHECK(ret == ACLNN_SUCCESS,
    OP_LOGE(ret, "Conv2DBackpropInputFp162Fp16 get aclTensor* output failed."), return nullptr);
  return output;
}

// fp32输入fp32输出, 仅Ascend910B系列支持
const aclTensor *Conv2DBackpropInputFp322Fp32(const aclTensor *input, const aclTensor *weight,
                                              const aclTensor *outBackprop, const aclIntArray *stride,
                                              const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                              aclOpExecutor *executor) {
  L0_DFX(Conv2DBackpropInputFp322Fp32, input, weight, outBackprop, stride, padding, dilation, groups);
  auto output = executor->AllocTensor(op::DataType::DT_FLOAT, input->GetStorageFormat(), op::Format::FORMAT_NCHW);
  int64_t useHf32Flag = 0x0;
  aclnnStatus ret = Conv2DBackpropInputWithFlag(input, weight, outBackprop, stride, padding, dilation, groups,
    useHf32Flag, output, executor);
  OP_CHECK(ret == ACLNN_SUCCESS,
    OP_LOGE(ret, "Conv2DBackpropInputFp322Fp32 get aclTensor* output failed."), return nullptr);
  return output;
}

// fp32输入fp32输出, 使用HF32计算
const aclTensor *Conv2DBackpropInputHf32(const aclTensor *input, const aclTensor *weight,
                                              const aclTensor *outBackprop, const aclIntArray *stride,
                                              const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                              aclOpExecutor *executor) {
  L0_DFX(Conv2DBackpropInputHf32, input, weight, outBackprop, stride, padding, dilation, groups);
  auto output = executor->AllocTensor(op::DataType::DT_FLOAT, input->GetStorageFormat(), op::Format::FORMAT_NCHW);
  int64_t useHf32Flag = 0x40;
  aclnnStatus ret = Conv2DBackpropInputWithFlag(input, weight, outBackprop, stride, padding, dilation, groups,
    useHf32Flag, output, executor);
  OP_CHECK(ret == ACLNN_SUCCESS,
    OP_LOGE(ret, "Conv2DBackpropInputHf32 get aclTensor* output failed."), return nullptr);
  return output;
}

const aclTensor *Conv2DBackpropInputBf162Bf16(const aclTensor *input, const aclTensor *weight,
                                              const aclTensor *outBackprop, const aclIntArray *stride,
                                              const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                              aclOpExecutor *executor) {
  L0_DFX(Conv2DBackpropInputBf162Bf16, input, weight, outBackprop, stride, padding, dilation, groups);
  auto output = executor->AllocTensor(op::DataType::DT_BF16, input->GetStorageFormat(), op::Format::FORMAT_NCHW);
  int64_t useHf32Flag = 0x0;
  aclnnStatus ret =  Conv2DBackpropInputWithFlag(input, weight, outBackprop, stride, padding, dilation, groups,
    useHf32Flag, output, executor);
  OP_CHECK(ret == ACLNN_SUCCESS,
    OP_LOGE(ret, "Conv2DBackpropInputBf162Bf16 get aclTensor* output failed."), return nullptr);
  return output;
}
const aclTensor *Conv2DBackpropInput(const aclTensor *input, const aclTensor *weight,
                                     const aclTensor *outBackprop, const aclIntArray *stride,
                                     const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                     aclOpExecutor *executor, bool useHf32Flag, op::DataType dataType) {
  L0_DFX(Conv2DBackpropInput, input, weight, outBackprop, stride, padding, dilation, groups, useHf32Flag, dataType);
  op::Format outputFormat = input->GetStorageFormat();
  auto output = executor->AllocTensor(dataType, outputFormat, input->GetOriginalFormat());
  int64_t useHf32 = useHf32Flag == true ? 0x40 : 0;
  aclnnStatus ret = Conv2DBackpropInputWithFlag(input, weight, outBackprop, stride, padding, dilation, groups, useHf32,
    output, executor);
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ret, "Conv2DBackpropInput get aclTensor output failed."), return nullptr);
  return output;
}
OP_TYPE_REGISTER(Conv2DBackpropFilter);
static aclnnStatus Conv2DBackpropFilterWithFlag(const aclTensor *input, const aclTensor *weight,
                                               const aclTensor *outBackprop, const aclIntArray *stride,
                                               const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                               int64_t useHf32Flag, aclTensor *&output, aclOpExecutor *executor) {
  L0_DFX(Conv2DBackpropFilterWithFlag, input, weight, outBackprop, stride, padding, dilation, groups, useHf32Flag);
  FVector<int64_t> newStrides{1, 1, (*stride)[0], (*stride)[1]};
  FVector<int64_t> newDalition{1, 1, (*dilation)[0], (*dilation)[1]};
  FVector<int64_t> newPad{(*padding)[0], (*padding)[0], (*padding)[1], (*padding)[1]};
  if (padding->Size() == PAD_DIM_4) {
    newPad = {(*padding)[0], (*padding)[1], (*padding)[2], (*padding)[3]};
  }
  auto stride4 = executor->AllocIntArray(newStrides.data(), 4);
  OP_CHECK(stride4 != nullptr, OP_LOGD("newStrides alloc failed."), return ACLNN_ERR_INNER_NULLPTR);
  auto dilation4 = executor->AllocIntArray(newDalition.data(), 4);
  OP_CHECK(dilation4 != nullptr, OP_LOGD("newDalition alloc failed."), return ACLNN_ERR_INNER_NULLPTR);
  auto pad4 = executor->AllocIntArray(newPad.data(), 4);
  OP_CHECK(pad4 != nullptr, OP_LOGD("newPad alloc failed."), return ACLNN_ERR_INNER_NULLPTR);
  const char *dataFormat = "NCHW";
  const char *paddingString = "";
  const bool fromDepthwise = false;
  auto weightSize = GetOutputSize(weight, executor);
  OP_LOGD("weightSize is: %s", weightSize->ToString().GetString());
  auto ret =
      INFER_SHAPE(Conv2DBackpropFilter, OP_INPUT(input, weightSize, outBackprop), OP_OUTPUT(output),
                  OP_ATTR(stride4, pad4, dilation4, groups, dataFormat, paddingString, fromDepthwise, useHf32Flag));
  if (ret != ACLNN_SUCCESS) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Conv2DBackpropFilter InferShape failed.");
    return ACLNN_ERR_INNER_INFERSHAPE_ERROR;
  }
  ret = ADD_TO_LAUNCHER_LIST_AICORE(
      Conv2DBackpropFilter, OP_INPUT(input, weightSize, outBackprop), OP_OUTPUT(output),
      OP_ATTR(stride4, pad4, dilation4, groups, dataFormat, paddingString, fromDepthwise, useHf32Flag));
  OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return ACLNN_ERR_INNER_NULLPTR,
                                      "Conv2DBackpropFilter ADD_TO_LAUNCHER_LIST_AICORE failed.");
  return ACLNN_SUCCESS;
}

// fp16输入fp32输出
const aclTensor *Conv2DBackpropFilterFp162Fp32(const aclTensor *input, const aclTensor *weight,
                                               const aclTensor *outBackprop, const aclIntArray *stride,
                                               const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                               aclOpExecutor *executor) {
  L0_DFX(Conv2DBackpropFilterFp162Fp32, input, weight, outBackprop, stride, padding, dilation, groups);
  int64_t useHf32Flag = 0x0;
  auto output = executor->AllocTensor(op::DataType::DT_FLOAT, weight->GetStorageFormat(), op::Format::FORMAT_NCHW);
  aclnnStatus ret = Conv2DBackpropFilterWithFlag(input, weight, outBackprop, stride, padding, dilation, groups,
    useHf32Flag, output, executor);
  OP_CHECK(ret == ACLNN_SUCCESS,
    OP_LOGE(ret, "Conv2DBackpropFilterFp162Fp32 get aclTensor* output failed."), return nullptr);
  return output;
}

// fp16输入fp32输出
const aclTensor *Conv2DBackpropFilterFp322Fp32(const aclTensor *input, const aclTensor *weight,
                                               const aclTensor *outBackprop, const aclIntArray *stride,
                                               const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                               aclOpExecutor *executor) {
  L0_DFX(Conv2DBackpropFilterFp322Fp32, input, weight, outBackprop, stride, padding, dilation, groups);
  int64_t useHf32Flag = 0x0;
  auto output = executor->AllocTensor(op::DataType::DT_FLOAT, weight->GetStorageFormat(), op::Format::FORMAT_NCHW);
  aclnnStatus ret = Conv2DBackpropFilterWithFlag(input, weight, outBackprop, stride, padding, dilation, groups,
    useHf32Flag, output, executor);
  OP_CHECK(ret == ACLNN_SUCCESS,
    OP_LOGE(ret, "Conv2DBackpropFilterFp322Fp32 get aclTensor* output failed."), return nullptr);
  return output;
}

// fp32输入fp32输出, 使用HF32计算
const aclTensor *Conv2DBackpropFilterHf32(const aclTensor *input, const aclTensor *weight,
                                          const aclTensor *outBackprop, const aclIntArray *stride,
                                          const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                          aclOpExecutor *executor) {
  L0_DFX(Conv2DBackpropFilterHf32, input, weight, outBackprop, stride, padding, dilation, groups);
  int64_t useHf32Flag = 0x40;
  auto output = executor->AllocTensor(op::DataType::DT_FLOAT, weight->GetStorageFormat(), op::Format::FORMAT_NCHW);
  aclnnStatus ret = Conv2DBackpropFilterWithFlag(input, weight, outBackprop, stride, padding, dilation, groups,
    useHf32Flag, output, executor);
  OP_CHECK(ret == ACLNN_SUCCESS,
    OP_LOGE(ret, "Conv2DBackpropFilterHf32 get aclTensor* output failed."), return nullptr);
  return output;
}

// Bf16输入fp32输出
const aclTensor *Conv2DBackpropFilterBf162Fp32(const aclTensor *input, const aclTensor *weight,
                                               const aclTensor *outBackprop, const aclIntArray *stride,
                                               const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                               aclOpExecutor *executor) {
  L0_DFX(Conv2DBackpropFilterBf162Fp32, input, weight, outBackprop, stride, padding, dilation, groups);
  int64_t useHf32Flag = 0x0;
  auto output = executor->AllocTensor(op::DataType::DT_FLOAT, weight->GetStorageFormat(), op::Format::FORMAT_NCHW);
  aclnnStatus ret = Conv2DBackpropFilterWithFlag(input, weight, outBackprop, stride, padding, dilation, groups,
    useHf32Flag, output, executor);
  OP_CHECK(ret == ACLNN_SUCCESS,
    OP_LOGE(ret, "Conv2DBackpropFilterBf162Fp32 get aclTensor* output failed."), return nullptr);
  return output;
}

OP_TYPE_REGISTER(Conv3DBackpropFilter);
OP_TYPE_REGISTER(Conv3DBackpropFilterV2);
struct tagAdaptParam {
  aclIntArray *adaptStride {0};
  aclIntArray *adaptDilation {0};
  aclIntArray *adaptPad {0};
};
using AdaptParam = struct tagAdaptParam;
static void GetConv3DBackpropAdapterParam(const aclTensor *input, const aclIntArray *stride,
                                          const aclIntArray *padding, const aclIntArray *dilation,
                                          aclOpExecutor *executor, AdaptParam *params)
{
  SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
  if (stride->Size() == DIM_3) {
    FVector<int64_t> newStrides {1, 1, 1, 1, 1};
    FVector<int64_t> newDilation {1, 1, 1, 1, 1};
    if (input->GetStorageFormat() == op::Format::FORMAT_NCDHW || socVersion != SocVersion::ASCEND910_95) {
      newStrides = {1, 1, (*stride)[0], (*stride)[1], (*stride)[2]};
      newDilation = {1, 1, (*dilation)[0], (*dilation)[1], (*dilation)[2]};
    } else {
      newStrides = {1, (*stride)[0], (*stride)[1], (*stride)[2], 1};
      newDilation = {1, (*dilation)[0], (*dilation)[1], (*dilation)[2], 1};
    }

    FVector<int64_t> newPad {(*padding)[0], (*padding)[0], (*padding)[1], (*padding)[1], (*padding)[2], (*padding)[2]};
    if (padding->Size() == 6) { // pad dim = 6
      newPad = {(*padding)[0], (*padding)[1], (*padding)[2], (*padding)[3], (*padding)[4], (*padding)[5]};
    }
    params->adaptStride = executor->AllocIntArray(newStrides.data(), 5); // conv3D stride dim = 5
    OP_CHECK(params->adaptStride != nullptr, OP_LOGD("newStrides alloc failed."), return);
    params->adaptDilation = executor->AllocIntArray(newDilation.data(), 5); // conv3D Dilation dim = 5;
    OP_CHECK(params->adaptDilation != nullptr, OP_LOGD("newDilation alloc failed."), return);
    params->adaptPad = executor->AllocIntArray(newPad.data(), 6); // conv3D Pad dim = 6;
    OP_CHECK(params->adaptPad != nullptr, OP_LOGD("newPad alloc failed."), return);
  } else {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GetConv3DBackpropAdapterParam not support: %s with input dim:%ld",
            op::ToString(socVersion).GetString(), stride->Size());
  }
}

static bool CheckV2BasicBlockSupport(const ConvBackpropParams &params) {
  const aclTensor &kernel = *(params.weight);
  const aclTensor &outBackprop = *(params.outBackprop);
  const aclTensor &input = *(params.input);
  if (outBackprop.GetStorageShape().GetDimNum() != FORMAT_6HD_DIM
    || input.GetStorageShape().GetDimNum() != FORMAT_6HD_DIM) {
    OP_LOGD("The storage dim of input is: %lu", input.GetStorageShape().GetDimNum());
    return false;
  }
  auto kernelShape = kernel.GetViewShape();
  auto outBackpropShape = outBackprop.GetStorageShape();
  auto inputShape = input.GetStorageShape();
  uint64_t kernelHW = kernelShape.GetDim(H_DIM_NCDHW_INDEX) * kernelShape.GetDim(W_DIM_NCDHW_INDEX);
  uint64_t mValue = outBackpropShape.GetDim(C1_DIM_NDC1HWC0_INDEX) * inputShape.GetDim(C0_DIM_NDC1HWC0_INDEX);
  uint64_t nValue = kernelShape.GetDim(D_DIM_NCDHW_INDEX) * kernelShape.GetDim(H_DIM_NCDHW_INDEX)
                      * kernelShape.GetDim(W_DIM_NCDHW_INDEX) * inputShape.GetDim(C1_DIM_NDC1HWC0_INDEX)
                      * inputShape.GetDim(C0_DIM_NDC1HWC0_INDEX);
  bool isKernelSupport = (kernelHW == 1)
        || ((kernelHW == KERNEL_HW_4 || kernelHW == KERNEL_HW_9 || kernelHW == KERNEL_HW_16)
            && mValue >= BASIC_BLOCK_SIZE_128 && nValue >= BASIC_BLOCK_SIZE_128);
  bool isDataTypeSuport = (GetSizeByDataType(params.input->GetDataType()) == GetSizeByDataType(DataType::DT_BF16));
  return isKernelSupport && isDataTypeSuport;
}

bool IsConv3DBackpropFilterToV2WithDeterministic(const ConvBackpropParams &params) {
  const aclTensor &kernel = *(params.weight);
  const aclTensor &outBackprop = *(params.outBackprop);
  const aclTensor &input = *(params.input);
  if (outBackprop.GetStorageShape().GetDimNum() != FORMAT_6HD_DIM ||
      input.GetStorageShape().GetDimNum() != FORMAT_6HD_DIM) {
    return false;
  }
  const aclIntArray &padding = *(params.padding);
  const aclIntArray &dilation = *(params.dilation);
  if (params.groups > 1) {
    return false;
  }

  if ((dilation[DIM_0] != 1) || (dilation[DIM_1] != 1) || (dilation[DIM_2] != 1)) {
    return false;
  }

  auto outBackpropShape = outBackprop.GetStorageShape();
  auto inputShape = input.GetStorageShape();

  uint64_t co = outBackpropShape.GetDim(C1_DIM_NDC1HWC0_INDEX) * outBackpropShape.GetDim(C0_DIM_NDC1HWC0_INDEX);
  uint64_t ci = inputShape.GetDim(C1_DIM_NDC1HWC0_INDEX) * inputShape.GetDim(C0_DIM_NDC1HWC0_INDEX);
  if (((co > DETERMINISTIC_CHANNEL_16) && (co % DETERMINISTIC_CHANNEL_16 > 0)) ||
      ((ci > DETERMINISTIC_CHANNEL_16) && (ci % DETERMINISTIC_CHANNEL_16 > 0))) {
    return false;
  }

  auto kernelShape = kernel.GetViewShape();
  uint64_t kd = kernelShape.GetDim(D_DIM_NCDHW_INDEX);
  uint64_t kh = kernelShape.GetDim(H_DIM_NCDHW_INDEX);
  uint64_t kw = kernelShape.GetDim(W_DIM_NCDHW_INDEX);
  if ((kd > DETERMINISTIC_FILTER_V2_STRIDE_MAX_VALUE) || (kh > DETERMINISTIC_FILTER_V2_STRIDE_MAX_VALUE) ||
      (kw > DETERMINISTIC_FILTER_V2_STRIDE_MAX_VALUE)) {
    return false;
  }
  const aclIntArray &stride = *(params.stride);
  if ((stride[DIM_0] > static_cast<int64_t>(kd)) || (stride[DIM_1] > static_cast<int64_t>(kh)) ||
      (stride[DIM_2] > static_cast<int64_t>(kw))) {
    return false;
  }

  if ((padding[DIM_0] * DETERMINISTIC_PADDING_COEF > kd) || (padding[DIM_1] * DETERMINISTIC_PADDING_COEF > kh) ||
      (padding[DIM_2] * DETERMINISTIC_PADDING_COEF > kw)) {
    return false;
  }
  
  return true;
}

bool IsConv3DBackpropFilterV2(const ConvBackpropParams &params) {
  SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
  if (socVersion == SocVersion::ASCEND910_95) {
    return true;
  }
  if (params.input->GetOriginalFormat() != op::Format::FORMAT_NCDHW) {
    OP_LOGD("Conv3d filter v2 not support except FORMAT_NCDHW");
    return false;
  }

  if (params.input->GetDataType() == DataType::DT_FLOAT) {
    return true;
  }

  // dilation > 1, 走V2
  if (CheckV2Dilation(params)) {
    return true;
  }

  vector<int64_t> caseInfo;
  ConstructCaseInfo(params, caseInfo);
  if (params.groups > 1 || IsConv3DV2WhiteListCase(caseInfo, CONV3D_BACKPROP_FILTER_V2_WHITE_LIST)
    || IsConv3DV2WhiteListCase(caseInfo, CONV3D_BACKPROP_FILTER_V2_TRANSDATA_MERGE_WHITE_LIST)) {
    return true;
  }

  int64_t deterministicValue = 0;
  rtError_t retRts = rtCtxGetSysParamOpt(SYS_OPT_DETERMINISTIC, &deterministicValue);
  if (retRts != RT_ERROR_NONE) {
    deterministicValue = 0;
  }
  if (static_cast<bool>(deterministicValue)) {
    return IsConv3DBackpropFilterToV2WithDeterministic(params);
  }

  return CheckV2BasicBlockSupport(params);
}

bool IsInputTransdataWhiteListCase(const ConvBackpropParams &params) {
  SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
  if (socVersion == SocVersion::ASCEND910_95) {
    return false;
  }
  vector<int64_t> caseInfo;
  ConstructCaseInfo(params, caseInfo);
  return IsConv3DV2WhiteListCase(caseInfo, CONV3D_BACKPROP_FILTER_V2_TRANSDATA_MERGE_WHITE_LIST);
}

static aclnnStatus Conv3DBackpropFilterWithFlag(const aclTensor *input, const aclTensor *weight,
                                                const aclTensor *outBackprop, const aclIntArray *stride,
                                                const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                                int64_t useHf32, aclTensor *&output, aclOpExecutor *executor) {
  L0_DFX(Conv3DBackpropFilterWithFlag, input, weight, outBackprop, stride, padding, dilation, groups, useHf32);
  AdaptParam adptParams = {0};
  GetConv3DBackpropAdapterParam(input, stride, padding, dilation, executor, &adptParams);
  aclIntArray *stride5 = adptParams.adaptStride;
  aclIntArray *dilation5 = adptParams.adaptDilation;
  aclIntArray *pad6 = adptParams.adaptPad;
  const char *dataFormat = "NCDHW";
  const char *paddingString = "";

  auto weightSize = GetOutputSize(weight, executor);
  OP_LOGD("weightSize is: %s", weightSize->ToString().GetString());
  auto ret =
      INFER_SHAPE(Conv3DBackpropFilter, OP_INPUT(input, weightSize, outBackprop), OP_OUTPUT(output),
                  OP_ATTR(stride5, pad6, dilation5, groups, dataFormat, paddingString, useHf32));
  if (ret != ACLNN_SUCCESS) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Conv3DBackpropFilter InferShape failed.");
    output = nullptr;
    return ACLNN_ERR_INNER_INFERSHAPE_ERROR;
  }

  ConvBackpropParams params = {input, weight, outBackprop, stride, padding, dilation, groups};
  bool useV2Flag = IsConv3DBackpropFilterV2(params);
  if (useV2Flag) {
    ret = ADD_TO_LAUNCHER_LIST_AICORE(Conv3DBackpropFilterV2, OP_INPUT(input, weightSize, outBackprop), OP_OUTPUT(output),
                                OP_ATTR(stride5, pad6, dilation5, groups, dataFormat, paddingString, useHf32));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return ACLNN_ERR_INNER_NULLPTR,
                                       "Conv3DBackpropFilterV2 ADD_TO_LAUNCHER_LIST_AICORE failed.");
  } else {
    ret = ADD_TO_LAUNCHER_LIST_AICORE(
      Conv3DBackpropFilter, OP_INPUT(input, weightSize, outBackprop), OP_OUTPUT(output),
        OP_ATTR(stride5, pad6, dilation5, groups, dataFormat, paddingString, useHf32));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return ACLNN_ERR_INNER_NULLPTR,
                                       "Conv3DBackpropFilter ADD_TO_LAUNCHER_LIST_AICORE failed.");
  }
  return ACLNN_SUCCESS;
}
// fp16输入fp32输出
const aclTensor *Conv3DBackpropFilterFp162Fp32(const aclTensor *input, const aclTensor *weight,
                                               const aclTensor *outBackprop, const aclIntArray *stride,
                                               const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                               aclOpExecutor *executor) {
  L0_DFX(Conv3DBackpropFilterFp162Fp32, input, weight, outBackprop, stride, padding, dilation, groups);
  int64_t useHf32 = 0x0;
  auto output = executor->AllocTensor(op::DataType::DT_FLOAT, weight->GetStorageFormat(), op::Format::FORMAT_NCDHW);
  OP_CHECK(
    Conv3DBackpropFilterWithFlag(input, weight, outBackprop, stride, padding,
                                 dilation, groups, useHf32, output, executor) == ACLNN_SUCCESS,
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Conv3DBackpropFilterFp162Fp32 fail due to Conv3DBackpropFilterWithFlag error."),
    return nullptr
  );
  return output;
}

// fp16输入fp32输出
const aclTensor *Conv3DBackpropFilterFp322Fp32(const aclTensor *input, const aclTensor *weight,
                                               const aclTensor *outBackprop, const aclIntArray *stride,
                                               const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                               aclOpExecutor *executor) {
  L0_DFX(Conv3DBackpropFilterFp322Fp32, input, weight, outBackprop, stride, padding, dilation, groups);
  int64_t useHf32 = 0x0;
  auto output = executor->AllocTensor(op::DataType::DT_FLOAT, weight->GetStorageFormat(), op::Format::FORMAT_NCDHW);
  OP_CHECK(
    Conv3DBackpropFilterWithFlag(input, weight, outBackprop, stride, padding,
                                 dilation, groups, useHf32, output, executor) == ACLNN_SUCCESS,
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Conv3DBackpropFilterWithFlag failed."),
    return nullptr
  );
  return output;
}

// fp32输入fp32输出, 使用HF32计算
const aclTensor *Conv3DBackpropFilterHf32(const aclTensor *input, const aclTensor *weight,
                                               const aclTensor *outBackprop, const aclIntArray *stride,
                                               const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                               aclOpExecutor *executor) {
  L0_DFX(Conv3DBackpropFilterHf32, input, weight, outBackprop, stride, padding, dilation, groups);
  int64_t useHf32 = 0x40;
  auto output = executor->AllocTensor(op::DataType::DT_FLOAT, weight->GetStorageFormat(), op::Format::FORMAT_NCDHW);
  OP_CHECK(
    Conv3DBackpropFilterWithFlag(input, weight, outBackprop, stride, padding,
                                 dilation, groups, useHf32, output, executor) == ACLNN_SUCCESS,
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Conv3DBackpropFilterHf32 fail due to Conv3DBackpropFilterWithFlag error."),
    return nullptr
  );
  return output;
}

// Bf16输入fp32输出
const aclTensor *Conv3DBackpropFilterBf162Fp32(const aclTensor *input, const aclTensor *weight,
                                               const aclTensor *outBackprop, const aclIntArray *stride,
                                               const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                               aclOpExecutor *executor) {
  L0_DFX(Conv3DBackpropFilterBf162Fp32, input, weight, outBackprop, stride, padding, dilation, groups);
  int64_t useHf32 = 0x0;
  auto output = executor->AllocTensor(op::DataType::DT_FLOAT, weight->GetStorageFormat(), op::Format::FORMAT_NCDHW);
  OP_CHECK(
    Conv3DBackpropFilterWithFlag(input, weight, outBackprop, stride, padding,
                                 dilation, groups, useHf32, output, executor) == ACLNN_SUCCESS,
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Conv3DBackpropFilterBf162Fp32 fail due to Conv3DBackpropFilterWithFlag error."),
    return nullptr
  );
  return output;
}

// 1971: 6HD->FZ
// 910_95: 5HD->FZ
const aclTensor *Conv3DBackpropFilter(ConvolutionBackwardInputTensor &inputTensor, ConvolutionBackwardParams &params,
                                    aclOpExecutor *executor, bool hf32Flag)
{
  L0_DFX(Conv3DBackpropFilter, inputTensor.input, inputTensor.weight, inputTensor.gradOutput,
        params.stride, params.padding, params.dilation, params.groups);
  int64_t useHf32 = hf32Flag == true ? 0x40 : 0;
  op::Format outputFormat = inputTensor.weight->GetStorageFormat();
  auto output = executor->AllocTensor(op::DataType::DT_FLOAT, outputFormat, inputTensor.weight->GetOriginalFormat());
  OP_CHECK(
    Conv3DBackpropFilterWithFlag(inputTensor.input, inputTensor.weight, inputTensor.gradOutput,
                               params.stride, params.padding, params.dilation, params.groups,
                               useHf32, output, executor) == ACLNN_SUCCESS,
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Conv3DBackpropFilterBf162Fp32 fail due to Conv3DBackpropFilterWithFlag error."),
    return nullptr
  );
  return output;
}

OP_TYPE_REGISTER(Conv3DBackpropInput);
OP_TYPE_REGISTER(Conv3DBackpropInputV2);

static aclnnStatus Conv3DBackpropInputWithFlag(const aclTensor *input, const aclTensor *weight,
                                              const aclTensor *outBackprop, const aclIntArray *stride,
                                              const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                              int64_t useHf32Flag, aclTensor *&output, aclOpExecutor *executor)
{
  L0_DFX(Conv3DBackpropInputWithFlag, input, weight, outBackprop, stride, padding, dilation, groups, useHf32Flag, output);
  AdaptParam adptParams = {0};
  GetConv3DBackpropAdapterParam(input, stride, padding, dilation, executor, &adptParams);
  aclIntArray *stride5 = adptParams.adaptStride;   // Conv3d stride维度为5
  aclIntArray *dilation5 = adptParams.adaptDilation; // Conv3d dilation维度为5
  aclIntArray *pad6 = adptParams.adaptPad;          // conv3d pad维度为6
  const char *dataFormat = "NCDHW";
  const char *paddingString = "";

  ConvBackpropParams params = {input, weight, outBackprop, stride, padding, dilation, groups};
  SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
  bool useV2Flag = IsConv3DBackpropInputV2(params);
  if (useV2Flag && !useHf32Flag && weight->GetDataType() != DataType::DT_FLOAT && socVersion != SocVersion::ASCEND910_95) {
    output->SetStorageFormat(op::Format::FORMAT_NCDHW);
  }
  auto inputSize = GetOutputSize(input, executor);
  OP_LOGD("inputSize is: %s", inputSize->ToString().GetString());
  auto ret =
      INFER_SHAPE(Conv3DBackpropInput, OP_INPUT(inputSize, weight, outBackprop), OP_OUTPUT(output),
                  OP_ATTR(stride5, pad6, dilation5, groups, dataFormat, paddingString, useHf32Flag));
  if (ret != ACLNN_SUCCESS) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Conv3DBackpropInput InferShape failed.");
    output = nullptr;
    return ACLNN_ERR_INNER_INFERSHAPE_ERROR;
  }

  if (useV2Flag) {
    ret = ADD_TO_LAUNCHER_LIST_AICORE(
      Conv3DBackpropInputV2, OP_INPUT(inputSize, weight, outBackprop), OP_OUTPUT(output),
      OP_ATTR(stride5, pad6, dilation5, groups, dataFormat, paddingString, useHf32Flag));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return ACLNN_ERR_INNER_NULLPTR,
                                       "Conv3DBackpropInputV2 ADD_TO_LAUNCHER_LIST_AICORE failed.");
  } else {
    ret = ADD_TO_LAUNCHER_LIST_AICORE(
      Conv3DBackpropInput, OP_INPUT(inputSize, weight, outBackprop), OP_OUTPUT(output),
      OP_ATTR(stride5, pad6, dilation5, groups, dataFormat, paddingString, useHf32Flag));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return ACLNN_ERR_INNER_NULLPTR,
                                       "Conv3DBackpropInput ADD_TO_LAUNCHER_LIST_AICORE failed.");
  }
  return ACLNN_SUCCESS;
}

const aclTensor *Conv3DBackpropInputFp162Fp16(const aclTensor *input, const aclTensor *weight,
                                              const aclTensor *outBackprop, const aclIntArray *stride,
                                              const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                              aclOpExecutor *executor)
{
  L0_DFX(Conv3DBackpropInputFp162Fp16, input, weight, outBackprop, stride, padding, dilation, groups);
  int64_t useHf32Flag = 0x0;
  auto output = executor->AllocTensor(op::DataType::DT_FLOAT16, input->GetStorageFormat(), op::Format::FORMAT_NCDHW);
  OP_CHECK(
    Conv3DBackpropInputWithFlag(input, weight, outBackprop, stride, padding, dilation, groups,
                                useHf32Flag, output, executor) == ACLNN_SUCCESS,
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Conv3DBackpropInputFp162Fp16 fail due to Conv3DBackpropInputWithFlag error."),
    return nullptr
  );
  return output;
}
// 1971 6HD->FZ with Fp32
const aclTensor *Conv3DBackpropInputFp322Fp32(const aclTensor *input, const aclTensor *weight,
                                              const aclTensor *outBackprop, const aclIntArray *stride,
                                              const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                              aclOpExecutor *executor)
{
  L0_DFX(Conv3DBackpropInputFp322Fp32, input, weight, outBackprop, stride, padding, dilation, groups);
  int64_t useHf32Flag = 0x0;
  auto output = executor->AllocTensor(op::DataType::DT_FLOAT, input->GetStorageFormat(), op::Format::FORMAT_NCDHW);
  OP_CHECK(
    Conv3DBackpropInputWithFlag(input, weight, outBackprop, stride, padding, dilation, groups,
                                useHf32Flag, output, executor) == ACLNN_SUCCESS,
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Conv3DBackpropInputFp322Fp32 due to Conv3DBackpropInputWithFlag error."),
    return nullptr
  );
  return output;
}
// 1971 6HD->FZ with Hf32
const aclTensor *Conv3DBackpropInputHf32(const aclTensor *input, const aclTensor *weight, const aclTensor *outBackprop,
                                         const aclIntArray *stride, const aclIntArray *padding,
                                         const aclIntArray *dilation, int groups, aclOpExecutor *executor)
{
  L0_DFX(Conv3DBackpropInputHf32, input, weight, outBackprop, stride, padding, dilation, groups);
  int64_t useHf32Flag = 0x40;
  auto output = executor->AllocTensor(op::DataType::DT_FLOAT, input->GetStorageFormat(), op::Format::FORMAT_NCDHW);
  OP_CHECK(
    Conv3DBackpropInputWithFlag(input, weight, outBackprop, stride, padding, dilation, groups,
                                useHf32Flag, output, executor) == ACLNN_SUCCESS,
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Conv3DBackpropInputHf32 fail due to Conv3DBackpropInputWithFlag error."),
    return nullptr
  );
  return output;
}
// 1971 6HD->FZ with Bf16
const aclTensor *Conv3DBackpropInputBf162Bf16(const aclTensor *input, const aclTensor *weight,
                                              const aclTensor *outBackprop, const aclIntArray *stride,
                                              const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                              aclOpExecutor *executor)
{
  L0_DFX(Conv3DBackpropInputBf162Bf16, input, weight, outBackprop, stride, padding, dilation, groups);
  int64_t useHf32Flag = 0x0;
  auto output = executor->AllocTensor(op::DataType::DT_BF16, input->GetStorageFormat(), op::Format::FORMAT_NCDHW);
  OP_CHECK(
    Conv3DBackpropInputWithFlag(input, weight, outBackprop, stride, padding, dilation, groups,
                                useHf32Flag, output, executor) == ACLNN_SUCCESS,
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Conv3DBackpropInputBf162Bf16 fail due to Conv3DBackpropInputWithFlag error."),
    return nullptr
  );
  return output;
}

// 1982: 5HD->FZ
const aclTensor *Conv3DBackpropInput(ConvolutionBackwardInputTensor &inputTensor, ConvolutionBackwardParams &params,
                                    aclOpExecutor *executor, bool hf32Flag)
{
  L0_DFX(Conv3DBackpropInput, inputTensor.input, inputTensor.weight, inputTensor.gradOutput, params.stride,
         params.padding, params.dilation, params.groups);
  op::Format outputFormat = inputTensor.input->GetStorageFormat();
  int64_t useHf32 = hf32Flag == true ? 0x40 : 0;
  auto output = executor->AllocTensor(inputTensor.input->GetDataType(), outputFormat, inputTensor.input->GetOriginalFormat());
  OP_CHECK(
    Conv3DBackpropInputWithFlag(inputTensor.input, inputTensor.weight, inputTensor.gradOutput,
                              params.stride, params.padding, params.dilation, params.groups,
                              useHf32, output, executor) == ACLNN_SUCCESS,
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Conv3DBackpropInput fail due to Conv3DBackpropInputWithFlag error."),
    return nullptr
  );
  return output;
}
}  // namespace l0op