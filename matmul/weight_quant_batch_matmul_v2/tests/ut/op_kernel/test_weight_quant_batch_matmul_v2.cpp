/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>

#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include <iostream>
#include <sstream>
#include <string>

#include "data_utils.h"
#include "string.h"
#include "tikicpulib.h"
#include "weight_quant_batch_matmul_v2_tiling_def.h"
#include "../../../op_kernel/weight_quant_batch_matmul_v2.cpp"
#endif

#include <cstdint>

using namespace std;
// using namespace AscendC;

class TestWeightQuantBatchMatmulV2 : public testing::Test
{
};

