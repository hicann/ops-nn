/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_quant_batch_matmul_inplace_add.cpp
 * \brief Kernel UT for the Ascend950 HiFloat8 cube basic-api path.
 */

#define DTYPE_X1 hifloat8_t
#define DTYPE_X2 hifloat8_t
#define DTYPE_Y float
#define ORIG_DTYPE_X1 DT_HIFLOAT8
#define ORIG_DTYPE_X2 DT_HIFLOAT8
#define ORIG_DTYPE_Y DT_FLOAT
#define __CCE_AICORE__ 310

#include <iostream>

#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "arch35/quant_batch_matmul_inplace_add_tiling_key.h"
#include "arch35/quant_batch_matmul_inplace_add.cpp"
#endif

using namespace std;

class QuantBatchMatmulInplaceAddKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "QuantBatchMatmulInplaceAddKernelTest SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "QuantBatchMatmulInplaceAddKernelTest TearDown" << endl;
    }
};

TEST_F(QuantBatchMatmulInplaceAddKernelTest, Hif8BasicApiKernelCompile)
{
#ifdef __CCE_KT_TEST__
    using KernelFunc = void (*)(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR x2Scale, GM_ADDR yIn, GM_ADDR x1Scale, GM_ADDR y, GM_ADDR workspace,
        GM_ADDR tiling);
    KernelFunc kernelFunc = quant_batch_matmul_inplace_add<1, 0, TPL_NO_VEC_EPILOGUE_WITH_MMAPI>;
    ASSERT_NE(kernelFunc, nullptr);
    ASSERT_GT(sizeof(QMMIA::QuantBatchMatmulInplaceAddTilingData), 0UL);
#endif
}
