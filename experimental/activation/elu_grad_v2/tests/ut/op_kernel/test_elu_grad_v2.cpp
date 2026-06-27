/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#endif

#include "../../../op_kernel/elu_grad_v2.cpp"
#include "../../../op_kernel/elu_grad_v2_tiling_data.h"

using namespace std;

namespace {

constexpr uint32_t kTestElems = 32 * 4 * 4 * 4; // 2048 elements, matching add_example reference

float EluGradRef(float grad, float x, float alpha, float scale, float inputScale, bool isResult)
{
    if (x > 0.0F) {
        return grad * scale;
    }
    if (isResult) {
        return grad * inputScale * (x + alpha * scale);
    }
    return grad * alpha * scale * inputScale * exp(x * inputScale);
}

template <uint32_t schMode>
void RunKernelTest(float alpha, float scale, float inputScale, bool isResult)
{
    size_t byteSize = kTestElems * sizeof(float);
    auto* grads = reinterpret_cast<float*>(AscendC::GmAlloc(byteSize));
    auto* acts = reinterpret_cast<float*>(AscendC::GmAlloc(byteSize));
    auto* y = reinterpret_cast<float*>(AscendC::GmAlloc(byteSize));
    auto* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(1024 * 1024 * 16));
    auto* tiling = reinterpret_cast<EluGradV2TilingData*>(AscendC::GmAlloc(sizeof(EluGradV2TilingData)));

    ASSERT_NE(grads, nullptr);
    ASSERT_NE(acts, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    vector<float> gradsHost(kTestElems);
    vector<float> actsHost(kTestElems);
    for (uint32_t i = 0; i < kTestElems; ++i) {
        gradsHost[i] = static_cast<float>(i % 5) - 2.0F; // [-2, -1, 0, 1, 2]
        actsHost[i] = static_cast<float>(i % 7) - 3.0F;  // [-3, -2, -1, 0, 1, 2, 3]
        grads[i] = gradsHost[i];
        acts[i] = actsHost[i];
        y[i] = 0.0F;
    }

    tiling->totalLength = kTestElems;
    tiling->tileDataNum = kTestElems;
    tiling->coreNum = 1U;
    tiling->bigCoreNum = 1U;
    tiling->bigCoreDataNum = kTestElems;
    tiling->smallCoreDataNum = kTestElems;
    tiling->lastCoreDataNum = kTestElems;
    tiling->bufferOpen = 0U;
    tiling->computeChunk = kTestElems;
    tiling->alpha = alpha;
    tiling->scale = scale;
    tiling->inputScale = inputScale;
    tiling->isResult = static_cast<uint8_t>(isResult);

    auto kernel = [](GM_ADDR gAddr, GM_ADDR aAddr, GM_ADDR yAddr, GM_ADDR wsAddr, GM_ADDR tAddr) {
        ::elu_grad_v2<schMode>(gAddr, aAddr, yAddr, wsAddr, tAddr);
    };

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(kernel, 1U, reinterpret_cast<uint8_t*>(grads), reinterpret_cast<uint8_t*>(acts),
                reinterpret_cast<uint8_t*>(y), workspace, reinterpret_cast<uint8_t*>(tiling));

    for (uint32_t i = 0; i < kTestElems; ++i) {
        float expected = EluGradRef(gradsHost[i], actsHost[i], alpha, scale, inputScale, isResult);
        EXPECT_NEAR(y[i], expected, 1e-4F) << "index=" << i;
    }

    AscendC::GmFree(grads);
    AscendC::GmFree(acts);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
}  // namespace

class elu_grad_v2_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "elu_grad_v2_test SetUp\n" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "elu_grad_v2_test TearDown\n" << endl;
    }
};

TEST_F(elu_grad_v2_test, test_exp_mode)
{
    RunKernelTest<ELU_GRAD_V2_TPL_SCH_MODE_SAFE_EXP>(1.2F, 0.7F, 1.1F, false);
}

TEST_F(elu_grad_v2_test, test_result_mode)
{
    RunKernelTest<ELU_GRAD_V2_TPL_SCH_MODE_SAFE_RESULT>(1.3F, 0.9F, 1.1F, true);
}
