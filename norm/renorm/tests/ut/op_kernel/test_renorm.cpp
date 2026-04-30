/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_renorm.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <sstream>
#include <string>
#include <iostream>
#include <cstdint>
#include <string>

#include "gtest/gtest.h"
// #include "renorm_tiling_def.h"
#include "../../../op_kernel/renorm_apt.cpp"
#include "../../../op_kernel/arch35/renorm_tiling_struct.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "register/op_def_registry.h"
#include "string.h"
#endif

using namespace std;

class renorm_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "renorm_kernel_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "renorm_kernel_test TearDown\n" << endl;
    }
};

std::string Shape2Str(const std::vector<int64_t>& shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        oss << shape[i];
        if (i != shape.size() - 1) {
            oss << ",";
        }
    }
    oss << "]";
    return oss.str();
}

static inline int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for(auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void ExcuteTestCase(const std::vector<int64_t> &xShape, const std::string &dtype,
                    int64_t tilingKey, uint32_t blockNum, uint8_t *tiling, uint32_t templateNum)
{
    uint32_t typeSize = 4;
    uint32_t fp32TypeSize = 4;
    if (dtype != "float") {
        typeSize = 2;
    }
    // 每个block对应 1 aic + 2 aiv
    uint32_t realBlockNum = (blockNum + 1) / 2;
    size_t xFileSize = GetShapeSize(xShape) * typeSize;

    size_t workspaceFileSize = 16 * 1024 * 1024;

    uint8_t *x = (uint8_t *)AscendC::GmAlloc((xFileSize + 31)/32*32);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc((xFileSize + 31)/32*32);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto KernelRenorm = [](GM_ADDR x, GM_ADDR boundaries, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::renorm<1, templateNum>(x, y, workspace, tiling);
    };
    ICPU_RUN_KF(KernelRenorm, realBlockNum, x, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(renorm_test, test_p1)
{
    std::vector<int64_t> xShape = {2, 2, 2, 2};
    std::string dtype = "float";
    uint64_t tilingKey = 33559615;
    uint32_t blockNum = 1;
    size_t tilingSize = sizeof(RenormTilingData);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    RenormTilingData* tilingDatafromBin = reinterpret_cast<RenormTilingData*>(tiling);

    tilingDatafromBin->ReduceOpTilingData.factorACntPerCore = 5;
    tilingDatafromBin->ReduceOpTilingData.factorATotalCnt = 5;
    tilingDatafromBin->ReduceOpTilingData.ubFactorA = 5;
    tilingDatafromBin->ReduceOpTilingData.factorRCntPerCore = 5;
    tilingDatafromBin->ReduceOpTilingData.factorRTotalCnt = 5;
    tilingDatafromBin->ReduceOpTilingData.ubFactorR = 5;
    tilingDatafromBin->ReduceOpTilingData.groupR = 5;
    tilingDatafromBin->ReduceOpTilingData.outSize = 5;
    tilingDatafromBin->ReduceOpTilingData.basicBlock = 5;
    tilingDatafromBin->ReduceOpTilingData.resultBlock = 5;
    tilingDatafromBin->ReduceOpTilingData.coreNum = 5;
    tilingDatafromBin->ReduceOpTilingData.useNddma = 5;
    tilingDatafromBin->ReduceOpTilingData.meanVar = 5;
    tilingDatafromBin->epsilon = 0;
    tilingDatafromBin->p = 0;
    tilingDatafromBin->recp = 0;
    tilingDatafromBin->maxnorm = 0;
    ExcuteTestCase(xShape, wShape, dtype, tilingKey,  blockNum, (uint8_t *)tilingDatafromBin);
}