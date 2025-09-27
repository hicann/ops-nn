/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file test_cross_entropy_loss_grad_apt.cpp
 * \brief
 */
 #include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "../../../op_kernel/cross_entropy_loss_grad_apt.cpp"
#include "../../../op_kernel/arch35/cross_entropy_loss_grad_tiling_data.h"
#include "../../../op_kernel/arch35/cross_entropy_loss_grad_tiling_key.h"
#include "data_utils.h"
#include "../tiling_test_help.h"
#include "gtest/gtest.h"
#include "tikicpulib.h"


#include <cstdint>

using namespace std;

class cross_entropy_loss_grad_regbase_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "cross_entropy_loss_grad_regbase_test SetUp\n" << endl;
  }
  static void TearDownTestCase() {
    cout << "cross_entropy_loss_grad_regbase_test TearDown\n" << endl;
  }
};

static bool CompareData()
{
    std::string cmd1 = "cd ./cross_entropy_loss_grad_data/ && python3 verify.py x_grad.bin output_x_grad.bin ";
    return system(cmd1.c_str()) == 0;
}

TEST_F(cross_entropy_loss_grad_regbase_test, test_case_weight_not_none_sum) {
  std::vector<int64_t> gradLossShape{1};
  std::vector<int64_t> logProbShape{1, 64};
  std::vector<int64_t> targetShape{1};
  std::vector<int64_t> weightShape{64};
  std::vector<int64_t> outputShape{1, 64};
  auto inDType = DT_FLOAT;
  uint32_t reduction = 2; // sum场景
  std::string reductionStr = "sum";
  TilingTestHelp::ParamManager manager;
  manager.Init(4, 1);
  manager.AddInputShape(gradLossShape, inDType, FORMAT_ND);
  manager.AddInputShape(logProbShape, inDType, FORMAT_ND);
  manager.AddInputShape(targetShape, DT_INT64, FORMAT_ND);
  manager.AddInputShape(weightShape, DT_FLOAT, FORMAT_ND);
  manager.AddAttr("reduction", reductionStr);
  manager.AddAttr("ignore_index", static_cast<int64_t>(-100));
  manager.AddAttr("label_smoothing", 0.0f);
  manager.AddOutputShape(outputShape, inDType, FORMAT_ND);
  TilingTestHelp::TilingInfo tilingInfo;
  // UT_SOC_VERSION pass by UT framework, such as "Ascend910D1"
  EXPECT_EQ(DoTiling("CrossEntropyLossGrad", manager, tilingInfo, UT_SOC_VERSION, "{}"), true);
  ASSERT_TRUE(tilingInfo.tilingDataSize > 0);

  size_t gradLossByteSize = TilingTestHelp::ShapeSize(gradLossShape) * sizeof(float);
  size_t logProbByteSize = TilingTestHelp::ShapeSize(logProbShape) * sizeof(float);
  size_t targetByteSize = TilingTestHelp::ShapeSize(targetShape) * sizeof(int64_t);
  size_t weightByteSize = TilingTestHelp::ShapeSize(weightShape) * sizeof(float);
  // output
  size_t xGradByteSize = TilingTestHelp::ShapeSize(outputShape) * sizeof(float);
  AscendC::SetKernelMode(KernelMode::AIV_MODE);
  uint8_t* grad_loss = (uint8_t*)AscendC::GmAlloc(gradLossByteSize);
  uint8_t* log_prob = (uint8_t*)AscendC::GmAlloc(logProbByteSize);
  uint8_t* target = (uint8_t*)AscendC::GmAlloc(targetByteSize);
  uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightByteSize);
  uint8_t* grad_zloss = (uint8_t*)AscendC::GmAlloc(0);
  uint8_t* lse_for_zloss = (uint8_t*)AscendC::GmAlloc(0);
  uint8_t* x_grad = (uint8_t*)AscendC::GmAlloc(xGradByteSize);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);

  system("cp -r ../../../../../../../ops/built-in/tests/ut/fast_op_test/cross_entropy_loss_grad/cross_entropy_loss_grad_data ./");
  system("chmod -R 755 ./cross_entropy_loss_grad_data/");
  system("cd ./cross_entropy_loss_grad_data/ && rm -rf ./*bin");
  //  gen_golden_data_simple(input_shape, input_dtype, amsgrad, maximize)
  std::string cmd = "cd ./cross_entropy_loss_grad_data/ && python3 gen_data.py 1 64 float32 sum 1 -100 0.0";
  system(cmd.c_str());
  char * path_ = get_current_dir_name();
  string curPath(path_);
  ReadFile(curPath + "/cross_entropy_loss_grad_data/grad_loss.bin", gradLossByteSize, grad_loss, gradLossByteSize);
  ReadFile(curPath + "/cross_entropy_loss_grad_data/log_prob.bin", logProbByteSize, log_prob, logProbByteSize);
  ReadFile(curPath + "/cross_entropy_loss_grad_data/target.bin", targetByteSize, target, targetByteSize);
  ReadFile(curPath + "/cross_entropy_loss_grad_data/weight.bin", weightByteSize, weight, weightByteSize);

  ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
  auto cross_entropy_loss_grad_func = [](GM_ADDR grad_loss, GM_ADDR log_prob, GM_ADDR target,
    GM_ADDR weight, GM_ADDR grad_zloss, GM_ADDR lse_for_zloss,
    GM_ADDR x_grad, GM_ADDR workspace, GM_ADDR tiling) {
        cross_entropy_loss_grad<1, 2, 1, 0, 0>(grad_loss, log_prob, target, weight, grad_zloss, lse_for_zloss, x_grad, workspace, tiling);
  };
  ICPU_RUN_KF(cross_entropy_loss_grad_func, tilingInfo.blockNum, grad_loss, log_prob, target, weight, grad_zloss, lse_for_zloss,
              x_grad, workspace, tilingInfo.tilingData.get());

  WriteFile(curPath + "/cross_entropy_loss_grad_data/x_grad.bin", x_grad, xGradByteSize);
  AscendC::GmFree(grad_loss);
  AscendC::GmFree(log_prob);
  AscendC::GmFree(target);
  AscendC::GmFree(weight);
  AscendC::GmFree(x_grad);
  AscendC::GmFree(workspace);

  EXPECT_EQ(CompareData(), true);
  free(path_);
}