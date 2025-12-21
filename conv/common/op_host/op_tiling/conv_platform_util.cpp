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
 * \file platform_util.cc
 * \brief function of platform_util
 */

#include <algorithm>
#include <cmath>
#include "platform/platform_info.h"
#include "log/log.h"
#include "platform_util.h"

namespace Ops {
namespace NN {
namespace Conv {

void PlatformUtil::GetLocalMemSize(fe::PlatFormInfos &platform_info, const string &lable, const string &mem_type,
                                      uint64_t &size) {
  std::string temp_str;
  size = 0;
  if (platform_info.GetPlatformRes(lable, mem_type, temp_str)) {
    OP_LOGD("NO_OP_NAME", "PLATFORM INFO %s: %s", mem_type.c_str(), temp_str.c_str());
    try {
      size = atol(temp_str.c_str());
    } catch (const std::exception &e) {
      OP_LOGE_WITHOUT_REPORT("NO_OP_NAME", "illegal PLATFORM INFO %s: %s", mem_type.c_str(), e.what());
    }
  } else {
    OP_LOGW("NO_OP_NAME", "NO PLATFORM INFO for %s", mem_type.c_str());
  }
}

void PlatformUtil::GetChipFeature(fe::PlatFormInfos &platform_info, const string &lable,
                                     const string &platform_info_key, const string &true_value, bool &value) {
  std::string temp_str;
  if (platform_info.GetPlatformRes(lable, platform_info_key, temp_str)) {
    OP_LOGD("NO_OP_NAME", "PLATFORM INFO %s: %s", platform_info_key.c_str(), temp_str.c_str());
    value = (temp_str.find(true_value) != string::npos);
  } else {
    OP_LOGW("NO_OP_NAME", "NO PLATFORM INFO for %s", platform_info_key.c_str());
    value = false;
  }
}

void PlatformUtil::GetAICoreIntrinsicDtype(fe::PlatFormInfos &platform_info, const string &intrinsic_name, bool &value) {
  std::string val;
  (void)platform_info.GetPlatformRes("AICoreintrinsicDtypeMap", intrinsic_name, val);
  if (!val.empty()) {
    OP_LOGD("NO_OP_NAME", "PLATFORM INFO %s: %s", intrinsic_name.c_str(), val.c_str());
    value = true;
  } else {
    value = false;
    OP_LOGW("NO_OP_NAME", "NO PLATFORM INFO for %s", intrinsic_name.c_str());
  }
}

void PlatformUtil::ParseRuntimePlatformInfo(Conv3DBackpropV2CompileInfo& compileInfo, const char *op_name, fe::PlatFormInfos &platform_info) {
  std::string cube_freq_str;
  compileInfo.core_num = platform_info.GetCoreNum();
  platform_info.GetLocalMemSize(fe::LocalMemType::UB, compileInfo.ub_size);
  platform_info.GetLocalMemSize(fe::LocalMemType::L1, compileInfo.l1_size);
  platform_info.GetLocalMemSize(fe::LocalMemType::L2, compileInfo.l2_size);
  platform_info.GetLocalMemSize(fe::LocalMemType::L0_A, compileInfo.l0a_size);
  platform_info.GetLocalMemSize(fe::LocalMemType::L0_B, compileInfo.l0b_size);
  platform_info.GetLocalMemSize(fe::LocalMemType::L0_C, compileInfo.l0c_size);
  platform_info.GetPlatformRes("AICoreSpec", "cube_freq", cube_freq_str);
  platform_info.GetPlatformRes("version", "SoC_version", compileInfo.soc_version);
  GetLocalMemSize(platform_info, "AICoreSpec", "bt_size", compileInfo.bt_size);
  compileInfo.cube_freq = std::atoi(cube_freq_str.c_str());
  OP_LOGD(op_name,
          "PLATFORM INFO CORE_NUM: %u, UB: %lu, L1: %lu, L2: %lu, L0_A: %lu, L0_B: %lu, L0_C: %lu, BT_SIZE: %lu",
          compileInfo.core_num, compileInfo.ub_size, compileInfo.l1_size, compileInfo.l2_size,
          compileInfo.l0a_size, compileInfo.l0b_size, compileInfo.l0c_size, compileInfo.bt_size);
  OP_LOGD(op_name, "The platform info of soc version: %s, cube_freq %i", compileInfo.soc_version.c_str(), compileInfo.cube_freq);

  GetChipFeature(platform_info, "AICoreintrinsicDtypeMap",
                 "Intrinsic_data_move_l12bt", "bf16", compileInfo.intrinsic_data_move_l12bt_bf16);
  GetChipFeature(platform_info, "AICoreSpec", "load3d_constraints", "1", compileInfo.load3d_constraints);
  GetAICoreIntrinsicDtype(platform_info, "Intrinsic_data_move_l12ub", compileInfo.intrinsic_data_move_l12ub);
  GetAICoreIntrinsicDtype(platform_info, "Intrinsic_data_move_l0c2ub", compileInfo.intrinsic_data_move_l0c2ub);
  GetAICoreIntrinsicDtype(platform_info, "Intrinsic_fix_pipe_l0c2out", compileInfo.intrinsic_fix_pipe_l0c2out);
  GetAICoreIntrinsicDtype(platform_info, "Intrinsic_fix_pipe_l0c2ub", compileInfo.intrinsic_fix_pipe_l0c2ub);
  GetAICoreIntrinsicDtype(platform_info, "Intrinsic_data_move_out2l1_nd2nz", compileInfo.intrinsic_data_move_out2l1_nd2nz);
  GetAICoreIntrinsicDtype(platform_info, "Intrinsic_matmul_ub_to_ub", compileInfo.intrinsic_matmul_ub_to_ub);
  GetAICoreIntrinsicDtype(platform_info, "Intrinsic_conv_ub_to_ub", compileInfo.intrinsic_conv_ub_to_ub);
}

}  // namespace Conv
}  // namespace NN
}  // namespace Ops
