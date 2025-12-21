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
 * \file platform_util.h
 * \brief
 */
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <tiling/platform/platform_ascendc.h>

namespace Ops {
namespace NN {
namespace Conv {

struct Conv3DBackpropV2CompileInfo {
  std::string soc_version = "";
  platform_ascendc::SocVersion shortSocVersion = platform_ascendc::SocVersion::ASCEND910B;

  uint32_t core_num = 0;
  uint64_t ub_size = 0;
  uint64_t l1_size = 0;
  uint64_t l2_size = 0;
  uint64_t l0a_size = 0;
  uint64_t l0b_size = 0;
  uint64_t l0c_size = 0;
  uint64_t bt_size = 0;
  int32_t cube_freq = 0;
  bool load3d_constraints = true;
  bool intrinsic_data_move_l12ub = true;
  bool intrinsic_matmul_ub_to_ub = false;
  bool intrinsic_conv_ub_to_ub = false;
  bool intrinsic_data_move_l0c2ub = true;
  bool intrinsic_fix_pipe_l0c2out = false;
  bool intrinsic_fix_pipe_l0c2ub = false;
  bool intrinsic_data_move_out2l1_nd2nz = false;
  bool intrinsic_data_move_l12bt_bf16 = false;
};

class PlatformUtil {
 public:
  static void ParseRuntimePlatformInfo(Conv3DBackpropV2CompileInfo& compileInfo, const char *op_name, fe::PlatFormInfos &platform_info);

 private:
  static void GetLocalMemSize(fe::PlatFormInfos &platform_info, const std::string &lable, const std::string &mem_type, uint64_t &size);
  static void GetChipFeature(fe::PlatFormInfos &platform_info, const std::string &lable,
                             const std::string &platform_info_key, const std::string &true_value, bool &value);
  static void GetAICoreIntrinsicDtype(fe::PlatFormInfos &platform_info, const std::string &intrinsic_name, bool &value);
};

}  // namespace Conv
}  // namespace NN
}  // namespace Ops
