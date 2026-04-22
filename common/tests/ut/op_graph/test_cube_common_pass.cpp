/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <map>
#include <string>
#include <vector>

#define protected public
#define private public
#include "cube_utils/cube_common.h"
#undef protected
#undef private
using namespace std;
using namespace ge;
namespace ops {
class post_cube_common_ut : public testing::Test {
protected:
  static void SetUpTestCase()
  {
    // set version
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_data_move_out2l1_dn2nz"] = std::vector<std::string>();
    optiCompilationInfo.soc_version = "Ascend950";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    fe::PlatFormInfos platformInfos;
    fe::OptionalInfos optiCompilationInfos;
    platformInfos.Init();
    optiCompilationInfos.Init();
    std::map<std::string, std::vector<std::string>> fixpipe_dtype_map;
    fixpipe_dtype_map["Intrinsic_fix_pipe_unit_list"] = {"pre_covn", "pre_act", "post_eltwise", "post_act"};
    platformInfos.SetFixPipeDtypeMap(fixpipe_dtype_map);
    optiCompilationInfos.SetSocVersion("Ascend950");
    fe::PlatformInfoManager::Instance().platform_infos_map_["Ascend950"] = platformInfos;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfos);
  }

  static void TearDownTestCase() {
    std::cout << "post_cube_common_ut TearDown" << std::endl;
  }
};

TEST_F(post_cube_common_ut, ReadPlatFormConfig01) {
  ge::CustomPassContext context;
  std::map<std::string, std::map<std::string, std::vector<CONFIGDTYPE>>> post_cube_map_;
  std::vector<std::string> unit_list;
  std::map<std::string, std::vector<std::string>> depends_units;
  bool ret = PostCubeComm::ReadPlatFormConfig(context, false, unit_list, depends_units, post_cube_map_);
  for (auto iter : post_cube_map_) {
    std::cout << "  unit_name = " << iter.first << std::endl;
    for (auto iiter : iter.second) {
      std::cout << "      opname = " << iiter.first << std::endl;
      for (auto iiiter : iiter.second) {
        std::cout << "          issecond = " << iiiter.has_output_dtype
                  << "  first = " << static_cast<int>(iiiter.input_dtype)
                  << "  second = " << static_cast<int>(iiiter.output_dtype) << std::endl;
      }
    }
  }
  for (auto iter : depends_units) {
    std::cout << "unit_name = " << iter.first << std::endl;
    for (auto iiter : iter.second) {
      std::cout << "depend unit_name = " << iiter << std::endl;
    }
  }
  EXPECT_EQ(true, ret);
}

TEST_F(post_cube_common_ut, ReadPlatFormConfig02) {
  std::string kPostCubeConfigKey = "Intrinsic_fix_pipe_";
  std::string UINT_LIST_NAME = "unit_list";
  std::string kFuncList = "_func_list";
  std::string soc_version = "Ascend910_93";

  fe::PlatFormInfos platform_infos;
  fe::OptionalInfos opti_compilation_infos;
  if (fe::PlatformInfoManager::Instance().GetPlatformInfos(soc_version, platform_infos, opti_compilation_infos) ==
      SUCCESS) {
    std::map<std::string, std::vector<std::string>> inputmap_ = opti_compilation_infos.GetFixPipeDtypeMap();
    for (auto iter = inputmap_.begin(); iter != inputmap_.end(); iter++) {
      std::string key = iter->first;
      std::vector<std::string> dtype_vector = iter->second;
      EXPECT_EQ(key.find(kPostCubeConfigKey), 0);
      std::cout << "post_cube_test 3.1 key = " << key << std::endl;
      for (auto values : iter->second) { 
        EXPECT_NE(values.find(UINT_LIST_NAME), 0);
        std::cout << "post_cube_test 3.2 value = " << values << "  "; 
      }
      std::cout << std::endl;
    }
  }
}
}  // namespace ops