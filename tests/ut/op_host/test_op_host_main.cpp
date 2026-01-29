/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <unistd.h>
#include <limits.h>
#include "platform/platform_info.h"
#include "base/registry/op_impl_space_registry_v2.h"

using namespace std;

class OpTilingUtEnvironment : public testing::Environment {
  public:
    OpTilingUtEnvironment() {}
    virtual void SetUp() {
      cout << "Global Environment SetpUp." << endl;
      fe::OptionalInfos opti_compilation_infos_ge;
      opti_compilation_infos_ge.Init();
      opti_compilation_infos_ge.SetSocVersion("soc_version");
      fe::PlatformInfoManager::GeInstance().SetOptionalCompilationInfo(opti_compilation_infos_ge);

      char exePath[PATH_MAX] = {0};
      (void)readlink("/proc/self/exe", exePath, sizeof(exePath) - 1);
      std::string exePathStr(exePath);
      auto pos = exePathStr.find_last_of('/');
      if (pos != std::string::npos) {
        exePathStr.erase(pos + 1);
      } else {
        exePathStr.assign("./");
      }
      string opHostSoPath = exePathStr + string("/libophost_nn_ut.so");
      gert::OppSoDesc oppSoDesc({ge::AscendString(opHostSoPath.c_str())}, "op_host_so");
      shared_ptr<gert::OpImplSpaceRegistryV2> opImplSpaceRegistryV2 = make_shared<gert::OpImplSpaceRegistryV2>();
      if (opImplSpaceRegistryV2->AddSoToRegistry(oppSoDesc) == ge::GRAPH_FAILED) {
          cout << "add so to registry failed." << endl;
          return;
      }

      gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(opImplSpaceRegistryV2);

    }

    virtual void TearDown() {
      cout << "Global Environment TearDown" << endl;
    }
};

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc,argv);
  testing::AddGlobalTestEnvironment(new OpTilingUtEnvironment());
  return RUN_ALL_TESTS();
}
