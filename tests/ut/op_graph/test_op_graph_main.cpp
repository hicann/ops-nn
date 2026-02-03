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
#include <unistd.h>
#include <limits.h>
#include "platform/platform_info.h"
#include "base/registry/op_impl_space_registry_v2.h"

using namespace std;

#ifdef __aarch64__
const string CPU_ARCH_STR = "aarch64";
#elif define(__x86_64__)
const string CPU_ARCH_STR = "x86_64";
#endif

class OpGraphUtEnvironment : public testing::Environment {
public:
    OpGraphUtEnvironment()
    {}
    virtual void SetUp()
    {
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
        string opHostSoPath = exePathStr + string("../op_host/libophost_nn_ut.so");

        string mathHostSoPath;
        const char* ascendToolkitHome = getenv("ASCEND_TOOLKIT_HOME");
        if (ascendToolkitHome == nullptr || strlen(ascendToolkitHome) == 0) {
            cout << "can not find env ASCEND_TOOLKIT_HOME." << endl; 
        }
        string ascendRoot(ascendToolkitHome);
        mathHostSoPath = ascendRoot + "/opp/built-in/op_impl/ai_core/tbe/op_host/lib/linux/"
                        + CPU_ARCH_STR
                        + "/libophost_math.so";
        if(access(mathHostSoPath.c_str(), R_OK) == -1) {
            cout << "libophost_math.so could not be found. Please install ops-math package." << endl;
        } else {
            cout << "Success found libophost_math.so." << endl;
        }

        gert::OppSoDesc oppSoDesc({ge::AscendString(opHostSoPath.c_str())}, "op_host_so");
        shared_ptr<gert::OpImplSpaceRegistryV2> opImplSpaceRegistryV2 = make_shared<gert::OpImplSpaceRegistryV2>();
        if (opImplSpaceRegistryV2->AddSoToRegistry(oppSoDesc) == ge::GRAPH_FAILED) {
            cout << "add op_host.so to registry failed." << endl;
            return;
        }
        gert::OppSoDesc mathSoDesc({ge::AscendString(mathHostSoPath.c_str())}, "math_so");
        if (opImplSpaceRegistryV2->AddSoToRegistry(mathSoDesc) == ge::GRAPH_FAILED) {
            cout << "add libophost_math.so to registry failed." << endl;
            return;
        }
        gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(opImplSpaceRegistryV2);
    }

    virtual void TearDown()
    {
        cout << "Global Environment TearDown" << endl;
    }
};

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    testing::AddGlobalTestEnvironment(new OpGraphUtEnvironment());
    _exit(RUN_ALL_TESTS());
}
