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
#include <cstdlib>
#include "base/registry/op_impl_space_registry_v2.h"
#include "kernel_ut_log.h"

using namespace std;

bool CheckPython3Available()
{
    int result = system("python3 --version > /dev/null 2>&1");
    if (result != 0) {
        KERNEL_UT_LOG_ERROR("[OpKernelUT] python3 is not available in the environment");
        return false;
    }
    KERNEL_UT_LOG_INFO("[OpKernelUT] python3 is available");
    return true;
}

class OpKernelUtEnvironment : public testing::Environment {
public:
    OpKernelUtEnvironment() {}
    virtual void SetUp()
    {
        cout << "Global Environment SetpUp." << endl;

        // Check Python3 availability once at startup
        if (!CheckPython3Available()) {
            cout << "Warning: python3 is not available, some tests may fail." << endl;
        }

        /* load libnn_op_kernel_ut_${socversion}_ut.so for init tiling funcs and infershape funcs */
        const char* buildPath = std::getenv("BUILD_PATH");
        if (buildPath == nullptr) {
            cout << "getenv BUILD_PATH failed." << endl;
            return;
        }

        string opKernelTilingSoPath = buildPath + string("/tests/ut/op_kernel/libnn_op_kernel_ut_tiling.so");
        gert::OppSoDesc oppSoDesc({ge::AscendString(opKernelTilingSoPath.c_str())}, "nn_op_kernel_ut_so");
        shared_ptr<gert::OpImplSpaceRegistryV2> opImplSpaceRegistryV2 = make_shared<gert::OpImplSpaceRegistryV2>();
        if (opImplSpaceRegistryV2->AddSoToRegistry(oppSoDesc) == ge::GRAPH_FAILED) {
            cout << "add so to registry failed." << endl;
            return;
        }

        gert::DefaultOpImplSpaceRegistryV2::GetInstance().SetSpaceRegistry(opImplSpaceRegistryV2);
    }

    virtual void TearDown() { cout << "Global Environment TearDown" << endl; }
};

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    testing::AddGlobalTestEnvironment(new OpKernelUtEnvironment());
    int ret = RUN_ALL_TESTS();

    return ret;
}