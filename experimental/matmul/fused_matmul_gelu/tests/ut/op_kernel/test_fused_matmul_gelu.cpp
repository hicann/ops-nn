/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdlib>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

#include <gtest/gtest.h>

namespace {

std::string FindRuntimeBinary(const std::string& binaryName)
{
    const char* runtimeDir = std::getenv("FUSED_MATMUL_GELU_RUNTIME_TEST_DIR");
    if (runtimeDir == nullptr || runtimeDir[0] == '\0') {
        return "";
    }

    std::string binaryPath = std::string(runtimeDir) + "/" + binaryName;
    if (access(binaryPath.c_str(), X_OK) != 0) {
        return "";
    }

    return binaryPath;
}

int RunBinary(const std::string& binaryPath)
{
    pid_t pid = fork();
    if (pid < 0) {
        return -1;
    }

    if (pid == 0) {
        execl(binaryPath.c_str(), binaryPath.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        return -1;
    }

    return status;
}

void RunNumericCase(const std::string& binaryName)
{
    auto binaryPath = FindRuntimeBinary(binaryName);
    if (binaryPath.empty()) {
        GTEST_SKIP() << "Skip local runtime numeric test. "
                     << "Set FUSED_MATMUL_GELU_RUNTIME_TEST_DIR to enable it: " << binaryName;
    }

    int status = RunBinary(binaryPath);
    ASSERT_TRUE(WIFEXITED(status)) << "Runtime binary did not exit normally: " << binaryName;
    EXPECT_EQ(WEXITSTATUS(status), 0) << "Runtime binary failed: " << binaryName;
}

} // namespace

TEST(FusedMatmulGeluKernelNumeric, Fp16TanhWithBias) { RunNumericCase("test_aclnn_fused_matmul_gelu_fp16"); }

TEST(FusedMatmulGeluKernelNumeric, Fp16TanhNoBias) { RunNumericCase("test_aclnn_fused_matmul_gelu_fp16_nobias"); }

TEST(FusedMatmulGeluKernelNumeric, Bf16TanhZeroWeightNoBias)
{
    RunNumericCase("test_aclnn_fused_matmul_gelu_bf16_zero_weight_nobias");
}

TEST(FusedMatmulGeluKernelNumeric, Bf16TanhZeroWeightBiasOnly)
{
    RunNumericCase("test_aclnn_fused_matmul_gelu_bf16_zero_weight");
}

TEST(FusedMatmulGeluKernelNumeric, Bf16TanhWithBias) { RunNumericCase("test_aclnn_fused_matmul_gelu_bf16_tanh"); }
