/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_ut_data_executor.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <sys/stat.h>

namespace kernel_ut {

bool CheckScriptExists(const std::string& scriptPath)
{
    struct stat buf;
    if (stat(scriptPath.c_str(), &buf) != 0) {
        KERNEL_UT_LOG_ERROR("[KernelUTExecutor] Script file does not exist: %s", scriptPath.c_str());
        return false;
    }

    if (!S_ISREG(buf.st_mode)) {
        KERNEL_UT_LOG_ERROR("[KernelUTExecutor] Script path is not a regular file: %s", scriptPath.c_str());
        return false;
    }

    KERNEL_UT_LOG_INFO("[KernelUTExecutor] Script exists: %s", scriptPath.c_str());
    return true;
}

int ExecuteCommand(const std::string& command, const std::string& workDir)
{
    std::string fullCommand;
    if (workDir.empty()) {
        fullCommand = command + " > /dev/null 2>&1";
    } else {
        fullCommand = "cd " + workDir + " && " + command + " > /dev/null 2>&1";
    }

    KERNEL_UT_LOG_INFO("[KernelUTExecutor] Executing command: %s", fullCommand.c_str());

    int exitCode = system(fullCommand.c_str());

    if (exitCode == -1) {
        KERNEL_UT_LOG_ERROR(
            "[KernelUTExecutor] Failed to execute command (system() returned -1): %s", fullCommand.c_str());
        return -1;
    } else if (exitCode != 0) {
        KERNEL_UT_LOG_ERROR("[KernelUTExecutor] Command failed with exit code %d: %s", exitCode, fullCommand.c_str());
        return -1;
    }

    KERNEL_UT_LOG_INFO("[KernelUTExecutor] Command executed successfully");
    return 0;
}

bool RunGenData(const std::string& dataDir, const std::vector<std::string>& args)
{
    KERNEL_UT_LOG_INFO("[KernelUTExecutor] Running gen_data.py in directory: %s", dataDir.c_str());

    std::string genDataScript = dataDir + "/gen_data.py";
    if (!CheckScriptExists(genDataScript)) {
        KERNEL_UT_LOG_ERROR("[KernelUTExecutor] Cannot run gen_data.py: script not found at %s", genDataScript.c_str());
        return false;
    }

    std::ostringstream cmdBuilder;
    cmdBuilder << "python3 gen_data.py";
    for (const auto& arg : args) {
        cmdBuilder << " " << arg;
    }

    int result = ExecuteCommand(cmdBuilder.str(), dataDir);

    if (result == -1) {
        KERNEL_UT_LOG_ERROR("[KernelUTExecutor] gen_data.py execution failed");
        return false;
    }

    KERNEL_UT_LOG_INFO("[KernelUTExecutor] gen_data.py completed successfully");
    return true;
}

bool RunCompareData(const std::string& dataDir, const std::vector<std::string>& args)
{
    KERNEL_UT_LOG_INFO("[KernelUTExecutor] Running compare_data.py in directory: %s", dataDir.c_str());

    std::string compareDataScript = dataDir + "/compare_data.py";
    if (!CheckScriptExists(compareDataScript)) {
        KERNEL_UT_LOG_ERROR(
            "[KernelUTExecutor] Cannot run compare_data.py: script not found at %s", compareDataScript.c_str());
        return false;
    }

    std::ostringstream cmdBuilder;
    cmdBuilder << "python3 compare_data.py";
    for (const auto& arg : args) {
        cmdBuilder << " " << arg;
    }

    int result = ExecuteCommand(cmdBuilder.str(), dataDir);

    if (result == -1) {
        KERNEL_UT_LOG_ERROR("[KernelUTExecutor] compare_data.py execution failed");
        return false;
    }

    KERNEL_UT_LOG_INFO("[KernelUTExecutor] compare_data.py completed successfully");
    return true;
}

bool RunGenTiling(const std::string& dataDir, const std::string& caseName)
{
    KERNEL_UT_LOG_INFO("[KernelUTExecutor] Running gen_tiling.py in directory: %s", dataDir.c_str());

    std::string genTilingScript = dataDir + "/gen_tiling.py";
    if (!CheckScriptExists(genTilingScript)) {
        KERNEL_UT_LOG_WARN(
            "[KernelUTExecutor] Cannot run gen_tiling.py: script not found at %s", genTilingScript.c_str());
        return false;
    }

    std::ostringstream cmdBuilder;
    cmdBuilder << "python3 gen_tiling.py " << caseName;

    int result = ExecuteCommand(cmdBuilder.str(), dataDir);

    if (result == -1) {
        KERNEL_UT_LOG_WARN("[KernelUTExecutor] gen_tiling.py execution failed");
        return false;
    }

    KERNEL_UT_LOG_INFO("[KernelUTExecutor] gen_tiling.py completed successfully");
    return true;
}

} // namespace kernel_ut