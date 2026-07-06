/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_ut_data_helper.h"
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>

namespace kernel_ut {

std::string GetRepoRootDir()
{
    const char* envRoot = std::getenv("OPS_NN_ROOT");
    if (envRoot != nullptr && strlen(envRoot) > 0) {
        return std::string(envRoot);
    }

    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) == nullptr) {
        KERNEL_UT_LOG_ERROR("[KernelUTHelper] Failed to get current working directory");
        return "./";
    }

    std::string currentPath(cwd);

    while (!currentPath.empty()) {
        std::string cmakeFile = currentPath + "/CMakeLists.txt";
        std::string readmeFile = currentPath + "/README.md";

        struct stat cmakeStat, readmeStat;
        bool cmakeExists = (stat(cmakeFile.c_str(), &cmakeStat) == 0);
        bool readmeExists = (stat(readmeFile.c_str(), &readmeStat) == 0);

        if (cmakeExists && readmeExists) {
            KERNEL_UT_LOG_INFO("[KernelUTHelper] Found repository root at: %s", currentPath.c_str());
            return currentPath;
        }

        size_t lastSlash = currentPath.find_last_of('/');
        if (lastSlash == std::string::npos || lastSlash == 0) {
            break;
        }

        currentPath = currentPath.substr(0, lastSlash);
        if (currentPath.empty()) {
            currentPath = "/";
        }
    }

    KERNEL_UT_LOG_ERROR("[KernelUTHelper] Failed to find repository root directory, using current path");
    return std::string(cwd);
}

std::string GetTestWorkDir()
{
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) == nullptr) {
        KERNEL_UT_LOG_ERROR("[KernelUTHelper] Failed to get current working directory");
        return "./";
    }
    return std::string(cwd);
}

bool DirectoryExists(const std::string& dirPath)
{
    struct stat buf;
    if (stat(dirPath.c_str(), &buf) != 0) {
        return false;
    }
    return S_ISDIR(buf.st_mode);
}

bool CopyDirectory(const std::string& srcDir, const std::string& destDir)
{
    struct stat srcStat;
    if (stat(srcDir.c_str(), &srcStat) != 0) {
        KERNEL_UT_LOG_ERROR("[KernelUTHelper] Source directory does not exist: %s", srcDir.c_str());
        return false;
    }

    if (!S_ISDIR(srcStat.st_mode)) {
        KERNEL_UT_LOG_ERROR("[KernelUTHelper] Source path is not a directory: %s", srcDir.c_str());
        return false;
    }

    struct stat destStat;
    if (stat(destDir.c_str(), &destStat) == 0) {
        std::string rmCmd = "rm -rf \"" + destDir + "\"";
        int rmResult = system(rmCmd.c_str());
        if (rmResult != 0) {
            KERNEL_UT_LOG_ERROR("[KernelUTHelper] Failed to remove existing directory: %s", destDir.c_str());
            return false;
        }
    }

    std::string copyCmd = "cp -r \"" + srcDir + "\" \"" + destDir + "\"";
    int copyResult = system(copyCmd.c_str());

    if (copyResult != 0) {
        KERNEL_UT_LOG_ERROR("[KernelUTHelper] CopyDirectory failed: %s -> %s, system() returned %d", srcDir.c_str(),
                            destDir.c_str(), copyResult);
        return false;
    }

    KERNEL_UT_LOG_INFO("[KernelUTHelper] Copied directory: %s -> %s", srcDir.c_str(), destDir.c_str());
    return true;
}

bool SetDirectoryPermissions(const std::string& dirPath)
{
    mode_t mode = S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH;

    if (chmod(dirPath.c_str(), mode) != 0) {
        KERNEL_UT_LOG_ERROR("[KernelUTHelper] SetDirectoryPermissions failed for %s, errno: %d", dirPath.c_str(),
                            errno);
        return false;
    }

    KERNEL_UT_LOG_INFO("[KernelUTHelper] Set permissions for directory: %s", dirPath.c_str());
    return true;
}

bool CleanGeneratedBinFiles(const std::string& dirPath)
{
    struct stat dirStat;
    if (stat(dirPath.c_str(), &dirStat) != 0) {
        KERNEL_UT_LOG_INFO("[KernelUTHelper] Directory does not exist, skip cleaning: %s", dirPath.c_str());
        return true;
    }

    if (!S_ISDIR(dirStat.st_mode)) {
        KERNEL_UT_LOG_ERROR("[KernelUTHelper] Path is not a directory: %s", dirPath.c_str());
        return false;
    }

    DIR* dir = opendir(dirPath.c_str());
    if (dir == nullptr) {
        KERNEL_UT_LOG_ERROR("[KernelUTHelper] Failed to open directory: %s, errno: %d", dirPath.c_str(), errno);
        return false;
    }

    int cleanedCount = 0;
    struct dirent* entry;

    while ((entry = readdir(dir)) != nullptr) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        if (entry->d_type != DT_REG) {
            continue;
        }

        std::string fileName = entry->d_name;
        if (fileName.size() >= 4 && fileName.substr(fileName.size() - 4) == ".bin") {
            std::string fullPath = dirPath + "/" + fileName;
            if (remove(fullPath.c_str()) == 0) {
                cleanedCount++;
            } else {
                KERNEL_UT_LOG_ERROR("[KernelUTHelper] Failed to remove file: %s, errno: %d", fullPath.c_str(), errno);
            }
        }
    }

    closedir(dir);
    KERNEL_UT_LOG_INFO("[KernelUTHelper] Cleaned %d .bin files in directory: %s", cleanedCount, dirPath.c_str());
    return true;
}

std::string BuildLocalDataPath(const std::string& localDirName)
{
    char cwd[PATH_MAX];
    if (getcwd(cwd, sizeof(cwd)) == nullptr) {
        KERNEL_UT_LOG_ERROR("[KernelUTHelper] Failed to get current working directory");
        return "./" + localDirName;
    }
    return std::string(cwd) + "/" + localDirName;
}

bool PrepareTestDataDir(const std::string& dataDirRelPath, const std::string& localName)
{
    std::string repoRoot = GetRepoRootDir();
    std::string srcDir = repoRoot + "/" + dataDirRelPath;

    std::string targetName = localName;
    if (targetName.empty()) {
        size_t lastSlash = dataDirRelPath.find_last_of('/');
        if (lastSlash != std::string::npos && lastSlash + 1 < dataDirRelPath.size()) {
            targetName = dataDirRelPath.substr(lastSlash + 1);
        } else {
            targetName = dataDirRelPath;
        }
    }

    std::string destDir = BuildLocalDataPath(targetName);

    if (!DirectoryExists(srcDir)) {
        KERNEL_UT_LOG_ERROR("[KernelUTHelper] Test data directory does not exist: %s", srcDir.c_str());
        return false;
    }

    if (!CopyDirectory(srcDir, destDir)) {
        return false;
    }

    if (!SetDirectoryPermissions(destDir)) {
        return false;
    }

    return true;
}

bool SetupTestEnvironment(const std::string& dataDirRelPath, const std::string& localName)
{
    KERNEL_UT_LOG_INFO("[KernelUTHelper] Setting up test environment for: %s", dataDirRelPath.c_str());

    std::string targetName = localName;
    if (targetName.empty()) {
        size_t lastSlash = dataDirRelPath.find_last_of('/');
        if (lastSlash != std::string::npos && lastSlash + 1 < dataDirRelPath.size()) {
            targetName = dataDirRelPath.substr(lastSlash + 1);
        } else {
            targetName = dataDirRelPath;
        }
    }

    if (!PrepareTestDataDir(dataDirRelPath, targetName)) {
        KERNEL_UT_LOG_ERROR("[KernelUTHelper] Failed to prepare test data directory");
        return false;
    }

    std::string localDataDir = BuildLocalDataPath(targetName);
    if (!CleanGeneratedBinFiles(localDataDir)) {
        KERNEL_UT_LOG_ERROR("[KernelUTHelper] Failed to clean generated bin files");
        return false;
    }

    KERNEL_UT_LOG_INFO("[KernelUTHelper] Test environment setup complete: %s", localDataDir.c_str());
    return true;
}

} // namespace kernel_ut