/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KERNEL_UT_DATA_HELPER_H
#define KERNEL_UT_DATA_HELPER_H

#include <string>
#include <iostream>
#include <sstream>
#include "kernel_ut_log.h"

namespace kernel_ut {

std::string GetRepoRootDir();

std::string GetTestWorkDir();

bool PrepareTestDataDir(const std::string& dataDirRelPath, const std::string& localName = "");

bool CleanGeneratedBinFiles(const std::string& dirPath);

std::string BuildLocalDataPath(const std::string& localDirName);

bool SetupTestEnvironment(const std::string& dataDirRelPath, const std::string& localName = "");

bool DirectoryExists(const std::string& dirPath);

bool CopyDirectory(const std::string& srcDir, const std::string& destDir);

bool SetDirectoryPermissions(const std::string& dirPath);

} // namespace kernel_ut

#endif