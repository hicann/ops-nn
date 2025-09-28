/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_cube_util.h
 */
#include <iostream>
#include <map>
using namespace std;
namespace testConvBp {

void GetPlatFormInfos(const char *compile_info_str, map<string, string> &soc_infos, map<string, string> &aicore_spec,
                      map<string, string> &intrinsics);

void GetPlatFormInfos(const char *compile_info_str, map<string, string> &soc_infos, map<string, string> &aicore_spec,
                      map<string, string> &intrinsics, map<string, string> &soc_version);

std::string GetExeDirPath();
}