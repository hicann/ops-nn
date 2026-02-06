/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file common_dtype.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_COMMON_DTYPE_H_
#define AIR_CXX_RUNTIME_V2_COMMON_DTYPE_H_

#include <string>
#include <sstream>

namespace optiling
{
/**
 ** function: ConcatString
 */
template <typename T>
std::string ConcatString(const T& arg)
{
    std::ostringstream oss;
    oss << arg;
    return oss.str();
}

template <typename T, typename... Ts>
std::string ConcatString(const T& arg, const Ts&... argLeft)
{
    std::ostringstream oss;
    oss << arg;
    oss << ConcatString(argLeft...);
    return oss.str();
}
} //namespace optiling

#endif