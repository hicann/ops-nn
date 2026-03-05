/**
 * Copyright (c) 20265 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file util.h
 * \brief
 */

#ifndef POOL_GRAD_COMMON_UTIL_H
#define POOL_GRAD_COMMON_UTIL_H

static const gert::Shape g_vec_1_shape = {1};
constexpr int64_t MAX_INT32 = 2147483647;
static constexpr int64_t DIGIT_TWO = 2;
static constexpr size_t WS_SYS_SIZE = static_cast<size_t>(16 * 1024 * 1024);

static bool IsInvalidPaddingMode(std::string padMode)
{
    const std::set<std::string> supportedPadModeList = {"SAME", "VALID"};
    bool padModeInValid = (supportedPadModeList.count(padMode) == 0);
    return padModeInValid;
}

static bool IsInvalidPaddingModeWithCalculated(std::string padMode)
{
    const std::set<std::string> supportedPadModeList = {"SAME", "VALID", "CALCULATED"};
    bool padModeInValid = (supportedPadModeList.count(padMode) == 0);
    return padModeInValid;
}
#endif