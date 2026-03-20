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
 * \file avg_pool_infershape_common.h
 * \brief Common includes, constants, and helper functions shared by
 *        avg_pool_infershape.cpp and avg_pool_v2_infershape.cpp.
 */
#pragma once

#include "log/log.h"
#include "util/shape_util.h"
#include "error_util.h"
#include "register/op_impl_registry.h"
#include <map>
#include "op_api/op_util.h"

using namespace ge;
using namespace gert;
using namespace Ops::Nn;

const size_t SUPPORTED_DIM_NUM = 4;

static std::map<ge::Format, std::string> format2str = {
    {ge::Format::FORMAT_NCHW, "NCHW"}, {ge::Format::FORMAT_NHWC, "NHWC"}, {ge::Format::FORMAT_HWCN, "HWCN"},
    {ge::Format::FORMAT_DHWNC, "DHWNC"}, {ge::Format::FORMAT_DHWCN, "DHWCN"}, {ge::Format::FORMAT_NDHWC, "NDHWC"},
    {ge::Format::FORMAT_NCDHW, "NCDHW"}};

inline std::string OtherErrMsg(const std::string& error_detail) {
  std::string msg = error_detail;
  return msg;
}

inline bool GetDimInFormat(const std::string& opName, const std::string& formatStr, const std::string& dimName,
                           int64_t& dimPosition)
{
    dimPosition = formatStr.find(dimName);
    if (dimPosition < 0) {
        CUBE_INNER_ERR_REPORT(opName.c_str(), "Position(%s) is invalid: %ld, which format is %s.",
                              dimName.c_str(), dimPosition, formatStr.c_str());
        return false;
    }
    return true;
}
