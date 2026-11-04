/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <string>
#include <vector>
#include "base/err_msg.h"
#include "log/log.h"

namespace Ops {
namespace NN {
inline void ReportMatmulParamError(const char* opName, const std::string& message)
{
    OP_LOGE_LIBOPAPI_REPORT(opName == nullptr ? "Matmul" : opName, "%s", message.c_str());
    const std::vector<const char*> keys = {"message"};
    const std::vector<const char*> values = {message.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1001", keys, values);
}

inline void ReportMatmulTilingError(const char* opName, const std::string& reason)
{
    const char* name = opName == nullptr ? "Matmul" : opName;
    OP_LOGE_LIBOPAPI_REPORT(name, "%s", reason.c_str());
    const std::vector<const char*> keys = {"opName", "reason"};
    const std::vector<const char*> values = {name, reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", keys, values);
}

inline void ReportMatmulExecutionError(const char* opName, const std::string& reason)
{
    const char* name = opName == nullptr ? "Matmul" : opName;
    OP_LOGE_LIBOPAPI_REPORT(name, "%s", reason.c_str());
    const std::vector<const char*> keys = {"opName", "reason"};
    const std::vector<const char*> values = {name, reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1009", keys, values);
}

inline void ReportMatmulInferShapeError(const char* opName, const std::string& reason)
{
    const char* name = opName == nullptr ? "Matmul" : opName;
    OP_LOGE_LIBOPAPI_REPORT(name, "%s", reason.c_str());
    const std::vector<const char*> keys = {"opName", "reason"};
    const std::vector<const char*> values = {name, reason.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1014", keys, values);
}
} // namespace NN
} // namespace Ops
