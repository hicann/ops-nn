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
 * \file plugin_util.h
 * \brief
 */

#ifndef FRAMEWORK_PLUGIN_UTIL_H
#define FRAMEWORK_PLUGIN_UTIL_H

#include "log/log.h"
#include "graph/ascend_string.h"

namespace domi {

template <typename T>
inline std::string GetOpName(const T& op)
{
  ge::AscendString op_ascend_name;
  ge::graphStatus ret = op.GetName(op_ascend_name);
  if (ret != ge::GRAPH_SUCCESS) {
    std::string op_name = "None";
    return op_name;
  }
  return op_ascend_name.GetString();
}

}  // namespace domi

#endif // FRAMEWORK_PLUGIN_UTIL_H
