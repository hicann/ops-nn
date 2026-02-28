/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_lstm.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_DYNAMIC_RNN_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_DYNAMIC_RNN_H
#include <cstdint>
#include <vector>
#include <nlohmann/json.hpp>
#include "log/log.h"                           // 如果涉及LOG日志打印
#include "register/op_impl_registry.h"         // 必需
#include "register/tilingdata_base.h"          // 必需
#include "op_host/tiling_base.h"                // 如果涉及TilingBaseClass类继承
#include "op_host/tiling_templates_registry.h" // 如果涉及TilingBaseClass类继承
#include "util/math_util.h"                   // 如果涉及CeilDiv等对齐运算
#include "tiling/tiling_api.h"
#include "error_util.h"
#include "dynamic_lstm_tiling.h"

namespace {
#define OPS_CHECK_NULL_WITH_CONTEXT_RET(context, ptr, ret)                                       \
if ((ptr) == nullptr) {                                                                        \
  const char* name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName(); \
  std::printf("EZ9999 op[%s], %s is nullptr!", name, #ptr);                                    \
  return ret;                                                                                  \
}
const gert::Shape g_vec_1_shape = {1};

}

namespace optiling {
  /*
 * @brief: get the json class of compile info from context
 * @param [in] context: gert::TilingContext
 * @return bool: std::unique_ptr<nlohmann::json>;
 */
inline std::unique_ptr<nlohmann::json> GetCompileInfoJson(gert::TilingParseContext* context) {
  auto json_str = context->GetCompiledJson();
  OPS_CHECK_NULL_WITH_CONTEXT_RET(context, json_str, nullptr);
  std::unique_ptr<nlohmann::json> parsed_object_cinfo = std::make_unique<nlohmann::json>(nlohmann::json::parse(json_str));
  return parsed_object_cinfo;
}

template <typename T>
bool ReadCompileItem(const nlohmann::json& all_vars, const std::string& name, T& value) {
  if (all_vars.empty()) {
    return false;
  }

  if (all_vars.count(name) == 0) {
    return false;
  }

  value = all_vars[name].get<T>();
  return true;
}

inline const gert::Shape &EnsureNotScalar(const gert::Shape &in_shape) {
  if (in_shape.IsScalar()) {
    return g_vec_1_shape;
  }
  return in_shape;
}


class CompileInfoBase;
using CompileInfoPtr = std::shared_ptr<CompileInfoBase>;

class CompileInfoBase {
public:
  CompileInfoBase() {}
  virtual ~CompileInfoBase() {}
};
}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_DYNAMIC_LSTM_H
