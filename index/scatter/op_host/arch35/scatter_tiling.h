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
 * \file scatter_tiling.h
 * \brief
 */
#ifndef CANN_SCATTER_H
#define CANN_SCATTER_H

#include <nlohmann/json.hpp>
#include "util/shape_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_host/util/platform_util.h"

namespace optiling {
struct ScatterKvCompileInfo {
  int64_t core_num;
  int64_t ub_size;
};


template <typename T>
static inline T* GetCompileInfoPtr(gert::TilingParseContext* context) {
  return context->GetCompiledInfo<T>();
}
}  // namespace optiling
#endif  // CANN_SCATTER_H
