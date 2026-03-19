/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file concat_offset_arch35.h
 * \brief
 */
#ifndef _CONCAT_OFFSET_H_
#define _CONCAT_OFFSET_H_

#include <cstdint>
#include <nlohmann/json.hpp>
#include <sstream>
#include "op_host/tiling_util.h"

namespace optiling {
struct ConcatOffsetCompileParams {
  int64_t core_num;
  int64_t ubSize{0};
  bool isAscendc{false};
};

struct ConcatOffsetTilingParams {
  int64_t input_num;
};
}  // namespace optiling

#endif // _CONCAT_OFFSET_H_
