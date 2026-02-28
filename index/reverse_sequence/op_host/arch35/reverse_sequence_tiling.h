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
* \file reverse_sequence_tiling.h
* \brief
*/
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_INDEX_FILL_D_TILING_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_INDEX_FILL_D_TILING_H
#include "register/op_def_registry.h"
#include "op_host/tiling_base.h"
#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"

namespace optiling {
using namespace Ops::NN::Optiling;

struct ReverseSequenceCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

}  // namespace optiling
#endif  // OPS_BUILD_IN_OP_TILING_RUNTIME_REVERSE_SEQUENCE_TILING_H