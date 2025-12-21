/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file fused_matmul_tiling.h
 * \brief
 */
#ifndef __OP_HOST_FUSED_MATMUL_TILING_H__
#define __OP_HOST_FUSED_MATMUL_TILING_H__

#include <map>
#include <nlohmann/json.hpp>

#include "error_util.h"
#include "log/log.h"

#include "platform/platform_infos_def.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_parse_context.h"

namespace optiling {
namespace fused_matmul {
class FusedMatMulTiling {
public:
    explicit FusedMatMulTiling(gert::TilingContext* context) : context_(context){};
    ~FusedMatMulTiling() = default;
    ge::graphStatus DoTiling();

protected:
    bool CheckArgs() const;
    ge::graphStatus CheckArgsPtr();
    bool GetShapeAttrsInfo();

private:
    gert::TilingContext* context_ = nullptr;
    std::string opType_;
    const char* opName_ = nullptr;
    bool hasX3Input_ = false;
    bool hasBias_ = false;
};
} // namespace fused_matmul
} // namespace optiling
#endif // __OP_HOST_FUSED_MATMUL_TILING_H__