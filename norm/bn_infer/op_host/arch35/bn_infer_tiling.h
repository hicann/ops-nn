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
 * \file bn_infer_tiling.h
 * \brief
 */

#ifndef BN_INFER_TILING_H
#define BN_INFER_TILING_H

#include <cmath>
#include "op_common/log/log.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "op_host/tiling_base.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../../op_kernel/arch35/bn_infer_tiling_data.h"

using namespace Ops::NN::Optiling;

namespace optiling {
struct BNInferCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
    uint32_t vectorLength;
    uint64_t blockSize;
};
} // namespace optiling
#endif // BN_INFER_TILING_H
