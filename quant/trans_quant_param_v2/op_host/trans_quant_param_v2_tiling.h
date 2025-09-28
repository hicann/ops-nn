/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file trans_quant_param_v2_tiling.h
 * \brief
 */
#ifndef TRANS_QUANT_PARAM_V2_TILING_H
#define TRANS_QUANT_PARAM_V2_TILING_H
#include "register/tilingdata_base.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "tiling_base/tiling_base.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "error_util.h"
#include "op_cache_tiling.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TransQuantParamV2TilingData)
TILING_DATA_FIELD_DEF(uint32_t, scaleLength);
TILING_DATA_FIELD_DEF(uint32_t, offsetLength);
TILING_DATA_FIELD_DEF(uint32_t, roundMode);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TransQuantParamV2, TransQuantParamV2TilingData)

} // namespace optiling
#endif // TRANS_QUANT_PARAM_V2_TILING_H