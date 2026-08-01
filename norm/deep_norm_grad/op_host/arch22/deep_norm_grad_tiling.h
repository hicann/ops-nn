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
 * \file deep_norm_grad_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_DEEP_NORM_GRAD_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_DEEP_NORM_GRAD_H_
#include "register/tilingdata_base.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "error_util.h"
#include "op_host/tiling_templates_registry.h"

using namespace Ops::NN::Optiling;

namespace optiling {
BEGIN_TILING_DATA_DEF(DeepNormGradTilingData)
TILING_DATA_FIELD_DEF(uint32_t, useCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, nDimNum);
TILING_DATA_FIELD_DEF(uint32_t, dDimNum);
TILING_DATA_FIELD_DEF(uint32_t, nDealPerCore);
TILING_DATA_FIELD_DEF(uint32_t, nDealLastCore);
TILING_DATA_FIELD_DEF(uint32_t, mergeNCount);
TILING_DATA_FIELD_DEF(uint32_t, cutDTime);
TILING_DATA_FIELD_DEF(uint32_t, cutDPerTime);
TILING_DATA_FIELD_DEF(uint32_t, cutDLastTime);
TILING_DATA_FIELD_DEF(uint32_t, alpha);
TILING_DATA_FIELD_DEF(uint32_t, fixedOutputFlag);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DeepNormGrad, DeepNormGradTilingData)
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_DEEP_NORM_GRAD_H_
