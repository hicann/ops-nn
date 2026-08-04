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
 * \file normalize_bbox_tiling_key.h
 * \brief normalize_bbox tiling key template argument declaration
 *
 * TilingKey space = {DTYPE_BOXES compile axis: half/float} x {reversedBox TPL bool: 0/1}
 * => 4 keys: K0(float,0) K1(float,1) K2(half,0) K3(half,1).
 * dtype is driven by the DTYPE_BOXES compile macro (binary.json variants), so only
 * reversedBox is carried by the TPL mechanism here. batch-vs-num split is a runtime
 * TilingData decision (splitMode), not a TilingKey.
 */

#ifndef NORMALIZE_BBOX_TILING_KEY_H
#define NORMALIZE_BBOX_TILING_KEY_H
#include "ascendc/host_api/tiling/template_argument.h"

// template parameter space
ASCENDC_TPL_ARGS_DECL(NormalizeBBox, ASCENDC_TPL_BOOL_DECL(reversedBox, 0, 1));

// template parameter combinations (host GET_TPL_TILING_KEY validates key legality)
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_BOOL_SEL(reversedBox, 0, 1)), );

#endif // NORMALIZE_BBOX_TILING_KEY_H
