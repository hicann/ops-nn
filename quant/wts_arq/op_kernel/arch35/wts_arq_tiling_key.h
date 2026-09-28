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
 * \file wts_arq_tiling_key.h
 * \brief WtsARQ TilingKey template argument declaration (arch35 / DAV_3510).
 *
 * TilingKey dimension: RANK in {4, 8} (collapsed shapeLen <= 4 -> RANK=4,
 * 5~8 -> RANK=8). dtype is derived at compile time by binary.json
 * (DTYPE_W macro), offset_flag/num_bits are runtime tiling fields.
 */
#ifndef WTS_ARQ_TILING_KEY_H_
#define WTS_ARQ_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define WTS_ARQ_RANK_4 4
#define WTS_ARQ_RANK_8 8

ASCENDC_TPL_ARGS_DECL(WtsARQ, ASCENDC_TPL_UINT_DECL(RANK, 8, ASCENDC_TPL_UI_LIST, WTS_ARQ_RANK_4, WTS_ARQ_RANK_8));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(RANK, ASCENDC_TPL_UI_LIST, WTS_ARQ_RANK_4)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(RANK, ASCENDC_TPL_UI_LIST, WTS_ARQ_RANK_8)));

#endif // WTS_ARQ_TILING_KEY_H_
