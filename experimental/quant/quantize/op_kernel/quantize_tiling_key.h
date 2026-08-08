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
 * \file quantize_tiling_key.h
 * \brief Quantize tiling-key template declaration, shared by host tiling and kernel.
 *
 * Single template axis `perMode` selects the schedule. Both host and kernel include this header, so the key
 * numbering has exactly one definition:
 *   - host  : ASCENDC_TPL_ARGS_DECL expands to the declare-params table used by GET_TPL_TILING_KEY(perMode).
 *   - kernel: ASCENDC_TPL_SEL enumerates the instantiations of the templated `quantize<perMode>` entry.
 *
 * perMode is a 1-bit field, so the encoded tiling key equals the perMode value itself
 * (QUANTIZE_PER_CHANNEL = 0, QUANTIZE_PER_TENSOR = 1).
 */
#ifndef QUANTIZE_TILING_KEY_H
#define QUANTIZE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define QUANTIZE_PER_CHANNEL 0
#define QUANTIZE_PER_TENSOR 1

ASCENDC_TPL_ARGS_DECL(Quantize,
                      ASCENDC_TPL_UINT_DECL(perMode, 1, ASCENDC_TPL_UI_LIST, QUANTIZE_PER_CHANNEL,
                                            QUANTIZE_PER_TENSOR), );

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(perMode, ASCENDC_TPL_UI_LIST, QUANTIZE_PER_CHANNEL,
                                                          QUANTIZE_PER_TENSOR)), );

#endif // QUANTIZE_TILING_KEY_H
