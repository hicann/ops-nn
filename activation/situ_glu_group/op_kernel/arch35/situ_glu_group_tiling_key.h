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
 * \file situ_glu_group_tiling_key.h
 * \brief SituGluGroup template-argument declaration (compile-time TilingKey dispatch).
 */
#ifndef SITU_GLU_GROUP_TILING_KEY_H_
#define SITU_GLU_GROUP_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

// The data type is not part of the TilingKey: the build registers DTYPE_X for the x input of each
// compiled binary, so the kernel reads the actual C++ type from that macro instead of encoding
// fp16/bf16/fp32 as a TilingKey axis.
#define SITU_GLU_GROUP_TPL_SCH_MODE_0 0

// Number of 32B slots reserved for the group_index reduce results: slot 0 is sum(group_index),
// slot 1 is the negative-element probe. Shared by the Host UB budget and the Kernel buffer pool so
// the two can never drift apart.
constexpr int64_t GROUP_INDEX_SUM_SLOT_NUM = 2;

ASCENDC_TPL_ARGS_DECL(SituGluGroup,
                      ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, SITU_GLU_GROUP_TPL_SCH_MODE_0));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST,
                                                          SITU_GLU_GROUP_TPL_SCH_MODE_0)));

#endif // SITU_GLU_GROUP_TILING_KEY_H_
