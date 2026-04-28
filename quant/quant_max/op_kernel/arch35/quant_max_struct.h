/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __QUANT_MAX_STRUCT_H__
#define __QUANT_MAX_STRUCT_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define TPL_ROUND_MODE_RINT 0
#define TPL_ROUND_MODE_ROUND 1
#define TPL_ROUND_MODE_HYBRID 2

namespace QuantMaxOp {

ASCENDC_TPL_ARGS_DECL(
    QuantMax, ASCENDC_TPL_UINT_DECL(
                  roundMode, 2, ASCENDC_TPL_UI_LIST, TPL_ROUND_MODE_ROUND, TPL_ROUND_MODE_RINT, TPL_ROUND_MODE_HYBRID));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(
    roundMode, ASCENDC_TPL_UI_LIST, TPL_ROUND_MODE_ROUND, TPL_ROUND_MODE_RINT, TPL_ROUND_MODE_HYBRID)));

} // namespace QuantMaxOp

#endif // _ASCEND_QUANT_STRUCT_H_