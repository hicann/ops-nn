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
 * \file mx_to_block_mx_quant_struct.h
 * \brief Template argument definitions for MxToBlockMxQuant kernel selection.
 *        rowMode: 0 = -2 axis is multiple of 64 (aligned), 1 = not aligned (has tail rows)
 */

#ifndef _MX_TO_BLOCK_MX_QUANT_STRUCT_H_
#define _MX_TO_BLOCK_MX_QUANT_STRUCT_H_

#include "ascendc/host_api/tiling/template_argument.h"

namespace MxToBlockMxQuantOp {
#define TPL_ROW_ALIGNED 0
#define TPL_ROW_NOT_ALIGNED 1

ASCENDC_TPL_ARGS_DECL(MxToBlockMxQuant,
                      ASCENDC_TPL_UINT_DECL(rowMode, 1, ASCENDC_TPL_UI_LIST, TPL_ROW_ALIGNED, TPL_ROW_NOT_ALIGNED));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(rowMode, ASCENDC_TPL_UI_LIST, TPL_ROW_ALIGNED,
                                                          TPL_ROW_NOT_ALIGNED)));

} // namespace MxToBlockMxQuantOp

#endif // _MX_TO_BLOCK_MX_QUANT_STRUCT_H_
