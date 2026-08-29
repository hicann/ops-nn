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
 * \file unsorted_segment_prod_tiling_key.h
 * \brief unsorted_segment_prod_tiling_key
 */

#ifndef UNSORTED_SEGMENT_PROD_TILING_KEY_H_
#define UNSORTED_SEGMENT_PROD_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define USS_TEMPLATE_SIMT 0

#define USS_CAST_NONE 0

namespace UnsortedSegmentProd {
ASCENDC_TPL_ARGS_DECL(UnsortedSegmentProd,
                      ASCENDC_TPL_UINT_DECL(TEMPLATE_MODE, 1, ASCENDC_TPL_UI_LIST, USS_TEMPLATE_SIMT),
                      ASCENDC_TPL_UINT_DECL(CAST_MODE, 1, ASCENDC_TPL_UI_LIST, USS_CAST_NONE));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIV_1_0),
                                     ASCENDC_TPL_UINT_SEL(TEMPLATE_MODE, ASCENDC_TPL_UI_LIST, USS_TEMPLATE_SIMT),
                                     ASCENDC_TPL_UINT_SEL(CAST_MODE, ASCENDC_TPL_UI_LIST, USS_CAST_NONE),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(UnsortedSegment::UnsortedSegmentSimtTilingData)));
} // namespace UnsortedSegmentProd

#endif // UNSORTED_SEGMENT_PROD_TILING_KEY_H_
