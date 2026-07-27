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
 * \file sync_batch_norm_backward_reduce_tiling_key.h
 * \brief tiling key template argument declaration.
 */
#ifndef __SYNCBNBR_TILING_KEY_H__
#define __SYNCBNBR_TILING_KEY_H__

#include "sync_batch_norm_backward_reduce_tiling_data.h"
#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(SyncBatchNormBackwardReduce,
                      ASCENDC_TPL_UINT_DECL(schMode, 2, ASCENDC_TPL_UI_LIST, SYNCBNBR_TPL_SCH_MODE_0,
                                            SYNCBNBR_TPL_SCH_MODE_1, SYNCBNBR_TPL_SCH_MODE_2));
ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, SYNCBNBR_TPL_SCH_MODE_0,
                                                          SYNCBNBR_TPL_SCH_MODE_1, SYNCBNBR_TPL_SCH_MODE_2)));
#endif
