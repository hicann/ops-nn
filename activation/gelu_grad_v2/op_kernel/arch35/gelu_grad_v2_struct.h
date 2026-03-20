/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file gelu_grad_v2_struct.h
 * \brief
 */

 #ifndef CANN_CUSTOM_OPS_GELU_GRAD_V2_STRUCT_H_
 #define CANN_CUSTOM_OPS_GELU_GRAD_V2_STRUCT_H_
 
 #include "atvoss/broadcast/broadcast_base_struct.h"
 
 #define TPL_NONE 0
 #define TPL_TANH 1
 
 namespace GeluGradV2Op
 {
 ASCENDC_TPL_ARGS_DECL(GeluGradV2,
     BRC_TEMP_SCH_MODE_KEY_DECL(schMode),
     ASCENDC_TPL_UINT_DECL(approximate, 1, ASCENDC_TPL_UI_LIST, TPL_NONE, TPL_TANH)
 );
 
 ASCENDC_TPL_SEL(
     ASCENDC_TPL_ARGS_SEL(
         BRC_TEMP_SCH_MODE_KEY_SEL(schMode),
         ASCENDC_TPL_UINT_SEL(approximate, ASCENDC_TPL_UI_LIST, TPL_NONE, TPL_TANH)
     )
 );
 }
 
 #endif //  CANN_CUSTOM_OPS_GELU_GRAD_V2_STRUCT_H_