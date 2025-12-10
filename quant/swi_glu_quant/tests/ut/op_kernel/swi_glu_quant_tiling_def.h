/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/


 #ifndef TEST_SWI_GLU_QUANT_H
 #define TEST_SWI_GLU_QUANT_H

 #include "kernel_tiling/kernel_tiling.h"

 #define DT_BF16 bfloat16_t
 #define ORIG_DTYPE_START DT_BF16
 #define DTYPE_Y int8_t
 #define __CCE_UT_TEST__

 #pragma pack(1)

 struct SwiGluQuantTilingData {
     uint32_t groupLen;
     uint32_t rowLen;
     uint32_t colLen;
     uint32_t rowLenPerHeadCore;
     uint32_t rowLenPerTailCore;
     uint32_t basicRowLenHeadCore;
     uint32_t basicRowLenTailCore;
     uint32_t basicColLen;
     uint32_t headCoreNum;
     uint32_t realCoreNum;
     uint32_t activateLeft;
     int64_t groupListType;
     int64_t dstType;
     int64_t hasGroup;
     uint32_t tilingKey;
 };


 #pragma pack()

 #define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer)  \
     __ubuf__ tilingStruct* tilingDataPointer =                                 \
         reinterpret_cast<__ubuf__ tilingStruct*>((__ubuf__ uint8_t*)(tilingPointer));

 #define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer)     \
     CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);

 #define GET_TILING_DATA(tilingData, tilingPointer)                            \
     SwiGluQuantTilingData tilingData;                                               \
     INIT_TILING_DATA(SwiGluQuantTilingData, tilingDataPointer, tilingPointer);  \
     (tilingData).groupLen = tilingDataPointer->groupLen;                          \
     (tilingData).rowLen = tilingDataPointer->rowLen;              \
     (tilingData).colLen = tilingDataPointer->colLen;            \
     (tilingData).rowLenPerHeadCore = tilingDataPointer->rowLenPerHeadCore;  \
     (tilingData).rowLenPerTailCore = tilingDataPointer->rowLenPerTailCore;  \
     (tilingData).basicRowLenHeadCore = tilingDataPointer->basicRowLenHeadCore;  \
     (tilingData).basicRowLenTailCore = tilingDataPointer->basicRowLenTailCore;  \
     (tilingData).basicColLen = tilingDataPointer->basicColLen;  \
     (tilingData).headCoreNum = tilingDataPointer->headCoreNum;  \
     (tilingData).realCoreNum = tilingDataPointer->realCoreNum;  \
     (tilingData).activateLeft = tilingDataPointer->activateLeft;  \
     (tilingData).groupListType = tilingDataPointer->groupListType;  \
     (tilingData).dstType = tilingDataPointer->dstType;  \
     (tilingData).hasGroup = tilingDataPointer->hasGroup;  \
     (tilingData).tilingKey = tilingDataPointer->tilingKey;
 #endif //TEST_SWI_GLU_QUANT_H