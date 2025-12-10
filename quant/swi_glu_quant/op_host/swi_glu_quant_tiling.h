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
 * \file swi_glu_quant_tiling.h
 * \brief
 */

 #ifndef SWI_GLU_QUANT_TILING_H
 #define SWI_GLU_QUANT_TILING_H

 #include "register/tilingdata_base.h"
 #include "register/op_impl_registry.h"
 #include "util/math_util.h"            // Ops::Base::CeilAlign等函数在此，
 #include "log/log.h"                   // OP_CHECK_IF ，OP_LOGD 函数在此。 可以在base仓查看。
 #include "tiling/platform/platform_ascendc.h"
 #include "platform/platform_infos_def.h"

 namespace optiling {
 BEGIN_TILING_DATA_DEF(SwiGluQuantTilingData)
     TILING_DATA_FIELD_DEF(uint32_t, groupLen);        // group_index的长度G
     TILING_DATA_FIELD_DEF(uint32_t, rowLen);          // 行数
     TILING_DATA_FIELD_DEF(uint32_t, colLen);          // 列数, 输入x的一半
     TILING_DATA_FIELD_DEF(uint32_t, rowLenPerHeadCore);   // 头核处理行数
     TILING_DATA_FIELD_DEF(uint32_t, rowLenPerTailCore);   // 尾核处理行数
     TILING_DATA_FIELD_DEF(uint32_t, basicRowLenHeadCore);     // 头核每次计算的行数
     TILING_DATA_FIELD_DEF(uint32_t, basicRowLenTailCore);     // 尾核每次计算的行数
     TILING_DATA_FIELD_DEF(uint32_t, basicColLen);     // 每次计算的列数
     TILING_DATA_FIELD_DEF(uint32_t, headCoreNum);     // 使用的head核数
     TILING_DATA_FIELD_DEF(uint32_t, realCoreNum);     // 实际使用的核数
     TILING_DATA_FIELD_DEF(uint32_t, activateLeft);
     TILING_DATA_FIELD_DEF(int64_t, groupListType);
     TILING_DATA_FIELD_DEF(int64_t, dstType);  // int8/int4量化
     TILING_DATA_FIELD_DEF(int64_t, hasGroup);  // 是否有分组
     TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
 END_TILING_DATA_DEF;
 
 REGISTER_TILING_DATA_CLASS(SwiGluQuant, SwiGluQuantTilingData)
 
 
 struct CoreCompileInfo {};
 
 struct SwiGluQuantCompileInfo {
     uint32_t totalCore = 1;
     uint32_t ubSize = 0;
     uint32_t inputDataByte = 2;
     uint32_t groupLength = 1;
     std::string curQuantMode = "dynamic";
     uint32_t activateLeft = 0;
     int64_t groupListType = 0;
     int64_t dstType = ge::DT_INT8;
     int64_t hasGroup = 0;
 
     uint32_t dataNumSingleUb = 1;      // UB空间可处理的最大数据量
     uint32_t block_num = 1;            // 32B对齐使用
     uint32_t cacheLineLen = 1;         // 512B对齐使用
     bool isPerTensor = true;           // 静态量化时判断是否为per-tensor
 };  // SwiGluQuantCompileInfo
 
 
 struct SwiGluQuantTilingParam {
     uint32_t optBaseRowLenHeadCore = 1;
     uint32_t optBaseRowLenTailCore = 1;
     uint32_t optBaseColLen = 1;
     uint32_t rowLenPerHeadCore = 0;
     uint32_t rowLenPerTailCore = 0;
     uint32_t headCoreNum = 0;
     uint32_t coreNumUsed = 0;
 }; // SwiGluQuantTilingParam
 
 enum class QuantMode : uint8_t {
    STATIC_PER_TENSOR = 0,
    STATIC_PER_CHANNEL,
    DYNAMIC
 };
 } // namespace optiling
 #endif // SWI_GLU_QUANT_TILING_H
 