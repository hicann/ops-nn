/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Li Wen <@liwenkkklll>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file logsigmoid_v2_tiling.cpp
 * \brief
 */

 #include "log/log.h"
 #include "util/math_util.h"
 #include "op_host/tiling_util.h"
 #include "op_host/tiling_templates_registry.h"
 #include "../op_kernel/logsigmoid_v2_tiling_data.h"
 #include "../op_kernel/logsigmoid_v2_tiling_key.h"
 
 namespace optiling {
 
 using namespace Ops::NN::OpTiling;
 const uint32_t BLOCK_SIZE = 32;
 const uint32_t BUFFER_NUM = 2;
 const uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;
 
 struct LogsigmoidV2CompileInfo {};
 
 // 获取平台信息如ubSize, coreNum
 static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
 {
     // 获取ubsize coreNum
     fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
     OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
     auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
     coreNum = ascendcPlatform.GetCoreNumAiv();
     OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
     ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
     OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
     return ge::GRAPH_SUCCESS;
 }
 
 // 获取属性，shape信息
 static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalIdx, ge::DataType& dataType)
 {
     // 获取输入shape信息
     auto inputX = context->GetInputShape(0);
     OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
     // 如果输入shape 是标量 转换为{1}，否则保持原 shape 不变
     auto inputShapeX = inputX->GetStorageShape();
     
     auto outZ = context->GetOutputShape(0);
     OP_CHECK_NULL_WITH_CONTEXT(context, outZ);
     auto outShapeZ = outZ->GetStorageShape();
 
     // shape校验
     bool shapeMatch = true;
     // 校验维度数一致
     if (inputShapeX.GetDimNum() != outShapeZ.GetDimNum()) {
         shapeMatch = false;
     } else {
         // 校验每个维度的大小一致
         size_t dimNum = inputShapeX.GetDimNum();
         for (size_t i = 0; i < dimNum; i++) {
             if (inputShapeX.GetDim(i) != outShapeZ.GetDim(i)) {
                 shapeMatch = false;
                 break;
             }
         }
     }
     // 形状不匹配则报错
     OP_CHECK_IF(
         !shapeMatch,
         OP_LOGE(
             context, "LogsigmoidV2: inputx,outputz shape not match! dim num: x=%zu,  z=%zu",
             inputShapeX.GetDimNum(), outShapeZ.GetDimNum()),
         return ge::GRAPH_FAILED);
     
     totalIdx = inputX->GetOriginShape().GetShapeSize();
     // dtype校验
     const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16};
     auto inputDesc = context->GetInputDesc(0);
     OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
     dataType = inputDesc->GetDataType();
     if (supportedDtype.count(dataType) == 0) {
         OP_LOGE(context, "invalid dtype");
         return ge::GRAPH_FAILED;
     }
     return ge::GRAPH_SUCCESS;
 }
 
 static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
 {
     size_t* currentWorkspace = context->GetWorkspaceSizes(1);
     OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
     currentWorkspace[0] = WS_SYS_SIZE;
     return ge::GRAPH_SUCCESS;
 }
 
 // tiling 分发入口
 // 可直接替换你的 LogsigmoidV2TilingFunc 内部实现（保留函数签名）
 static ge::graphStatus LogsigmoidV2TilingFunc(gert::TilingContext* context)
 {
     // 1. platform
     uint64_t ubSize = 0;
     int64_t coreNum = 0;
     OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                 OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
 
     // 2. shapes & dtype
     int64_t totalIdx = 0;
     ge::DataType dataType;
     OP_CHECK_IF(GetShapeAttrsInfo(context, totalIdx, dataType) != ge::GRAPH_SUCCESS,
                 OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);
 
     // 3. workspace
     OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                 OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);
 
     LogsigmoidV2TilingData* tiling = context->GetTilingData<LogsigmoidV2TilingData>();
     OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
     OP_CHECK_IF(memset_s(tiling, sizeof(LogsigmoidV2TilingData), 0, sizeof(LogsigmoidV2TilingData)) != EOK,
                 OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
 
     // --- safer numeric types ---
     uint32_t typeLength = 0;
     ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
     if (typeLength == 0) {
         OP_LOGE(context, "typeLength is 0");
         return ge::GRAPH_FAILED;
     }
     uint64_t inputBytes = static_cast<uint64_t>(typeLength);
     uint64_t inputLengthBytes = static_cast<uint64_t>(totalIdx) * inputBytes;
 
     // ub-based tileBlockNum guard (避免为0)
     uint32_t ubDataNumber = (inputBytes == 1ULL) ? 4U : 2U;
     uint64_t tmp = (ubSize / BLOCK_SIZE / BUFFER_NUM);
     uint32_t tileBlockNum = 1U;
     if (tmp > 0) {
         uint64_t tb = tmp / ubDataNumber;
         tileBlockNum = (tb == 0) ? 1U : static_cast<uint32_t>(tb);
     }
 
     // 每个 tile 包含的元素数（至少 1）
     uint32_t tileDataNum = static_cast<uint32_t>((static_cast<uint64_t>(tileBlockNum) * BLOCK_SIZE) / inputBytes);
     if (tileDataNum == 0U) tileDataNum = 1U;
 
     // 总 block 数（向上取整）
     uint64_t blocksTotal = (inputLengthBytes + BLOCK_SIZE - 1ULL) / BLOCK_SIZE;
     uint64_t coreNum64 = static_cast<uint64_t>(coreNum);
     if (coreNum64 > blocksTotal){
         coreNum64 = blocksTotal;
     }
     if (coreNum64 == 0ULL) coreNum64 = 1ULL; // 最少 1 core
     uint32_t finalCoreNum = static_cast<uint32_t>(coreNum64);
 
     uint64_t everyCoreInputBlockNum = blocksTotal / coreNum64; // 基本块数
     uint32_t tailBlockNum = static_cast<uint32_t>(blocksTotal % coreNum64); // 前 tailBlockNum 个核是 big-core
 
     // small-core 数量（元素）
     uint64_t smallCoreDataNum_u = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
     uint32_t smallCoreDataNum = static_cast<uint32_t>(smallCoreDataNum_u);
 
     uint32_t smallTileNum = static_cast<uint32_t>(everyCoreInputBlockNum / static_cast<uint64_t>(tileBlockNum));
     uint32_t finalSmallTileNum = ((everyCoreInputBlockNum % tileBlockNum) == 0) ? smallTileNum : (smallTileNum + 1);
     int64_t smallTailDataNum_i = static_cast<int64_t>(smallCoreDataNum) - static_cast<int64_t>(tileDataNum) * static_cast<int64_t>(smallTileNum);
     uint32_t smallTailDataNum = (smallTailDataNum_i <= 0) ? tileDataNum : static_cast<uint32_t>(smallTailDataNum_i);
 
     // big-core（每个多一个 block）
     uint64_t bigEveryCoreBlockNum = everyCoreInputBlockNum + 1ULL;
     uint64_t bigCoreDataNum_u = bigEveryCoreBlockNum * BLOCK_SIZE / inputBytes;
     uint32_t bigCoreDataNum = static_cast<uint32_t>(bigCoreDataNum_u);
     uint32_t bigTileNum = static_cast<uint32_t>(bigEveryCoreBlockNum / static_cast<uint64_t>(tileBlockNum));
     uint32_t finalBigTileNum = ((bigEveryCoreBlockNum % tileBlockNum) == 0) ? bigTileNum : (bigTileNum + 1);
     int64_t bigTailDataNum_i = static_cast<int64_t>(bigCoreDataNum) - static_cast<int64_t>(tileDataNum) * static_cast<int64_t>(bigTileNum);
     uint32_t bigTailDataNum = (bigTailDataNum_i <= 0) ? tileDataNum : static_cast<uint32_t>(bigTailDataNum_i);
 
     // write back
     tiling->smallCoreDataNum = static_cast<int64_t>(smallCoreDataNum);
     tiling->bigCoreDataNum = static_cast<int64_t>(bigCoreDataNum);
     tiling->tileDataNum = static_cast<int64_t>(tileDataNum);
     tiling->smallTailDataNum = static_cast<int64_t>(smallTailDataNum);
     tiling->bigTailDataNum = static_cast<int64_t>(bigTailDataNum);
     tiling->finalSmallTileNum = static_cast<int64_t>(finalSmallTileNum);
     tiling->finalBigTileNum = static_cast<int64_t>(finalBigTileNum);
     tiling->tailBlockNum = static_cast<int64_t>(tailBlockNum);
 
     context->SetBlockDim(finalCoreNum);
     return ge::GRAPH_SUCCESS;
 }
 
 
 static ge::graphStatus TilingParseForLogsigmoidV2([[maybe_unused]] gert::TilingParseContext* context)
 {
     return ge::GRAPH_SUCCESS;
 }
 
 // tiling注册入口.
 IMPL_OP_OPTILING(LogsigmoidV2).Tiling(LogsigmoidV2TilingFunc).TilingParse<LogsigmoidV2CompileInfo>(TilingParseForLogsigmoidV2);
 } // namespace optiling