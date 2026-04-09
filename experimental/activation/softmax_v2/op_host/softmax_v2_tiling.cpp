/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):

 * - Tu Yuanhang <@TuYHAAAAAA>
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
 * \file softmax_v2_tiling.cpp
 * \brief
*/

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/softmax_v2_tiling_data.h"
#include "../op_kernel/softmax_v2_tiling_key.h"

namespace optiling {

using namespace Ops::NN::OpTiling;
const uint32_t BLOCK_SIZE = 32;
const uint32_t BUFFER_NUM = 2;
const uint32_t BLOCK_DIM = 8;
const uint32_t DOUBLE = 2;//倍增的参数
static inline uint64_t AlignUp(uint64_t x, uint64_t a) {
    if (a == 0) {
        return x;  
    }
    return (x + a - 1) / a * a;
}

constexpr uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;

struct SoftmaxV2CompileInfo {};

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
    totalIdx = inputX->GetStorageShape().GetShapeSize();
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
    auto ascendcPlatform = platform_ascendc:: PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
// 可直接替换你的 SoftmaxV2TilingFunc 内部实现（保留函数签名）
static ge::graphStatus SoftmaxV2TilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    int64_t totalIdx = 0;
    ge::DataType dataType;
    OP_CHECK_IF(GetShapeAttrsInfo(context, totalIdx, dataType) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);
   
    SoftmaxV2TilingData* tiling = context->GetTilingData<SoftmaxV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(SoftmaxV2TilingData), 0, sizeof(SoftmaxV2TilingData)) != EOK,
    OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto xTensor = context->GetInputTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xTensor);
    const auto xShape = context->GetInputTensor(0)->GetOriginShape();
    const auto inputDataType = context->GetInputTensor(0)->GetDataType();
    auto attrs = context->GetAttrs();
    int32_t dim = 3;  // 默认值
    if (attrs) {
        const int64_t* attrA = attrs->GetInt(0);  
        if (attrA != nullptr) {
            dim = *attrA;
        }
    }
    uint32_t typeSize = sizeof(float);
    switch (inputDataType) {
        case ge::DT_FLOAT16: 
            typeSize = sizeof(uint16_t); 
            break;
        case ge::DT_FLOAT:   
            typeSize = sizeof(float);    
            break;
        default: // 不支持的数据类型
            return ge::GRAPH_FAILED;
        }
     //第一个输入 ->得到输入的形状 -> 每个维度相乘
    uint32_t totalLengthx = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    uint32_t totalLengthz = totalLengthx;
    int32_t dimNum = xShape.GetDimNum();
    int dimarr[dimNum]={};
    //八维存储
    for(int i=0;i<dimNum;i++){
        dimarr[i] = xShape.GetDim(i);
    }
    uint32_t rows =1;
    //总的行数
    for(int i=0;i<dimNum-1;i++){
        rows = rows*dimarr[i];
    }
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    //第一次求和的空间大小--除开选择的维度其余的相乘
    uint32_t sumspace = totalLengthx / dimarr[dim];
    uint32_t sumspacerows = sumspace/dimarr[dimNum-1];

    //核间划分
    uint32_t big_core_num = rows % BLOCK_DIM; 
    uint32_t small_core_num=BLOCK_DIM - big_core_num;
    uint32_t small_tile_length = rows/BLOCK_DIM; //小核分到的行数
    uint32_t big_tile_length = rows/BLOCK_DIM + 1; //大核分到的行数

    //核内划分
    int64_t core_tile_x1 = 1; // 对行维度进行划分
    //ub内的数据划分-- 考虑temp就是第一次循环需要求和的数量 
    auto FitsUB = [&](int64_t tile_x1) -> bool {
        uint64_t xBytes  = AlignUp((uint64_t)tile_x1  * dimarr[dimNum-1] * typeSize, BLOCK_SIZE);
        // 总 UB = 双缓冲 * (x + z + y + temp)
        uint64_t total = BUFFER_NUM * (xBytes *4 );
         // 预留 5% 余量
        return total <= ubSize ;
    };

    while (FitsUB(core_tile_x1) && core_tile_x1 * DOUBLE < big_tile_length) {// 调整核内分块参数以适应UB
        core_tile_x1 *= DOUBLE;//倍增
    }
    if (core_tile_x1 != 1) { // 判断是否倍增了 
        core_tile_x1 /= DOUBLE;  // 溯到倍增前的结果
    }
    if (!FitsUB(core_tile_x1)) {
            return ge::GRAPH_FAILED;
    }

    uint32_t small_tile_times = small_tile_length/core_tile_x1; 
    uint32_t big_tile_times = big_tile_length / core_tile_x1; 
    uint32_t small_tail_num= small_tile_length % core_tile_x1;
    uint32_t big_tail_num= big_tile_length % core_tile_x1;
    if(small_tail_num!=0){
        small_tile_times ++;
    }else{
        small_tail_num = core_tile_x1;
    }
    if(big_tail_num!=0){
        big_tile_times ++;
    }else{
        big_tail_num = core_tile_x1;
    }
    
    //求和的次数
    uint32_t sumtimes = dimarr[dim];
    //求和空间的份数
    //如果为零 则默认为1 ，否则为之前维度的乘积，
    uint32_t partnum = 1;
    if(dim==0){
        partnum = 1;
    }else{
        for(int i=0;i<dim;i++){
            partnum = partnum*dimarr[i];
        }
    }
    //每次可以搬的元素个数
    //如果为最后一个，则每次搬运一个，否则为之后所有维度的乘积
    uint32_t elenum = 1;
    if(dim==dimNum -1 ){
        elenum = 1;
    }else{
        for(int i= dim+1;i<dimNum;i++){
            elenum = elenum*dimarr[i];
        }
    }

    tiling->sumtimes = dim;
    tiling->small_tile_times = small_tile_times;
    tiling->big_tile_times = big_tile_times;
    tiling->small_tail_num = small_tail_num;
    tiling->big_tail_num = big_tail_num;
    tiling->totalLengthx = totalLengthx;
    tiling->totalLengthz = totalLengthz;
    tiling->big_core_num = big_core_num;
    tiling->small_core_num = small_core_num;
    tiling->small_tile_length = small_tile_length;
    tiling->big_tile_length = big_tile_length;
    tiling->core_tile_x1 = core_tile_x1;
    tiling->dimNum = dimNum;
    for (int i = 0; i < dimNum; ++i) {
    tiling->dimarr[i] = dimarr[i];
    }
    tiling->dim = dim;
    tiling->sumtimes = sumtimes;
    tiling->partnum = partnum;
    tiling->elenum = elenum;
    tiling->sumspace = sumspace;
    tiling->sumspacerows = sumspacerows;
    tiling->rows = rows;
    context->SetBlockDim(BLOCK_DIM);
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(SoftmaxV2).Tiling(SoftmaxV2TilingFunc);
} // namespace optiling