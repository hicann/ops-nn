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
 * \file modulate_grad_tiling.cc
 * \brief
 */
#include "modulate_grad_tiling.h"
#include <cassert>
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "util/math_util.h"

namespace optiling{
    struct ModulateGradTilingData{
        uint32_t B;
        uint32_t L;
        uint32_t D;
        uint32_t dataType = 2;
        uint32_t block_dim;
        uint32_t dataTypeSize = 4;

        uint32_t splitB;
        uint32_t coresPerB;
        uint32_t usedCores;
        uint32_t formerNum;
        uint32_t formerLength;
        uint32_t tailNum;
        uint32_t tailLength;
        uint32_t has_scale;
        uint32_t has_shift;
    };

    constexpr uint32_t DATA_TYPE_SIZE[] = {2,2,4};
    constexpr uint32_t BLOCK_SIZE = 32;
    constexpr uint32_t BUFFER_NUM = 2;
    constexpr uint32_t UB_BLOCK_NUM = 384;
    constexpr uint32_t MAX_AVAILABLE_UB_BLOCK_NUM = UB_BLOCK_NUM / BUFFER_NUM * BUFFER_NUM;

    void PrintTilingData(gert::TilingContext* context, ModulateGradTilingData& tiling)
    {
        OP_LOGD(context,"B:%u.", tiling.B);
        OP_LOGD(context,"L:%u", tiling.L);
        OP_LOGD(context,"D:%u.", tiling.D);
        OP_LOGD(context,"dataType:%u.", tiling.dataType);
        OP_LOGD(context,"block_dim:%u.", tiling.block_dim);
        OP_LOGD(context,"dataTypeSize:%u.", tiling.dataTypeSize);
        OP_LOGD(context,"splitB:%u.", tiling.splitB);
        OP_LOGD(context,"coresPerB:%u.", tiling.coresPerB);
        OP_LOGD(context,"usedCores:%u.", tiling.usedCores);
        OP_LOGD(context,"formerLength:%u.", tiling.formerLength);
        OP_LOGD(context,"tailNum:%u.", tiling.tailNum);
        OP_LOGD(context,"tailLength:%u.", tiling.tailLength);
        OP_LOGD(context,"has_scale:%u.", tiling.has_scale);
        OP_LOGD(context,"has_shift:%u.", tiling.has_shift);
    }
    void GenerateTilingData(ModulateGradTilingData* tiling, uint32_t numBlocks){
        uint32_t B = tiling->B;
        uint32_t D = tiling->D;
        if (numBlocks == 0){
            return ;
        }
        numBlocks = numBlocks < B * D ? numBlocks : B * D;
        uint32_t splitBlock = numBlocks / 2; 
        tiling -> splitB = (B < splitBlock) ? 1 : 0;

        if (tiling->splitB){
            tiling->coresPerB = numBlocks / B;
            tiling->usedCores = tiling->coresPerB * B;

            uint32_t dPerCore = D / tiling -> coresPerB;
            uint32_t remainderD = D % tiling -> coresPerB;

            tiling->formerNum = remainderD;
            tiling->formerLength = dPerCore + 1;
            tiling->tailNum = tiling-> coresPerB - remainderD;
            tiling->tailLength = dPerCore; 
        }else{
            numBlocks = numBlocks < B ? numBlocks : B;
            tiling->coresPerB = 1;
            uint32_t bPerCore = B / numBlocks;
            tiling->usedCores = numBlocks;
            uint32_t remainderB = B % numBlocks;

            tiling->formerNum = (remainderB == 0) ? 0 : remainderB;
            tiling->formerLength = (remainderB == 0) ? 0 : bPerCore+1;
            tiling->tailNum = (remainderB == 0)? numBlocks : (numBlocks - remainderB);
            tiling->tailLength = bPerCore; 
        }
    }
    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        constexpr uint32_t INPUT_INDEX = 1;
        constexpr uint32_t SCALE_INDEX = 2;
        constexpr uint32_t SHIFT_INDEX = 3;
        uint32_t has_scale = 0;
        uint32_t has_shift = 0;
        context->SetTilingKey(0);
        ModulateGradTiling tilingData;
        ModulateGradTilingData tiling;
        const gert::StorageShape* input_shape = context->GetInputShape(INPUT_INDEX);
        auto scale_input = context->GetOptionalInputShape(SCALE_INDEX);
        auto shift_input = context->GetOptionalInputShape(SHIFT_INDEX);
        if (scale_input){
            has_scale = 1;
        }
        if (shift_input){
            has_shift = 1;
        }
        const uint32_t B = input_shape->GetStorageShape().GetDim(0);
        const uint32_t L = input_shape->GetStorageShape().GetDim(1);
        const uint32_t D = input_shape->GetStorageShape().GetDim(2);
        tiling.B = B;
        tiling.L = L;
        tiling.D = D;
        tiling.has_scale = has_scale;
        tiling.has_shift = has_shift;
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        const uint32_t block_dim = ascendcPlatform.GetCoreNumAiv();
        OP_LOGD(context, "ascendcPlatform CoreNum: %u.",block_dim);
        tiling.block_dim = block_dim;
        GenerateTilingData(&tiling, block_dim);
        context -> SetBlockDim(tiling.usedCores);

        tilingData.set_B(tiling.B);
        tilingData.set_L(tiling.L);
        tilingData.set_D(tiling.D);
        tilingData.set_dataType(tiling.dataType);
        tilingData.set_block_dim(tiling.block_dim);
        tilingData.set_dataTypeSize(tiling.dataTypeSize);
        tilingData.set_splitB(tiling.splitB);
        tilingData.set_coresPerB(tiling.coresPerB);
        tilingData.set_usedCores(tiling.usedCores);
        tilingData.set_formerNum(tiling.formerNum);
        tilingData.set_formerLength(tiling.formerLength);
        tilingData.set_tailNum(tiling.tailNum);
        tilingData.set_tailLength(tiling.tailLength);
        tilingData.set_has_scale(tiling.has_scale);
        tilingData.set_has_shift(tiling.has_shift);

        PrintTilingData(context, tiling);
        tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize();
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus TilingPrepareForModulateGrad(gert::TilingParseContext* context){
        OP_LOGD(context, "Tiling Prepare For ModulateGrad start.");
        auto compileInfo = context->GetCompiledInfo<ModulateGradCompileInfo>();
        OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
        auto platformInfo = context->GetPlatformInfo();
        OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
        if (compileInfo->totalCoreNum == 0) {
            OP_LOGE(context, "coreNum %u", compileInfo->totalCoreNum);
            return ge::GRAPH_FAILED;
        }
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        compileInfo->ubSizePlatForm = ubSizePlatForm;
        OP_LOGD(context, "ub_size_platform is %lu.", compileInfo->ubSizePlatForm);
        OP_LOGD(context, "Tiling Prepare For ModulateGrad end.");
        return ge::GRAPH_SUCCESS;
    }
    IMPL_OP_OPTILING(ModulateGrad)
            .Tiling(TilingFunc).TilingParse<ModulateGradCompileInfo>(TilingPrepareForModulateGrad);
}

