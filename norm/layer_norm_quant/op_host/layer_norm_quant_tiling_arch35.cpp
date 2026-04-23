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
 * \file layer_norm_quant_tiling.cpp
 * \brief
 */

#include "layer_norm_quant_tiling.h"
#include "layer_norm_quant_tiling_base.h"
#include "register/op_def_registry.h"
#include "op_host/tiling_templates_registry.h"

namespace optiling {
using namespace Ops::NN::Optiling;
constexpr uint32_t LAYER_NORM_TILING_KEY_BASE = 2000000000;
constexpr uint32_t LAYER_NORM_TILING_KEY_BF16_DTYPE = 200000000; 
constexpr uint32_t LAYER_NORM_TILING_KEY_FP16_DTYPE = 300000000;
constexpr uint32_t LAYER_NORM_TILING_KEY_FP32_DTYPE = 400000000;
constexpr uint32_t LAYER_NORM_TILING_KEY_FAST = 10000000;   // 0: fast; 1:slice

constexpr uint32_t LAYER_NORM_HALF_SIZE = 2;
constexpr uint32_t LAYER_NORM_FP32_SIZE = 4;

constexpr int64_t FP16_DATA_USED = 5;
constexpr int64_t FP16_OTHER_USED = 12;
constexpr uint32_t SCALAR_USED = 50;
constexpr uint32_t NUM_TEMP_BUF = 64;
constexpr uint32_t MEAN_AND_VAR_SIZE = 256;
uint32_t dtypeKey = LAYER_NORM_TILING_KEY_FP32_DTYPE;
constexpr uint32_t BLOCK_SIZE = 32;

size_t roundDownReg(size_t size, size_t divisor)
{
    if (divisor == 0) {
        return size;
    }
    return size / divisor * divisor;
}

uint32_t GetDataTypeSize(ge::DataType XType) {
    if (XType == ge::DataType::DT_BF16 || XType == ge::DataType::DT_FLOAT16){
        return LAYER_NORM_HALF_SIZE;
    } else if (XType == ge::DataType::DT_FLOAT) {
        return LAYER_NORM_FP32_SIZE;
    }
    return LAYER_NORM_FP32_SIZE;
}

uint32_t GetDataTypeKey(ge::DataType XType) {
    if (XType == ge::DataType::DT_BF16){
        return LAYER_NORM_TILING_KEY_BF16_DTYPE;
    } else if (XType == ge::DataType::DT_FLOAT16) {
        return LAYER_NORM_TILING_KEY_FP16_DTYPE;
    } else if (XType == ge::DataType::DT_FLOAT) {
        return LAYER_NORM_TILING_KEY_FP32_DTYPE;
    }
    return LAYER_NORM_TILING_KEY_FP32_DTYPE;
}

ge::graphStatus LayerNormQuantRegTiling::GetTilingSliceInfo()
{
    uint32_t singleRowSizePerElem = fp32BufNum * sizeof(uint32_t) + fp16BufNum * dtypeSize; // 3*4 + 2*2  --  3*4 + 2*4
    uint32_t multiRowSizePerElem = fp16BufNumForMulRow * dtypeSize + i8BufNumForMulRow * sizeof(uint8_t); // 2*2 +1  --  2*4 +1

    OP_CHECK_IF(colsAligned > (UINT_MAX / (singleRowSizePerElem + multiRowSizePerElem)),
                    OP_LOGE(context->GetNodeName(), "RowBufferSize invalid!"),
                    return ge::GRAPH_FAILED);
    uint32_t singleRowBufferSize = singleRowSizePerElem * colsAligned;
    uint32_t multiRowBufferSize = multiRowSizePerElem * colsAligned;

    if ((maxUbSize - MEAN_AND_VAR_SIZE) < (singleRowBufferSize + multiRowBufferSize)) {
        uint32_t oneRepeatElemCount = 256U / dtypeSize;
        uint32_t elemSize = roundDownReg((maxUbSize - MEAN_AND_VAR_SIZE) / (singleRowSizePerElem + multiRowSizePerElem),
                                      oneRepeatElemCount);
        tilingData.set_sliceNum(CeilDiv(numCol, elemSize));
        tilingData.set_sliceSize(elemSize);
        tilingData.set_tailSliceSize(numCol - (tilingData.get_sliceNum() - 1) * elemSize);
    } else {
        tilingData.set_sliceNum(1);
        tilingData.set_sliceSize(colsAligned);
        tilingData.set_tailSliceSize(numCol);
    }
    return ge::GRAPH_SUCCESS;
}

void LayerNormQuantRegTiling::GetTilingBasicInfo()
{
    float tempAve = float(1.0 / numCol);
    tilingData.set_aveStr(tempAve);
    tilingData.set_numLastDim(numCol);
    tilingData.set_colsAligned(colsAligned);
    uint32_t numCore = layerNormPtrCon.numCore;
    tilingData.set_numCore(numCore);
    uint32_t numRow = layerNormPtrCon.numRow;
    tilingData.set_numFirstDim(numRow);
    uint32_t nlFirstdimPerCoreNum = layerNormPtrCon.nlFirstdimPerCoreNum;
    tilingData.set_nlFirstdimPerCore(nlFirstdimPerCoreNum);
    tilingData.set_lFirstdimPerCore(numRow - nlFirstdimPerCoreNum * (numCore - 1));
}

ge::graphStatus LayerNormQuantRegTiling::DoTiling()
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto eps = attrs->GetAttrPointer<float>(EPSILON_ATTR_INDEX);
    this->tilingData.set_epsStr(*eps);

    ge::graphStatus ret =
        PostLayerNormPtrFunc<LayerNormQuantRegTilingData>(&this->tilingData, this->layerNormPtrCon, this->context);
    if (ret == ge::GRAPH_FAILED) {  // OP_TILING_CHECK_STATUS_RETURN(ret);
        return ret;
    }

    this->maxUbSize = layerNormPtrCon.maxUbSize;  // maxUb
    this->numCol = layerNormPtrCon.numCol;
    this->colsAligned = (this->numCol + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;  // 对齐后

    auto XDesc = context->GetInputDesc(0);
    auto XType = ge::DT_FLOAT;
    if (XDesc != nullptr) {
        XType = XDesc->GetDataType();
    }
    this->dtypeSize = GetDataTypeSize(XType);

    GetTilingBasicInfo();
    GetTilingSliceInfo();

    if (tilingData.get_sliceNum() == 1) { 
        uint64_t totalMemNeed =
            static_cast<uint64_t>(FP16_DATA_USED) * layerNormPtrCon.nlFirstdimPerCoreNum * colsAligned;
        OP_CHECK_IF(colsAligned > (UINT_MAX / FP16_OTHER_USED),
                        OP_LOGE(context->GetNodeName(), "sumData is invalid!"), return ge::GRAPH_FAILED);
        uint32_t sumData = (layerNormPtrCon.maxUbSize - NUM_TEMP_BUF -
                           static_cast<uint32_t>(FP16_OTHER_USED) * colsAligned - SCALAR_USED) / dtypeSize;
        OP_CHECK_IF(CeilDiv(totalMemNeed, static_cast<uint64_t>(sumData)) > UINT_MAX ,
                        OP_LOGE(context->GetNodeName(), "totalMemNeed is invalid!"), return ge::GRAPH_FAILED);
        ret = CheckSplit(&tilingData, totalMemNeed, static_cast<uint64_t>(sumData), layerNormPtrCon, context);
        if (ret == ge::GRAPH_FAILED) {
            return ret;
        }
    } else {
        tilingData.set_firstDimPerTimes(1);
    }
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    dtypeKey = GetDataTypeKey(XType);

    uint64_t tilingKey = LAYER_NORM_TILING_KEY_BASE;
    tilingKey += dtypeKey;
    tilingKey += tilingData.get_sliceNum() == 1 ? LAYER_NORM_TILING_KEY_FAST : 0;
    context->SetTilingKey(tilingKey);  // 2000000000 + 100000000 + 10000000
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    PrintTilingRegBaseData();

    return ge::GRAPH_SUCCESS;
}

void LayerNormQuantRegTiling::PrintTilingRegBaseData()
{
    OP_LOGD(context, "Start LayerNormQuantRegTiling printing");
    OP_LOGD(context, "------------------------------------------");
    OP_LOGD(context, "------------------------------------------");
    OP_LOGD(context, "numCore is %u", tilingData.get_numCore());
    OP_LOGD(context, "numLastDim is %u", tilingData.get_numLastDim());
    OP_LOGD(context, "colsAligned is %u", tilingData.get_colsAligned());
    OP_LOGD(context, "numFirstDim is %u", tilingData.get_numFirstDim());   
    OP_LOGD(context, "nlFirstdimPerCore is %u", tilingData.get_nlFirstdimPerCore());
    OP_LOGD(context, "lFirstdimPerCore is %u", tilingData.get_lFirstdimPerCore());
    OP_LOGD(context, "firstDimPerTimes is %u", tilingData.get_firstDimPerTimes());
    OP_LOGD(context, "epsStr is %f", tilingData.get_epsStr());
    OP_LOGD(context, "aveStr is %f", tilingData.get_aveStr());
    OP_LOGD(context, "sliceNum is %u", tilingData.get_sliceNum());
    OP_LOGD(context, "sliceSize is %u", tilingData.get_sliceSize());
    OP_LOGD(context, "tailSliceSize is %u", tilingData.get_tailSliceSize());
    OP_LOGD(context, "tilingKey is %lu", context->GetTilingKey());
    OP_LOGD(context, "------------------------------------------");
    OP_LOGD(context, "------------------------------------------");
    OP_LOGD(context, "End LayerNormQuantRegTiling printing");
}

}   // namespace optiling
