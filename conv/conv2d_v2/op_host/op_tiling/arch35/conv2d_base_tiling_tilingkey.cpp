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
 * \file conv2d_base_tiling_tilingkey.cpp
 * \brief
 */
#include "conv2d_base_tiling.h"
#include "../../../op_kernel/arch35/conv2d_v2_tilingkey.h"

namespace optiling {
namespace conv_ops_tiling {
using namespace Conv2DV2Key;

uint64_t Conv2dBaseTiling::GetL0PingPongVal()
{
    return static_cast<uint64_t>(tilingData_.conv2dApiTiling.get_pBufferFlag() & L0A_L0B_PB_FLAG_MASK);
}

uint64_t Conv2dBaseTiling::GetWeightTilingVal()
{
    if (flagInfo_.convGroupType != ConvGroupType::NORMAL_CONV) {
        return WEIGHT_OTHER;
    }
    bool kBL1FullloadFlag = false;
    bool nBL1FullloadFlag = false;
    uint64_t ci1 = ConvCeilDiv(shapeInfo_.ci, convOpsConstParams_.k0);
    uint64_t weightKSize = flagInfo_.enableC04Flag ? ConvAlignB(C04_CIN_SIZE *
                           shapeInfo_.kh * shapeInfo_.kw, convOpsConstParams_.k0) :
                           ci1 * shapeInfo_.kh * shapeInfo_.kw * convOpsConstParams_.k0;
    uint64_t singleCoreNSize = ConvAlignB(ConvCeilDiv(shapeInfo_.co, blockDimRes.nDim), convOpsConstParams_.n0);
    if (tilingData_.conv2dApiTiling.get_kBL1() == weightKSize) {
        kBL1FullloadFlag = true;
    }
 
    if (tilingData_.conv2dApiTiling.get_nBL1() == singleCoreNSize) {
        nBL1FullloadFlag = true;
    }
 
    if (kBL1FullloadFlag && nBL1FullloadFlag) {
        return FULLLOAD_BL1;
    }
 
    if (!kBL1FullloadFlag && tilingData_.conv2dApiTiling.get_nL0() == singleCoreNSize) {
        return ONLY_N_FULLLOAD_BL1_BL0;
    }
    return WEIGHT_OTHER;
}
 
uint64_t Conv2dBaseTiling::GetOutputOrderVal()
{
    if (flagInfo_.mSplitModeFlag) {
        return 1;
    }
    return 0;
}

uint64_t Conv2dBaseTiling::GetFmpTilingVal()
{
    if (flagInfo_.convGroupType != ConvGroupType::NORMAL_CONV) {
        return FMP_OTHER;
    }
    uint64_t ci1 = ConvCeilDiv(shapeInfo_.ci, convOpsConstParams_.k0);
    uint64_t fmpKSize = flagInfo_.enableC04Flag ? ConvAlignB(C04_CIN_SIZE *
                        shapeInfo_.kh * shapeInfo_.kw, convOpsConstParams_.k0) :
                        ci1 * shapeInfo_.kh * shapeInfo_.kw * convOpsConstParams_.k0;
    bool kAL1FullloadFlag = tilingData_.conv2dApiTiling.get_kAL1() == fmpKSize;
    if (flagInfo_.mSplitModeFlag) {
        return GetFmpTilingValForMSplit(kAL1FullloadFlag);
    }
    return GetFmpTilingValForHWSplit(kAL1FullloadFlag);
}
 
uint64_t Conv2dBaseTiling::GetFmpTilingValForMSplit(bool kAL1FullloadFlag)
{
    bool mL1FullloadFlag = tilingData_.conv2dApiTiling.get_innerBatch() == 1 ?
        tilingData_.conv2dApiTiling.get_singleCoreHo() <= tilingData_.conv2dApiTiling.get_hoL1() :
        tilingData_.conv2dApiTiling.get_innerBatch() == tilingData_.conv2dApiTiling.get_singleCoreBatch();
    bool mL0FullloadFlag = tilingData_.conv2dApiTiling.get_hoL1() == tilingData_.conv2dApiTiling.get_hoL0();
    if (kAL1FullloadFlag && mL1FullloadFlag) {
        return FULLLOAD_AL1;
    } else if (!kAL1FullloadFlag && mL1FullloadFlag && mL0FullloadFlag) {
        return ONLY_M_FULLLOAD_AL1_AL0;
    }
    return FMP_OTHER;    
}
 
uint64_t Conv2dBaseTiling::GetFmpTilingValForHWSplit(bool kAL1FullloadFlag)
{
    bool hoL1FullloadFlag = tilingData_.conv2dApiTiling.get_singleCoreHo() <= tilingData_.conv2dApiTiling.get_hoL1();
    bool woL1FullloadFlag = false;
    if (tilingData_.conv2dApiTiling.get_singleCoreWo() <= tilingData_.conv2dApiTiling.get_woL1() &&
        !(ConvCeilDiv(tilingData_.conv2dApiTiling.get_singleCoreWo(), tilingData_.conv2dApiTiling.get_woL0()) > 1 &&
        tilingData_.conv2dApiTiling.get_singleCoreWo() % convOpsConstParams_.m0 > 0 &&
        tilingData_.conv2dApiTiling.get_hoL0() > 1)) {
        woL1FullloadFlag = true;
    }
    bool hoL0FullloadFlag = tilingData_.conv2dApiTiling.get_hoL1() == tilingData_.conv2dApiTiling.get_hoL0();
    bool woL0FullloadFlag = tilingData_.conv2dApiTiling.get_woL1() == tilingData_.conv2dApiTiling.get_woL0();
    if (kAL1FullloadFlag && hoL1FullloadFlag && woL1FullloadFlag) {
        return FULLLOAD_AL1;
    } else if (!kAL1FullloadFlag && hoL1FullloadFlag && hoL0FullloadFlag && woL1FullloadFlag && woL0FullloadFlag) {
        return ONLY_M_FULLLOAD_AL1_AL0;
    }
    return FMP_OTHER;
}
 
uint64_t Conv2dBaseTiling::GetL1PingPongVal()
{
    uint64_t l1PingPong = static_cast<uint64_t>(tilingData_.conv2dApiTiling.get_pBufferFlag() &
        L1A_L1B_PB_FLAG_MASK) >> L1_PB_OFFSET;
    // in group conv: only care about bl1 pingpong
    if (flagInfo_.convGroupType != ConvGroupType::NORMAL_CONV) {
        if (l1PingPong == L1_PB_BL1_OPEN || l1PingPong == L1_PB_ALL_OPEN) {
            return L1_PB_ALL_OPEN; // BL1_OPEN
        } else {
            return L1_PB_ALL_CLOSE; // BL1_CLOSE
        }
    }
    return l1PingPong;
}

uint64_t Conv2dBaseTiling::GetWeightUbTrans()
{
    // bUbKStep is always 0 except weight ub trans mode.
    if (tilingData_.conv2dApiTiling.get_bUbNStep() > 0 && tilingData_.conv2dApiTiling.get_bUbKStep() > 0) {
        return WEIGHT_UB_TRANS_OPEN;
    }

    return WEIGHT_UB_TRANS_CLOSE;
}

uint64_t Conv2dBaseTiling::GetEnableInnerBatch()
{
    if (tilingData_.conv2dApiTiling.get_innerBatch() > 1) {
        return shapeInfo_.kh == 1 && shapeInfo_.kw == 1 && attrInfo_.padTop == 0 && attrInfo_.padBottom == 0 &&
            attrInfo_.padLeft == 0 && attrInfo_.padRight == 0 && attrInfo_.strideH == 1 && attrInfo_.strideW == 1 &&
            attrInfo_.dilationH == 1 && attrInfo_.dilationW == 1 ?
            CONV_INNER_BATCH_KERNEL_1X1_MULTI : CONV_INNER_BATCH_MULTI;
    }
    return CONV_INNER_BATCH_SINGLE;
}

uint64_t Conv2dBaseTiling::GetFmapCopyMode()
{
    // bUbKStep is always 0 except weight ub trans mode.
    if (tilingData_.conv2dApiTiling.get_khUb() > 0 && tilingData_.conv2dApiTiling.get_kwUb() > 0) {
        return FMAP_DMA_MODE;
    }
 
    return FMAP_LOAD3D_MODE;
}

void Conv2dBaseTiling::ReSetTilingKeyPara()
{
    if (flagInfo_.convGroupType != ConvGroupType::NORMAL_CONV) {
        return;
    }
    bool weightTilingResetFlag = tilingKeyPara_.weightTiling == 1 &&
                                 !(tilingKeyPara_.fmpTiling == 1 && tilingKeyPara_.outputOrder == 1);
    if (weightTilingResetFlag) {
        tilingKeyPara_.weightTiling = WEIGHT_OTHER;
    }
 
    bool fmpTilingResetFlag = tilingKeyPara_.fmpTiling == 1 &&
                              !(tilingKeyPara_.weightTiling == 1 && tilingKeyPara_.outputOrder == 1);
    if (fmpTilingResetFlag) {
        tilingKeyPara_.fmpTiling = FMP_OTHER;
    }
}

ge::graphStatus Conv2dBaseTiling::SetTilingKey()
{
    tilingKeyPara_.fmpTiling = GetFmpTilingVal();
    tilingKeyPara_.weightTiling = GetWeightTilingVal();
    tilingKeyPara_.l1PingPong = GetL1PingPongVal();
    tilingKeyPara_.l0PingPong = GetL0PingPongVal();
    tilingKeyPara_.outputOrder = GetOutputOrderVal();
    tilingKeyPara_.iterOrder = static_cast<uint64_t>(tilingData_.conv2dApiTiling.get_iterateMNOrder());
    tilingKeyPara_.groupType = static_cast<uint64_t>(flagInfo_.convGroupType);
    tilingKeyPara_.enableSmallChannel = flagInfo_.enableC04Flag;
    tilingKeyPara_.weightUbTrans = GetWeightUbTrans();
    tilingKeyPara_.fmapCppyMode = GetFmapCopyMode();
    tilingKeyPara_.innerBatch =  GetEnableInnerBatch();
    ReSetTilingKeyPara();
    tilingKey_ = GET_TPL_TILING_KEY(tilingKeyPara_.fmpTiling,
                                    tilingKeyPara_.weightTiling,
                                    tilingKeyPara_.l1PingPong,
                                    tilingKeyPara_.l0PingPong,
                                    tilingKeyPara_.outputOrder,
                                    tilingKeyPara_.iterOrder,
                                    tilingKeyPara_.groupType,
                                    tilingKeyPara_.enableSmallChannel,
                                    tilingKeyPara_.weightUbTrans,
                                    tilingKeyPara_.fmapCppyMode,
                                    tilingKeyPara_.innerBatch);

    OP_LOGD(context_->GetNodeName(), "%s AscendC: c04 mode status is: %d",
            paramInfo_.nodeType.c_str(), flagInfo_.enableC04Flag);
    OP_LOGD(context_->GetNodeName(), "%s AscendC: tiling key: %lu.",
            paramInfo_.nodeType.c_str(), tilingKey_);
    return ge::GRAPH_SUCCESS;
}
}
}