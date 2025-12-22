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
 * \file conv3d_v2_base_tiling_tilingkey.cpp
 * \brief
 */
#include "conv3d_v2_base_tiling.h"
#include "conv/common/op_kernel/arch35/conv_tilingkey.h"
#include "conv/conv3d_v2/op_kernel/arch35/conv3d_v2_tilingkey.h"

namespace optiling {
namespace conv_ops_tiling {
using namespace Conv3DV2Key;

uint64_t Conv3dBaseTilingV2::GetL0PingPongVal()
{
    return static_cast<uint64_t>(tilingData_.conv3dApiTiling.get_pBufferFlag() & L0A_L0B_PB_FLAG_MASK);
}

uint64_t Conv3dBaseTilingV2::GetWeightTilingVal()
{
    if (flagInfo_.convGroupType != ConvGroupType::NORMAL_CONV) {
        return WEIGHT_OTHER;
    }
    bool kBL1FullloadFlag = false;
    bool nBL1FullloadFlag = false;
    uint32_t k0 = CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo_.fMapDtype), MKN_K_IDX);
    uint32_t n0 = CUBE_MKN_MAP.GetMKN(dtypeMap.at(descInfo_.fMapDtype), MKN_N_IDX);
    uint64_t ci1 = ConvCeilDiv(shapeInfo_.ci, k0);
    uint64_t weightKSize = ci1 * shapeInfo_.kd * shapeInfo_.kh * shapeInfo_.kw * k0;
    uint64_t singleCoreNSize = ConvAlignB(ConvCeilDiv(shapeInfo_.co, blockDimRes.nDim), n0);
    if (tilingData_.conv3dApiTiling.get_kBL1() == weightKSize) {
        kBL1FullloadFlag = true;
    }
 
    if (tilingData_.conv3dApiTiling.get_nBL1() == singleCoreNSize) {
        nBL1FullloadFlag = true;
    }
 
    if (kBL1FullloadFlag && nBL1FullloadFlag) {
        return FULLLOAD_BL1;
    }
 
    if (!kBL1FullloadFlag && tilingData_.conv3dApiTiling.get_nL0() == singleCoreNSize) {
        return ONLY_N_FULLLOAD_BL1_BL0;
    }
    return WEIGHT_OTHER;
}

uint64_t Conv3dBaseTilingV2::GetOutputOrderVal()
{
    if (flagInfo_.mSplitModeFlag) {
        return 1;
    }
    return 0;
}

uint64_t Conv3dBaseTilingV2::GetFmpTilingValMMode(const bool kAL1FullloadFlag)
{
    bool mL1FullloadFlag = false;
    bool mL0FullloadFlag = false;
    mL1FullloadFlag = tilingData_.conv3dApiTiling.get_singleCoreHo() <= tilingData_.conv3dApiTiling.get_hoL1();
    mL0FullloadFlag = tilingData_.conv3dApiTiling.get_hoL1() == tilingData_.conv3dApiTiling.get_hoL0();
    if (kAL1FullloadFlag && mL1FullloadFlag) {
        return FULLLOAD_AL1;
    } else if (!kAL1FullloadFlag && mL1FullloadFlag && mL0FullloadFlag) {
        return ONLY_M_FULLLOAD_AL1_AL0;
    }
    return FMP_OTHER;
}

uint64_t Conv3dBaseTilingV2::GetFmpTilingValHWMode(const bool kAL1FullloadFlag)
{
    bool hoL1FullloadFlag = false;
    bool woL1FullloadFlag = false;
    bool hoL0FullloadFlag = false;
    bool woL0FullloadFlag = false;
    hoL1FullloadFlag = tilingData_.conv3dApiTiling.get_singleCoreHo() <= tilingData_.conv3dApiTiling.get_hoL1();
    if (tilingData_.conv3dApiTiling.get_singleCoreWo() <= tilingData_.conv3dApiTiling.get_woL1() &&
        !(ConvCeilDiv(tilingData_.conv3dApiTiling.get_singleCoreWo(), tilingData_.conv3dApiTiling.get_woL0()) > 1 &&
        tilingData_.conv3dApiTiling.get_singleCoreWo() % convOpsConstParams_.m0 > 0 &&
        tilingData_.conv3dApiTiling.get_hoL0() > 1)) {
        woL1FullloadFlag = true;
    }
    hoL0FullloadFlag = tilingData_.conv3dApiTiling.get_hoL1() == tilingData_.conv3dApiTiling.get_hoL0();
    woL0FullloadFlag = tilingData_.conv3dApiTiling.get_woL1() == tilingData_.conv3dApiTiling.get_woL0();
    if (kAL1FullloadFlag && hoL1FullloadFlag && woL1FullloadFlag) {
        return FULLLOAD_AL1;
    } else if (!kAL1FullloadFlag && hoL1FullloadFlag && hoL0FullloadFlag && woL1FullloadFlag && woL0FullloadFlag) {
        return ONLY_M_FULLLOAD_AL1_AL0;
    }
    return FMP_OTHER;
}

uint64_t Conv3dBaseTilingV2::GetFmpTilingVal()
{
    if (flagInfo_.convGroupType != ConvGroupType::NORMAL_CONV) {
        return FMP_OTHER;
    }
    bool kAL1FullloadFlag = false;

    uint64_t ci1 = ConvCeilDiv(shapeInfo_.ci, convOpsConstParams_.k0);
    uint64_t fmpKSize = ci1 * shapeInfo_.kh * shapeInfo_.kd * shapeInfo_.kw * convOpsConstParams_.k0;
    if (tilingData_.conv3dApiTiling.get_kAL1() == fmpKSize) {
        kAL1FullloadFlag = true;
    }
    if (flagInfo_.mSplitModeFlag) {
        return GetFmpTilingValMMode(kAL1FullloadFlag);
    } else {
        return GetFmpTilingValHWMode(kAL1FullloadFlag);
    }

    return FMP_OTHER;
}
 
uint64_t Conv3dBaseTilingV2::GetL1PingPongVal()
{
    uint64_t l1PingPong = static_cast<uint64_t>(tilingData_.conv3dApiTiling.get_pBufferFlag() & 0x18) >> L1_PB_OFFSET;
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
 
void Conv3dBaseTilingV2::ReSetTilingKeyPara()
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

ge::graphStatus Conv3dBaseTilingV2::SetTilingKey()
{
    tilingKeyPara_.fmpTiling = GetFmpTilingVal();
    tilingKeyPara_.weightTiling = GetWeightTilingVal();
    tilingKeyPara_.l1PingPong = GetL1PingPongVal();
    tilingKeyPara_.l0PingPong = GetL0PingPongVal();
    tilingKeyPara_.outputOrder = GetOutputOrderVal();
    tilingKeyPara_.iterOrder = static_cast<uint64_t>(tilingData_.conv3dApiTiling.get_iterateMNOrder());
    tilingKeyPara_.groupType = static_cast<uint64_t>(flagInfo_.convGroupType);
    ReSetTilingKeyPara();
    tilingKey_ = GET_TPL_TILING_KEY(tilingKeyPara_.fmpTiling,
                                    tilingKeyPara_.weightTiling,
                                    tilingKeyPara_.l1PingPong,
                                    tilingKeyPara_.l0PingPong,
                                    tilingKeyPara_.outputOrder,
                                    tilingKeyPara_.iterOrder,
                                    tilingKeyPara_.groupType);

    OP_LOGD(context_->GetNodeName(), "%s AscendC: tiling key: %lu.", paramInfo_.nodeType.c_str(), tilingKey_);
    return ge::GRAPH_SUCCESS;
}
}
}