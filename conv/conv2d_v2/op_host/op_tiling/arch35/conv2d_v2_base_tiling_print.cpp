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
 * \file conv2d_v2_base_tiling_print.cpp
 * \brief
 */
#include "conv2d_v2_base_tiling.h"

namespace optiling {
namespace conv_ops_tiling {
void Conv2dBaseTiling::PrintOpTilingData()
{
    /*
      maily used to check the Validity of ops tilingdata
      do not modify the param name if not necessarily
    */
    std::stringstream ss;
    ss << "hin: " << tilingData_.convRunInfo.get_hin() << ", win: " << tilingData_.convRunInfo.get_win()
       << ", hout: " << tilingData_.convRunInfo.get_hout() << ", wout: " << tilingData_.convRunInfo.get_wout()
       << ", batch: " << tilingData_.convRunInfo.get_batch() << ", cin: " << tilingData_.convRunInfo.get_cin()
       << ", cout: " << tilingData_.convRunInfo.get_cout() << ", kh: " << tilingData_.convRunInfo.get_kh()
       << ", kw: " << tilingData_.convRunInfo.get_kw() << ", batchDim: " << tilingData_.convRunInfo.get_batchDim()
       << ", hoDim: " << tilingData_.convRunInfo.get_hoDim() << ", woDim: " << tilingData_.convRunInfo.get_woDim()
       << ", nDim: " << tilingData_.convRunInfo.get_nDim()
       << ", strideH: " << tilingData_.convRunInfo.get_strideH()
       << ", strideW: " << tilingData_.convRunInfo.get_strideW()
       << ", dilationH: " << tilingData_.convRunInfo.get_dilationH()
       << ", dilationW: " << tilingData_.convRunInfo.get_dilationW()
       << ", padTop: " << tilingData_.convRunInfo.get_padTop()
       << ", padLeft: " << tilingData_.convRunInfo.get_padLeft()
       << ", groups: " << tilingData_.convRunInfo.get_groups()
       << ", cinOpt: " << tilingData_.convRunInfo.get_cinOpt()
       << ", coutOpt: " << tilingData_.convRunInfo.get_coutOpt()
       << ", groupOpt: " << tilingData_.convRunInfo.get_groupOpt()
       << ", enlarge: " << tilingData_.convRunInfo.get_enlarge()
       << ", groupDim: " << tilingData_.convRunInfo.get_groupDim()
       << ", hasBias: " << static_cast<uint32_t>(tilingData_.convRunInfo.get_hasBias());
    OP_LOGD(context_->GetNodeName(), "%s AscendC: ops tilingdata: %s", paramInfo_.nodeType.c_str(), ss.str().c_str());
}

void Conv2dBaseTiling::PrintTilingInfo() const
{
    OP_LOGD(context_->GetNodeName(), "%s AscendC: tiling running mode: %s.",
            paramInfo_.nodeType.c_str(), FeatureFlagEnumToString(featureFlagInfo_));
    OP_LOGD(context_->GetNodeName(), "%s AscendC: weight desc: format: %s, dtype: %s.",
            paramInfo_.nodeType.c_str(), formatToStrTab.at(descInfo_.weightFormat).c_str(),
            dtypeToStrTab.at(descInfo_.weightDtype).c_str());
    OP_LOGD(context_->GetNodeName(), "%s AscendC: featuremap desc: format: %s, dtype: %s.",
            paramInfo_.nodeType.c_str(), formatToStrTab.at(descInfo_.fMapFormat).c_str(),
            dtypeToStrTab.at(descInfo_.fMapDtype).c_str());
    if (flagInfo_.hasBias) {
        OP_LOGD(context_->GetNodeName(), "%s AscendC: bias desc: format %s, dtype: %s.",
                paramInfo_.nodeType.c_str(), formatToStrTab.at(descInfo_.biasFormat).c_str(),
                dtypeToStrTab.at(descInfo_.biasDtype).c_str());
    }
    OP_LOGD(context_->GetNodeName(), "%s AscendC: output desc: format: %s, dtype: %s.",
            paramInfo_.nodeType.c_str(), formatToStrTab.at(descInfo_.outFormat).c_str(),
            dtypeToStrTab.at(descInfo_.outDtype).c_str());
    if (flagInfo_.extendConvFlag && (attrInfo_.dualOutput != 0)) {
        OP_LOGD(context_->GetNodeName(), "%s AscendC: output1 desc: format: %s, dtype: %s.",
                paramInfo_.nodeType.c_str(), formatToStrTab.at(descInfo_.out1Format).c_str(),
                dtypeToStrTab.at(descInfo_.out1Dtype).c_str());
    }
}

void Conv2dBaseTiling::PrintLibApiTilingDataPartOne(std::stringstream &ss)
{
    ss << "singleCoreHo: " << tilingData_.convApiTiling.get_singleCoreHo()
    << ", singleCoreWo: " << tilingData_.convApiTiling.get_singleCoreWo()
    << ", singleCoreBatch: " << tilingData_.convApiTiling.get_singleCoreBatch()
    << ", orgHi: " << tilingData_.convApiTiling.get_orgHi()
    << ", orgWi: " << tilingData_.convApiTiling.get_orgWi()
    << ", orgHo: " << tilingData_.convApiTiling.get_orgHo()
    << ", orgWo: " << tilingData_.convApiTiling.get_orgWo()
    << ", groups: " << tilingData_.convApiTiling.get_groups()
    << ", orgCi: " << tilingData_.convApiTiling.get_orgCi()
    << ", orgCo: " << tilingData_.convApiTiling.get_orgCo()
    << ", kernelH: " << tilingData_.convApiTiling.get_kernelH()
    << ", kernelW: " << tilingData_.convApiTiling.get_kernelW()
    << ", singleCoreCi: " << tilingData_.convApiTiling.get_singleCoreCi()
    << ", singleCoreCo: " << tilingData_.convApiTiling.get_singleCoreCo()
    << ", hoL1: " << tilingData_.convApiTiling.get_hoL1()
    << ", woL1: " << tilingData_.convApiTiling.get_woL1()
    << ", kAL1: " << tilingData_.convApiTiling.get_kAL1()
    << ", kBL1: " << tilingData_.convApiTiling.get_kBL1()
    << ", nBL1: " << tilingData_.convApiTiling.get_nBL1()
    << ", hoL0: " << tilingData_.convApiTiling.get_hoL0()
    << ", woL0: " << tilingData_.convApiTiling.get_woL0()
    << ", kL0: " << tilingData_.convApiTiling.get_kL0()
    << ", nL0: " << tilingData_.convApiTiling.get_nL0()
    << ", pBufferFlag: " << tilingData_.convApiTiling.get_pBufferFlag()
    << ", multiNBL1: " << tilingData_.convApiTiling.get_multiNBL1()
    << ", strideH: " << tilingData_.convApiTiling.get_strideH()
    << ", strideW: " << tilingData_.convApiTiling.get_strideW()
    << ", dilationH: " << tilingData_.convApiTiling.get_dilationH()
    << ", dilationW: " << tilingData_.convApiTiling.get_dilationW()
    << ", padTop: " << tilingData_.convApiTiling.get_padTop()
    << ", padBottom: " << tilingData_.convApiTiling.get_padBottom()
    << ", padLeft: " << tilingData_.convApiTiling.get_padLeft()
    << ", padRight: " << tilingData_.convApiTiling.get_padRight()
    << ", aL1SpaceSize: " << tilingData_.convApiTiling.get_aL1SpaceSize()
    << ", singleCoreGroups: " << tilingData_.convApiTiling.get_singleCoreGroups()
    << ", singleCoreGroupOpt: " << tilingData_.convApiTiling.get_singleCoreGroupOpt()
    << ", enlarge: " << tilingData_.convApiTiling.get_enlarge()
    << ", bUbNStep: " << tilingData_.convApiTiling.get_bUbNStep()
    << ", iterateMNOrder: " << static_cast<uint32_t>(tilingData_.convApiTiling.get_iterateMNOrder())
    << ", biasFullLoadFlag: " << static_cast<uint32_t>(tilingData_.convApiTiling.get_biasFullLoadFlag())
    << ", fixpParamsFullLoadFlag: " << static_cast<uint32_t>(tilingData_.convApiTiling.get_fixpParamsFullLoadFlag())
    << ", hf32Enable: " << static_cast<uint32_t>(tilingData_.convApiTiling.get_hf32Enable())
    << ", hf32TransMode: " << static_cast<uint32_t>(tilingData_.convApiTiling.get_hf32TransMode())
    << ", hasBias: " << static_cast<uint32_t>(tilingData_.convApiTiling.get_hasBias())
    << ", hasScale: " << static_cast<uint32_t>(tilingData_.convApiTiling.get_hasScale())
    << ", offsetx: " << static_cast<uint32_t>(tilingData_.convApiTiling.get_offsetx())
    << ", roundMode: " << static_cast<uint32_t>(tilingData_.convApiTiling.get_roundMode())
    << ", innerBatch: " << static_cast<uint32_t>(tilingData_.convApiTiling.get_innerBatch());
}


void Conv2dBaseTiling::PrintLibApiTilingData()
{
    /*
      maily used to check the Validity of api tilingdata
      do not modify the param name if not necessarily
    */
    std::stringstream ss;
    PrintLibApiTilingDataPartOne(ss);
    std::stringstream ssPartTwo;
    ssPartTwo << "bUbKStep: " << tilingData_.convApiTiling.get_bUbKStep()
    << ", orgHixWi: " << tilingData_.convApiTiling.get_orgHixWi()
    << ", kernelHxkernelW: " << tilingData_.convApiTiling.get_kernelHxkernelW()
    << ", kernelHxkernelWxkernelD: " << tilingData_.convApiTiling.get_kernelHxkernelWxkernelD()
    << ", cinAInCore: " << tilingData_.convApiTiling.get_cinAInCore()
    << ", cinATailInCore: " << tilingData_.convApiTiling.get_cinATailInCore()
    << ", cinBInCore: " << tilingData_.convApiTiling.get_cinBInCore()
    << ", cinBTailInCore: " << tilingData_.convApiTiling.get_cinBTailInCore()
    << ", mStep: " << tilingData_.convApiTiling.get_mStep()
    << ", kStep: " << tilingData_.convApiTiling.get_kStep()
    << ", nStep: " << tilingData_.convApiTiling.get_nStep()
    << ", fmapKStride: " << tilingData_.convApiTiling.get_fmapKStride()
    << ", weightKStride: " << tilingData_.convApiTiling.get_weightKStride()
    << ", cinOffsetBlockInGM: " << tilingData_.convApiTiling.get_cinOffsetBlockInGM()
    << ", coutOffsetBlock: " << tilingData_.convApiTiling.get_coutOffsetBlock()
    << ", nL1DivBlockSize: " << tilingData_.convApiTiling.get_nL1DivBlockSize()
    << ", dualOutput: " << static_cast<uint32_t>(tilingData_.convApiTiling.get_dualOutput())
    << ", quantMode0: " << static_cast<uint32_t>(tilingData_.convApiTiling.get_quantMode0())
    << ", reluMode0: " << static_cast<uint32_t>(tilingData_.convApiTiling.get_reluMode0())
    << ", clipMode0: " << static_cast<uint32_t>(tilingData_.convApiTiling.get_clipMode0())
    << ", quantMode1: " << static_cast<uint32_t>(tilingData_.convApiTiling.get_quantMode1())
    << ", reluMode1: " << static_cast<uint32_t>(tilingData_.convApiTiling.get_reluMode1())
    << ", clipMode1: " << static_cast<uint32_t>(tilingData_.convApiTiling.get_clipMode1())
    << ", khL1: " << static_cast<uint32_t>(tilingData_.convApiTiling.get_khL1())
    << ", kwL1: " << static_cast<uint32_t>(tilingData_.convApiTiling.get_kwL1())
    << ", khUb: " << static_cast<uint32_t>(tilingData_.convApiTiling.get_khUb())
    << ", kwUb: " << static_cast<uint32_t>(tilingData_.convApiTiling.get_kwUb())
    << ", unionDataXt: " << static_cast<uint32_t>(tilingData_.convApiTiling.get_unionDataXt());
    OP_LOGD(context_->GetNodeName(), "%s AscendC: api tilingdata: %s", paramInfo_.nodeType.c_str(), ss.str().c_str());
    OP_LOGD(context_->GetNodeName(), "%s", ssPartTwo.str().c_str());
}

}
}