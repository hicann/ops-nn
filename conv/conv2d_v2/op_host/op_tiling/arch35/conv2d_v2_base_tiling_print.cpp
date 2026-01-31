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
    ss << "hin: " << tilingData_.conv2dRunInfo.get_hin() << ", win: " << tilingData_.conv2dRunInfo.get_win()
       << ", hout: " << tilingData_.conv2dRunInfo.get_hout() << ", wout: " << tilingData_.conv2dRunInfo.get_wout()
       << ", batch: " << tilingData_.conv2dRunInfo.get_batch() << ", cin: " << tilingData_.conv2dRunInfo.get_cin()
       << ", cout: " << tilingData_.conv2dRunInfo.get_cout() << ", kh: " << tilingData_.conv2dRunInfo.get_kh()
       << ", kw: " << tilingData_.conv2dRunInfo.get_kw() << ", batchDim: " << tilingData_.conv2dRunInfo.get_batchDim()
       << ", hoDim: " << tilingData_.conv2dRunInfo.get_hoDim() << ", woDim: " << tilingData_.conv2dRunInfo.get_woDim()
       << ", nDim: " << tilingData_.conv2dRunInfo.get_nDim()
       << ", strideH: " << tilingData_.conv2dRunInfo.get_strideH()
       << ", strideW: " << tilingData_.conv2dRunInfo.get_strideW()
       << ", dilationH: " << tilingData_.conv2dRunInfo.get_dilationH()
       << ", dilationW: " << tilingData_.conv2dRunInfo.get_dilationW()
       << ", padTop: " << tilingData_.conv2dRunInfo.get_padTop()
       << ", padLeft: " << tilingData_.conv2dRunInfo.get_padLeft()
       << ", groups: " << tilingData_.conv2dRunInfo.get_groups()
       << ", cinOpt: " << tilingData_.conv2dRunInfo.get_cinOpt()
       << ", coutOpt: " << tilingData_.conv2dRunInfo.get_coutOpt()
       << ", groupOpt: " << tilingData_.conv2dRunInfo.get_groupOpt()
       << ", enlarge: " << tilingData_.conv2dRunInfo.get_enlarge()
       << ", groupDim: " << tilingData_.conv2dRunInfo.get_groupDim()
       << ", hasBias: " << static_cast<uint32_t>(tilingData_.conv2dRunInfo.get_hasBias());
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
    ss << "singleCoreHo: " << tilingData_.conv2dApiTiling.get_singleCoreHo()
    << ", singleCoreWo: " << tilingData_.conv2dApiTiling.get_singleCoreWo()
    << ", singleCoreBatch: " << tilingData_.conv2dApiTiling.get_singleCoreBatch()
    << ", orgHi: " << tilingData_.conv2dApiTiling.get_orgHi()
    << ", orgWi: " << tilingData_.conv2dApiTiling.get_orgWi()
    << ", orgHo: " << tilingData_.conv2dApiTiling.get_orgHo()
    << ", orgWo: " << tilingData_.conv2dApiTiling.get_orgWo()
    << ", groups: " << tilingData_.conv2dApiTiling.get_groups()
    << ", orgCi: " << tilingData_.conv2dApiTiling.get_orgCi()
    << ", orgCo: " << tilingData_.conv2dApiTiling.get_orgCo()
    << ", kernelH: " << tilingData_.conv2dApiTiling.get_kernelH()
    << ", kernelW: " << tilingData_.conv2dApiTiling.get_kernelW()
    << ", singleCoreCi: " << tilingData_.conv2dApiTiling.get_singleCoreCi()
    << ", singleCoreCo: " << tilingData_.conv2dApiTiling.get_singleCoreCo()
    << ", hoL1: " << tilingData_.conv2dApiTiling.get_hoL1()
    << ", woL1: " << tilingData_.conv2dApiTiling.get_woL1()
    << ", kAL1: " << tilingData_.conv2dApiTiling.get_kAL1()
    << ", kBL1: " << tilingData_.conv2dApiTiling.get_kBL1()
    << ", nBL1: " << tilingData_.conv2dApiTiling.get_nBL1()
    << ", hoL0: " << tilingData_.conv2dApiTiling.get_hoL0()
    << ", woL0: " << tilingData_.conv2dApiTiling.get_woL0()
    << ", kL0: " << tilingData_.conv2dApiTiling.get_kL0()
    << ", nL0: " << tilingData_.conv2dApiTiling.get_nL0()
    << ", pBufferFlag: " << tilingData_.conv2dApiTiling.get_pBufferFlag()
    << ", multiNBL1: " << tilingData_.conv2dApiTiling.get_multiNBL1()
    << ", strideH: " << tilingData_.conv2dApiTiling.get_strideH()
    << ", strideW: " << tilingData_.conv2dApiTiling.get_strideW()
    << ", dilationH: " << tilingData_.conv2dApiTiling.get_dilationH()
    << ", dilationW: " << tilingData_.conv2dApiTiling.get_dilationW()
    << ", padTop: " << tilingData_.conv2dApiTiling.get_padTop()
    << ", padBottom: " << tilingData_.conv2dApiTiling.get_padBottom()
    << ", padLeft: " << tilingData_.conv2dApiTiling.get_padLeft()
    << ", padRight: " << tilingData_.conv2dApiTiling.get_padRight()
    << ", aL1SpaceSize: " << tilingData_.conv2dApiTiling.get_aL1SpaceSize()
    << ", singleCoreGroups: " << tilingData_.conv2dApiTiling.get_singleCoreGroups()
    << ", singleCoreGroupOpt: " << tilingData_.conv2dApiTiling.get_singleCoreGroupOpt()
    << ", enlarge: " << tilingData_.conv2dApiTiling.get_enlarge()
    << ", bUbNStep: " << tilingData_.conv2dApiTiling.get_bUbNStep()
    << ", iterateMNOrder: " << static_cast<uint32_t>(tilingData_.conv2dApiTiling.get_iterateMNOrder())
    << ", biasFullLoadFlag: " << static_cast<uint32_t>(tilingData_.conv2dApiTiling.get_biasFullLoadFlag())
    << ", fixpParamsFullLoadFlag: " << static_cast<uint32_t>(tilingData_.conv2dApiTiling.get_fixpParamsFullLoadFlag())
    << ", hf32Enable: " << static_cast<uint32_t>(tilingData_.conv2dApiTiling.get_hf32Enable())
    << ", hf32TransMode: " << static_cast<uint32_t>(tilingData_.conv2dApiTiling.get_hf32TransMode())
    << ", hasBias: " << static_cast<uint32_t>(tilingData_.conv2dApiTiling.get_hasBias())
    << ", hasScale: " << static_cast<uint32_t>(tilingData_.conv2dApiTiling.get_hasScale())
    << ", offsetx: " << static_cast<uint32_t>(tilingData_.conv2dApiTiling.get_offsetx())
    << ", roundMode: " << static_cast<uint32_t>(tilingData_.conv2dApiTiling.get_roundMode())
    << ", innerBatch: " << static_cast<uint32_t>(tilingData_.conv2dApiTiling.get_innerBatch());
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
    ssPartTwo << "bUbKStep: " << tilingData_.conv2dApiTiling.get_bUbKStep()
    << ", orgHixWi: " << tilingData_.conv2dApiTiling.get_orgHixWi()
    << ", kernelHxkernelW: " << tilingData_.conv2dApiTiling.get_kernelHxkernelW()
    << ", kernelHxkernelWxkernelD: " << tilingData_.conv2dApiTiling.get_kernelHxkernelWxkernelD()
    << ", cinAInCore: " << tilingData_.conv2dApiTiling.get_cinAInCore()
    << ", cinATailInCore: " << tilingData_.conv2dApiTiling.get_cinATailInCore()
    << ", cinBInCore: " << tilingData_.conv2dApiTiling.get_cinBInCore()
    << ", cinBTailInCore: " << tilingData_.conv2dApiTiling.get_cinBTailInCore()
    << ", mStep: " << tilingData_.conv2dApiTiling.get_mStep()
    << ", kStep: " << tilingData_.conv2dApiTiling.get_kStep()
    << ", nStep: " << tilingData_.conv2dApiTiling.get_nStep()
    << ", fmapKStride: " << tilingData_.conv2dApiTiling.get_fmapKStride()
    << ", weightKStride: " << tilingData_.conv2dApiTiling.get_weightKStride()
    << ", cinOffsetBlockInGM: " << tilingData_.conv2dApiTiling.get_cinOffsetBlockInGM()
    << ", coutOffsetBlock: " << tilingData_.conv2dApiTiling.get_coutOffsetBlock()
    << ", nL1DivBlockSize: " << tilingData_.conv2dApiTiling.get_nL1DivBlockSize()
    << ", dualOutput: " << static_cast<uint32_t>(tilingData_.conv2dApiTiling.get_dualOutput())
    << ", quantMode0: " << static_cast<uint32_t>(tilingData_.conv2dApiTiling.get_quantMode0())
    << ", reluMode0: " << static_cast<uint32_t>(tilingData_.conv2dApiTiling.get_reluMode0())
    << ", clipMode0: " << static_cast<uint32_t>(tilingData_.conv2dApiTiling.get_clipMode0())
    << ", quantMode1: " << static_cast<uint32_t>(tilingData_.conv2dApiTiling.get_quantMode1())
    << ", reluMode1: " << static_cast<uint32_t>(tilingData_.conv2dApiTiling.get_reluMode1())
    << ", clipMode1: " << static_cast<uint32_t>(tilingData_.conv2dApiTiling.get_clipMode1())
    << ", khL1: " << static_cast<uint32_t>(tilingData_.conv2dApiTiling.get_khL1())
    << ", kwL1: " << static_cast<uint32_t>(tilingData_.conv2dApiTiling.get_kwL1())
    << ", khUb: " << static_cast<uint32_t>(tilingData_.conv2dApiTiling.get_khUb())
    << ", kwUb: " << static_cast<uint32_t>(tilingData_.conv2dApiTiling.get_kwUb());
    OP_LOGD(context_->GetNodeName(), "%s AscendC: api tilingdata: %s", paramInfo_.nodeType.c_str(), ss.str().c_str());
    OP_LOGD(context_->GetNodeName(), "%s", ssPartTwo.str().c_str());
}

}
}