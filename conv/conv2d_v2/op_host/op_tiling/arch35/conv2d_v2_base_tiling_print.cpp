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
    ss << "hin: " << tilingData_.get_hin() << ", win: " << tilingData_.get_win() << ", hout: " << tilingData_.get_hout()
       << ", wout: " << tilingData_.get_wout() << ", batch: " << tilingData_.get_batch()
       << ", cin: " << tilingData_.get_cin() << ", cout: " << tilingData_.get_cout() << ", kh: " << tilingData_.get_kh()
       << ", kw: " << tilingData_.get_kw() << ", batchDim: " << tilingData_.get_batchDim()
       << ", hoDim: " << tilingData_.get_hoDim() << ", woDim: " << tilingData_.get_woDim()
       << ", nDim: " << tilingData_.get_nDim() << ", strideH: " << tilingData_.get_strideH()
       << ", strideW: " << tilingData_.get_strideW() << ", dilationH: " << tilingData_.get_dilationH()
       << ", dilationW: " << tilingData_.get_dilationW() << ", padTop: " << tilingData_.get_padTop()
       << ", padLeft: " << tilingData_.get_padLeft() << ", groups: " << tilingData_.get_groups()
       << ", cinOpt: " << tilingData_.get_cinOpt() << ", coutOpt: " << tilingData_.get_coutOpt()
       << ", groupOpt: " << tilingData_.get_groupOpt() << ", enlarge: " << tilingData_.get_enlarge()
       << ", groupDim: " << tilingData_.get_groupDim()
       << ", hasBias: " << static_cast<uint32_t>(tilingData_.get_hasBias());
    OP_LOGD(context_->GetNodeName(), "%s AscendC: ops tilingdata: %s", paramInfo_.nodeType.c_str(), ss.str().c_str());
}

void Conv2dBaseTiling::PrintTilingInfo() const
{
    OP_LOGD(context_->GetNodeName(), "%s AscendC: tiling running mode: %s.", paramInfo_.nodeType.c_str(),
            FeatureFlagEnumToString(featureFlagInfo_));
    OP_LOGD(context_->GetNodeName(), "%s AscendC: weight desc: format: %s, dtype: %s.", paramInfo_.nodeType.c_str(),
            formatToStrTab.at(descInfo_.weightFormat).c_str(), dtypeToStrTab.at(descInfo_.weightDtype).c_str());
    OP_LOGD(context_->GetNodeName(), "%s AscendC: featuremap desc: format: %s, dtype: %s.", paramInfo_.nodeType.c_str(),
            formatToStrTab.at(descInfo_.fMapFormat).c_str(), dtypeToStrTab.at(descInfo_.fMapDtype).c_str());
    if (flagInfo_.hasBias) {
        OP_LOGD(context_->GetNodeName(), "%s AscendC: bias desc: format %s, dtype: %s.", paramInfo_.nodeType.c_str(),
                formatToStrTab.at(descInfo_.biasFormat).c_str(), dtypeToStrTab.at(descInfo_.biasDtype).c_str());
    }
    OP_LOGD(context_->GetNodeName(), "%s AscendC: output desc: format: %s, dtype: %s.", paramInfo_.nodeType.c_str(),
            formatToStrTab.at(descInfo_.outFormat).c_str(), dtypeToStrTab.at(descInfo_.outDtype).c_str());
    if (flagInfo_.extendConvFlag && (attrInfo_.dualOutput != 0)) {
        OP_LOGD(context_->GetNodeName(), "%s AscendC: output1 desc: format: %s, dtype: %s.",
                paramInfo_.nodeType.c_str(), formatToStrTab.at(descInfo_.out1Format).c_str(),
                dtypeToStrTab.at(descInfo_.out1Dtype).c_str());
    }
}

void Conv2dBaseTiling::PrintLibApiTilingDataPartOne(std::stringstream& ss)
{
    ss << "singleCoreHo: " << tilingData_.get_singleCoreHo() << ", singleCoreWo: " << tilingData_.get_singleCoreWo()
       << ", singleCoreBatch: " << tilingData_.get_singleCoreBatch() << ", orgHi: " << tilingData_.get_orgHi()
       << ", orgWi: " << tilingData_.get_orgWi() << ", orgHo: " << tilingData_.get_orgHo()
       << ", orgWo: " << tilingData_.get_orgWo() << ", groups: " << tilingData_.get_groups()
       << ", orgCi: " << tilingData_.get_orgCi() << ", orgCo: " << tilingData_.get_orgCo()
       << ", kernelH: " << tilingData_.get_kernelH() << ", kernelW: " << tilingData_.get_kernelW()
       << ", singleCoreCi: " << tilingData_.get_singleCoreCi() << ", singleCoreCo: " << tilingData_.get_singleCoreCo()
       << ", hoL1: " << tilingData_.get_hoL1() << ", woL1: " << tilingData_.get_woL1()
       << ", kAL1: " << tilingData_.get_kAL1() << ", kBL1: " << tilingData_.get_kBL1()
       << ", nBL1: " << tilingData_.get_nBL1() << ", hoL0: " << tilingData_.get_hoL0()
       << ", woL0: " << tilingData_.get_woL0() << ", kL0: " << tilingData_.get_kL0()
       << ", nL0: " << tilingData_.get_nL0() << ", pBufferFlag: " << tilingData_.get_pBufferFlag()
       << ", multiNBL1: " << tilingData_.get_multiNBL1() << ", strideH: " << tilingData_.get_strideH()
       << ", strideW: " << tilingData_.get_strideW() << ", dilationH: " << tilingData_.get_dilationH()
       << ", dilationW: " << tilingData_.get_dilationW() << ", padTop: " << tilingData_.get_padTop()
       << ", padBottom: " << tilingData_.get_padBottom() << ", padLeft: " << tilingData_.get_padLeft()
       << ", padRight: " << tilingData_.get_padRight() << ", aL1SpaceSize: " << tilingData_.get_aL1SpaceSize()
       << ", singleCoreGroups: " << tilingData_.get_singleCoreGroups()
       << ", singleCoreGroupOpt: " << tilingData_.get_singleCoreGroupOpt() << ", enlarge: " << tilingData_.get_enlarge()
       << ", bUbNStep: " << tilingData_.get_bUbNStep()
       << ", iterateMNOrder: " << static_cast<uint32_t>(tilingData_.get_iterateMNOrder())
       << ", biasFullLoadFlag: " << static_cast<uint32_t>(tilingData_.get_biasFullLoadFlag())
       << ", fixpParamsFullLoadFlag: " << static_cast<uint32_t>(tilingData_.get_fixpParamsFullLoadFlag())
       << ", hf32Enable: " << static_cast<uint32_t>(tilingData_.get_hf32Enable())
       << ", hf32TransMode: " << static_cast<uint32_t>(tilingData_.get_hf32TransMode())
       << ", hasBias: " << static_cast<uint32_t>(tilingData_.get_hasBias())
       << ", hasScale: " << static_cast<uint32_t>(tilingData_.get_hasScale())
       << ", offsetx: " << static_cast<uint32_t>(tilingData_.get_offsetx())
       << ", roundMode: " << static_cast<uint32_t>(tilingData_.get_roundMode())
       << ", innerBatch: " << static_cast<uint32_t>(tilingData_.get_innerBatch());
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
    ssPartTwo << "bUbKStep: " << tilingData_.get_bUbKStep() << ", orgHixWi: " << tilingData_.get_orgHixWi()
              << ", kernelHxkernelW: " << tilingData_.get_kernelHxkernelW()
              << ", kernelHxkernelWxkernelD: " << tilingData_.get_kernelHxkernelWxkernelD()
              << ", cinAInCore: " << tilingData_.get_cinAInCore()
              << ", cinATailInCore: " << tilingData_.get_cinATailInCore()
              << ", cinBInCore: " << tilingData_.get_cinBInCore()
              << ", cinBTailInCore: " << tilingData_.get_cinBTailInCore() << ", mStep: " << tilingData_.get_mStep()
              << ", kStep: " << tilingData_.get_kStep() << ", nStep: " << tilingData_.get_nStep()
              << ", fmapKStride: " << tilingData_.get_fmapKStride()
              << ", weightKStride: " << tilingData_.get_weightKStride()
              << ", cinOffsetBlockInGM: " << tilingData_.get_cinOffsetBlockInGM()
              << ", coutOffsetBlock: " << tilingData_.get_coutOffsetBlock()
              << ", nL1DivBlockSize: " << tilingData_.get_nL1DivBlockSize()
              << ", dualOutput: " << static_cast<uint32_t>(tilingData_.get_dualOutput())
              << ", quantMode0: " << static_cast<uint32_t>(tilingData_.get_quantMode0())
              << ", reluMode0: " << static_cast<uint32_t>(tilingData_.get_reluMode0())
              << ", clipMode0: " << static_cast<uint32_t>(tilingData_.get_clipMode0())
              << ", quantMode1: " << static_cast<uint32_t>(tilingData_.get_quantMode1())
              << ", reluMode1: " << static_cast<uint32_t>(tilingData_.get_reluMode1())
              << ", clipMode1: " << static_cast<uint32_t>(tilingData_.get_clipMode1())
              << ", khL1: " << static_cast<uint32_t>(tilingData_.get_khL1())
              << ", kwL1: " << static_cast<uint32_t>(tilingData_.get_kwL1())
              << ", khUb: " << static_cast<uint32_t>(tilingData_.get_khUb())
              << ", kwUb: " << static_cast<uint32_t>(tilingData_.get_kwUb())
              << ", unionDataXt: " << static_cast<uint32_t>(tilingData_.get_unionDataXt());
    OP_LOGD(context_->GetNodeName(), "%s AscendC: api tilingdata: %s", paramInfo_.nodeType.c_str(), ss.str().c_str());
    OP_LOGD(context_->GetNodeName(), "%s", ssPartTwo.str().c_str());
}

} // namespace conv_ops_tiling
} // namespace optiling