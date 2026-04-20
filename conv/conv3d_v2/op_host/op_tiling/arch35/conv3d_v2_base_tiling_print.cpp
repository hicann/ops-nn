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
 * \file conv3d_v2_base_tiling_print.cpp
 * \brief
 */
#include "conv3d_v2_base_tiling.h"
 
namespace optiling {
namespace conv_ops_tiling {
void Conv3dBaseTilingV2::PrintOpTilingData()
{
    stringstream ss;
    ss << paramInfo_.nodeType.c_str() << " AscendC: ops tilingdata: batch: " << tilingData_.convRunInfo.batch
       << ", cin: " << tilingData_.convRunInfo.cin
       << ", din: " << tilingData_.convRunInfo.din
       << ", hi: " << tilingData_.convRunInfo.hin
       << ", wi: " << tilingData_.convRunInfo.win
       << ", cout: " << tilingData_.convRunInfo.cout
       << ", kd: " << tilingData_.convRunInfo.kd
       << ", kh: " << tilingData_.convRunInfo.kh
       << ", kw: " << tilingData_.convRunInfo.kw
       << ", dout: " << tilingData_.convRunInfo.dout
       << ", hout: " << tilingData_.convRunInfo.hout
       << ", wout: " << tilingData_.convRunInfo.wout
       << ", batchDim: " << tilingData_.convRunInfo.batchDim
       << ", doDim: " << tilingData_.convRunInfo.doDim
       << ", hoDim: " << tilingData_.convRunInfo.hoDim
       << ", nDim: " << tilingData_.convRunInfo.nDim
       << ", groupDim: " << tilingData_.convRunInfo.groupDim
       << ", strideH: " << tilingData_.convRunInfo.strideH
       << ", strideD: " << tilingData_.convRunInfo.strideD
       << ", dilationH: " << tilingData_.convRunInfo.dilationH
       << ", dilationD: " << tilingData_.convRunInfo.dilationD
       << ", padHead: " << tilingData_.convRunInfo.padHead
       << ", padTop: " << tilingData_.convRunInfo.padTop
       << ", hasBias: " << static_cast<uint32_t>(tilingData_.convRunInfo.hasBias)
       << ", groups: " << tilingData_.convRunInfo.groups
       << ", cinOpt: " << tilingData_.convRunInfo.cinOpt
       << ", coutOpt: " << tilingData_.convRunInfo.coutOpt
       << ", groupOpt: " << tilingData_.convRunInfo.groupOpt
       << ", enlarge: " << tilingData_.convRunInfo.enlarge;
    OP_LOGD(context_->GetNodeName(), "%s", ss.str().c_str());
}

void Conv3dBaseTilingV2::PrintLibApiTilingData()
{
    stringstream ss;
    ss << paramInfo_.nodeType.c_str() << " AscendC: api tilingdata: groups: "
       << tilingData_.convApiTiling.groups << ", orgDo: " << tilingData_.convApiTiling.orgDo
       << ", orgCo: " << tilingData_.convApiTiling.orgCo
       << ", orgHo: " << tilingData_.convApiTiling.orgHo
       << ", orgWo: " << tilingData_.convApiTiling.orgWo
       << ", orgDi: " << tilingData_.convApiTiling.orgDi
       << ", orgCi: " << tilingData_.convApiTiling.orgCi
       << ", orgHi: " << tilingData_.convApiTiling.orgHi
       << ", orgWi: " << tilingData_.convApiTiling.orgWi
       << ", kernelD: " << tilingData_.convApiTiling.kernelD
       << ", kernelH: " << tilingData_.convApiTiling.kernelH
       << ", kernelW: " << tilingData_.convApiTiling.kernelW
       << ", singleCoreBatch: " << tilingData_.convApiTiling.singleCoreBatch
       << ", singleCoreCo: " << tilingData_.convApiTiling.singleCoreCo
       << ", singleCoreCi: " << tilingData_.convApiTiling.singleCoreCi
       << ", singleCoreDo: " << tilingData_.convApiTiling.singleCoreDo
       << ", singleCoreHo: " << tilingData_.convApiTiling.singleCoreHo
       << ", singleCoreWo: " << tilingData_.convApiTiling.singleCoreWo
       << ", strideH: " << tilingData_.convApiTiling.strideH << ", strideW: "
       << tilingData_.convApiTiling.strideW << ", strideD: " << tilingData_.convApiTiling.strideD
       << ", dilationH: " << tilingData_.convApiTiling.dilationH << ", dilationW: "
       << tilingData_.convApiTiling.dilationW << ", dilationD: " << tilingData_.convApiTiling.dilationD
       << ", padHead: " << tilingData_.convApiTiling.padHead
       << ", padTail: " << tilingData_.convApiTiling.padTail
       << ", padTop: " << tilingData_.convApiTiling.padTop
       << ", padBottom: " << tilingData_.convApiTiling.padBottom
       << ", padLeft: " << tilingData_.convApiTiling.padLeft
       << ", padRight: " << tilingData_.convApiTiling.padRight
       << ", hoL0: " << tilingData_.convApiTiling.hoL0 << ", woL0: " << tilingData_.convApiTiling.woL0
       << ", kL0: " << tilingData_.convApiTiling.kL0 << ", nL0: " << tilingData_.convApiTiling.nL0
       << ", kAL1: " << tilingData_.convApiTiling.kAL1 << ", kBL1: " << tilingData_.convApiTiling.kBL1
       << ", nBL1: " << tilingData_.convApiTiling.nBL1 << ", hoL1: " << tilingData_.convApiTiling.hoL1
       << ", woL1: " << tilingData_.convApiTiling.woL1
       << ", mUB: " << tilingData_.convApiTiling.mUB
       << ", nUB: " << tilingData_.convApiTiling.nUB
       << ", pBufferFlag: " << tilingData_.convApiTiling.pBufferFlag
       << ", offsetx: " << static_cast<int32_t>(tilingData_.convApiTiling.offsetx)
       << ", iterateMNOrder: " << static_cast<uint32_t>(tilingData_.convApiTiling.iterateMNOrder)
       << ", biasFullLoadFlag: " << static_cast<uint32_t>(tilingData_.convApiTiling.biasFullLoadFlag)
       << ", fixpParamsFullLoadFlag: "
       << static_cast<uint32_t>(tilingData_.convApiTiling.fixpParamsFullLoadFlag)
       << ", enlarge: " << tilingData_.convApiTiling.enlarge
       << ", singleCoreGroups: " << tilingData_.convApiTiling.singleCoreGroups
       << ", singleCoreGroupOpt: " << tilingData_.convApiTiling.singleCoreGroupOpt
       << ", hf32Enable: " << static_cast<uint32_t>(tilingData_.convApiTiling.hf32Enable)
       << ", hf32TransMode: " << static_cast<uint32_t>(tilingData_.convApiTiling.hf32TransMode);
    OP_LOGD(context_->GetNodeName(), "%s", ss.str().c_str());

    stringstream ssPartTwo;
    ssPartTwo << paramInfo_.nodeType.c_str() << " AscendC: api tilingdata: groups: " << tilingData_.convApiTiling.groups
       << ", hasBias: " << static_cast<uint32_t>(tilingData_.convApiTiling.hasBias)
       << ", hasScale: " << static_cast<uint32_t>(tilingData_.convApiTiling.hasScale)
       << ", roundMode: " << static_cast<uint32_t>(tilingData_.convApiTiling.roundMode)
       << ", unionDataXt: " << static_cast<uint32_t>(tilingData_.convApiTiling.unionDataXt);
    OP_LOGD(context_->GetNodeName(), "%s", ssPartTwo.str().c_str());

    PrintLibApiScalarTilingData();
    PrintLibApiSpaceSize();
}

void Conv3dBaseTilingV2::PrintLibApiSpaceSize()
{
    uint32_t pbAL0 = (tilingData_.convApiTiling.pBufferFlag & PB_AL0_IDX) + 1;
    uint32_t pbBL0 = ((tilingData_.convApiTiling.pBufferFlag & PB_BL0_IDX) >> PB_BL0_IDX) + 1;
    uint32_t pbCL0 = ((tilingData_.convApiTiling.pBufferFlag & PB_CL0_IDX) >> PB_CL0_IDX) + 1;
    uint32_t pbBL1 = ((tilingData_.convApiTiling.pBufferFlag & PB_BL1_IDX) >> PB_BL1_IDX) + 1;
    uint64_t biasL1Size = flagInfo_.hasBias ? ConvAlignB(tilingData_.convApiTiling.nBL1 *
        static_cast<uint64_t>(descInfo_.biasDtype), C0_SIZE) : 0;
    uint64_t scaleL1Size = flagInfo_.quantFlag ?
        ConvAlignB(tilingData_.convApiTiling.nBL1 * static_cast<uint64_t>(descInfo_.scaleDtype),
        C0_SIZE) : 0;
    uint64_t apiL1Size = (tilingData_.convApiTiling.aL1SpaceSize + tilingData_.convApiTiling.kBL1 *
        tilingData_.convApiTiling.nBL1 * dtypeSizeTab.at(descInfo_.weightDtype) * pbBL1 +
        biasL1Size + scaleL1Size);
    uint64_t apiL0ASize = (tilingData_.convApiTiling.kL0 * tilingData_.convApiTiling.hoL0 *
        dtypeSizeTab.at(descInfo_.fMapDtype) * pbAL0);
    uint64_t apiL0BSize = (tilingData_.convApiTiling.kL0 * tilingData_.convApiTiling.nL0 *
        dtypeSizeTab.at(descInfo_.weightDtype) * pbBL0);
    uint64_t apiL0CSize = (tilingData_.convApiTiling.hoL0 * tilingData_.convApiTiling.nL0 *
        dtypeSizeTab.at(descInfo_.outDtype) * pbCL0);
    OP_LOGD(context_->GetNodeName(),
        "%s AscendC: api space size: apiL1Size: %lu, apiL0ASize: %lu, apiL0BSize: %lu, apiL0CSize: %lu",
        paramInfo_.nodeType.c_str(), apiL1Size, apiL0ASize, apiL0BSize, apiL0CSize);
}

void Conv3dBaseTilingV2::PrintLibApiScalarTilingData()
{
    OP_LOGD(context_->GetNodeName(),
        "Conv3D AscendC: api scalar tilingdata: orgHixWi: %lu, kernelHxkernelW: %lu, kernelHxkernelWxkernelD: %u,"\
        "aL1SpaceSize: %u, multiNBL1: %u, cinAInCore: %u, cinATailInCore: %u, cinBInCore: %u,"\
        "cinBTailInCore: %u, mStep: %u, kStep: %u, nStep: %u, fmapKStride: %u, weightKStride: %u,"\
        "cinOffsetBlockInGM: %u, coutOffsetBlock: %u, nL1DivBlockSize: %u",
        tilingData_.convApiTiling.orgHixWi, tilingData_.convApiTiling.kernelHxkernelW,
        tilingData_.convApiTiling.kernelHxkernelWxkernelD, tilingData_.convApiTiling.aL1SpaceSize,
        tilingData_.convApiTiling.multiNBL1, tilingData_.convApiTiling.cinAInCore,
        tilingData_.convApiTiling.cinATailInCore, tilingData_.convApiTiling.cinBInCore,
        tilingData_.convApiTiling.cinBTailInCore, tilingData_.convApiTiling.mStep,
        tilingData_.convApiTiling.kStep, tilingData_.convApiTiling.nStep,
        tilingData_.convApiTiling.fmapKStride, tilingData_.convApiTiling.weightKStride,
        tilingData_.convApiTiling.cinOffsetBlockInGM, tilingData_.convApiTiling.coutOffsetBlock,
        tilingData_.convApiTiling.nL1DivBlockSize);
}

void Conv3dBaseTilingV2::PrintTilingInfo()
{
    OP_LOGD(context_->GetNodeName(), "%s AscendC: tiling running mode: conv3d_load3d_flag.",
            paramInfo_.nodeType.c_str());
    OP_LOGD(context_->GetNodeName(), "%s AscendC: weight desc: format: %s, dtype: %s.",
            paramInfo_.nodeType.c_str(), formatToStrTab[descInfo_.weightFormat].c_str(),
            dtypeToStrTab[descInfo_.weightDtype].c_str());
    OP_LOGD(context_->GetNodeName(), "%s AscendC: featuremap desc: format: %s, dtype: %s.",
            paramInfo_.nodeType.c_str(), formatToStrTab[descInfo_.fMapFormat].c_str(),
            dtypeToStrTab[descInfo_.fMapDtype].c_str());
    if (flagInfo_.hasBias) {
        OP_LOGD(context_->GetNodeName(), "%s AscendC: bias desc: format %s, dtype: %s.",
                paramInfo_.nodeType.c_str(), formatToStrTab[descInfo_.biasFormat].c_str(),
                dtypeToStrTab[descInfo_.biasDtype].c_str());
    }
    OP_LOGD(context_->GetNodeName(), "%s AscendC: output desc: format: %s, dtype: %s.",
            paramInfo_.nodeType.c_str(), formatToStrTab[descInfo_.outFormat].c_str(),
            dtypeToStrTab[descInfo_.outDtype].c_str());
}
}
}