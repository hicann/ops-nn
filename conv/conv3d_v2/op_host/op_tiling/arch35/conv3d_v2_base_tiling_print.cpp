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
    ss << paramInfo_.nodeType.c_str() << " AscendC: ops tilingdata: batch: " << tilingData_.conv3dRunInfo.batch
       << ", cin: " << tilingData_.conv3dRunInfo.cin
       << ", din: " << tilingData_.conv3dRunInfo.din
       << ", hi: " << tilingData_.conv3dRunInfo.hin
       << ", wi: " << tilingData_.conv3dRunInfo.win
       << ", cout: " << tilingData_.conv3dRunInfo.cout
       << ", kd: " << tilingData_.conv3dRunInfo.kd
       << ", kh: " << tilingData_.conv3dRunInfo.kh
       << ", kw: " << tilingData_.conv3dRunInfo.kw
       << ", dout: " << tilingData_.conv3dRunInfo.dout
       << ", hout: " << tilingData_.conv3dRunInfo.hout
       << ", wout: " << tilingData_.conv3dRunInfo.wout
       << ", batchDim: " << tilingData_.conv3dRunInfo.batchDim
       << ", doDim: " << tilingData_.conv3dRunInfo.doDim
       << ", hoDim: " << tilingData_.conv3dRunInfo.hoDim
       << ", nDim: " << tilingData_.conv3dRunInfo.nDim
       << ", groupDim: " << tilingData_.conv3dRunInfo.groupDim
       << ", strideH: " << tilingData_.conv3dRunInfo.strideH
       << ", strideD: " << tilingData_.conv3dRunInfo.strideD
       << ", dilationH: " << tilingData_.conv3dRunInfo.dilationH
       << ", dilationD: " << tilingData_.conv3dRunInfo.dilationD
       << ", padHead: " << tilingData_.conv3dRunInfo.padHead
       << ", padTop: " << tilingData_.conv3dRunInfo.padTop
       << ", hasBias: " << static_cast<uint32_t>(tilingData_.conv3dRunInfo.hasBias)
       << ", groups: " << tilingData_.conv3dRunInfo.groups
       << ", cinOpt: " << tilingData_.conv3dRunInfo.cinOpt
       << ", coutOpt: " << tilingData_.conv3dRunInfo.coutOpt
       << ", groupOpt: " << tilingData_.conv3dRunInfo.groupOpt
       << ", enlarge: " << tilingData_.conv3dRunInfo.enlarge;
    OP_LOGD(context_->GetNodeName(), "%s", ss.str().c_str());
}

void Conv3dBaseTilingV2::PrintLibApiTilingData()
{
    stringstream ss;
    ss << paramInfo_.nodeType.c_str() << " AscendC: api tilingdata: groups: "
       << tilingData_.conv3dApiTiling.groups << ", orgDo: " << tilingData_.conv3dApiTiling.orgDo
       << ", orgCo: " << tilingData_.conv3dApiTiling.orgCo
       << ", orgHo: " << tilingData_.conv3dApiTiling.orgHo
       << ", orgWo: " << tilingData_.conv3dApiTiling.orgWo
       << ", orgDi: " << tilingData_.conv3dApiTiling.orgDi
       << ", orgCi: " << tilingData_.conv3dApiTiling.orgCi
       << ", orgHi: " << tilingData_.conv3dApiTiling.orgHi
       << ", orgWi: " << tilingData_.conv3dApiTiling.orgWi
       << ", kernelD: " << tilingData_.conv3dApiTiling.kernelD
       << ", kernelH: " << tilingData_.conv3dApiTiling.kernelH
       << ", kernelW: " << tilingData_.conv3dApiTiling.kernelW
       << ", singleCoreBatch: " << tilingData_.conv3dApiTiling.singleCoreBatch
       << ", singleCoreCo: " << tilingData_.conv3dApiTiling.singleCoreCo
       << ", singleCoreCi: " << tilingData_.conv3dApiTiling.singleCoreCi
       << ", singleCoreDo: " << tilingData_.conv3dApiTiling.singleCoreDo
       << ", singleCoreHo: " << tilingData_.conv3dApiTiling.singleCoreHo
       << ", singleCoreWo: " << tilingData_.conv3dApiTiling.singleCoreWo
       << ", strideH: " << tilingData_.conv3dApiTiling.strideH << ", strideW: "
       << tilingData_.conv3dApiTiling.strideW << ", strideD: " << tilingData_.conv3dApiTiling.strideD
       << ", dilationH: " << tilingData_.conv3dApiTiling.dilationH << ", dilationW: "
       << tilingData_.conv3dApiTiling.dilationW << ", dilationD: " << tilingData_.conv3dApiTiling.dilationD
       << ", padHead: " << tilingData_.conv3dApiTiling.padHead
       << ", padTail: " << tilingData_.conv3dApiTiling.padTail
       << ", padTop: " << tilingData_.conv3dApiTiling.padTop
       << ", padBottom: " << tilingData_.conv3dApiTiling.padBottom
       << ", padLeft: " << tilingData_.conv3dApiTiling.padLeft
       << ", padRight: " << tilingData_.conv3dApiTiling.padRight
       << ", hoL0: " << tilingData_.conv3dApiTiling.hoL0 << ", woL0: " << tilingData_.conv3dApiTiling.woL0
       << ", kL0: " << tilingData_.conv3dApiTiling.kL0 << ", nL0: " << tilingData_.conv3dApiTiling.nL0
       << ", kAL1: " << tilingData_.conv3dApiTiling.kAL1 << ", kBL1: " << tilingData_.conv3dApiTiling.kBL1
       << ", nBL1: " << tilingData_.conv3dApiTiling.nBL1 << ", hoL1: " << tilingData_.conv3dApiTiling.hoL1
       << ", woL1: " << tilingData_.conv3dApiTiling.woL1
       << ", mUB: " << tilingData_.conv3dApiTiling.mUB
       << ", nUB: " << tilingData_.conv3dApiTiling.nUB
       << ", pBufferFlag: " << tilingData_.conv3dApiTiling.pBufferFlag
       << ", offsetx: " << static_cast<int32_t>(tilingData_.conv3dApiTiling.offsetx)
       << ", iterateMNOrder: " << static_cast<uint32_t>(tilingData_.conv3dApiTiling.iterateMNOrder)
       << ", biasFullLoadFlag: " << static_cast<uint32_t>(tilingData_.conv3dApiTiling.biasFullLoadFlag)
       << ", fixpParamsFullLoadFlag: "
       << static_cast<uint32_t>(tilingData_.conv3dApiTiling.fixpParamsFullLoadFlag)
       << ", enlarge: " << tilingData_.conv3dApiTiling.enlarge
       << ", singleCoreGroups: " << tilingData_.conv3dApiTiling.singleCoreGroups
       << ", singleCoreGroupOpt: " << tilingData_.conv3dApiTiling.singleCoreGroupOpt
       << ", hf32Enable: " << static_cast<uint32_t>(tilingData_.conv3dApiTiling.hf32Enable)
       << ", hf32TransMode: " << static_cast<uint32_t>(tilingData_.conv3dApiTiling.hf32TransMode)
       << ", hasBias: " << static_cast<uint32_t>(tilingData_.conv3dApiTiling.hasBias)
       << ", hasScale: " << static_cast<uint32_t>(tilingData_.conv3dApiTiling.hasScale)
       << ", roundMode: " << static_cast<uint32_t>(tilingData_.conv3dApiTiling.roundMode);
    OP_LOGD(context_->GetNodeName(), "%s", ss.str().c_str());
    PrintLibApiScalarTilingData();
    PrintLibApiSpaceSize();
}

void Conv3dBaseTilingV2::PrintLibApiSpaceSize()
{
    uint32_t pbAL0 = (tilingData_.conv3dApiTiling.pBufferFlag & PB_AL0_IDX) + 1;
    uint32_t pbBL0 = ((tilingData_.conv3dApiTiling.pBufferFlag & PB_BL0_IDX) >> PB_BL0_IDX) + 1;
    uint32_t pbCL0 = ((tilingData_.conv3dApiTiling.pBufferFlag & PB_CL0_IDX) >> PB_CL0_IDX) + 1;
    uint32_t pbBL1 = ((tilingData_.conv3dApiTiling.pBufferFlag & PB_BL1_IDX) >> PB_BL1_IDX) + 1;
    uint64_t biasL1Size = flagInfo_.hasBias ? ConvAlignB(tilingData_.conv3dApiTiling.nBL1 * descInfo_.biasDtype,
        C0_SIZE) : 0;
    uint64_t scaleL1Size = flagInfo_.quantFlag ?
        ConvAlignB(tilingData_.conv3dApiTiling.nBL1 * descInfo_.scaleDtype,
        C0_SIZE) : 0;
    uint64_t apiL1Size = (tilingData_.conv3dApiTiling.aL1SpaceSize + tilingData_.conv3dApiTiling.kBL1 *
        tilingData_.conv3dApiTiling.nBL1 * dtypeSizeTab.at(descInfo_.weightDtype) * pbBL1 +
        biasL1Size + scaleL1Size);
    uint64_t apiL0ASize = (tilingData_.conv3dApiTiling.kL0 * tilingData_.conv3dApiTiling.hoL0 *
        dtypeSizeTab.at(descInfo_.fMapDtype) * pbAL0);
    uint64_t apiL0BSize = (tilingData_.conv3dApiTiling.kL0 * tilingData_.conv3dApiTiling.nL0 *
        dtypeSizeTab.at(descInfo_.weightDtype) * pbBL0);
    uint64_t apiL0CSize = (tilingData_.conv3dApiTiling.hoL0 * tilingData_.conv3dApiTiling.nL0 *
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
        tilingData_.conv3dApiTiling.orgHixWi, tilingData_.conv3dApiTiling.kernelHxkernelW,
        tilingData_.conv3dApiTiling.kernelHxkernelWxkernelD, tilingData_.conv3dApiTiling.aL1SpaceSize,
        tilingData_.conv3dApiTiling.multiNBL1, tilingData_.conv3dApiTiling.cinAInCore,
        tilingData_.conv3dApiTiling.cinATailInCore, tilingData_.conv3dApiTiling.cinBInCore,
        tilingData_.conv3dApiTiling.cinBTailInCore, tilingData_.conv3dApiTiling.mStep,
        tilingData_.conv3dApiTiling.kStep, tilingData_.conv3dApiTiling.nStep,
        tilingData_.conv3dApiTiling.fmapKStride, tilingData_.conv3dApiTiling.weightKStride,
        tilingData_.conv3dApiTiling.cinOffsetBlockInGM, tilingData_.conv3dApiTiling.coutOffsetBlock,
        tilingData_.conv3dApiTiling.nL1DivBlockSize);
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