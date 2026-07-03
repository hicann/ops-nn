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
    ss << paramInfo_.nodeType.c_str() << " AscendC: ops tilingdata: batch: " << tilingData_.batch
       << ", cin: " << tilingData_.cin
       << ", din: " << tilingData_.din
       << ", hi: " << tilingData_.hin
       << ", wi: " << tilingData_.win
       << ", cout: " << tilingData_.cout
       << ", kd: " << tilingData_.kd
       << ", kh: " << tilingData_.kh
       << ", kw: " << tilingData_.kw
       << ", dout: " << tilingData_.dout
       << ", hout: " << tilingData_.hout
       << ", wout: " << tilingData_.wout
       << ", batchDim: " << tilingData_.batchDim
       << ", doDim: " << tilingData_.doDim
       << ", hoDim: " << tilingData_.hoDim
       << ", nDim: " << tilingData_.nDim
       << ", groupDim: " << tilingData_.groupDim
       << ", strideH: " << tilingData_.strideH
       << ", strideD: " << tilingData_.strideD
       << ", dilationH: " << tilingData_.dilationH
       << ", dilationD: " << tilingData_.dilationD
       << ", padHead: " << tilingData_.padHead
       << ", padTop: " << tilingData_.padTop
       << ", hasBias: " << static_cast<uint32_t>(tilingData_.hasBias)
       << ", groups: " << tilingData_.groups
       << ", cinOpt: " << tilingData_.cinOpt
       << ", coutOpt: " << tilingData_.coutOpt
       << ", groupOpt: " << tilingData_.groupOpt
       << ", enlarge: " << tilingData_.enlarge;
    OP_LOGD(context_->GetNodeName(), "%s", ss.str().c_str());
}

void Conv3dBaseTilingV2::PrintLibApiTilingData()
{
    stringstream ss;
    ss << paramInfo_.nodeType.c_str() << " AscendC: api tilingdata: groups: "
       << tilingData_.groups << ", orgDo: " << tilingData_.orgDo
       << ", orgCo: " << tilingData_.orgCo
       << ", orgHo: " << tilingData_.orgHo
       << ", orgWo: " << tilingData_.orgWo
       << ", orgDi: " << tilingData_.orgDi
       << ", orgCi: " << tilingData_.orgCi
       << ", orgHi: " << tilingData_.orgHi
       << ", orgWi: " << tilingData_.orgWi
       << ", kernelD: " << tilingData_.kernelD
       << ", kernelH: " << tilingData_.kernelH
       << ", kernelW: " << tilingData_.kernelW
       << ", singleCoreBatch: " << tilingData_.singleCoreBatch
       << ", singleCoreCo: " << tilingData_.singleCoreCo
       << ", singleCoreCi: " << tilingData_.singleCoreCi
       << ", singleCoreDo: " << tilingData_.singleCoreDo
       << ", singleCoreHo: " << tilingData_.singleCoreHo
       << ", singleCoreWo: " << tilingData_.singleCoreWo
       << ", strideH: " << tilingData_.strideH << ", strideW: "
       << tilingData_.strideW << ", strideD: " << tilingData_.strideD
       << ", dilationH: " << tilingData_.dilationH << ", dilationW: "
       << tilingData_.dilationW << ", dilationD: " << tilingData_.dilationD
       << ", padHead: " << tilingData_.padHead
       << ", padTail: " << tilingData_.padTail
       << ", padTop: " << tilingData_.padTop
       << ", padBottom: " << tilingData_.padBottom
       << ", padLeft: " << tilingData_.padLeft
       << ", padRight: " << tilingData_.padRight
       << ", hoL0: " << tilingData_.hoL0 << ", woL0: " << tilingData_.woL0
       << ", kL0: " << tilingData_.kL0 << ", nL0: " << tilingData_.nL0
       << ", kAL1: " << tilingData_.kAL1 << ", kBL1: " << tilingData_.kBL1
       << ", nBL1: " << tilingData_.nBL1 << ", hoL1: " << tilingData_.hoL1
       << ", woL1: " << tilingData_.woL1
       << ", mUB: " << tilingData_.mUB
       << ", nUB: " << tilingData_.nUB
       << ", khL1: " << tilingData_.khL1
       << ", kwL1: " << tilingData_.kwL1
       << ", pBufferFlag: " << tilingData_.pBufferFlag
       << ", offsetx: " << static_cast<int32_t>(tilingData_.offsetx)
       << ", iterateMNOrder: " << static_cast<uint32_t>(tilingData_.iterateMNOrder)
       << ", biasFullLoadFlag: " << static_cast<uint32_t>(tilingData_.biasFullLoadFlag)
       << ", fixpParamsFullLoadFlag: "
       << static_cast<uint32_t>(tilingData_.fixpParamsFullLoadFlag)
       << ", enlarge: " << tilingData_.enlarge
       << ", singleCoreGroups: " << tilingData_.singleCoreGroups
       << ", singleCoreGroupOpt: " << tilingData_.singleCoreGroupOpt
       << ", hf32Enable: " << static_cast<uint32_t>(tilingData_.hf32Enable)
       << ", hf32TransMode: " << static_cast<uint32_t>(tilingData_.hf32TransMode);
    OP_LOGD(context_->GetNodeName(), "%s", ss.str().c_str());

    stringstream ssPartTwo;
    ssPartTwo << paramInfo_.nodeType.c_str() << " AscendC: api tilingdata: groups: " << tilingData_.groups
       << ", hasBias: " << static_cast<uint32_t>(tilingData_.hasBias)
       << ", hasScale: " << static_cast<uint32_t>(tilingData_.hasScale)
       << ", roundMode: " << static_cast<uint32_t>(tilingData_.roundMode)
       << ", unionDataXt: " << static_cast<uint32_t>(tilingData_.unionDataXt);
    OP_LOGD(context_->GetNodeName(), "%s", ssPartTwo.str().c_str());

    PrintLibApiScalarTilingData();
    PrintLibApiSpaceSize();
}

void Conv3dBaseTilingV2::PrintLibApiSpaceSize()
{
    uint32_t pbAL0 = (tilingData_.pBufferFlag & PB_AL0_IDX) + 1;
    uint32_t pbBL0 = ((tilingData_.pBufferFlag & PB_BL0_IDX) >> PB_BL0_IDX) + 1;
    uint32_t pbCL0 = ((tilingData_.pBufferFlag & PB_CL0_IDX) >> PB_CL0_IDX) + 1;
    uint32_t pbAL1 = ((tilingData_.pBufferFlag & PB_AL1_IDX) >> PB_AL1_IDX) + 1;
    uint32_t pbBL1 = ((tilingData_.pBufferFlag & PB_BL1_IDX) >> PB_BL1_IDX) + 1;
    uint64_t biasL1Size = flagInfo_.hasBias ? ConvAlignB(tilingData_.nBL1 *
        static_cast<uint64_t>(descInfo_.biasDtype), C0_SIZE) : 0;
    uint64_t scaleL1Size = flagInfo_.quantFlag ?
        ConvAlignB(tilingData_.nBL1 * static_cast<uint64_t>(descInfo_.scaleDtype),
        C0_SIZE) : 0;
    uint64_t apiL1Size = (tilingData_.aL1SpaceSize * pbAL1 + tilingData_.kBL1 *
        tilingData_.nBL1 * dtypeSizeTab.at(descInfo_.weightDtype) * pbBL1 +
        biasL1Size + scaleL1Size);
    uint64_t apiL0ASize = (tilingData_.kL0 * tilingData_.hoL0 *
        dtypeSizeTab.at(descInfo_.fMapDtype) * pbAL0);
    uint64_t apiL0BSize = (tilingData_.kL0 * tilingData_.nL0 *
        dtypeSizeTab.at(descInfo_.weightDtype) * pbBL0);
    uint64_t apiL0CSize = (tilingData_.hoL0 * tilingData_.nL0 *
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
        tilingData_.orgHixWi, tilingData_.kernelHxkernelW,
        tilingData_.kernelHxkernelWxkernelD, tilingData_.aL1SpaceSize,
        tilingData_.multiNBL1, tilingData_.cinAInCore,
        tilingData_.cinATailInCore, tilingData_.cinBInCore,
        tilingData_.cinBTailInCore, tilingData_.mStep,
        tilingData_.kStep, tilingData_.nStep,
        tilingData_.fmapKStride, tilingData_.weightKStride,
        tilingData_.cinOffsetBlockInGM, tilingData_.coutOffsetBlock,
        tilingData_.nL1DivBlockSize);
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