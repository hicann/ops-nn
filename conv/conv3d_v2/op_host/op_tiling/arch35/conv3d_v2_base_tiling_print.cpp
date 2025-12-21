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
    ss << paramInfo_.nodeType.c_str() << " AscendC: ops tilingdata: batch: " << tilingData_.conv3dRunInfo.get_batch()
       << ", cin: " << tilingData_.conv3dRunInfo.get_cin()
       << ", din: " << tilingData_.conv3dRunInfo.get_din()
       << ", hi: " << tilingData_.conv3dRunInfo.get_hin()
       << ", wi: " << tilingData_.conv3dRunInfo.get_win()
       << ", cout: " << tilingData_.conv3dRunInfo.get_cout()
       << ", kd: " << tilingData_.conv3dRunInfo.get_kd()
       << ", kh: " << tilingData_.conv3dRunInfo.get_kh()
       << ", kw: " << tilingData_.conv3dRunInfo.get_kw()
       << ", dout: " << tilingData_.conv3dRunInfo.get_dout()
       << ", hout: " << tilingData_.conv3dRunInfo.get_hout()
       << ", wout: " << tilingData_.conv3dRunInfo.get_wout()
       << ", batchDim: " << tilingData_.conv3dRunInfo.get_batchDim()
       << ", doDim: " << tilingData_.conv3dRunInfo.get_doDim()
       << ", hoDim: " << tilingData_.conv3dRunInfo.get_hoDim()
       << ", nDim: " << tilingData_.conv3dRunInfo.get_nDim()
       << ", groupDim: " << tilingData_.conv3dRunInfo.get_groupDim()
       << ", strideH: " << tilingData_.conv3dRunInfo.get_strideH()
       << ", strideD: " << tilingData_.conv3dRunInfo.get_strideD()
       << ", dilationH: " << tilingData_.conv3dRunInfo.get_dilationH()
       << ", dilationD: " << tilingData_.conv3dRunInfo.get_dilationD()
       << ", padHead: " << tilingData_.conv3dRunInfo.get_padHead()
       << ", padTop: " << tilingData_.conv3dRunInfo.get_padTop()
       << ", hasBias: " << static_cast<uint32_t>(tilingData_.conv3dRunInfo.get_hasBias())
       << ", groups: " << tilingData_.conv3dRunInfo.get_groups()
       << ", cinOpt: " << tilingData_.conv3dRunInfo.get_cinOpt()
       << ", coutOpt: " << tilingData_.conv3dRunInfo.get_coutOpt()
       << ", groupOpt: " << tilingData_.conv3dRunInfo.get_groupOpt()
       << ", enlarge: " << tilingData_.conv3dRunInfo.get_enlarge();
    OP_LOGD(context_->GetNodeName(), "%s", ss.str().c_str());
}

void Conv3dBaseTilingV2::PrintLibApiTilingData()
{
    stringstream ss;
    ss << paramInfo_.nodeType.c_str() << " AscendC: api tilingdata: groups: "
       << tilingData_.conv3dApiTiling.get_groups() << ", orgDo: " << tilingData_.conv3dApiTiling.get_orgDo()
       << ", orgCo: " << tilingData_.conv3dApiTiling.get_orgCo()
       << ", orgHo: " << tilingData_.conv3dApiTiling.get_orgHo()
       << ", orgWo: " << tilingData_.conv3dApiTiling.get_orgWo()
       << ", orgDi: " << tilingData_.conv3dApiTiling.get_orgDi()
       << ", orgCi: " << tilingData_.conv3dApiTiling.get_orgCi()
       << ", orgHi: " << tilingData_.conv3dApiTiling.get_orgHi()
       << ", orgWi: " << tilingData_.conv3dApiTiling.get_orgWi()
       << ", kernelD: " << tilingData_.conv3dApiTiling.get_kernelD()
       << ", kernelH: " << tilingData_.conv3dApiTiling.get_kernelH()
       << ", kernelW: " << tilingData_.conv3dApiTiling.get_kernelW()
       << ", singleCoreBatch: " << tilingData_.conv3dApiTiling.get_singleCoreBatch()
       << ", singleCoreCo: " << tilingData_.conv3dApiTiling.get_singleCoreCo()
       << ", singleCoreCi: " << tilingData_.conv3dApiTiling.get_singleCoreCi()
       << ", singleCoreDo: " << tilingData_.conv3dApiTiling.get_singleCoreDo()
       << ", singleCoreHo: " << tilingData_.conv3dApiTiling.get_singleCoreHo()
       << ", singleCoreWo: " << tilingData_.conv3dApiTiling.get_singleCoreWo()
       << ", strideH: " << tilingData_.conv3dApiTiling.get_strideH() << ", strideW: "
       << tilingData_.conv3dApiTiling.get_strideW() << ", strideD: " << tilingData_.conv3dApiTiling.get_strideD()
       << ", dilationH: " << tilingData_.conv3dApiTiling.get_dilationH() << ", dilationW: "
       << tilingData_.conv3dApiTiling.get_dilationW() << ", dilationD: " << tilingData_.conv3dApiTiling.get_dilationD()
       << ", padHead: " << tilingData_.conv3dApiTiling.get_padHead()
       << ", padTail: " << tilingData_.conv3dApiTiling.get_padTail()
       << ", padTop: " << tilingData_.conv3dApiTiling.get_padTop()
       << ", padBottom: " << tilingData_.conv3dApiTiling.get_padBottom()
       << ", padLeft: " << tilingData_.conv3dApiTiling.get_padLeft()
       << ", padRight: " << tilingData_.conv3dApiTiling.get_padRight()
       << ", hoL0: " << tilingData_.conv3dApiTiling.get_hoL0() << ", woL0: " << tilingData_.conv3dApiTiling.get_woL0()
       << ", kL0: " << tilingData_.conv3dApiTiling.get_kL0() << ", nL0: " << tilingData_.conv3dApiTiling.get_nL0()
       << ", kAL1: " << tilingData_.conv3dApiTiling.get_kAL1() << ", kBL1: " << tilingData_.conv3dApiTiling.get_kBL1()
       << ", nBL1: " << tilingData_.conv3dApiTiling.get_nBL1() << ", hoL1: " << tilingData_.conv3dApiTiling.get_hoL1()
       << ", woL1: " << tilingData_.conv3dApiTiling.get_woL1()
       << ", pBufferFlag: " << tilingData_.conv3dApiTiling.get_pBufferFlag()
       << ", offsetx: " << static_cast<int32_t>(tilingData_.conv3dApiTiling.get_offsetx())
       << ", iterateMNOrder: " << static_cast<uint32_t>(tilingData_.conv3dApiTiling.get_iterateMNOrder())
       << ", biasFullLoadFlag: " << static_cast<uint32_t>(tilingData_.conv3dApiTiling.get_biasFullLoadFlag())
       << ", fixpParamsFullLoadFlag: "
       << static_cast<uint32_t>(tilingData_.conv3dApiTiling.get_fixpParamsFullLoadFlag())
       << ", enlarge: " << tilingData_.conv3dApiTiling.get_enlarge()
       << ", singleCoreGroups: " << tilingData_.conv3dApiTiling.get_singleCoreGroups()
       << ", singleCoreGroupOpt: " << tilingData_.conv3dApiTiling.get_singleCoreGroupOpt()
       << ", hf32Enable: " << static_cast<uint32_t>(tilingData_.conv3dApiTiling.get_hf32Enable())
       << ", hf32TransMode: " << static_cast<uint32_t>(tilingData_.conv3dApiTiling.get_hf32TransMode())
       << ", hasBias: " << static_cast<uint32_t>(tilingData_.conv3dApiTiling.get_hasBias())
       << ", hasScale: " << static_cast<uint32_t>(tilingData_.conv3dApiTiling.get_hasScale())
       << ", roundMode: " << static_cast<uint32_t>(tilingData_.conv3dApiTiling.get_roundMode());
    OP_LOGD(context_->GetNodeName(), "%s", ss.str().c_str());
    PrintLibApiScalarTilingData();
    PrintLibApiSpaceSize();
}

void Conv3dBaseTilingV2::PrintLibApiScalarTilingData()
{
    OP_LOGD(context_->GetNodeName(),
        "Conv3D AscendC: api scalar tilingdata: orgHixWi: %lu, kernelHxkernelW: %lu, kernelHxkernelWxkernelD: %u,"\
        "aL1SpaceSize: %u, multiNBL1: %u, cinAInCore: %u, cinATailInCore: %u, cinBInCore: %u,"\
        "cinBTailInCore: %u, mStep: %u, kStep: %u, nStep: %u, fmapKStride: %u, weightKStride: %u,"\
        "cinOffsetBlockInGM: %u, coutOffsetBlock: %u, nL1DivBlockSize: %u",
        tilingData_.conv3dApiTiling.get_orgHixWi(), tilingData_.conv3dApiTiling.get_kernelHxkernelW(),
        tilingData_.conv3dApiTiling.get_kernelHxkernelWxkernelD(), tilingData_.conv3dApiTiling.get_aL1SpaceSize(),
        tilingData_.conv3dApiTiling.get_multiNBL1(), tilingData_.conv3dApiTiling.get_cinAInCore(),
        tilingData_.conv3dApiTiling.get_cinATailInCore(), tilingData_.conv3dApiTiling.get_cinBInCore(),
        tilingData_.conv3dApiTiling.get_cinBTailInCore(), tilingData_.conv3dApiTiling.get_mStep(),
        tilingData_.conv3dApiTiling.get_kStep(), tilingData_.conv3dApiTiling.get_nStep(),
        tilingData_.conv3dApiTiling.get_fmapKStride(), tilingData_.conv3dApiTiling.get_weightKStride(),
        tilingData_.conv3dApiTiling.get_cinOffsetBlockInGM(), tilingData_.conv3dApiTiling.get_coutOffsetBlock(),
        tilingData_.conv3dApiTiling.get_nL1DivBlockSize());
}

void Conv3dBaseTilingV2::PrintLibApiSpaceSize()
{
    uint32_t pbAL0 = (tilingData_.conv3dApiTiling.get_pBufferFlag() & PB_AL0_IDX) + 1;
    uint32_t pbBL0 = ((tilingData_.conv3dApiTiling.get_pBufferFlag() & PB_BL0_IDX) >> PB_BL0_IDX) + 1;
    uint32_t pbCL0 = ((tilingData_.conv3dApiTiling.get_pBufferFlag() & PB_CL0_IDX) >> PB_CL0_IDX) + 1;
    uint32_t pbBL1 = ((tilingData_.conv3dApiTiling.get_pBufferFlag() & PB_BL1_IDX) >> PB_BL1_IDX) + 1;
    uint64_t biasL1Size = flagInfo_.hasBias ? ConvAlignB(tilingData_.conv3dApiTiling.get_nBL1() * descInfo_.biasDtype,
        C0_SIZE) : 0;
    uint64_t scaleL1Size = flagInfo_.quantFlag ?
        ConvAlignB(tilingData_.conv3dApiTiling.get_nBL1() * descInfo_.quantScaleDtype,
        C0_SIZE) : 0;
    uint64_t apiL1Size = (tilingData_.conv3dApiTiling.get_aL1SpaceSize() + tilingData_.conv3dApiTiling.get_kBL1() *
        tilingData_.conv3dApiTiling.get_nBL1() * dtypeSizeTab.at(descInfo_.weightDtype) * pbBL1 +
        biasL1Size + scaleL1Size);
    uint64_t apiL0ASize = (tilingData_.conv3dApiTiling.get_kL0() * tilingData_.conv3dApiTiling.get_hoL0() *
        dtypeSizeTab.at(descInfo_.fMapDtype) * pbAL0);
    uint64_t apiL0BSize = (tilingData_.conv3dApiTiling.get_kL0() * tilingData_.conv3dApiTiling.get_nL0() *
        dtypeSizeTab.at(descInfo_.weightDtype) * pbBL0);
    uint64_t apiL0CSize = (tilingData_.conv3dApiTiling.get_hoL0() * tilingData_.conv3dApiTiling.get_nL0() *
        dtypeSizeTab.at(descInfo_.outDtype) * pbCL0);
    OP_LOGD(context_->GetNodeName(),
        "%s AscendC: api space size: apiL1Size: %lu, apiL0ASize: %lu, apiL0BSize: %lu, apiL0CSize: %lu",
        paramInfo_.nodeType.c_str(), apiL1Size, apiL0ASize, apiL0BSize, apiL0CSize);
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