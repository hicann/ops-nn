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
 * \file fused_linear_cross_entropy_loss_grad_tiling.cpp
 * \brief
 */
#include "fused_linear_cross_entropy_loss_grad_tiling.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include "log/log.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

static ge::graphStatus HighPerfTilingFunc(gert::TilingContext* context)
{
    FusedLinearCrossEntropyLossGradHighPerfTilingData tiling;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubByteSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubByteSize);
    auto inputShape = context->GetInputShape(INDEX_1)->GetOriginShape();
    auto softmaxShape = context->GetOptionalInputTensor(INDEX_7)->GetOriginShape();
    auto inputDtype = context->GetInputTensor(INDEX_1)->GetDataType();

    // all val def
    uint32_t aicNum;    // aic核心数
    uint32_t aivNum;    // aiv核心数
    uint64_t BT;    // softmax为[BT, V]，weight为[V, H]
    uint64_t V;
    uint64_t H;
    uint64_t BT_16ALIGNED;
    uint64_t btPerCore; // 每个满载核心任务数（update操作，下同）
    uint64_t btLastCore;    // 最后一个核心的任务数。如果BT_LAST_CORE为0，则无跑而未满核心，可作为LAST_CORE是否启动条件
    uint32_t btLastCoreId;  // 满载核心数，即LAST_CORE的ID
    uint64_t targetMaskSize;    // target_mask的形式长度
    uint64_t VByteSize; // V * sizeof(float)
    uint64_t VOutByteSize;  // V * sizeof(half)
    uint64_t V64BAlignedSize;   // 每行64B对齐的元素数量
    uint32_t copyOutDstStride;  // 每行输出对齐到512B还需填充的字节大小
    uint64_t VOut512BAlignedSize;   // 每行输出512B对齐的元素数量
    uint64_t VOut32BAlignedSize;    // 每行输出32B对齐的元素数量
    uint32_t btPerEpochSingle = 0;  // 每轮单核处理的行数（muls & cast操作，下同），如果单核处理不到1行，该值为0
    uint32_t btPerEpochTotal;   // 每轮处理的总行数
    uint32_t epochCnt;  // 处理singleA所需轮数
    uint32_t lastEpochCnt;  // 处理singleA尾块所需轮数
    uint32_t quebindSoftmaxByteSize;    // que buffer字节大小
    uint32_t copyInDstStride;   // 已满足64B对齐就不插stride
    uint32_t using32BlockCnt;   // 将使用的UB空间块数（偶数个32B块数）
    uint32_t sizePerTask;   // 每个任务块处理的输入元素长度。如果单核处理不到1行，将对1行分块成多个任务块，交给多核处理
    uint32_t taskCntPerV;   // 一行需几块任务
    uint32_t sizeLastTask;  // 尾块任务含有的实际元素数量
    uint64_t taskCntTotal;  // 总任务块数
    uint32_t mm1BaseMCnt;   // 第一轮mm的左矩阵能划分多少块baseM
    uint32_t mm1BaseNCnt;   // 第一轮mm的左矩阵能划分多少块baseN
    uint32_t mm1TailM;  // 第一轮mm尾块有效M长度
    uint32_t mm1TailN;  // 第一轮mm尾块有效N长度
    uint32_t mm2BaseMCnt;   // 第二轮mm，参数含义同上
    uint32_t mm2BaseNCnt;
    uint32_t mm2BaseKCnt;
    uint32_t mm2TailM;
    uint32_t mm2TailN;
    uint32_t mm2TailK;
    uint32_t mm2BasePerVec;    // 每个满载核心任务数（mm2左矩阵nd2nz操作，1个基本块为1个任务，下同）
    uint32_t mm2BaseLastVec;    // 最后一个核心的任务数。如果BT_LAST_CORE为0，则无跑而未满核心，可作为LAST_CORE是否启动条件
    uint32_t mm2BaseLastVecId;  // 满载核心数，即LAST_CORE的ID
    uint32_t mm2LastBaseK;  // K方向最后一个基本块的K方向的有效大小
    uint32_t wsNum;  // workspace空间包含singleA的块数
    uint64_t userWorkspaceSize;

    // basic info
    aicNum = ascendcPlatform.GetCoreNumAic();
    aivNum = aicNum * SIZE_2;
    BT = (uint64_t)softmaxShape[0];
    V = (uint64_t)softmaxShape[1];
    H = (uint64_t)inputShape[1];
    BT_16ALIGNED = (BT - 1U) / SIZE_16 * SIZE_16 + SIZE_16;
    
    if (BT >= BT_BOUNDARY) {
        baseM = SIZE_256;
        baseN = SIZE_128;
    }

    // cube info
    matmul_tiling::DataType halfDtype = matmul_tiling::DataType::DT_FLOAT16;
    if (inputDtype == ge::DT_BF16) {
        halfDtype = matmul_tiling::DataType::DT_BFLOAT16;
    }
    // mm1 api
    matmul_tiling::MultiCoreMatmulTiling mm1Tiling(ascendcPlatform);
    mm1Tiling.SetDim(aicNum); // 参与计算的cube核数
    mm1Tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, halfDtype);
    mm1Tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, halfDtype);
    mm1Tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, halfDtype);
    mm1Tiling.SetBias(false);
    mm1Tiling.SetShape(baseM, H, V);
    mm1Tiling.SetOrgShape(baseM, H, V);
    mm1Tiling.SetSingleShape(baseM, baseN, V);
    mm1Tiling.SetFixSplit(baseM, baseN, baseK);
    int64_t ret = mm1Tiling.GetTiling(tiling.mm1Tiling);  // if ret = -1, get tiling failed
    if (ret == -1) {
        OP_LOGE(context->GetNodeName(), "GetTiling for FusedLinearCrossEntropyLossGrad mm1 failed.");
        return ge::GRAPH_FAILED;
    };
    tiling.mm1Tiling.set_stepKa(SIZE_4);
    tiling.mm1Tiling.set_stepKb(SIZE_4);
    tiling.mm1Tiling.set_depthA1(SIZE_8);
    tiling.mm1Tiling.set_depthB1(SIZE_8);
    // mm1 other
    mm1BaseMCnt = (BT - 1) / mm1Tiling.GetBaseM() + 1;
    mm1BaseNCnt = (H - 1) / mm1Tiling.GetBaseN() + 1;
    mm1TailM = BT % mm1Tiling.GetBaseM();  // 尾块有效M长度
    mm1TailN = H % mm1Tiling.GetBaseN();  // 尾块有效N长度
    if (mm1TailM == 0) mm1TailM = mm1Tiling.GetBaseM();
    if (mm1TailN == 0) mm1TailN = mm1Tiling.GetBaseN();

    // mm2 api
    matmul_tiling::MultiCoreMatmulTiling mm2Tiling(ascendcPlatform);
    mm2Tiling.SetDim(aicNum); // 参与计算的cube核数
    mm2Tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, halfDtype, true);
    mm2Tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, halfDtype);
    mm2Tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, halfDtype);
    mm2Tiling.SetBias(false);
    mm2Tiling.SetShape(baseM, H, BT);
    mm2Tiling.SetOrgShape(baseM, H, BT_16ALIGNED, BT);
    mm2Tiling.SetSingleShape(baseM, baseN, BT);
    mm2Tiling.SetFixSplit(baseM, baseN, baseK);
    ret = mm2Tiling.GetTiling(tiling.mm2Tiling);  // if ret = -1, get tiling failed
    if (ret == -1) {
        OP_LOGE(context->GetNodeName(), "GetTiling for FusedLinearCrossEntropyLossGrad mm2 failed.");
        return ge::GRAPH_FAILED;
    };
    tiling.mm2Tiling.set_stepKa(SIZE_4);
    tiling.mm2Tiling.set_stepKb(SIZE_4);
    tiling.mm2Tiling.set_depthA1(SIZE_8);
    tiling.mm2Tiling.set_depthB1(SIZE_8);
    tiling.mm2Tiling.set_baseM(baseM);
    tiling.mm2Tiling.set_baseN(baseN);
    tiling.mm2Tiling.set_baseK(baseK);
    // mm2 other
    mm2BaseMCnt = (V - 1) / mm2Tiling.GetBaseM() + 1;
    mm2BaseNCnt = (H - 1) / mm2Tiling.GetBaseN() + 1;
    mm2BaseKCnt = (BT - 1) / mm2Tiling.GetBaseK() + 1;
    mm2TailM = V % mm2Tiling.GetBaseM();  // 尾块有效M长度
    mm2TailN = H % mm2Tiling.GetBaseN();  // 尾块有效N长度
    mm2TailK = BT % mm2Tiling.GetBaseK();  // 尾块有效K长度
    if (mm2TailM == 0) mm2TailM = mm2Tiling.GetBaseM();
    if (mm2TailN == 0) mm2TailN = mm2Tiling.GetBaseN();
    if (mm2TailK == 0) mm2TailK = mm2Tiling.GetBaseK();

    // vec info
    VByteSize = V * IN_BYTE_SIZE;
    VOutByteSize = V * OUT_BYTE_SIZE;
    btPerCore = (BT - 1U) / aivNum + 1U;  // 每个满载核心任务数
    btLastCore = BT % btPerCore;  // 最后一个核心的任务数。如果BT_LAST_CORE为0，则无跑而未满核心，可作为LAST_CORE是否启动条件
    btLastCoreId = BT / btPerCore;  // 满载核心数，即LAST_CORE的ID
    targetMaskSize = context->GetInputTensor(INDEX_3)->GetShapeSize();
    uint64_t V32BAlignedBlockCnt = (VByteSize - 1U) / SIZE_32 + 1U;  // 每行32B对齐的元素数量，用于推测搬入参数
    V64BAlignedSize = ((VByteSize - 1U) / SIZE_64 + 1U) * (SIZE_64 / IN_BYTE_SIZE);  // 每行64B对齐的元素数量
    uint64_t VOut512BAlignedByteSize = ((VOutByteSize - 1U) / SIZE_512 + 1U) * SIZE_512;  // 每行输出512B对齐的字节大小
    uint64_t VOut32BAlignedByteSize = ((VOutByteSize - 1U) / SIZE_32 + 1U) * SIZE_32;  // 每行输出32B对齐的字节大小
    copyOutDstStride = VOut512BAlignedByteSize - VOutByteSize;  // 每行输出对齐到512B还需填充的字节大小
    VOut512BAlignedSize = VOut512BAlignedByteSize / OUT_BYTE_SIZE;  // 每行输出512B对齐的元素数量
    VOut32BAlignedSize = VOut32BAlignedByteSize / OUT_BYTE_SIZE;  // 每行输出32B对齐的元素数量
    btPerEpochSingle = ((ubByteSize - TBUF_BYTE_SIZE) / BUFFER_NUM) / (V64BAlignedSize * IN_BYTE_SIZE);  // 每轮单核处理的行数
    if (btPerEpochSingle) {
        // full case
        btPerEpochTotal = btPerEpochSingle * aivNum;  // 每轮能够处理的总行数
        epochCnt = (baseM - 1U) / btPerEpochTotal + 1U;  // singleA所需轮数
        btPerEpochSingle = (baseM - 1U) / (epochCnt * aivNum) + 1U;
        btPerEpochTotal = btPerEpochSingle * aivNum;
        lastEpochCnt = (mm1TailM - 1U) / btPerEpochTotal + 1U;  // singleA尾块所需轮数
        quebindSoftmaxByteSize = V64BAlignedSize * btPerEpochSingle * IN_BYTE_SIZE;
        copyInDstStride = (V32BAlignedBlockCnt % SIZE_2 == 0) ? 0U : 1U;  // 已满足64B对齐就不插stride
    } else {
        // partial case
        using32BlockCnt = ((ubByteSize - TBUF_BYTE_SIZE) / BUFFER_NUM) / SIZE_32 / SIZE_2 * SIZE_2;  // 将使用的UB空间块数（偶数个32B块数）
        sizePerTask = using32BlockCnt * SIZE_32 / IN_BYTE_SIZE;  // 每个任务块处理的输入元素长度
        quebindSoftmaxByteSize = sizePerTask;
        taskCntPerV = (V - 1U) / sizePerTask + 1U;  // 一行需几块任务
        sizeLastTask = (uint32_t)(V - (taskCntPerV - 1U) * sizePerTask);  // 尾块任务含有的实际元素数量
        taskCntTotal = taskCntPerV * BT;  // 总任务块数
        epochCnt = (taskCntTotal - 1U) / aivNum + 1U;  // 总轮数
    }
    quebindSoftmaxByteSize = std::max(quebindSoftmaxByteSize, MAX_LEFT_BASE_BYTE_SIZE * 2);  // 同时需要保证mm2左矩阵nd2nz必需的空间
    
    // vec mm2 nd2nz
    mm2BasePerVec = (mm2BaseKCnt - 1U) / aivNum + 1U;
    mm2BaseLastVec = mm2BaseKCnt % mm2BasePerVec;
    mm2BaseLastVecId = mm2BaseKCnt / mm2BasePerVec;
    mm2LastBaseK = (uint64_t)mm2BaseKCnt * (uint64_t)baseK - BT;

    // workspace info
    wsNum = ((aicNum * SIZE_2 - 1U - 1U) / mm1BaseNCnt + 1U) + 1U;
    wsNum = std::min(wsNum, std::max(mm1BaseMCnt, mm2BaseMCnt));
    userWorkspaceSize = std::max(VOut32BAlignedSize, BT_16ALIGNED) * baseM * wsNum;  // 用于存放左矩阵nd2nz的结果，多块用以保证vec在处理时没有cube空闲

    // validation
    // bt
    if (BT_16ALIGNED - mm2TailK > SIZE_65535) {
        OP_LOGE(context->GetNodeName(), "[FusedLinearCrossEntropyLossGrad] softmax.size(0) greater than 65536 is not supported now.");
        return ge::GRAPH_FAILED;
    };
    // v
    if (V * IN_BYTE_SIZE < SIZE_512) {
        OP_LOGE(context->GetNodeName(), "[FusedLinearCrossEntropyLossGrad] bytesize of softmax.size(1) smaller than 512B is not supported.");
        return ge::GRAPH_FAILED;
    };
    if (V > (uint64_t)((SIZE_256 - 1) * SIZE_256)) {  // VOut512BAlignedByteSize < 65536
        OP_LOGE(context->GetNodeName(), "[FusedLinearCrossEntropyLossGrad] softmax.size(1) greater than 65280 is not supported now.");
        return ge::GRAPH_FAILED;
    };
    // h
    if (H > SIZE_65535) {
        OP_LOGE(context->GetNodeName(), "[FusedLinearCrossEntropyLossGrad] input.size(1) greater than 65535 is not supported now.");
        return ge::GRAPH_FAILED;
    };
    // others
    if (btPerEpochSingle == 0U) {
        OP_LOGE(context->GetNodeName(), "[FusedLinearCrossEntropyLossGrad] softmax.size(1) is too large and is not supported now.");
        return ge::GRAPH_FAILED;
    }

    // tiling填充
    tiling.set_aicNum(aicNum);
    tiling.set_aivNum(aivNum);
    tiling.set_BT(BT);
    tiling.set_V(V);
    tiling.set_H(H);
    tiling.set_BT_16ALIGNED(BT_16ALIGNED);
    tiling.set_btPerCore(btPerCore);
    tiling.set_btLastCore(btLastCore);
    tiling.set_btLastCoreId(btLastCoreId);
    tiling.set_targetMaskSize(targetMaskSize);
    tiling.set_V64BAlignedSize(V64BAlignedSize);
    tiling.set_copyOutDstStride(copyOutDstStride);
    tiling.set_VOut512BAlignedSize(VOut512BAlignedSize);
    tiling.set_VOut32BAlignedSize(VOut32BAlignedSize);
    tiling.set_btPerEpochSingle(btPerEpochSingle);
    tiling.set_btPerEpochTotal(btPerEpochTotal);
    tiling.set_quebindSoftmaxByteSize(quebindSoftmaxByteSize);
    tiling.set_copyInDstStride(copyInDstStride);
    tiling.set_epochCnt(epochCnt);
    tiling.set_lastEpochCnt(lastEpochCnt);
    tiling.set_using32BlockCnt(using32BlockCnt);
    tiling.set_sizePerTask(sizePerTask);
    tiling.set_taskCntPerV(taskCntPerV);
    tiling.set_sizeLastTask(sizeLastTask);
    tiling.set_taskCntTotal(taskCntTotal);
    tiling.set_mm1BaseMCnt(mm1BaseMCnt);
    tiling.set_mm1BaseNCnt(mm1BaseNCnt);
    tiling.set_mm1TailM(mm1TailM);
    tiling.set_mm1TailN(mm1TailN);
    tiling.set_mm2BaseMCnt(mm2BaseMCnt);
    tiling.set_mm2BaseNCnt(mm2BaseNCnt);
    tiling.set_mm2BaseKCnt(mm2BaseKCnt);
    tiling.set_mm2TailM(mm2TailM);
    tiling.set_mm2TailN(mm2TailN);
    tiling.set_mm2TailK(mm2TailK);
    tiling.set_mm2BasePerVec(mm2BasePerVec);
    tiling.set_mm2BaseLastVec(mm2BaseLastVec);
    tiling.set_mm2BaseLastVecId(mm2BaseLastVecId);
    tiling.set_mm2LastBaseK(mm2LastBaseK);
    tiling.set_userWorkspaceSize(userWorkspaceSize);
    tiling.set_wsNum(wsNum);

    // tiling设定
    context->SetBlockDim(aicNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(HIGH_PERF_KEY);
    // 设置workspace大小
    size_t userWorkspaceByteSize = userWorkspaceSize * OUT_BYTE_SIZE;
    size_t sysWorkspaceByteSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = userWorkspaceByteSize + sysWorkspaceByteSize;

    return ge::GRAPH_SUCCESS;
}

static void CalcUBUsage(
    uint32_t ubByteSize,
    uint32_t H,
    uint32_t& mmAccumRowOnetime,
    uint32_t& mmAccumCyclePerRow,
    uint32_t& mmAccumCntPerCycle,
    uint32_t& mmAccumCntLastCycle)
{
    // 1. 每行占用的字节数（float32 = 4 Bytes）
    uint64_t rowBytes = static_cast<uint64_t>(H) * SIZE_4;

    // 2. 计算一次能搬运的最大行数（不足 1 行仍算 1 行）
    if (rowBytes == 0U) {               // 防止除 0（理论上 H 不会为 0）
        mmAccumRowOnetime = 0U;
        mmAccumCyclePerRow = mmAccumCntPerCycle = mmAccumCntLastCycle = 0U;
        return;
    }

    mmAccumRowOnetime = static_cast<uint32_t>(
        std::max<uint64_t>(1ULL, ubByteSize / rowBytes));

    // 3. 多行情况：只需要返回行数，其他参数置 0 即可
    if (mmAccumRowOnetime >= SIZE_2) {
        mmAccumCyclePerRow = mmAccumCntPerCycle = mmAccumCntLastCycle = 0U;
        return;
    }

    // 4. 单行（或不足一行）情况，需要分批搬运
    // 每次最多搬运的元素个数（UBByteSize / 4 向下取整）
    mmAccumCntPerCycle = ubByteSize / SIZE_4;

    // 为避免除 0，若 ubByteSize < 4，则强制每次搬运 1 个元素（最小粒度）
    if (mmAccumCntPerCycle == 0U) {
        mmAccumCntPerCycle = 1U;
    }

    // 每次搬运的字节数
    uint64_t bytesPerCycle = static_cast<uint64_t>(mmAccumCntPerCycle) * SIZE_4;

    // 需要的循环次数（向上取整）
    mmAccumCyclePerRow = static_cast<uint32_t>(
        (rowBytes + bytesPerCycle - 1ULL) / bytesPerCycle);

    // 最后一轮搬运的元素数
    uint64_t remainingBytes = rowBytes % bytesPerCycle;
    if (remainingBytes == 0U) {
        mmAccumCntLastCycle = mmAccumCntPerCycle;
    } else {
        // 这里的除法一定能整除，因为 remainingBytes 是 4 的倍数
        mmAccumCntLastCycle = static_cast<uint32_t>(remainingBytes / SIZE_4);
    }
}

static ge::graphStatus MemFriendlyTilingFunc(gert::TilingContext* context)
{
    FusedLinearCrossEntropyLossGradMemFriendlyTilingData tiling;
    // 场景检查
    auto logitsMax = context->GetInputTensor(INDEX_5);
    auto sumExpLogits = context->GetInputTensor(INDEX_6);
    if (logitsMax == nullptr or sumExpLogits == nullptr) {
        OP_LOGE(context->GetNodeName(), "logitsMax or sumExpLogits can not be nullptr when softmax is nullptr.");
        return ge::GRAPH_FAILED;
    }
    // 信息获取
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubByteSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubByteSize);
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    if (aicNum == 0U) {
        aicNum = 1U;
    }
    uint32_t aivNum = aicNum * SIZE_2;
    auto gradShape = context->GetInputShape(INDEX_0)->GetOriginShape();
    auto weightShape = context->GetInputShape(INDEX_2)->GetOriginShape();
    auto inputDtype = context->GetInputTensor(INDEX_1)->GetDataType();
    uint32_t BT = static_cast<uint32_t>(gradShape[0]);
    uint32_t V = static_cast<uint32_t>(weightShape[0]);
    uint32_t H = static_cast<uint32_t>(weightShape[1]);
    uint32_t targetMaskSize = static_cast<uint32_t>(context->GetInputTensor(INDEX_3)->GetShapeSize());  // target_mask的形式长度

    // 数据准备
    uint32_t baseBT = SIZE_128;
    uint32_t baseV = SIZE_256;
    uint32_t baseBTTail;  // 行方向尾块行数量
    uint32_t baseVTail;  // 列方向尾块列数量
    uint32_t mm0ResultSize;  // mm0结果大小
    uint32_t mm0ResultSizePerEpoch;  // 一轮mm0结果总大小
    uint32_t V1MainSize;  // V1单核单轮任务所需处理元素数量
    uint32_t mm1ResultSize;  // mm1结果大小
    uint32_t mm1ResultSizePerEpoch;  // 一轮mm1结果总大小
    uint32_t mm2ResultSize;  // mm2结果大小
    uint32_t mm2ResultSizePerEpoch;  // 一轮mm2结果总大小
    uint32_t V1InUBStride;  // V1搬入参数
    uint32_t V1OutUBStride;  // V1搬入参数
    uint32_t mmAccumRowOnetime;  // 一次处理几行mm累加。由于行长均为H，不区分mm1、2
    // mmAccumRowOnetime > 1时，以下3个参数无效
    uint32_t mmAccumCyclePerRow;  // 1行需要几套累加流水
    uint32_t mmAccumCntPerCycle;  // 每套处理多少元素
    uint32_t mmAccumCntLastCycle;  // 最后一套处理多少元素
    uint32_t V2MainSize;  // V2单核单轮任务所需处理元素数量
    uint32_t mmCastRowOnetime;  // 一次处理几行cast。由于行长均为H，不区分mm1、2
    // mmCastRowOnetime > 1时，以下3个参数无效
    uint32_t mmCastCyclePerRow;  // 1行需要几套cast流水
    uint32_t mmCastCntPerCycle;  // 每套处理多少元素
    uint32_t mmCastCntLastCycle;  // 最后一套处理多少元素
    uint32_t castMainSize;  // cast单核单轮任务所需处理元素数量
    uint32_t poolByteSize;  // pool大小
    uint32_t mm1CastRowPerVec;  // 每个vec至少要处理的cast行数
    uint32_t mm1CastExtraRowCnt;  // 额外需要处理的cast行数
    uint32_t mm2CastRowPerVec;
    uint32_t mm2CastExtraRowCnt;

    // 计算尾块大小
    baseBTTail = BT % baseBT;
    baseVTail = V % baseV;
    if (baseBTTail == 0U) {
        baseBTTail = baseBT;
    }
    if (baseVTail == 0U) {
        baseVTail = baseV;
    }
    
    // 空间大小计算
    mm0ResultSize = baseBT * baseV;
    mm0ResultSizePerEpoch = mm0ResultSize * aicNum;
    mm1ResultSize = baseBT * H;
    mm1ResultSizePerEpoch = mm1ResultSize * aicNum;
    mm2ResultSize = baseV * H;
    mm2ResultSizePerEpoch = mm2ResultSize * aicNum;
    // V1 UB
    V1MainSize = static_cast<uint32_t>(mm0ResultSize / SIZE_2);
    V1InUBStride = SIZE_32 - (baseVTail + SIZE_8 - 1) / SIZE_8;  // baseV=256，列尾块需要补的32B块数
    V1OutUBStride = SIZE_16 - (baseVTail + SIZE_16 - 1) / SIZE_16;  // 同上，是基本任务块cast后
    // V2 UB，按H大小区分情形
    uint32_t targetByteSize = ubByteSize / SIZE_32 / BUFFER_NUM / SIZE_2 * SIZE_32;  // 32B对齐、double buffer与两个加数空间预留
    CalcUBUsage(targetByteSize, H, mmAccumRowOnetime, mmAccumCyclePerRow, mmAccumCntPerCycle, mmAccumCntLastCycle);
    if (mmAccumRowOnetime >= SIZE_2) {
        V2MainSize = H * mmAccumRowOnetime;
    } else {
        V2MainSize = mmAccumCntPerCycle;
    }
    // Cast UB，同上，少用一块加数空间
    targetByteSize = ubByteSize / SIZE_32 / BUFFER_NUM * SIZE_32;  // 32B对齐、double buffer
    CalcUBUsage(targetByteSize, H, mmCastRowOnetime, mmCastCyclePerRow, mmCastCntPerCycle, mmCastCntLastCycle);
    if (mmCastRowOnetime >= SIZE_2) {
        castMainSize = H * mmCastRowOnetime;
    } else {
        castMainSize = mmCastCntPerCycle;
    }
    mm1CastRowPerVec = BT / aivNum;
    mm1CastExtraRowCnt = BT % aivNum;
    mm2CastRowPerVec = V / aivNum;
    mm2CastExtraRowCnt = V % aivNum;

    poolByteSize = std::max(targetByteSize * BUFFER_NUM, (V1MainSize + baseV) * IN_BYTE_SIZE);  // V1始终一次性搬运，没有db，且元素量为256倍数。V2采用了最大设计量，此处应由V2决定

    // cube相关
    matmul_tiling::DataType halfDtype = matmul_tiling::DataType::DT_FLOAT16;
    if (inputDtype == ge::DT_BF16) {
        halfDtype = matmul_tiling::DataType::DT_BFLOAT16;
    }
    uint32_t M;
    uint32_t N;
    uint32_t K;
    // mm0
    matmul_tiling::MultiCoreMatmulTiling mm0Tiling(ascendcPlatform);
    mm0Tiling.SetDim(aicNum); // 参与计算的cube核数
    mm0Tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, halfDtype);
    mm0Tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, halfDtype, true);
    mm0Tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm0Tiling.SetBias(false);
    M = std::min(baseBT, BT);
    N = std::min(baseV, V);
    K = H;
    mm0Tiling.SetShape(M, N, K);
    mm0Tiling.SetOrgShape(M, N, K);
    mm0Tiling.SetSingleShape(M, N, K);
    int64_t ret = mm0Tiling.GetTiling(tiling.mm0Tiling);
    if (ret == -1) {
        OP_LOGE(context->GetNodeName(), "GetTiling for FusedLinearCrossEntropyLossGrad mm0 failed.");
        return ge::GRAPH_FAILED;
    };

    // mm1
    matmul_tiling::MultiCoreMatmulTiling mm1Tiling(ascendcPlatform);
    mm1Tiling.SetDim(aicNum); // 参与计算的cube核数
    mm1Tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, halfDtype);
    mm1Tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, halfDtype);
    mm1Tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm1Tiling.SetBias(false);
    M = std::min(baseBT, BT);
    N = H;
    K = std::min(baseV, V);
    mm1Tiling.SetShape(M, N, K);
    mm1Tiling.SetOrgShape(M, N, K);
    mm1Tiling.SetSingleShape(M, N, K);
    ret = mm1Tiling.GetTiling(tiling.mm1Tiling);
    if (ret == -1) {
        OP_LOGE(context->GetNodeName(), "GetTiling for FusedLinearCrossEntropyLossGrad mm1 failed.");
        return ge::GRAPH_FAILED;
    };

    // mm2
    matmul_tiling::MultiCoreMatmulTiling mm2Tiling(ascendcPlatform);
    mm2Tiling.SetDim(aicNum); // 参与计算的cube核数
    mm2Tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, halfDtype, true);
    mm2Tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, halfDtype);
    mm2Tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm2Tiling.SetBias(false);
    M = std::min(baseV, V);
    N = H;
    K = std::min(baseBT, BT);
    mm2Tiling.SetShape(M, N, K);
    mm2Tiling.SetOrgShape(M, N, K);
    mm2Tiling.SetSingleShape(M, N, K);
    ret = mm2Tiling.GetTiling(tiling.mm2Tiling);
    if (ret == -1) {
        OP_LOGE(context->GetNodeName(), "GetTiling for FusedLinearCrossEntropyLossGrad mm2 failed.");
        return ge::GRAPH_FAILED;
    };

    // validation
    // h
    if (H > SIZE_65535) {
        OP_LOGE(context->GetNodeName(), "[FusedLinearCrossEntropyLossGrad] input.size(1) greater than 65535 is not supported now.");
        return ge::GRAPH_FAILED;
    };

    // tiling填充
    tiling.set_aicNum(aicNum);
    tiling.set_aivNum(aivNum);
    tiling.set_BT(BT);
    tiling.set_V(V);
    tiling.set_H(H);
    tiling.set_targetMaskSize(targetMaskSize);
    tiling.set_baseBT(baseBT);
    tiling.set_baseV(baseV);
    tiling.set_baseBTTail(baseBTTail);
    tiling.set_baseVTail(baseVTail);
    tiling.set_mm0ResultSize(mm0ResultSize);
    tiling.set_mm0ResultSizePerEpoch(mm0ResultSizePerEpoch);
    tiling.set_V1MainSize(V1MainSize);
    tiling.set_mm1ResultSize(mm1ResultSize);
    tiling.set_mm1ResultSizePerEpoch(mm1ResultSizePerEpoch);
    tiling.set_mm2ResultSize(mm2ResultSize);
    tiling.set_mm2ResultSizePerEpoch(mm2ResultSizePerEpoch);
    tiling.set_V1InUBStride(V1InUBStride);
    tiling.set_V1OutUBStride(V1OutUBStride);
    tiling.set_mmAccumRowOnetime(mmAccumRowOnetime);
    tiling.set_mmAccumCyclePerRow(mmAccumCyclePerRow);
    tiling.set_mmAccumCntPerCycle(mmAccumCntPerCycle);
    tiling.set_mmAccumCntLastCycle(mmAccumCntLastCycle);
    tiling.set_V2MainSize(V2MainSize);
    tiling.set_mmCastRowOnetime(mmCastRowOnetime);
    tiling.set_mmCastCyclePerRow(mmCastCyclePerRow);
    tiling.set_mmCastCntPerCycle(mmCastCntPerCycle);
    tiling.set_mmCastCntLastCycle(mmCastCntLastCycle);
    tiling.set_castMainSize(castMainSize);
    tiling.set_poolByteSize(poolByteSize);
    tiling.set_mm1CastRowPerVec(mm1CastRowPerVec);
    tiling.set_mm1CastExtraRowCnt(mm1CastExtraRowCnt);
    tiling.set_mm2CastRowPerVec(mm2CastRowPerVec);
    tiling.set_mm2CastExtraRowCnt(mm2CastExtraRowCnt);
    
    // tiling设定
    context->SetBlockDim(aicNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(MEM_FRIENDLY_KEY);
    // 设置workspace大小
    size_t userWorkspaceByteSize = 0;
    userWorkspaceByteSize += mm0ResultSizePerEpoch * BUFFER_NUM * IN_BYTE_SIZE;
    userWorkspaceByteSize += mm1ResultSizePerEpoch * BUFFER_NUM * IN_BYTE_SIZE;
    userWorkspaceByteSize += mm2ResultSizePerEpoch * BUFFER_NUM * IN_BYTE_SIZE;
    userWorkspaceByteSize += (uint64_t)BT * H * IN_BYTE_SIZE;
    userWorkspaceByteSize += (uint64_t)V * H * IN_BYTE_SIZE;
    size_t sysWorkspaceByteSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = userWorkspaceByteSize + sysWorkspaceByteSize;
    OP_LOGD(
        context->GetNodeName(),
        "[FusedLinearCrossEntropyLossGrad] userWorkspaceByteSize=%lld totalWorkspace=%lld.",
        userWorkspaceByteSize, currentWorkspace[0]
    );

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FusedLinearCrossEntropyLossGradTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Entering FusedLinearCrossEntropyLossGradTilingFunc.");
    auto softmax = context->GetOptionalInputTensor(INDEX_7);
    if (softmax == nullptr) {
        OP_LOGD(context->GetNodeName(), "Entering MemFriendlyTilingFunc.");
        return MemFriendlyTilingFunc(context);
    } else {
        OP_LOGD(context->GetNodeName(), "Entering HighPerfTilingFunc.");
        return HighPerfTilingFunc(context);
    }
}

static ge::graphStatus FusedLinearCrossEntropyLossGradTilingParseFunc([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FusedLinearCrossEntropyLossGrad)
    .Tiling(FusedLinearCrossEntropyLossGradTilingFunc)
    .TilingParse<FusedLinearCrossEntropyLossGradCompileInfo>(FusedLinearCrossEntropyLossGradTilingParseFunc);
}  // namespace optiling