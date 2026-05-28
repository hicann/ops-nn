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
 * \file pp_matmul_default.cc
 * \brief
 */
#include "pp_matmul_default.h"
#include "pp_matmul_common_tiling.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_tiling.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "error_util.h"
#include "platform/platform_infos_def.h"

// using namespace optiling::pp_matmul; // todo：使用别的地方定义在pp_matmul的东西

namespace {
constexpr uint64_t L1_DESCALE_BUFFER_SIZE_MAX = 6144;
constexpr uint64_t CONST_3 = 3;
constexpr uint64_t CONST_4 = 4;
constexpr uint64_t CONST_16 = 16;
constexpr uint64_t CONST_32 = 32;
constexpr uint64_t CONST_256 = 256;
constexpr uint64_t CONST_512 = 512;

constexpr size_t DIM_2 = 2;
constexpr size_t DIM_3 = 3;
constexpr size_t DIM_4 = 4;
constexpr size_t COMPUTE_TYPE_IDX_CANN = 3;
constexpr size_t COMPUTE_TYPE_IDX_ATB = 10;
constexpr size_t EINSUM_MODE = 4;

constexpr uint32_t DTYPE_BIT_COUNT = 2;
constexpr uint32_t FORMAT_BIT_COUNT = 1;
constexpr uint32_t COMPUTE_TYPE_BIT_COUNT = 3;
constexpr uint32_t MAX_ATTRS_NUM = 4;
constexpr uint32_t EN_SHUFFFLE_IDX_ATB = 7;

constexpr uint32_t ALIGNMENT_16 = 16;
constexpr uint64_t BLOCK_SIZE_INT8_K = 32;
constexpr uint32_t ALIGNMENT_256 = 256;
constexpr uint32_t L0AB_PINGPONG_BUFFER_LEN_FP16 = 131072;
constexpr uint64_t L1AB_PINGPONG_BUFFER_LEN_INT8_SPARSE = 160*1024;
constexpr uint32_t TRANS_B_MASK = 0b001000;
}

namespace optiling {
namespace pp_matmul {

void PpMatmulDefaultTilingData::SetBaseShape(uint64_t batchSize, uint64_t m, uint64_t k, uint64_t n) {
    opShape.batchSize = batchSize;
    opShape.m = m;
    opShape.k = k;
    opShape.n = n;
}

void PpMatmulDefaultTilingData::SetBaseOp(uint64_t coreNum, uint64_t l0cSize, uint64_t mBase, uint64_t nBase, const MatMulInfo &mmInfo, bool isNpuArch2002) {
    opShape.m0 = mBase;
    opShape.n0 = nBase;
    mLoop = CeilDiv(opShape.m, opShape.m0);
    nLoop = CeilDiv(opShape.n, opShape.n0);
    coreLoop = opShape.batchSize * mLoop * nLoop;
    if (mmInfo.isQuantBatchMatmulV3) {
        bool transB = tilingKey & TRANS_B_MASK;
        if (mLoop == 1 && (tilingK == 0) && transB && coreLoop % coreNum < coreNum / CONST_4 * CONST_3) {
            uint32_t x = CeilDiv(opShape.n, coreNum);
            uint32_t y = CeilDiv(x, CONST_256);
            nBase = RoundUp(CeilDiv(x, y), CONST_16);
            if (mBase * nBase * sizeof(float) < l0cSize) {
                opShape.n0 = nBase;
                nLoop = CeilDiv(opShape.n, opShape.n0);
                coreLoop = opShape.batchSize * nLoop;
            }
        }
    }
    if (!isNpuArch2002 && mLoop == 1UL && mmInfo.transB && static_cast<uint64_t>(coreLoop % coreNum) <
        static_cast<uint64_t>(coreNum / CONST_4) * CONST_3) {
        mBase = RoundUp(opShape.m, CONST_16);
        opShape.m0 = mBase;
        uint64_t maxN0 = l0cSize / (mBase * sizeof(float));
        uint64_t x = CeilDiv(opShape.n, coreNum);
        uint64_t y = CeilDiv(x, maxN0);
        nBase = RoundUp(CeilDiv(x, y), CONST_16);
        uint64_t rqdL0CSize = mBase * nBase * sizeof(float);
        if (rqdL0CSize < l0cSize &&
            (mBase + nBase) * CONST_256 * sizeof(uint16_t) < L1AB_PINGPONG_BUFFER_SIZE) {
            opShape.n0 = nBase;
            nLoop = CeilDiv(opShape.n, opShape.n0);
            coreLoop = opShape.batchSize * nLoop;
        }
    }
    blockDim = std::min(coreLoop, coreNum);
}

void PpMatmulDefaultTilingData::End(const MatMulInfo &mmInfo, bool isNpuArch2002) {
    uint64_t shapeSum = opShape.m0 + opShape.n0;
    if (isNpuArch2002) {
        uint32_t l1AbPpBuffLen = L0AB_PINGPONG_BUFFER_LEN_FP16;
        if (mmInfo.isCompress) {
            l1AbPpBuffLen = L1AB_PINGPONG_BUFFER_LEN_INT8_SPARSE;
            uint32_t compressNTile = CeilDiv((opShape.n % opShape.n0), CONST_16) % mmInfo.tilingN;
            compressOverlapN = compressNTile == 0 ? 0 : mmInfo.tilingN - compressNTile;
            tilingK = mmInfo.tilingK;
            tilingN = mmInfo.tilingN;
        }
        uint32_t k0Max = (shapeSum == 0) ? l1AbPpBuffLen : (l1AbPpBuffLen / shapeSum);
        uint32_t k0Init = mmInfo.isCompress ? mmInfo.tilingK * BLOCK_SIZE_INT8_K : ALIGNMENT_256;
        opShape.k0 = k0Max < k0Init ? k0Max / ALIGNMENT_16 * ALIGNMENT_16 : k0Max / k0Init * k0Init;
    }
    if (!isNpuArch2002) {
        uint64_t cubeBlockSize = mmInfo.isInt8 ? CUBE_BLOCK_SIZE_INT8 : CUBE_BLOCK_SIZE;
        uint64_t kBlockSize = mmInfo.isInt8 ? BLOCK_SIZE_INT8_K : BLOCK_SIZE;
        uint64_t scaleBlockSize = mmInfo.isInt8 ? L1_DESCALE_BUFFER_SIZE_MAX : 0UL;
        if (mmInfo.isInt8 && (mmInfo.transA || !mmInfo.transB)) {
            shapeSum = RoundUp(opShape.m0, CONST_32) + RoundUp(opShape.n0, CONST_32);
        }
        uint64_t k0Max = shapeSum == 0UL
                        ? L1AB_PINGPONG_BUFFER_SIZE
                        : static_cast<uint64_t>(static_cast<float>(L1AB_PINGPONG_BUFFER_SIZE - scaleBlockSize)
                            / (shapeSum * mmInfo.sizeInDtype));
        opShape.k0 = k0Max < cubeBlockSize ? RoundDown(k0Max, kBlockSize) : RoundDown(k0Max, cubeBlockSize);
        if (opShape.k0 > CONST_512) {
            opShape.k0 = RoundDown(opShape.k0, CONST_512);
        }
    }
    kLoop = CeilDiv(opShape.k, opShape.k0);
}


void PpMatMulDefault::GetHardwareInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE("[PpMatMul]", "platformInfo is nullptr");
        return;
    }
    HardwareInfo hardwareInfo;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    platformInfo->GetPlatformRes("version", "SoC_version", hardwareInfo.socVersionStr);

    hardwareInfo.coreNum = static_cast<uint64_t>(ascendcPlatform.GetCoreNumAic());
    hardwareInfo.socVersion = ascendcPlatform.GetSocVersion();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, hardwareInfo.l2Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, hardwareInfo.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, hardwareInfo.l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, hardwareInfo.l0bSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, hardwareInfo.l0cSize);
    hardwareInfo_ = hardwareInfo;
}


bool PpMatMulDefault::GetMatMulTilingData()
{
    ppMatmulDefaultTilingData_.SetBaseShape(matMulInfo_.batchSize, matMulInfo_.m, matMulInfo_.k, matMulInfo_.n);
    OpShape opShape = ppMatmulDefaultTilingData_.opShape;
    if (opShape.m < opShape.n) {
        TilingFunc<false, OpShape, PpMatmulDefaultTilingData, HardwareInfo, MatMulInfo>(opShape, ppMatmulDefaultTilingData_, hardwareInfo_, matMulInfo_, matMulInfo_.isCompress, matMulInfo_.tilingN);
    } else {
        TilingFunc<true, OpShape, PpMatmulDefaultTilingData, HardwareInfo, MatMulInfo>(opShape, ppMatmulDefaultTilingData_, hardwareInfo_, matMulInfo_, matMulInfo_.isCompress, matMulInfo_.tilingN);
    }
    Swizzle<PpMatmulDefaultTilingData>(ppMatmulDefaultTilingData_);
    ppMatmulDefaultTilingData_.End(matMulInfo_, hardwareInfo_.socVersion == platform_ascendc::SocVersion::ASCEND310P);
    return true;
}


void PpMatMulDefault::PrintTiling() {
    OP_LOGD(context_->GetNodeName(), "PpMatMul batchSize: %ld.", ppMatmulDefaultTilingData_.opShape.batchSize);
    OP_LOGD(context_->GetNodeName(), "PpMatMul m: %ld.", ppMatmulDefaultTilingData_.opShape.m);
    OP_LOGD(context_->GetNodeName(), "PpMatMul k: %ld.", ppMatmulDefaultTilingData_.opShape.k);
    OP_LOGD(context_->GetNodeName(), "PpMatMul n: %ld.", ppMatmulDefaultTilingData_.opShape.n);
    OP_LOGD(context_->GetNodeName(), "PpMatMul m0: %ld.", ppMatmulDefaultTilingData_.opShape.m0);
    OP_LOGD(context_->GetNodeName(), "PpMatMul k0: %ld.", ppMatmulDefaultTilingData_.opShape.k0);
    OP_LOGD(context_->GetNodeName(), "PpMatMul n0: %ld.", ppMatmulDefaultTilingData_.opShape.n0);
    OP_LOGD(context_->GetNodeName(), "PpMatMul mLoop: %ld.", ppMatmulDefaultTilingData_.mLoop);
    OP_LOGD(context_->GetNodeName(), "PpMatMul kLoop: %ld.", ppMatmulDefaultTilingData_.kLoop);
    OP_LOGD(context_->GetNodeName(), "PpMatMul nLoop: %ld.", ppMatmulDefaultTilingData_.nLoop);
    OP_LOGD(context_->GetNodeName(), "PpMatMul coreLoop: %ld.", ppMatmulDefaultTilingData_.coreLoop);
    OP_LOGD(context_->GetNodeName(), "PpMatMul swizzleCount: %ld.", ppMatmulDefaultTilingData_.swizzleCount);
    OP_LOGD(context_->GetNodeName(), "PpMatMul tilingKey: %d.", ppMatmulDefaultTilingData_.tilingKey);
    OP_LOGD(context_->GetNodeName(), "PpMatMul blockDim: %ld.", ppMatmulDefaultTilingData_.blockDim);
    OP_LOGD(context_->GetNodeName(), "PpMatMul swizzleDirect: %ld.", ppMatmulDefaultTilingData_.swizzleDirect);
    OP_LOGD(context_->GetNodeName(), "PpMatMul splitk: %ld.", ppMatmulDefaultTilingData_.splitk);
    OP_LOGD(context_->GetNodeName(), "PpMatMul enShuffleK: %ld.", ppMatmulDefaultTilingData_.enShuffleK);
}


} // namespace pp_matmul
} // namespace optiling