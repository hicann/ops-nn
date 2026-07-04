/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mat_mul_optimized_fixpipe_algorithm.h
 * \brief Optimized fixpipe algorithm kernels with AIC+AIV cooperation.
 *        MatmulBaseUnalignedNKernel inherits from MatmulBaseFixPipeKernel for shared AIC processing
 *        and NZ2ND utilities. Supports both aligned output (AivProcess) and NZ2ND unaligned output
 *        (AivNz2NdProcess) via FIXPIPE_OPT template parameter.
 */
 #ifndef OP_KERNEL_MATMUL_V3_OPTIMIZED_FIXPIPE_ALGORITHM_H
 #define OP_KERNEL_MATMUL_V3_OPTIMIZED_FIXPIPE_ALGORITHM_H
 
 #include "mat_mul_base_fixpipe_kernel.h"
 #include "mat_mul_l1_full_load.h"
 
 using namespace AscendC;
 using namespace matmul;
 using namespace MatmulV3;
 
 #if defined(__CCE_KT_TEST__)
 using namespace std;
 #endif
 
 const uint32_t MM_ALIGN_SIZE = 512;
 
 template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = MatmulBaseBlock,
           const MatmulConfig& MM_CFG = MM_CFG_NO_PRELOAD, class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>,
           FIXPIPE_OPT_SELECT FIXPIPE_OPT = FIXPIPE_OPT_SELECT::BASE_ENABLE_ALIGNOUT>
 class MatmulBaseUnalignedNKernel : public MatmulBaseFixPipeKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE,
      BLOCK_TYPE, MM_CFG, MM_CB> {
 public:
     using C_T = typename C_TYPE::T;
     __aicore__ inline MatmulBaseUnalignedNKernel() {}
     __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                 GM_ADDR workspaceGM, const void* tilingData, TPipe* pipe);
     __aicore__ inline void AivProcess(GlobalTensor<C_T>& cTensor, uint8_t pingPongId);
     __aicore__ inline void Process(uint64_t index = 0UL, uint8_t enAtomic = 0UL);
     __aicore__ inline void AivNz2NdProcess(GlobalTensor<C_T> cNzGlobal, uint8_t pingPongId);
 
     // 供基类 RunFixpipeTileLoop 回调的每块 AIV 处理步骤：按 FIXPIPE_OPT 选择 NZ2ND 或对齐输出路径。
     __aicore__ inline void AivStep(GlobalTensor<C_T>& cTensor, uint8_t pingPongId)
     {
         if constexpr (FIXPIPE_OPT == FIXPIPE_OPT_SELECT::VEC_NZ2ND_UNALIGNOUT) {
             AivNz2NdProcess(cTensor, pingPongId);
         } else {
             AivProcess(cTensor, pingPongId);
         }
     }
 };
 
 template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
           class MM_CB, FIXPIPE_OPT_SELECT FIXPIPE_OPT>
 __aicore__ inline void
 MatmulBaseUnalignedNKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB, FIXPIPE_OPT>::Init(
     GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
     const void* tilingData, TPipe* pipe)
 {
     GetSizeC0<C_T>(this->c0Size_);
     uint64_t cDtypeSize = sizeof(C_T);
     this->block_.template Init<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(tilingData);
     this->InitInputs(aGM, bGM, cGM, biasGM);
 
     this->pipe_ = pipe;
     this->pipe_->InitBuffer(this->ubBuf_, TOTAL_UB_SIZE);
 
     int64_t originShapeM = 0;
     if constexpr (FIXPIPE_OPT == FIXPIPE_OPT_SELECT::VEC_NZ2ND_UNALIGNOUT) {
         this->alignedN_ = AlignUp(this->block_.matmulTilingData_->matmulTiling.N, BLOCK_SIZE);
         originShapeM = this->block_.matmulTilingData_->matmulTiling.baseM;
     } else {
         this->alignedN_ = AlignUp(this->block_.matmulTilingData_->matmulTiling.N, MM_ALIGN_SIZE / cDtypeSize);
         originShapeM = this->block_.matmulTilingData_->matmulTiling.M;
     }
     this->baseSize_ = this->alignedN_ * this->block_.matmulTilingData_->matmulTiling.baseM;
     this->params_.src0BlockStride = 1;
     this->params_.src0RepeatStride = this->alignedN_ / this->c0Size_;
     this->params_.src1RepeatStride = 0;
 
     this->tempCGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ C_T*>(workspaceGM),
                                  this->baseSize_ * NUM_TWO * this->block_.matmulTilingData_->matmulTiling.usedCoreNum);
     this->mm_.SetSubBlockIdx(0);
     this->mm_.Init(&this->block_.matmulTilingData_->matmulTiling, pipe);
     this->mm_.SetUserDefInfo(reinterpret_cast<uint64_t>(tilingData));
     this->mm_.SetOrgShape(originShapeM, this->block_.params_.alignedOriN,
                           this->block_.matmulTilingData_->matmulTiling.singleCoreK, this->block_.params_.alignedKbSize,
                           this->alignedN_);
     if (this->block_.params_.isHf32) {
         this->mm_.SetHF32(true, 1);
     } else {
         this->mm_.SetHF32(false, 0);
     }
 }
 
 template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
           class MM_CB, FIXPIPE_OPT_SELECT FIXPIPE_OPT>
 __aicore__ inline void
 MatmulBaseUnalignedNKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB, FIXPIPE_OPT>::AivProcess(
     GlobalTensor<C_T>& cTensor, uint8_t pingPongId)
 {
     if ASCEND_IS_AIV {
         uint32_t vBlockIndex = GetBlockIdx();
         if (vBlockIndex >= (this->block_.matmulTilingData_->matmulTiling.usedCoreNum * NUM_TWO)) {
             return;
         }
         CrossCoreWaitFlag(AIV_SYNC_AIC_FLAG + pingPongId);
 
         uint64_t cDtypeSize = sizeof(C_T);
         LocalTensor<C_T> ubTensor = this->ubBuf_.template Get<C_T>();
         uint64_t vecM = min(MMV3CeilAlign(this->block_.params_.singleCoreM / NUM_TWO, this->c0Size_),
                             static_cast<uint64_t>(this->block_.params_.singleCoreM));
         uint64_t subIdx = GetSubBlockIdx();
         uint64_t srcOffset = 0UL;
         uint64_t dstOffset = 0UL;
         if (subIdx == 1) {
             srcOffset = this->alignedN_ * vecM;
             dstOffset = vecM * this->block_.matmulTilingData_->matmulTiling.N;
             vecM = this->block_.params_.singleCoreM - vecM;
         }
         if (vecM == 0UL) {
             return;
         }
         uint64_t ubOffset = (pingPongId * TOTAL_UB_SIZE >> 1) / cDtypeSize;
         DataCopy<C_T>(ubTensor[ubOffset], cTensor[srcOffset], vecM * this->alignedN_);
         CrossCoreSetFlag<0x2, PIPE_MTE2>(AIC_SYNC_AIV_FLAG + pingPongId);
 
         SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(pingPongId));
         WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(pingPongId));
         this->params_.repeatTimes = vecM;
         uint64_t rsvdCnt = 0UL;
         GatherMask(ubTensor[ubOffset], ubTensor[ubOffset], 7, true, this->block_.matmulTilingData_->matmulTiling.N,
                    this->params_, rsvdCnt);
         SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(pingPongId));
         WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(pingPongId));
         DataCopy<C_T>(this->cGlobal_[this->block_.offset_.offsetC + dstOffset], ubTensor[ubOffset],
                       AlignUp(vecM * this->block_.matmulTilingData_->matmulTiling.N, this->c0Size_));
         SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(pingPongId));
         WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(pingPongId));
     }
 }
 
 template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
           class MM_CB, FIXPIPE_OPT_SELECT FIXPIPE_OPT>
 __aicore__ inline void
 MatmulBaseUnalignedNKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB, FIXPIPE_OPT>::AivNz2NdProcess(
     GlobalTensor<C_T> cNzGlobal, uint8_t pingPongId)
 {
     if ASCEND_IS_AIV {
         LocalTensor<C_T> tensorNZ;
         LocalTensor<C_T> tensorND;
         uint64_t ubProcessMNum = 0UL;
         uint64_t srcGmOffset = 0UL;
         uint64_t dstGmOffset = 0UL;
         if (!this->PrepareNz2NdConversion(pingPongId, tensorNZ, tensorND, ubProcessMNum, srcGmOffset, dstGmOffset)) {
             return;
         }
         if (ubProcessMNum == 0UL) {
             return;
         }
 
         WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(AIV_DB_SYNC_FLAG + pingPongId));
         this->Nz2NdCopyAndConvert(cNzGlobal, tensorNZ, tensorND, ubProcessMNum, srcGmOffset, pingPongId,
             this->block_.matmulTilingData_->matmulTiling.N);
 
         DataCopy(this->cGlobal_[this->block_.offset_.offsetC + dstGmOffset], tensorND,
                  AlignUp(ubProcessMNum * this->block_.matmulTilingData_->matmulTiling.N, this->c0Size_));
         SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(AIV_DB_SYNC_FLAG + pingPongId));
     }
 }
 
 template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
           class MM_CB, FIXPIPE_OPT_SELECT FIXPIPE_OPT>
 __aicore__ inline void
 MatmulBaseUnalignedNKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB, FIXPIPE_OPT>::Process(
     uint64_t index, uint8_t enAtomic)
 {
     ctx.isFirst = true;
     ctx.inputDtypeSize = sizeof(typename A_TYPE::T);
     this->template RunFixpipeTileLoop<
         MatmulBaseUnalignedNKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB, FIXPIPE_OPT>>(
         index, enAtomic);
 }
 
 template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = MatmulBaseBlock,
           const MatmulConfig& MM_CFG = MM_CFG_NO_PRELOAD, class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>,
           FIXPIPE_OPT_SELECT FIXPIPE_OPT = FIXPIPE_OPT_SELECT::BASE_ENABLE_ALIGNOUT>
 class MatmulBaseAToNZWithBL1FixpipeKernel
     : public MatmulBaseUnalignedNKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB, FIXPIPE_OPT> {
     struct BaseUnAlignedNKernelParams {
         GM_ADDR aGMNZ;
         GM_ADDR workspaceGMNZ;
         uint64_t baseAN;
         uint64_t baseAD;
     };
 
 public:
     __aicore__ inline MatmulBaseAToNZWithBL1FixpipeKernel() {}
 
     __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                 GM_ADDR workspaceGM, const MatmulTilingData* tilingData, TPipe* pipe);
 
     __aicore__ inline void Process(uint64_t index = 0, uint8_t enAtomic = 0);
 
 protected:
     using C_T = typename C_TYPE::T;
     BaseUnAlignedNKernelParams fixpipeInnerParams_;
 };
 
 template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
           class MM_CB, FIXPIPE_OPT_SELECT FIXPIPE_OPT>
 __aicore__ inline void
 MatmulBaseAToNZWithBL1FixpipeKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB, FIXPIPE_OPT>::Init(
     GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR workspaceGM,
     const MatmulTilingData* matmulTilingData, TPipe* pipe)
 {
     GetSizeC0<C_T>(this->c0Size_);
     uint64_t cDtypeSize = sizeof(C_T);
     uint64_t alignedN = AlignUp(matmulTilingData->matmulTiling.N, MM_ALIGN_SIZE / cDtypeSize);
     uint64_t baseSize = alignedN * matmulTilingData->matmulTiling.baseM;
 
     MatmulBaseUnalignedNKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB, FIXPIPE_OPT>::Init(
         workspaceGM + baseSize * NUM_TWO * matmulTilingData->matmulTiling.usedCoreNum * cDtypeSize, bGM, cGM, biasGM,
         offsetWGM, workspaceGM, matmulTilingData, pipe);
     this->fixpipeInnerParams_.baseAN = this->block_.matmulTilingData_->baseAN;
     this->fixpipeInnerParams_.baseAD = this->block_.matmulTilingData_->baseAD;
     this->fixpipeInnerParams_.aGMNZ = aGM;
     this->fixpipeInnerParams_.workspaceGMNZ =
         workspaceGM + this->baseSize_ * NUM_TWO * this->block_.matmulTilingData_->matmulTiling.usedCoreNum * cDtypeSize;
     this->mm_.SetOrgShape(this->block_.params_.alignedOriM, this->block_.matmulTilingData_->matmulTiling.N,
                           this->block_.params_.alignedKaSize, this->block_.matmulTilingData_->matmulTiling.Kb,
                           this->alignedN_);
 }
 
 template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig& MM_CFG,
           class MM_CB, FIXPIPE_OPT_SELECT FIXPIPE_OPT>
 __aicore__ inline void
 MatmulBaseAToNZWithBL1FixpipeKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB, FIXPIPE_OPT>::Process(
     uint64_t index, uint8_t enAtomic)
 {
     if ASCEND_IS_AIV {
         MatrixAtoNZV2<typename A_TYPE::T>(this->fixpipeInnerParams_.workspaceGMNZ, this->fixpipeInnerParams_.aGMNZ,
                                           this->block_.matmulTilingData_->matmulTiling, A_TYPE::isTrans, this->ubBuf_,
                                           this->fixpipeInnerParams_.baseAN, this->fixpipeInnerParams_.baseAD);
         SyncAll();
         CrossCoreSetFlag<0x2, PIPE_MTE3>(CV_FLAG);
     }
     if ASCEND_IS_AIC {
         CrossCoreWaitFlag(CV_FLAG);
     }
     MatmulBaseUnalignedNKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB, FIXPIPE_OPT>::Process(
         index, enAtomic);
 }
 
 #endif
 