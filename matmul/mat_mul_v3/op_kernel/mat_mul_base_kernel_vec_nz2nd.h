/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mat_mul_base_vector_nz2nd_kernel.h
 * \brief Matmul kernel with AIV-based NZ2ND conversion for half/bfloat16 output
 */
 #ifndef __OP_KERNEL_MATMUL_V3_BASE_VECTOR_NZ2ND_KERNEL_H__
 #define __OP_KERNEL_MATMUL_V3_BASE_VECTOR_NZ2ND_KERNEL_H__
 
 #include "mat_mul_base_fixpipe_kernel.h"
 
 using namespace AscendC;
 using namespace matmul;
 
 template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE = MatmulBaseBlock,
     const MatmulConfig &MM_CFG = MM_CFG_NO_PRELOAD, class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>>
 class MatmulBaseVectorNz2NdKernel
     : public MatmulBaseFixPipeKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB> {
 public:
     using C_T = typename C_TYPE::T;
     __aicore__ inline MatmulBaseVectorNz2NdKernel() {}
 
     __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
         GM_ADDR workspaceGM, const void *tilingData, TPipe *pipe);
 
     __aicore__ inline void AivNz2NdProcess(GlobalTensor<C_T> cNzGlobal, uint8_t pingPongId);
 
     // 读 A 与写 C 的 orgM 取值不同，需在 Iterate 与 GetTensorC 之间切换 orgM。
     __aicore__ inline void AicProcess(GlobalTensor<C_T>& cTensor, uint8_t enAtomic, bool aicNeedWaitAiv,
         uint8_t pingPongId);
 
     // 供基类 RunFixpipeTileLoop 回调的每块 AIV 处理步骤。
     __aicore__ inline void AivStep(GlobalTensor<C_T>& cTensor, uint8_t pingPongId)
     {
         AivNz2NdProcess(cTensor, pingPongId);
     }
 
     __aicore__ inline void UpdateGlobalTensor(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
         GM_ADDR workspaceGM);
 
     __aicore__ inline void Process(uint64_t index = 0, uint8_t enAtomic = 0);
 };
 
 template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
     class MM_CB>
 __aicore__ inline void MatmulBaseVectorNz2NdKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::Init(
     GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
     GM_ADDR workspaceGM, const void *tilingData, TPipe *pipe)
 {
     GetSizeC0<C_T>(this->c0Size_);
     this->block_.template Init<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE>(tilingData);
     this->pipe_ = pipe;
     this->InitInputs(aGM, bGM, cGM, biasGM);
 
     this->pipe_->InitBuffer(this->ubBuf_, TOTAL_UB_SIZE);
 
     int64_t originShapeM = this->block_.matmulTilingData_->matmulTiling.baseM;
 
     this->baseSize_ = this->block_.matmulTilingData_->matmulTiling.baseN *
         this->block_.matmulTilingData_->matmulTiling.baseM;
 
     this->tempCGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ C_T*>(workspaceGM),
                                 this->baseSize_ * NUM_TWO * this->block_.matmulTilingData_->matmulTiling.usedCoreNum);
     this->mm_.SetSubBlockIdx(0);
     this->mm_.Init(&this->block_.matmulTilingData_->matmulTiling, pipe);
     this->mm_.SetUserDefInfo(reinterpret_cast<uint64_t>(tilingData));
 
     this->mm_.SetOrgShape(this->block_.params_.alignedOriM, this->block_.params_.alignedOriN,
         this->block_.params_.alignedKaSize, this->block_.params_.alignedKbSize,
         AlignUp(this->block_.matmulTilingData_->matmulTiling.N, BLOCK_SIZE));
 
     if (this->block_.params_.isHf32) {
         this->mm_.SetHF32(true, 1);
     } else {
         this->mm_.SetHF32(false, 0);
     }
 }
 
 
 template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
     class MM_CB>
 __aicore__ inline void MatmulBaseVectorNz2NdKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::AicProcess(
     GlobalTensor<C_T>& cTensor, uint8_t enAtomic, bool aicNeedWaitAiv, uint8_t pingPongId)
 {
     if ASCEND_IS_AIC {
         // 读 A 阶段：A 为整块 FRACTAL_NZ，沿 K 方向跨步使用 orgM，需用 alignedOriM
         this->mm_.SetOrgShape(this->block_.params_.alignedOriM, this->block_.params_.alignedOriN,
             this->block_.params_.alignedKaSize, this->block_.params_.alignedKbSize,
             AlignUp(this->block_.matmulTilingData_->matmulTiling.N, BLOCK_SIZE));
         this->AicLoadAndIterate(aicNeedWaitAiv, pingPongId);
         // 写 C 阶段：C 写入每个 base 块独立的 NZ workspace，N 分形跨步使用 orgM，需用 baseM
         this->mm_.SetOrgShape(this->block_.matmulTilingData_->matmulTiling.baseM, this->block_.params_.alignedOriN,
             this->block_.params_.alignedKaSize, this->block_.params_.alignedKbSize,
             AlignUp(this->block_.params_.singleCoreN, BLOCK_SIZE));
         this->AicStoreC(cTensor, enAtomic, pingPongId);
     }
 }
 
 
 template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
     class MM_CB>
 __aicore__ inline void MatmulBaseVectorNz2NdKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::AivNz2NdProcess(
     GlobalTensor<C_T> cNzGlobal, uint8_t pingPongId)
 {
     if ASCEND_IS_AIV {
         uint64_t nBlocks = this->block_.params_.singleCoreN;
         this->alignedN_ = AlignUp(nBlocks, BLOCK_SIZE);
 
         LocalTensor<C_T> tensorNZ;
         LocalTensor<C_T> tensorND;
         uint64_t ubProcessMNum = 0UL;
         uint64_t srcGmOffset = 0UL;
         uint64_t dstGmOffset = 0UL;
         if (!this->PrepareNz2NdConversion(pingPongId, tensorNZ, tensorND, ubProcessMNum, srcGmOffset, dstGmOffset)) {
             return;
         }
 
         if (ubProcessMNum == 0UL) {
             //补充同步信号
             WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(AIV_DB_SYNC_FLAG + pingPongId));
             CrossCoreSetFlag<0x2, PIPE_MTE2>(AIC_SYNC_AIV_FLAG + pingPongId);
             SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(AIV_DB_SYNC_FLAG + pingPongId));
             return;
         }
 
         WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(AIV_DB_SYNC_FLAG + pingPongId));
         if (std::is_same_v<C_T, bfloat16_t> || std::is_same_v<C_T, half>) {
             this->Nz2NdCopyAndConvert(cNzGlobal, tensorNZ, tensorND, ubProcessMNum, srcGmOffset, pingPongId,
                 this->alignedN_);
 
             DataCopyExtParams dataCopyExtParams;
             dataCopyExtParams.blockCount = ubProcessMNum;
             dataCopyExtParams.blockLen = static_cast<uint32_t>(nBlocks * sizeof(C_T));
             dataCopyExtParams.srcStride = 0;
             dataCopyExtParams.dstStride = static_cast<uint32_t>((this->block_.matmulTilingData_->matmulTiling.N - nBlocks) * sizeof(C_T));
             dataCopyExtParams.rsv = 0;
             DataCopyPad(this->cGlobal_[this->block_.offset_.offsetC + dstGmOffset], tensorND, dataCopyExtParams);
         } else {
             CrossCoreSetFlag<0x2, PIPE_MTE2>(AIC_SYNC_AIV_FLAG + pingPongId);
         }
         SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(AIV_DB_SYNC_FLAG + pingPongId));
     }
 }
 
 
 template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
     class MM_CB>
 __aicore__ inline void MatmulBaseVectorNz2NdKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG,
     MM_CB>::UpdateGlobalTensor(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
     GM_ADDR workspaceGM)
 {
     this->InitInputs(aGM, bGM, cGM, biasGM);
 }
 
 
 template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class BLOCK_TYPE, const MatmulConfig &MM_CFG,
     class MM_CB>
 __aicore__ inline void MatmulBaseVectorNz2NdKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>::Process(
     uint64_t index, uint8_t enAtomic)
 {
     this->template RunFixpipeTileLoop<
         MatmulBaseVectorNz2NdKernel<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, BLOCK_TYPE, MM_CFG, MM_CB>>(index, enAtomic);
 }
 #endif // __OP_KERNEL_MATMUL_V3_BASE_VECTOR_NZ2ND_KERNEL_H__
 