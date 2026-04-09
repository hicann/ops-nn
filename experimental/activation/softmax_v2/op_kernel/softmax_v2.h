/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):

 * - Tu Yuanhang <@TuYHAAAAAA>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file softmax_v2.h
 * \brief
*/
#ifndef SOFTMAXV2_H
#define SOFTMAXV2_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "softmax_v2_tiling_data.h"
#include "softmax_v2_tiling_key.h"

namespace NsSoftmaxV2 {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class SoftmaxV2 {
public:
    __aicore__ inline SoftmaxV2(){};

    __aicore__ inline void Init(GM_ADDR x,  GM_ADDR z, GM_ADDR workspace,const SoftmaxV2TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress,int32_t row,int32_t circle);
    __aicore__ inline void CopyOut(int32_t progress,int32_t row,int32_t circle);
    __aicore__ inline void Compute(int32_t progress,int32_t row,int32_t circle);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueS,inQueueM;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ,outQueueS,outQueueM;
    AscendC::TBuf<QuePosition::VECCALC> temp0,temp1;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> sGm;
    AscendC::GlobalTensor<T> zGm;
    AscendC::GlobalTensor<T> mGm;
    AscendC::DataCopyPadExtParams<T> padParams;
    AscendC::DataCopyPadExtParams<T> padParams0;
    AscendC::DataCopyExtParams copyParams;
    AscendC::DataCopyExtParams copyParams0;
    AscendC::DataCopyExtParams copyParams1;
    AscendC::DataCopyExtParams copyParams2;
    AscendC::DataCopyExtParams copyParams3;

    uint32_t rowsss=0;
    float sum=0;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tailNum;
    uint32_t processNum;
    uint32_t tileLength;
    uint32_t dim;
    uint32_t lastdimnum;
    uint32_t dimNum;
    int32_t  dimarr[4];   
    SoftmaxV2TilingData tiling;
};

template <typename T>
__aicore__ inline void SoftmaxV2<T>::Init(GM_ADDR x,  GM_ADDR z,GM_ADDR workspace, const SoftmaxV2TilingData* tilingData)
{
        this->tiling = *tilingData;
        dim=tiling.dim;
        dimNum = tiling.dimNum;
        int32_t* dimarr = tiling.dimarr;  
        lastdimnum = dimarr[dimNum-1];
        if(AscendC::GetBlockIdx()<tiling.big_core_num){
            this->blockLength = tiling.big_tile_length;
            this->tileNum = tiling.big_tile_times;
            this->tailNum = tiling.big_tail_num;
            rowsss = this->blockLength * AscendC::GetBlockIdx();
        }else{
            this->blockLength = tiling.small_tile_length;
            this->tileNum = tiling.small_tile_times;
            this->tailNum = tiling.small_tail_num;
            rowsss = this->blockLength * AscendC::GetBlockIdx() + tiling.big_core_num;
        }
        if(AscendC::GetBlockIdx()<tiling.big_core_num){
            xGm.SetGlobalBuffer((__gm__ T *)x + this->blockLength * AscendC::GetBlockIdx()*lastdimnum, this->blockLength*lastdimnum);
            sGm.SetGlobalBuffer((__gm__ T *)workspace , tiling.sumspace);
            mGm.SetGlobalBuffer((__gm__ T *)workspace +tiling.sumspace , tiling.sumspace);
            zGm.SetGlobalBuffer((__gm__ T *)z + this->blockLength * AscendC::GetBlockIdx()*lastdimnum, this->blockLength*lastdimnum);
        }else{
            xGm.SetGlobalBuffer((__gm__ T *)x + (this->blockLength * AscendC::GetBlockIdx()+tiling.big_core_num)*lastdimnum, this->blockLength *lastdimnum);
            sGm.SetGlobalBuffer((__gm__ T *)workspace , tiling.sumspace);
            mGm.SetGlobalBuffer((__gm__ T *)workspace +tiling.sumspace , tiling.sumspace);
            zGm.SetGlobalBuffer((__gm__ T *)z + (this->blockLength * AscendC::GetBlockIdx()+tiling.big_core_num)*lastdimnum, this->blockLength *lastdimnum);
        }

        pipe.InitBuffer(inQueueX, BUFFER_NUM, lastdimnum* sizeof(T));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, lastdimnum* sizeof(T));
        pipe.InitBuffer(inQueueS, BUFFER_NUM, lastdimnum * sizeof(T));//往ub搬
        pipe.InitBuffer(outQueueS, BUFFER_NUM, lastdimnum * sizeof(T));//往gm搬
        pipe.InitBuffer(inQueueM, BUFFER_NUM, lastdimnum * sizeof(T));//往ub搬
        pipe.InitBuffer(outQueueM, BUFFER_NUM, lastdimnum * sizeof(T));//往gm搬
        pipe.InitBuffer(temp0, lastdimnum * sizeof(T));
        pipe.InitBuffer(temp1, lastdimnum * sizeof(T));
    }

template <typename T>
 __aicore__ inline void SoftmaxV2<T>::CopyIn(int32_t progress,int32_t row,int32_t circle)//
    {        
        if(circle==0||circle==1){
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        AscendC::DataCopyExtParams copyParams{1,  static_cast<uint32_t>(lastdimnum * sizeof(T)), 0, 0, 0}; 
        AscendC::DataCopyPadExtParams<T> padParams{true, 0, 0, 0};//
        AscendC::DataCopyPad(xLocal, xGm[(progress * tiling.core_tile_x1 + row) * lastdimnum], copyParams,padParams); 
        inQueueX.EnQue(xLocal);
            if(dim == dimNum-1){//求这一行的对应的行数
                int hangshu =  progress* tiling.core_tile_x1 + row  + rowsss;
                AscendC::LocalTensor<T> mLocalin = inQueueM.AllocTensor<T>();
                AscendC::DataCopyExtParams copyParams{1,  static_cast<uint32_t>(sizeof(T)), 0, 0, 0}; 
                AscendC::DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
                AscendC::DataCopyPad(mLocalin, mGm[hangshu], copyParams,padParams); 
                inQueueM.EnQue(mLocalin);
            }else{  //不是最后一个维度就搬入一行
                //分到每一个的行数
                int hangshu=  rowsss + progress*tiling.core_tile_x1 +row;
                int parts= tiling.rows / tiling.partnum;//一个部分最多的行数 
                int inparts= tiling.sumspacerows / tiling.partnum;//一个部分的行数 
                int elepartnum =  hangshu / parts;//部分偏移
                int eleinparts =  hangshu % inparts;//部内偏移
                //要从搬到这个mLocalin里面
                AscendC::LocalTensor<T> mLocalin = inQueueM.AllocTensor<T>();
                AscendC::DataCopyExtParams copyParams{1,  static_cast<uint32_t>(lastdimnum*sizeof(T)), 0, 0, 0}; 
                AscendC::DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
                AscendC::DataCopyPad(mLocalin, mGm[elepartnum * lastdimnum *inparts  + lastdimnum*eleinparts], copyParams,padParams); 
                inQueueM.EnQue(mLocalin);
            }
        }else{
            AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
            //得搬相对行数
            AscendC::DataCopyExtParams copyParams{1,  static_cast<uint32_t>(lastdimnum * sizeof(T)), 0, 0, 0}; 
            AscendC::DataCopyPadExtParams<T> padParams{true, 0, 0, 0};//
            AscendC::DataCopyPad(xLocal, xGm[(progress * tiling.core_tile_x1 + row) * lastdimnum], copyParams,padParams);
            inQueueX.EnQue(xLocal);
            if(dim == dimNum-1){//求这一行的对应的行数
                int hangshu =  progress* tiling.core_tile_x1 + row  + rowsss;
                AscendC::LocalTensor<T> sLocalin = inQueueS.AllocTensor<T>();
                AscendC::DataCopyExtParams copyParams{1,  static_cast<uint32_t>(sizeof(T)), 0, 0, 0}; 
                AscendC::DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
                AscendC::DataCopyPad(sLocalin, sGm[hangshu], copyParams,padParams); 
                AscendC::LocalTensor<T> mLocalin = inQueueM.AllocTensor<T>();
                AscendC::DataCopyPad(mLocalin, mGm[hangshu], copyParams,padParams); 
                inQueueM.EnQue(mLocalin);
                inQueueS.EnQue(sLocalin);
            }else{  //不是最后一个维度就搬入一行
                //分到每一个的行数
                int hangshu=  rowsss + progress*tiling.core_tile_x1 +row;
                int parts= tiling.rows / tiling.partnum;//一个部分最多的行数
                int inparts= tiling.sumspacerows / tiling.partnum;//一个部分的行数 
                int elepartnum =  hangshu / parts;//部分偏移
                int eleinparts =  hangshu % inparts;//部内偏移
                //要从搬到这个sLocalin里面
                AscendC::LocalTensor<T> sLocalin = inQueueS.AllocTensor<T>();
                AscendC::DataCopyExtParams copyParams{1,  static_cast<uint32_t>(lastdimnum*sizeof(T)), 0, 0, 0}; 
                AscendC::DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
                AscendC::DataCopyPad(sLocalin, sGm[elepartnum * lastdimnum *inparts  + lastdimnum*eleinparts], copyParams,padParams); 
                AscendC::LocalTensor<T> mLocalin = inQueueM.AllocTensor<T>();
                AscendC::DataCopyPad(mLocalin, mGm[elepartnum * lastdimnum *inparts  + lastdimnum*eleinparts], copyParams,padParams); 
                inQueueM.EnQue(mLocalin);
                inQueueS.EnQue(sLocalin);
            }
        }
    }

template <typename T>
  __aicore__ inline void SoftmaxV2<T>::CopyOut(int32_t progress,int32_t row,int32_t circle)
    {   
        if(circle==0){
            if(dim == dimNum-1){
            AscendC::LocalTensor<T> mLocalout = outQueueM.DeQue<T>();
            AscendC::DataCopyExtParams copyParams2{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
            AscendC::DataCopyPad(mGm[progress * tiling.core_tile_x1 + row + rowsss], mLocalout, copyParams2); 
            outQueueM.FreeTensor(mLocalout);
            }else{
                //分到每一个的行数
                int hangshu=  rowsss + progress*tiling.core_tile_x1 +row;
                int parts= tiling.rows / tiling.partnum;//一个部分最多的行数
                int inparts= tiling.sumspacerows / tiling.partnum;//一个部分的行数 
                int elepartnum =  hangshu / parts;//部分偏移
                int eleinparts =  hangshu % inparts;//部内偏移
                AscendC::LocalTensor<T> mLocalout = outQueueM.DeQue<T>();
                AscendC::DataCopyExtParams copyParams2{static_cast<uint16_t>(1),  static_cast<uint32_t>(lastdimnum*sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
                AscendC::DataCopyPad(mGm[elepartnum * lastdimnum *inparts  +  lastdimnum * eleinparts], mLocalout,copyParams2); 
                outQueueM.FreeTensor(mLocalout);
            }
        }else if(circle==1){
            if(dim == dimNum-1){
                AscendC::LocalTensor<T> sLocalout = outQueueS.DeQue<T>();
                AscendC::LocalTensor<T> temp = temp1.Get<T>();
                AscendC::ReduceSum<T>(sLocalout, sLocalout, temp, lastdimnum);//规约到第一个数
                
                AscendC::DataCopyExtParams copyParams2{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
                AscendC::DataCopyPad(sGm[progress * tiling.core_tile_x1 + row + rowsss], sLocalout, copyParams2); 
                outQueueS.FreeTensor(sLocalout);
            }else{
                //分到每一个的行数
                int hangshu=  rowsss + progress*tiling.core_tile_x1 +row;
                int parts= tiling.rows / tiling.partnum;//一个部分最多的行数
                int inparts= tiling.sumspacerows / tiling.partnum;//一个部分的行数 
                int elepartnum =  hangshu / parts;//部分偏移
                int eleinparts =  hangshu % inparts;//部内偏移
                AscendC::LocalTensor<T> sLocalout = outQueueS.DeQue<T>();
                AscendC::DataCopyExtParams copyParams2{static_cast<uint16_t>(1),  static_cast<uint32_t>(lastdimnum*sizeof(T)), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
                AscendC::SetAtomicAdd<T>();
                AscendC::DataCopyPad(sGm[elepartnum * lastdimnum *inparts  +  lastdimnum * eleinparts], sLocalout,copyParams2); 
                AscendC::SetAtomicNone();
                outQueueS.FreeTensor(sLocalout);
            }
        }else{
            AscendC::LocalTensor<T> zlocal = outQueueZ.DeQue<T>();
            AscendC::DataCopyExtParams copyParams2{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(T)*lastdimnum), 0, 0, 0}; // 结构体DataCopyExtParams最后一个参数是rsv保留位
            AscendC::DataCopyPad(zGm[(progress * tiling.core_tile_x1 + row) * lastdimnum], zlocal, copyParams2); 
            outQueueZ.FreeTensor(zlocal);
        }
    }

template <typename T>
    __aicore__ inline void SoftmaxV2<T>::Compute(int32_t progress,int32_t row,int32_t circle)
    {
        if(circle==0){//第一个循环获取每个维度的max
            if(dim == dimNum-1){
                AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
                AscendC::LocalTensor<T> mLocalin = inQueueM.DeQue<T>();
                AscendC::LocalTensor<T> mLocalout = outQueueM.AllocTensor<T>();
                AscendC::LocalTensor<T> temp = temp0.Get<T>();

                AscendC::ReduceMax<T>(mLocalout, xLocal, temp, lastdimnum);//规约到第一个数

                inQueueX.FreeTensor(xLocal);
                inQueueM.FreeTensor(mLocalin);
                outQueueM.EnQue<T>(mLocalout);
                }else{
                AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
                AscendC::LocalTensor<T> mLocalin = inQueueM.DeQue<T>();
                AscendC::LocalTensor<T> mLocalout = outQueueM.AllocTensor<T>();

                AscendC::Max(mLocalout,xLocal, mLocalin, lastdimnum);

                inQueueX.FreeTensor(xLocal);
                inQueueM.FreeTensor(mLocalin);
                outQueueM.EnQue<T>(mLocalout);
            }
        }else if(circle==1){
            if(dim == dimNum-1){
                AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
                AscendC::LocalTensor<T> mLocalin = inQueueM.DeQue<T>();
                AscendC::LocalTensor<T> sLocalout = outQueueS.AllocTensor<T>();

                float max = static_cast<float>(mLocalin.GetValue(0));
                T max_t = static_cast<T>(-max);

                AscendC::Adds(xLocal,xLocal, max_t, lastdimnum);
                AscendC::Exp(sLocalout, xLocal, lastdimnum);

                inQueueX.FreeTensor(xLocal);
                inQueueM.FreeTensor(mLocalin);
                outQueueS.EnQue<T>(sLocalout);
            }else{
                AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
                AscendC::LocalTensor<T> mLocalin = inQueueM.DeQue<T>();
                AscendC::LocalTensor<T> sLocalout = outQueueS.AllocTensor<T>();
                AscendC::Sub(xLocal,xLocal, mLocalin, lastdimnum);
                AscendC::Exp(sLocalout, xLocal, lastdimnum);
                inQueueX.FreeTensor(xLocal);
                inQueueM.FreeTensor(mLocalin);
                outQueueS.EnQue<T>(sLocalout);
            }
        }else{
            if(dim == dimNum-1){
                AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
                AscendC::LocalTensor<T> sLocalin = inQueueS.DeQue<T>();
                AscendC::LocalTensor<T> mLocalin = inQueueM.DeQue<T>();
                AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
                //s取求和所得值
                float max = static_cast<float>(mLocalin.GetValue(0));
                T max_t = static_cast<T>(-max);
                AscendC::Adds(xLocal,xLocal, max_t, lastdimnum);
                AscendC::Exp(xLocal, xLocal, lastdimnum);
                T temp = sLocalin.GetValue(0);
                AscendC::Duplicate<T>(sLocalin, temp, lastdimnum);
                AscendC::Div(zLocal, xLocal, sLocalin, lastdimnum);
                inQueueX.FreeTensor(xLocal);
                inQueueS.FreeTensor(sLocalin);
                inQueueM.FreeTensor(mLocalin);
                outQueueZ.EnQue<T>(zLocal);
            }else{
                AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
                AscendC::LocalTensor<T> sLocalin = inQueueS.DeQue<T>();
                AscendC::LocalTensor<T> mLocalin = inQueueM.DeQue<T>();
                AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
    
                AscendC::Sub(xLocal,xLocal, mLocalin, lastdimnum);
                AscendC::Exp(xLocal, xLocal, lastdimnum);
                AscendC::Div(zLocal, xLocal, sLocalin, lastdimnum);

                inQueueX.FreeTensor(xLocal);
                inQueueS.FreeTensor(sLocalin);
                inQueueM.FreeTensor(mLocalin);
                outQueueZ.EnQue<T>(zLocal);
            }
    }
}

template <typename T>
    __aicore__ inline void SoftmaxV2<T>::Process()
    {
        for(int k=0;k<3;k++){
            int32_t loopCount = this->tileNum;
            this->processNum = tiling.core_tile_x1;
            for (int32_t i = 0; i < loopCount-1; i++)
            {
                for(int32_t j = 0; j<this->processNum ;j++){
                    CopyIn(i,j,k);
                    Compute(i,j,k);
                    CopyOut(i,j,k);
                }
            }
            this->processNum = this->tailNum;
            for(int32_t j =0;j<this->processNum ; j++){
                CopyIn(loopCount-1,j,k);
                Compute(loopCount-1,j,k);
                CopyOut(loopCount-1,j,k);
            }
            AscendC::SyncAll();
        }
    } 

} // namespace NsSoftmaxV2
#endif // SoftmaxV2_H