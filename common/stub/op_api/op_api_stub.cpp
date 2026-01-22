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
 * \file op_legacy_api.cpp
 * \brief
 */

#include "opdev/op_executor.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/pad.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/slice.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "level0/abs.h"
#include "level0/adaptive_avg_pool2d_assist_matrix.h"
#include "level0/add.h"
#include "level0/adjacent_difference.h"
#include "level0/arange.h"
#include "level0/avgpool_update.h"
#include "level0/avgpool_v2.h"
#include "level0/axpy.h"
#include "level0/batch_norm_backward.h"
#include "level0/broadcast_to.h"
#include "level0/concat.h"
#include "level0/cumsum.h"
#include "level0/dilation.h"
#include "level0/div.h"
#include "level0/dot.h"
#include "level0/dsa_random_uniform.h"
#include "level0/equal.h"
#include "level0/expand.h"
#include "level0/fault_injection.h"
#include "level0/fill.h"
#include "level0/gather_elements.h"
#include "level0/gather_v2.h"
#include "level0/gather_v3.h"
#include "level0/greater.h"
#include "level0/index_put.h"
#include "level0/index_put_with_sort.h"
#include "level0/index_put_with_sort_v2.h"
#include "level0/inplace_index_add.h"
#include "level0/less.h"
#include "level0/logical_and.h"
#include "level0/logical_or.h"
#include "level0/lp_norm_reduce_v2.h"
#include "level0/lp_norm_update_v2.h"
#include "level0/masked_fill.h"
#include "level0/max_pool_grad_with_argmax_v1.h"
#include "level0/max_pool_grad_with_argmax_v3.h"
#include "level0/max_pool_with_argmax_v1.h"
#include "level0/max_pool_with_argmax_v3.h"
#include "level0/max_pool3d_grad_with_argmax.h"
#include "level0/max_pool3d_with_argmax_v2.h"
#include "level0/maximum.h"
#include "level0/median_dim.h"
#include "level0/minimum.h"
#include "level0/mod.h"
#include "level0/mul.h"
#include "level0/muls.h"
#include "level0/not_equal.h"
#include "level0/ones_like.h"
#include "level0/padv3.h"
#include "level0/pooling.h"
#include "level0/pow.h"
#include "level0/realdiv.h"
#include "level0/reduce_any.h"
#include "level0/reduce_mean.h"
#include "level0/reduce_mean_with_count.h"
#include "level0/reduce_min.h"
#include "level0/reduce_sum_op.h"
#include "level0/reduce_var.h"
#include "level0/renorm.h"
#include "level0/right_shift.h"
#include "level0/rsqrt.h"
#include "level0/scatter_nd_add.h"
#include "level0/scatter_update.h"
#include "level0/select.h"
#include "level0/shape_op.h"
#include "level0/sort.h"
#include "level0/split_v.h"
#include "level0/squeeze.h"
#include "level0/stateless_random_uniform_v2.h"
#include "level0/sub.h"
#include "level0/sync_bn_training_update.h"
#include "level0/tensor_move.h"
#include "level0/threshold.h"
#include "level0/topk.h"
#include "level0/unique_with_counts_and_sorting.h"
#include "level0/unsqueeze.h"
#include "level0/zero_op.h"
namespace l0op {
const aclTensor* TensorMove(const aclTensor* x, const aclTensor* /*y*/, aclOpExecutor* /*executor*/)
{
    return x;
}
const aclTensor* Threshold(
    const aclTensor* self, const aclScalar* threshold, const aclScalar* /*value*/, aclOpExecutor* /*executor*/)
{
    return self;
}
const aclTensor* ZerosLike(const aclTensor* self, aclOpExecutor* /*executor*/)
{
    return self;
}
const aclTensor* Maximum(const aclTensor* self, const aclTensor* /*other*/, aclOpExecutor* /*executor*/)
{
    return self;
}
const aclTensor* Minimum(const aclTensor* self, const aclTensor* /*other*/, aclOpExecutor* /*executor*/)
{
    return self;
}
const aclTensor* Cast(const aclTensor* self, op::DataType /*dstDtype*/, aclOpExecutor* /*executor*/)
{
    return self;
}

aclTensor* ConcatD(const aclTensorList* inputs, int64_t dim, op::DataType outDtype, aclOpExecutor* executor)
{
    if (inputs->Size() == 0) {
        return nullptr;
    }
    auto out = executor->AllocTensor((*inputs)[0]->GetViewShape(), outDtype);
    return out;
}

aclTensor* ConcatD(const aclTensorList* inputs, int64_t dim, aclOpExecutor* executor)
{
    if (inputs->Size() == 0) {
        return nullptr;
    }
    auto out = executor->AllocTensor((*inputs)[0]->GetViewShape(), (*inputs)[0]->GetDataType());
    return out;
}

const aclTensor* CastOnlyForConvBackward(const aclTensor* self, op::DataType /*dstDtype*/, aclOpExecutor* /*executor*/)
{
    return self;
}

const aclTensor* Contiguous(const aclTensor* x, aclOpExecutor* /*executor*/)
{
    return x;
}

const aclTensor* ViewCopy(const aclTensor* x, const aclTensor* /*y*/, aclOpExecutor* /*executor*/)
{
    return x;
}

const aclTensor* PickViewAsContiguous(const aclTensor* x, aclOpExecutor* /*executor*/)
{
    return x;
}

const aclTensor* ReViewToOut(const aclTensor* x, const aclTensor* /*y*/, aclOpExecutor* /*executor*/)
{
    return x;
}

const std::tuple<aclTensor*, aclTensor*> Sort(
    const aclTensor* /*self*/, int64_t /*dim*/, bool /*descending*/, bool /*stable*/, op::DataType /*indicesType*/,
    aclOpExecutor* /*executor*/)
{
    return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
}

bool CanOptimizeContiguous(
    const op::Shape& /*viewShape*/, const op::Strides& /*strides*/, int64_t /*offset*/, int64_t /*storageSize*/,
    ContiguousParam& /*param*/)
{
    return true;
}

bool CanOptimizeView(
    const op::Shape& /*viewShape*/, const op::Strides& /*strides*/, int64_t /*offset*/, ContiguousParam& /*param*/)
{
    return true;
}

const aclTensor* Pad(const aclTensor* self, const aclTensor* /*paddings*/, aclOpExecutor* /*executor*/)
{
    return self;
}

const aclTensor* Reshape(const aclTensor* x, const op::Shape& /*shape*/, aclOpExecutor* /*executor*/)
{
    return x;
}

const aclTensor* Reshape(const aclTensor* x, const aclIntArray* /*shape*/, aclOpExecutor* /*executor*/)
{
    return x;
}

const aclTensor* Slice(
    const aclTensor* x, const aclTensor* /*y*/, const aclTensor* /*offset*/, const aclTensor* /*size*/,
    aclOpExecutor* /*executor*/)
{
    return x;
}

const aclTensor* Slice(
    const aclTensor* x, const aclIntArray* /*offsets*/, const aclIntArray* /*size*/, aclOpExecutor* /*executor*/)
{
    return x;
}

const aclTensorList *SplitV(const aclTensor* self, const aclIntArray* splitSize, int64_t dim, aclOpExecutor* executor)
{
    op::FVector<const aclTensor*> splitVector;
    for (size_t index = 0; index < splitSize->Size(); index++) {
        auto out = executor->AllocTensor(self->GetViewShape(), self->GetDataType());
        splitVector.emplace_back(out);
    }
    auto splitOut = executor->AllocTensorList(splitVector.data(), splitSize->Size());
    return splitOut;
}

const aclTensor *StatelessRandomUniformV2(
    const aclTensor* self, uint64_t /*seed*/, uint64_t /*offset*/, int32_t /*alg*/, aclOpExecutor* /*executor*/)
{
    return self;
}

const aclTensor* ReFormat(const aclTensor* x, const op::Format& /*format*/, aclOpExecutor* /*executor*/)
{
    return x;
}

const aclTensor* TransData(
    const aclTensor* x, op::Format /*dstPrimaryFormat*/, int64_t /*groups*/, aclOpExecutor* /*executor*/)
{
    return x;
}

const aclTensor* TransDataSpecial(
    const aclTensor* x, op::Format /*dstPrimaryFormat*/, int64_t /*groups*/, aclOpExecutor* /*executor*/)
{
    return x;
}

const aclTensor* Transpose(
    const aclTensor* x, const aclTensor* /*y*/, const aclTensor* /*perm*/, aclOpExecutor* /*executor*/)
{
    return x;
}
const aclTensor* Transpose(const aclTensor* x, const aclIntArray* /*perm*/, aclOpExecutor* /*executor*/)
{
    return x;
}

const aclTensor* Add(const aclTensor* self, const aclTensor* /*other*/, aclOpExecutor* /*executor*/)
{
    return self;
}
const aclTensor* Axpy(const aclTensor* self, const aclTensor* /*other*/, float /*alpha*/, aclOpExecutor* /*executor*/)
{
    return self;
}
const aclTensor* BroadcastTo(
    const aclTensor* x, const aclTensor* /*y*/, const aclTensor* /*shape*/, aclOpExecutor* /*executor*/)
{
    return x;
}
const aclTensor* BroadcastTo(const aclTensor* x, const aclIntArray* /*shape*/, aclOpExecutor* /*executor*/)
{
    return x;
}
const aclTensor* Dot(const aclTensor* self, const aclTensor* /*tensor*/, aclOpExecutor* /*executor*/)
{
    return self;
}

const aclTensor *DSARandomUniform(
    const aclIntArray* /*outShape*/, uint64_t /*seed*/, uint64_t /*offset*/,
    const aclScalar* /*low*/, const aclScalar* /*high*/, aclOpExecutor* /*executor*/)
{
    return nullptr;
}

const aclTensor* Fill(
    const aclTensor* /*dims*/, const aclTensor* value, const aclIntArray* /*outShape*/, aclOpExecutor* /*executor*/)
{
    return value;
}
const aclTensor* MaskedScatterWithPosition(const aclTensor* x, const aclTensor* /*mask*/, const aclTensor* /*position*/, const aclTensor* /*updates*/, aclOpExecutor* /*executor*/)
{
    return x;
}
const aclTensor* Mul(const aclTensor* self, const aclTensor* /*other*/, aclOpExecutor* /*executor*/)
{
    return self;
}
const aclTensor* Muls(const aclTensor* self, float /*alpha*/, aclOpExecutor* /*executor*/)
{
    return self;
}
const aclTensor* ReduceMean(
    const aclTensor* self, const aclIntArray* /*dim*/, bool /*keepDim*/, aclOpExecutor* /*executor*/)
{
    return self;
}
const aclTensor* ReduceMean(
    const aclTensor* self, const aclIntArray* /*dim*/, bool /*keepDim*/, bool /*noopWithEmptyAxes*/,
    aclOpExecutor* /*executor*/)
{
    return self;
}

const aclTensor* ReduceAny(const aclTensor* self, const aclIntArray* /*dim*/, bool /*keepDim*/, aclOpExecutor* /*executor*/)
{
    return self;
}

const aclTensor *LogicalAnd(const aclTensor *self, const aclTensor* /*other*/, aclOpExecutor* /*executor*/)
{
    return self;
}

const aclTensor *LogicalOr(const aclTensor *self, const aclTensor* /*other*/, aclOpExecutor* /*executor*/)
{
    return self;
}

const aclTensor *Less(const aclTensor *self, const aclTensor* /*other*/, aclOpExecutor* /*executor*/)
{
    return self;
}

const std::array<aclTensor*, 3> AdaptiveAvgPool2dAssistMatrix(
    const aclTensor* /*input*/, const aclTensor* /*origin_input*/, const aclIntArray* /*output_size*/,
    aclOpExecutor* /*executor*/)
{
    return {nullptr, nullptr, nullptr};
}
const aclTensor* SqueezeNd(const aclTensor* x, const aclIntArray* /*dim*/, aclOpExecutor* /*executor*/)
{
    return x;
}
const aclTensor* SqueezeNd(const aclTensor* x, int64_t /*dim*/, aclOpExecutor* /*executor*/)
{
    return x;
}
const std::tuple<const aclTensor*, const aclTensor*> MaxPool3DWithArgmaxV2Ncdhw(
    const aclTensor* /*self*/, const aclIntArray* /*kernelSize*/, const aclIntArray* /*stride*/,
    const aclIntArray* /*padding*/, const aclIntArray* /*dilation*/, bool /*ceilMode*/, std::string /*dataFormat*/,
    aclOpExecutor* /*executor*/)
{
    return std::make_tuple(nullptr, nullptr);
}
const aclTensor* MaxPool3DGradWithArgmax(
    const aclTensor* /*gradOutput*/, const aclTensor* self, const aclTensor* /*indices*/,
    const aclIntArray* /*kernelSize*/, const aclIntArray* /*stride*/, const aclIntArray* /*padding*/,
    const aclIntArray* /*dilation*/, bool /*ceilMode*/, aclOpExecutor* /*executor*/)
{
    return self;
}
const aclTensor* PadV3(
    const aclTensor* self, const aclTensor* /*paddings*/, const aclTensor* /*constant_values*/,
    const std::string& /*mode*/, const bool /*paddingsContiguous*/, aclOpExecutor* /*executor*/)
{
    return self;
}
const aclTensor* UnsqueezeNd(const aclTensor* x, const aclIntArray* /*dim*/, aclOpExecutor* /*executor*/)
{
    return x;
}

const aclTensor* UnsqueezeNd(const aclTensor* x, int64_t /*dim*/, aclOpExecutor* /*executor*/)
{
    return x;
}

const aclTensor* ReduceSumOp(
    const aclTensor* x, const aclIntArray* /*axes*/, bool /*keep_dims*/, aclOpExecutor* /*executor*/)
{
    return x;
}

const aclTensor* ReduceMeanWithCount(
    const aclTensor* input, const aclTensor* /*count*/, const aclTensor* /*countSum*/, const aclIntArray* /*axes*/,
    bool /*keepDims*/, aclOpExecutor* /*executor*/)
{
    return input;
}

const aclTensor* Rsqrt(const aclTensor* self, aclOpExecutor* /*executor*/)
{
    return self;
}

const aclTensor* OnesLike(const aclTensor* self, aclOpExecutor* /*executor*/)
{
    return self;
}

const aclTensor* Div(const aclTensor* self, const aclTensor* /*other*/, aclOpExecutor* /*executor*/)
{
    return self;
}

const aclTensor* RealDiv(const aclTensor* self, const aclTensor* /*other*/, aclOpExecutor* /*executor*/)
{
    return self;
}
std::tuple<aclTensor*, aclTensor*> Topk(
    const aclTensor* self, int64_t /*k*/, int64_t /*dim*/, bool /*largest*/, bool /*sorted*/,
    op::DataType /*indicesDType*/, aclOpExecutor* /*executor*/)
{
    return std::make_tuple(
        const_cast<aclTensor*>(self), // 移除 const，转为 aclTensor*
        const_cast<aclTensor*>(self));
}
const aclTensor* Abs(const aclTensor* self, aclOpExecutor* executor)
{
    return self;
}
aclTensor *AdjacentDifference(const aclTensor *x, op::DataType yDtype, aclOpExecutor *executor)
{
    return nullptr;
}
const aclTensor* Arange(const aclScalar* start, const aclScalar* end, const aclScalar* step, const aclTensor* out,
                        const bool isClosed, aclOpExecutor* executor)
{
    return out;
}
const aclTensor *Cumsum(const aclTensor *self, const aclTensor* dim, aclOpExecutor *executor)
{
    return self;
}
const aclTensor *Cumsum(const aclTensor *self, const aclTensor* dim, bool exclusive, bool reverse, aclOpExecutor *executor)
{
    return self;
}
const aclTensor *DSARandomUniformTensor(const aclIntArray *outShape,
                                  const aclTensor *seed,
                                  const aclTensor *offset,
                                  const aclScalar *low,
                                  const aclScalar *high,
                                  aclOpExecutor *executor)
{
    return seed;
}
const aclTensor *Equal(const aclTensor *self,
                     const aclTensor *other,
                     aclOpExecutor *executor)
{
    return self;
}
const aclTensor *Expand(const aclTensor *self, const aclIntArray *shape, aclOpExecutor *executor)
{
    return self;
}
const aclTensor* FaultInjection(const aclTensor *injectObj, const aclTensor *injectPara, aclTensor *out, aclOpExecutor *executor)
{
    return injectObj;
}
const aclTensor *Greater(const aclTensor *self,
                     const aclTensor *other,
                     aclOpExecutor *executor)
{
    return self;
}
const aclTensor *MaskedFill(const aclTensor *self, const aclTensor *mask,
                            const aclTensor *value, aclOpExecutor *executor)
{
    return self;
}
const aclTensor* Mod(const aclTensor* self, const aclTensor* other, aclOpExecutor* executor)
{
    return self;
}
const aclTensor *NotEqual(const aclTensor *self, const aclTensor *other, aclOpExecutor *executor)
{
    return self;
}
const aclTensor *Pow(const aclTensor *self, const aclTensor *exponent, aclOpExecutor *executor)
{
    return self;
}
const aclTensor *ReduceMin(const aclTensor *self, const aclIntArray *dims, bool keepDim, aclOpExecutor *executor)
{
    return self;
}
const std::tuple<const aclTensor *, const aclTensor *> ReduceVar(const aclTensor* self, const aclIntArray* dim,
    int64_t correction, bool keepdim, bool isMeanOut, aclOpExecutor* executor)
{
    return std::tuple<const aclTensor *, const aclTensor *>(nullptr, nullptr);
}
 const aclTensor* RightShift(const aclTensor* x, const aclTensor* y, aclOpExecutor* executor)
{
    return x;
}
const aclTensor *SelectV2(const aclTensor *condition,
                          const aclTensor *x1,
                          const aclTensor *x2,
                          aclOpExecutor *executor)
{
    return x1;
}
const aclTensor *Sub(const aclTensor *self, const aclTensor *other, aclOpExecutor *executor)
{
    return self;
}
const aclTensor *Dilation(const aclTensor *x, const aclIntArray *dilations, const aclIntArray *pads,
                          float paddingValue, aclOpExecutor *executor)
{
    return x;
}
const aclTensor *InplaceIndexAddAiCore(const aclTensor *self, const int64_t dim, const aclTensor *index,
                                       const aclTensor *source, const aclTensor *alphaTensor,
                                       aclOpExecutor *executor)
{
    return self;
}

const aclTensor *InplaceIndexAddAiCpu(const aclTensor *self, const int64_t dim, const aclTensor *index,
                                      const aclTensor *source, const aclTensor *alphaTensor,
                                      aclOpExecutor *executor)
{
    return self;
}

const aclTensor *InplaceIndexAddWithSorted(const aclTensor *self, const int64_t dim, const aclTensor *sortedIndices,
                                           const aclTensor *pos, const aclTensor *value, const aclTensor *alphaTensor,
                                           aclOpExecutor *executor)
{
    return self;
}
const aclTensor *GatherV3(const aclTensor *self, int64_t axis, const aclTensor *indices, aclOpExecutor *executor,
                          int batchDims, bool negativeIndexSupport)
{
    return self;
}
const aclTensor *IndexPut(const aclTensor *selfRef, const aclTensorList *indices, const aclTensor *values,
                          const aclTensor *masks, const bool accumulate, aclTensor *out, aclOpExecutor *executor)
{
    return selfRef;
}
const std::tuple<const aclTensor*, const aclTensor*> MaxPoolWithArgmaxV3(
    const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding,
    const op::DataType dtype, const aclIntArray* dilation, bool ceilMode, std::string& dataFormat,
    aclOpExecutor* executor)
{
    auto out1 = executor->AllocTensor(self->GetViewShape(), self->GetDataType());
    auto out2 = executor->AllocTensor(self->GetViewShape(), self->GetDataType());
    return std::tuple<const aclTensor*, const aclTensor*>(out1, out2);
}
const aclTensor* ScatterNdAdd(const aclTensor* self, const aclTensor* indices, const aclTensor* updates,
                              bool use_locking, aclOpExecutor* executor)
{
    return self;
}
const aclTensor *AvgPoolNchw(
    const aclTensor *x, const aclIntArray *window, const aclIntArray *stride, const std::string &paddingMode,
    const aclIntArray *pad, const std::string &dataFormat, const bool globalPooling, const bool ceilMode,
    const bool exclusive, const int64_t divisorOverride, aclOpExecutor *executor)
{
    return x;
}
const std::tuple<const aclTensor*, const aclTensor*> MaxPoolWithArgmaxV1(const aclTensor *self,
                                                                         const aclIntArray *kernelSize,
                                                                         const aclIntArray *stride,
                                                                         const aclIntArray *padding,
                                                                         const aclIntArray *dilation, bool ceilMode,
                                                                         aclOpExecutor *executor)
{
    return std::tuple<const aclTensor*, const aclTensor*>(nullptr, nullptr);
}
const aclTensor* LpNormUpdateV2(const aclTensor* x, float p, float epsilon, aclOpExecutor* executor)
{
    return x;
}
const std::tuple<aclTensor*, aclTensor*> MedianDim(const aclTensor *self, int64_t dim, bool keepDim, aclOpExecutor *executor)
{
    return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
}
const aclTensor *AvgPoolUpdate5Hd(
    const aclTensor *x1, const aclTensor *x2, const aclIntArray *ksize, const aclIntArray *strides,
    const char *paddingMode, const aclIntArray *pads, const char *dataFormat, const bool ceilMode,
    const bool exclusive, aclOpExecutor *executor)
{
    return x1;
}
const aclTensor* MaxPoolGradWithArgmaxV1(const aclTensor *gradOutput, const aclTensor *self,
                                         const aclTensor *indices, const aclIntArray *kernelSize,
                                         const aclIntArray *stride, const aclIntArray *padding,
                                         const aclIntArray *dilation, bool ceilMode,
                                         aclOpExecutor *executor)
{
    return self;
}
const aclTensor* LpNormReduceV2(
    const aclTensor* x, float p, const aclIntArray* dims, bool keepDim, float epsilon, aclOpExecutor* executor)
{
    return x;
}
const aclTensor *Shape_op(const aclTensor *x, aclOpExecutor *executor)
{
    return x;
}
const aclTensor* SyncBNTrainingUpdate(
    const aclTensor* meanAll, const aclTensor* runningMean, const float momentum, aclOpExecutor* executor)
{
    return meanAll;
}
const aclTensor *Pooling5Hd(
    const aclTensor *x, const aclTensor *weight, int64_t mode, bool globalPooling, const aclIntArray *window,
    const aclIntArray *stride, const aclIntArray *pad, const int64_t ceilMode, const char *dataFormat,
    aclOpExecutor *executor)
{
    return x;
}
const aclTensor* MaxPoolGradWithArgmaxV3(const aclTensor *gradOutput, const aclTensor *self,
                                         const aclTensor *indices, const aclIntArray *kernelSize,
                                         const aclIntArray *stride, const aclIntArray *padding,
                                         const ge::DataType dtype, const aclIntArray *dilation, bool ceilMode,
                                         std::string& dataFormat, aclOpExecutor *executor)
{
    return self;
}

const aclTensor *IndexPutWithSortV2(const aclTensor *self, const aclTensor *linearIndex, const aclTensor *posIdx,
                                    const aclTensor *values, const aclIntArray* indexed_sizes, const bool accumulate,
                                    aclTensor *out, aclOpExecutor *executor)
{
    return self;
}
aclnnStatus UniqueWithCountsAndSorting(const aclTensor* self, bool sorted, bool returnInverse, bool returnCounts,
                                       aclTensor* valueOut, aclTensor* inverseOut, aclTensor* countsOut,
                                       aclOpExecutor* executer)
{
    return ACLNN_SUCCESS;
}

aclnnStatus UniqueWithCountsAndSorting(const aclTensor* self, bool sorted, bool returnInverse, aclTensor* valueOut,
                                       aclTensor* inverseOut, aclOpExecutor* executor)
{
    return ACLNN_SUCCESS;
}
} // namespace l0op