/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scatter_elements_aicpu.h"

#include <atomic>
#include <complex>
#include <memory>
#include <string>
#include <vector>

#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace {
using namespace aicpu;

const int32_t kInputNum = 3;
const int32_t kOutputNum = 1;
const int32_t kSplitSize = 64 * 1024;
const char *const kScatterElements = "ScatterElements";
const uint32_t kUpdatesInputIndex = 2;

const uint8_t kReductionNone = 0;
const uint8_t kReductionAdd = 1;
const uint8_t kReductionMul = 2;
const uint8_t kReductionMax = 3;
const uint8_t kReductionMin = 4;
const uint8_t kReductionMean = 5;

inline int64_t FloorDiv(int64_t value, int64_t divisor) {
  if (divisor == 0) {
    return 0;
  }
  int64_t quotient = value / divisor;
  int64_t remainder = value % divisor;
  if (remainder != 0 && value < 0) {
    --quotient;
  }
  return quotient;
}

template <typename T>
inline T MeanDivide(T sum, int64_t count) {
  if (count == 0) {
    return T(0);
  }
  return sum / static_cast<T>(count);
}

template <>
inline int8_t MeanDivide<int8_t>(int8_t sum, int64_t count) {
  if (count == 0) {
    return 0;
  }
  return static_cast<int8_t>(FloorDiv(static_cast<int64_t>(sum), count));
}

template <>
inline int16_t MeanDivide<int16_t>(int16_t sum, int64_t count) {
  if (count == 0) {
    return 0;
  }
  return static_cast<int16_t>(FloorDiv(static_cast<int64_t>(sum), count));
}

template <>
inline int32_t MeanDivide<int32_t>(int32_t sum, int64_t count) {
  if (count == 0) {
    return 0;
  }
  return static_cast<int32_t>(FloorDiv(static_cast<int64_t>(sum), count));
}

template <>
inline int64_t MeanDivide<int64_t>(int64_t sum, int64_t count) {
  if (count == 0) {
    return 0;
  }
  return FloorDiv(sum, count);
}

template <typename T>
class MeanReductionHelper {
 public:
  MeanReductionHelper(int64_t size, bool include_self) : size_(size), include_self_(include_self) {
    counts_.resize(size, 0);
    sums_.resize(size, T(0));
  }

  void AddUpdate(int64_t index, T value) {
    sums_[index] += value;
    counts_[index]++;
  }

  void Finalize(T *output) {
    for (int64_t i = 0; i < size_; ++i) {
      if (counts_[i] > 0) {
        output[i] = MeanDivide(sums_[i], counts_[i]);
      }
    }
  }

 private:
  int64_t size_;
  bool include_self_;
  std::vector<int64_t> counts_;
  std::vector<T> sums_;
};

template <>
class MeanReductionHelper<std::complex<float>> {
 public:
  MeanReductionHelper(int64_t size, bool include_self) : size_(size), include_self_(include_self) {
    counts_.resize(size, 0);
    sums_.resize(size, std::complex<float>(0.0f, 0.0f));
  }

  void AddUpdate(int64_t index, std::complex<float> value) {
    sums_[index] += value;
    counts_[index]++;
  }

  void Finalize(std::complex<float> *output) {
    for (int64_t i = 0; i < size_; ++i) {
      if (counts_[i] > 0) {
        output[i] = sums_[i] / static_cast<float>(counts_[i]);
      }
    }
  }

 private:
  int64_t size_;
  bool include_self_;
  std::vector<int64_t> counts_;
  std::vector<std::complex<float>> sums_;
};

template <>
class MeanReductionHelper<std::complex<double>> {
 public:
  MeanReductionHelper(int64_t size, bool include_self) : size_(size), include_self_(include_self) {
    counts_.resize(size, 0);
    sums_.resize(size, std::complex<double>(0.0, 0.0));
  }

  void AddUpdate(int64_t index, std::complex<double> value) {
    sums_[index] += value;
    counts_[index]++;
  }

  void Finalize(std::complex<double> *output) {
    for (int64_t i = 0; i < size_; ++i) {
      if (counts_[i] > 0) {
        output[i] = sums_[i] / static_cast<double>(counts_[i]);
      }
    }
  }

 private:
  int64_t size_;
  bool include_self_;
  std::vector<int64_t> counts_;
  std::vector<std::complex<double>> sums_;
};

template <>
class MeanReductionHelper<bool> {
 public:
  MeanReductionHelper(int64_t size, bool include_self) : size_(size), include_self_(include_self) {
    counts_.resize(size, 0);
    sums_.resize(size, 0);
  }

  void AddUpdate(int64_t index, bool value) {
    sums_[index] += value ? 1 : 0;
    counts_[index]++;
  }

  void Finalize(bool *output) {
    for (int64_t i = 0; i < size_; ++i) {
      if (counts_[i] > 0) {
        output[i] = sums_[i] > (counts_[i] / 2);
      }
    }
  }

 private:
  int64_t size_;
  bool include_self_;
  std::vector<int64_t> counts_;
  std::vector<int64_t> sums_;
};

struct ScatterElementsComputeInfo {
  int64_t total_value_num = 0;
  int64_t axis_value = 0;
  uint8_t reduction_flag = kReductionNone;
  bool include_self = true;
  int64_t value_dim_num_x1 = 0;
  int64_t value_dim_num_x2 = 0;
  int64_t value_dim_num_x3 = 0;
  std::vector<int64_t> value_dim_x1;
  std::vector<int64_t> value_dim_x2;
  std::vector<int64_t> value_dim_x3;
  std::vector<int64_t> data_dim_vec;
  std::vector<int64_t> index_dim_vec;
  std::vector<int64_t> src_dim_vec;
  int64_t axis_dim_value = 0;
  int64_t update_value_num = 0;
  int64_t update_src_num = 0;
  int64_t data_dim_size = -1;
};

std::vector<int64_t> GetTensorDims(const std::shared_ptr<TensorShape> &shape, int64_t &dim_num) {
  if (dim_num == 0) {
    dim_num = 1;
    return {1};
  }

  return shape->GetDimSizes();
}

uint8_t GetReductionFlag(const std::string &rdt_value) {
  if (rdt_value == "add" || rdt_value == "sum") {
    return kReductionAdd;
  }
  if (rdt_value == "mul" || rdt_value == "prod") {
    return kReductionMul;
  }
  if (rdt_value == "max" || rdt_value == "amax") {
    return kReductionMax;
  }
  if (rdt_value == "min" || rdt_value == "amin") {
    return kReductionMin;
  }
  if (rdt_value == "mean") {
    return kReductionMean;
  }
  return kReductionNone;
}

template <typename T>
inline void AddComputeWithSelf(T *output, const T *update, int64_t index, int64_t update_idx) {
  output[index] += update[update_idx];
}

template <typename T>
inline void MulComputeWithSelf(T *output, const T *update, int64_t index, int64_t update_idx) {
  output[index] *= update[update_idx];
}

template <typename T>
inline void MaxComputeWithSelf(T *output, const T *update, int64_t index, int64_t update_idx) {
  if (update[update_idx] > output[index]) {
    output[index] = update[update_idx];
  }
}

template <typename T>
inline void MinComputeWithSelf(T *output, const T *update, int64_t index, int64_t update_idx) {
  if (update[update_idx] < output[index]) {
    output[index] = update[update_idx];
  }
}

template <typename T>
inline void AddComputeWithoutSelf(T *output, const T *update, int64_t index, int64_t update_idx,
                                  int64_t update_count) {
  if (update_count == 0) {
    output[index] = update[update_idx];
  } else {
    output[index] += update[update_idx];
  }
}

template <typename T>
inline void MulComputeWithoutSelf(T *output, const T *update, int64_t index, int64_t update_idx,
                                  int64_t update_count) {
  if (update_count == 0) {
    output[index] = update[update_idx];
  } else {
    output[index] *= update[update_idx];
  }
}

template <typename T>
inline void MaxComputeWithoutSelf(T *output, const T *update, int64_t index, int64_t update_idx,
                                  int64_t update_count) {
  if (update_count == 0) {
    output[index] = update[update_idx];
  } else if (update[update_idx] > output[index]) {
    output[index] = update[update_idx];
  }
}

template <typename T>
inline void MinComputeWithoutSelf(T *output, const T *update, int64_t index, int64_t update_idx,
                                  int64_t update_count) {
  if (update_count == 0) {
    output[index] = update[update_idx];
  } else if (update[update_idx] < output[index]) {
    output[index] = update[update_idx];
  }
}

template <>
inline void AddComputeWithSelf<bool>(bool *output, const bool *update, int64_t index, int64_t update_idx) {
  output[index] = output[index] || update[update_idx];
}

template <>
inline void MulComputeWithSelf<bool>(bool *output, const bool *update, int64_t index, int64_t update_idx) {
  output[index] = output[index] && update[update_idx];
}

template <>
inline void MaxComputeWithSelf<bool>(bool *output, const bool *update, int64_t index, int64_t update_idx) {
  output[index] = output[index] || update[update_idx];
}

template <>
inline void MinComputeWithSelf<bool>(bool *output, const bool *update, int64_t index, int64_t update_idx) {
  output[index] = output[index] && update[update_idx];
}

template <>
inline void AddComputeWithoutSelf<bool>(bool *output, const bool *update, int64_t index, int64_t update_idx,
                                        int64_t update_count) {
  if (update_count == 0) {
    output[index] = update[update_idx];
  } else {
    output[index] = output[index] || update[update_idx];
  }
}

template <>
inline void MulComputeWithoutSelf<bool>(bool *output, const bool *update, int64_t index, int64_t update_idx,
                                        int64_t update_count) {
  if (update_count == 0) {
    output[index] = update[update_idx];
  } else {
    output[index] = output[index] && update[update_idx];
  }
}

template <>
inline void MaxComputeWithoutSelf<bool>(bool *output, const bool *update, int64_t index, int64_t update_idx,
                                        int64_t update_count) {
  if (update_count == 0) {
    output[index] = update[update_idx];
  } else {
    output[index] = output[index] || update[update_idx];
  }
}

template <>
inline void MinComputeWithoutSelf<bool>(bool *output, const bool *update, int64_t index, int64_t update_idx,
                                        int64_t update_count) {
  if (update_count == 0) {
    output[index] = update[update_idx];
  } else {
    output[index] = output[index] && update[update_idx];
  }
}

template <>
inline void MaxComputeWithSelf<std::complex<float>>(std::complex<float> *output, const std::complex<float> *update,
                                                    int64_t index, int64_t update_idx) {
  auto update_norm = std::norm(update[update_idx]);
  auto output_norm = std::norm(output[index]);
  if (update_norm > output_norm) {
    output[index] = update[update_idx];
  }
}

template <>
inline void MinComputeWithSelf<std::complex<float>>(std::complex<float> *output, const std::complex<float> *update,
                                                    int64_t index, int64_t update_idx) {
  auto update_norm = std::norm(update[update_idx]);
  auto output_norm = std::norm(output[index]);
  if (update_norm < output_norm) {
    output[index] = update[update_idx];
  }
}

template <>
inline void MaxComputeWithSelf<std::complex<double>>(std::complex<double> *output,
                                                     const std::complex<double> *update, int64_t index,
                                                     int64_t update_idx) {
  auto update_norm = std::norm(update[update_idx]);
  auto output_norm = std::norm(output[index]);
  if (update_norm > output_norm) {
    output[index] = update[update_idx];
  }
}

template <>
inline void MinComputeWithSelf<std::complex<double>>(std::complex<double> *output,
                                                     const std::complex<double> *update, int64_t index,
                                                     int64_t update_idx) {
  auto update_norm = std::norm(update[update_idx]);
  auto output_norm = std::norm(output[index]);
  if (update_norm < output_norm) {
    output[index] = update[update_idx];
  }
}

template <typename T, bool kGreater>
inline void ComplexComputeWithoutSelf(T *output, const T *update, int64_t index, int64_t update_idx,
                                      int64_t update_count) {
  auto update_norm = std::norm(update[update_idx]);
  if (update_count == 0) {
    output[index] = update[update_idx];
    return;
  }
  auto output_norm = std::norm(output[index]);
  if ((kGreater && update_norm > output_norm) || (!kGreater && update_norm < output_norm)) {
    output[index] = update[update_idx];
  }
}

template <>
inline void MaxComputeWithoutSelf<std::complex<float>>(std::complex<float> *output,
                                                       const std::complex<float> *update, int64_t index,
                                                       int64_t update_idx, int64_t update_count) {
  ComplexComputeWithoutSelf<std::complex<float>, true>(output, update, index, update_idx, update_count);
}

template <>
inline void MinComputeWithoutSelf<std::complex<float>>(std::complex<float> *output,
                                                       const std::complex<float> *update, int64_t index,
                                                       int64_t update_idx, int64_t update_count) {
  ComplexComputeWithoutSelf<std::complex<float>, false>(output, update, index, update_idx, update_count);
}

template <>
inline void MaxComputeWithoutSelf<std::complex<double>>(std::complex<double> *output,
                                                        const std::complex<double> *update, int64_t index,
                                                        int64_t update_idx, int64_t update_count) {
  ComplexComputeWithoutSelf<std::complex<double>, true>(output, update, index, update_idx, update_count);
}

template <>
inline void MinComputeWithoutSelf<std::complex<double>>(std::complex<double> *output,
                                                        const std::complex<double> *update, int64_t index,
                                                        int64_t update_idx, int64_t update_count) {
  ComplexComputeWithoutSelf<std::complex<double>, false>(output, update, index, update_idx, update_count);
}

template <typename T>
inline void NoneCompute(T *output, const T *update, int64_t index, int64_t update_idx) {
  output[index] = update[update_idx];
}

template <typename T>
inline void ApplyScatterReduction(const ScatterElementsComputeInfo &info, const T *updates, T *output,
                                  int64_t index_value, int64_t update_idx, std::vector<int64_t> &update_counts,
                                  MeanReductionHelper<T> *mean_helper) {
  if (info.reduction_flag == kReductionNone) {
    NoneCompute(output, updates, index_value, update_idx);
  } else if (info.reduction_flag == kReductionAdd) {
    info.include_self ? AddComputeWithSelf(output, updates, index_value, update_idx)
                      : AddComputeWithoutSelf(output, updates, index_value, update_idx, update_counts[index_value]);
    update_counts[index_value]++;
  } else if (info.reduction_flag == kReductionMul) {
    info.include_self ? MulComputeWithSelf(output, updates, index_value, update_idx)
                      : MulComputeWithoutSelf(output, updates, index_value, update_idx, update_counts[index_value]);
    update_counts[index_value]++;
  } else if (info.reduction_flag == kReductionMax) {
    info.include_self ? MaxComputeWithSelf(output, updates, index_value, update_idx)
                      : MaxComputeWithoutSelf(output, updates, index_value, update_idx, update_counts[index_value]);
    update_counts[index_value]++;
  } else if (info.reduction_flag == kReductionMin) {
    info.include_self ? MinComputeWithSelf(output, updates, index_value, update_idx)
                      : MinComputeWithoutSelf(output, updates, index_value, update_idx, update_counts[index_value]);
    update_counts[index_value]++;
  } else if (info.reduction_flag == kReductionMean && mean_helper != nullptr) {
    mean_helper->AddUpdate(index_value, updates[update_idx]);
  }
}

uint32_t InitScatterElementsInfo(const CpuKernelContext &ctx, ScatterElementsComputeInfo &info) {
  auto *data_tensor = ctx.Input(0);
  auto *indices_tensor = ctx.Input(1);
  auto *updates_tensor = ctx.Input(kUpdatesInputIndex);
  auto *axis = ctx.GetAttr("axis");
  auto *reduction = ctx.GetAttr("reduction");
  auto *include_self_attr = ctx.GetAttr("include_self");
  info.total_value_num = data_tensor->NumElements();
  info.axis_value = axis == nullptr ? 0 : axis->GetInt();
  info.include_self = include_self_attr == nullptr ? true : include_self_attr->GetBool();
  info.reduction_flag = GetReductionFlag(reduction == nullptr ? "none" : reduction->GetString());
  info.value_dim_num_x1 = data_tensor->GetTensorShape()->GetDims();
  info.value_dim_num_x2 = indices_tensor->GetTensorShape()->GetDims();
  info.value_dim_num_x3 = updates_tensor->GetTensorShape()->GetDims();
  info.value_dim_x1 = GetTensorDims(data_tensor->GetTensorShape(), info.value_dim_num_x1);
  info.value_dim_x2 = GetTensorDims(indices_tensor->GetTensorShape(), info.value_dim_num_x2);
  info.value_dim_x3 = GetTensorDims(updates_tensor->GetTensorShape(), info.value_dim_num_x3);
  info.update_value_num = indices_tensor->NumElements();
  info.update_src_num = updates_tensor->NumElements();
  KERNEL_CHECK_FALSE(info.value_dim_x2 <= info.value_dim_x3, KERNEL_STATUS_PARAM_INVALID,
                     "The shape of input indices must be smaller than or equal to that of input updates");
  KERNEL_CHECK_FALSE(info.value_dim_num_x1 == info.value_dim_num_x2 && info.value_dim_num_x2 == info.value_dim_num_x3,
                     KERNEL_STATUS_PARAM_INVALID, "Input dim values are different; data:%ld,indices:%ld,update:%ld",
                     info.value_dim_num_x1, info.value_dim_num_x2, info.value_dim_num_x3);
  KERNEL_CHECK_FALSE(info.axis_value >= info.value_dim_num_x1 * -1 && info.axis_value < info.value_dim_num_x1,
                     KERNEL_STATUS_PARAM_INVALID, "Axis_value %ld is out of range %ld", info.axis_value,
                     info.value_dim_num_x1);
  info.axis_value = info.axis_value < 0 ? info.axis_value + info.value_dim_num_x1 : info.axis_value;
  return KERNEL_STATUS_OK;
}

uint32_t BuildScatterElementsInfo(ScatterElementsComputeInfo &info) {
  int64_t sub_data_fix = 1;
  int64_t sub_index_fix = 1;
  int64_t sub_src_fix = 1;
  for (int64_t i = info.value_dim_num_x2 - 1; i >= 0; --i) {
    if (i != info.axis_value && info.value_dim_x1[i] < info.value_dim_x2[i]) {
      KERNEL_LOG_ERROR("The %ld dimension verfication failed:input0[%ld],input1[%ld]", i, info.value_dim_x1[i],
                       info.value_dim_x2[i]);
      return KERNEL_STATUS_PARAM_INVALID;
    }
    if (i > 0) {
      sub_data_fix *= info.value_dim_x1[i];
      info.data_dim_vec.push_back(sub_data_fix);
      sub_index_fix *= info.value_dim_x2[i];
      info.index_dim_vec.push_back(sub_index_fix);
      sub_src_fix *= info.value_dim_x3[i];
      info.src_dim_vec.push_back(sub_src_fix);
    }
  }
  info.axis_dim_value = info.value_dim_x1[info.axis_value];
  info.data_dim_size = static_cast<int64_t>(info.index_dim_vec.size()) - 1;
  return KERNEL_STATUS_OK;
}

template <typename TI>
uint32_t NormalizeIndicesValue(const ScatterElementsComputeInfo &info, TI raw_value, int64_t &indices_value) {
  KERNEL_CHECK_FALSE(raw_value >= info.axis_dim_value * -1 && raw_value < info.axis_dim_value,
                     KERNEL_STATUS_PARAM_INVALID, "Indices value %ld is out of bound %ld",
                     static_cast<int64_t>(raw_value), static_cast<int64_t>(info.axis_dim_value));
  indices_value = raw_value < 0 ? raw_value + info.axis_dim_value : raw_value;
  return KERNEL_STATUS_OK;
}

uint32_t CalcScatterIndices(const ScatterElementsComputeInfo &info, int64_t flat_index, int64_t indices_value,
                            int64_t &index_value, int64_t &src_index) {
  int64_t remain_index = flat_index;
  int64_t counter = 0;
  index_value = 0;
  src_index = 0;
  for (int64_t j = info.data_dim_size; j >= 0; --j) {
    index_value += ((counter == info.axis_value ? indices_value : remain_index / info.index_dim_vec[j])) *
                   info.data_dim_vec[j];
    src_index += (remain_index / info.index_dim_vec[j]) * info.src_dim_vec[j];
    remain_index %= info.index_dim_vec[j];
    ++counter;
  }
  index_value += counter == info.axis_value ? indices_value : remain_index;
  src_index += remain_index;
  return KERNEL_STATUS_OK;
}

template <typename TI>
uint32_t ResolveScatterIndex(const ScatterElementsComputeInfo &info, const TI *indices_data, int64_t flat_index,
                             bool check_src_index, int64_t &index_value, int64_t &src_index) {
  int64_t indices_value = 0;
  auto ret = NormalizeIndicesValue(info, indices_data[flat_index], indices_value);
  KERNEL_CHECK_FALSE(ret == KERNEL_STATUS_OK, ret, "NormalizeIndicesValue failed");
  CalcScatterIndices(info, flat_index, indices_value, index_value, src_index);
  KERNEL_CHECK_FALSE(index_value < info.total_value_num, KERNEL_STATUS_PARAM_INVALID,
                     "Update index %ld greater than %ld which is overflow", index_value, info.total_value_num);
  if (check_src_index) {
    KERNEL_CHECK_FALSE(src_index < info.update_src_num, KERNEL_STATUS_PARAM_INVALID,
                       "src index %ld greater than src total numbers %ld which is overflow", src_index,
                       info.update_src_num);
  }
  return KERNEL_STATUS_OK;
}

template <typename T, typename TI>
uint32_t ScatterSameNum(const ScatterElementsComputeInfo &info, const TI *indices_data, const T *updates, T *output,
                        std::vector<int64_t> &update_counts, MeanReductionHelper<T> *mean_helper) {
  for (int64_t i = 0; i < info.update_value_num; ++i) {
    int64_t index_value = 0;
    int64_t src_index = 0;
    auto ret = ResolveScatterIndex(info, indices_data, i, false, index_value, src_index);
    KERNEL_CHECK_FALSE(ret == KERNEL_STATUS_OK, ret, "ResolveScatterIndex failed");
    ApplyScatterReduction(info, updates, output, index_value, i, update_counts, mean_helper);
  }
  return KERNEL_STATUS_OK;
}

template <typename T, typename TI>
uint32_t ScatterDiffNum(const ScatterElementsComputeInfo &info, const TI *indices_data, const T *updates, T *output,
                        std::vector<int64_t> &update_counts, MeanReductionHelper<T> *mean_helper) {
  for (int64_t i = 0; i < info.update_value_num; ++i) {
    int64_t index_value = 0;
    int64_t src_index = 0;
    auto ret = ResolveScatterIndex(info, indices_data, i, true, index_value, src_index);
    KERNEL_CHECK_FALSE(ret == KERNEL_STATUS_OK, ret, "ResolveScatterIndex failed");
    ApplyScatterReduction(info, updates, output, index_value, src_index, update_counts, mean_helper);
  }
  return KERNEL_STATUS_OK;
}

}  // namespace

namespace aicpu {
uint32_t ScatterElementsCpuKernel::Compute(CpuKernelContext &ctx) {
  KERNEL_HANDLE_ERROR(NormalCheck(ctx, kInputNum, kOutputNum), "ScatterElements check input or output failed");
  auto data_type = ctx.Input(0)->GetDataType();
  KERNEL_CHECK_FALSE(ctx.Input(kUpdatesInputIndex)->GetDataType() == data_type, KERNEL_STATUS_PARAM_INVALID,
                     "Input[0] and Input[2] data types must be the same");
  auto indices_type = ctx.Input(1)->GetDataType();
  KERNEL_CHECK_FALSE(indices_type == DT_INT32 || indices_type == DT_INT64, KERNEL_STATUS_PARAM_INVALID,
                     "Input[1] data type[%s] is unsupported", DTypeStr(indices_type).c_str());
  return indices_type == DT_INT32 ? DispatchByDataType<int32_t>(ctx) : DispatchByDataType<int64_t>(ctx);
}

template <typename T>
uint32_t ScatterElementsCpuKernel::UpdateOutput(const CpuKernelContext &ctx, int64_t total_value_num) {
  auto *input_data = reinterpret_cast<T *>(ctx.Input(0)->GetData());
  auto *output_data = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  if (total_value_num == 0) {
    return KERNEL_STATUS_OK;
  }

  int64_t initial_size = total_value_num * static_cast<int64_t>(sizeof(T));
  int64_t max_thread_num = initial_size / kSplitSize;
  if (max_thread_num > total_value_num || max_thread_num == 0) {
    max_thread_num = total_value_num;
  }

  std::atomic<uint32_t> work_ret(KERNEL_STATUS_OK);
  size_t per_core_size = static_cast<size_t>(total_value_num / max_thread_num) * sizeof(T);
  size_t last_core_size = static_cast<size_t>(total_value_num % max_thread_num) * sizeof(T) + per_core_size;
  auto shard_copy = [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; ++i) {
      size_t core_size = i == max_thread_num - 1 ? last_core_size : per_core_size;
      int64_t ptr_offset = i * (total_value_num / max_thread_num);
      if (ptr_offset >= total_value_num) {
        work_ret = KERNEL_STATUS_PARAM_INVALID;
        KERNEL_LOG_ERROR("Pointer offset %ld more than %ld which is overflow", ptr_offset, total_value_num);
        return;
      }
      auto mem_ret = memcpy_s(output_data + ptr_offset, core_size, input_data + ptr_offset, core_size);
      if (mem_ret != EOK) {
        work_ret = KERNEL_STATUS_INNER_ERROR;
        KERNEL_LOG_ERROR("Initial memory copy failed[%d].Offerst is %ld, copy size is %lu", mem_ret, ptr_offset,
                         core_size);
        return;
      }
    }
  };
  KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, max_thread_num, 1, shard_copy), "Initial output data failed!");
  return work_ret;
}

template <typename TI>
uint32_t ScatterElementsCpuKernel::DispatchByDataType(CpuKernelContext &ctx) {
  switch (ctx.Input(0)->GetDataType()) {
    case DT_FLOAT16:
      return DoCompute<Eigen::half, TI>(ctx);
    case DT_BFLOAT16:
      return DoCompute<Eigen::bfloat16, TI>(ctx);
    case DT_FLOAT:
      return DoCompute<float, TI>(ctx);
    case DT_DOUBLE:
      return DoCompute<double, TI>(ctx);
    case DT_INT8:
      return DoCompute<int8_t, TI>(ctx);
    case DT_INT16:
      return DoCompute<int16_t, TI>(ctx);
    case DT_INT32:
      return DoCompute<int32_t, TI>(ctx);
    case DT_INT64:
      return DoCompute<int64_t, TI>(ctx);
    case DT_UINT8:
      return DoCompute<uint8_t, TI>(ctx);
    case DT_UINT16:
      return DoCompute<uint16_t, TI>(ctx);
    case DT_UINT32:
      return DoCompute<uint32_t, TI>(ctx);
    case DT_UINT64:
      return DoCompute<uint64_t, TI>(ctx);
    case DT_BOOL:
      return DoCompute<bool, TI>(ctx);
    case DT_COMPLEX64:
      return DoCompute<std::complex<float>, TI>(ctx);
    case DT_COMPLEX128:
      return DoCompute<std::complex<double>, TI>(ctx);
    default:
      KERNEL_LOG_ERROR("Input[0] data type[%s] is unsupported", DTypeStr(ctx.Input(0)->GetDataType()).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
}

template <typename T, typename TI>
uint32_t ScatterElementsCpuKernel::DoCompute(const CpuKernelContext &ctx) {
  ScatterElementsComputeInfo info;
  auto ret = InitScatterElementsInfo(ctx, info);
  KERNEL_CHECK_FALSE(ret == KERNEL_STATUS_OK, ret, "InitScatterElementsInfo failed");
  if (ctx.Input(kUpdatesInputIndex)->GetDataSize() == 0) {
    KERNEL_LOG_INFO("[%s] input of updates is empty tensor.", ctx.GetOpType().c_str());
    return UpdateOutput<T>(ctx, info.total_value_num);
  }

  ret = BuildScatterElementsInfo(info);
  KERNEL_CHECK_FALSE(ret == KERNEL_STATUS_OK, ret, "BuildScatterElementsInfo failed");
  ret = UpdateOutput<T>(ctx, info.total_value_num);
  KERNEL_CHECK_FALSE(ret == KERNEL_STATUS_OK, ret, "UpdateOutput failed");

  auto *indices_data = reinterpret_cast<TI *>(ctx.Input(1)->GetData());
  auto *updates = reinterpret_cast<T *>(ctx.Input(kUpdatesInputIndex)->GetData());
  auto *output = reinterpret_cast<T *>(ctx.Output(0)->GetData());
  std::vector<int64_t> update_counts(info.total_value_num, 0);
  std::unique_ptr<MeanReductionHelper<T>> mean_helper;
  if (info.reduction_flag == kReductionMean) {
    mean_helper.reset(new MeanReductionHelper<T>(info.total_value_num, info.include_self));
    if (info.include_self) {
      auto *input_data = reinterpret_cast<T *>(ctx.Input(0)->GetData());
      for (int64_t i = 0; i < info.total_value_num; ++i) {
        mean_helper->AddUpdate(i, input_data[i]);
      }
    }
  }

  ret = info.update_value_num == info.update_src_num
            ? ScatterSameNum(info, indices_data, updates, output, update_counts, mean_helper.get())
            : ScatterDiffNum(info, indices_data, updates, output, update_counts, mean_helper.get());
  KERNEL_CHECK_FALSE(ret == KERNEL_STATUS_OK, ret, "ScatterElements reduce failed");
  if (info.reduction_flag == kReductionMean && mean_helper != nullptr) {
    mean_helper->Finalize(output);
  }
  return KERNEL_STATUS_OK;
}

REGISTER_CPU_KERNEL(kScatterElements, ScatterElementsCpuKernel);
}  // namespace aicpu
