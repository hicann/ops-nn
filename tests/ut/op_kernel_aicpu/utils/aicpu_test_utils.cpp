/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#include "aicpu_test_utils.h"

using namespace std;

uint64_t CalTotalElements(std::vector<std::vector<int64_t>> &shapes,
                          uint32_t index) {
  if(index < 0U) {
    return 0;
  }
  uint64_t nums = 1;
  for(auto shape : shapes[index]) {
    nums = nums * shape;
  }
  return nums;
}

template <>
bool CompareResult(float output[], float expect_output[], uint64_t num) {
  bool result = true;
  for (uint64_t i = 0; i < num; ++i) {
    double absolute_error = std::fabs(output[i] - expect_output[i]);
    double relative_error = 0;
    if (expect_output[i] == 0) {
      relative_error = 2e-6;
    } else {
      relative_error = absolute_error / std::fabs(expect_output[i]);
    }
    if ((absolute_error > 1e-6) && (relative_error > 1e-6)) {
      std::cout << "output[" << i << "] = ";
      std::cout << output[i];
      std::cout << ", expect_output[" << i << "] = ";
      std::cout << expect_output[i] << std::endl;
      result = false;
    }
  }
  return result;
}

bool CompareResultAllClose(float output[], float expect_output[], uint64_t num) {
  bool result = true;
  for (uint64_t i = 0; i < num; ++i) {
    double absolute_error = std::fabs(output[i] - expect_output[i]);
    float atol = 1e-4;
    float rtol = 1e-4;
    double relative_error = 0;
    if (!(absolute_error <= (atol + rtol * std::fabs(expect_output[i])))) {
      std::cout << "output[" << i << "] = ";
      std::cout << output[i];
      std::cout << ", expect_output[" << i << "] = ";
      std::cout << expect_output[i] << std::endl;
      result = false;
    }
  }

  return result;
}

template <>
bool CompareResult(double output[], double expect_output[], uint64_t num) {
  bool result_code = true;
  for (uint64_t i = 0; i < num; ++i) {
    double absolute_error = std::fabs(output[i] - expect_output[i]);
    double relative_error = 0;
    if (expect_output[i] == 0) {
      relative_error = 2e-10;
    } else {
      relative_error = absolute_error / std::fabs(expect_output[i]);
    }
    if ((absolute_error > 1e-12) && (relative_error > 1e-10)) {
      std::cout << "output[" << i << "] = ";
      std::cout << output[i];
      std::cout << ", expect_output[" << i << "] = ";
      std::cout << expect_output[i] << std::endl;
      result_code = false;
    }
  }
  return result_code;
}

template <>
bool CompareResult(complex<float> output[], complex<float> expect_output[], uint64_t num) {
  bool result = true;
  for (uint64_t i = 0; i < num; ++i) {
    double absolute_real_error = std::fabs(output[i].real() - expect_output[i].real());
    double relative_real_error = 0;
    if (expect_output[i].real() == 0) {
      relative_real_error = 2e-6;
    } else {
      relative_real_error = absolute_real_error / std::fabs(expect_output[i].real());
    }

    double absolute_imag_error = std::fabs(output[i].imag() - expect_output[i].imag());
    double relative_imag_error = 0;
    if (expect_output[i].imag() == 0) {
      relative_imag_error = 2e-6;
    } else {
      relative_imag_error = absolute_imag_error / std::fabs(expect_output[i].imag());
    }

    if ((absolute_real_error > 1e-6) && (relative_real_error > 1e-6) ||
        (absolute_imag_error > 1e-6) && (relative_imag_error > 1e-6)) {
      result = false;
      std::cout << "output[" << i << "] = ";
      std::cout << output[i];
      std::cout << ", expect_output[" << i << "] = ";
      std::cout << expect_output[i] << std::endl;
    }
  }

  return result;
}

template <>
bool CompareResult(complex<double> output[], complex<double> expect_output[], uint64_t num) {
  bool result = true;
  for (uint64_t i = 0; i < num; ++i) {
    double absolute_real_error = std::fabs(output[i].real() - expect_output[i].real());
    double relative_real_error = 0;
    if (expect_output[i].real() == 0) {
      relative_real_error = 2e-10;
    } else {
      relative_real_error = absolute_real_error / std::fabs(expect_output[i].real());
    }

    double absolute_imag_error = std::fabs(output[i].imag() - expect_output[i].imag());
    double relative_imag_error = 0;
    if (expect_output[i].imag() == 0) {
      relative_imag_error = 2e-10;
    } else {
      relative_imag_error = absolute_imag_error / std::fabs(expect_output[i].imag());
    }

    if ((absolute_real_error > 10e-8) && (relative_real_error > 10e-8) ||
        (absolute_imag_error > 10e-8) && (relative_imag_error > 10e-8)) {
      std::cout << "output[" << i << "] = ";
      std::cout << output[i];
      std::cout << ", expect_output[" << i << "] = ";
      std::cout << expect_output[i] << std::endl;
      result = false;
    }
  }
  return result;
}

bool CompareResult(Eigen::half output[], Eigen::half expect_output[],
    uint64_t num) {
  bool result = true;
  for (uint64_t i = 0; i < num; ++i) {
    Eigen::half absolute_error = (output[i] - expect_output[i]);
    absolute_error =
        absolute_error >= Eigen::half(0) ? absolute_error : -absolute_error;
    Eigen::half relative_error(0);
    if (expect_output[i] == Eigen::half(0)) {
      relative_error = Eigen::half(2e-3);
    } else {
      relative_error = absolute_error / expect_output[i];
      relative_error =
          relative_error >= Eigen::half(0) ? relative_error : -relative_error;
    }
    if ((absolute_error > Eigen::half(1e-3)) &&
        (relative_error > Eigen::half(1e-3))) {
      std::cout << "output[" << i << "] = ";
      std::cout << output[i];
      std::cout << ", expect_output[" << i << "] = ";
      std::cout << expect_output[i] << std::endl;
      result = false;
    }
  }
  return result;
}