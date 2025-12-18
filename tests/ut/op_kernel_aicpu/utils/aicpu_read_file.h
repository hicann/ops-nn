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

#ifndef OPS_TEST_UTILS_AICPU_READ_FILE_H_
#define OPS_TEST_UTILS_AICPU_READ_FILE_H_
#include <iostream>
#include <string>
#include <fstream>
#include <exception>
#include <vector>
#include "Eigen/Core"

bool ReadFile(std::string file_name, Eigen::half output[], uint64_t size);

template<typename T>
bool ReadFile(std::string file_name, std::vector<T> &output) {
  try {
    std::ifstream in_file{file_name};
    if (!in_file.is_open()) {
      std::cout << "open file: " << file_name << " failed." << std::endl;
      return false;
    }
    T tmp;
    while (in_file >> tmp) {
      output.push_back(tmp);
    }
    in_file.close();
  } catch (std::exception &e) {
    std::cout << "read file " << file_name << " failed, "
              << e.what() << std::endl;
    return false;
  }
  return true;
}

template<typename T>
bool ReadFile(std::string file_name, T output[], uint64_t size) {
  try {
    std::ifstream in_file{file_name};
    if (!in_file.is_open()) {
      std::cout << "open file: " << file_name << " failed." << std::endl;
      return false;
    }

    T tmp;
    uint64_t index = 0;
    while (in_file >> tmp) {
      if (index >= size) {
        break;
      }
      output[index] = tmp;
      index++;
    }
    in_file.close();
  } catch (std::exception &e) {
    std::cout << "read file " << file_name << " failed,  " << e.what() << std::endl;
    return false;
  }
  return true;
}

template <typename T,
          typename std::enable_if<!(std::is_floating_point<T>::value || std::is_same<T, Eigen::half>::value ||
                                    std::is_same<T, Eigen::bfloat16>::value),
                                  T>::type* = nullptr>
bool ReadFile2(std::string file_name, T output[], uint64_t size) {
  try {
    std::ifstream in_file{file_name};
    if (!in_file.is_open()) {
      std::cout << "open file: " << file_name << " failed." << std::endl;
      return false;
    }

    T tmp;
    uint64_t index = 0;
    while (in_file >> tmp) {
      if (index >= size) {
        break;
      }
      output[index] = tmp;
      index++;
    }
    in_file.close();
  } catch (std::exception& e) {
    std::cout << "read file " << file_name << " failed, " << e.what() << std::endl;
    return false;
  }
  return true;
}

template <typename T, typename std::enable_if<std::is_floating_point<T>::value || std::is_same<T, Eigen::half>::value ||
                                                  std::is_same<T, Eigen::bfloat16>::value,
                                              T>::type* = nullptr>
bool ReadFile2(std::string file_name, T output[], uint64_t size) {
  try {
    std::ifstream in_file{file_name};
    if (!in_file.is_open()) {
      std::cout << "open file: " << file_name << " failed." << std::endl;
      return false;
    }

    std::string tmp;
    uint64_t index = 0;
    while (in_file >> tmp) {
      if (index >= size) {
        break;
      }
      if (tmp == "inf") {
        output[index++] = static_cast<T>(INFINITY);
      } else if (tmp == "nan") {
        output[index++] = static_cast<T>(std::nan("1"));
      } else {
        output[index++] = static_cast<T>(std::stod(tmp));
      }
    }
    in_file.close();
  } catch (std::exception& e) {
    std::cout << "read file " << file_name << " failed, " << e.what() << std::endl;
    return false;
  }
  return true;
}

#endif