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

#include "aicpu_read_file.h"
#include <sstream>
#include <algorithm>
#include <cmath>

bool StringToBool(const std::string &str, bool &result) {
  result = false;
  std::string buff = str;
  try {
    std::transform(buff.begin(), buff.end(), buff.begin(),
        [](unsigned char c) -> unsigned char {
          return std::tolower(c);
        });
    if ((buff == "false") || (buff == "true")) {
      std::istringstream(buff) >> std::boolalpha >> result;
      return true;
    } else {
      std::cout << "convert to bool failed, " << buff
                << " is not a bool value." << std::endl;
      return false;
    }
  } catch (std::exception &e) {
    std::cout << "convert " << str << " to bool failed, "
              << e.what() << std::endl;
  }
  return false;
}

template<>
bool ReadFile(std::string file_name, std::vector<bool> &output) {
  try {
    std::ifstream input_file{file_name};
    if (!input_file.is_open()) {
      std::cout << "open file: " << file_name << " failed." <<std::endl;
      return false;
    }
    bool tmp;
    std::string read_str;
    while (input_file >> read_str) {
      if (StringToBool(read_str, tmp)) {
        output.push_back(tmp);
      }
    }
    input_file.close();
  } catch (std::exception &e) {
    std::cout << "read file " << file_name << " failed, " << e.what() << std::endl;
    return false;
  }
  return true;
}

template<>
bool ReadFile(std::string file_name, bool output[], uint64_t size) {
  try {
    std::ifstream in_file{file_name};
    if (!in_file.is_open()) {
      std::cout << "open file: " << file_name << " failed." <<std::endl;
      return false;
    }
    bool tmp;
    std::string read_str;
    uint64_t index = 0;
    while (in_file >> read_str) {
      if (StringToBool(read_str, tmp)) {
        if (index >= size) {
          break;
        }
        output[index] = tmp;
        index++;
      }
    }

    in_file.close();
  } catch (std::exception &e) {
    std::cout << "read file " << file_name << " failed, " << e.what() << std::endl;
    return false;
  }
  return true;
}

template<>
bool ReadFile(std::string file_name, int8_t output[], uint64_t size) {
  try {
    std::ifstream in_file{file_name};
    if (!in_file.is_open()) {
      std::cout << "open file: " << file_name << " failed." <<std::endl;
      return false;
    }
    uint64_t index = 0;
    int32_t tmp;
    while (in_file >> tmp) {
      if (index >= size) {
        break;
      }

      output[index] = static_cast<int8_t>(tmp);
      index++;
    }

    in_file.close();
  } catch (std::exception &e) {
    std::cout << "read file " << file_name << " failed, "
              << e.what() << std::endl;
    return false;
  }

  return true;
}

template<>
bool ReadFile(std::string file_name, uint8_t output[], uint64_t size) {
  try {
    std::ifstream in_file{file_name};
    if (!in_file.is_open()) {
      std::cout << "open file: " << file_name << " failed." <<std::endl;
      return false;
    }
    uint32_t tmp;
    uint64_t index = 0;
    while (in_file >> tmp) {
      if (index >= size) {
        break;
      }
      output[index] = static_cast<uint8_t>(tmp);
      index++;
    }
    in_file.close();
  } catch (std::exception &e) {
    std::cout << "read file " << file_name << " failed, " << e.what() << std::endl;
    return false;
  }
  return true;
}

std::string GetComplexReal(std::string tmp) {
  std::string tmp_connector = "+";
  std::string tmp_connector_minus = "-";
  std::string tmp_left = "(";
  std::string tmp_real;
  std::size_t tmp_connector_rfind =
      (tmp.rfind(tmp_connector) == std::string::npos) ? tmp.rfind(tmp_connector_minus)
                                                      : tmp.rfind(tmp_connector);
  std::size_t tmp_left_find = tmp.find(tmp_left);
  tmp_real = (tmp_left_find == std::string::npos) ? tmp.substr(0, tmp_connector_rfind - 1U)
                                                  : tmp.substr(tmp_left_find + 1U,
                                                               tmp_connector_rfind - 1U);
  return tmp_real;
}

std::string GetComplexImag(std::string tmp) {
  std::string tmp_connector = "+";
  std::string tmp_connector_minus = "-";
  std::string tmp_right = "j";
  std::string tmp_imag;
  std::size_t tmp_connector_rfind =
      (tmp.rfind(tmp_connector) == std::string::npos) ? tmp.rfind(tmp_connector_minus)
                                                      : tmp.rfind(tmp_connector);
  std::size_t tmp_right_find = tmp.find(tmp_right);
  tmp_imag = tmp_right_find == std::string::npos
                 ? tmp.substr(tmp_connector_rfind)
                 : tmp.substr(tmp_connector_rfind, tmp_right_find - tmp_connector_rfind);
  return tmp_imag;
}

template <>
bool ReadFile(std::string file_name, std::complex<float> output[], uint64_t size) {
  try {
    std::ifstream input_file{file_name};
    if (!input_file.is_open()) {
      std::cout << "open file: " << file_name << " failed." << std::endl;
      return false;
    }
    uint64_t index = 0;
    std::string tmp;
    while (input_file >> tmp) {
      if (index >= size) {
        break;
      }
      output[index] = std::complex<float>(stof(GetComplexReal(tmp)), stof(GetComplexImag(tmp)));
      index++;
    }

    input_file.close();
  } catch (std::exception& e) {
    std::cout << "read file " << file_name << " failed, " << e.what()
              << std::endl;
    return false;
  }
  return true;
}

template <>
bool ReadFile(std::string file_name, std::complex<double> output[],
              uint64_t size) {
  try {
    std::ifstream input_file{file_name};
    if (!input_file.is_open()) {
      std::cout << "open file: " << file_name << " failed." << std::endl;
      return false;
    }

    uint64_t index = 0;
    std::string tmp;
    while (input_file >> tmp) {
      if (index >= size) {
        break;
      }
      output[index] = std::complex<double>(stod(GetComplexReal(tmp)), stod(GetComplexImag(tmp)));
      index++;
    }
    input_file.close();
  } catch (std::exception& e) {
    std::cout << "read file " << file_name << " failed, " << e.what()
              << std::endl;
    return false;
  }
  return true;
}

bool ReadFile(std::string file_name, Eigen::half output[], uint64_t size) {
  try {
    std::ifstream in_file{file_name};
    if (!in_file.is_open()) {
      std::cout << "open file: " << file_name << " failed." <<std::endl;
      return false;
    }
    float tmp;
    uint64_t index = 0;
    while (in_file >> tmp) {
      if (index >= size) {
        break;
      }

      output[index] = static_cast<Eigen::half>(tmp);
      index++;
    }
    in_file.close();
  } catch (std::exception &e) {
    std::cout << "read file " << file_name << " failed, "
              << e.what() << std::endl;
    return false;
  }
  return true;
}

template <>
bool ReadFile2(std::string file_name, bool output[], uint64_t size) {
  return ReadFile(file_name, output, size);
}

template <>
bool ReadFile2(std::string file_name, int8_t output[], uint64_t size) {
  return ReadFile(file_name, output, size);
}

template <>
bool ReadFile2(std::string file_name, uint8_t output[], uint64_t size) {
  return ReadFile(file_name, output, size);
}

template <>
bool ReadFile2(std::string file_name, std::complex<float> output[], uint64_t size) {
  return ReadFile(file_name, output, size);
}

template <>
bool ReadFile2(std::string file_name, std::complex<double> output[], uint64_t size) {
  return ReadFile(file_name, output, size);
}