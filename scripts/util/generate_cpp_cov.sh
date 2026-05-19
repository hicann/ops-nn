#!/bin/bash
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

logging() {
  echo "[INFO] $@" >&2
}

mk_dir() {
  local create_dir="$1"
  mkdir -pv "${create_dir}"
  logging "Created ${create_dir}"
}

get_lcov_version() {
  local version
  version=$(lcov --version 2>&1 | grep -oP 'version \K[0-9]+\.[0-9]+' | head -1)
  echo "$version"
}

version_ge() {
  local v1="$1"
  local v2="$2"
  
  local v1_major v1_minor v2_major v2_minor
  v1_major=$(echo "$v1" | cut -d. -f1)
  v1_minor=$(echo "$v1" | cut -d. -f2)
  v2_major=$(echo "$v2" | cut -d. -f1)
  v2_minor=$(echo "$v2" | cut -d. -f2)
  
  logging "Comparing versions: v1=$v1 (major=$v1_major, minor=$v1_minor) vs v2=$v2 (major=$v2_major, minor=$v2_minor)"
  
  if [[ $v1_major -gt $v2_major ]]; then
    logging "Result: v1 > v2 (major comparison)"
    return 0
  elif [[ $v1_major -eq $v2_major && $v1_minor -ge $v2_minor ]]; then
    logging "Result: v1 >= v2 (minor comparison)"
    return 0
  else
    logging "Result: v1 < v2"
    return 1
  fi
}

build_ignore_errors_remove() {
  local base_errors="inconsistent"
  local version
  version=$(get_lcov_version)
  
  if [[ -n "$version" ]] && version_ge "$version" "2.0"; then
    echo "empty,mismatch,unused,negative,source, $base_errors"
  else
    echo "$base_errors"
  fi
}

# using lcov to generate coverage for cpp files
generate_coverage() {
  local _source_dir="$1"
  local _coverage_file="$2"

  if [[ -z "${_source_dir}" ]]; then
    logging "directory required to find the .da files"
    exit 1
  fi

  if [[ ! -d "${_source_dir}" ]]; then
    logging "directory is not exist, please check ${_source_dir}"
    exit 1
  fi

  if [[ -z "${_coverage_file}" ]]; then
    _coverage_file="coverage.info"
    logging "using default file name to generate coverage"
  fi

  which lcov >/dev/null 2>&1
  if [[ $? -ne 0 ]]; then
    logging "lcov is required to generate coverage data, please install"
    exit 1
  fi

  local _path_to_gen="$(dirname ${_coverage_file})"
  if [[ ! -d "${_path_to_gen}" ]]; then
    mk_dir "${_path_to_gen}"
  fi

  local version
  version=$(get_lcov_version)
  if [[ -n "$version" ]] && version_ge "$version" "2.0"; then
    lcov --ignore-errors empty,mismatch,negative,source -c -d "${_source_dir}" -o "${_coverage_file}"
  else
    lcov  -c -d "${_source_dir}" -o "${_coverage_file}"
  fi
  local _ignore_errors_remove
  _ignore_errors_remove=$(build_ignore_errors_remove)
  lcov --ignore-errors ${_ignore_errors_remove} -r "${_coverage_file}" "${ASCEND_PARENT_PATH}/*" -o "${_coverage_file}"
 	
  logging "generated coverage file ${_coverage_file}"
}

# filter out some unused directories or files
filter_coverage() {
  local _coverage_file="$1"
  local _filtered_file="$2"

  if [[ ! -f "${_coverage_file}" ]]; then
    logging "coverage data file required"
    exit 1
  fi

  which lcov >/dev/null 2>&1
  if [[ $? -ne 0 ]]; then
    logging "lcov is required to generate coverage data, please install"
    exit 1
  fi

  local _ignore_errors_remove
  _ignore_errors_remove=$(build_ignore_errors_remove)

  lcov --remove "${_coverage_file}" '${ASCEND_PARENT_PATH}/*'\
                                    '/usr/include/*'          \
                                    '*/third_party/*' \
                                    '*/opensource/*' \
                                    '*/common/*'    \
                                    '*/ops-nn-dev/tests/*'  \
                                    '*/ops-nn-dev/*/*/tests/*' -o "${_filtered_file}" --ignore-errors ${_ignore_errors_remove}
}

# generate html report
generate_html() {
  local _filtered_file="$1"
  local _out_path="$2"

  which genhtml >/dev/null 2>&1
  if [[ $? -ne 0 ]]; then
    logging "genhtml is required to generate coverage html report, please install"
    exit 1
  fi

  if [[ ! -d "${_out_path}" ]]; then
    mk_dir "${_out_path}"
  fi

  if [[ ! -s "${_filtered_file}" ]]; then
    logging "empty coverage data"
  else
    genhtml "${_filtered_file}" -o "${_out_path}"
  fi
}


if [[ $# -ne 3 ]]; then
  logging "Usage: $0 DIR COV_FILE OUT_PATH"
  exit 1
fi

_src="$1"
_cov_file="$2"
_out="$3"

ASCEND_PARENT_PATH=$(dirname "${ASCEND_HOME_PATH}")
if [[ -z "${ASCEND_PARENT_PATH}" ]]; then
  logging "ASCEND_HOME_PATH is not set"
  exit 1
fi

# 检查目录是否存在且有覆盖率数据
if [ -d "${_src}" ] && [ -n "$(find "${_src}" -name "*.gcda" -print -quit 2>/dev/null)" ]; then
  generate_coverage "${_src}" "${_cov_file}"
  filter_coverage   "${_cov_file}" "${_cov_file}_filtered"
  generate_html     "${_cov_file}_filtered" "${_out}"
else
  logging "No coverage data found"
fi
