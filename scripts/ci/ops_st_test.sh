#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

declare -A SOC_TO_ST_ARCH
SOC_TO_ST_ARCH=(["ascend910b"]="arch22" ["ascend950"]="arch35")

dotted_line="----------------------------------------------------------------"
print_msg() {
    local msg="$1"
    local date_time
    date_time=$(date +%Y-%m-%d/%H.%M.%S)
    echo "[INFO]${date_time}: ${msg}" >&2
}

print_error() {
    echo >&2
    echo $dotted_line >&2
    local msg="$1"
    echo -e "\033[31m[ERROR] ${msg}\033[0m" >&2
    echo $dotted_line >&2
    echo >&2
}

print_success() {
    echo >&2
    echo $dotted_line >&2
    local msg="$1"
    echo -e "\033[32m[SUCCESS] ${msg}\033[0m" >&2
    echo $dotted_line >&2
    echo >&2
}

print_warning() {
    echo >&2
    echo $dotted_line >&2
    local msg="$1"
    echo -e "\033[33m[WARNING] ${msg}\033[0m" >&2
    echo $dotted_line >&2
    echo >&2
}

get_op_categories() {
    local cmake_file="${framework_path}/cmake/variables.cmake"
    local categories=""
    
    if [[ -f "${cmake_file}" ]]; then
        categories=$(grep "OP_CATEGORY_LIST" "${cmake_file}" | \
            sed -n 's/set(OP_CATEGORY_LIST "\(.*\)")/\1/p' | \
            tr -d '"')
    else
        categories="matmul conv activation foreach hash vfusion index loss norm optim pooling quant rnn control"
    fi
    
    categories="${categories} common"
    echo "${categories}"
}

extract_op_from_path() {
    local file_path="$1"
    local op_categories="$2"
    local op_name=""
    
    local rel_path="${file_path#${framework_path}/}"
    local parts=(${rel_path//\// })
    
    if [[ ${#parts[@]} -ge 2 ]]; then
        local first_dir="${parts[0]}"
        local second_dir="${parts[1]}"
        
        local is_category=0
        for cat in ${op_categories}; do
            if [[ "${first_dir}" == "${cat}" ]]; then
                is_category=1
                break
            fi
        done
        
        if [[ ${is_category} -eq 1 ]]; then
            if [[ "${second_dir}" == "common" ]]; then
                op_name="${first_dir}.common"
            else
                op_name="${second_dir}"
            fi
        elif [[ "${first_dir}" == "experimental" && ${#parts[@]} -ge 3 ]]; then
            local exp_type="${parts[1]}"
            local exp_name="${parts[2]}"
            for cat in ${op_categories}; do
                if [[ "${exp_type}" == "${cat}" ]]; then
                    if [[ "${exp_name}" == "common" ]]; then
                        op_name="${exp_type}.common"
                    else
                        op_name="${exp_name}"
                    fi
                    break
                fi
            done
        fi
    fi
    
    if [[ -n "${op_name}" ]]; then
        echo "${op_name}"
    fi
}

parse_ops_from_filelist() {
    local pr_filelist="$1"
    
    if [[ ! -f "${pr_filelist}" ]]; then
        print_error "pr_filelist not found: ${pr_filelist}"
        return 1
    fi
    
    local op_categories=$(get_op_categories)
    local changed_files=$(cat "${pr_filelist}" | grep -v '^$' | grep -v '^#' || echo "")
    
    if [[ -z "${changed_files}" ]]; then
        print_msg "No changed files in pr_filelist"
        return 0
    fi
    
    local ops_set=""
    while IFS= read -r file_line; do
        [[ -z "${file_line}" ]] && continue
        file_line=$(echo "${file_line}" | sed 's/^[MADRC]\t//')
        local op_name=$(extract_op_from_path "${file_line}" "${op_categories}")
        if [[ -n "${op_name}" ]]; then
            if [[ -z "${ops_set}" ]]; then
                ops_set="${op_name}"
            elif [[ ",${ops_set}," != *",${op_name},"* ]]; then
                ops_set="${ops_set},${op_name}"
            fi
        fi
    done <<< "${changed_files}"
    
    echo "${ops_set}"
}

merge_ops_lists() {
    local list1="$1"
    local list2="$2"
    local merged=""
    
    for op in ${list1//,/ }; do
        if [[ -z "${merged}" ]]; then
            merged="${op}"
        elif [[ ",${merged}," != *",${op},"* ]]; then
            merged="${merged},${op}"
        fi
    done
    
    for op in ${list2//,/ }; do
        if [[ -z "${merged}" ]]; then
            merged="${op}"
        elif [[ ",${merged}," != *",${op},"* ]]; then
            merged="${merged},${op}"
        fi
    done
    
    echo "${merged}"
}

usage() {
    echo "Usage: bash ops_st_test.sh [--soc_version=ascend950] [--ops=op1,op2,op3] [--test_type=kernel,aclnn,e2e] [--pr_filelist=pr_filelist.txt]"
    echo "       bash ops_st_test.sh pr_filelist.txt"
    echo "Options:"
    echo "    --soc_version   (Optional) Specify soc version. Supported: ascend910b, ascend950. If not specified, auto-detect via 'asys info -r=status'."
    echo "    --ops           (Optional) Specify operators to test (comma-separated). If not specified, extract from git diff."
    echo "    --test_type     (Optional) Specify test types to run (comma-separated). Supported: kernel, aclnn, e2e. Default: all types."
    echo "    --pr_filelist   (Optional) Path to file containing list of changed files (one per line). If not specified, extract from git diff."
    echo "    --case_path     (Optional) Custom base path for test cases. If specified, st_path will be {case_path}/${op_type}/${op_name}"
    echo "Examples:"
    echo "    bash ops_st_test.sh"
    echo "    bash ops_st_test.sh pr_filelist.txt"
    echo "    bash ops_st_test.sh --soc_version=ascend950"
    echo "    bash ops_st_test.sh --pr_filelist=pr_filelist.txt"
    echo "    bash ops_st_test.sh --soc_version=ascend910b --ops=mat_mul_v3,conv2d_v2"
    echo "    bash ops_st_test.sh --soc_version=ascend950 --test_type=kernel"
    echo "    bash ops_st_test.sh --soc_version=ascend950 --test_type=kernel,e2e"
}

get_changed_ops() {
    local base_branch="master"
    local changed_files
    
    changed_files=$(git diff --name-only "${base_branch}...HEAD" 2>/dev/null || git diff --name-only HEAD~1 HEAD 2>/dev/null || echo "")
    
    if [[ -z "${changed_files}" ]]; then
        print_msg "No changed files detected"
        return 0
    fi
    
    mkdir -p "${build_path}"
    cd "${build_path}"
    rm -f CMakeCache.txt
    cmake -DENABLE_EXPERIMENTAL=FALSE -DDOWNLOAD_OPS_TEST_KIT=ON -DPREPROCESS_ONLY=ON "${framework_path}" >/dev/null 2>&1 || {
        print_error "Failed to run cmake preprocess"
        return 1
    }
    cd "${framework_path}"
    
    local changed_files_tmp="${build_path}/tmp/changed_files_tmp.txt"
    mkdir -p "${build_path}/tmp"
    echo "${changed_files}" > "${changed_files_tmp}"
    
    export BASE_PATH="${framework_path}"
    export BUILD_PATH="${build_path}"
    
    local is_experimental="FALSE"
    
    local result
    result=$(python3 "${framework_path}/scripts/util/parse_compile_changed_files.py" "${changed_files_tmp}" "${is_experimental}" 2>&1)
    rm -f "${changed_files_tmp}"
    
    if [[ -z "${result}" ]]; then
        print_msg "No ops detected from changed files"
        return 0
    fi
    
    if [[ "${DEBUG_DEPENDENCIES:-}" == "TRUE" ]]; then
        local reverse_deps=$(echo "${result}" | cut -d':' -f1)
        local compile_deps=$(echo "${result}" | cut -d':' -f2)
        print_msg "Reverse dependencies (ops that depend on changed ops): ${reverse_deps}"
        print_msg "Compile dependencies (ops needed to compile): ${compile_deps}"
    fi
    
    local ops_list=$(echo "${result}" | cut -d':' -f1)
    echo "${ops_list}"
}

download_ops_test_kit() {
    print_msg "Preparing build environment..."
    
    mkdir -p "${build_path}"
    
    local ops_config_file="${build_path}/tmp/ops_config.txt"
    if [[ ! -f "${ops_config_file}" ]]; then
        print_msg "Running cmake to generate ops_config.txt..."
        [ -f "${build_path}/CMakeCache.txt" ] && rm -f "${build_path}/CMakeCache.txt"
        cd "${build_path}"
        cmake -DDOWNLOAD_OPS_TEST_KIT=ON -DPREPROCESS_ONLY=ON "${framework_path}" || {
            print_error "Failed to run cmake preprocess"
            exit 1
        }
        print_msg "ops_config.txt generated successfully"
    else
        print_msg "ops_config.txt already exists, skipping cmake"
    fi
    
    if [[ ! -d "${ttk_path}" ]]; then
        print_msg "Downloading ops-test-kit..."
        cd "${build_path}"
        cmake -DDOWNLOAD_OPS_TEST_KIT=ON -DPREPROCESS_ONLY=ON "${framework_path}" || {
            print_error "Failed to download ops-test-kit via cmake"
            exit 1
        }
        print_msg "ops-test-kit downloaded successfully"
    else
        print_msg "ops-test-kit already exists, skipping download"
    fi
    
    if [[ ! -d "${ttk_path}" ]]; then
        print_error "ttk_path does not exist after download"
        exit 1
    fi
    
    if [[ ! -f "${ops_config_file}" ]]; then
        print_error "ops_config.txt does not exist"
        exit 1
    fi
}

find_op_code_path() {
    local op_name="$1"
    local code_path=$(find "${framework_path}" -type d -name "${op_name}" -not -path "*/build/*" -not -path "*/.git/*" -not -path "*/build_out/*" | head -1)
    
    if [[ -z "${code_path}" ]]; then
        return 1
    fi
    
    echo "${code_path}"
}

get_op_type() {
    local code_path="$1"
    local subdir_path=$(realpath "${code_path}")
    local op_type=$(basename "$(dirname "${subdir_path}")")
    echo "${op_type}"
}

find_test_cases() {
    local op_name="$1"
    local op_type="$2"
    local arch="$3"
    local test_case_files=()
    
    local st_path
    if [[ -n "${case_path}" ]]; then
        st_path="${case_path}/${op_type}/${op_name}/"
    else
        st_path="${framework_path}/${op_type}/${op_name}/tests/st"
    fi
    
    if [[ ! -d "${st_path}" ]]; then
        [[ "${DEBUG_DEPENDENCIES:-}" == "TRUE" ]] && print_msg "No st test directory found for ${op_name} at ${st_path}"
        return 0
    fi
    
    local all_prefixes=("ttk_kernel" "ttk_aclnn" "ttk_e2e")
    local search_prefixes=()
    
    if [[ -n "${test_type_list}" ]]; then
        IFS=',' read -r -a input_types <<< "${test_type_list}"
        for input_type in "${input_types[@]}"; do
            # 兼容旧参数 pta，转换为 e2e
            if [[ "${input_type}" == "pta" ]]; then
                search_prefixes+=("ttk_e2e")
            else
                search_prefixes+=("ttk_${input_type}")
            fi
        done
    else
        search_prefixes=("${all_prefixes[@]}")
    fi
    
    # 第一步：查找通用用例（st/ttk_kernel_*.csv 等）
    for prefix in "${search_prefixes[@]}"; do
        local csv_files=$(find "${st_path}" -maxdepth 1 -name "${prefix}_*.csv" -type f 2>/dev/null)
        for csv_file in ${csv_files}; do
            local test_type="${prefix#ttk_}"
            test_case_files+=("${test_type}:${csv_file}")
        done
    done
    
    # 第二步：查找架构专用用例（st/arch35/ 或 st/arch22/）
    if [[ -n "${arch}" ]]; then
        local arch_path="${st_path}/${arch}"
        if [[ -d "${arch_path}" ]]; then
            for prefix in "${search_prefixes[@]}"; do
                local csv_files=$(find "${arch_path}" -maxdepth 1 -name "${prefix}_*.csv" -type f 2>/dev/null)
                for csv_file in ${csv_files}; do
                    local test_type="${prefix#ttk_}"
                    test_case_files+=("${test_type}:${csv_file}")
                done
            done
        fi
    fi
    
    echo "${test_case_files[*]}"
}

get_ops_test_path() {
    local op_name="$1"
    local op_type="$2"
    local ops_test_path="${framework_path}/${op_type}/${op_name}/tests"
    
    if [[ ! -d "${ops_test_path}" ]]; then
        print_msg "No tests directory found for ${op_name} at ${op_type}/${op_name}/tests"
        return 1
    fi
    
    echo "${ops_test_path}"
}

check_precision_status() {
    local result_csv="$1"
    local op_name="$2"
    local testcase_name="$3"
    
    if [[ ! -f "${result_csv}" ]]; then
        print_warning "Result csv file not found: ${result_csv}"
        return 1
    fi
    
    python3 "${framework_path}/scripts/ci/ops_test_util.py" \
        --action=check_precision \
        --result_csv="${result_csv}" \
        --op_name="${op_name}" \
        --testcase_name="${testcase_name}"
    
    return $?
}

check_plugin_assets() {
    local plugin_path="$1"
    local op_name="$2"
    
    local assets_path="${plugin_path}/assets"
    
    if [[ ! -d "${assets_path}" ]]; then
        print_warning "assets directory not found for ${op_name}: ${assets_path}"
        return 1
    fi
    
    local py_files=$(find "${assets_path}" -maxdepth 1 -name "*.py" -type f 2>/dev/null | head -1)
    if [[ -z "${py_files}" ]]; then
        print_warning "No .py files found in assets directory for ${op_name}: ${assets_path}"
        return 1
    fi
    
    return 0
}

run_kernel_test() {
    local op_name="$1"
    local test_csv="$2"
    local ops_test_path="$3"
    
    if [[ ! -f "${test_csv}" ]]; then
        print_warning "Test csv file not found: ${test_csv}, skipping this test case"
        return 0
    fi
    
    if [[ ! -d "${ops_test_path}" ]]; then
        print_warning "Plugin directory not found: ${ops_test_path}, skipping this test case"
        return 0
    fi
    
    if ! check_plugin_assets "${ops_test_path}" "${op_name}"; then
        return 0
    fi
    
    local testcase_name=$(basename "${test_csv}" .csv)
    local log_op_dir="${log_path}/${op_name}"
    mkdir -p "${log_op_dir}"
    
    print_msg "Running kernel test for ${op_name}, testcase: ${testcase_name}"
    
    cd "${ttk_path}"
    
    local cmd="python3 -m ttk kernel -i ${test_csv} -o ${log_op_dir}/${testcase_name}_result.csv --plugin ${ops_test_path} -c --pc=8 --warmup=false"
    print_msg "Executing: ${cmd}"
    
    local start_time=$(date +%s)
    ${cmd} 2>&1 | tee "${log_op_dir}/${testcase_name}_run.log" > /dev/null
    local test_failed=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    
    if [[ ${test_failed} -ne 0 ]]; then
        print_error "kernel test failed for ${op_name}, testcase: ${testcase_name}, elapsed: ${elapsed}s"
    else
        print_msg "kernel test completed for ${op_name}, testcase: ${testcase_name}, elapsed: ${elapsed}s"
    fi
    
    local result_csv="${log_op_dir}/${testcase_name}_result.csv"
    echo "${result_csv}"
    
    if [[ ${test_failed} -ne 0 ]]; then
        return 1
    fi
}

run_aclnn_test() {
    local op_name="$1"
    local test_csv="$2"
    local ops_test_path="$3"
    
    local start_time=$(date +%s)
    print_warning "aclnn test not implemented yet for ${op_name}"
    # TODO: 待确定aclnn测试命令后实现
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    print_msg "aclnn test elapsed: ${elapsed}s"
    return 0
}

run_e2e_test() {
    local op_name="$1"
    local test_csv="$2"
    local ops_test_path="$3"
    
    local start_time=$(date +%s)
    print_warning "e2e test not implemented yet for ${op_name}"
    # TODO: 待确定e2e测试命令后实现
    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    print_msg "e2e test elapsed: ${elapsed}s"
    return 0
}

summarize_op_results() {
    local op_name="$1"
    local test_type="$2"
    local result_csvs="$3"
    
    local summary_file="${log_path}/${test_type}_summary.csv"
    local summary_header="op_name,testcase_name,test_type,result_csv,status,precision"
    
    if [[ ! -f "${summary_file}" ]]; then
        echo "${summary_header}" > "${summary_file}"
    fi
    
    if [[ -z "${result_csvs}" ]]; then
        return 0
    fi
    
    for result_csv in ${result_csvs}; do
        if [[ ! -f "${result_csv}" ]]; then
            continue
        fi
        
        python3 "${framework_path}/scripts/ci/ops_test_util.py" \
            --action=summarize \
            --result_csv="${result_csv}" \
            --op_name="${op_name}" \
            --test_type="${test_type}" \
            --summary_file="${summary_file}"
    done
}

print_summary_table() {
    python3 "${framework_path}/scripts/ci/ops_test_util.py" \
        --action=print_table \
        --log_path="${log_path}"
}

run_single_op_test() {
    local op_name="$1"
    
    print_msg "=== Testing op: ${op_name} ==="
    
    local code_path=$(find_op_code_path "${op_name}")
    if [[ -z "${code_path}" ]]; then
        print_warning "Cannot find op directory for ${op_name}, skipping"
        return 0
    fi
    
    local op_type=$(get_op_type "${code_path}")
    print_msg "op_type: ${op_type}, op_name: ${op_name}"
    
    local arch="${SOC_TO_ST_ARCH[${soc_version}]:-}"
    if [[ -n "${arch}" ]]; then
        print_msg "soc_version: ${soc_version}, arch: ${arch}"
    fi
    
    local ops_test_path=$(get_ops_test_path "${op_name}" "${op_type}")
    if [[ -z "${ops_test_path}" ]]; then
        print_msg "No tests directory found, skipping ${op_name}"
        return 0
    fi
    
    local test_cases=$(find_test_cases "${op_name}" "${op_type}" "${arch}")
    
    if [[ -z "${test_cases}" ]]; then
        print_msg "No test cases found for ${op_name}"
        return 0
    fi
    
    local result_csvs=()
    local kernel_csvs=()
    local aclnn_csvs=()
    local e2e_csvs=()
    local test_case_array=(${test_cases})
    local result_csv
    local testcase_name
    local op_error_flag=0
    
    for test_item in "${test_case_array[@]}"; do
        local test_type=$(echo "${test_item}" | cut -d':' -f1)
        local test_csv=$(echo "${test_item}" | cut -d':' -f2-)
        local test_ret=0
        
        case "${test_type}" in
            kernel)
                result_csv=$(run_kernel_test "${op_name}" "${test_csv}" "${ops_test_path}")
                test_ret=$?
                if [[ ${test_ret} -ne 0 ]]; then
                    op_error_flag=1
                fi
                if [[ -n "${result_csv}" ]]; then
                    kernel_csvs+=("${result_csv}")
                fi
                ;;
            aclnn)
                result_csv=$(run_aclnn_test "${op_name}" "${test_csv}" "${ops_test_path}")
                test_ret=$?
                if [[ ${test_ret} -ne 0 ]]; then
                    op_error_flag=1
                fi
                if [[ -n "${result_csv}" ]]; then
                    aclnn_csvs+=("${result_csv}")
                fi
                ;;
            e2e)
                result_csv=$(run_e2e_test "${op_name}" "${test_csv}" "${ops_test_path}")
                test_ret=$?
                if [[ ${test_ret} -ne 0 ]]; then
                    op_error_flag=1
                fi
                if [[ -n "${result_csv}" ]]; then
                    e2e_csvs+=("${result_csv}")
                fi
                ;;
            *)
                print_warning "Unknown test type: ${test_type}, skipping"
                continue
                ;;
        esac
        
        if [[ -n "${result_csv}" ]]; then
            result_csvs+=("${result_csv}")
        fi
    done
    
    summarize_op_results "${op_name}" "kernel" "${kernel_csvs[*]}"
    summarize_op_results "${op_name}" "aclnn" "${aclnn_csvs[*]}"
    summarize_op_results "${op_name}" "e2e" "${e2e_csvs[*]}"
    
    for csv in "${result_csvs[@]}"; do
        testcase_name=$(basename "${csv}" _result.csv)
        echo "${op_name}:${testcase_name}:${csv}"
    done
    
    if [[ ${op_error_flag} -ne 0 ]]; then
        return 1
    fi
}

parse_args() {
    ops_list=""
    soc_version=""
    test_type_list=""
    pr_filelist=""
    case_path=""
    
    for arg in "$@"; do
        case "${arg}" in
            --ops=*)
                ops_list="${arg#*=}"
                ;;
            --soc_version=*)
                soc_version="${arg#*=}"
                ;;
            --test_type=*)
                test_type_list="${arg#*=}"
                ;;
            --pr_filelist=*)
                pr_filelist="${arg#*=}"
                ;;
            --case_path=*)
                case_path="${arg#*=}"
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            -*)
                print_error "Unknown argument: ${arg}"
                usage
                exit 1
                ;;
            *)
                if [[ -z "${pr_filelist}" ]]; then
                    pr_filelist="${arg}"
                else
                    print_error "Multiple pr_filelist arguments: ${pr_filelist} and ${arg}"
                    usage
                    exit 1
                fi
                ;;
        esac
    done
    
    if [[ -n "${pr_filelist}" && ! -f "${pr_filelist}" ]]; then
        print_error "pr_filelist not found: ${pr_filelist}"
        exit 1
    fi
    
    if [[ -n "${test_type_list}" ]]; then
        IFS=',' read -r -a valid_types <<< "kernel,aclnn,e2e"
        IFS=',' read -r -a input_types <<< "${test_type_list}"
        for input_type in "${input_types[@]}"; do
            local found=0
            for valid_type in "${valid_types[@]}"; do
                if [[ "${input_type}" == "${valid_type}" ]]; then
                    found=1
                    break
                fi
            done
            if [[ ${found} -eq 0 ]]; then
                print_error "Unsupported test_type: ${input_type}. Supported: kernel, aclnn, e2e"
                exit 1
            fi
        done
    fi
    
    local chip_info=$(asys info -r=status 2>/dev/null || echo "")
    local detected_soc=""
    if echo "${chip_info}" | grep -q "Ascend 950"; then
        detected_soc="ascend950"
    elif echo "${chip_info}" | grep -q "Ascend 910"; then
        detected_soc="ascend910b"
    fi
    
    if [[ -z "${soc_version}" ]]; then
        if [[ -z "${detected_soc}" ]]; then
            print_error "Failed to detect SOC version via 'asys info -r=status'. Current environment does not support auto-detection. Please specify --soc_version manually."
            exit 1
        fi
        soc_version="${detected_soc}"
        print_msg "Auto-detected soc_version: ${soc_version}"
    else
        if [[ "${soc_version}" != "ascend910b" && "${soc_version}" != "ascend950" ]]; then
            print_error "Unsupported soc_version: ${soc_version}. Supported: ascend910b, ascend950"
            exit 1
        fi
        
        if [[ -n "${detected_soc}" && "${detected_soc}" != "${soc_version}" ]]; then
            print_error "SOC version mismatch: specified '${soc_version}' but detected '${detected_soc}' from 'asys info -r=status'"
            exit 1
        fi
    fi
    
    print_msg "soc_version: ${soc_version}"
    print_msg "ops_list: ${ops_list:-'auto detect from git diff or pr_filelist'}"
    print_msg "test_type_list: ${test_type_list:-'all types'}"
    if [[ -n "${pr_filelist}" ]]; then
        print_msg "pr_filelist: ${pr_filelist}"
    fi
    if [[ -n "${case_path}" ]]; then
        print_msg "case_path: ${case_path}"
    fi
}

framework_path="$(cd "$(dirname "$0")/../.." && pwd)"
build_path="${framework_path}/build"
log_path="${framework_path}/st/log"
ttk_path="${build_path}/third_party/ops-test-kit"

parse_args "$@"

rm -rf "${log_path:?}"/*
mkdir -p "${log_path}"

download_ops_test_kit

if [[ -n "${ops_list}" && -z "${pr_filelist}" ]]; then
    print_msg "Using ops from --ops parameter: ${ops_list}"
elif [[ -z "${ops_list}" && -n "${pr_filelist}" ]]; then
    print_msg "Extracting ops from pr_filelist..."
    print_msg "pr_filelist content:"
    cat "${pr_filelist}" | grep -v '^$' | grep -v '^#' | sed 's/^[MADRC]\t//' >&2
    ops_list=$(parse_ops_from_filelist "${pr_filelist}")
elif [[ -n "${ops_list}" && -n "${pr_filelist}" ]]; then
    print_msg "Merging ops from pr_filelist and --ops parameter..."
    print_msg "--ops input: ${ops_list}"
    print_msg "pr_filelist content:"
    cat "${pr_filelist}" | grep -v '^$' | grep -v '^#' | sed 's/^[MADRC]\t//' >&2
    ops_from_filelist=$(parse_ops_from_filelist "${pr_filelist}")
    ops_list=$(merge_ops_lists "${ops_from_filelist}" "${ops_list}")
else
    print_msg "Extracting ops from git diff..."
    ops_list=$(get_changed_ops)
    ops_list=$(echo "${ops_list}" | tr ';' ',')
fi

if [[ -z "${ops_list}" ]]; then
    print_msg "No ops to test"
    exit 0
fi

print_msg "Ops to test: ${ops_list}"

IFS=',' read -r -a op_name_array <<< "${ops_list}"

source /usr/local/Ascend/cann/bin/setenv.bash 2>/dev/null || true

all_result_csvs=()
result_flag=0
for op_name in "${op_name_array[@]}"; do
    op_results=$(run_single_op_test "${op_name}") || result_flag=1
    if [[ -n "${op_results}" ]]; then
        while IFS= read -r line; do
            all_result_csvs+=("${line}")
        done <<< "${op_results}"
    fi
done

print_msg "=== Starting precision check for all test cases ==="
precision_flag=0
for result_info in "${all_result_csvs[@]}"; do
    op_name=$(echo "${result_info}" | cut -d':' -f1)
    testcase_name=$(echo "${result_info}" | cut -d':' -f2)
    result_csv=$(echo "${result_info}" | cut -d':' -f3)
    check_precision_status "${result_csv}" "${op_name}" "${testcase_name}" || precision_flag=1
done

print_summary_table

if [[ ${result_flag} -ne 0 ]]; then
    print_error "Some tests failed, please check the log for details."
    exit 1
elif [[ ${precision_flag} -ne 0 ]]; then
    print_error "Some precision checks failed, please check the details above."
    exit 1
else
    print_success "All tests and precision checks passed."
    exit 0
fi