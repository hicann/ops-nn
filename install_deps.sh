#!/bin/bash
# ============================================================================
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

set -euo pipefail

run_command() {
    local cmd="$*"
    echo "Executing command: $cmd"

    if ! output=$("$@" 2>&1); then
         local exit_code=$?
         echo -e "\nCommand execution failed!"
         echo -e "\nFailed command: $cmd"
         echo -e "\nError output: $output"
         echo -e "\nExit code: $exit_code"
         exit $exit_code
    fi
}

version_ge() {
    # Version comparison, format: xx.xx.xx
    IFS='.' read -r -a curr_arr <<< "$1"
    IFS='.' read -r -a req_arr <<< "$2"

    for ((i=0; i<${#req_arr[@]}; i++)); do
        curr=${curr_arr[i]:-0}
        req=${req_arr[i]}
        if (( curr > req )); then
            return 0
        elif (( curr < req )); then
            return 1
        fi
    done
    return 0
}

detect_os() {
    # OS detection, supports debian (uses apt), rhel (uses dnf or yum), macos
    if [[ "$(uname -s)" == "Linux" ]]; then
        if [[ -f /etc/debian_version ]]; then
            OS="debian"
            PKG_MANAGER="apt"
        elif [[ -f /etc/redhat-release ]]; then
            OS="rhel"
            if command -v dnf &> /dev/null; then
                PKG_MANAGER="dnf"
            else
                PKG_MANAGER="yum"
            fi
        elif grep -qE '^NAME="openEuler"$|^NAME="EulerOS"$' /etc/os-release 2>/dev/null; then
            OS="euler"
            PKG_MANAGER="dnf"
        else
            echo "Unsupported Linux distribution, please install manually"
            exit 1
        fi
    elif [[ "$(uname -s)" == "Darwin" ]]; then
        OS="macos"
        if ! command -v brew &> /dev/null; then
            echo "Please install Homebrew first"
            exit 1
        fi
        PKG_MANAGER="brew"
    else
        echo "Unsupported OS type, please install manually"
        exit 1
    fi
}

install_gawk() {
    echo -e "\n==== Checking gawk ===="

    if command -v gawk &> /dev/null; then
        echo "gawk has been installed"
        return
    fi

    echo "Installing gawk..."
    case "$OS" in
        debian)
            run_command sudo $PKG_MANAGER update
            run_command sudo $PKG_MANAGER install -y gawk
            ;;
        rhel|euler)
            run_command sudo $PKG_MANAGER install -y gawk
            ;;
        macos)
            run_command brew install gawk
            ;;
    esac

    if command -v gawk &> /dev/null; then
        echo "gawk installed successfully"
    else
        echo "gawk installation failed"
        exit 1
    fi
}

install_python() {
    # Python version >= 3.7.0
    echo -e "\n==== Checking Python ===="
    local req_ver="3.7.0"
    local curr_ver=""

    if command -v python3 &> /dev/null; then
        curr_ver=$(python3 --version 2>&1 | awk '{print $2}')
        echo "Current Python version: $curr_ver"
        if version_ge "$curr_ver" "$req_ver"; then
            echo "Python version meets requirements"
            return
        fi
    fi
    echo "Installing Python..."
    case "$OS" in
        debian)
            run_command sudo $PKG_MANAGER update
            run_command sudo $PKG_MANAGER install -y python3 python3-pip python3-dev
            ;;
        rhel)
            if grep -q "release 7" /etc/redhat-release; then
                run_command sudo $PKG_MANAGER install -y centos-release-scl
                run_command sudo $PKG_MANAGER install -y rh-python38 rh-python38-python-devel
                run_command source /opt/rh/rh-python38/enable
                echo "Need to execute 'source /opt/rh/rh-python38/enable' to activate python3.8"
            else
                run_command sudo $PKG_MANAGER install -y python3 python3-pip python3-devel
            fi
            ;;
        macos)
            run_command brew install python@3.10
            echo 'export PATH="/usr/local/opt/python@3.10/bin:$PATH"' >> ~/.zshrc
            run_command source ~/.zshrc
            ;;
        euler)
            run_command sudo $PKG_MANAGER install -y python3 python3-pip python3-devel
            ;;
    esac

    if command -v python3 &> /dev/null; then
        curr_ver=$(python3 --version 2>&1 | awk '{print $2}')
        if version_ge "$curr_ver" "$req_ver"; then
            echo "Python installed successfully ($curr_ver)"
        else
            echo "Python version still doesn't meet requirements, please install manually"
            exit 1
        fi
    else
        echo "Python installation failed"
        exit 1
    fi
}

install_gcc() {
    # GCC version >= 7.3.0
    echo -e "\n==== Checking GCC ===="
    local req_ver="7.3.0"
    local curr_ver=""

    if command -v gcc &> /dev/null; then
        curr_ver=$(gcc --version | awk '/^gcc/ {print $NF}')
    elif command -v g++ &> /dev/null; then
        curr_ver=$(g++ --version | awk '/^g\+\+/ {print $NF}')
    else
        curr_ver="0.0.0"
    fi
    echo "Current GCC version: $curr_ver"
    if version_ge "$curr_ver" "$req_ver"; then
        echo "GCC version meets requirements ($curr_ver)"
        return
    fi

    echo "Installing GCC..."
    case "$OS" in
        debian)
            run_command sudo $PKG_MANAGER update
            run_command sudo $PKG_MANAGER install -y gcc-9 g++-9
            run_command sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 \
                --slave /usr/bin/g++ g++ /usr/bin/g++-9
            ;;
        rhel)
            if grep -q "release 7" /etc/redhat-release; then
                run_command sudo $PKG_MANAGER install -y centos-release-scl
                run_command sudo $PKG_MANAGER install -y devtoolset-9-gcc devtoolset-9-gcc-c++
                run_command source /opt/rh/devtoolset-9/enable
                echo "Need to execute 'source /opt/rh/devtoolset-9/enable' to activate GCC9"
            else
                run_command sudo $PKG_MANAGER install -y gcc gcc-c++
            fi
            ;;
        macos)
            if ! xcode-select -p &> /dev/null; then
                xcode-select --install
            fi
            run_command brew install gcc@11
            echo 'export CC=/usr/local/bin/gcc-11' >> ~/.zshrc
            echo 'export CXX=/usr/local/bin/g++-11' >> ~/.zshrc
            run_command source ~/.zshrc
            ;;
        euler)
            run_command sudo $PKG_MANAGER install -y gcc gcc-c++
            ;;
    esac

    if command -v gcc &> /dev/null; then
        curr_ver=$(gcc --version | awk '/^gcc/ {print $NF}')
        if version_ge "$curr_ver" "$req_ver"; then
            echo "GCC installed successfully ($curr_ver)"
        else
            echo "GCC version still doesn't meet requirements, please install manually."
            exit 1
        fi
    else
        echo "GCC installation failed"
        exit 1
    fi
}

install_cmake() {
    # CMake version >= 3.16.0
    echo -e "\n==== Checking CMake ===="
    local req_ver="3.16.0"
    local curr_ver=""

    if command -v cmake &> /dev/null; then
        curr_ver=$(cmake --version | awk '/^cmake/ {print $3}')
        echo "Current CMake version: $curr_ver"
        if version_ge "$curr_ver" "$req_ver"; then
            echo "CMake meets requirements"
            return
        fi
    fi

    echo "Installing CMake..."
    case "$OS" in
        debian)
            if grep -q "Ubuntu 18.04" /etc/os-release; then
                run_command wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
                run_command echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ bionic main' | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null
                run_command sudo apt update
                run_command sudo apt install -y cmake make
            else
                run_command sudo $PKG_MANAGER update
                run_command sudo $PKG_MANAGER install -y cmake make
            fi
            ;;
        rhel)
            if grep -q "release 7" /etc/redhat-release; then
                run_command sudo $PKG_MANAGER install -y epel-release
                run_command sudo $PKG_MANAGER install -y cmake3 make
                run_command sudo ln -s /usr/bin/cmake3 /usr/bin/cmake
            else
                run_command sudo $PKG_MANAGER install -y cmake make
            fi
            ;;
        macos)
            run_command brew install cmake
            ;;
        euler)
            run_command sudo $PKG_MANAGER install -y cmake make
            ;;
    esac

    if command -v cmake &> /dev/null; then
        curr_ver=$(cmake --version | awk '/^cmake/ {print $3}')
        if version_ge "$curr_ver" "$req_ver"; then
            echo "CMake installed successfully ($curr_ver)"
        else
            echo "CMake version still doesn't meet requirements, please install manually"
            exit 1
        fi
    else
        echo "CMake installation failed"
        exit 1
    fi
}

install_pigz() {
    # pigz version >= 2.4
    echo -e "\n==== Checking pigz ===="
    local req_ver="2.4"
    local curr_ver=""

    if command -v pigz &> /dev/null; then
        curr_ver=$(pigz --version 2>&1 | awk '{print $2}')
        echo "Current pigz version: $curr_ver"
        if version_ge "$curr_ver" "$req_ver"; then
            echo "pigz meets requirements"
            return
        fi
    fi

    read -p "Install pigz? [Y/n] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping pigz installation"
        return
    fi

    echo "Installing pigz..."
    case "$OS" in
        debian|rhel|euler)
            run_command sudo $PKG_MANAGER install -y pigz
            ;;
        macos)
            run_command brew install pigz
            ;;
    esac

    if command -v pigz &> /dev/null; then
        curr_ver=$(pigz --version 2>&1 | awk '{print $2}')
        echo "pigz installed successfully ($curr_ver)"
    else
        echo "pigz installation failed, can be ignored"
    fi
}

install_dos2unix() {
    echo -e "\n==== Checking dos2unix ===="

    if command -v dos2unix &> /dev/null; then
        echo "dos2unix has been installed"
        return
    fi

    echo "Installing dos2unix..."
    case "$OS" in
        debian|rhel|euler)
            run_command sudo $PKG_MANAGER install -y dos2unix
            ;;
        macos)
            run_command brew install dos2unix
            ;;
    esac

    if command -v dos2unix &> /dev/null; then
        echo "dos2unix installed successfully"
    else
        echo "dos2unix installation failed"
        exit 1
    fi
}

check_dependencies_silent() {
    local args=("$@")
    local check_pkgz="false"
    local check_dos2unix="false"

    for arg in "${args[@]}"; do
        case "$arg" in
            --pkg)
                check_pkgz="true"
                check_dos2unix="true"
                ;;
            --opkernel)
                check_dos2unix="true"
                ;;
        esac
    done

    local missing_deps=()
    declare -A req_versions
    req_versions["gawk"]=""
    req_versions["Python"]="3.7.0"
    req_versions["GCC"]="7.3.0"
    req_versions["CMake"]="3.16.0"
    req_versions["pigz"]="2.4"
    req_versions["dos2unix"]=""

    check_deps() {
        local name="$1"
        local cmd="$2"
        local req_ver="$3"

        if ! command -v "$cmd" &> /dev/null; then
            missing_deps+=("$name")
            return
        fi

        if [[ -n "$req_ver" ]]; then
            local curr_ver=""
            case "$cmd" in
                python3)
                    curr_ver=$(python3 --version 2>&1 | awk '{print $2}')
                    ;;
                gcc|g++)
                    curr_ver=$(gcc --version | awk '/^gcc/ {print $NF}') 
                    ;;
                cmake)
                    curr_ver=$(cmake --version | awk '/^cmake/ {print $3}') 
                    ;;
                pigz)
                    curr_ver=$(pigz --version 2>&1 | awk '{print $2}') 
                    ;;
            esac
        
            if [[ -z "$curr_ver" ]] || ! version_ge "$curr_ver" "$req_ver"; then
                missing_deps+=("$name")
            fi
        fi
    }

    check_deps "gawk" "gawk" "${req_versions["gawk"]}"
    check_deps "Python" "python3" "${req_versions["Python"]}"
    check_deps "GCC" "gcc" "${req_versions["GCC"]}"
    check_deps "CMake" "cmake" "${req_versions["CMake"]}"
    if [[ "$check_dos2unix" == "true" ]]; then
        check_deps "dos2unix" "dos2unix" "${req_versions["dos2unix"]}"
    fi
    if [[ "$check_pkgz" == "true" ]]; then
        check_deps "pigz" "pigz" "${req_versions["pigz"]}"
    fi

    if [[ ${#missing_deps[@]} -eq 0 ]]; then
        return 0
    else
        echo -e "\n Missing dependencies:"
        for dep in "${missing_deps[@]}"; do
            local req_ver="${req_versions[$dep]}"
            if [[ -n "$req_ver" ]]; then
                echo " - $dep (required: >= $req_ver)"
            else
                echo " - $dep"
            fi
        done
        echo -e "\n Please run:"
        echo -e "\n    bash install_deps.sh\n"
        echo -e "    to install all missing dependencies."
        echo -e "    After installation, re-run this script.\n"
        return 1
    fi
}

main() {
    echo "===================================================="
    echo "Starting project dependency installation"
    echo "===================================================="

    detect_os
    install_gawk
    install_python
    install_gcc
    install_cmake
    install_pigz
    install_dos2unix

    echo -e "===================================================="
    echo "All dependencies installed successfully!"
    echo "===================================================="
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi