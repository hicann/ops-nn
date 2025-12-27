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

_CURR_OPERATE_USER="$(id -nu 2> /dev/null)"
_CURR_OPERATE_GROUP="$(id -ng 2> /dev/null)"
_DEFAULT_USERNAME="root"
_DEFAULT_USERGROUP="root"
_DEFAULT_INSTALL_PATH="/usr/local/Ascend"
# defaults for general user
if [ "$(id -u)" != "0" ]; then
    _DEFAULT_USERNAME="${_CURR_OPERATE_USER}"
    _DEFAULT_USERGROUP="${_CURR_OPERATE_GROUP}"
    _DEFAULT_INSTALL_PATH="${HOME}/Ascend"
fi

# run package's files info
_CURR_PATH=$(dirname $(readlink -f $0))
_FILELIST_FILE="${_CURR_PATH}""/filelist.csv"
_INSTALL_SHELL_FILE="${_CURR_PATH}""/opp_install.sh"
_UPGRADE_SHELL_FILE="${_CURR_PATH}""/opp_upgrade.sh"
_RUN_PKG_INFO_FILE="${_CURR_PATH}""/../scene.info"
_VERSION_INFO_FILE="${_CURR_PATH}""/../version.info"
_COMMON_INC_FILE="${_CURR_PATH}""/common_func.inc"
_VERCHECK_FILE="${_CURR_PATH}""/ver_check.sh"
_PRE_CHECK_FILE="${_CURR_PATH}""/../bin/prereq_check.bash"
VERSION_COMPAT_FUNC_PATH="${_CURR_PATH}/version_compatiable.inc"
common_func_v2_path="${_CURR_PATH}/common_func_v2.inc"
version_cfg_path="${_CURR_PATH}/version_cfg.inc"
_OPP_COMMON_FILE="${_CURR_PATH}/opp_common.sh"
. "${VERSION_COMPAT_FUNC_PATH}"
. "${_COMMON_INC_FILE}"
. "${common_func_v2_path}"
. "${version_cfg_path}"
. "${_OPP_COMMON_FILE}"
platform_data=$(grep -e "arch" "$_RUN_PKG_INFO_FILE" | cut --only-delimited -d"=" -f2-)
opp_old_platform_dir=ops_nn_$platform_data-linux
opp_platform_dir=ops_nn
upper_opp_platform=$(echo "${opp_platform_dir}" | tr 'a-z' 'A-Z')
# defaluts info determinated by user's inputs
_INSTALL_LOG_DIR="ops_nn/install_log"
_INSTALL_INFO_SUFFIX="${opp_platform_dir}/ascend_install.info"
_VERSION_INFO_SUFFIX="${opp_platform_dir}/version.info"
_TARGET_INSTALL_PATH=""
_TARGET_USERNAME=""
_TARGET_USERGROUP=""
_PRECHECK_PATH=""
_PRECHECK_BASH_INFO=""
_PRECHECK_CSH_INFO=""
_PRECHECK_FISH_INFO=""
_OPP_PRECHECK_PATH=""
_OPP_PRECHECK_BASH_CMD=""
_OPP_PRECHECK_CSH_CMD=""
_OPP_PRECHECK_FISH_CMD=""

# error number and description
OPERATE_FAILED="0x0001"
PARAM_INVALID="0x0002"
FILE_NOT_EXIST="0x0080"
FILE_NOT_EXIST_DES="File not found."
FILE_WRITE_FAILED="0x0081"
FILE_WRITE_FAILED_DES="File write failed."
FILE_READ_FAILED="0x0082"
FILE_READ_FAILED_DES="File read failed."
FILE_REMOVE_FAILED="0x0090"
FILE_REMOVE_FAILED_DES="Failed to remove file."
UNAME_NOT_EXIST="0x0091"
UNAME_NOT_EXIST_DES="Username not exists."
OPP_COMPATIBILITY_CEHCK_ERR="0x0092"
OPP_COMPATIBILITY_CEHCK_ERR_DES="OpsNN compatibility check error."
PERM_DENIED="0x0093"
PERM_DENIED_DES="Permission denied."

# log functions
# start info before shell executing
startlog() {
    echo "[OpsNN] [$(getdate)] [INFO]: Start Time: $(getdate)"
}

exitlog() {
    echo "[OpsNN] [$(getdate)] [INFO]: End Time: $(getdate)"
}

checkgroupvalidwithuser(){
    _ugroup="$1"
    _uname_param="$2"
    if [ $(groups "${_uname_param}" | grep "${_uname_param} :" -c) -eq 1 ]; then
        _related=$(groups "${_uname_param}" 2> /dev/null |awk -F":" '{print $2}'|grep -w "${_ugroup}")
    else
        _related=$(groups "${_uname_param}" 2> /dev/null |grep -w "${_ugroup}")
    fi
    if [ "${_related}" != "" ];then
        return 0
    else
        return 1
    fi
}

getinstallpath() {
    docker_root_tmp="$(echo "${docker_root}" | sed "s#/\+\$##g")"
    docker_root_regex="$(echo "${docker_root_tmp}" | sed "s#\/#\\\/#g")"
    relative_path=$(echo "${target_dir}" | sed "s/^${docker_root_regex}//g" | sed "s/\/\+\$//g")
    return
}

# check user name and user group is valid or not
checkusergrouprelationvalid() {
    _uname_val="$1"
    _ugroup_val="$2"
    if [ "$_uname_val" = "" ] || [ "$_ugroup_val" = "" ]; then
        logandprint "[ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:Input empty username or usergroup is invalid."
        return 1
    fi
    checkgroupvalidwithuser "${_ugroup_val}" "${_uname_val}"
    if [ $? -ne 0 ];then
        logandprint "[ERROR]: ERR_NO:${UNAME_NOT_EXIST};ERR_DES:Usergroup ${_ugroup_val} \
not right! Please check the relatianship of user ${_uname_val} and the group ${_ugroup_val}."
        return 1
    fi
    return 0
}


checkemtpyuser() {
    _cmd_type="$1"
    _uname_value="$2"
    _ugroup_value="$3"
    if [ "${_uname_value}" != "" ] || [ "${_ugroup_value}" != "" ]; then
        logandprint "[ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:Operation of \
${_cmd_type} ops_nn not support specific user name or user group. Please \
use [--help] to see the useage."notepadd
        return 1
    fi
}

logoperationretstatus() {
    _cmd_type_param="$1"
    _install_type="$2"
    _ret_status="$3"
    _cmd_list="$*"
    _cmd_list=$(echo $_cmd_list | awk -F" " '{$1="";$2="";$3=""; print $0}' | awk -F"   " '{print $2}')

    _event_level="SUGGESTION"
    if [ "${_cmd_type_param}" = "upgrade" ]; then
        _event_level="MINOR"
    elif [ "${_cmd_type_param}" = "uninstall" ]; then
        _event_level="MAJOR"
    fi
    _ret_status_des="success"
    if [ "${_ret_status}" != 0 ]; then
        _ret_status_des="failed"
    fi

    _curr_user="${_CURR_OPERATE_USER}"
    _curr_ip="127.0.0.1"
    _pkg_name="OpsNN"
    _cur_date_res=$(getdate)
    echo "Install ${_event_level} ${_curr_user} ${_cur_date_res} ${_curr_ip} \
${_pkg_name} ${_ret_status_des} install_type=${_install_type}; \
cmdlist=${_cmd_list}."
    if [ -f "${_OPERATE_LOG_FILE}" ]; then
        echo "Install ${_event_level} ${_curr_user} ${_cur_date_res} ${_curr_ip} \
${_pkg_name} ${_ret_status_des} install_type=${_install_type}; \
cmdlist=${_cmd_list}." >> "${_OPERATE_LOG_FILE}"
    else
        echo "[WARNING]: Operation log file not exist."
    fi

    if [ "${_ret_status}" != 0 ]; then
        exitlog
        exit 1
    else
        exitlog
        exit 0
    fi
}

#check ascend_install.info for the change in code warning
checkascendinfo() {
    file_param="${target_dir}/${_INSTALL_INFO_SUFFIX}"
    if [ -f "${file_param}" ]; then
        inst_type=$(cat ${file_param} | grep "Opp_Install_Type" | awk -F = '{print $2}')
        uname_param=$(cat ${file_param} | grep "UserName" | awk -F = '{print $2}')
        ugroup_param=$(cat ${file_param} | grep "UserGroup" | awk -F = '{print $2}')
        path_param=$(cat ${file_param} | grep "Opp_Install_path_Param" | awk -F = '{print $2}')
        path_params=$(cat ${file_param} | grep "Opp_Install_Path_Param" | awk -F = '{print $2}')
        version_param=$(cat ${file_param} | grep "Opp_Version" | awk -F = '{print $2}')
        if [ "$inst_type" != "" ]; then
            echo "${upper_opp_platform}_INSTALL_TYPE=${inst_type}" >> ${file_param}
        fi
        if [ "$uname_param" != "" ]; then
            echo "USERNAME=${uname_param}" >> ${file_param}
        fi
        if [ "$ugroup_param" != "" ]; then
            echo "USERGROUP=${ugroup_param}" >> ${file_param}
        fi
        if [ "$path_param" != "" ]; then
            echo "${upper_opp_platform}_INSTALL_PATH_VAL=${path_param}" >> ${file_param}
        fi
        if [ "$path_params" != "" ]; then
            echo "${upper_opp_platform}_INSTALL_PATH_PARAM=${path_params}" >> ${file_param}
        fi

        if [ "$version_param" != "" ]; then
            echo "${upper_opp_platform}_VERSION=${version_param}" >> ${file_param}
        fi
    fi
}

# keys of infos in ascend_install.info
KEY_INSTALLED_UNAME="USERNAME"
KEY_INSTALLED_UGROUP="USERGROUP"
KEY_INSTALLED_TYPE="${upper_opp_platform}_INSTALL_TYPE"
KEY_INSTALLED_PATH="${upper_opp_platform}_INSTALL_PATH_VAL"
KEY_INSTALLED_PATH_BUGFIX="${upper_opp_platform}_INSTALL_PATH_PARAM"
KEY_INSTALLED_VERSION="${upper_opp_platform}_VERSION"
KEY_INSTALLED_FEATURE="${upper_opp_platform}_Install_Feature"
KEY_INSTALLED_CHIP="${upper_opp_platform}_INSTALL_CHIP"

getinstalledinfo() {
    _key="$1"
    _res=""
    if [ -f "${_INSTALL_INFO_FILE}" ]; then
        chmod 644 "${_INSTALL_INFO_FILE}"> /dev/null 2>&1
        checkascendinfo
        case "${_key}" in
        USERNAME)
            res=$(cat ${_INSTALL_INFO_FILE} | grep "USERNAME" | awk -F = '{print $2}')
            ;;
        USERGROUP)
            res=$(cat ${_INSTALL_INFO_FILE} | grep "USERGROUP" | awk -F = '{print $2}')
            ;;
        ${upper_opp_platform}_INSTALL_TYPE)
            type="INSTALL_TYPE"
            res=$(cat ${_INSTALL_INFO_FILE} | grep "${type}" | awk -F = '{print $2}')
            ;;
        ${upper_opp_platform}_INSTALL_CHIP)
            chip="INSTALL_CHIP"
            res=$(cat ${_INSTALL_INFO_FILE} | grep "${chip}" | awk -F = '{print $2}')
            ;;
        ${upper_opp_platform}_INSTALL_PATH_VAL)
            val="INSTALL_PATH_VAL"
            res=$(cat ${_INSTALL_INFO_FILE} | grep ${val} | awk -F = '{print $2}')
            ;;
        ${upper_opp_platform}_VERSION)
            version="VERSION"
            res=$(cat ${_INSTALL_INFO_FILE} | grep ${version} | awk -F = '{print $2}')
            ;;
        ${upper_opp_platform}_INSTALL_PATH_PARAM)
            param="INSTALL_PATH_PARAM"
            res=$(cat ${_INSTALL_INFO_FILE} | grep ${param} | awk -F = '{print $2}')
            ;;
        esac
    fi
    echo "${res}"
}

# keys of infos in run package
KEY_RUNPKG_VERSION="Version"
getrunpkginfo() {
    _key_param="$1"
    if [ -f "${_VERSION_INFO_FILE}" ]; then
        . "${_VERSION_INFO_FILE}" 2> /dev/null
        case "${_key_param}" in
        Version)
            echo ${Version}
            ;;
        esac
    fi
}

precleanbeforeinstall() {
    _path="$1"
    if [ "${_path}" = "" ]; then
        logandprint "[ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:Input empty path is invalid."
        return 1
    fi
    if [ "$pkg_is_multi_version" = "true" ]; then
        _opp_sub_dir="${_path}/$pkg_version_dir""/share/info/${opp_platform_dir}"
    else
        _opp_sub_dir="${_path}""/${opp_platform_dir}"
    fi
    _installed_path=$(getinstalledinfo "${KEY_INSTALLED_PATH}")
    _files_existed=1
    # check the installation folder has files or ops_nn module existed or not
    _existed_files=$(find ${_opp_sub_dir} -path ${_opp_sub_dir}/aicpu -prune -o -type f -print 2> /dev/null)
    _arrs="aicpu Ascend310 Ascend910 Ascend310P Ascend"

    for i in ${_arrs}
    do
        _existed_files=$(find ${_opp_sub_dir} -path ${_opp_sub_dir}/${i} -prune -o -type f -print 2> /dev/null)
        if [ "${_existed_files}" != "" ]; then
            break
        fi
    done

    if [ "${_existed_files}" = "" ]; then
        _files_existed=1
    else
        _files_existed=0
        logandprint "[WARNING]: Install folder has files existed. Some files are listed below:"
	_ret_array=$(echo ${_existed_files}|awk '{split($0,arr," ");for(i in arr) print arr[i]}')
	for i in $_ret_array;do id=$((${id:=-1}+1)); eval _ret_array_$id=$i;done
	for idx in $(seq 0 5); do
	    if [ "$(eval echo '$'_ret_array_$idx)" != "" ]; then
		    logandprint "[WARNING]: file: "$(eval echo '$'_ret_array_$idx)""
            fi
        done
    fi

    if [ "${_files_existed}" = "0" ]; then
        if [ "${is_quiet}" = y ]; then
            logandprint "[WARNING]: Directory has file existed or installed ops_nn \
module, are you sure to keep installing ops_nn module in it? y"
        else
            if [ ! -f  "${_opp_sub_dir}""/ascend_install.info" ]; then
                logandprint "[INFO]: Directory has file existed, do you want to continue? [y/n]"
            else
                logandprint "[INFO]: OpsNN package has been installed on the path $(getinstalledinfo "${KEY_INSTALLED_PATH}"), \
the version is $(getinstalledinfo "${KEY_INSTALLED_VERSION}"), \
and the version of this package is $(getrunpkginfo "${KEY_RUNPKG_VERSION}"), do you want to continue? [y/n]"
            fi
            while true
            do
                read yn
                if [ "$yn" = n ]; then
                    logandprint "[INFO]: Exit to install ops_nn module."
                    exitlog
                    exit 0
                elif [ "$yn" = y ]; then
                    break;
                else
                    echo "[WARNING]: Input error, please input y or n to choose!"
                fi
            done
        fi
    else
        logandprint "[INFO]: Directory is empty, directly install ops_nn module."
    fi

    ret=0
    if [ $(id -u) -eq 0 ]; then
        parent_dirs_permission_check "${_path}" && ret=$? || ret=$?
        if [ "${is_quiet}" = "y" ]; then
            if [ ${ret} -ne 0 ]; then
                logandprint "[ERROR]: ERR_NO:0x0095;ERR_DES:the given dir, or its parents, permission is invalid. \
Please install without quiet mode and check permission."
                exitlog
                exit 1
            fi
        else
            if [ ${ret} -ne 0 ]; then
                logandprint "[WARNING]: You are going to put run-files on a unsecure install-path, do you want to continue? [y/n]"
                while true
                do
                    read yn
                    if [ "$yn" = n ]; then
                        exit 1
                    elif [ "$yn" = y ]; then
                        break;
                    else
                        echo "[ERROR]: ERR_NO:0x0002;ERR_DES:input error, please input again!"
                    fi
                done
            fi
        fi
    fi

    if [ "${_files_existed}" = "0" ] && [ "${_installed_path}" = "${_path}" ]; then
        logandprint "[INFO]: Clean the installed ops_nn module before install."
        if [ ! -f "${_UNINSTALL_SHELL_FILE}" ]; then
            logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST};ERR_DES:The file\
(${_UNINSTALL_SHELL_FILE}) not exists. Please set the correct install \
path or clean the previous version ops_nn install info (/etc/ascend_install.info) and then reinstall it."
            return 1
        fi
#        aicpuinfofile "remove"
        uninstall_file_path="${target_dir}/share/info/${opp_platform_dir}/script"
	keyword=$(cat "${uninstall_file_path}/uninstall.sh" | grep function)
        if [ "${keyword}" != "" ];then
            #uninstall the file before compatible
            bash "${uninstall_file_path}/uninstall.sh"
            if [ "$?" != 0 ]; then
                logandprint "[ERROR]: ERR_NO:${INSTALL_FAILED};ERR_DES:Clean the installed directory failed."
                return 1
            fi
        else
            sh "${_UNINSTALL_SHELL_FILE}" "${target_dir}" "uninstall" "${is_quiet}" $in_feature "${is_docker_install}" "${docker_root}" "$pkg_version_dir"
            if [ "$?" != 0 ]; then
                logandprint "[ERROR]: ERR_NO:${INSTALL_FAILED};ERR_DES:Clean the installed directory failed."
                return 1
            fi
        fi

    fi

}

checkemptyuserandgroup() {
    _username="$1"
    _usergroup="$2"
    if [ $(id -u) -ne 0 ]; then
        if [ "${_username}" != "" ] || [ "${_usergroup}" != "" ]; then
            logandprint "[ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:\
Only root user can specific user name or user group."
            return 1
        else
            return 0
        fi
    else
        return 0
    fi
}


parent_dirs_permission_check() {
    current_dir="$1" parent_dir="" short_install_dir=""
    owner="" mod_num=""

    parent_dir=$(dirname "${current_dir}")
    short_install_dir=$(basename "${current_dir}")
    logandprint "[INFO]: parent_dir value is [${parent_dir}] and children_dir value is [${short_install_dir}]"

    if [ "${current_dir}"x = "/"x ]; then
        logandprint "[INFO]: parent dirs permission checked successfully"
        return 0
    else
        owner=""
        if [ -d "${parent_dir}"/"${short_install_dir}" ]; then
            owner=$(stat -c %U "${parent_dir}"/"${short_install_dir}")
        fi
        if [ "${owner}" = "" ]; then
            logandprint "[WARNING]: [${parent_dir}/${short_install_dir}] is not exist."
            return 1
        elif [ "${owner}" != "root" ]; then
            logandprint "[WARNING]: [${short_install_dir}] permission isn't right, it should belong to root."
            return 1
        fi
        logandprint "[INFO]: [${short_install_dir}] belongs to root."

        mod_num=$(stat -c %a "${parent_dir}"/"${short_install_dir}")
        if [ ${mod_num} -lt 755 ]; then
            logandprint "[WARNING]: [${short_install_dir}] permission is too small, it is recommended that the permission be 755 for the root user."
            return 2
        elif [ ${mod_num} -eq 755 ]; then
            logandprint "[INFO]: [${short_install_dir}] permission is ok."
        else
            logandprint "[WARNING]: [${short_install_dir}] permission is too high, it is recommended that the permission be 755 for the root user."
            [ "${is_quiet}" = n ] && return 3
        fi

        parent_dirs_permission_check "${parent_dir}"
    fi
}

getoperationname() {
    _cmd_name="install"
    if [ "${is_upgrade}" = "y" ]; then
        _cmd_name="upgrade"
    elif [ "${is_uninstall}" = "y" ]; then
        _cmd_name="uninstall"
    fi
    echo "${_cmd_name}"
}

getoperationinstalltype() {
    _cmd_name_res=$(getoperationname)
    _cmd_install_type="${in_install_type}"
    if [ "${_cmd_name_res}" != "install" ]; then
        _cmd_install_type=$(getinstalledinfo "${KEY_INSTALLED_TYPE}")
    fi
    echo "${_cmd_install_type}"
}

getfirstnotexistdir() {
    in_tmp_dir="$1"
    arr=$(echo ${in_tmp_dir} | tr '/' "\n")
    index=0
    id=-1
    for i in $arr;do
	id=$((${id:=-1}+1))
	eval arr_$id=$i
	index=$(expr $index + 1)
    done
    len=${index}
    tmp_dir_0=""
    tmp_str_val=""
    for i in $(seq 0 ${len}); do
        tmp_str_val="${tmp_str_val}""/""$(eval echo '$'arr_$i)"
        eval tmp_dir_$i=${tmp_str_val}
    done
    for i in $(seq 0 ${len}); do
        if [ ! -d "$(eval echo '$'tmp_dir_$i)" ]; then
            echo "$(eval echo '$'tmp_dir_$i)"
            break
        fi
    done
}

isonlylastnotexistdir() {
    in_tmp_dir_val="$1"
    tmp_arr=$(echo ${in_tmp_dir_val} | tr '/' ' ')
    index_val=0
    for i in $tmp_arr;do id=$((${id:=-1}+1)); eval arr_$id=$i;index_val=$(expr $index_val + 1);done
    tmp_str_value=""
    eval tmp_dir_0="/"
    arr_len=$(expr $index_val + 1)
    for i in $(seq 1 ${arr_len}); do
	tmp_i=$(expr $i - 1)
	tmp_str_value="${tmp_str_value}""/""$(eval echo '$'arr_${tmp_i})"
        eval tmp_dir_$i=${tmp_str_value}
    done
    less_len=$(expr $index_val - 1)
    for i in $(seq 0 ${less_len}); do
	if [ ! -d "$(eval echo '$'tmp_dir_$i)" ]; then
            isonlylastnotexistdir_path=""
	    THE_NUM_LAST_NOT_EXISTDIR="$(eval echo '$'tmp_dir_$i)"
            break
        else
	    isonlylastnotexistdir_path="$(eval echo '$'tmp_dir_$i)"
        fi
    done
    return
}

matchfullpath() {
    absolute_path=""
    if [ -d "${1}" ]; then
        absolute_path=${1}
        return
    fi
    paths="/"*
    for i in ${paths}; do
        if [ "${1}" = "${i}" ] && [ "${1%/*}" = "" ]; then
            absolute_path="/${1##*/}"
        else
            absolute_path="$(cd ${1%/*} >/dev/null 2>&1; pwd)/${1##*/}"
        fi
    done
    return
}

checkprefolderspermission() {
    in_tmp_dir_value="$1"
    _uname_info="$2"
    arr_val=$(echo ${in_tmp_dir_value} | awk '{split($0,arr,"/");for(i in arr) print arr[i]}')
    index_value=0
    for i in $arr_val;do id=$((${id:=-1}+1)); eval arr_$id=$i;index_value=$(expr $index_value + 1);done
    len_val="${index_value}"
    tmp_str=""
    for i in $(seq 0 ${len_val}); do
        tmp_str_info="${tmp_str}""/""$(eval echo '$'arr_$i)"
        eval tmp_dir_$i=${tmp_str_info}
    done
    for i in $(seq 0 ${len_val}); do
        if [ -d "$(eval echo '$'tmp_dir_$i)" ]; then
            # general user only can install for himself
            cd $(eval echo '$'tmp_dir_$i) >> /dev/null 2>&1
            if [ "$?" != "0" ]; then
                logandprint "[ERROR]: ERR_NO:${PERM_DENIED};ERR_DES:The ${_uname_info} do \
not have the permission to access $(eval echo '$'tmp_dir_$i), please reset the directory \
to a right permission."
                return 1
            fi
            cd - >> /dev/null 2>&1
        fi
    done
    return 0
}

checklastfolderspermissionforcheckpath() {
    in_tmp_dir_info="$1"
    _uname_res="$2"

    if [ -d "${in_tmp_dir_info}" ]; then
        if [ "$(id -u)" = "0" ]; then
            if [ "${is_uninstall}" = "y" ];then
              return 0
            fi

            sh -c "test -w ${in_tmp_dir_info}"  2> /dev/null
            if [ "$?" != "0" ]; then
                logandprint "[ERROR]: ERR_NO:${PERM_DENIED};ERR_DES:The ${_uname_res} do \
access ${in_tmp_dir_info} failed, please reset the directory \
to a right permission."
                return 1
            fi
        else
                # general user only can install for himself
            test -w ${in_tmp_dir_info} >> /dev/null 2>&1
            if [ "$?" != "0" ]; then
                logandprint "[ERROR]: ERR_NO:${PERM_DENIED};ERR_DES:The ${_uname_res} \
access ${in_tmp_dir_info} failed, please reset the directory \
to a right permission."
                return 1
            fi
        fi
    fi
    return 0
}

creat_checkpath() {
    created_path=$1
    _target_username=$2
    _target_usergroup=$3
    mkdir -p "${created_path}"
    _created_path_perm=755
    if [ "$(id -u)" != "0" ] && [ "${is_for_all}" != "y" ]; then
         _created_path_perm=750
    fi
    chmod ${_created_path_perm} "${created_path}" 2> /dev/null
    if [ "$(id -u)" != "0" ]; then
        chown "${_target_username}":"${_target_usergroup}" "${created_path}" 2> /dev/null
    else
        chown root:root "${created_path}" 2> /dev/nul
    fi

    ret_created_path=${created_path}
    return
}


select_last_dir_component (){
    path="$1"
    last_component=$(basename ${path})
    if [ "${last_component}" = "atc" ] ;then
        last_component="atc"
        return
    elif [ "${last_component}" = "fwkacllib" ]; then
        last_component="fwkacllib"
        return
    elif [ "${last_component}" = "compiler" ]; then
        last_component="compiler"
        return
    fi
}

check_version_file () {
    pkg_path="$1"
    component_ret="$2"
    run_pkg_path_temp=$(dirname "${pkg_path}")
    run_pkg_path_temp2=${run_pkg_path_temp%/*}
    run_pkg_path="${run_pkg_path_temp}""/${component_ret}"
    run_pkg_path_temp2=${run_pkg_path%/*}
    version_file="${run_pkg_path}""/version.info"
    version_file_tmp="${run_pkg_path_temp2}""/version.info"
    if [ -f "${version_file_tmp}" ];then
        version_file=${version_file_tmp}
    fi
    if [ -f "${version_file}" ];then
        echo "${version_file}" >> /dev/null 2
    else
        logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST}; The [${component_ret}] version.info in path [${pkg_path}] not exists."
        exitlog
        exit 1
    fi
    return
}


check_opp_version_file () {
    if [ -f "${_CURR_PATH}/../version.info" ];then
        opp_ver_info="${_CURR_PATH}/../version.info"
    elif [ -f "${_DEFAULT_INSTALL_PATH}/${opp_platform_dir}/version.info" ];then #TODO:
        opp_ver_info="${_DEFAULT_INSTALL_PATH}/${opp_platform_dir}/version.info"
    else
        logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST}; The [${opp_platform_dir}] version.info not exists."
        exitlog
        exit 1
    fi
    return
}

check_relation () {
    opp_ver_info_val="$1"
    req_pkg_name="$2"
    req_pkg_version="$3"
    if [ -f "${_COMMON_INC_FILE}" ];then
    . "${_COMMON_INC_FILE}"
    check_pkg_ver_deps "${opp_ver_info_val}" "${req_pkg_name}" "${req_pkg_version}"
    ret_situation=$ver_check_status
    else
        logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST}; The ${_COMMON_INC_FILE} not exists."
        exitlog
        exit 1
    fi
    return
}

show_relation () {
    relation_situation="$1"
    req_pkg_name_val="$2"
    req_pkg_path="$3"
    if [ "$relation_situation" = "SUCC" ] ;then
        logandprint "[INFO]: Relationship of ops_nn with ${req_pkg_name_val} in path ${req_pkg_path} checked successfully"
    else
        logandprint "[WARNING]: Relationship of ops_nn with ${req_pkg_name_val} in path ${req_pkg_path} checked failed."
    fi
    return
}

find_version_check(){
    if [ "$(id -u)" != "0" ]; then
        atc_res=$(find ${HOME} -name "ccec_compiler" | grep Ascend | grep atc)
        fwk_res=$(find ${HOME} -name "ccec_compiler" | grep Ascend| grep fwk)
        comp_res=$(find ${HOME} -name "ccec_compiler" | grep Ascend| grep Ascend/compiler)
        ccec_compiler_path="$atc_res $fwk_res $comp_res"
    else
        atc_res=$(find /usr/local -name "ccec_compiler" | grep Ascend | grep atc)
        fwk_res=$(find /usr/local -name "ccec_compiler" | grep Ascend| grep fwk)
        comp_res=$(find /usr/local -name "ccec_compiler" | grep Ascend| grep Ascend/compiler)
        ccec_compiler_path="$atc_res $fwk_res $comp_res"
    fi
    check_opp_version_file
    ret_check_opp_version_file=$opp_ver_info
    for var in ${ccec_compiler_path}
        do
        run_pkg_path_val=$(dirname "${var}")
# find run pkg name
        select_last_dir_component "${run_pkg_path_val}"
        ret_pkg_name=$last_component
#get check version
        check_version_file "${run_pkg_path_val}" "${ret_pkg_name}"
        ret_check_version_file=$version_file
#check relation
        check_relation "${ret_check_opp_version_file}" "${ret_pkg_name}" "${ret_check_version_file}"
        ret_check_relation_val=$ret_situation
#show relation
        show_relation "${ret_check_relation_val}" "${ret_pkg_name}" "${run_pkg_path_val}"
         done
    return
}


path_version_check(){
    path_env_list="$1"
    check_opp_version_file
    ret_check_opp_version_file_name=$opp_ver_info
    path_list=$(echo "${path_env_list}" | cut -d"=" -f2 )
    array=$(echo ${path_list} | awk '{split($0,arr,":");for(i in arr) print arr[i]}')
    for var in ${array}
    do
        path_ccec_compile=$(echo ${var} | grep -w "ccec_compiler")
        if [ "${path_ccec_compile}" != "" ]; then
            pkg_path_val=$(dirname $(dirname "${path_ccec_compile}"))
# find run pkg name
            select_last_dir_component "${pkg_path_val}"
            ret_pkg_name_val=$last_component
#get check version
            check_version_file "${pkg_path_val}" "${ret_pkg_name_val}"
            ret_check_version_file_val=$version_file
#check relation
            check_relation "${ret_check_opp_version_file_name}" "${ret_pkg_name}" "${ret_check_version_file_val}"
            ret_check_relation=$ret_situation
#show relation
            show_relation "${ret_check_relation}" "${ret_pkg_name}" "${pkg_path_val}"
        else
            echo "the var_case does not contains ccec_compiler" >> /dev/null 2
        fi
    done
    return
}

check_docker_path(){
    docker_path="$1"
    if [ "${docker_path}" != "/"* ]; then
        echo "[OpsNN] [ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:Parameter --docker-root \
must with absolute path that which is start with root directory /. Such as --docker-root=/${docker_path}"
        exitlog
        exit 1
    fi
    if [ ! -d "${docker_path}" ]; then
        echo "[OpsNN] [ERROR]: ERR_NO:${FILE_NOT_EXIST}; The directory:${docker_path} not exist, please create this directory."
        exitlog
        exit 1
    fi
}

judgment_path() {
    . "${_COMMON_INC_FILE}"
    check_install_path_valid "${1}"
    if [ $? -ne 0 ]; then
        echo "[OpsNN][ERROR]: The ops_nn install path ${1} is invalid, only characters in [a-z,A-Z,0-9,-,_] are supported!"
        exitlog
        exit 1
    fi
}

check_install_path(){
    in_install_path_param="$1"
    param_name="$2"
    # empty patch check
    if [ "x${in_install_path_param}" = "x" ]; then
        echo "[OpsNN] [ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:Parameter ${param_name} \
not support that the install path is empty."
        exitlog
        exit 1
    fi
    # space check
    if echo "x${in_install_path_param}" | grep -q " "; then
        echo "[OpsNN] [ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:Parameter ${param_name} \
not support that the install path contains space character."
        exitlog
        exit 1
    fi
    # delete last "/"
    temp_path="${in_install_path_param}"
    temp_path=$(echo "${temp_path}" | sed "s/\/*$//g")
    if [ x"${temp_path}" = "x" ]; then
        temp_path="/"
    fi
    # covert relative path to absolute path
    prefix=$(echo "${temp_path}" | cut -d"/" -f1 | cut -d"~" -f1)
    if [ x"${prefix}" = "x" ]; then
        in_install_path_param="${temp_path}"
    else
	    prefix=$(echo "${run_path}" | cut -d"/" -f1 | cut -d"~" -f1)
        if [ x"${prefix}" = "x" ]; then
            in_install_path_param="${run_path}/${temp_path}"
        else
            echo "[OpsNN] [ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES: Run package path is invalid: $run_path"
            exitlog
            exit 1
        fi
    fi
    # covert '~' to home path
    home=$(echo "${in_install_path_param}" | cut -d"~" -f1)
    if [ "x${home}" = "x" ]; then
	temp_path_value=$(echo "${in_install_path_param}" | cut -d"~" -f2)
        if [ "$(id -u)" -eq 0 ]; then
            in_install_path_param="/root$temp_path_value"
        else
	    home_path=$(eval echo "~${USER}")
	    home_path=$(echo "${home_path}" | sed "s/\/*$//g")
            in_install_path_param="$home_path$temp_path_value"
        fi
    fi
}

# execute prereq_check file
exec_pre_check(){
    sh "${_PRE_CHECK_FILE}"
}

# execute prereq_check file and interact with user
interact_pre_check() {
    exec_pre_check
    if [ "$?" != 0 ]; then
        if [ "${is_quiet}" = y ]; then
            logandprint "[WARNING]: Precheck of ops_nn module execute failed! do you want to continue install? y"
        else
            logandprint "[WARNING]: Precheck of ops_nn module execute failed! do you want to continue install?  [y/n] "
            while true
            do
            read yn
            if [ "$yn" = "n" ]; then
                echo "stop install ops_nn module!"
                exit 1
            elif [ "$yn" = y ]; then
                break;
            else
                echo "[WARNING]: Input error, please input y or n to choose!"
            fi
            done
        fi
    fi
}

uninstall_none_multi_version() {
    local install_path="$1"
    if [ "$pkg_is_multi_version" = "true" ] && [ ! -L "$install_path" ] && [ -d "$install_path" ]; then
	if [ "$is_install" = "y" ] || [ "$is_upgrade" = "y" ] || [ "$is_uninstall" = "y" ]; then
            path_version="$install_path/version.info"
            path_install="$install_path/ascend_install.info"
            if [ -f "$path_version" ] && [ -f "$path_install" ] && [ -f "$install_path/script/uninstall.sh" ]; then
                $install_path/script/uninstall.sh
            fi
        fi
    fi
}

switchchip(){
    if [ "$tmp_chip_type" = "Ascend910" ];then
        chip_type="Ascend910"
        chip_type_new="Ascend910"
    elif [ "$tmp_chip_type" = "Ascend310" ];then
        chip_type="Ascend310"
        chip_type_new="Ascend310"
    elif [ "$tmp_chip_type" = "Ascend310-minirc" ] || [ "$tmp_chip_type" = "Ascend310RC" ];then
        chip_type="Ascend310RC"
        chip_type_new="Ascend310RC"
    elif [ "$tmp_chip_type" = "Ascend310P" ];then
        chip_type="Ascend310P"
        chip_type_new="Ascend310P"
    elif [ "$tmp_chip_type" = "Ascend" ];then
        chip_type="Ascend"
        chip_type_new="Ascend610,Ascend910B,Ascend910_93,Ascend310B,Ascend310BRC,Ascend"
    elif [ "${tmp_chip_type}" = "Ascend610" ] || [ "$tmp_chip_type" = "Ascend910B" ] || [ "$tmp_chip_type" = "Ascend910_93" ] || [ "$tmp_chip_type" = "Ascend310B" ] || [ "$tmp_chip_type" = "Ascend310BRC" ]; then
        chip_type="Ascend"
        chip_type_new="${tmp_chip_type},Ascend"
    else
       echo "[OpsNN] [ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:Parameter --chip \
not support that $tmp_chip_type"
        exitlog
        exit 1
    fi
}

aicputarsize(){
    if [ -d /usr/lib64/aicpu_kernels ];then
        if [ ! -f /usr/lib64/aicpu_kernels/aicpu_package_install.info ];then
            touch /usr/lib64/aicpu_kernels/aicpu_package_install.info  > /dev/null 2>&1
            chown HwHiAiUser:HwHiAiUser /usr/lib64/aicpu_kernels/aicpu_package_install.info  > /dev/null 2>&1
        fi
        if [ ! -d /usr/lib64/aicpu_kernels/0 ];then
            mkdir /usr/lib64/aicpu_kernels/0  > /dev/null 2>&1
            chown HwHiAiUser:HwHiAiUser /usr/lib64/aicpu_kernels/0  > /dev/null 2>&1
        fi
        if [ ! -d /usr/lib64/aicpu_kernels/0/aicpu_kernels_device ];then
            mkdir /usr/lib64/aicpu_kernels/0/aicpu_kernels_device  > /dev/null 2>&1
            chown HwHiAiUser:HwHiAiUser /usr/lib64/aicpu_kernels/0/aicpu_kernels_device  > /dev/null 2>&1
        fi
        if [ ! -d /usr/lib64/aicpu_kernels/0/aicpu_kernels_device/sand_box ];then
            mkdir /usr/lib64/aicpu_kernels/0/aicpu_kernels_device/sand_box   > /dev/null 2>&1
            chown HwHiAiUser:HwHiAiUser /usr/lib64/aicpu_kernels/0/aicpu_kernels_device/sand_box   > /dev/null 2>&1
        fi
        chmod 664 /usr/lib64/aicpu_kernels/aicpu_package_install.info > /dev/null 2>&1
        grep '^export LD_LIBRARY_PATH' ~/.bashrc | grep /usr/lib64/aicpu_kernels/0/aicpu_kernels_device  > /dev/null 2>&1
        if [ $? -ne 0 ];then
        echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/lib64/aicpu_kernels/0/aicpu_kernels_device" >> ~/.bashrc > /dev/null 2>&1
        fi
        grep '^export LD_LIBRARY_PATH' ~/.bashrc | grep /usr/lib64/aicpu_kernels/0/aicpu_kernels_device/sand_box > /dev/null 2>&1
        if [ $? -ne 0 ];then
        echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/sand_box " >> ~/.bashrc > /dev/null 2>&1
        fi
    fi
}

versioninfoadd(){
    aicpuversioninfo="$1"
    chmod 640 ${aicpuversioninfo}
    echo "irProto_version=1.0" >> "${aicpuversioninfo}"
    echo "required_driver_ascendhal_version=4.0.0" >> "${aicpuversioninfo}"
    echo "required_driver_aicpu_version=1.0" >> "${aicpuversioninfo}"
    echo "required_driver_tdt_version=1.0" >> "${aicpuversioninfo}"
    chmod 440 ${aicpuversioninfo}
}

check_dir_permission_for_common_user() {
    current_dir_val="$1"
    parent_dir_val=$(dirname "${current_dir_val}")
    if [ "${parent_dir_val}"x = "/"x ]; then
        return
    fi
    if [ -d "${current_dir_val}" ]; then
        mod_num_val=$(stat -c %a "${current_dir_val}")
        num_0=${mod_num_val:0:1}
        num_1=${mod_num_val:1:1}
        num_2=${mod_num_val:2:1}
        if [ ${num_0} -lt 7 ] || [ ${num_1} -lt 5 ] || [ ${num_2} -lt 5 ]; then
            logandprint "[ERROR]: ${current_dir_val} permission is ${mod_num_val}, this permission is too small, and it should be 755."
            exitlog
            exit 1
        elif [ ${num_0} -eq 7 ] && [ ${num_1} -eq 5 ] && [ ${num_2} -eq 5 ]; then
            logandprint "[INFO]: ${current_dir_val} permission is ok."
        else
            if [ "${is_quiet}" = y ]; then
                logandprint "[WARNING]: ${current_dir_val} permission is ${mod_num_val}, this permission is too high, do you want to continue install? y"
            else
                logandprint "[WARNING]: ${current_dir_val} permission is ${mod_num_val}, this permission is too high, do you want to continue install?  [y/n] "
                while true
                do
                    read yn
                    if [ "$yn" = n ]; then
                        logandprint "[INFO]: stop install ops_nn module!"
                        exitlog
                        exit 1
                    elif [ "$yn" = y ]; then
                        break;
                    else
                        logandprint "[WARNING]: Input error, please input y or n to choose!"
                    fi
                done
            fi
        fi
    fi
    check_dir_permission_for_common_user "${parent_dir_val}"
}

#get the dir of xxx.run
#opp_install_path_curr=`echo "$2" | cut -d"/" -f2- `
run_path=$(echo "$2" | cut -d"-" -f3-)
if [ x"${run_path}" = x"" ]; then
    run_path=$(pwd)
else
    # delete last "/"
    run_path=$(echo "${run_path}" | sed "s/\/*$//g")
    if [ x"${run_path}" = x"" ]; then
        # root path
	run_path=$(pwd)
    fi
fi

# cut first two params from *.run
i=0
while true
do
    if [ x"$1" = x"" ]; then
        break
    fi
    if [ "$(expr substr "$1" 1 2 )" = "--" ]; then
	i=$(expr $i + 1)
    fi
    if [ $i -gt 2 ]; then
        break
    fi
    shift 1
done

if [ "$*" = "" ]; then
    echo "[ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:Unrecognized parameters. \
Try './xxx.run --help for more information.'"
    exitlog
    exit 1
fi

# init install cmd status, set default as n
in_cmd_list="$*"
full_install=n
run_install=n
devel_install=n
is_uninstall=n
is_install=n
is_upgrade=n
is_quiet=n
is_viewlog=n
is_input_path=n
is_check=n
is_precheck=n
in_install_type=""
in_install_path=""
is_docker_install=n
is_setenv=n
docker_root=""
is_feature=n
iter_i=0
startlog

is_multi_version_pkg "pkg_is_multi_version" "$_VERSION_INFO_FILE"
while true
do
    # skip 2 parameters avoid run pkg and directory as input parameter
    case "$1" in
    --full)
        in_install_type=$(echo ${1} | awk -F"--" '{print $2}')
        is_install=y
        iter_i=$(( ${iter_i} + 1 ))
        shift
        ;;
    --upgrade)
        is_upgrade=y
        iter_i=$(( ${iter_i} + 1 ))
        shift
        ;;
    --uninstall)
        is_uninstall=y
        iter_i=$(( ${iter_i} + 1 ))
        shift
        ;;
    --install-path=*)
        is_input_path=y
        in_install_path=$(echo ${1} | cut -d"=" -f2- )
        # check path
        judgment_path "${in_install_path}"
        check_install_path "${in_install_path}" "--install-path"
        shift
        ;;
    --feature=*)
        feature_choice=$(echo $1 | cut -d"=" -f2 )
        if test -z "$feature_choice"; then
            echo "[OpsNN] [ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:Paramter --feature cannot be null."
            exitlog
            exit 1
        fi
        contain_feature "ret" "$feature_choice" "${_FILELIST_FILE}"
        if [ "$ret" = "false" ]; then
            log "WARNING" "OpsNN package doesn't contain features $feature_choice, skip installation."
            exit 0
        fi
        is_feature=y
        shift
        ;;
    --quiet)
        is_quiet=y
        shift
        ;;
    --install-for-all)
        is_for_all=y
        shift
        ;;
    --check)
        is_check=y
        iter_i=$(( ${iter_i} + 1 ))
        shift
        ;;
    --setenv)
        is_setenv=y
        shift
        ;;
    -*)
        echo "[OpsNN] [ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:Unsupported parameters [$1], \
operation execute failed. Please use [--help] to see the useage."
        exitlog
        exit 1
        ;;
    *)
        break
        ;;
    esac
done

logandprint "[INFO]: Command install"
architecture=$(uname -m)
# check platform
if [ "${architecture}" != "${platform_data}" ] ; then
    logandprint "[ERROR]: ERR_NO:${OPERATE_FAILED};ERR_DES:the architecture of the run package \
is inconsistent with that of the current environment. "
    exitlog
    exit 1
fi
#root install set is_for_all y
if [ "$(id -u)" = "0" ] ; then
    is_for_all=y
fi

# pre-check
if [ "${iter_i}" -eq 0 ] && [ "x${is_precheck}" = "xy" ]; then
    interact_pre_check
    exitlog
    exit 0
fi


if [ "${iter_i}" != 1 ]; then
    echo "[OpsNN] [ERROR]: ERR_NO:${PARAM_INVALID};ERR_DES:only support one type: full/run/devel/upgrade/uninstall/check, operation execute failed! \
Please use [--help] to see the usage."
    exitlog
    exit 1
fi

# must init target install path first before installation
if [ "${is_input_path}" != y ]; then
    _TARGET_INSTALL_PATH="${_DEFAULT_INSTALL_PATH}"
else
    _TARGET_INSTALL_PATH="${in_install_path_param}"
fi

if is_version_dirpath "$_TARGET_INSTALL_PATH"; then
    pkg_version_dir="$(basename "$_TARGET_INSTALL_PATH")"
    _TARGET_INSTALL_PATH="$(dirname "$_TARGET_INSTALL_PATH")"
else
    pkg_version_dir="cann"
fi

if [ "$pkg_is_multi_version" = "true" ]; then
    target_dir="${_TARGET_INSTALL_PATH}/$pkg_version_dir"
else
    target_dir="${_TARGET_INSTALL_PATH}"
fi

target_dir_before_docker=${target_dir}
# Splicing docker-root and install-path
if [ "${is_docker_install}" = y ] ; then
    # delete last "/"
    temp_path_param="${docker_root}"
    temp_path_val=$(echo "${temp_path_param}" | sed "s/\/*$//g")
    if [ x"${temp_path_val}" = "x" ]; then
        temp_path_val="/"
    fi
    target_dir=${temp_path_val}${target_dir}
fi

uninstall_none_multi_version "$_TARGET_INSTALL_PATH/${opp_platform_dir}"
if [ "$?" = "0" ]; then
    echo "[OpsNN] [$(getdate)] [INFO]: Uninstall the version before multi version of ops_nn successfully!"
else
    echo "[OpsNN] [$(getdate)] [ERROR]: Uninstall the version before multi version of ops_nn failed!"
    exit 1
fi
if [ "${is_input_path}" = y ]; then
    isonlylastnotexistdir "${_TARGET_INSTALL_PATH}"
    ret_isonlylastnotexistdir_path=$isonlylastnotexistdir_path
    RET_THE_NUM_LAST_NOT_EXISTDIR=$THE_NUM_LAST_NOT_EXISTDIR
fi

_UNINSTALL_SHELL_FILE="${target_dir}""/share/info/${opp_platform_dir}/script/opp_uninstall.sh"
# adpter for old version's path
if [ ! -f "${_UNINSTALL_SHELL_FILE}" ]; then
    _UNINSTALL_SHELL_FILE="${target_dir}""/share/info/${opp_platform_dir}/scripts/opp_uninstall.sh"
fi

# init log file path before installation
_INSTALL_INFO_FILE="${target_dir}/${_INSTALL_INFO_SUFFIX}"

if [ "$(id -u)" != "0" ]; then
    _LOG_PATH_AND_FILE_GROUP="root"
    _LOG_PATH=$(echo "${HOME}")"/var/log/ascend_seclog"
    _INSTALL_LOG_FILE="${_LOG_PATH}/ascend_install.log"
    _OPERATE_LOG_FILE="${_LOG_PATH}/operation.log"
else
    _LOG_PATH="/var/log/ascend_seclog"
    _INSTALL_LOG_FILE="${_LOG_PATH}/ascend_install.log"
    _OPERATE_LOG_FILE="${_LOG_PATH}/operation.log"
fi

# creat log folder and log file
if [ ! -d "${_LOG_PATH}" ]; then
    mkdir -p "${_LOG_PATH}"
fi
if [ ! -f "${_INSTALL_LOG_FILE}" ]; then
    touch "${_INSTALL_LOG_FILE}"
fi
if [ ! -f "${_OPERATE_LOG_FILE}" ]; then
    touch "${_OPERATE_LOG_FILE}"
fi


logandprint "[INFO]: Execute the ops_nn run package."
logandprint "[INFO]: OperationLogFile path: ${_INSTALL_LOG_FILE}."
logandprint "[INFO]: Input params: $in_cmd_list"


if [ "${is_check}" = "y" ] && [ "${check_path}" = "" ]; then
    path_env_list_val=$(env | grep -w PATH)
    path_ccec_compile_val=$(echo ${path_env_list} | grep -w "ccec_compiler")
    if [ "${path_ccec_compile_val}" != "" ]; then
        path_version_check "${path_env_list_val}"
    else
        find_version_check
    fi
    exitlog
    exit 0
fi

if [ "${is_check}"="y" ] && [ "${check_path}" != "" ]; then
    _VERCHECK_FILE="${_CURR_PATH}""/ver_check.sh"
    if [ ! -f "${_VERCHECK_FILE}" ]; then
        logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST};ERR_DES:The file\
(${_VERCHECK_FILE}) not exists. Please make sure that the ops_nn module\
 installed in (${_VERCHECK_FILE}) and then set the correct install path."
    fi
    sh "${_VERCHECK_FILE}" "${check_path}"
    exitlog
    exit 0
fi

# installed version check and print
installed_version=$(getinstalledinfo "${KEY_INSTALLED_VERSION}")
runpkg_version=$(getrunpkginfo "${KEY_RUNPKG_VERSION}")
installed_user=$(getinstalledinfo "${KEY_INSTALLED_UNAME}")
installed_group=$(getinstalledinfo "${KEY_INSTALLED_UGROUP}")
installed_feature=$(getinstalledinfo "${KEY_INSTALLED_FEATURE}")
if [ "${installed_version}" = "" ]; then
    logandprint "[INFO]: Version of installing ops_nn module is ${runpkg_version}."
else
    if [ "${runpkg_version}" != "" ]; then
        logandprint "[INFO]: Existed ops_nn module version is ${installed_version}, \
the new ops_nn module version is ${runpkg_version}."
    fi
fi

if [ "${installed_version}" = "${runpkg_version}" ]; then
    inc_compilation_flag=y
else
    inc_compilation_flag=n
fi

if [ "${is_chip}" = "y" ] && [ -n "${tmp_chip_type}" ]; then
    switchchip
else
    chip_type_new=""
fi

in_feature_new=""
if [ "${is_feature}" = "n" ];then
    if [ -n "$tmp_chip_type" ]; then
        switchchip
        in_feature=${chip_type}
    else
        in_feature="All"
    fi
    aicputarsize
else
    if [ -n "${feature_choice}" ]; then
        in_feature_new=${feature_choice}
        in_feature="All"
    fi
    if [ -n "$tmp_chip_type" ]; then
        switchchip
        in_feature=${chip_type}
    fi
    if [ "${is_uninstall}" = "y" ];then
        if [ ! "$in_feature" = "All" ];then
            logandprint "[WARNING]: chip=$in_feature parameter is invalid"
        fi
        in_feature="All"
    fi
    aicputarsize
fi

# get installed version's user and group
if [ "${installed_version}" != "" ] && [ "${is_uninstall}" != "y" ] && [ "${is_upgrade}" != "y" ]; then
    if [ "${in_username}" != "" ] && [ "${in_usergroup}" != "" ]; then
        if [ "${installed_user}" != "${in_username}" ] || [ "${installed_group}" != "${in_usergroup}" ]; then
            logandprint "[ERROR]: ERR_NO:0x0095;ERR_DES:The user and group are not same with last installation,do not support overwriting installation!"
            exitlog
            exit 1
        fi
    else
        in_username=${installed_user}
        in_usergroup=${installed_group}
    fi
fi

# input username and usergroup valid check
_TARGET_USERNAME="${_CURR_OPERATE_USER}"
_TARGET_USERGROUP="${_CURR_OPERATE_GROUP}"

#Support the installation script when the specified path (relative path and absolute path) does not exist
if [ "${is_input_path}" = "y" ];then
    checkgroupvalidwithuser "${_TARGET_USERGROUP}" "${_TARGET_USERNAME}"
    if [ "${ret_isonlylastnotexistdir_path}" = "" ]; then
#Penultimate path not exists
        logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST}; The directory:${RET_THE_NUM_LAST_NOT_EXISTDIR} not exist, please create this directory."
        exitlog
        exit 1
    else
        if [ -d "${_TARGET_INSTALL_PATH}" ]; then
            checklastfolderspermissionforcheckpath "${_TARGET_INSTALL_PATH}" "${_TARGET_USERNAME}"
            if [ "$?" = 0 ]; then
#All paths exist with write permission
                _TARGET_INSTALL_PATH=${_TARGET_INSTALL_PATH}
            else
#All paths exist, no write permission
                exit 1
            fi
        else
            checklastfolderspermissionforcheckpath "${ret_isonlylastnotexistdir_path}" "${_TARGET_USERNAME}"
            if [ "$?" = 0 ]; then
#penultimate path exists with write permission
                if [ "${target_dir}" = "${_DEFAULT_INSTALL_PATH}" ] && [ ! -d "${_DEFAULT_INSTALL_PATH}" ] && [ "$(id -u)" = "0" ]; then
                    creat_checkpath "${_TARGET_INSTALL_PATH}" "${_TARGET_USERNAME}" "${_TARGET_USERGROUP}"
                else
                    creat_checkpath "${_TARGET_INSTALL_PATH}" "${_TARGET_USERNAME}" "${_TARGET_USERGROUP}"
                fi
            else
#Penultimate path exists, no write permission
                exit 1
            fi
        fi
    fi
else
    if [ "${is_quiet}" = y ] && [ $(id -u) -eq 0 ] && [ ! -d "${_TARGET_INSTALL_PATH}" ]; then
        mkdir -p ${_TARGET_INSTALL_PATH} >> /dev/null 2>&1
        if [ "${is_for_all}" = "y" ]; then
            chmod 755 ${_TARGET_INSTALL_PATH}
        fi
        logandprint "[INFO]: Created default install path: ${_TARGET_INSTALL_PATH}."
    fi
fi

matchfullpath "${target_dir}"
#target_dir=${absolute_path}

if [ "${is_check}" = "y" ]; then
    preinstall_check --install-path="${target_dir}" --docker-root="${docker_root}" --script-dir="${_CURR_PATH}" --package="${opp_platform_dir}" --logfile="${_INSTALL_LOG_FILE}"
    if [ $? -ne 0 ]; then
        logandprint "[ERROR]: ERR_NO:${OPP_COMPATIBILITY_CEHCK_ERR};ERR_DES:Check the compatibility of ops_nn package fail,please confirm the true version package."
        exit 1
	fi
fi


if [ "${is_for_all}" = y ] && [ $(id -u) -ne 0 ]; then
    check_dir_permission_for_common_user "${target_dir}"
fi

if [ "$in_feature" = "All" ];then
    if [ -f "${target_dir}/${opp_platform_dir}/aicpu/script/uninstall.sh" ];then
        ${target_dir}/${opp_platform_dir}/aicpu/script/uninstall.sh  > /dev/null 2>&1
    fi
    if [ -f "${target_dir}/${opp_platform_dir}/Ascend910/aicpu/script/uninstall.sh" ];then
    ${target_dir}/${opp_platform_dir}/Ascend910/aicpu/script/uninstall.sh  > /dev/null 2>&1
    elif [ -f "${target_dir}/${opp_platform_dir}/Ascend310P/aicpu/script/uninstall.sh" ];then
    ${target_dir}/${opp_platform_dir}/Ascend310P/aicpu/script/uninstall.sh  > /dev/null 2>&1
    elif [ -f "${target_dir}/${opp_platform_dir}/Ascend310/aicpu/script/uninstall.sh" ];then
    ${target_dir}/${opp_platform_dir}/Ascend310/aicpu/script/uninstall.sh  > /dev/null 2>&1
    elif [ -f "${target_dir}/${opp_platform_dir}/Ascend310RC/aicpu/script/uninstall.sh" ];then
    ${target_dir}/${opp_platform_dir}/Ascend310RC/aicpu/script/uninstall.sh  > /dev/null 2>&1
    elif [ -f "${target_dir}/${opp_platform_dir}/Ascend/aicpu/script/uninstall.sh" ];then
    ${target_dir}/${opp_platform_dir}/Ascend/aicpu/script/uninstall.sh  > /dev/null 2>&1
    fi
elif [ "$in_feature" = "exec_mode" ]; then
    if [ -f "${target_dir}/${opp_platform_dir}/script/uninstall.sh" ];then
        ${target_dir}/${opp_platform_dir}/script/uninstall.sh  > /dev/null 2>&1
    fi
elif [ "$in_feature" = "exec_mode" ];then
    logandprint "[INFO]: Install only files required at runtime (not compile time), include [libopsproto.so] and [liboptiling.so]"
else
    if [ -f "${target_dir}/${opp_platform_dir}/aicpu/script/uninstall.sh" ];then
        ${target_dir}/${opp_platform_dir}/aicpu/script/uninstall.sh  > /dev/null 2>&1
    fi
    if [ -f "${target_dir}/${opp_platform_dir}/${in_feature}/aicpu/script/uninstall.sh" ];then
    ${target_dir}/${opp_platform_dir}/${in_feature}/aicpu/script/uninstall.sh  > /dev/null 2>&1
    fi
fi

if [ "${is_install}" = "y" ];then
    if [ "${is_for_all}" = y ] && [ $(id -u) -ne 0 ]; then
        check_dir_permission_for_common_user "${target_dir}"
    fi
    # precheck before install ops_nn module
    if [ "${is_precheck}" = "y" ];then
        interact_pre_check
    fi

    # devel mode no need set the correct install user
    if [ "${in_install_type}" != "devel" ]; then
        checkusergrouprelationvalid "${_TARGET_USERNAME}" "${_TARGET_USERGROUP}"
        if [ "$?" != 0 ]; then
            logandprint "[ERROR]: ERR_NO:${INSTALL_FAILED};ERR_DES:Installation of ops_nn\
 module execute failed!"
            logoperationretstatus "$(getoperationname)" "$(getoperationinstalltype)" "1" "${in_cmd_list}"
        fi
    else
        checkgroupvalidwithuser "${_TARGET_USERGROUP}" "${_TARGET_USERNAME}"
        devel_group_ret="$?"
        if [ "${devel_group_ret}" != "0" ]; then
            _TARGET_USERNAME="${_DEFAULT_USERNAME}"
            _TARGET_USERGROUP="${_DEFAULT_USERGROUP}"
            if [ "${is_quiet}" = "y" ]; then
                logandprint "[WARNING]: Input user name or user group is invalid. \
Would you like to set as the default user name (${_DEFAULT_USERNAME}),\
user group (${_DEFAULT_USERGROUP}) for devel mode? y"
            else
                logandprint "[WARNING]: Input user name or user group is invalid. \
Would you like to set as the default user name (${_DEFAULT_USERNAME}),\
user group (${_DEFAULT_USERGROUP}) for devel mode? [y/n]"
                while true
                do
                    read yn
                    if [ "$yn" = "n" ]; then
                        logandprint "[INFO]: Exit to install ops_nn module."
                        logoperationretstatus "$(getoperationname)" "$(getoperationinstalltype)" "1" "${in_cmd_list}"
                    elif [ "$yn" = y ]; then
                        break;
                    else
                        echo "[WARNING]: Input error, please input y or n to choose!"
                    fi
                done
            fi
        fi
    fi
    # install need check whether the custom user can access the folders or not
    if [ $(id -u) -ne 0 ]; then
        checkprefolderspermission "${target_dir}/${opp_platform_dir}" "${_TARGET_USERNAME}"
    fi

    if [ "$?" != 0 ]; then
        logoperationretstatus "install" "${in_install_type}" "1" "${in_cmd_list}"
    fi

    # use uninstall to clean the install folder
    precleanbeforeinstall "${_TARGET_INSTALL_PATH}"
    if [ "$?" != 0 ]; then
        logoperationretstatus "install" "${in_install_type}" "1" "${in_cmd_list}"
    fi

    _FIRST_NOT_EXIST_DIR=$(getfirstnotexistdir "${target_dir}/${opp_platform_dir}")
    #getfirstnotexistdir "/usr/local/tmp/CANN-1.81/ops_nn"
    is_the_last_dir_opp=""
    if [ "${_FIRST_NOT_EXIST_DIR}" = "${target_dir}/${opp_platform_dir}" ]; then
        is_the_last_dir_opp=1
    else
        is_the_last_dir_opp=0
    fi
    if [ -f "${target_dir}/${opp_platform_dir}/ascend_install.info" ];then
        install_type=$(getinstalledinfo "${KEY_INSTALLED_FEATURE}")
        if [ ! "$install_type" = "$in_feature" ];then
#            aicpuinfofile "remove"
            if [ "${inc_compilation_flag}" = "y" ] && [ "${is_chip}" = "y" ]; then
                old_chip_type=$(getinstalledinfo "${KEY_INSTALLED_CHIP}")
                if [ "${old_chip_type}" != "" ]; then
                    chip_type_new="${old_chip_type},${chip_type_new}"
                    chip_type_new=$(echo ${chip_type_new} | awk -F "," '{for(i=1;i<=NF;i++) if(!a[$i]) {a[$i]=1;printf("%s ", $i)}}' | tr ' ' ',') # 去重
                    chip_type_new=${chip_type_new%%,}
                else
                    chip_type_new=${chip_type_new}
                fi
            else
                chip_type_new=${chip_type_new}
            fi
            sh "${_UPGRADE_SHELL_FILE}" "${_TARGET_INSTALL_PATH}" "${_DEFAULT_USERNAME}" "${_DEFAULT_USERGROUP}" ${in_feature} "${is_quiet}" "${is_for_all}" "${is_setenv}" "${is_docker_install}" "${docker_root}" "" "n" "${in_feature_new}" "${chip_type_new}" "$$pkg_version_dir"
            chmod -R 555 "${target_dir}/${opp_platform_dir}/script"> /dev/null 2>&1
            chmod -R 555 "${target_dir}/${opp_platform_dir}/bin"> /dev/null 2>&1
            if [ $(id -u) -eq 0 ]; then
                chown -R "root":"root" "${target_dir}/${opp_platform_dir}/script"> /dev/null 2>&1
                chown "root":"root" "${target_dir}/${opp_platform_dir}"> /dev/null 2>&1
            else
                chmod 750 "${target_dir}"> /dev/null 2>&1
            fi
            # uprate precheck info to ${target_dir}/bin/prereq_check.bash
            logandprint "[INFO]: Set precheck info."
            logoperationretstatus "upgrade" "${install_type}" "$?" "${in_cmd_list}"
        fi
    fi
    # check the compatibility of ops_nn
    preinstall_process --install-path="${target_dir}" --docker-root="${docker_root}" --script-dir="${_CURR_PATH}" --package="${opp_platform_dir}" --logfile="${_INSTALL_LOG_FILE}"
    if [ $? -ne 0 ]; then
        logandprint "[ERROR]: ERR_NO:${OPP_COMPATIBILITY_CEHCK_ERR};ERR_DES:Check the compatibility of ops_nn package fail,please confirm the true version package."
        exit 1
    fi
    # call opp_install.sh
    if [ -d "${target_dir}/ops_nn" ] && [ ! -L "${target_dir}/ops_nn" ] && [ -f "${target_dir}/ops_nn/script/uninstall.sh" ]; then
        bash "${target_dir}/ops_nn/script/uninstall.sh"
    fi
    sh "${_INSTALL_SHELL_FILE}" "${_TARGET_INSTALL_PATH}" "${_DEFAULT_USERNAME}" "${_DEFAULT_USERGROUP}" "${in_feature}" "${in_install_type}" "${is_quiet}" "${_FIRST_NOT_EXIST_DIR}" "${is_the_last_dir_opp}" "${is_for_all}" "${is_setenv}" "${is_docker_install}" "${docker_root}" "${is_input_path}" "${in_feature_new}" "${chip_type_new}" "$pkg_version_dir"
    if [ "$?" != 0 ]; then
        logoperationretstatus "install" "${in_install_type}" "1" "${in_cmd_list}"
    fi
fi

if [ "${is_upgrade}" = "y" ];then
    # if latest dir has no ops_nn module exit first!
    if [ "${is_for_all}" = y ] && [ $(id -u) -ne 0 ]; then
        check_dir_permission_for_common_user "${target_dir}"
    fi
    install_type=$(getinstalledinfo "${KEY_INSTALLED_TYPE}")
    # precheck before upgrade ops_nn module
    if [ "${is_precheck}" = "y" ];then
        interact_pre_check
    fi
    # check the compatibility of ops_nn
    preinstall_process --install-path="${target_dir}" --docker-root="${docker_root}" --script-dir="${_CURR_PATH}" --package="${opp_platform_dir}" --logfile="${_INSTALL_LOG_FILE}"
    if [ $? -ne 0 ]; then
        logandprint "[ERROR]: ERR_NO:${OPP_COMPATIBILITY_CEHCK_ERR};ERR_DES:Check the compatibility of ops_nn package fail,please confirm the true version package."
        exit 1
    fi
    sh "${_UPGRADE_SHELL_FILE}" "${_TARGET_INSTALL_PATH}" "${_DEFAULT_USERNAME}" "${_DEFAULT_USERGROUP}" ${in_feature} "${is_quiet}" "${is_for_all}" "${is_setenv}" "${is_docker_install}" "${docker_root}" "${is_input_path}" "${is_upgrade}" "${in_feature_new}" "${chip_type_new}" "$pkg_version_dir"
    if [ $(id -u) -eq 0 ]; then
        chown -R "root":"root" "${target_dir}/share/info/${opp_platform_dir}/script" 2> /dev/null
        chown "root":"root" "${target_dir}/share/info/${opp_platform_dir}" 2> /dev/null
    fi

    if [ $(id -u) -eq 0 ]; then
        chmod -R 555 "${target_dir}/share/info/${opp_platform_dir}/script" 2> /dev/null
        chmod 444 "${target_dir}/share/info/${opp_platform_dir}/script/filelist.csv" 2> /dev/null
    else
        chmod -R 550 "${target_dir}/share/info/${opp_platform_dir}/script" 2> /dev/null
        chmod 440 "${target_dir}/share/info/${opp_platform_dir}/script/filelist.csv" 2> /dev/null
    fi
    # uprate precheck info to ${target_dir}/bin/prereq_check.bash
    logandprint "[INFO]: Set precheck info."
    logoperationretstatus "upgrade" "${install_type}" "$?" "${in_cmd_list}"
fi

if [ "${is_uninstall}" = "y" ];then
    install_type=$(getinstalledinfo "${KEY_INSTALLED_TYPE}")
    # uninstall not support specific username and usergroup
    checkemtpyuser "uninstall" "${in_username}" "${in_usergroup}"
    if [ "$?" != 0 ]; then
        logoperationretstatus "uninstall" "${install_type}" "1" "${in_cmd_list}"
    fi
    aicpustatus=0
    if [ ! -f "${_UNINSTALL_SHELL_FILE}" ]; then
        logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST};ERR_DES:The file\
    (${_UNINSTALL_SHELL_FILE}) not exists. Please make sure that the ops_nn module\
    installed in (${target_dir}) and then set the correct install path."
	uninstall_path=$(ls "${_TARGET_INSTALL_PATH}" 2> /dev/null)
        if [ "${uninstall_path}" = "" ]; then
            rm -rf "${_TARGET_INSTALL_PATH}"
        fi
            logoperationretstatus "uninstall" "${install_type}" "1" "${in_cmd_list}"
        exit 0
    fi
    # call opp_uninstall.sh
#    aicpuinfofile "remove"
    sh "${_UNINSTALL_SHELL_FILE}" "${_TARGET_INSTALL_PATH}" "uninstall" "${is_quiet}" $in_feature "${is_docker_install}" "${docker_root}" "$pkg_version_dir"
    # remove precheck info in ${target_dir}/bin/prereq_check.bash
    logandprint "[INFO]: Remove precheck info."

    logoperationretstatus "uninstall" "${install_type}" "$?" "${in_cmd_list}"
fi

if [ "${is_precheck}" = "y" ];then
    exec_pre_check
    exit $?
fi
exit 0

