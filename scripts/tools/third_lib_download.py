#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import urllib.request
import os
import re
from pathlib import Path


def down_files_native(url_list):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 创建子目录（例如：downloads）
    download_dir = os.path.join(current_dir, "cann_3rd_lib_path_download")
    os.makedirs(download_dir, exist_ok=True)  # 如果目录不存在则创建

    for url in url_list:

        file_name = url.split('/')[-1]

        if not file_name:
            file_name = "downloaded_file"

        # 将文件保存到新建的目录下
        file_path = os.path.join(download_dir, file_name)

        urllib.request.urlretrieve(url, file_path)


def extract_urls_from_cmake(cmake_file):
    """
    从单个 .cmake 文件中提取所有 HTTPS URL

    支持格式：
    - set(REQ_URL "https://...")
    - set(DOWNLOAD_URL "https://...")
    - URL https://...
    - "https://..."
    """
    urls = []
    content = cmake_file.read_text(encoding='utf-8')

    # 匹配模式1: set(XXX "https://...")
    # 匹配模式2: set(XXX https://...)
    # 匹配模式3: 裸 URL
    patterns = [
        # set(VAR "https://..." ) 或 set(VAR https://... )
        r'set\s*\(\s*\w+\s+["\']?(https://[^"\'\s)]+)["\']?\s*\)',
        # URL https://... (ExternalProject_Add)
        r'URL\s+["\']?(https://[^"\'\s)]+)["\']?',
        # 引号包裹的 URL
        r'["\'](https://[^"\']+)["\']',
        # 裸 URL (行首或空格后)
        r'(^|\s)(https://\S+)',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, content, re.MULTILINE)
        for match in matches:
            # 处理捕获组的情况
            if isinstance(match, tuple):
                url = match[1] if len(match) > 1 and match[1] else match[0]
            else:
                url = match
            # 清理 URL
            url = url.strip().rstrip(')')
            if url not in urls:
                urls.append(url)

    return urls


def scan_cmake_files(directory: Path) -> dict[str, list[str]]:
    """
    扫描目录下所有 .cmake 文件，提取 URL

    Returns:
        {文件名: [url1, url2, ...]}
    """
    results = {}

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    cmake_files = sorted(directory.glob("*.cmake"))

    if not cmake_files:
        print(f"Warning: No .cmake files found in {directory}")
        return results

    for cmake_file in cmake_files:
        urls = extract_urls_from_cmake(cmake_file)
        if urls:
            results[cmake_file.name] = urls
        else:
            print(f"[{cmake_file.name}] No URLs found")

    return results


def get_all_urls(directory: Path) -> list[str]:
    """
    获取所有 .cmake 文件中的 URL合并为单一列表
    """
    results = scan_cmake_files(directory)
    all_urls = []
    for urls in results.values():
        all_urls.extend(urls)
    return list(dict.fromkeys(all_urls))  # 去重保持顺序

if __name__ == "__main__":

    script_path = Path(__file__).resolve()
    cmake_dir = script_path.parent.parent.parent / "cmake" / "third_party"
    all_urls = get_all_urls(cmake_dir)
    down_files_native(all_urls)