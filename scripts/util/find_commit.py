#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import argparse
import os
import subprocess
import sys


def find_commit(repo_path: str, public_commit_id: str) -> str:
    if not os.path.isdir(repo_path):
        print(f"Error: Repository path does not exist: {repo_path}", file=sys.stderr)
        return ""
    
    if not os.path.isdir(os.path.join(repo_path, ".git")):
        print(f"Error: Not a git repository: {repo_path}", file=sys.stderr)
        return ""
    
    try:
        result = subprocess.run(
            ["git", "log", "--all", "--format=%H", "--grep=" + public_commit_id],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            print(f"Error: Git log command failed: {result.stderr}", file=sys.stderr)
            return ""
        
        output = result.stdout.strip()
        if not output:
            print(f"Warning: No commit found with public commit id: {public_commit_id}", file=sys.stderr)
            return ""
        
        lines = output.split("\n")
        first_line = lines[0].strip()
        commit_id = first_line.split()[0] if first_line else ""
        
        if not commit_id:
            print(f"Error: Failed to parse commit id from: {first_line}", file=sys.stderr)
            return ""
        
        return commit_id
        
    except subprocess.SubprocessError as e:
        print(f"Error: Failed to execute git command: {e}", file=sys.stderr)
        return ""
    except Exception as e:
        print(f"Error: Unexpected error: {e}", file=sys.stderr)
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-path",
        required=True,
        help="Path to the git repository"
    )
    parser.add_argument(
        "--commit",
        required=True,
        help="Commit id to search for"
    )
    
    args = parser.parse_args()
    
    commit = find_commit(args.repo_path, args.commit)
    
    if commit:
        print(commit)
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()