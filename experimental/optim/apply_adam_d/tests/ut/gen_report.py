#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""
将 gtest 的 JSON 输出（真实 UT 执行结果）转换为 test-report.json。

本脚本只做"格式转换"，不伪造任何通过证据：
  - status / total / passed / failed / 每条 case 的 status 全部从 gtest JSON 实测结果推导；
  - 若 gtest JSON 缺失或解析失败 -> 报错并以非 0 退出（不会写出"全 passed"）。

用法: gen_report.py <gtest_json_in> <report_json_out>
"""
import json
import sys


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: gen_report.py <gtest_json_in> <report_json_out>\n")
        return 2
    gtest_path, report_path = sys.argv[1], sys.argv[2]

    try:
        with open(gtest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:  # gtest 没有产生 JSON（崩溃/未编译/未运行）
        sys.stderr.write("[gen_report] cannot read gtest json '%s': %s\n" % (gtest_path, e))
        # 写一个明确 failed 的报告，避免下游误读为通过
        report = {"status": "failed",
                  "summary": {"total": 0, "passed": 0, "failed": 0},
                  "results": [],
                  "error": "gtest json missing or invalid: %s" % e}
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return 1

    results = []
    passed = 0
    failed = 0
    for suite in data.get("testsuites", []):
        suite_name = suite.get("name", "")
        for case in suite.get("testsuite", []):
            name = case.get("name", "")
            classname = case.get("classname", suite_name)
            case_id = "%s.%s" % (classname, name)
            case_failures = case.get("failures", [])
            case_result = case.get("result", "COMPLETED")
            ok = (not case_failures) and (case_result == "COMPLETED")
            if ok:
                passed += 1
                results.append({"case_id": case_id, "status": "passed"})
            else:
                failed += 1
                entry = {"case_id": case_id, "status": "failed"}
                if case_failures:
                    msgs = [fl.get("failure", "") for fl in case_failures]
                    entry["detail"] = "; ".join(m.splitlines()[0] for m in msgs if m)
                results.append(entry)

    total = passed + failed
    status = "passed" if (total > 0 and failed == 0) else "failed"
    report = {
        "status": status,
        "summary": {"total": total, "passed": passed, "failed": failed},
        "results": results,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    sys.stderr.write("[gen_report] total=%d passed=%d failed=%d status=%s\n"
                     % (total, passed, failed, status))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
