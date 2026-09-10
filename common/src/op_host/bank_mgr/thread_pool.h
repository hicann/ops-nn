/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef __AOE_RUNTIME_KB_COMMON_THREAD_POOL_H__
#define __AOE_RUNTIME_KB_COMMON_THREAD_POOL_H__
#include <cstdint>
#include <pthread.h>
#include <thread>
#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <future>
#include <atomic>
#include <functional>
#include "error_util.h"
#include "kb_common.h"
#include "system_utils.h"

namespace RuntimeKb {
constexpr int32_t MAX_THREAD_POOL_NUM = 64;
constexpr int32_t DEFAULT_THREAD_POOL_NUM = 8;

class ThreadPool {
public:
    static ThreadPool* GetInstance()
    {
        static ThreadPool instance;
        return &instance;
    }

    template <typename F, typename... Args>
    auto Submit(F&& func, Args&&... args) -> std::future<decltype(func(args...))>
    {
        using RetType = decltype(func(args...));
        std::lock_guard<std::mutex> lk(workerMtx_);
        std::function<RetType()> f = std::bind(std::forward<F>(func), std::forward<Args>(args)...);
        auto task = MakeShared<std::packaged_task<RetType()>>(f);
        if (task == nullptr) {
            std::future<RetType> errFuture;
            return errFuture;
        }
        std::future<RetType> future = task->get_future();
        if (idle_ == 0 && running_.load() < size_) {
            running_++;
            static_cast<void>(workers_.emplace_back([this] { PushTask(); }));
        }
        if (this->idle_.load() > 0) {
            this->idle_--;
        }

        {
            std::unique_lock<std::mutex> lock(queTaskMtx_);
            static_cast<void>(taskQueue_.emplace([task]() { (*task)(); }));
        }
        cvRun_.notify_one();
        return future;
    }

    int32_t GetSize() { return size_; }

private:
    ThreadPool() {};
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void PushTask()
    {
        std::string threadName = "AOE_RTKB";
        auto ret = pthread_setname_np(pthread_self(), threadName.c_str());
        if (ret != 0) {
            CANNKB_LOGW("can not set thread name, ret = %d", ret);
        }
        for (;;) {
            std::function<void()> doTask;
            {
                std::unique_lock<std::mutex> lock(this->queTaskMtx_);
                this->cvRun_.wait(lock, [this] { return this->stop_.load() || !this->taskQueue_.empty(); });
                if (this->stop_.load() && this->taskQueue_.empty()) {
                    return;
                }
                doTask = std::move(this->taskQueue_.front());
                this->taskQueue_.pop();
            }
            doTask();
            this->idle_++;
        }
    }

    void Join()
    {
        for (auto& it : workers_) {
            if (it.joinable()) {
                it.join();
            }
        }
    }

    void Stop()
    {
        if (!stop_.load()) {
            {
                std::unique_lock<std::mutex> lock(this->queTaskMtx_);
                stop_.store(true);
            }
            cvRun_.notify_all();
            Join();
        }
    }

    ~ThreadPool() { Stop(); }

    std::atomic<bool> stop_;
    std::atomic<int32_t> running_;
    std::atomic<int32_t> idle_;
    int32_t size_ = std::min(SystemUtils::GetCpuCoreNum() > 0 ? SystemUtils::GetCpuCoreNum() : DEFAULT_THREAD_POOL_NUM,
                             MAX_THREAD_POOL_NUM);
    std::mutex queTaskMtx_;
    std::mutex workerMtx_;
    std::queue<std::function<void()>> taskQueue_;
    std::vector<std::thread> workers_;
    std::condition_variable cvRun_;
};
} // namespace RuntimeKb
#endif // __AOE_RUNTIME_KB_COMMON_THREAD_POOL_H__
