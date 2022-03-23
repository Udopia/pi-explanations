/*************************************************************************************************
CNFTools -- Copyright (c) 2021, Markus Iser, KIT - Karlsruhe Institute of Technology

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 **************************************************************************************************/

#ifndef SRC_UTIL_RESOURCELIMITS_H_
#define SRC_UTIL_RESOURCELIMITS_H_

#include <iomanip>
#include <iostream>
#include <cstdint>
#include <exception>

#ifdef _WIN32
    #include <Windows.h>
    #include <psapi.h>
#else
    #include <time.h>
    #include <sys/time.h>
    #include <unistd.h>
    #include <sys/resource.h>
    #include <csignal>

    #ifdef __APPLE__
        #include <mach/mach.h>
    #else
        #include <stdio.h>
    #endif
#endif


struct ResourceLimitsExceeded : public std::exception {
    const char* what() const throw() {
        return "Exceeded Resource Limits";
    }
};

struct TimeLimitExceeded : public std::exception {
    const char* what() const throw() {
        return "Exceeded Time Limit";
    }
};

struct MemoryLimitExceeded : public std::exception {
    const char* what() const throw() {
        return "Exceeded Memory Limit";
    }
};

struct FileSizeLimitExceeded : public std::exception {
    const char* what() const throw() {
        return "Exceeded File Size Limit";
    }
};

struct ResourceLimitsNotSupported : public std::exception {
    const char* what() const throw() {
        return "rlimit is not supported by windows";
    }
};

struct rlimit cpu_limit;
static void timeout(int signal) {
    setrlimit(RLIMIT_CPU, &cpu_limit);
    throw TimeLimitExceeded();
}

struct rlimit as_limit;
static void memout() {
    setrlimit(RLIMIT_AS, &as_limit);
    throw MemoryLimitExceeded();
}

struct rlimit fsize_limit;
static void fileout(int signal) {
    setrlimit(RLIMIT_FSIZE, &fsize_limit);
    throw FileSizeLimitExceeded();
}


class ResourceLimits {
    unsigned rlim_;  // runtime limit (seconds)
    unsigned mlim_;  // memory limit (mega bytes)
    unsigned flim_;  // file size limit (mega bytes)

    unsigned time_;

 public:
    explicit ResourceLimits(unsigned rlim = 0, unsigned mlim = 0, unsigned flim = 0)
     : rlim_(rlim), mlim_(mlim), flim_(flim) {
        time_ = get_cpu_time();
    }

    unsigned get_runtime() const {
        return get_cpu_time() - time_;
    }

    unsigned get_memory() const {
        return get_peak_memory();
    }

    bool within_memory_limit() const {
        // std::cout << "Memory: " << get_memory() << " MB" << std::endl;
        return mlim_ == 0 || get_memory() <= mlim_;
    }

    bool within_time_limit() const {
        // std::cout << "Runtime: " << get_runtime() << " sec" << std::endl;
        return rlim_ == 0 || get_runtime() <= rlim_;
    }

    bool within_limits() const {
        return within_time_limit() && within_memory_limit();
    }

    void within_limits_or_throw() const {
        if (!within_limits()) throw ResourceLimitsExceeded();
    }

    void set_rlimits() const {
    #ifdef _WIN32
        throw ResourceLimitsNotSupported();
    #else
        struct rlimit limit;

        if (mlim_ > 0) {
            getrlimit(RLIMIT_AS, &limit);
            uint64_t mlim = static_cast<uint64_t>(mlim_) << 20;  // mega bytes to bytes
            if (mlim <= limit.rlim_max) {
                limit.rlim_cur = mlim;
            } else {
                limit.rlim_cur = limit.rlim_max;
            }

            if (setrlimit(RLIMIT_AS, &limit) != 0) {
                std::cerr << "Warning: Memory limit could not be set" << std::endl;
            }

            /**
             * Thanks to Sascha Witt for pointing me to std::set_new_handler().
             * 
             * For graceful out-of-memory handling, the new-handler 'memout()' expands memory limits before exiting. 
             * Otherwise, python process pool implementations either stall or crash, depending on the used library. 
             */
            as_limit = limit;
            as_limit.rlim_cur = limit.rlim_max;
            std::set_new_handler(memout);
        }

        if (rlim_ > 0) {
            getrlimit(RLIMIT_CPU, &limit);
            uint64_t rlim = static_cast<uint64_t>(rlim_);  // seconds
            if (rlim <= limit.rlim_max) {
                limit.rlim_cur = rlim;
            } else {
                limit.rlim_cur = limit.rlim_max;
            }

            if (setrlimit(RLIMIT_CPU, &limit) != 0) {
                std::cerr << "Warning: Runtime limit could not be set" << std::endl;
            }

            cpu_limit = limit;
            cpu_limit.rlim_cur = limit.rlim_max;
            signal(SIGXCPU, timeout);
        }

        if (flim_ > 0) {
            getrlimit(RLIMIT_FSIZE, &limit);
            uint64_t flim = static_cast<uint64_t>(flim_) << 20;  // mega bytes to bytes
            if (flim <= limit.rlim_max) {
                limit.rlim_cur = flim;
            } else {
                limit.rlim_cur = limit.rlim_max;
            }

            if (setrlimit(RLIMIT_FSIZE, &limit) != 0) {
                std::cerr << "Warning: File size limit could not be set" << std::endl;
            }

            fsize_limit = limit;
            fsize_limit.rlim_cur = limit.rlim_max;
            signal(SIGXFSZ, fileout);
        }

        #endif
    }

 private:
#ifdef _WIN32
    // wallclock time in seconds
    unsigned get_wallclock_time() const {
        LARGE_INTEGER time, freq;
        if (!QueryPerformanceFrequency(&freq)) {
            return 0;
        }
        if (!QueryPerformanceCounter(&time)) {
            return 0;
        }
        return static_cast<unsigned>(time.QuadPart / freq.QuadPart);
    }

    // cpu time in seconds
    unsigned get_cpu_time() const {
        FILETIME a, b, c, d;
        if (GetProcessTimes(GetCurrentProcess(), &a, &b, &c, &d) != 0) {
            uint64_t time = static_cast<uint64_t>(d.dwHighDateTime) << 32 | d.dwLowDateTime;  // 100-nanosecond intervals
            return static_cast<unsigned>(time / 10000000);
        } else {
            return 0;
        }
    }

    // peak memory in megabytes
    unsigned get_peak_memory() const {
        PROCESS_MEMORY_COUNTERS info;
        GetProcessMemoryInfo( GetCurrentProcess( ), &info, sizeof(info) );
        return static_cast<unsigned>(info.PeakWorkingSetSize) >> 20;
    }

    // current memory in megabytes
    unsigned get_current_memory() const {
        PROCESS_MEMORY_COUNTERS info;
        GetProcessMemoryInfo( GetCurrentProcess( ), &info, sizeof(info) );
        return static_cast<unsigned>(info.WorkingSetSize) >> 20;
    }
#else
    // wallclock time in seconds
    unsigned get_wallclock_time() const {
        struct timeval time;
        if (gettimeofday(&time, NULL)) {
            return 0;
        }
        return static_cast<unsigned>(time.tv_sec);
    }

    // cpu time in seconds
    unsigned get_cpu_time() const {
        return static_cast<unsigned>(clock() / CLOCKS_PER_SEC);
    }

    // peak memory in megabytes
    unsigned get_peak_memory() const {
        struct rusage rusage;
        getrusage(RUSAGE_SELF, &rusage);
        #if defined(__APPLE__) && defined(__MACH__)
            return static_cast<unsigned>(rusage.ru_maxrss) >> 20;
        #else
            return static_cast<unsigned>(rusage.ru_maxrss) >> 10;
        #endif
    }

    // current memory in megabytes
    unsigned get_current_memory() const {
    #ifdef __APPLE__
        struct mach_task_basic_info info;
        mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
        if (task_info( mach_task_self( ), MACH_TASK_BASIC_INFO, (task_info_t)&info, &infoCount ) != KERN_SUCCESS)
            return 0;
        return static_cast<unsigned>(info.resident_size) >> 20;
    #else
        int64_t rss = 0;
        FILE* fp = fopen("/proc/self/statm", "r");
        if (fp == NULL) {
            return 0;
        }
        if (fscanf(fp, "%*s%ld", &rss) != 1) {
            fclose(fp);
            return 0;
        }
        fclose(fp);
        return static_cast<unsigned>(rss * sysconf(_SC_PAGESIZE)) >> 20;
    #endif
    }
#endif
};

#endif  // SRC_UTIL_RESOURCELIMITS_H_
