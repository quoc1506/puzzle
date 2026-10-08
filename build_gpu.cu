// ============================================================================
// BITCOIN PUZZLE SOLVER - ULTRA-OPTIMIZED NVIDIA CUDA GPU SOLVER (Zero-Copy Register Pipeline + Warp Synchronous)
// Designed for NVIDIA GPUs (Tesla T4, RTX 3080/3090, RTX 4090, A100, H100)
// Features:
//   - Inlined PTX assembly for 256-bit multiprecision arithmetic
//   - Single-cycle 3-input bitwise logic via hardware lop3.b32 instructions
//   - 16-way Lockstep SIMD Montgomery Batch Inversion (0 warp divergence)
//   - In-register SHA-256 + RIPEMD-160 pipeline with 64-bit early hash rejection
//   - Dynamic SM detection & grid dimension auto-tuning
// ============================================================================

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <cstring>
#include <csignal>
#include <cstdint>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <random>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#if defined(__NVCC__) || defined(__CUDACC__)
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_DEV __device__
#define CUDA_GLOBAL __global__
#define CUDA_INLINE __forceinline__
#define CUDA_CONSTANT __constant__
#else
#define CUDA_HOSTDEV
#define CUDA_DEV
#define CUDA_GLOBAL
#define CUDA_INLINE inline
#define CUDA_CONSTANT
struct uint3 { unsigned int x, y, z; };
struct dim3 { unsigned int x, y, z; dim3(unsigned int _x=1, unsigned int _y=1, unsigned int _z=1): x(_x), y(_y), z(_z) {} };
static uint3 threadIdx = {0,0,0};
static uint3 blockIdx = {0,0,0};
static dim3 blockDim = {1,1,1};
static dim3 gridDim = {1,1,1};
typedef int cudaError_t;
#define cudaSuccess 0
#define cudaMemcpyDeviceToHost 2
#define cudaMemcpyHostToDevice 1
#define cudaLimitStackSize 0
inline const char* cudaGetErrorString(cudaError_t) { return "ok"; }
inline cudaError_t cudaSetDevice(int) { return 0; }
struct cudaDeviceProp { char name[256]; int major; int minor; int multiProcessorCount; size_t totalGlobalMem; int regsPerBlock; };
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int) { if (p) { p->multiProcessorCount = 40; } return 0; }
inline cudaError_t cudaDeviceSetLimit(int, size_t) { return 0; }
inline cudaError_t cudaDeviceSynchronize() { return 0; }
inline cudaError_t cudaGetLastError() { return 0; }
template<typename T> inline cudaError_t cudaMemcpyToSymbol(T& dst, const void* src, size_t count, size_t offset=0, int kind=0) {
    std::memcpy(((char*)&dst) + offset, src, count);
    return 0;
}
template<typename T> inline cudaError_t cudaMemcpyFromSymbol(void* dst, const T& src, size_t count, size_t offset=0, int kind=0) {
    std::memcpy(dst, ((const char*)&src) + offset, count);
    return 0;
}
template<typename T> inline cudaError_t cudaMalloc(T** devPtr, size_t size) {
    *devPtr = (T*)std::malloc(size);
    return 0;
}
inline cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
    std::memset(devPtr, value, count);
    return 0;
}
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, int kind=0) {
    std::memcpy(dst, src, count);
    return 0;
}
inline cudaError_t cudaFree(void* devPtr) {
    if (devPtr) std::free(devPtr);
    return 0;
}
template<typename T> inline T atomicExch(T* address, T val) {
    T old = *address;
    *address = val;
    return old;
}
#endif

#ifndef CURL_STATICLIB
#define CURL_STATICLIB
#endif

#include <curl/curl.h>

#if defined(__clang__)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

static std::atomic<bool> g_running(true);

void sigint_handler(int signum) {
    (void)signum;
    g_running = false;
}

template <typename T>
CUDA_HOSTDEV CUDA_INLINE T host_min(T a, T b) {
    return (a < b) ? a : b;
}

inline void portable_sleep_ms(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

typedef __uint128_t u128;

struct u256 {
    __uint128_t low;
    __uint128_t high;

    CUDA_HOSTDEV u256() : low(0), high(0) {}
    CUDA_HOSTDEV u256(int v) : low((uint64_t)v), high(0) {}
    CUDA_HOSTDEV u256(uint32_t v) : low(v), high(0) {}
    CUDA_HOSTDEV u256(uint64_t v) : low(v), high(0) {}
    CUDA_HOSTDEV u256(__uint128_t l) : low(l), high(0) {}
    CUDA_HOSTDEV u256(__uint128_t l, __uint128_t h) : low(l), high(h) {}

    CUDA_HOSTDEV CUDA_INLINE bool is_zero() const {
        return (low | high) == 0;
    }

    CUDA_HOSTDEV CUDA_INLINE void to_bytes_be(uint8_t out[32]) const {
        for (int i = 0; i < 16; ++i) {
            out[31 - i] = (uint8_t)(low >> (i * 8));
            out[15 - i] = (uint8_t)(high >> (i * 8));
        }
    }
};

CUDA_HOSTDEV CUDA_INLINE bool operator==(const u256& a, const u256& b) {
    return a.low == b.low && a.high == b.high;
}
CUDA_HOSTDEV CUDA_INLINE bool operator!=(const u256& a, const u256& b) {
    return !(a == b);
}
CUDA_HOSTDEV CUDA_INLINE bool operator<(const u256& a, const u256& b) {
    if (a.high != b.high) return a.high < b.high;
    return a.low < b.low;
}
CUDA_HOSTDEV CUDA_INLINE bool operator<=(const u256& a, const u256& b) {
    return (a < b) || (a == b);
}
CUDA_HOSTDEV CUDA_INLINE bool operator>(const u256& a, const u256& b) {
    return b < a;
}
CUDA_HOSTDEV CUDA_INLINE bool operator>=(const u256& a, const u256& b) {
    return !(a < b);
}

CUDA_HOSTDEV CUDA_INLINE u256 operator+(const u256& a, const u256& b) {
    u256 r;
    r.low = a.low + b.low;
    r.high = a.high + b.high + (r.low < a.low ? 1 : 0);
    return r;
}
CUDA_HOSTDEV CUDA_INLINE u256 operator+(const u256& a, uint64_t b) {
    u256 r;
    r.low = a.low + b;
    r.high = a.high + (r.low < a.low ? 1 : 0);
    return r;
}
CUDA_HOSTDEV CUDA_INLINE u256 operator-(const u256& a, const u256& b) {
    u256 r;
    r.low = a.low - b.low;
    r.high = a.high - b.high - (a.low < b.low ? 1 : 0);
    return r;
}
CUDA_HOSTDEV CUDA_INLINE u256 operator-(const u256& a, uint64_t b) {
    u256 r;
    r.low = a.low - b;
    r.high = a.high - (a.low < b ? 1 : 0);
    return r;
}

u256 parse_u256(const std::string& s) {
    u256 r;
    if (s.empty()) return r;
    bool is_hex = (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X'));
    if (!is_hex) {
        for (char c : s) {
            if ((c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
                is_hex = true;
                break;
            }
        }
    }
    if (is_hex) {
        size_t start = (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) ? 2 : 0;
        for (size_t i = start; i < s.size(); ++i) {
            char c = s[i];
            int v = 0;
            if (c >= '0' && c <= '9') v = c - '0';
            else if (c >= 'a' && c <= 'f') v = c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') v = c - 'A' + 10;
            else continue;
            uint64_t carry = (uint64_t)(r.low >> 124);
            r.low = (r.low << 4) | v;
            r.high = (r.high << 4) | carry;
        }
    } else {
        for (char c : s) {
            if (c >= '0' && c <= '9') {
                int digit = c - '0';
                u128 low_prod = (r.low & 0xFFFFFFFFFFFFFFFFULL) * 10 + digit;
                u128 low_carry = low_prod >> 64;
                u128 high_part = (r.low >> 64) * 10 + low_carry;
                r.low = (high_part << 64) | (low_prod & 0xFFFFFFFFFFFFFFFFULL);
                u128 high_carry = high_part >> 64;

                u128 h_low_prod = (r.high & 0xFFFFFFFFFFFFFFFFULL) * 10 + high_carry;
                u128 h_low_carry = h_low_prod >> 64;
                u128 h_high_part = (r.high >> 64) * 10 + h_low_carry;
                r.high = (h_high_part << 64) | (h_low_prod & 0xFFFFFFFFFFFFFFFFULL);
            }
        }
    }
    return r;
}

std::string u256_to_dec(u256 val) {
    if (val.is_zero()) return "0";
    std::string s;
    while (!val.is_zero()) {
        u128 rem = 0;
        u128 h_hi = val.high >> 64;
        u128 q_h_hi = (rem << 64 | h_hi) / 10;
        rem = (rem << 64 | h_hi) % 10;

        u128 h_lo = val.high & 0xFFFFFFFFFFFFFFFFULL;
        u128 q_h_lo = (rem << 64 | h_lo) / 10;
        rem = (rem << 64 | h_lo) % 10;

        val.high = (q_h_hi << 64) | q_h_lo;

        u128 l_hi = val.low >> 64;
        u128 q_l_hi = (rem << 64 | l_hi) / 10;
        rem = (rem << 64 | l_hi) % 10;

        u128 l_lo = val.low & 0xFFFFFFFFFFFFFFFFULL;
        u128 q_l_lo = (rem << 64 | l_lo) / 10;
        rem = (rem << 64 | l_lo) % 10;

        val.low = (q_l_hi << 64) | q_l_lo;

        s.push_back((char)('0' + rem));
    }
    std::reverse(s.begin(), s.end());
    return s;
}

std::string u256_to_hex64(u256 val) {
    std::stringstream ss;
    uint8_t bytes[32];
    val.to_bytes_be(bytes);
    for (int i = 0; i < 32; ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)bytes[i];
    }
    return ss.str();
}

// Standalone SHA-256 for Base58Check decoding (Zero OpenSSL dependency)
static inline uint32_t b58_rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }

static void b58_sha256(const uint8_t* data, size_t len, uint8_t out[32]) {
    static const uint32_t K[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
    };
    uint32_t h[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    std::vector<uint8_t> msg(data, data + len);
    msg.push_back(0x80);
    while ((msg.size() % 64) != 56) msg.push_back(0x00);
    uint64_t bit_len = (uint64_t)len * 8ULL;
    for (int i = 7; i >= 0; --i) msg.push_back((uint8_t)((bit_len >> (i * 8)) & 0xff));

    for (size_t chunk = 0; chunk < msg.size(); chunk += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = ((uint32_t)msg[chunk + i * 4] << 24) |
                   ((uint32_t)msg[chunk + i * 4 + 1] << 16) |
                   ((uint32_t)msg[chunk + i * 4 + 2] << 8) |
                   ((uint32_t)msg[chunk + i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = b58_rotr(w[i - 15], 7) ^ b58_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = b58_rotr(w[i - 2], 17) ^ b58_rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
        uint32_t e = h[4], f = h[5], g = h[6], h_val = h[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = b58_rotr(e, 6) ^ b58_rotr(e, 11) ^ b58_rotr(e, 25);
            uint32_t ch = (e & f) ^ ((~e) & g);
            uint32_t temp1 = h_val + S1 + ch + K[i] + w[i];
            uint32_t S0 = b58_rotr(a, 2) ^ b58_rotr(a, 13) ^ b58_rotr(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temp2 = S0 + maj;
            h_val = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += h_val;
    }
    for (int i = 0; i < 8; ++i) {
        out[i * 4]     = (uint8_t)(h[i] >> 24);
        out[i * 4 + 1] = (uint8_t)(h[i] >> 16);
        out[i * 4 + 2] = (uint8_t)(h[i] >> 8);
        out[i * 4 + 3] = (uint8_t)(h[i]);
    }
}

bool b58check_decode_hash160(const std::string& raw_addr, uint8_t out_hash160[20]) {
    std::string addr = raw_addr;
    while (!addr.empty() && (addr.back() == '\r' || addr.back() == '\n' || addr.back() == ' ' || addr.back() == '\"' || addr.back() == '\t')) addr.pop_back();
    while (!addr.empty() && (addr.front() == ' ' || addr.front() == '\"' || addr.front() == '\t')) addr.erase(0, 1);
    static const char* B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    std::vector<uint8_t> bytes = {0};
    int leading_zeros = 0;
    for (char c : addr) {
        if (c == '1' && bytes.size() == 1 && bytes[0] == 0) leading_zeros++;
        else break;
    }
    for (char c : addr) {
        const char* p = std::strchr(B58, c);
        if (!p) return false;
        int carry = (int)(p - B58);
        for (size_t i = 0; i < bytes.size(); ++i) {
            int cur = (int)bytes[i] * 58 + carry;
            bytes[i] = (uint8_t)(cur & 0xff);
            carry = cur >> 8;
        }
        while (carry > 0) {
            bytes.push_back((uint8_t)(carry & 0xff));
            carry >>= 8;
        }
    }
    std::vector<uint8_t> full(leading_zeros, 0);
    for (int i = (int)bytes.size() - 1; i >= 0; --i) {
        full.push_back(bytes[i]);
    }
    if (full.size() != 25) return false;
    uint8_t h1[32], h2[32];
    b58_sha256(full.data(), 21, h1);
    b58_sha256(h1, 32, h2);
    if (std::memcmp(h2, full.data() + 21, 4) != 0) return false;
    std::memcpy(out_hash160, full.data() + 1, 20);
    return true;
}

// Secp256k1 Field Element Representation (4 x 64-bit limbs)
struct Fe {
    uint64_t d[4];
};

CUDA_HOSTDEV CUDA_INLINE bool operator==(const Fe& a, const Fe& b) {
    return (a.d[0] == b.d[0]) && (a.d[1] == b.d[1]) &&
           (a.d[2] == b.d[2]) && (a.d[3] == b.d[3]);
}

CUDA_HOSTDEV CUDA_INLINE bool operator!=(const Fe& a, const Fe& b) {
    return !(a == b);
}

CUDA_HOSTDEV CUDA_INLINE bool fe_is_zero(const Fe& a) {
    return (a.d[0] | a.d[1] | a.d[2] | a.d[3]) == 0;
}

CUDA_HOSTDEV CUDA_INLINE Fe fe_add(const Fe& a, const Fe& b) {
#if defined(__CUDA_ARCH__)
    Fe r;
    uint64_t c0 = 0;
    asm volatile(
        "add.cc.u64 %0, %5, %9;\n\t"
        "addc.cc.u64 %1, %6, %10;\n\t"
        "addc.cc.u64 %2, %7, %11;\n\t"
        "addc.cc.u64 %3, %8, %12;\n\t"
        "addc.u64 %4, 0, 0;\n\t"
        : "=l"(r.d[0]), "=l"(r.d[1]), "=l"(r.d[2]), "=l"(r.d[3]), "=l"(c0)
        : "l"(a.d[0]), "l"(a.d[1]), "l"(a.d[2]), "l"(a.d[3]),
          "l"(b.d[0]), "l"(b.d[1]), "l"(b.d[2]), "l"(b.d[3])
    );
    Fe r_k;
    uint64_t c1 = 0;
    const uint64_t K = 0x1000003D1ULL;
    asm volatile(
        "add.cc.u64 %0, %5, %9;\n\t"
        "addc.cc.u64 %1, %6, 0;\n\t"
        "addc.cc.u64 %2, %7, 0;\n\t"
        "addc.cc.u64 %3, %8, 0;\n\t"
        "addc.u64 %4, 0, 0;\n\t"
        : "=l"(r_k.d[0]), "=l"(r_k.d[1]), "=l"(r_k.d[2]), "=l"(r_k.d[3]), "=l"(c1)
        : "l"(r.d[0]), "l"(r.d[1]), "l"(r.d[2]), "l"(r.d[3]), "l"(K)
    );
    uint64_t need_reduce = c0 | c1;
    r.d[0] = need_reduce ? r_k.d[0] : r.d[0];
    r.d[1] = need_reduce ? r_k.d[1] : r.d[1];
    r.d[2] = need_reduce ? r_k.d[2] : r.d[2];
    r.d[3] = need_reduce ? r_k.d[3] : r.d[3];
    return r;
#elif (defined(__x86_64__) || defined(_M_X64))
    Fe r;
    unsigned char c = 0;
    c = _addcarry_u64(c, a.d[0], b.d[0], (unsigned long long*)&r.d[0]);
    c = _addcarry_u64(c, a.d[1], b.d[1], (unsigned long long*)&r.d[1]);
    c = _addcarry_u64(c, a.d[2], b.d[2], (unsigned long long*)&r.d[2]);
    c = _addcarry_u64(c, a.d[3], b.d[3], (unsigned long long*)&r.d[3]);

    if (__builtin_expect(c != 0, 0)) {
        const uint64_t K = 0x1000003D1ULL;
        c = _addcarry_u64(0, r.d[0], K, (unsigned long long*)&r.d[0]);
        c = _addcarry_u64(c, r.d[1], 0, (unsigned long long*)&r.d[1]);
        c = _addcarry_u64(c, r.d[2], 0, (unsigned long long*)&r.d[2]);
        _addcarry_u64(c, r.d[3], 0, (unsigned long long*)&r.d[3]);
    } else {
        if (__builtin_expect(r.d[3] == 0xFFFFFFFFFFFFFFFFULL &&
            r.d[2] == 0xFFFFFFFFFFFFFFFFULL &&
            r.d[1] == 0xFFFFFFFFFFFFFFFFULL &&
            r.d[0] >= 0xFFFFFFFEFFFFFC2FULL, 0)) {
            r.d[0] -= 0xFFFFFFFEFFFFFC2FULL;
            r.d[1] = 0;
            r.d[2] = 0;
            r.d[3] = 0;
        }
    }
    return r;
#elif defined(__SIZEOF_INT128__) && !defined(__CUDA_ARCH__)
    Fe r;
    u128 c = (u128)a.d[0] + b.d[0];
    r.d[0] = (uint64_t)c; c >>= 64;
    c += (u128)a.d[1] + b.d[1];
    r.d[1] = (uint64_t)c; c >>= 64;
    c += (u128)a.d[2] + b.d[2];
    r.d[2] = (uint64_t)c; c >>= 64;
    c += (u128)a.d[3] + b.d[3];
    r.d[3] = (uint64_t)c;
    uint64_t carry = (uint64_t)(c >> 64);

    if (carry) {
        const uint64_t K = 0x1000003D1ULL;
        u128 c2 = (u128)r.d[0] + K;
        r.d[0] = (uint64_t)c2; c2 >>= 64;
        c2 += r.d[1]; r.d[1] = (uint64_t)c2; c2 >>= 64;
        c2 += r.d[2]; r.d[2] = (uint64_t)c2; c2 >>= 64;
        r.d[3] += (uint64_t)c2;
    } else {
        if (r.d[3] == 0xFFFFFFFFFFFFFFFFULL &&
            r.d[2] == 0xFFFFFFFFFFFFFFFFULL &&
            r.d[1] == 0xFFFFFFFFFFFFFFFFULL &&
            r.d[0] >= 0xFFFFFFFEFFFFFC2FULL) {
            r.d[0] -= 0xFFFFFFFEFFFFFC2FULL;
            r.d[1] = 0;
            r.d[2] = 0;
            r.d[3] = 0;
        }
    }
    return r;
#else
    Fe r;
    uint64_t c = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint64_t s = a.d[i] + b.d[i];
        uint64_t c1 = (s < a.d[i]);
        s += c;
        uint64_t c2 = (s < c);
        r.d[i] = s;
        c = c1 + c2;
    }
    if (c) {
        const uint64_t K = 0x1000003D1ULL;
        uint64_t s = r.d[0] + K;
        uint64_t c2 = (s < K);
        r.d[0] = s;
        #pragma unroll
        for (int i = 1; i < 4; ++i) {
            s = r.d[i] + c2;
            c2 = (s < c2);
            r.d[i] = s;
        }
    } else {
        if (r.d[3] == 0xFFFFFFFFFFFFFFFFULL &&
            r.d[2] == 0xFFFFFFFFFFFFFFFFULL &&
            r.d[1] == 0xFFFFFFFFFFFFFFFFULL &&
            r.d[0] >= 0xFFFFFFFEFFFFFC2FULL) {
            r.d[0] -= 0xFFFFFFFEFFFFFC2FULL;
            r.d[1] = 0;
            r.d[2] = 0;
            r.d[3] = 0;
        }
    }
    return r;
#endif
}

CUDA_HOSTDEV CUDA_INLINE Fe fe_sub(const Fe& a, const Fe& b) {
#if defined(__CUDA_ARCH__)
    Fe r;
    uint64_t borrow = 0;
    asm volatile(
        "sub.cc.u64 %0, %5, %9;\n\t"
        "subc.cc.u64 %1, %6, %10;\n\t"
        "subc.cc.u64 %2, %7, %11;\n\t"
        "subc.cc.u64 %3, %8, %12;\n\t"
        "subc.u64 %4, 0, 0;\n\t"
        : "=l"(r.d[0]), "=l"(r.d[1]), "=l"(r.d[2]), "=l"(r.d[3]), "=l"(borrow)
        : "l"(a.d[0]), "l"(a.d[1]), "l"(a.d[2]), "l"(a.d[3]),
          "l"(b.d[0]), "l"(b.d[1]), "l"(b.d[2]), "l"(b.d[3])
    );
    const uint64_t K = 0x1000003D1ULL;
    uint64_t sub_k = borrow ? K : 0ULL;
    asm volatile(
        "sub.cc.u64 %0, %0, %4;\n\t"
        "subc.cc.u64 %1, %1, 0;\n\t"
        "subc.cc.u64 %2, %2, 0;\n\t"
        "subc.u64 %3, %3, 0;\n\t"
        : "+l"(r.d[0]), "+l"(r.d[1]), "+l"(r.d[2]), "+l"(r.d[3])
        : "l"(sub_k)
    );
    return r;
#elif (defined(__x86_64__) || defined(_M_X64))
    Fe r;
    unsigned char borrow = 0;
    borrow = _subborrow_u64(borrow, a.d[0], b.d[0], (unsigned long long*)&r.d[0]);
    borrow = _subborrow_u64(borrow, a.d[1], b.d[1], (unsigned long long*)&r.d[1]);
    borrow = _subborrow_u64(borrow, a.d[2], b.d[2], (unsigned long long*)&r.d[2]);
    borrow = _subborrow_u64(borrow, a.d[3], b.d[3], (unsigned long long*)&r.d[3]);

    if (__builtin_expect(borrow != 0, 0)) {
        const uint64_t K = 0x1000003D1ULL;
        borrow = _subborrow_u64(0, r.d[0], K, (unsigned long long*)&r.d[0]);
        borrow = _subborrow_u64(borrow, r.d[1], 0, (unsigned long long*)&r.d[1]);
        borrow = _subborrow_u64(borrow, r.d[2], 0, (unsigned long long*)&r.d[2]);
        _subborrow_u64(borrow, r.d[3], 0, (unsigned long long*)&r.d[3]);
    }
    return r;
#elif defined(__SIZEOF_INT128__) && !defined(__CUDA_ARCH__)
    Fe r;
    u128 c = (u128)a.d[0] - b.d[0];
    r.d[0] = (uint64_t)c;
    c = (u128)a.d[1] - b.d[1] - ((c >> 64) & 1);
    r.d[1] = (uint64_t)c;
    c = (u128)a.d[2] - b.d[2] - ((c >> 64) & 1);
    r.d[2] = (uint64_t)c;
    c = (u128)a.d[3] - b.d[3] - ((c >> 64) & 1);
    r.d[3] = (uint64_t)c;

    if ((c >> 64) & 1) {
        const uint64_t K = 0x1000003D1ULL;
        c = (u128)r.d[0] - K;
        r.d[0] = (uint64_t)c;
        c = (u128)r.d[1] - ((c >> 64) & 1);
        r.d[1] = (uint64_t)c;
        c = (u128)r.d[2] - ((c >> 64) & 1);
        r.d[2] = (uint64_t)c;
        r.d[3] -= (uint64_t)((c >> 64) & 1);
    }
    return r;
#else
    Fe r;
    uint64_t borrow = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint64_t diff = a.d[i] - b.d[i];
        uint64_t b1 = (a.d[i] < b.d[i]);
        uint64_t diff2 = diff - borrow;
        uint64_t b2 = (diff < borrow);
        r.d[i] = diff2;
        borrow = b1 + b2;
    }
    if (borrow) {
        const uint64_t K = 0x1000003D1ULL;
        uint64_t diff = r.d[0] - K;
        uint64_t b2 = (r.d[0] < K);
        r.d[0] = diff;
        #pragma unroll
        for (int i = 1; i < 4; ++i) {
            diff = r.d[i] - b2;
            b2 = (r.d[i] < b2);
            r.d[i] = diff;
        }
    }
    return r;
#endif
}

CUDA_HOSTDEV CUDA_INLINE Fe fe_reduce(uint64_t t[8]) {
    const uint64_t K = 0x1000003D1ULL;
    uint64_t c = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint64_t hi = t[4 + i];
        uint64_t prod_lo = hi * K;
#if defined(__CUDA_ARCH__)
        uint64_t prod_hi = __umul64hi(hi, K);
#elif defined(__SIZEOF_INT128__)
        uint64_t prod_hi = (uint64_t)(((unsigned __int128)hi * K) >> 64);
#else
        uint64_t prod_hi = 0;
#endif
        uint64_t s = t[i] + prod_lo;
        uint64_t c1 = (s < t[i]);
        s += c;
        uint64_t c2 = (s < c);
        t[i] = s;
        c = prod_hi + c1 + c2;
    }

    while (c > 0) {
        uint64_t prod_lo = c * K;
#if defined(__CUDA_ARCH__)
        uint64_t prod_hi = __umul64hi(c, K);
#elif defined(__SIZEOF_INT128__)
        uint64_t prod_hi = (uint64_t)(((unsigned __int128)c * K) >> 64);
#else
        uint64_t prod_hi = 0;
#endif
        c = 0;

        uint64_t s = t[0] + prod_lo;
        uint64_t carry = (s < t[0]);
        t[0] = s;

        s = t[1] + prod_hi;
        uint64_t c1 = (s < t[1]);
        s += carry;
        carry = c1 + (s < carry);
        t[1] = s;

        s = t[2] + carry;
        carry = (s < carry);
        t[2] = s;

        s = t[3] + carry;
        carry = (s < carry);
        t[3] = s;
        c = carry;
    }

    if (t[3] == 0xFFFFFFFFFFFFFFFFULL &&
        t[2] == 0xFFFFFFFFFFFFFFFFULL &&
        t[1] == 0xFFFFFFFFFFFFFFFFULL &&
        t[0] >= 0xFFFFFFFEFFFFFC2FULL) {
        t[0] -= 0xFFFFFFFEFFFFFC2FULL;
        t[1] = 0;
        t[2] = 0;
        t[3] = 0;
    }

    Fe r;
    r.d[0] = t[0]; r.d[1] = t[1]; r.d[2] = t[2]; r.d[3] = t[3];
    return r;
}

CUDA_HOSTDEV CUDA_INLINE Fe fe_mul(const Fe& a, const Fe& b) {
#if defined(__SIZEOF_INT128__) || defined(__CUDA_ARCH__)
    uint64_t a0 = a.d[0], a1 = a.d[1], a2 = a.d[2], a3 = a.d[3];
    uint64_t b0 = b.d[0], b1 = b.d[1], b2 = b.d[2], b3 = b.d[3];

    // Branchless 4x4 limb schoolbook multiplier
    u128 p = (u128)a0 * b0;
    uint64_t t0 = (uint64_t)p;
    u128 c = p >> 64;

    p = (u128)a0 * b1 + c;
    uint64_t t1 = (uint64_t)p;
    c = p >> 64;

    p = (u128)a0 * b2 + c;
    uint64_t t2 = (uint64_t)p;
    c = p >> 64;

    p = (u128)a0 * b3 + c;
    uint64_t t3 = (uint64_t)p;
    uint64_t t4 = (uint64_t)(p >> 64);
    uint64_t t5 = 0, t6 = 0, t7 = 0;

    p = (u128)a1 * b0 + t1;
    t1 = (uint64_t)p;
    c = p >> 64;

    p = (u128)a1 * b1 + t2 + c;
    t2 = (uint64_t)p;
    c = p >> 64;

    p = (u128)a1 * b2 + t3 + c;
    t3 = (uint64_t)p;
    c = p >> 64;

    p = (u128)a1 * b3 + t4 + c;
    t4 = (uint64_t)p;
    t5 = (uint64_t)(p >> 64);

    p = (u128)a2 * b0 + t2;
    t2 = (uint64_t)p;
    c = p >> 64;

    p = (u128)a2 * b1 + t3 + c;
    t3 = (uint64_t)p;
    c = p >> 64;

    p = (u128)a2 * b2 + t4 + c;
    t4 = (uint64_t)p;
    c = p >> 64;

    p = (u128)a2 * b3 + t5 + c;
    t5 = (uint64_t)p;
    t6 = (uint64_t)(p >> 64);

    p = (u128)a3 * b0 + t3;
    t3 = (uint64_t)p;
    c = p >> 64;

    p = (u128)a3 * b1 + t4 + c;
    t4 = (uint64_t)p;
    c = p >> 64;

    p = (u128)a3 * b2 + t5 + c;
    t5 = (uint64_t)p;
    c = p >> 64;

    p = (u128)a3 * b3 + t6 + c;
    t6 = (uint64_t)p;
    t7 = (uint64_t)(p >> 64);

    // Fast Secp256k1 Modular Reduction mod p = 2^256 - 0x1000003D1
    const uint64_t SECP_K = 0x1000003D1ULL;
    u128 prod0 = (u128)t4 * SECP_K + t0;
    uint64_t r0 = (uint64_t)prod0;
    u128 carry = prod0 >> 64;

    u128 prod1 = (u128)t5 * SECP_K + t1 + carry;
    uint64_t r1 = (uint64_t)prod1;
    carry = prod1 >> 64;

    u128 prod2 = (u128)t6 * SECP_K + t2 + carry;
    uint64_t r2 = (uint64_t)prod2;
    carry = prod2 >> 64;

    u128 prod3 = (u128)t7 * SECP_K + t3 + carry;
    uint64_t r3 = (uint64_t)prod3;
    carry = prod3 >> 64;

#if (defined(__x86_64__) || defined(_M_X64)) && !defined(__CUDA_ARCH__)
    u128 k_prod = (u128)carry * SECP_K;
    uint64_t k_lo = (uint64_t)k_prod;
    uint64_t k_hi = (uint64_t)(k_prod >> 64);
    unsigned char carry_flag = 0;
    carry_flag = _addcarry_u64(carry_flag, r0, k_lo, (unsigned long long*)&r0);
    carry_flag = _addcarry_u64(carry_flag, r1, k_hi, (unsigned long long*)&r1);
    carry_flag = _addcarry_u64(carry_flag, r2, 0,    (unsigned long long*)&r2);
    carry_flag = _addcarry_u64(carry_flag, r3, 0,    (unsigned long long*)&r3);
    if (__builtin_expect(carry_flag != 0, 0)) {
        carry_flag = _addcarry_u64(0, r0, SECP_K, (unsigned long long*)&r0);
        carry_flag = _addcarry_u64(carry_flag, r1, 0, (unsigned long long*)&r1);
        carry_flag = _addcarry_u64(carry_flag, r2, 0, (unsigned long long*)&r2);
        _addcarry_u64(carry_flag, r3, 0, (unsigned long long*)&r3);
    }
#else
    u128 c2_red = (u128)r0 + (u128)carry * SECP_K;
    r0 = (uint64_t)c2_red; c2_red >>= 64;
    c2_red += r1; r1 = (uint64_t)c2_red; c2_red >>= 64;
    c2_red += r2; r2 = (uint64_t)c2_red; c2_red >>= 64;
    c2_red += r3; r3 = (uint64_t)c2_red; c2_red >>= 64;
    uint64_t extra = (uint64_t)c2_red;
    if (__builtin_expect(extra != 0, 0)) {
        u128 c3 = (u128)r0 + (u128)extra * SECP_K;
        r0 = (uint64_t)c3; c3 >>= 64;
        c3 += r1; r1 = (uint64_t)c3; c3 >>= 64;
        c3 += r2; r2 = (uint64_t)c3; c3 >>= 64;
        r3 += (uint64_t)c3;
    }
#endif

    if (__builtin_expect(r3 == 0xFFFFFFFFFFFFFFFFULL &&
        r2 == 0xFFFFFFFFFFFFFFFFULL &&
        r1 == 0xFFFFFFFFFFFFFFFFULL &&
        r0 >= 0xFFFFFFFEFFFFFC2FULL, 0)) {
        r0 -= 0xFFFFFFFEFFFFFC2FULL;
        r1 = 0; r2 = 0; r3 = 0;
    }
    return Fe{{r0, r1, r2, r3}};
#else
    Fe r = {0};
    return r;
#endif
}
CUDA_HOSTDEV CUDA_INLINE Fe fe_sqr(const Fe& a) {
#if defined(__SIZEOF_INT128__)
    uint64_t a0 = a.d[0], a1 = a.d[1], a2 = a.d[2], a3 = a.d[3];

    // High-performance Comba Column Squaring (Registers Only)
    // Cross products
    u128 c = (u128)a0 * a1;
    uint64_t c1 = (uint64_t)c;
    c >>= 64;

    c += (u128)a0 * a2;
    uint64_t c2 = (uint64_t)c;
    c >>= 64;

    c += (u128)a0 * a3;
    uint64_t c3 = (uint64_t)c;
    uint64_t c4 = (uint64_t)(c >> 64);

    c = (u128)c3 + (u128)a1 * a2;
    c3 = (uint64_t)c;
    c >>= 64;

    c += (u128)c4 + (u128)a1 * a3;
    c4 = (uint64_t)c;
    uint64_t c5 = (uint64_t)(c >> 64);

    c = (u128)c5 + (u128)a2 * a3;
    c5 = (uint64_t)c;
    uint64_t c6 = (uint64_t)(c >> 64);

    // Double cross-products
    uint64_t t7 = c6 >> 63;
    uint64_t t6 = (c6 << 1) | (c5 >> 63);
    uint64_t t5 = (c5 << 1) | (c4 >> 63);
    uint64_t t4 = (c4 << 1) | (c3 >> 63);
    uint64_t t3 = (c3 << 1) | (c2 >> 63);
    uint64_t t2 = (c2 << 1) | (c1 >> 63);
    uint64_t t1 = (c1 << 1);

    // Add squares
    c = (u128)a0 * a0;
    uint64_t t0 = (uint64_t)c;
    c >>= 64;

    c += (u128)t1;
    t1 = (uint64_t)c;
    c >>= 64;

    c += (u128)t2 + (u128)a1 * a1;
    t2 = (uint64_t)c;
    c >>= 64;

    c += (u128)t3;
    t3 = (uint64_t)c;
    c >>= 64;

    c += (u128)t4 + (u128)a2 * a2;
    t4 = (uint64_t)c;
    c >>= 64;

    c += (u128)t5;
    t5 = (uint64_t)c;
    c >>= 64;

    c += (u128)t6 + (u128)a3 * a3;
    t6 = (uint64_t)c;
    t7 += (uint64_t)(c >> 64);

    // Fast reduction mod p
    const uint64_t SECP_K = 0x1000003D1ULL;
    u128 prod0 = (u128)t4 * SECP_K + t0;
    uint64_t r0 = (uint64_t)prod0;
    u128 carry = prod0 >> 64;

    u128 prod1 = (u128)t5 * SECP_K + t1 + carry;
    uint64_t r1 = (uint64_t)prod1;
    carry = prod1 >> 64;

    u128 prod2 = (u128)t6 * SECP_K + t2 + carry;
    uint64_t r2 = (uint64_t)prod2;
    carry = prod2 >> 64;

    u128 prod3 = (u128)t7 * SECP_K + t3 + carry;
    uint64_t r3 = (uint64_t)prod3;
    carry = prod3 >> 64;

#if (defined(__x86_64__) || defined(_M_X64)) && !defined(__CUDA_ARCH__)
    u128 k_prod = (u128)carry * SECP_K;
    uint64_t k_lo = (uint64_t)k_prod;
    uint64_t k_hi = (uint64_t)(k_prod >> 64);
    unsigned char carry_flag = 0;
    carry_flag = _addcarry_u64(carry_flag, r0, k_lo, (unsigned long long*)&r0);
    carry_flag = _addcarry_u64(carry_flag, r1, k_hi, (unsigned long long*)&r1);
    carry_flag = _addcarry_u64(carry_flag, r2, 0,    (unsigned long long*)&r2);
    carry_flag = _addcarry_u64(carry_flag, r3, 0,    (unsigned long long*)&r3);
    if (__builtin_expect(carry_flag != 0, 0)) {
        carry_flag = _addcarry_u64(0, r0, SECP_K, (unsigned long long*)&r0);
        carry_flag = _addcarry_u64(carry_flag, r1, 0, (unsigned long long*)&r1);
        carry_flag = _addcarry_u64(carry_flag, r2, 0, (unsigned long long*)&r2);
        _addcarry_u64(carry_flag, r3, 0, (unsigned long long*)&r3);
    }
#else
    u128 c2_red = (u128)r0 + (u128)carry * SECP_K;
    r0 = (uint64_t)c2_red; c2_red >>= 64;
    c2_red += r1; r1 = (uint64_t)c2_red; c2_red >>= 64;
    c2_red += r2; r2 = (uint64_t)c2_red; c2_red >>= 64;
    c2_red += r3; r3 = (uint64_t)c2_red; c2_red >>= 64;
    uint64_t extra = (uint64_t)c2_red;
    if (__builtin_expect(extra != 0, 0)) {
        u128 c3 = (u128)r0 + (u128)extra * SECP_K;
        r0 = (uint64_t)c3; c3 >>= 64;
        c3 += r1; r1 = (uint64_t)c3; c3 >>= 64;
        c3 += r2; r2 = (uint64_t)c3; c3 >>= 64;
        r3 += (uint64_t)c3;
    }
#endif

    if (__builtin_expect(r3 == 0xFFFFFFFFFFFFFFFFULL &&
        r2 == 0xFFFFFFFFFFFFFFFFULL &&
        r1 == 0xFFFFFFFFFFFFFFFFULL &&
        r0 >= 0xFFFFFFFEFFFFFC2FULL, 0)) {
        r0 -= 0xFFFFFFFEFFFFFC2FULL;
        r1 = 0; r2 = 0; r3 = 0;
    }

    return Fe{{r0, r1, r2, r3}};
#else
    Fe r = {0};
    return r;
#endif
}
// ============================================================================
// MODULAR INVERSION VIA FERMAT'S LITTLE THEOREM: a^(p - 2) mod p
// ============================================================================
CUDA_HOSTDEV CUDA_INLINE Fe fe_sqr_n(Fe a, int n) {
    for (int i = 0; i < n; ++i) {
        a = fe_sqr(a);
    }
    return a;
}

CUDA_HOSTDEV CUDA_INLINE Fe fe_inv(const Fe& a) {
    Fe x2 = fe_mul(fe_sqr(a), a);
    Fe x3 = fe_mul(fe_sqr(x2), a);
    Fe x6 = fe_mul(fe_sqr_n(x3, 3), x3);
    Fe x9 = fe_mul(fe_sqr_n(x6, 3), x3);
    Fe x11 = fe_mul(fe_sqr_n(x9, 2), x2);
    Fe x22 = fe_mul(fe_sqr_n(x11, 11), x11);
    Fe x44 = fe_mul(fe_sqr_n(x22, 22), x22);
    Fe x88 = fe_mul(fe_sqr_n(x44, 44), x44);
    Fe x176 = fe_mul(fe_sqr_n(x88, 88), x88);
    Fe x220 = fe_mul(fe_sqr_n(x176, 44), x44);
    Fe x223 = fe_mul(fe_sqr_n(x220, 3), x3);

    // Exact Fermat addition chain for 2^256 - 2^32 - 979:
    Fe t1 = fe_sqr(x223);
    Fe t2 = fe_mul(fe_sqr_n(t1, 22), x22);
    Fe t3 = fe_sqr_n(t2, 10);

    Fe a2 = fe_sqr(a);
    Fe a4 = fe_sqr(a2);
    Fe a8 = fe_sqr(a4);
    Fe a16 = fe_sqr(a8);
    Fe a32 = fe_sqr(a16);
    Fe a45 = fe_mul(fe_mul(fe_mul(a32, a8), a4), a);

    return fe_mul(t3, a45);
}

// ============================================================================
// SECP256K1 ELLIPTIC CURVE POINT ARITHMETIC
// ============================================================================
struct AffinePoint {
    Fe x;
    Fe y;
};

struct JacobianPoint {
    Fe x;
    Fe y;
    Fe z;
    bool infinity;
};

CUDA_HOSTDEV CUDA_INLINE JacobianPoint jacobian_double(const JacobianPoint& p) {
    if (p.infinity) return p;
    Fe y2 = fe_sqr(p.y);
    Fe s = fe_mul(fe_add(p.x, p.x), fe_add(y2, y2));
    Fe m = fe_mul(fe_add(fe_sqr(p.x), fe_add(fe_sqr(p.x), fe_sqr(p.x))), Fe{{1, 0, 0, 0}});
    Fe x3 = fe_sub(fe_sqr(m), fe_add(s, s));
    Fe y3 = fe_sub(fe_mul(m, fe_sub(s, x3)), fe_mul(fe_sqr(y2), Fe{{8, 0, 0, 0}}));
    Fe z3 = fe_mul(fe_add(p.y, p.y), p.z);
    return JacobianPoint{x3, y3, z3, false};
}

CUDA_HOSTDEV CUDA_INLINE JacobianPoint jacobian_add_affine(const JacobianPoint& p, const AffinePoint& q) {
    if (p.infinity) return JacobianPoint{q.x, q.y, Fe{{1, 0, 0, 0}}, false};
    Fe z1z1 = fe_sqr(p.z);
    Fe u2 = fe_mul(q.x, z1z1);
    Fe s2 = fe_mul(q.y, fe_mul(p.z, z1z1));
    if (p.x == u2) {
        if (p.y == s2) return jacobian_double(p);
        return JacobianPoint{Fe{{0, 0, 0, 0}}, Fe{{0, 0, 0, 0}}, Fe{{0, 0, 0, 0}}, true};
    }
    Fe h = fe_sub(u2, p.x);
    Fe i = fe_sqr(fe_add(h, h));
    Fe j = fe_mul(h, i);
    Fe r = fe_sub(s2, p.y);
    Fe r2 = fe_add(r, r);
    Fe v = fe_mul(p.x, i);
    Fe x3 = fe_sub(fe_sub(fe_sqr(r2), j), fe_add(v, v));
    Fe y3 = fe_sub(fe_mul(r2, fe_sub(v, x3)), fe_mul(fe_add(p.y, p.y), j));
    Fe z3 = fe_mul(fe_add(p.z, h), fe_add(p.z, h));
    z3 = fe_sub(fe_sub(z3, z1z1), fe_sqr(h));
    return JacobianPoint{x3, y3, z3, false};
}

CUDA_HOSTDEV CUDA_INLINE AffinePoint jacobian_to_affine(const JacobianPoint& p) {
    if (p.infinity) return AffinePoint{Fe{{0, 0, 0, 0}}, Fe{{0, 0, 0, 0}}};
    Fe z_inv = fe_inv(p.z);
    Fe z_inv2 = fe_sqr(z_inv);
    Fe z_inv3 = fe_mul(z_inv2, z_inv);
    return AffinePoint{fe_mul(p.x, z_inv2), fe_mul(p.y, z_inv3)};
}

static const AffinePoint G_POINT = {
    Fe{{0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL}},
    Fe{{0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL}}
};

inline AffinePoint scalar_mul_G(const uint64_t limbs[4]) {
    JacobianPoint res{Fe{{0, 0, 0, 0}}, Fe{{0, 0, 0, 0}}, Fe{{0, 0, 0, 0}}, true};
    for (int limb = 3; limb >= 0; --limb) {
        uint64_t w = limbs[limb];
        for (int b = 63; b >= 0; --b) {
            if (!res.infinity) res = jacobian_double(res);
            if ((w >> b) & 1) res = jacobian_add_affine(res, G_POINT);
        }
    }
    return jacobian_to_affine(res);
}

CUDA_CONSTANT AffinePoint dev_G_table[16];
CUDA_CONSTANT AffinePoint dev_batch_G[32];

CUDA_DEV AffinePoint scalar_mul_G_windowed(uint64_t s0, uint64_t s1, uint64_t s2, uint64_t s3) {
    uint64_t limbs[4] = { s0, s1, s2, s3 };
    JacobianPoint res{Fe{{0, 0, 0, 0}}, Fe{{0, 0, 0, 0}}, Fe{{0, 0, 0, 0}}, true};
    for (int limb = 3; limb >= 0; --limb) {
        uint64_t w = limbs[limb];
        for (int b = 60; b >= 0; b -= 4) {
            if (!res.infinity) {
                res = jacobian_double(res);
                res = jacobian_double(res);
                res = jacobian_double(res);
                res = jacobian_double(res);
            }
            uint32_t nibble = (w >> b) & 0x0F;
            if (nibble > 0) {
                res = jacobian_add_affine(res, dev_G_table[nibble]);
            }
        }
    }
    return jacobian_to_affine(res);
}

// ============================================================================
// HASHING: SHA-256 + RIPEMD-160 FOR GPU & CPU
// ============================================================================
CUDA_HOSTDEV CUDA_INLINE uint32_t rotr32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

CUDA_HOSTDEV CUDA_INLINE uint32_t bswap32(uint32_t x) {
#if defined(__CUDA_ARCH__)
    return __byte_perm(x, 0, 0x0123);
#else
    return __builtin_bswap32(x);
#endif
}


#if defined(__CUDA_ARCH__)
// Hardware single-cycle 3-input bitwise LUT instructions (Compute Capability >= 5.0)
CUDA_DEV CUDA_INLINE uint32_t lop3_ch(uint32_t e, uint32_t f, uint32_t g) {
    uint32_t ret;
    asm("lop3.b32 %0, %1, %2, %3, 0xca;" : "=r"(ret) : "r"(e), "r"(f), "r"(g));
    return ret;
}

CUDA_DEV CUDA_INLINE uint32_t lop3_maj(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t ret;
    asm("lop3.b32 %0, %1, %2, %3, 0xe8;" : "=r"(ret) : "r"(a), "r"(b), "r"(c));
    return ret;
}
#endif

CUDA_HOSTDEV CUDA_INLINE void fast_sha256_into_ripemd_X(uint8_t prefix, const Fe& x, uint32_t X[8]) {
    uint32_t w[16];
    uint64_t x3 = x.d[3], x2 = x.d[2], x1 = x.d[1], x0 = x.d[0];
    w[0] = ((uint32_t)prefix << 24) | (uint32_t)(x3 >> 40);
    w[1] = (uint32_t)(x3 >> 8);
    w[2] = ((uint32_t)x3 << 24) | (uint32_t)(x2 >> 40);
    w[3] = (uint32_t)(x2 >> 8);
    w[4] = ((uint32_t)x2 << 24) | (uint32_t)(x1 >> 40);
    w[5] = (uint32_t)(x1 >> 8);
    w[6] = ((uint32_t)x1 << 24) | (uint32_t)(x0 >> 40);
    w[7] = (uint32_t)(x0 >> 8);
    w[8] = ((uint32_t)x0 << 24) | 0x00800000U;
    w[9] = 0;
    w[10] = 0;
    w[11] = 0;
    w[12] = 0;
    w[13] = 0;
    w[14] = 0;
    w[15] = 0x00000108U;

    static const uint32_t K256[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
    };

    uint32_t a = 0x6a09e667, b = 0xbb67ae85, c = 0x3c6ef372, d = 0xa54ff53a;
    uint32_t e = 0x510e527f, f = 0x9b05688c, g = 0x1f83d9ab, h = 0x5be0cd19;

    #pragma unroll 16
    for (int i = 0; i < 16; ++i) {
        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
#if defined(__CUDA_ARCH__)
        uint32_t ch = lop3_ch(e, f, g);
        uint32_t maj = lop3_maj(a, b, c);
#else
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
#endif
        uint32_t temp1 = h + S1 + ch + K256[i] + w[i];
        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t temp2 = S0 + maj;

        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }

    #pragma unroll 48
    for (int i = 16; i < 64; ++i) {
        uint32_t w15 = w[(i - 15) & 15];
        uint32_t s0 = rotr32(w15, 7) ^ rotr32(w15, 18) ^ (w15 >> 3);
        uint32_t w2 = w[(i - 2) & 15];
        uint32_t s1 = rotr32(w2, 17) ^ rotr32(w2, 19) ^ (w2 >> 10);
        uint32_t wi = w[(i - 16) & 15] + s0 + w[(i - 7) & 15] + s1;
        w[i & 15] = wi;

        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
#if defined(__CUDA_ARCH__)
        uint32_t ch = lop3_ch(e, f, g);
        uint32_t maj = lop3_maj(a, b, c);
#else
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
#endif
        uint32_t temp1 = h + S1 + ch + K256[i] + wi;
        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t temp2 = S0 + maj;

        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }

    X[0] = bswap32(0x6a09e667 + a);
    X[1] = bswap32(0xbb67ae85 + b);
    X[2] = bswap32(0x3c6ef372 + c);
    X[3] = bswap32(0xa54ff53a + d);
    X[4] = bswap32(0x510e527f + e);
    X[5] = bswap32(0x9b05688c + f);
    X[6] = bswap32(0x1f83d9ab + g);
    X[7] = bswap32(0x5be0cd19 + h);
}

CUDA_HOSTDEV CUDA_INLINE uint32_t btc_rol(uint32_t x, int i) { return (x << i) | (x >> (32 - i)); }
CUDA_HOSTDEV CUDA_INLINE uint32_t btc_f1(uint32_t x, uint32_t y, uint32_t z) { return x ^ y ^ z; }
CUDA_HOSTDEV CUDA_INLINE uint32_t btc_f2(uint32_t x, uint32_t y, uint32_t z) { return (x & y) | (~x & z); }
CUDA_HOSTDEV CUDA_INLINE uint32_t btc_f3(uint32_t x, uint32_t y, uint32_t z) { return (x | ~y) ^ z; }
CUDA_HOSTDEV CUDA_INLINE uint32_t btc_f4(uint32_t x, uint32_t y, uint32_t z) { return (x & z) | (y & ~z); }
CUDA_HOSTDEV CUDA_INLINE uint32_t btc_f5(uint32_t x, uint32_t y, uint32_t z) { return x ^ (y | ~z); }

CUDA_HOSTDEV CUDA_INLINE void btc_round(uint32_t& a, uint32_t b, uint32_t& c, uint32_t d, uint32_t e, uint32_t f, uint32_t x, uint32_t k, int r) {
    a = btc_rol(a + f + x + k, r) + e;
    c = btc_rol(c, 10);
}

CUDA_HOSTDEV CUDA_INLINE bool fast_ripemd160_32_check(const uint32_t X[8], const uint32_t target_w[5]) {
    uint32_t a1 = 0x67452301U, b1 = 0xEFCDAB89U, c1 = 0x98BADCFEU, d1 = 0x10325476U, e1 = 0xC3D2E1F0U;
    uint32_t a2 = a1, b2 = b1, c2 = c1, d2 = d1, e2 = e1;

    btc_round(a1, b1, c1, d1, e1, btc_f1(b1, c1, d1), X[0], 0U, 11);
    btc_round(a2, b2, c2, d2, e2, btc_f5(b2, c2, d2), X[5], 0x50A28BE6U, 8);
    btc_round(e1, a1, b1, c1, d1, btc_f1(a1, b1, c1), X[1], 0U, 14);
    btc_round(e2, a2, b2, c2, d2, btc_f5(a2, b2, c2), 256U, 0x50A28BE6U, 9);
    btc_round(d1, e1, a1, b1, c1, btc_f1(e1, a1, b1), X[2], 0U, 15);
    btc_round(d2, e2, a2, b2, c2, btc_f5(e2, a2, b2), X[7], 0x50A28BE6U, 9);
    btc_round(c1, d1, e1, a1, b1, btc_f1(d1, e1, a1), X[3], 0U, 12);
    btc_round(c2, d2, e2, a2, b2, btc_f5(d2, e2, a2), X[0], 0x50A28BE6U, 11);
    btc_round(b1, c1, d1, e1, a1, btc_f1(c1, d1, e1), X[4], 0U, 5);
    btc_round(b2, c2, d2, e2, a2, btc_f5(c2, d2, e2), 0U, 0x50A28BE6U, 13);
    btc_round(a1, b1, c1, d1, e1, btc_f1(b1, c1, d1), X[5], 0U, 8);
    btc_round(a2, b2, c2, d2, e2, btc_f5(b2, c2, d2), X[2], 0x50A28BE6U, 15);
    btc_round(e1, a1, b1, c1, d1, btc_f1(a1, b1, c1), X[6], 0U, 7);
    btc_round(e2, a2, b2, c2, d2, btc_f5(a2, b2, c2), 0U, 0x50A28BE6U, 15);
    btc_round(d1, e1, a1, b1, c1, btc_f1(e1, a1, b1), X[7], 0U, 9);
    btc_round(d2, e2, a2, b2, c2, btc_f5(e2, a2, b2), X[4], 0x50A28BE6U, 5);
    btc_round(c1, d1, e1, a1, b1, btc_f1(d1, e1, a1), 0x00000080U, 0U, 11);
    btc_round(c2, d2, e2, a2, b2, btc_f5(d2, e2, a2), 0U, 0x50A28BE6U, 7);
    btc_round(b1, c1, d1, e1, a1, btc_f1(c1, d1, e1), 0U, 0U, 13);
    btc_round(b2, c2, d2, e2, a2, btc_f5(c2, d2, e2), X[6], 0x50A28BE6U, 7);
    btc_round(a1, b1, c1, d1, e1, btc_f1(b1, c1, d1), 0U, 0U, 14);
    btc_round(a2, b2, c2, d2, e2, btc_f5(b2, c2, d2), 0U, 0x50A28BE6U, 8);
    btc_round(e1, a1, b1, c1, d1, btc_f1(a1, b1, c1), 0U, 0U, 15);
    btc_round(e2, a2, b2, c2, d2, btc_f5(a2, b2, c2), 0x00000080U, 0x50A28BE6U, 11);
    btc_round(d1, e1, a1, b1, c1, btc_f1(e1, a1, b1), 0U, 0U, 6);
    btc_round(d2, e2, a2, b2, c2, btc_f5(e2, a2, b2), X[1], 0x50A28BE6U, 14);
    btc_round(c1, d1, e1, a1, b1, btc_f1(d1, e1, a1), 0U, 0U, 7);
    btc_round(c2, d2, e2, a2, b2, btc_f5(d2, e2, a2), 0U, 0x50A28BE6U, 14);
    btc_round(b1, c1, d1, e1, a1, btc_f1(c1, d1, e1), 256U, 0U, 9);
    btc_round(b2, c2, d2, e2, a2, btc_f5(c2, d2, e2), X[3], 0x50A28BE6U, 12);
    btc_round(a1, b1, c1, d1, e1, btc_f1(b1, c1, d1), 0U, 0U, 8);
    btc_round(a2, b2, c2, d2, e2, btc_f5(b2, c2, d2), 0U, 0x50A28BE6U, 6);
    btc_round(e1, a1, b1, c1, d1, btc_f2(a1, b1, c1), X[7], 0x5A827999U, 7);
    btc_round(e2, a2, b2, c2, d2, btc_f4(a2, b2, c2), X[6], 0x5C4DD124U, 9);
    btc_round(d1, e1, a1, b1, c1, btc_f2(e1, a1, b1), X[4], 0x5A827999U, 6);
    btc_round(d2, e2, a2, b2, c2, btc_f4(e2, a2, b2), 0U, 0x5C4DD124U, 13);
    btc_round(c1, d1, e1, a1, b1, btc_f2(d1, e1, a1), 0U, 0x5A827999U, 8);
    btc_round(c2, d2, e2, a2, b2, btc_f4(d2, e2, a2), X[3], 0x5C4DD124U, 15);
    btc_round(b1, c1, d1, e1, a1, btc_f2(c1, d1, e1), X[1], 0x5A827999U, 13);
    btc_round(b2, c2, d2, e2, a2, btc_f4(c2, d2, e2), X[7], 0x5C4DD124U, 7);
    btc_round(a1, b1, c1, d1, e1, btc_f2(b1, c1, d1), 0U, 0x5A827999U, 11);
    btc_round(a2, b2, c2, d2, e2, btc_f4(b2, c2, d2), X[0], 0x5C4DD124U, 12);
    btc_round(e1, a1, b1, c1, d1, btc_f2(a1, b1, c1), X[6], 0x5A827999U, 9);
    btc_round(e2, a2, b2, c2, d2, btc_f4(a2, b2, c2), 0U, 0x5C4DD124U, 8);
    btc_round(d1, e1, a1, b1, c1, btc_f2(e1, a1, b1), 0U, 0x5A827999U, 7);
    btc_round(d2, e2, a2, b2, c2, btc_f4(e2, a2, b2), X[5], 0x5C4DD124U, 9);
    btc_round(c1, d1, e1, a1, b1, btc_f2(d1, e1, a1), X[3], 0x5A827999U, 15);
    btc_round(c2, d2, e2, a2, b2, btc_f4(d2, e2, a2), 0U, 0x5C4DD124U, 11);
    btc_round(b1, c1, d1, e1, a1, btc_f2(c1, d1, e1), 0U, 0x5A827999U, 7);
    btc_round(b2, c2, d2, e2, a2, btc_f4(c2, d2, e2), 256U, 0x5C4DD124U, 7);
    btc_round(a1, b1, c1, d1, e1, btc_f2(b1, c1, d1), X[0], 0x5A827999U, 12);
    btc_round(a2, b2, c2, d2, e2, btc_f4(b2, c2, d2), 0U, 0x5C4DD124U, 7);
    btc_round(e1, a1, b1, c1, d1, btc_f2(a1, b1, c1), 0U, 0x5A827999U, 15);
    btc_round(e2, a2, b2, c2, d2, btc_f4(a2, b2, c2), 0x00000080U, 0x5C4DD124U, 12);
    btc_round(d1, e1, a1, b1, c1, btc_f2(e1, a1, b1), X[5], 0x5A827999U, 9);
    btc_round(d2, e2, a2, b2, c2, btc_f4(e2, a2, b2), 0U, 0x5C4DD124U, 7);
    btc_round(c1, d1, e1, a1, b1, btc_f2(d1, e1, a1), X[2], 0x5A827999U, 11);
    btc_round(c2, d2, e2, a2, b2, btc_f4(d2, e2, a2), X[4], 0x5C4DD124U, 6);
    btc_round(b1, c1, d1, e1, a1, btc_f2(c1, d1, e1), 256U, 0x5A827999U, 7);
    btc_round(b2, c2, d2, e2, a2, btc_f4(c2, d2, e2), 0U, 0x5C4DD124U, 15);
    btc_round(a1, b1, c1, d1, e1, btc_f2(b1, c1, d1), 0U, 0x5A827999U, 13);
    btc_round(a2, b2, c2, d2, e2, btc_f4(b2, c2, d2), X[1], 0x5C4DD124U, 13);
    btc_round(e1, a1, b1, c1, d1, btc_f2(a1, b1, c1), 0x00000080U, 0x5A827999U, 12);
    btc_round(e2, a2, b2, c2, d2, btc_f4(a2, b2, c2), X[2], 0x5C4DD124U, 11);
    btc_round(d1, e1, a1, b1, c1, btc_f3(e1, a1, b1), X[3], 0x6ED9EBA1U, 11);
    btc_round(d2, e2, a2, b2, c2, btc_f3(e2, a2, b2), 0U, 0x6D703EF3U, 9);
    btc_round(c1, d1, e1, a1, b1, btc_f3(d1, e1, a1), 0U, 0x6ED9EBA1U, 13);
    btc_round(c2, d2, e2, a2, b2, btc_f3(d2, e2, a2), X[5], 0x6D703EF3U, 7);
    btc_round(b1, c1, d1, e1, a1, btc_f3(c1, d1, e1), 256U, 0x6ED9EBA1U, 6);
    btc_round(b2, c2, d2, e2, a2, btc_f3(c2, d2, e2), X[1], 0x6D703EF3U, 15);
    btc_round(a1, b1, c1, d1, e1, btc_f3(b1, c1, d1), X[4], 0x6ED9EBA1U, 7);
    btc_round(a2, b2, c2, d2, e2, btc_f3(b2, c2, d2), X[3], 0x6D703EF3U, 11);
    btc_round(e1, a1, b1, c1, d1, btc_f3(a1, b1, c1), 0U, 0x6ED9EBA1U, 14);
    btc_round(e2, a2, b2, c2, d2, btc_f3(a2, b2, c2), X[7], 0x6D703EF3U, 8);
    btc_round(d1, e1, a1, b1, c1, btc_f3(e1, a1, b1), 0U, 0x6ED9EBA1U, 9);
    btc_round(d2, e2, a2, b2, c2, btc_f3(e2, a2, b2), 256U, 0x6D703EF3U, 6);
    btc_round(c1, d1, e1, a1, b1, btc_f3(d1, e1, a1), 0x00000080U, 0x6ED9EBA1U, 13);
    btc_round(c2, d2, e2, a2, b2, btc_f3(d2, e2, a2), X[6], 0x6D703EF3U, 6);
    btc_round(b1, c1, d1, e1, a1, btc_f3(c1, d1, e1), X[1], 0x6ED9EBA1U, 15);
    btc_round(b2, c2, d2, e2, a2, btc_f3(c2, d2, e2), 0U, 0x6D703EF3U, 14);
    btc_round(a1, b1, c1, d1, e1, btc_f3(b1, c1, d1), X[2], 0x6ED9EBA1U, 14);
    btc_round(a2, b2, c2, d2, e2, btc_f3(b2, c2, d2), 0U, 0x6D703EF3U, 12);
    btc_round(e1, a1, b1, c1, d1, btc_f3(a1, b1, c1), X[7], 0x6ED9EBA1U, 8);
    btc_round(e2, a2, b2, c2, d2, btc_f3(a2, b2, c2), 0x00000080U, 0x6D703EF3U, 13);
    btc_round(d1, e1, a1, b1, c1, btc_f3(e1, a1, b1), X[0], 0x6ED9EBA1U, 13);
    btc_round(d2, e2, a2, b2, c2, btc_f3(e2, a2, b2), 0U, 0x6D703EF3U, 5);
    btc_round(c1, d1, e1, a1, b1, btc_f3(d1, e1, a1), X[6], 0x6ED9EBA1U, 6);
    btc_round(c2, d2, e2, a2, b2, btc_f3(d2, e2, a2), X[2], 0x6D703EF3U, 14);
    btc_round(b1, c1, d1, e1, a1, btc_f3(c1, d1, e1), 0U, 0x6ED9EBA1U, 5);
    btc_round(b2, c2, d2, e2, a2, btc_f3(c2, d2, e2), 0U, 0x6D703EF3U, 13);
    btc_round(a1, b1, c1, d1, e1, btc_f3(b1, c1, d1), 0U, 0x6ED9EBA1U, 12);
    btc_round(a2, b2, c2, d2, e2, btc_f3(b2, c2, d2), X[0], 0x6D703EF3U, 13);
    btc_round(e1, a1, b1, c1, d1, btc_f3(a1, b1, c1), X[5], 0x6ED9EBA1U, 7);
    btc_round(e2, a2, b2, c2, d2, btc_f3(a2, b2, c2), X[4], 0x6D703EF3U, 7);
    btc_round(d1, e1, a1, b1, c1, btc_f3(e1, a1, b1), 0U, 0x6ED9EBA1U, 5);
    btc_round(d2, e2, a2, b2, c2, btc_f3(e2, a2, b2), 0U, 0x6D703EF3U, 5);
    btc_round(c1, d1, e1, a1, b1, btc_f4(d1, e1, a1), X[1], 0x8F1BBCDCU, 11);
    btc_round(c2, d2, e2, a2, b2, btc_f2(d2, e2, a2), 0x00000080U, 0x7A6D76E9U, 15);
    btc_round(b1, c1, d1, e1, a1, btc_f4(c1, d1, e1), 0U, 0x8F1BBCDCU, 12);
    btc_round(b2, c2, d2, e2, a2, btc_f2(c2, d2, e2), X[6], 0x7A6D76E9U, 5);
    btc_round(a1, b1, c1, d1, e1, btc_f4(b1, c1, d1), 0U, 0x8F1BBCDCU, 14);
    btc_round(a2, b2, c2, d2, e2, btc_f2(b2, c2, d2), X[4], 0x7A6D76E9U, 8);
    btc_round(e1, a1, b1, c1, d1, btc_f4(a1, b1, c1), 0U, 0x8F1BBCDCU, 15);
    btc_round(e2, a2, b2, c2, d2, btc_f2(a2, b2, c2), X[1], 0x7A6D76E9U, 11);
    btc_round(d1, e1, a1, b1, c1, btc_f4(e1, a1, b1), X[0], 0x8F1BBCDCU, 14);
    btc_round(d2, e2, a2, b2, c2, btc_f2(e2, a2, b2), X[3], 0x7A6D76E9U, 14);
    btc_round(c1, d1, e1, a1, b1, btc_f4(d1, e1, a1), 0x00000080U, 0x8F1BBCDCU, 15);
    btc_round(c2, d2, e2, a2, b2, btc_f2(d2, e2, a2), 0U, 0x7A6D76E9U, 14);
    btc_round(b1, c1, d1, e1, a1, btc_f4(c1, d1, e1), 0U, 0x8F1BBCDCU, 9);
    btc_round(b2, c2, d2, e2, a2, btc_f2(c2, d2, e2), 0U, 0x7A6D76E9U, 6);
    btc_round(a1, b1, c1, d1, e1, btc_f4(b1, c1, d1), X[4], 0x8F1BBCDCU, 8);
    btc_round(a2, b2, c2, d2, e2, btc_f2(b2, c2, d2), X[0], 0x7A6D76E9U, 14);
    btc_round(e1, a1, b1, c1, d1, btc_f4(a1, b1, c1), 0U, 0x8F1BBCDCU, 9);
    btc_round(e2, a2, b2, c2, d2, btc_f2(a2, b2, c2), X[5], 0x7A6D76E9U, 6);
    btc_round(d1, e1, a1, b1, c1, btc_f4(e1, a1, b1), X[3], 0x8F1BBCDCU, 14);
    btc_round(d2, e2, a2, b2, c2, btc_f2(e2, a2, b2), 0U, 0x7A6D76E9U, 9);
    btc_round(c1, d1, e1, a1, b1, btc_f4(d1, e1, a1), X[7], 0x8F1BBCDCU, 5);
    btc_round(c2, d2, e2, a2, b2, btc_f2(d2, e2, a2), X[2], 0x7A6D76E9U, 12);
    btc_round(b1, c1, d1, e1, a1, btc_f4(c1, d1, e1), 0U, 0x8F1BBCDCU, 6);
    btc_round(b2, c2, d2, e2, a2, btc_f2(c2, d2, e2), 0U, 0x7A6D76E9U, 9);
    btc_round(a1, b1, c1, d1, e1, btc_f4(b1, c1, d1), 256U, 0x8F1BBCDCU, 8);
    btc_round(a2, b2, c2, d2, e2, btc_f2(b2, c2, d2), 0U, 0x7A6D76E9U, 12);
    btc_round(e1, a1, b1, c1, d1, btc_f4(a1, b1, c1), X[5], 0x8F1BBCDCU, 6);
    btc_round(e2, a2, b2, c2, d2, btc_f2(a2, b2, c2), X[7], 0x7A6D76E9U, 5);
    btc_round(d1, e1, a1, b1, c1, btc_f4(e1, a1, b1), X[6], 0x8F1BBCDCU, 5);
    btc_round(d2, e2, a2, b2, c2, btc_f2(e2, a2, b2), 0U, 0x7A6D76E9U, 15);
    btc_round(c1, d1, e1, a1, b1, btc_f4(d1, e1, a1), X[2], 0x8F1BBCDCU, 12);
    btc_round(c2, d2, e2, a2, b2, btc_f2(d2, e2, a2), 256U, 0x7A6D76E9U, 8);
    btc_round(b1, c1, d1, e1, a1, btc_f5(c1, d1, e1), X[4], 0xA953FD4EU, 9);
    btc_round(b2, c2, d2, e2, a2, btc_f1(c2, d2, e2), 0U, 0U, 8);
    btc_round(a1, b1, c1, d1, e1, btc_f5(b1, c1, d1), X[0], 0xA953FD4EU, 15);
    btc_round(a2, b2, c2, d2, e2, btc_f1(b2, c2, d2), 0U, 0U, 5);
    btc_round(e1, a1, b1, c1, d1, btc_f5(a1, b1, c1), X[5], 0xA953FD4EU, 5);
    btc_round(e2, a2, b2, c2, d2, btc_f1(a2, b2, c2), 0U, 0U, 12);
    btc_round(d1, e1, a1, b1, c1, btc_f5(e1, a1, b1), 0U, 0xA953FD4EU, 11);
    btc_round(d2, e2, a2, b2, c2, btc_f1(e2, a2, b2), X[4], 0U, 9);
    btc_round(c1, d1, e1, a1, b1, btc_f5(d1, e1, a1), X[7], 0xA953FD4EU, 6);
    btc_round(c2, d2, e2, a2, b2, btc_f1(d2, e2, a2), X[1], 0U, 12);
    btc_round(b1, c1, d1, e1, a1, btc_f5(c1, d1, e1), 0U, 0xA953FD4EU, 8);
    btc_round(b2, c2, d2, e2, a2, btc_f1(c2, d2, e2), X[5], 0U, 5);
    btc_round(a1, b1, c1, d1, e1, btc_f5(b1, c1, d1), X[2], 0xA953FD4EU, 13);
    btc_round(a2, b2, c2, d2, e2, btc_f1(b2, c2, d2), 0x00000080U, 0U, 14);
    btc_round(e1, a1, b1, c1, d1, btc_f5(a1, b1, c1), 0U, 0xA953FD4EU, 12);
    btc_round(e2, a2, b2, c2, d2, btc_f1(a2, b2, c2), X[7], 0U, 6);
    btc_round(d1, e1, a1, b1, c1, btc_f5(e1, a1, b1), 256U, 0xA953FD4EU, 5);
    btc_round(d2, e2, a2, b2, c2, btc_f1(e2, a2, b2), X[6], 0U, 8);
    btc_round(c1, d1, e1, a1, b1, btc_f5(d1, e1, a1), X[1], 0xA953FD4EU, 12);
    btc_round(c2, d2, e2, a2, b2, btc_f1(d2, e2, a2), X[2], 0U, 13);
    btc_round(b1, c1, d1, e1, a1, btc_f5(c1, d1, e1), X[3], 0xA953FD4EU, 13);
    btc_round(b2, c2, d2, e2, a2, btc_f1(c2, d2, e2), 0U, 0U, 6);
    btc_round(a1, b1, c1, d1, e1, btc_f5(b1, c1, d1), 0x00000080U, 0xA953FD4EU, 14);
    btc_round(a2, b2, c2, d2, e2, btc_f1(b2, c2, d2), 256U, 0U, 5);
    btc_round(e1, a1, b1, c1, d1, btc_f5(a1, b1, c1), 0U, 0xA953FD4EU, 11);
    btc_round(e2, a2, b2, c2, d2, btc_f1(a2, b2, c2), X[0], 0U, 15);
    btc_round(d1, e1, a1, b1, c1, btc_f5(e1, a1, b1), X[6], 0xA953FD4EU, 8);
    btc_round(d2, e2, a2, b2, c2, btc_f1(e2, a2, b2), X[3], 0U, 13);
    btc_round(c1, d1, e1, a1, b1, btc_f5(d1, e1, a1), 0U, 0xA953FD4EU, 5);
    btc_round(c2, d2, e2, a2, b2, btc_f1(d2, e2, a2), 0U, 0U, 11);
    btc_round(b1, c1, d1, e1, a1, btc_f5(c1, d1, e1), 0U, 0xA953FD4EU, 6);
    btc_round(b2, c2, d2, e2, a2, btc_f1(c2, d2, e2), 0U, 0U, 11);

    uint32_t s0 = 0x67452301U, s1 = 0xEFCDAB89U, s2 = 0x98BADCFEU, s3 = 0x10325476U, s4 = 0xC3D2E1F0U;
    if ((s1 + c1 + d2) != target_w[0]) return false;
    if ((s2 + d1 + e2) != target_w[1]) return false;
    if ((s3 + e1 + a2) != target_w[2]) return false;
    if ((s4 + a1 + b2) != target_w[3]) return false;
    return ((s0 + b1 + c2) == target_w[4]);
}

// ============================================================================
// CUDA SCAN KERNEL (LOCKSTEP REGISTER-ONLY PIPELINE)
// ============================================================================
CUDA_GLOBAL void cuda_scan_kernel(
    uint64_t start_k0, uint64_t start_k1, uint64_t start_k2, uint64_t start_k3,
    uint64_t total_chunk_keys,
    uint32_t grid_threads,
    uint32_t batches,
    int* d_found_flag,
    uint64_t* d_found_offset,
    uint32_t tw0, uint32_t tw1, uint32_t tw2, uint32_t tw3, uint32_t tw4) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= grid_threads) return;
    uint64_t thread_start_offset = (uint64_t)tid;
    if (thread_start_offset >= total_chunk_keys) return;

    uint32_t tw[5] = { tw0, tw1, tw2, tw3, tw4 };

    uint64_t cur_k0 = start_k0 + thread_start_offset;
    uint64_t carry = (cur_k0 < start_k0) ? 1 : 0;
    uint64_t cur_k1 = start_k1 + carry;
    carry = (cur_k1 < carry) ? 1 : 0;
    uint64_t cur_k2 = start_k2 + carry;
    carry = (cur_k2 < carry) ? 1 : 0;
    uint64_t cur_k3 = start_k3 + carry;

    AffinePoint cur_P = scalar_mul_G_windowed(cur_k0, cur_k1, cur_k2, cur_k3);

    uint8_t pfx = (cur_P.y.d[0] & 1) ? 0x03 : 0x02;
    uint32_t Xinit[8];
    fast_sha256_into_ripemd_X(pfx, cur_P.x, Xinit);
    if (fast_ripemd160_32_check(Xinit, tw)) {
        if (atomicExch(d_found_flag, 1) == 0) {
            *d_found_offset = thread_start_offset;
        }
        return;
    }

    uint64_t step_keys = (uint64_t)grid_threads;

    for (uint32_t b = 0; b < batches; ++b) {
        if (*d_found_flag) return;
        uint64_t batch_base_offset = thread_start_offset + (uint64_t)b * 32ULL * step_keys;
        if (batch_base_offset >= total_chunk_keys) return;

        Fe dx[32];
        Fe prod[32];

        for (int i = 0; i < 32; ++i) {
            dx[i] = fe_sub(dev_batch_G[i].x, cur_P.x);
        }

        prod[0] = dx[0];
        for (int i = 1; i < 32; ++i) {
            prod[i] = fe_mul(prod[i - 1], dx[i]);
        }

        Fe inv_all = fe_inv(prod[31]);

        for (int i = 31; i >= 1; --i) {
            Fe inv_dx_i = fe_mul(inv_all, prod[i - 1]);
            inv_all = fe_mul(inv_all, dx[i]);
            prod[i] = inv_dx_i;
        }
        prod[0] = inv_all;

        AffinePoint next_cur_P;
        for (int i = 0; i < 32; ++i) {
            uint64_t key_offset = batch_base_offset + (uint64_t)(i + 1) * step_keys;
            if (key_offset < total_chunk_keys) {
                Fe dy_i = fe_sub(dev_batch_G[i].y, cur_P.y);
                Fe lambda = fe_mul(dy_i, prod[i]);
                Fe lambda_sq = fe_sqr(lambda);
                Fe next_x = fe_sub(fe_sub(lambda_sq, cur_P.x), dev_batch_G[i].x);
                Fe diff_x = fe_sub(cur_P.x, next_x);
                Fe next_y = fe_sub(fe_mul(lambda, diff_x), cur_P.y);

                if (i == 31) {
                    next_cur_P = AffinePoint{next_x, next_y};
                }

                uint8_t prefix = (next_y.d[0] & 1) ? 0x03 : 0x02;
                uint32_t X[8];
                fast_sha256_into_ripemd_X(prefix, next_x, X);
                if (fast_ripemd160_32_check(X, tw)) {
                    if (atomicExch(d_found_flag, 1) == 0) {
                        *d_found_offset = key_offset;
                    }
                    return;
                }
            } else if (i == 31) {
                Fe dy_31 = fe_sub(dev_batch_G[31].y, cur_P.y);
                Fe lambda = fe_mul(dy_31, prod[31]);
                Fe lambda_sq = fe_sqr(lambda);
                Fe next_x = fe_sub(fe_sub(lambda_sq, cur_P.x), dev_batch_G[31].x);
                Fe diff_x = fe_sub(cur_P.x, next_x);
                Fe next_y = fe_sub(fe_mul(lambda, diff_x), cur_P.y);
                next_cur_P = AffinePoint{next_x, next_y};
            }
        }
        cur_P = next_cur_P;
    }
}

// ============================================================================
// ============================================================================
// CUDA GPU HOST DRIVER & VERIFICATION RUNNER
// ============================================================================
static size_t curl_cb(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t total = size * nmemb;
    ((std::string*)userp)->append((char*)contents, total);
    return total;
}

bool http_get(const std::string& url, std::string* response) {
    CURL* curl = curl_easy_init();
    if (!curl) return false;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 15L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    return (res == CURLE_OK);
}

bool http_post(const std::string& url, const std::string& json_data, std::string* response) {
    CURL* curl = curl_easy_init();
    if (!curl) return false;
    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_data.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 15L);
    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return (res == CURLE_OK);
}

std::string json_get_string(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    size_t p = json.find(pattern);
    if (p == std::string::npos) return "";
    p += pattern.length();
    while (p < json.length() && (json[p] == ' ' || json[p] == '\"')) p++;
    size_t end = p;
    while (end < json.length() && json[end] != '\"' && json[end] != ',' && json[end] != '}') end++;
    return json.substr(p, end - p);
}

std::string format_speed(double speed) {
    std::stringstream ss;
    if (speed >= 1000000.0) {
        ss << std::fixed << std::setprecision(2) << (speed / 1000000.0) << " Mkeys/s";
    } else if (speed >= 1000.0) {
        ss << std::fixed << std::setprecision(2) << (speed / 1000.0) << " Kkeys/s";
    } else {
        ss << std::fixed << std::setprecision(0) << speed << " keys/s";
    }
    return ss.str();
}





void parse_hex64_limbs(const std::string& str, uint64_t limbs[4]) {
    limbs[0] = limbs[1] = limbs[2] = limbs[3] = 0;
    if (str.empty()) return;
    if (str.rfind("0x", 0) == 0 || str.rfind("0X", 0) == 0) {
        std::string s = str.substr(2);
        while (s.length() < 64) s = "0" + s;
        for (int i = 0; i < 4; ++i) {
            std::string part = s.substr((3 - i) * 16, 16);
            limbs[i] = std::stoull(part, nullptr, 16);
        }
    } else {
        u256 r = parse_u256(str);
        limbs[0] = (uint64_t)r.low;
        limbs[1] = (uint64_t)(r.low >> 64);
        limbs[2] = (uint64_t)r.high;
        limbs[3] = (uint64_t)(r.high >> 64);
    }
}

std::string limbs_to_hex(const uint64_t limbs[4]) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    ss << std::setw(16) << limbs[3]
       << std::setw(16) << limbs[2]
       << std::setw(16) << limbs[1]
       << std::setw(16) << limbs[0];
    return ss.str();
}

inline void init_cuda_tables(uint32_t grid_threads) {
    AffinePoint h_table[16];
    std::memset(&h_table[0], 0, sizeof(AffinePoint));
    for (int i = 1; i < 16; ++i) {
        uint64_t s[4] = { (uint64_t)i, 0, 0, 0 };
        h_table[i] = scalar_mul_G(s);
    }
    cudaMemcpyToSymbol(dev_G_table, h_table, sizeof(h_table));

    AffinePoint h_batch_G[32];
    for (int i = 0; i < 32; ++i) {
        uint64_t step_mult = (uint64_t)grid_threads * (uint64_t)(i + 1);
        uint64_t s[4] = { step_mult, 0, 0, 0 };
        h_batch_G[i] = scalar_mul_G(s);
    }
    cudaMemcpyToSymbol(dev_batch_G, h_batch_G, sizeof(h_batch_G));
}

int run_gpu_verify(const std::string& api_base, const std::string& current_user = "verify-node", int target_id = 0, int device_id = 0) {
    cudaSetDevice(device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    uint32_t sm_count = (prop.multiProcessorCount > 0) ? (uint32_t)prop.multiProcessorCount : 32;
    uint32_t block_size = 256;
    uint32_t num_blocks = sm_count * 8;
    if (num_blocks < 256) num_blocks = 256;
    if (num_blocks > 4096) num_blocks = 4096;
    uint32_t grid_threads = num_blocks * block_size;
    init_cuda_tables(grid_threads);

    std::vector<int> puzzle_ids;
    if (target_id > 0) {
        puzzle_ids.push_back(target_id);
    } else {
        puzzle_ids = {
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
            31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
            51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
            61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
            75, 80, 85, 90, 95, 100, 105, 110, 115, 120,
            125, 130, 135
        };
    }

    int tested = 0;
    int passed = 0;
    int failed = 0;
    uint64_t total_keys_verified = 0;
    auto t_global_start = std::chrono::high_resolution_clock::now();

    std::cout << "[VERIFY] Connecting to coordinator: " << api_base << "\n";
    std::cout << "[VERIFY] Running CUDA GPU Verification on device " << device_id << "...\n";

    int* d_found_flag = nullptr;
    uint64_t* d_found_offset = nullptr;
    cudaMalloc(&d_found_flag, sizeof(int));
    cudaMalloc(&d_found_offset, sizeof(uint64_t));

    for (int pid : puzzle_ids) {
        tested++;
        std::string req_url = api_base + "?action=range&puzzle=" + std::to_string(pid) + "&test=1&user=" + current_user;
        std::string resp;
        bool got_server = http_get(req_url, &resp);
        std::string str_start = got_server ? json_get_string(resp, "start") : "";
        std::string str_end = got_server ? json_get_string(resp, "end") : "";
        std::string str_target = got_server ? json_get_string(resp, "target_address") : "";

        if (!got_server || str_start.empty() || str_target.empty()) {
            std::cerr << "[SKIP] Target #" << pid << ": Server did not provide test range.\n";
            failed++;
            continue;
        }

        u256 start_k = parse_u256(str_start);
        u256 end_k = str_end.empty() ? (start_k + 65536) : parse_u256(str_end);
        uint64_t total_keys_count = 65536;

        uint8_t target_h160[20];
        if (!b58check_decode_hash160(str_target, target_h160)) {
            failed++;
            continue;
        }

        uint32_t tw[5];
        for (int j = 0; j < 5; ++j) {
            tw[j] = (uint32_t)target_h160[j * 4] |
                    ((uint32_t)target_h160[j * 4 + 1] << 8) |
                    ((uint32_t)target_h160[j * 4 + 2] << 16) |
                    ((uint32_t)target_h160[j * 4 + 3] << 24);
        }

        int h_flag = 0;
        uint64_t h_offset = 0;
        cudaMemcpy(d_found_flag, &h_flag, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_found_offset, &h_offset, sizeof(uint64_t), cudaMemcpyHostToDevice);

        uint64_t chunk_step = (uint64_t)grid_threads * 32ULL;
        uint32_t batches = (uint32_t)((total_keys_count + chunk_step - 1) / chunk_step);
        if (batches == 0) batches = 1;

        uint64_t start_limbs[4] = {
            (uint64_t)start_k.low,
            (uint64_t)(start_k.low >> 64),
            (uint64_t)start_k.high,
            (uint64_t)(start_k.high >> 64)
        };

        auto t_scan_start = std::chrono::high_resolution_clock::now();
        cuda_scan_kernel<<<num_blocks, block_size>>>(
            start_limbs[0], start_limbs[1], start_limbs[2], start_limbs[3],
            total_keys_count, grid_threads, batches,
            d_found_flag, d_found_offset,
            tw[0], tw[1], tw[2], tw[3], tw[4]
        );
        cudaDeviceSynchronize();
        auto t_scan_end = std::chrono::high_resolution_clock::now();

        cudaMemcpy(&h_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_offset, d_found_offset, sizeof(uint64_t), cudaMemcpyDeviceToHost);

        double elapsed_sec = std::chrono::duration<double>(t_scan_end - t_scan_start).count();
        if (elapsed_sec <= 0.0) elapsed_sec = 0.0001;
        total_keys_verified += total_keys_count;
        double target_speed = (double)total_keys_count / elapsed_sec;

        if (h_flag == 1) {
            passed++;
            std::cout << "[PASS] Target #" << pid
                      << " | Speed: " << format_speed(target_speed)
                      << " -> Matched Server Target\n";
        } else {
            failed++;
            std::cerr << "[FAIL] Target #" << pid << " -> Key not found in range\n";
        }
    }

    cudaFree(d_found_flag);
    cudaFree(d_found_offset);

    auto t_global_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_global_end - t_global_start).count();
    if (total_sec <= 0.0) total_sec = 0.0001;
    double avg_verify_speed = (double)total_keys_verified / total_sec;

    std::cout << "\n";
    if (failed == 0 && passed > 0) {
        std::cout << "[OK] " << passed << "/" << tested << " CUDA targets verified successfully.\n";
        std::cout << "[SPEED] Average CUDA Throughput: " << format_speed(avg_verify_speed)
                  << " (" << total_keys_verified << " keys in " << std::fixed << std::setprecision(2) << total_sec << "s)\n";
    }

    if (passed > 0) {
        std::stringstream stat_json;
        stat_json << "{\"action\":\"telemetry\",\"user\":\"" << current_user
                  << "\",\"speed\":" << (uint64_t)avg_verify_speed
                  << ",\"avg_speed\":" << (uint64_t)avg_verify_speed
                  << ",\"status\":\"idle\",\"verified_count\":" << passed << "}";
        std::string stat_resp;
        http_post(api_base, stat_json.str(), &stat_resp);
    }
    return (failed == 0) ? 0 : 1;
}

int main(int argc, char* argv[]) {
    std::string api_base = "http://65.20.91.208/puzzle_server.php";
    std::string user = "cuda-worker-1";
    int device_id = 0;
    int puzzle_id = 0; // 0 = dynamic from server
    bool verify_mode = false;
    bool explicit_puzzle = false;
    bool no_limit = false;
    int max_ranges = 50;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if ((arg == "--server" || arg == "-s" || arg == "--api") && i + 1 < argc) {
            api_base = argv[++i];
        } else if (arg.rfind("--server=", 0) == 0) {
            api_base = arg.substr(9);
        } else if (arg.rfind("-s=", 0) == 0) {
            api_base = arg.substr(3);
        } else if (arg.rfind("--api=", 0) == 0) {
            api_base = arg.substr(6);
        }
        else if ((arg == "--user" || arg == "-u") && i + 1 < argc) {
            user = argv[++i];
        } else if (arg.rfind("--user=", 0) == 0) {
            user = arg.substr(7);
        } else if (arg.rfind("-u=", 0) == 0) {
            user = arg.substr(3);
        }
        else if ((arg == "--gpu") && i + 1 < argc) {
            device_id = std::atoi(argv[++i]);
        } else if (arg.rfind("--gpu=", 0) == 0) {
            device_id = std::atoi(arg.substr(6).c_str());
        }
        else if (arg == "-nl" || arg == "--no-limit") {
            no_limit = true;
        }
        else if (arg == "--verify" || arg == "-v") {
            verify_mode = true;
        }
        else if ((arg == "-p" || arg == "--puzzle") && i + 1 < argc) {
            puzzle_id = std::atoi(argv[++i]);
            explicit_puzzle = true;
        } else if (arg.rfind("-p=", 0) == 0) {
            puzzle_id = std::atoi(arg.substr(3).c_str());
            explicit_puzzle = true;
        } else if (arg.rfind("--puzzle=", 0) == 0) {
            puzzle_id = std::atoi(arg.substr(9).c_str());
            explicit_puzzle = true;
        }
    }

    if (verify_mode) {
        int verify_target = explicit_puzzle ? puzzle_id : 0;
        return run_gpu_verify(api_base, user, verify_target, device_id);
    }

        std::cout << "[WORKER] CUDA Device: " << device_id << " | Privacy: ON\n";
    

    cudaSetDevice(device_id);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    uint32_t sm_count = (prop.multiProcessorCount > 0) ? (uint32_t)prop.multiProcessorCount : 32;
    uint32_t block_size = 256;
    uint32_t num_blocks = sm_count * 8;
    if (num_blocks < 256) num_blocks = 256;
    if (num_blocks > 4096) num_blocks = 4096;
    uint32_t grid_threads = num_blocks * block_size;
    init_cuda_tables(grid_threads);

    int* d_found_flag = nullptr;
    uint64_t* d_found_offset = nullptr;
    cudaMalloc(&d_found_flag, sizeof(int));
    cudaMalloc(&d_found_offset, sizeof(uint64_t));

    

    int ranges_completed = 0;
    double last_measured_speed = 0.0;
    uint64_t single_range_size = 268435456ULL; // default 2^28 keys

    while (g_running.load()) {
        if (!no_limit && ranges_completed >= max_ranges) {
            std::cout << "\n[STOP] Reached limit of " << max_ranges << " ranges completed without -nl. Exiting cleanly.\n";
            break;
        }

        // Dynamic multiple calculation: target submitting roughly every 30 seconds
        int req_multiple = 1;
        if (last_measured_speed > 0.0) {
            double target_keys = last_measured_speed * 30.0;
            int calc = (int)std::round(target_keys / (double)single_range_size);
            req_multiple = std::max(1, std::min(128, calc));
        }

        std::string req_url = api_base + "?action=range&user=" + user + "&multiple=" + std::to_string(req_multiple);
        if (explicit_puzzle || puzzle_id > 0) {
            req_url += "&puzzle=" + std::to_string(puzzle_id);
        }

        std::string resp;
        if (!http_get(req_url, &resp)) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            continue;
        }

        std::string s_puz = json_get_string(resp, "puzzle");
        if (!s_puz.empty()) {
            try { puzzle_id = std::stoi(s_puz); } catch (...) {}
        }

        std::string str_single_sz = json_get_string(resp, "single_range_size");
        if (!str_single_sz.empty()) {
            try { single_range_size = std::stoull(str_single_sz); } catch (...) {}
        }

        std::string str_rc = json_get_string(resp, "range_count");
        if (str_rc.empty()) str_rc = json_get_string(resp, "multiple");
        int actual_multiple = str_rc.empty() ? req_multiple : std::max(1, std::stoi(str_rc));

        std::string str_block = json_get_string(resp, "block");
        std::string str_range = json_get_string(resp, "range_idx");
        std::string str_start = json_get_string(resp, "start");
        std::string str_end = json_get_string(resp, "end");
        std::string str_target = json_get_string(resp, "target_address");
        if (str_start.empty() || str_target.empty()) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            continue;
        }

        u256 start_k = parse_u256(str_start);
        u256 end_k = str_end.empty() ? (start_k + 268435456ULL) : parse_u256(str_end);
        u256 diff = end_k - start_k;
        uint64_t total_keys_count = (diff.high > 0) ? 268435456ULL : (uint64_t)diff.low;
        if (total_keys_count == 0) total_keys_count = 268435456ULL;

        uint8_t target_h160[20];
        if (!b58check_decode_hash160(str_target, target_h160)) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            continue;
        }

        uint32_t target_w[5];
        for (int j = 0; j < 5; ++j) {
            target_w[j] = (uint32_t)target_h160[j * 4] |
                          ((uint32_t)target_h160[j * 4 + 1] << 8) |
                          ((uint32_t)target_h160[j * 4 + 2] << 16) |
                          ((uint32_t)target_h160[j * 4 + 3] << 24);
        }

        int h_flag = 0;
        uint64_t h_offset = 0;
        cudaMemcpy(d_found_flag, &h_flag, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_found_offset, &h_offset, sizeof(uint64_t), cudaMemcpyHostToDevice);

        uint64_t chunk_step = (uint64_t)grid_threads * 32ULL;
        uint32_t batches = (uint32_t)((total_keys_count + chunk_step - 1) / chunk_step);
        if (batches == 0) batches = 1;

        uint64_t start_limbs[4] = {
            (uint64_t)start_k.low,
            (uint64_t)(start_k.low >> 64),
            (uint64_t)start_k.high,
            (uint64_t)(start_k.high >> 64)
        };

        auto t_start = std::chrono::high_resolution_clock::now();
        cuda_scan_kernel<<<num_blocks, block_size>>>(
            start_limbs[0], start_limbs[1], start_limbs[2], start_limbs[3],
            total_keys_count, grid_threads, batches,
            d_found_flag, d_found_offset,
            target_w[0], target_w[1], target_w[2], target_w[3], target_w[4]
        );
        cudaDeviceSynchronize();
        auto t_end = std::chrono::high_resolution_clock::now();

        cudaMemcpy(&h_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_offset, d_found_offset, sizeof(uint64_t), cudaMemcpyDeviceToHost);

        double elapsed = std::chrono::duration<double>(t_end - t_start).count();
        if (elapsed <= 0.0) elapsed = 0.0001;
        double final_spd = (double)total_keys_count / elapsed;
        ranges_completed += actual_multiple; last_measured_speed = final_spd;

        std::cout << "\r[*] Speed: " << format_speed(final_spd)
                  << " | Done: " << ranges_completed << " ranges" << std::flush;

        if (h_flag == 1) {
            uint64_t found_limbs[4];
            std::memcpy(found_limbs, start_limbs, sizeof(start_limbs));
            u128 s_carry = (u128)found_limbs[0] + h_offset;
            found_limbs[0] = (uint64_t)s_carry;
            s_carry >>= 64;
            s_carry += found_limbs[1];
            found_limbs[1] = (uint64_t)s_carry;
            s_carry >>= 64;
            s_carry += found_limbs[2];
            found_limbs[2] = (uint64_t)s_carry;
            found_limbs[3] += (uint64_t)(s_carry >> 64);

            std::string priv_hex = limbs_to_hex(found_limbs);
            std::cout << "\n[WINNER] TARGET MATCHED! Submitting solution to server...\n";

            std::stringstream res_json;
            res_json << "{\"action\":\"result\",\"puzzle\":" << puzzle_id
                     << ",\"block\":" << str_block
                     << ",\"range_idx\":" << str_range
                     << ",\"range_count\":" << actual_multiple << ",\"multiple\":" << actual_multiple << ",\"status\":\"found\",\"user\":\"" << user
                     << "\",\"private_key\":\"0x" << priv_hex
                     << "\",\"speed\":" << (uint64_t)final_spd << "}";
            std::string ack;
            http_post(api_base, res_json.str(), &ack);
            break;
        } else {
            std::stringstream res_json;
            res_json << "{\"action\":\"result\",\"puzzle\":" << puzzle_id
                     << ",\"block\":" << str_block
                     << ",\"range_idx\":" << str_range
                     << ",\"range_count\":" << actual_multiple << ",\"multiple\":" << actual_multiple << ",\"status\":\"done\",\"user\":\"" << user
                     << "\",\"speed\":" << (uint64_t)final_spd << "}";
            std::string ack;
            http_post(api_base, res_json.str(), &ack);
        }
    }

    cudaFree(d_found_flag);
    cudaFree(d_found_offset);
    return 0;
}
