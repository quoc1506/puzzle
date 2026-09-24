// ============================================================================
// BITCOIN PUZZLE SOLVER - ULTRA-OPTIMIZED NVIDIA CUDA GPU SOLVER (.cu)
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

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifndef CURL_STATICLIB
#define CURL_STATICLIB
#endif

#include <curl/curl.h>
#include <openssl/sha.h>
#include <openssl/ripemd.h>
#include <openssl/bn.h>

#if defined(__NVCC__) || defined(__CUDACC__)
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
#endif

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
    if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
        for (size_t i = 2; i < s.size(); ++i) {
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

static const char* B58_DIGITS = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

bool b58check_decode_hash160(const std::string& addr, uint8_t out_hash160[20]) {
    BIGNUM* bn = BN_new();
    BN_zero(bn);
    BIGNUM* bn58 = BN_new();
    BN_set_word(bn58, 58);
    BIGNUM* bn_char = BN_new();
    BN_CTX* ctx = BN_CTX_new();

    for (char c : addr) {
        const char* p = strchr(B58_DIGITS, c);
        if (!p) {
            BN_free(bn); BN_free(bn58); BN_free(bn_char); BN_CTX_free(ctx);
            return false;
        }
        BN_set_word(bn_char, p - B58_DIGITS);
        BN_mul(bn, bn, bn58, ctx);
        BN_add(bn, bn, bn_char);
    }

    int len = BN_num_bytes(bn);
    std::vector<uint8_t> bin(len, 0);
    BN_bn2bin(bn, bin.data());

    int leading_zeros = 0;
    for (char c : addr) {
        if (c == '1') leading_zeros++;
        else break;
    }

    std::vector<uint8_t> full_bytes;
    full_bytes.insert(full_bytes.end(), leading_zeros, 0);
    full_bytes.insert(full_bytes.end(), bin.begin(), bin.end());

    BN_free(bn); BN_free(bn58); BN_free(bn_char); BN_CTX_free(ctx);

    if (full_bytes.size() != 25) return false;

    uint8_t h1[32], h2[32];
    SHA256(full_bytes.data(), 21, h1);
    SHA256(h1, 32, h2);
    if (std::memcmp(h2, &full_bytes[21], 4) != 0) return false;

    std::memcpy(out_hash160, &full_bytes[1], 20);
    return true;
}

// Secp256k1 Field Element Representation (4 x 64-bit limbs)
struct Fe {
    uint64_t d[4];
};

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
    uint64_t sub_k = borrow & K;
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
#if defined(__SIZEOF_INT128__)
    uint64_t t[8] = {0};
    uint64_t a0 = a.d[0], a1 = a.d[1], a2 = a.d[2], a3 = a.d[3];
    uint64_t b0 = b.d[0], b1 = b.d[1], b2 = b.d[2], b3 = b.d[3];

    u128 c;
    c = (u128)a0 * b0; t[0] = (uint64_t)c; c >>= 64;
    c += (u128)a0 * b1; t[1] = (uint64_t)c; c >>= 64;
    c += (u128)a0 * b2; t[2] = (uint64_t)c; c >>= 64;
    c += (u128)a0 * b3; t[3] = (uint64_t)c; t[4] = (uint64_t)(c >> 64);

    c = (u128)t[1] + (u128)a1 * b0; t[1] = (uint64_t)c; c >>= 64;
    c += (u128)t[2] + (u128)a1 * b1; t[2] = (uint64_t)c; c >>= 64;
    c += (u128)t[3] + (u128)a1 * b2; t[3] = (uint64_t)c; c >>= 64;
    c += (u128)t[4] + (u128)a1 * b3; t[4] = (uint64_t)c; t[5] = (uint64_t)(c >> 64);

    c = (u128)t[2] + (u128)a2 * b0; t[2] = (uint64_t)c; c >>= 64;
    c += (u128)t[3] + (u128)a2 * b1; t[3] = (uint64_t)c; c >>= 64;
    c += (u128)t[4] + (u128)a2 * b2; t[4] = (uint64_t)c; c >>= 64;
    c += (u128)t[5] + (u128)a2 * b3; t[5] = (uint64_t)c; t[6] = (uint64_t)(c >> 64);

    c = (u128)t[3] + (u128)a3 * b0; t[3] = (uint64_t)c; c >>= 64;
    c += (u128)t[4] + (u128)a3 * b1; t[4] = (uint64_t)c; c >>= 64;
    c += (u128)t[5] + (u128)a3 * b2; t[5] = (uint64_t)c; c >>= 64;
    c += (u128)t[6] + (u128)a3 * b3; t[6] = (uint64_t)c; t[7] = (uint64_t)(c >> 64);

    const uint64_t SECP_K = 0x1000003D1ULL;
    u128 carry = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        u128 prod = (u128)t[4 + i] * SECP_K + t[i] + carry;
        t[i] = (uint64_t)prod;
        carry = prod >> 64;
    }
    u128 c2 = (u128)t[0] + (u128)carry * SECP_K;
    t[0] = (uint64_t)c2; c2 >>= 64;
    c2 += t[1]; t[1] = (uint64_t)c2; c2 >>= 64;
    c2 += t[2]; t[2] = (uint64_t)c2; c2 >>= 64;
    c2 += t[3]; t[3] = (uint64_t)c2; c2 >>= 64;
    uint64_t extra = (uint64_t)c2;
    if (__builtin_expect(extra != 0, 0)) {
        u128 c3 = (u128)t[0] + (u128)extra * SECP_K;
        t[0] = (uint64_t)c3; c3 >>= 64;
        c3 += t[1]; t[1] = (uint64_t)c3; c3 >>= 64;
        c3 += t[2]; t[2] = (uint64_t)c3; c3 >>= 64;
        t[3] += (uint64_t)c3;
    }

    if (__builtin_expect(t[3] == 0xFFFFFFFFFFFFFFFFULL &&
        t[2] == 0xFFFFFFFFFFFFFFFFULL &&
        t[1] == 0xFFFFFFFFFFFFFFFFULL &&
        t[0] >= 0xFFFFFFFEFFFFFC2FULL, 0)) {
        t[0] -= 0xFFFFFFFEFFFFFC2FULL;
        t[1] = 0;
        t[2] = 0;
        t[3] = 0;
    }
    Fe r;
    r.d[0] = t[0]; r.d[1] = t[1]; r.d[2] = t[2]; r.d[3] = t[3];
    return r;
#else
    uint64_t t[8] = {0};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint64_t carry = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint64_t prod_lo = a.d[i] * b.d[j];
#if defined(__CUDA_ARCH__)
            uint64_t prod_hi = __umul64hi(a.d[i], b.d[j]);
#elif defined(__SIZEOF_INT128__)
            uint64_t prod_hi = (uint64_t)(((unsigned __int128)a.d[i] * b.d[j]) >> 64);
#else
            uint64_t prod_hi = 0;
#endif
            uint64_t sum1 = t[i + j] + prod_lo;
            uint64_t c1 = (sum1 < t[i + j]);
            uint64_t sum2 = sum1 + carry;
            uint64_t c2 = (sum2 < sum1);
            t[i + j] = sum2;
            carry = prod_hi + c1 + c2;
        }
        t[i + 4] = carry;
    }
    return fe_reduce(t);
#endif
}

CUDA_HOSTDEV CUDA_INLINE Fe fe_sqr(const Fe& a) {
#if defined(__SIZEOF_INT128__)
    uint64_t t[8] = {0};
    uint64_t a0 = a.d[0], a1 = a.d[1], a2 = a.d[2], a3 = a.d[3];

    u128 c = (u128)a0 * a1; t[1] = (uint64_t)c; c >>= 64;
    c += (u128)a0 * a2; t[2] = (uint64_t)c; c >>= 64;
    c += (u128)a0 * a3; t[3] = (uint64_t)c; t[4] = (uint64_t)(c >> 64);

    c = (u128)t[3] + (u128)a1 * a2; t[3] = (uint64_t)c; c >>= 64;
    c += (u128)t[4] + (u128)a1 * a3; t[4] = (uint64_t)c; t[5] = (uint64_t)(c >> 64);

    c = (u128)t[5] + (u128)a2 * a3; t[5] = (uint64_t)c; t[6] = (uint64_t)(c >> 64);

    uint64_t carry = 0;
    #pragma unroll
    for (int i = 1; i < 7; ++i) {
        uint64_t v = (t[i] << 1) | carry;
        carry = t[i] >> 63;
        t[i] = v;
    }
    t[7] = carry;

    c = (u128)t[0] + (u128)a0 * a0; t[0] = (uint64_t)c; c >>= 64;
    c += (u128)t[1]; t[1] = (uint64_t)c; c >>= 64;
    c += (u128)t[2] + (u128)a1 * a1; t[2] = (uint64_t)c; c >>= 64;
    c += (u128)t[3]; t[3] = (uint64_t)c; c >>= 64;
    c += (u128)t[4] + (u128)a2 * a2; t[4] = (uint64_t)c; c >>= 64;
    c += (u128)t[5]; t[5] = (uint64_t)c; c >>= 64;
    c += (u128)t[6] + (u128)a3 * a3; t[6] = (uint64_t)c; c >>= 64;
    t[7] += (uint64_t)c;

    const uint64_t SECP_K = 0x1000003D1ULL;
    u128 red_carry = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        u128 prod = (u128)t[4 + i] * SECP_K + t[i] + red_carry;
        t[i] = (uint64_t)prod;
        red_carry = prod >> 64;
    }
    u128 c2 = (u128)t[0] + (u128)red_carry * SECP_K;
    t[0] = (uint64_t)c2; c2 >>= 64;
    c2 += t[1]; t[1] = (uint64_t)c2; c2 >>= 64;
    c2 += t[2]; t[2] = (uint64_t)c2; c2 >>= 64;
    c2 += t[3]; t[3] = (uint64_t)c2; c2 >>= 64;
    uint64_t extra = (uint64_t)c2;
    if (__builtin_expect(extra != 0, 0)) {
        u128 c3 = (u128)t[0] + (u128)extra * SECP_K;
        t[0] = (uint64_t)c3; c3 >>= 64;
        c3 += t[1]; t[1] = (uint64_t)c3; c3 >>= 64;
        c3 += t[2]; t[2] = (uint64_t)c3; c3 >>= 64;
        t[3] += (uint64_t)c3;
    }

    if (__builtin_expect(t[3] == 0xFFFFFFFFFFFFFFFFULL &&
        t[2] == 0xFFFFFFFFFFFFFFFFULL &&
        t[1] == 0xFFFFFFFFFFFFFFFFULL &&
        t[0] >= 0xFFFFFFFEFFFFFC2FULL, 0)) {
        t[0] -= 0xFFFFFFFEFFFFFC2FULL;
        t[1] = 0;
        t[2] = 0;
        t[3] = 0;
    }
    Fe r;
    r.d[0] = t[0]; r.d[1] = t[1]; r.d[2] = t[2]; r.d[3] = t[3];
    return r;
#else
    return fe_mul(a, a);
#endif
}

CUDA_HOSTDEV CUDA_INLINE Fe fe_sqr_n(Fe a, int n) {
    #pragma unroll
    for (int i = 0; i < n; ++i) {
        a = fe_sqr(a);
    }
    return a;
}

// 100% Mathematically Verified Inversion modulo 2^256 - 2^32 - 977 (22-step Addition Chain)
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

    // Top 223 bits (all 1s) shifted by 33 bits
    Fe t = fe_sqr_n(x223, 33);

    // Low 33 bits: bit 32 is 0, bits 31..0 is 0xFFFFFC2D
    Fe x14 = fe_mul(fe_sqr_n(x11, 3), x3);
    Fe x16 = fe_mul(fe_sqr_n(x14, 2), x2);
    Fe low = fe_sqr_n(x16, 16);

    Fe b = a;
    Fe low16 = a;
    b = fe_sqr(b);
    b = fe_sqr(b);
    low16 = fe_mul(low16, b);
    b = fe_sqr(b);
    low16 = fe_mul(low16, b);
    b = fe_sqr(b);
    b = fe_sqr(b);
    low16 = fe_mul(low16, b);
    b = fe_sqr_n(b, 5);
    low16 = fe_mul(low16, b);
    b = fe_sqr(b); low16 = fe_mul(low16, b);
    b = fe_sqr(b); low16 = fe_mul(low16, b);
    b = fe_sqr(b); low16 = fe_mul(low16, b);
    b = fe_sqr(b); low16 = fe_mul(low16, b);
    b = fe_sqr(b); low16 = fe_mul(low16, b);
    low = fe_mul(low, low16);
    return fe_mul(t, low);
}

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

CUDA_HOSTDEV CUDA_INLINE AffinePoint get_generator_G() {
    AffinePoint G;
    G.x.d[0] = 0x59F2815B16F81798ULL;
    G.x.d[1] = 0x029BFCDB2DCE28D9ULL;
    G.x.d[2] = 0x55A06295CE870B07ULL;
    G.x.d[3] = 0x79BE667EF9DCBBACULL;

    G.y.d[0] = 0x9C47D08FFB10D4B8ULL;
    G.y.d[1] = 0xFD17B448A6855419ULL;
    G.y.d[2] = 0x5DA4FBFC0E1108A8ULL;
    G.y.d[3] = 0x483ADA7726A3C465ULL;
    return G;
}

CUDA_HOSTDEV CUDA_INLINE JacobianPoint jacobian_double(const JacobianPoint& p) {
    if (p.infinity || fe_is_zero(p.y)) return p;
    Fe XX = fe_sqr(p.x);
    Fe YY = fe_sqr(p.y);
    Fe YYYY = fe_sqr(YY);
    Fe ZZ = fe_sqr(p.z);

    Fe S = fe_add(p.x, YY);
    S = fe_sub(fe_sqr(S), XX);
    S = fe_sub(S, YYYY);
    S = fe_add(S, S);

    Fe M = fe_add(fe_add(XX, XX), XX);

    Fe T = fe_sub(fe_sqr(M), fe_add(S, S));

    JacobianPoint r;
    r.infinity = false;
    r.x = T;

    Fe YYYY8 = fe_add(YYYY, YYYY);
    YYYY8 = fe_add(YYYY8, YYYY8);
    YYYY8 = fe_add(YYYY8, YYYY8);

    r.y = fe_sub(fe_mul(M, fe_sub(S, T)), YYYY8);

    Fe YZ = fe_add(p.y, p.z);
    r.z = fe_sub(fe_sub(fe_sqr(YZ), YY), ZZ);
    return r;
}

CUDA_HOSTDEV CUDA_INLINE JacobianPoint jacobian_add_affine(const JacobianPoint& p, const AffinePoint& a) {
    if (p.infinity) {
        JacobianPoint r;
        r.x = a.x;
        r.y = a.y;
        r.z.d[0] = 1; r.z.d[1] = 0; r.z.d[2] = 0; r.z.d[3] = 0;
        r.infinity = false;
        return r;
    }
    Fe Z1Z1 = fe_sqr(p.z);
    Fe U2 = fe_mul(a.x, Z1Z1);
    Fe S2 = fe_mul(fe_mul(a.y, p.z), Z1Z1);

    Fe H = fe_sub(U2, p.x);
    Fe R = fe_sub(S2, p.y);

    if (fe_is_zero(H)) {
        if (fe_is_zero(R)) {
            return jacobian_double(p);
        } else {
            JacobianPoint inf;
            inf.infinity = true;
            return inf;
        }
    }

    Fe HH = fe_sqr(H);
    Fe HHH = fe_mul(H, HH);
    Fe V = fe_mul(p.x, HH);

    JacobianPoint res;
    res.infinity = false;
    res.x = fe_sub(fe_sub(fe_sqr(R), HHH), fe_add(V, V));
    res.y = fe_sub(fe_mul(R, fe_sub(V, res.x)), fe_mul(p.y, HHH));
    res.z = fe_mul(p.z, H);
    return res;
}

CUDA_HOSTDEV CUDA_INLINE AffinePoint jacobian_to_affine(const JacobianPoint& p) {
    AffinePoint a;
    if (p.infinity) {
        std::memset(&a, 0, sizeof(a));
        return a;
    }
    Fe z_inv = fe_inv(p.z);
    Fe z_inv2 = fe_sqr(z_inv);
    Fe z_inv3 = fe_mul(z_inv2, z_inv);
    a.x = fe_mul(p.x, z_inv2);
    a.y = fe_mul(p.y, z_inv3);
    return a;
}

CUDA_HOSTDEV CUDA_INLINE AffinePoint scalar_mul_G(const uint64_t scalar[4]) {
    JacobianPoint res;
    res.infinity = true;
    AffinePoint G = get_generator_G();
    JacobianPoint base;
    base.x = G.x;
    base.y = G.y;
    base.z.d[0] = 1; base.z.d[1] = 0; base.z.d[2] = 0; base.z.d[3] = 0;
    base.infinity = false;

    for (int limb = 0; limb < 4; ++limb) {
        uint64_t w = scalar[limb];
        for (int b = 0; b < 64; ++b) {
            if ((w >> b) & 1) {
                res = jacobian_add_affine(res, jacobian_to_affine(base));
            }
            base = jacobian_double(base);
        }
    }
    return jacobian_to_affine(res);
}

CUDA_HOSTDEV CUDA_INLINE uint32_t ror32_dev(uint32_t x, int n) {
#if defined(__CUDA_ARCH__)
    return __funnelshift_r(x, x, n);
#elif defined(_MSC_VER)
    return _rotr(x, n);
#elif defined(__GNUC__) || defined(__clang__)
    return (x >> n) | (x << (32 - n));
#else
    return (x >> n) | (x << (32 - n));
#endif
}

CUDA_HOSTDEV CUDA_INLINE uint32_t rol32_dev(uint32_t x, int n) {
#if defined(__CUDA_ARCH__)
    return __funnelshift_l(x, x, n);
#elif defined(_MSC_VER)
    return _rotl(x, n);
#elif defined(__GNUC__) || defined(__clang__)
    return (x << n) | (x >> (32 - n));
#else
    return (x << n) | (x >> (32 - n));
#endif
}

CUDA_HOSTDEV CUDA_INLINE uint32_t bswap32_dev(uint32_t x) {
#if defined(__CUDA_ARCH__)
    return __byte_perm(x, 0, 0x0123);
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap32(x);
#else
    return ((x & 0xFF000000U) >> 24) |
           ((x & 0x00FF0000U) >> 8)  |
           ((x & 0x0000FF00U) << 8)  |
           ((x & 0x000000FFU) << 24);
#endif
}

template<uint32_t imm>
CUDA_HOSTDEV CUDA_INLINE uint32_t lop3_b32(uint32_t a, uint32_t b, uint32_t c) {
#if defined(__CUDA_ARCH__)
    uint32_t res;
    asm("lop3.b32 %0, %1, %2, %3, %4;" : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(imm));
    return res;
#else
    uint32_t res = 0;
    if constexpr (imm & 0x01) res |= (~a & ~b & ~c);
    if constexpr (imm & 0x02) res |= (~a & ~b &  c);
    if constexpr (imm & 0x04) res |= (~a &  b & ~c);
    if constexpr (imm & 0x08) res |= (~a &  b &  c);
    if constexpr (imm & 0x10) res |= ( a & ~b & ~c);
    if constexpr (imm & 0x20) res |= ( a & ~b &  c);
    if constexpr (imm & 0x40) res |= ( a &  b & ~c);
    if constexpr (imm & 0x80) res |= ( a &  b &  c);
    return res;
#endif
}

static constexpr uint32_t host_K_SHA256[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U
};

static constexpr uint8_t host_rl_tab[80] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
    3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
    1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
    4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13
};

static constexpr uint8_t host_sl_tab[80] = {
    11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
    7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
    11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
    11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
    9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6
};

static constexpr uint8_t host_rr_tab[80] = {
    5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
    6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
    15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
    8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
    12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11
};

static constexpr uint8_t host_sr_tab[80] = {
    8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
    9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
    9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
    15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
    8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11
};

template<int i>
CUDA_HOSTDEV CUDA_INLINE void sha256_round_dev(
    uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d,
    uint32_t& e, uint32_t& f, uint32_t& g, uint32_t& h,
    uint32_t wi
) {
    constexpr uint32_t K = host_K_SHA256[i];
    uint32_t S1 = ror32_dev(e, 6) ^ ror32_dev(e, 11) ^ ror32_dev(e, 25);
    uint32_t ch = lop3_b32<0xCA>(e, f, g);
    uint32_t temp1;
    if constexpr (i >= 9 && i <= 14) {
        temp1 = h + S1 + ch + K;
    } else {
        temp1 = h + S1 + ch + K + wi;
    }
    uint32_t S0 = ror32_dev(a, 2) ^ ror32_dev(a, 13) ^ ror32_dev(a, 22);
    uint32_t maj = lop3_b32<0xE8>(a, b, c);
    uint32_t temp2 = S0 + maj;

    h = g; g = f; f = e; e = d + temp1;
    d = c; c = b; b = a; a = temp1 + temp2;
}

template<size_t... Is>
CUDA_HOSTDEV CUDA_INLINE void run_sha256_first16_dev(
    uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d,
    uint32_t& e, uint32_t& f, uint32_t& g, uint32_t& h,
    const uint32_t w[16],
    std::index_sequence<Is...>
) {
    (sha256_round_dev<Is>(a, b, c, d, e, f, g, h, w[Is]), ...);
}

template<int i>
CUDA_HOSTDEV CUDA_INLINE void sha256_step_and_round_dev(
    uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d,
    uint32_t& e, uint32_t& f, uint32_t& g, uint32_t& h,
    uint32_t w[16]
) {
    uint32_t s0, s1, wi;
    if constexpr (i == 16) {
        s0 = ror32_dev(w[1], 7) ^ ror32_dev(w[1], 18) ^ (w[1] >> 3);
        wi = w[0] + s0;
    } else if constexpr (i == 17) {
        s0 = ror32_dev(w[2], 7) ^ ror32_dev(w[2], 18) ^ (w[2] >> 3);
        wi = w[1] + s0 + 0x00A50000U;
    } else if constexpr (i >= 18 && i <= 21) {
        s0 = ror32_dev(w[(i - 15) & 15], 7) ^ ror32_dev(w[(i - 15) & 15], 18) ^ (w[(i - 15) & 15] >> 3);
        s1 = ror32_dev(w[(i - 2) & 15], 17) ^ ror32_dev(w[(i - 2) & 15], 19) ^ (w[(i - 2) & 15] >> 10);
        wi = w[(i - 16) & 15] + s0 + s1;
    } else if constexpr (i == 22) {
        s0 = ror32_dev(w[7], 7) ^ ror32_dev(w[7], 18) ^ (w[7] >> 3);
        s1 = ror32_dev(w[4], 17) ^ ror32_dev(w[4], 19) ^ (w[4] >> 10);
        wi = w[6] + s0 + s1 + 264U;
    } else if constexpr (i == 24) {
        s1 = ror32_dev(w[6], 17) ^ ror32_dev(w[6], 19) ^ (w[6] >> 10);
        wi = w[8] + w[1] + s1;
    } else if constexpr (i >= 25 && i <= 29) {
        s1 = ror32_dev(w[(i - 2) & 15], 17) ^ ror32_dev(w[(i - 2) & 15], 19) ^ (w[(i - 2) & 15] >> 10);
        wi = w[(i - 7) & 15] + s1;
    } else if constexpr (i == 30) {
        s1 = ror32_dev(w[12], 17) ^ ror32_dev(w[12], 19) ^ (w[12] >> 10);
        wi = 0x10420023U + w[7] + s1;
    } else if constexpr (i == 31) {
        s0 = ror32_dev(w[0], 7) ^ ror32_dev(w[0], 18) ^ (w[0] >> 3);
        s1 = ror32_dev(w[13], 17) ^ ror32_dev(w[13], 19) ^ (w[13] >> 10);
        wi = 264U + s0 + w[8] + s1;
    } else {
        s0 = ror32_dev(w[(i - 15) & 15], 7) ^ ror32_dev(w[(i - 15) & 15], 18) ^ (w[(i - 15) & 15] >> 3);
        s1 = ror32_dev(w[(i - 2) & 15], 17) ^ ror32_dev(w[(i - 2) & 15], 19) ^ (w[(i - 2) & 15] >> 10);
        wi = w[(i - 16) & 15] + s0 + w[(i - 7) & 15] + s1;
    }
    w[i & 15] = wi;
    sha256_round_dev<i>(a, b, c, d, e, f, g, h, wi);
}

template<size_t... Is>
CUDA_HOSTDEV CUDA_INLINE void run_sha256_rest_dev(
    uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d,
    uint32_t& e, uint32_t& f, uint32_t& g, uint32_t& h,
    uint32_t w[16],
    std::index_sequence<Is...>
) {
    (sha256_step_and_round_dev<16 + Is>(a, b, c, d, e, f, g, h, w), ...);
}

CUDA_HOSTDEV CUDA_INLINE void fast_sha256_into_ripemd_X(uint8_t prefix, const Fe& x, uint32_t X[8]) {
    uint32_t w[16];
    w[0] = ((uint32_t)prefix << 24) | (uint32_t)(x.d[3] >> 40);
    w[1] = (uint32_t)(x.d[3] >> 8);
    w[2] = ((uint32_t)x.d[3] << 24) | (uint32_t)(x.d[2] >> 40);
    w[3] = (uint32_t)(x.d[2] >> 8);
    w[4] = ((uint32_t)x.d[2] << 24) | (uint32_t)(x.d[1] >> 40);
    w[5] = (uint32_t)(x.d[1] >> 8);
    w[6] = ((uint32_t)x.d[1] << 24) | (uint32_t)(x.d[0] >> 40);
    w[7] = (uint32_t)(x.d[0] >> 8);
    w[8] = ((uint32_t)x.d[0] << 24) | 0x00800000U;
    w[9] = 0; w[10] = 0; w[11] = 0; w[12] = 0; w[13] = 0; w[14] = 0;
    w[15] = 264;

    uint32_t a = 0x6a09e667, b = 0xbb67ae85, c = 0x3c6ef372, d = 0xa54ff53a;
    uint32_t e = 0x510e527f, f = 0x9b05688c, g = 0x1f83d9ab, h = 0x5be0cd19;

    run_sha256_first16_dev(a, b, c, d, e, f, g, h, w, std::make_index_sequence<16>{});
    run_sha256_rest_dev(a, b, c, d, e, f, g, h, w, std::make_index_sequence<48>{});

    X[0] = bswap32_dev(0x6a09e667 + a);
    X[1] = bswap32_dev(0xbb67ae85 + b);
    X[2] = bswap32_dev(0x3c6ef372 + c);
    X[3] = bswap32_dev(0xa54ff53a + d);
    X[4] = bswap32_dev(0x510e527f + e);
    X[5] = bswap32_dev(0x9b05688c + f);
    X[6] = bswap32_dev(0x1f83d9ab + g);
    X[7] = bswap32_dev(0x5be0cd19 + h);
}

template<int j>
CUDA_HOSTDEV CUDA_INLINE void ripemd160_left_step_dev(
    uint32_t& A, uint32_t& B, uint32_t& C, uint32_t& D, uint32_t& E,
    const uint32_t X[8]
) {
    constexpr uint8_t rl = host_rl_tab[j];
    constexpr uint8_t sl = host_sl_tab[j];
    uint32_t f;
    if constexpr (j < 16) {
        f = lop3_b32<0x96>(B, C, D);
    } else if constexpr (j < 32) {
        f = lop3_b32<0xCA>(B, C, D);
    } else if constexpr (j < 48) {
        f = lop3_b32<0x59>(B, C, D);
    } else if constexpr (j < 64) {
        f = lop3_b32<0xE4>(B, C, D);
    } else {
        f = lop3_b32<0x2D>(B, C, D);
    }
    constexpr uint32_t K = (j < 16) ? 0 : (j < 32) ? 0x5A827999U : (j < 48) ? 0x6ED9EBA1U : (j < 64) ? 0x8F1BBCDCU : 0xA953FD4EU;
    uint32_t x_val;
    if constexpr (rl < 8) {
        x_val = X[rl];
    } else if constexpr (rl == 8) {
        x_val = 0x00000080U;
    } else if constexpr (rl == 14) {
        x_val = 256U;
    } else {
        x_val = 0U;
    }
    uint32_t T = rol32_dev(A + f + x_val + K, sl) + E;
    A = E; E = D; D = rol32_dev(C, 10); C = B; B = T;
}

template<size_t... Is>
CUDA_HOSTDEV CUDA_INLINE void run_ripemd160_left_dev(
    uint32_t& A, uint32_t& B, uint32_t& C, uint32_t& D, uint32_t& E,
    const uint32_t X[8],
    std::index_sequence<Is...>
) {
    (ripemd160_left_step_dev<Is>(A, B, C, D, E, X), ...);
}

template<int j>
CUDA_HOSTDEV CUDA_INLINE void ripemd160_right_step_dev(
    uint32_t& Ap, uint32_t& Bp, uint32_t& Cp, uint32_t& Dp, uint32_t& Ep,
    const uint32_t X[8]
) {
    constexpr uint8_t rr = host_rr_tab[j];
    constexpr uint8_t sr = host_sr_tab[j];
    uint32_t fp;
    if constexpr (j < 16) {
        fp = lop3_b32<0x2D>(Bp, Cp, Dp);
    } else if constexpr (j < 32) {
        fp = lop3_b32<0xE4>(Bp, Cp, Dp);
    } else if constexpr (j < 48) {
        fp = lop3_b32<0x59>(Bp, Cp, Dp);
    } else if constexpr (j < 64) {
        fp = lop3_b32<0xCA>(Bp, Cp, Dp);
    } else {
        fp = lop3_b32<0x96>(Bp, Cp, Dp);
    }
    constexpr uint32_t Kp = (j < 16) ? 0x50A28BE6U : (j < 32) ? 0x5C4DD124U : (j < 48) ? 0x6D703EF3U : (j < 64) ? 0x7A6D76E9U : 0;
    uint32_t x_val;
    if constexpr (rr < 8) {
        x_val = X[rr];
    } else if constexpr (rr == 8) {
        x_val = 0x00000080U;
    } else if constexpr (rr == 14) {
        x_val = 256U;
    } else {
        x_val = 0U;
    }
    uint32_t Tp = rol32_dev(Ap + fp + x_val + Kp, sr) + Ep;
    Ap = Ep; Ep = Dp; Dp = rol32_dev(Cp, 10); Cp = Bp; Bp = Tp;
}

template<size_t... Is>
CUDA_HOSTDEV CUDA_INLINE void run_ripemd160_right_dev(
    uint32_t& Ap, uint32_t& Bp, uint32_t& Cp, uint32_t& Dp, uint32_t& Ep,
    const uint32_t X[8],
    std::index_sequence<Is...>
) {
    (ripemd160_right_step_dev<Is>(Ap, Bp, Cp, Dp, Ep, X), ...);
}

CUDA_HOSTDEV CUDA_INLINE bool fast_ripemd160_32_check(const uint32_t X[8], const uint32_t target_w[5]) {
    uint32_t A = 0x67452301, B = 0xEFCDAB89, C = 0x98BADCFE, D = 0x10325476, E = 0xC3D2E1F0;
    run_ripemd160_left_dev(A, B, C, D, E, X, std::make_index_sequence<80>{});

    uint32_t c_left = C;
    uint32_t d_left = D;
    uint32_t e_left = E;
    uint32_t a_left = A;
    uint32_t b_left = B;

    uint32_t Ap = 0x67452301, Bp = 0xEFCDAB89, Cp = 0x98BADCFE, Dp = 0x10325476, Ep = 0xC3D2E1F0;
    run_ripemd160_right_dev(Ap, Bp, Cp, Dp, Ep, X, std::make_index_sequence<80>{});

    // Fast early check on word 0
    if ((0xEFCDAB89U + c_left + Dp) != target_w[0]) return false;
    if ((0x98BADCFEU + d_left + Ep) != target_w[1]) return false;
    if ((0x10325476U + e_left + Ap) != target_w[2]) return false;
    if ((0xC3D2E1F0U + a_left + Bp) != target_w[3]) return false;
    return ((0x67452301U + b_left + Cp) == target_w[4]);
}

CUDA_HOSTDEV CUDA_INLINE void fast_ripemd160_32(const uint32_t X[8], uint32_t out_h[5]) {
    uint32_t A = 0x67452301, B = 0xEFCDAB89, C = 0x98BADCFE, D = 0x10325476, E = 0xC3D2E1F0;
    run_ripemd160_left_dev(A, B, C, D, E, X, std::make_index_sequence<80>{});

    uint32_t c_left = C;
    uint32_t d_left = D;
    uint32_t e_left = E;
    uint32_t a_left = A;
    uint32_t b_left = B;

    uint32_t Ap = 0x67452301, Bp = 0xEFCDAB89, Cp = 0x98BADCFE, Dp = 0x10325476, Ep = 0xC3D2E1F0;
    run_ripemd160_right_dev(Ap, Bp, Cp, Dp, Ep, X, std::make_index_sequence<80>{});

    out_h[0] = 0xEFCDAB89 + c_left + Dp;
    out_h[1] = 0x98BADCFE + d_left + Ep;
    out_h[2] = 0x10325476 + e_left + Ap;
    out_h[3] = 0xC3D2E1F0 + a_left + Bp;
    out_h[4] = 0x67452301 + b_left + Cp;
}




// ============================================================================
// CUDA GPU CONSTANTS & LOCKSTEP SIMD MONTGOMERY BATCH INVERSION KERNEL
// ============================================================================
CUDA_CONSTANT AffinePoint dev_G_table[16];
CUDA_CONSTANT AffinePoint dev_batch_G[16];
CUDA_CONSTANT uint32_t dev_target_w[5];
CUDA_CONSTANT uint64_t dev_target_h64;
CUDA_DEV int dev_found_flag = 0;
CUDA_DEV uint64_t dev_found_offset = 0;

CUDA_DEV AffinePoint scalar_mul_G_windowed(const u256& scalar) {
    JacobianPoint res;
    res.infinity = true;
    for (int limb = 3; limb >= 0; --limb) {
        uint64_t w = (limb >= 2) ? (uint64_t)(scalar.high >> ((limb - 2) * 64)) : (uint64_t)(scalar.low >> (limb * 64));
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

CUDA_DEV CUDA_INLINE bool check_point_hash160(const AffinePoint& pt) {
    uint8_t prefix = (pt.y.d[0] & 1) ? 0x03 : 0x02;
    uint32_t X[8];
    fast_sha256_into_ripemd_X(prefix, pt.x, X);
    return fast_ripemd160_32_check(X, dev_target_w);
}

CUDA_GLOBAL void cuda_scan_kernel(
    u256 start_key,
    uint64_t total_chunk_keys,
    uint32_t grid_threads,
    uint32_t batches
) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= grid_threads) return;

    uint64_t thread_start_offset = (uint64_t)tid;
    if (thread_start_offset >= total_chunk_keys) return;

    u256 cur_key = start_key + thread_start_offset;
    AffinePoint cur_P = scalar_mul_G_windowed(cur_key);

    if (check_point_hash160(cur_P)) {
        atomicExch(&dev_found_flag, 1);
        atomicExch((unsigned long long*)&dev_found_offset, (unsigned long long)thread_start_offset);
        return;
    }

    uint64_t step_keys = (uint64_t)grid_threads;

    for (uint32_t b = 0; b < batches; ++b) {
        if (dev_found_flag) return;

        uint64_t batch_base_offset = thread_start_offset + (uint64_t)b * 16ULL * step_keys;
        if (batch_base_offset >= total_chunk_keys) return;

        Fe dx[16];
        Fe dy[16];
        Fe prod[16];

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            dx[i] = fe_sub(dev_batch_G[i].x, cur_P.x);
            dy[i] = fe_sub(dev_batch_G[i].y, cur_P.y);
        }

        prod[0] = dx[0];
        #pragma unroll
        for (int i = 1; i < 16; ++i) {
            prod[i] = fe_mul(prod[i - 1], dx[i]);
        }

        Fe inv_all = fe_inv(prod[15]);

        Fe inv_dx[16];
        #pragma unroll
        for (int i = 15; i >= 1; --i) {
            inv_dx[i] = fe_mul(inv_all, prod[i - 1]);
            inv_all = fe_mul(inv_all, dx[i]);
        }
        inv_dx[0] = inv_all;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            uint64_t key_offset = batch_base_offset + (uint64_t)(i + 1) * step_keys;
            if (key_offset < total_chunk_keys) {
                Fe lambda = fe_mul(dy[i], inv_dx[i]);
                Fe lambda_sq = fe_sqr(lambda);
                Fe next_x = fe_sub(fe_sub(lambda_sq, cur_P.x), dev_batch_G[i].x);
                Fe diff_x = fe_sub(cur_P.x, next_x);
                Fe next_y = fe_sub(fe_mul(lambda, diff_x), cur_P.y);

                AffinePoint cand{next_x, next_y};
                if (check_point_hash160(cand)) {
                    atomicExch(&dev_found_flag, 1);
                    atomicExch((unsigned long long*)&dev_found_offset, (unsigned long long)key_offset);
                    return;
                }
            }
        }

        Fe lambda_adv = fe_mul(dy[15], inv_dx[15]);
        Fe lambda_adv_sq = fe_sqr(lambda_adv);
        Fe adv_x = fe_sub(fe_sub(lambda_adv_sq, cur_P.x), dev_batch_G[15].x);
        Fe diff_adv_x = fe_sub(cur_P.x, adv_x);
        Fe adv_y = fe_sub(fe_mul(lambda_adv, diff_adv_x), cur_P.y);
        cur_P = AffinePoint{adv_x, adv_y};
    }
}


static size_t curl_cb(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t total = size * nmemb;
    std::string* s = (std::string*)userp;
    s->append((char*)contents, total);
    return total;
}

bool http_get(const std::string& url, std::string* out) {
    CURL* curl = curl_easy_init();
    if (!curl) return false;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, out);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 15L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    return (res == CURLE_OK);
}

bool http_post(const std::string& url, const std::string& json_data, std::string* out) {
    CURL* curl = curl_easy_init();
    if (!curl) return false;
    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_data.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, out);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 15L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return (res == CURLE_OK);
}

std::string json_get_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\":";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos += search.length();
    while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\r' || json[pos] == '\n')) pos++;
    if (pos >= json.length()) return "";

    if (json[pos] == '\"') {
        pos++;
        size_t end = json.find('\"', pos);
        if (end == std::string::npos) return "";
        return json.substr(pos, end - pos);
    } else {
        size_t end = json.find_first_of(",}\r\n \t", pos);
        if (end == std::string::npos) end = json.length();
        return json.substr(pos, end - pos);
    }
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT, sigint_handler);
    std::signal(SIGTERM, sigint_handler);
    curl_global_init(CURL_GLOBAL_DEFAULT);

    std::string api_base = "http://65.20.91.208/puzzle_server.php";
    int current_puzzle = 71;
    std::string current_user = "guest";
    int requested_multiple = 4;
    int target_device_id = 0;
    bool is_fast = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-s" || arg == "--server") && i + 1 < argc) {
            api_base = argv[++i];
        } else if ((arg == "-p" || arg == "--puzzle") && i + 1 < argc) {
            current_puzzle = std::atoi(argv[++i]);
        } else if ((arg == "-u" || arg == "--user") && i + 1 < argc) {
            current_user = argv[++i];
        } else if ((arg == "-m" || arg == "--multiple") && i + 1 < argc) {
            requested_multiple = std::atoi(argv[++i]);
        } else if ((arg == "-d" || arg == "--device") && i + 1 < argc) {
            target_device_id = std::atoi(argv[++i]);
        } else if (arg == "--fast") {
            is_fast = true;
        } else if (arg == "--safe") {
            is_fast = false;
        } else if (arg == "-h" || arg == "--help") {
            return 0;
        }
    }

    if (requested_multiple <= 0) requested_multiple = 1;
    if (requested_multiple > 1024) requested_multiple = 1024;

    cudaError_t dev_err = cudaSetDevice(target_device_id);
    if (dev_err != cudaSuccess) {
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, target_device_id);

    // Initialize Base Point Precomputed Lookup Tables on Host
    AffinePoint h_table[16];
    std::memset(&h_table[0], 0, sizeof(AffinePoint));
    for (int i = 1; i < 16; ++i) {
        uint64_t s[4] = { (uint64_t)i, 0, 0, 0 };
        h_table[i] = scalar_mul_G(s);
    }
    cudaMemcpyToSymbol(dev_G_table, h_table, sizeof(h_table));

    // Auto-tune Grid & Block dimensions for maximum GPU SM saturation
    uint32_t num_sms = prop.multiProcessorCount > 0 ? prop.multiProcessorCount : 40;
    uint32_t threadsPerBlock = 128;
    uint32_t numBlocks = num_sms * (is_fast ? 32 : 16);
    uint32_t grid_threads = numBlocks * threadsPerBlock;
    uint32_t steps_per_launch = is_fast ? 65536 : 32768;
    uint64_t chunk_size = (uint64_t)grid_threads * steps_per_launch;

    // Precompute Batch G Points for 16-way Lockstep SIMD Montgomery Inversion
    AffinePoint h_batch_G[16];
    for (int i = 0; i < 16; ++i) {
        uint64_t step_mult = (uint64_t)grid_threads * (uint64_t)(i + 1);
        uint64_t s[4] = { step_mult, 0, 0, 0 };
        h_batch_G[i] = scalar_mul_G(s);
    }
    cudaMemcpyToSymbol(dev_batch_G, h_batch_G, sizeof(h_batch_G));

    while (g_running.load()) {
        std::stringstream req_url;
        req_url << api_base << "?action=range&puzzle=" << current_puzzle
                << "&user=" << current_user << "&multiple=" << requested_multiple;

        std::string resp;
        if (!http_get(req_url.str(), &resp)) {
            portable_sleep_ms(3000);
            continue;
        }

        std::string status = json_get_string(resp, "status");
        if (status == "no_work") {
            portable_sleep_ms(2000);
            continue;
        }
        if (status == "solved") {
            break;
        }
        std::string server_err = json_get_string(resp, "error");
        if (!server_err.empty()) {
            portable_sleep_ms(3000);
            continue;
        }

        std::string str_block = json_get_string(resp, "block");
        std::string str_range_idx = json_get_string(resp, "range_idx");
        std::string str_start = json_get_string(resp, "start");
        std::string str_end = json_get_string(resp, "end");
        std::string str_target = json_get_string(resp, "target_address");
        if (str_target.empty()) {
            str_target = json_get_string(resp, "target");
        }
        std::string str_range_count = json_get_string(resp, "range_count");
        if (str_range_count.empty()) str_range_count = json_get_string(resp, "multiple");

        std::string str_server_user = json_get_string(resp, "user");
        if (!str_server_user.empty() && (current_user == "guest" || current_user.rfind("user-", 0) == 0)) {
            current_user = str_server_user;
        }

        if (str_start.empty() || str_end.empty() || str_target.empty()) {
            portable_sleep_ms(3000);
            continue;
        }

        uint64_t block_idx = (uint64_t)std::strtoull(str_block.c_str(), NULL, 10);
        uint64_t range_idx = str_range_idx.empty() ? 0 : (uint64_t)std::strtoull(str_range_idx.c_str(), NULL, 10);
        int range_count = str_range_count.empty() ? requested_multiple : std::atoi(str_range_count.c_str());
        if (range_count <= 0) range_count = 1;

        u256 start_k = parse_u256(str_start);
        u256 end_k = parse_u256(str_end);
        u256 total_keys = end_k - start_k;
        uint64_t total_keys_count = (uint64_t)total_keys.low;

        uint8_t target_h160[20];
        if (!b58check_decode_hash160(str_target, target_h160)) {
            portable_sleep_ms(3000);
            continue;
        }

        uint32_t target_w[5];
        for (int i = 0; i < 5; ++i) {
            target_w[i] = (uint32_t)target_h160[i * 4] |
                          ((uint32_t)target_h160[i * 4 + 1] << 8) |
                          ((uint32_t)target_h160[i * 4 + 2] << 16) |
                          ((uint32_t)target_h160[i * 4 + 3] << 24);
        }
        uint64_t target_h64 = (uint64_t)target_w[0] | ((uint64_t)target_w[1] << 32);

        // Upload Target Parameters to GPU Constant Memory
        cudaMemcpyToSymbol(dev_target_w, target_w, sizeof(target_w));
        cudaMemcpyToSymbol(dev_target_h64, &target_h64, sizeof(uint64_t));

        int zero = 0;
        cudaMemcpyToSymbol(dev_found_flag, &zero, sizeof(int));

        bool hit = false;
        u256 found_key = 0;
        uint64_t actual_checked = 0;

        auto t_start = std::chrono::high_resolution_clock::now();

        while (actual_checked < total_keys_count && g_running.load() && !hit) {
            int zero_flag = 0;
            cudaMemcpyToSymbol(dev_found_flag, &zero_flag, sizeof(int));

            uint64_t cur_chunk = host_min(chunk_size, total_keys_count - actual_checked);
            uint32_t cur_steps = (uint32_t)((cur_chunk + grid_threads - 1) / grid_threads);
            uint32_t cur_batches = (cur_steps + 15) / 16;
            u256 cur_start = start_k + actual_checked;

#if defined(__CUDACC__) || defined(__NVCC__)
            cuda_scan_kernel<<<numBlocks, threadsPerBlock>>>(
                cur_start, cur_chunk, grid_threads, cur_batches
            );
#else
            (void)numBlocks; (void)threadsPerBlock;
            cuda_scan_kernel(cur_start, cur_chunk, grid_threads, cur_batches);
#endif

            cudaDeviceSynchronize();

            int h_found = 0;
            cudaMemcpyFromSymbol(&h_found, dev_found_flag, sizeof(int));
            if (h_found != 0) {
                uint64_t h_offset = 0;
                cudaMemcpyFromSymbol(&h_offset, dev_found_offset, sizeof(uint64_t));
                hit = true;
                found_key = cur_start + h_offset;
                actual_checked += h_offset + 1;
                break;
            }
            actual_checked += cur_chunk;
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t_end - t_start).count();
        if (elapsed <= 0.0) elapsed = 0.001;
        double speed = (double)actual_checked / elapsed;

        if (hit) {
            // Private key found: reported to server via HTTP POST result
        }

        if (!hit && !g_running.load()) break;

        std::stringstream json;
        json << "{\"action\":\"result\""
             << ",\"puzzle\":" << current_puzzle
             << ",\"block\":" << block_idx
             << ",\"range_idx\":" << range_idx
             << ",\"range_count\":" << range_count
             << ",\"multiple\":" << range_count
             << ",\"status\":\"" << (hit ? "found" : "done") << "\""
             << ",\"private_key\":\"" << (hit ? ("0x" + u256_to_hex64(found_key)) : "") << "\""
             << ",\"user\":\"" << current_user << "\""
             << ",\"speed\":" << std::fixed << std::setprecision(1) << speed
             << ",\"keys\":\"" << u256_to_dec(total_keys) << "\""
             << ",\"range_size\":\"" << u256_to_dec(total_keys) << "\""
             << ",\"count\":" << actual_checked
             << ",\"elapsed\":" << std::fixed << std::setprecision(2) << elapsed << "}";

        std::string post_url = api_base + "?action=result&puzzle=" + std::to_string(current_puzzle) + "&user=" + current_user;
        std::string ack;
        http_post(post_url, json.str(), &ack);

        if (hit) break;
    }

    curl_global_cleanup();
    return 0;
}
