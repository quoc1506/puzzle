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

#ifndef CURL_STATICLIB
#define CURL_STATICLIB
#endif

#include <curl/curl.h>
#include <openssl/sha.h>
#include <openssl/ripemd.h>
#include <openssl/bn.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#if defined(__GNUC__) || defined(__clang__)
#include <x86intrin.h>
#endif
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_DEV __device__
#define CUDA_GLOBAL __global__
#define CUDA_INLINE __forceinline__
#else
#define CUDA_HOSTDEV
#define CUDA_DEV
#define CUDA_GLOBAL
#define CUDA_INLINE inline
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

template <typename T>
CUDA_HOSTDEV CUDA_INLINE void host_reverse(T* first, T* last) {
    while ((first != last) && (first != --last)) {
        T temp = *first;
        *first = *last;
        *last = temp;
        ++first;
    }
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

struct Fe {
    uint64_t d[4];
};

CUDA_HOSTDEV CUDA_INLINE bool fe_is_zero(const Fe& a) {
    return (a.d[0] | a.d[1] | a.d[2] | a.d[3]) == 0;
}

CUDA_HOSTDEV CUDA_INLINE Fe fe_add(const Fe& a, const Fe& b) {
    Fe r;
#if defined(__CUDA_ARCH__)
    unsigned long long c = 0;
    asm volatile(
        "add.cc.u64 %0, %4, %8;\n\t"
        "addc.cc.u64 %1, %5, %9;\n\t"
        "addc.cc.u64 %2, %6, %10;\n\t"
        "addc.cc.u64 %3, %7, %11;\n\t"
        "addc.u64 %12, 0, 0;\n\t"
        : "=l"(r.d[0]), "=l"(r.d[1]), "=l"(r.d[2]), "=l"(r.d[3]), "=l"(c)
        : "l"(a.d[0]), "l"(a.d[1]), "l"(a.d[2]), "l"(a.d[3]),
          "l"(b.d[0]), "l"(b.d[1]), "l"(b.d[2]), "l"(b.d[3])
    );
    unsigned long long t0, t1, t2, t3;
    unsigned long long borrow;
    asm volatile(
        "sub.cc.u64 %0, %4, 0xFFFFEEFFFFFC2FULL;\n\t"
        "subc.cc.u64 %1, %5, 0xFFFFFFFFFFFFFFFFULL;\n\t"
        "subc.cc.u64 %2, %6, 0xFFFFFFFFFFFFFFFFULL;\n\t"
        "subc.cc.u64 %3, %7, 0xFFFFFFFFFFFFFFFFULL;\n\t"
        "subc.u64 %8, %9, 0;\n\t"
        : "=l"(t0), "=l"(t1), "=l"(t2), "=l"(t3), "=l"(borrow)
        : "l"(r.d[0]), "l"(r.d[1]), "l"(r.d[2]), "l"(r.d[3]), "l"(c)
    );
    if (borrow == 0) {
        r.d[0] = t0; r.d[1] = t1; r.d[2] = t2; r.d[3] = t3;
    }
#elif defined(__x86_64__) || defined(_M_X64)
    unsigned char c = 0;
    c = _addcarry_u64(c, a.d[0], b.d[0], (unsigned long long*)&r.d[0]);
    c = _addcarry_u64(c, a.d[1], b.d[1], (unsigned long long*)&r.d[1]);
    c = _addcarry_u64(c, a.d[2], b.d[2], (unsigned long long*)&r.d[2]);
    c = _addcarry_u64(c, a.d[3], b.d[3], (unsigned long long*)&r.d[3]);
    uint64_t t0, t1, t2, t3;
    unsigned char b_borrow = 0;
    b_borrow = _subborrow_u64(b_borrow, r.d[0], 0xFFFFEEFFFFFC2FULL, (unsigned long long*)&t0);
    b_borrow = _subborrow_u64(b_borrow, r.d[1], 0xFFFFFFFFFFFFFFFFULL, (unsigned long long*)&t1);
    b_borrow = _subborrow_u64(b_borrow, r.d[2], 0xFFFFFFFFFFFFFFFFULL, (unsigned long long*)&t2);
    b_borrow = _subborrow_u64(b_borrow, r.d[3], 0xFFFFFFFFFFFFFFFFULL, (unsigned long long*)&t3);
    b_borrow = _subborrow_u64(b_borrow, c, 0, (unsigned long long*)&c);
    if (!b_borrow) {
        r.d[0] = t0; r.d[1] = t1; r.d[2] = t2; r.d[3] = t3;
    }
#else
    u128 c = 0;
    for (int i = 0; i < 4; ++i) {
        c += (u128)a.d[i] + b.d[i];
        r.d[i] = (uint64_t)c;
        c >>= 64;
    }
    const uint64_t P[4] = { 0xFFFFEEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
    u128 borrow = 0;
    uint64_t t[4];
    for (int i = 0; i < 4; ++i) {
        u128 diff = (u128)r.d[i] - P[i] - borrow;
        t[i] = (uint64_t)diff;
        borrow = (diff >> 127) & 1;
    }
    if (c || !borrow) {
        for (int i = 0; i < 4; ++i) r.d[i] = t[i];
    }
#endif
    return r;
}

CUDA_HOSTDEV CUDA_INLINE Fe fe_sub(const Fe& a, const Fe& b) {
    Fe r;
#if defined(__CUDA_ARCH__)
    unsigned long long borrow;
    asm volatile(
        "sub.cc.u64 %0, %4, %8;\n\t"
        "subc.cc.u64 %1, %5, %9;\n\t"
        "subc.cc.u64 %2, %6, %10;\n\t"
        "subc.cc.u64 %3, %7, %11;\n\t"
        "subc.u64 %12, 0, 0;\n\t"
        : "=l"(r.d[0]), "=l"(r.d[1]), "=l"(r.d[2]), "=l"(r.d[3]), "=l"(borrow)
        : "l"(a.d[0]), "l"(a.d[1]), "l"(a.d[2]), "l"(a.d[3]),
          "l"(b.d[0]), "l"(b.d[1]), "l"(b.d[2]), "l"(b.d[3])
    );
    if (borrow != 0) {
        asm volatile(
            "add.cc.u64 %0, %0, 0xFFFFEEFFFFFC2FULL;\n\t"
            "addc.cc.u64 %1, %1, 0xFFFFFFFFFFFFFFFFULL;\n\t"
            "addc.cc.u64 %2, %2, 0xFFFFFFFFFFFFFFFFULL;\n\t"
            "addc.u64 %3, %3, 0xFFFFFFFFFFFFFFFFULL;\n\t"
            : "+l"(r.d[0]), "+l"(r.d[1]), "+l"(r.d[2]), "+l"(r.d[3])
        );
    }
#elif defined(__x86_64__) || defined(_M_X64)
    unsigned char borrow = 0;
    borrow = _subborrow_u64(borrow, a.d[0], b.d[0], (unsigned long long*)&r.d[0]);
    borrow = _subborrow_u64(borrow, a.d[1], b.d[1], (unsigned long long*)&r.d[1]);
    borrow = _subborrow_u64(borrow, a.d[2], b.d[2], (unsigned long long*)&r.d[2]);
    borrow = _subborrow_u64(borrow, a.d[3], b.d[3], (unsigned long long*)&r.d[3]);
    if (borrow) {
        unsigned char c = 0;
        c = _addcarry_u64(c, r.d[0], 0xFFFFEEFFFFFC2FULL, (unsigned long long*)&r.d[0]);
        c = _addcarry_u64(c, r.d[1], 0xFFFFFFFFFFFFFFFFULL, (unsigned long long*)&r.d[1]);
        c = _addcarry_u64(c, r.d[2], 0xFFFFFFFFFFFFFFFFULL, (unsigned long long*)&r.d[2]);
        c = _addcarry_u64(c, r.d[3], 0xFFFFFFFFFFFFFFFFULL, (unsigned long long*)&r.d[3]);
    }
#else
    u128 borrow = 0;
    for (int i = 0; i < 4; ++i) {
        u128 diff = (u128)a.d[i] - b.d[i] - borrow;
        r.d[i] = (uint64_t)diff;
        borrow = (diff >> 127) & 1;
    }
    if (borrow) {
        const uint64_t P[4] = { 0xFFFFEEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
        u128 c = 0;
        for (int i = 0; i < 4; ++i) {
            c += (u128)r.d[i] + P[i];
            r.d[i] = (uint64_t)c;
            c >>= 64;
        }
    }
#endif
    return r;
}

#if defined(__CUDA_ARCH__)
CUDA_DEV CUDA_INLINE Fe fe_reduce_cuda(uint64_t t[8]) {
    uint64_t c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0;
    for (int i = 0; i < 4; ++i) {
        uint64_t hi = t[4 + i];
        if (hi == 0) continue;
        uint64_t m_lo = hi * 0x1000003D1ULL;
        uint64_t m_hi = __umul64hi(hi, 0x1000003D1ULL);
        unsigned long long carry = 0;
        if (i == 0) {
            asm volatile(
                "add.cc.u64 %0, %0, %5;\n\t"
                "addc.cc.u64 %1, %1, %6;\n\t"
                "addc.cc.u64 %2, %2, 0;\n\t"
                "addc.cc.u64 %3, %3, 0;\n\t"
                "addc.u64 %4, %4, 0;\n\t"
                : "+l"(c0), "+l"(c1), "+l"(c2), "+l"(c3), "+l"(c4)
                : "l"(m_lo), "l"(m_hi)
            );
        } else if (i == 1) {
            asm volatile(
                "add.cc.u64 %1, %1, %5;\n\t"
                "addc.cc.u64 %2, %2, %6;\n\t"
                "addc.cc.u64 %3, %3, 0;\n\t"
                "addc.u64 %4, %4, 0;\n\t"
                : "+l"(c0), "+l"(c1), "+l"(c2), "+l"(c3), "+l"(c4)
                : "l"(m_lo), "l"(m_hi)
            );
        } else if (i == 2) {
            asm volatile(
                "add.cc.u64 %2, %2, %5;\n\t"
                "addc.cc.u64 %3, %3, %6;\n\t"
                "addc.u64 %4, %4, 0;\n\t"
                : "+l"(c0), "+l"(c1), "+l"(c2), "+l"(c3), "+l"(c4)
                : "l"(m_lo), "l"(m_hi)
            );
        } else {
            asm volatile(
                "add.cc.u64 %3, %3, %5;\n\t"
                "addc.u64 %4, %4, %6;\n\t"
                : "+l"(c0), "+l"(c1), "+l"(c2), "+l"(c3), "+l"(c4)
                : "l"(m_lo), "l"(m_hi)
            );
        }
    }
    Fe r;
    unsigned long long c_out = 0;
    asm volatile(
        "add.cc.u64 %0, %5, %10;\n\t"
        "addc.cc.u64 %1, %6, %11;\n\t"
        "addc.cc.u64 %2, %7, %12;\n\t"
        "addc.cc.u64 %3, %8, %13;\n\t"
        "addc.u64 %4, %9, 0;\n\t"
        : "=l"(r.d[0]), "=l"(r.d[1]), "=l"(r.d[2]), "=l"(r.d[3]), "=l"(c_out)
        : "l"(t[0]), "l"(t[1]), "l"(t[2]), "l"(t[3]), "l"(c4),
          "l"(c0), "l"(c1), "l"(c2), "l"(c3)
    );
    while (c_out != 0) {
        uint64_t m_lo = c_out * 0x1000003D1ULL;
        uint64_t m_hi = __umul64hi(c_out, 0x1000003D1ULL);
        c_out = 0;
        asm volatile(
            "add.cc.u64 %0, %0, %4;\n\t"
            "addc.cc.u64 %1, %1, %5;\n\t"
            "addc.cc.u64 %2, %2, 0;\n\t"
            "addc.cc.u64 %3, %3, 0;\n\t"
            "addc.u64 %4, 0, 0;\n\t"
            : "+l"(r.d[0]), "+l"(r.d[1]), "+l"(r.d[2]), "+l"(r.d[3]), "+l"(c_out)
            : "l"(m_lo), "l"(m_hi)
        );
    }
    unsigned long long t0, t1, t2, t3;
    unsigned long long borrow;
    asm volatile(
        "sub.cc.u64 %0, %4, 0xFFFFEEFFFFFC2FULL;\n\t"
        "subc.cc.u64 %1, %5, 0xFFFFFFFFFFFFFFFFULL;\n\t"
        "subc.cc.u64 %2, %6, 0xFFFFFFFFFFFFFFFFULL;\n\t"
        "subc.cc.u64 %3, %7, 0xFFFFFFFFFFFFFFFFULL;\n\t"
        "subc.u64 %8, 0, 0;\n\t"
        : "=l"(t0), "=l"(t1), "=l"(t2), "=l"(t3), "=l"(borrow)
        : "l"(r.d[0]), "l"(r.d[1]), "l"(r.d[2]), "l"(r.d[3])
    );
    if (borrow == 0) {
        r.d[0] = t0; r.d[1] = t1; r.d[2] = t2; r.d[3] = t3;
    }
    return r;
}
#endif

CUDA_HOSTDEV CUDA_INLINE Fe fe_mul(const Fe& a, const Fe& b) {
#if defined(__CUDA_ARCH__)
    uint64_t t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int i = 0; i < 4; ++i) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; ++j) {
            uint64_t prod_lo = a.d[i] * b.d[j];
            uint64_t prod_hi = __umul64hi(a.d[i], b.d[j]);
            unsigned long long c1 = 0, c2 = 0;
            asm volatile(
                "add.cc.u64 %0, %0, %3;\n\t"
                "addc.u64 %1, 0, 0;\n\t"
                : "+l"(t[i + j]), "=l"(c1)
                : "l"(prod_lo)
            );
            asm volatile(
                "add.cc.u64 %0, %0, %3;\n\t"
                "addc.u64 %1, 0, 0;\n\t"
                : "+l"(t[i + j]), "=l"(c2)
                : "l"(carry)
            );
            carry = prod_hi + c1 + c2;
        }
        t[i + 4] = carry;
    }
    return fe_reduce_cuda(t);
#elif (defined(__x86_64__) || defined(_M_X64)) && defined(__BMI2__) && defined(__ADX__)
    uint64_t r0, r1, r2, r3, r4, r5, r6, r7;
    uint64_t zero = 0;
    asm volatile(
        "xorq %4, %4\n\t"
        "movq 0(%8), %%rdx\n\t"
        "mulxq 0(%9), %0, %1\n\t"
        "mulxq 8(%9), %%rax, %2\n\t"
        "addq %%rax, %1\n\t"
        "mulxq 16(%9), %%rax, %3\n\t"
        "adcq %%rax, %2\n\t"
        "mulxq 24(%9), %%rax, %4\n\t"
        "adcq %%rax, %3\n\t"
        "adcq $0, %4\n\t"

        "movq 8(%8), %%rdx\n\t"
        "mulxq 0(%9), %%rax, %%rbx\n\t"
        "addq %%rax, %1\n\t"
        "adcq %%rbx, %2\n\t"
        "mulxq 8(%9), %%rax, %%rbx\n\t"
        "adcq %%rax, %2\n\t"
        "adcq %%rbx, %3\n\t"
        "mulxq 16(%9), %%rax, %%rbx\n\t"
        "adcq %%rax, %3\n\t"
        "adcq %%rbx, %4\n\t"
        "mulxq 24(%9), %%rax, %5\n\t"
        "adcq %%rax, %4\n\t"
        "adcq $0, %5\n\t"

        "movq 16(%8), %%rdx\n\t"
        "mulxq 0(%9), %%rax, %%rbx\n\t"
        "addq %%rax, %2\n\t"
        "adcq %%rbx, %3\n\t"
        "mulxq 8(%9), %%rax, %%rbx\n\t"
        "adcq %%rax, %3\n\t"
        "adcq %%rbx, %4\n\t"
        "mulxq 16(%9), %%rax, %%rbx\n\t"
        "adcq %%rax, %4\n\t"
        "adcq %%rbx, %5\n\t"
        "mulxq 24(%9), %%rax, %6\n\t"
        "adcq %%rax, %5\n\t"
        "adcq $0, %6\n\t"

        "movq 24(%8), %%rdx\n\t"
        "mulxq 0(%9), %%rax, %%rbx\n\t"
        "addq %%rax, %3\n\t"
        "adcq %%rbx, %4\n\t"
        "mulxq 8(%9), %%rax, %%rbx\n\t"
        "adcq %%rax, %4\n\t"
        "adcq %%rbx, %5\n\t"
        "mulxq 16(%9), %%rax, %%rbx\n\t"
        "adcq %%rax, %5\n\t"
        "adcq %%rbx, %6\n\t"
        "mulxq 24(%9), %%rax, %7\n\t"
        "adcq %%rax, %6\n\t"
        "adcq $0, %7\n\t"
        : "=&r"(r0), "=&r"(r1), "=&r"(r2), "=&r"(r3),
          "=&r"(r4), "=&r"(r5), "=&r"(r6), "=&r"(r7)
        : "r"(a.d), "r"(b.d)
        : "%rax", "%rbx", "%rdx", "cc", "memory"
    );

    uint64_t t[8] = { r0, r1, r2, r3, r4, r5, r6, r7 };
    u128 c = 0;
    const uint64_t K = 0x1000003D1ULL;
    for (int i = 0; i < 4; ++i) {
        u128 hi = t[4 + i];
        u128 prod = hi * K;
        c += (u128)t[i] + (uint64_t)prod;
        t[i] = (uint64_t)c;
        c >>= 64;
        c += (u128)t[i + 1] + (prod >> 64);
        t[i + 1] = (uint64_t)c;
        c >>= 64;
    }
    for (int i = 4; i < 7; ++i) {
        c += t[i];
        t[i] = (uint64_t)c;
        c >>= 64;
    }
    uint64_t carry_top = t[7] + (uint64_t)c;
    u128 c2 = (u128)carry_top * K;
    for (int i = 0; i < 4; ++i) {
        c2 += t[i];
        t[i] = (uint64_t)c2;
        c2 >>= 64;
    }
    while (c2 != 0) {
        u128 extra = c2 * K;
        c2 = 0;
        for (int i = 0; i < 4; ++i) {
            extra += t[i];
            t[i] = (uint64_t)extra;
            extra >>= 64;
        }
        c2 = extra;
    }
    Fe r;
    for (int i = 0; i < 4; ++i) r.d[i] = t[i];
    const uint64_t P[4] = { 0xFFFFEEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
    unsigned char borrow = 0;
    uint64_t sub_t[4];
    borrow = _subborrow_u64(borrow, r.d[0], P[0], (unsigned long long*)&sub_t[0]);
    borrow = _subborrow_u64(borrow, r.d[1], P[1], (unsigned long long*)&sub_t[1]);
    borrow = _subborrow_u64(borrow, r.d[2], P[2], (unsigned long long*)&sub_t[2]);
    borrow = _subborrow_u64(borrow, r.d[3], P[3], (unsigned long long*)&sub_t[3]);
    if (!borrow) {
        for (int i = 0; i < 4; ++i) r.d[i] = sub_t[i];
    }
    return r;
#else
    u128 t[8] = {0};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            t[i + j] += (u128)a.d[i] * b.d[j];
        }
    }
    u128 carry = 0;
    uint64_t t64[8];
    for (int i = 0; i < 8; ++i) {
        carry += t[i];
        t64[i] = (uint64_t)carry;
        carry >>= 64;
    }
    const uint64_t K = 0x1000003D1ULL;
    u128 c = 0;
    for (int i = 0; i < 4; ++i) {
        u128 hi = t64[4 + i];
        u128 prod = hi * K;
        c += (u128)t64[i] + (uint64_t)prod;
        t64[i] = (uint64_t)c;
        c >>= 64;
        c += (u128)t64[i + 1] + (prod >> 64);
        t64[i + 1] = (uint64_t)c;
        c >>= 64;
    }
    for (int i = 4; i < 7; ++i) {
        c += t64[i];
        t64[i] = (uint64_t)c;
        c >>= 64;
    }
    uint64_t carry_top = t64[7] + (uint64_t)c;
    u128 c2 = (u128)carry_top * K;
    for (int i = 0; i < 4; ++i) {
        c2 += t64[i];
        t64[i] = (uint64_t)c2;
        c2 >>= 64;
    }
    while (c2 != 0) {
        u128 extra = c2 * K;
        c2 = 0;
        for (int i = 0; i < 4; ++i) {
            extra += t64[i];
            t64[i] = (uint64_t)extra;
            extra >>= 64;
        }
        c2 = extra;
    }
    Fe r;
    for (int i = 0; i < 4; ++i) r.d[i] = t64[i];
    const uint64_t P[4] = { 0xFFFFEEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
    u128 b_borrow = 0;
    uint64_t sub_t[4];
    for (int i = 0; i < 4; ++i) {
        u128 diff = (u128)r.d[i] - P[i] - b_borrow;
        sub_t[i] = (uint64_t)diff;
        b_borrow = (diff >> 127) & 1;
    }
    if (!b_borrow) {
        for (int i = 0; i < 4; ++i) r.d[i] = sub_t[i];
    }
    return r;
#endif
}

CUDA_HOSTDEV CUDA_INLINE Fe fe_sqr(const Fe& a) {
    return fe_mul(a, a);
}

CUDA_HOSTDEV CUDA_INLINE Fe fe_inv(const Fe& a) {
    Fe x2 = fe_mul(fe_sqr(a), a);
    Fe x3 = fe_mul(fe_sqr(x2), a);
    Fe x6 = x3;
    for (int i = 0; i < 3; ++i) x6 = fe_sqr(x6);
    x6 = fe_mul(x6, x3);

    Fe x12 = x6;
    for (int i = 0; i < 6; ++i) x12 = fe_sqr(x12);
    x12 = fe_mul(x12, x6);

    Fe x24 = x12;
    for (int i = 0; i < 12; ++i) x24 = fe_sqr(x24);
    x24 = fe_mul(x24, x12);

    Fe x30 = x24;
    for (int i = 0; i < 6; ++i) x30 = fe_sqr(x30);
    x30 = fe_mul(x30, x6);

    Fe x31 = fe_mul(fe_sqr(x30), a);

    Fe t = x31;
    for (int i = 0; i < 31; ++i) t = fe_sqr(t);
    t = fe_mul(t, x31);

    for (int i = 0; i < 62; ++i) t = fe_sqr(t);
    t = fe_mul(t, x31);

    for (int i = 0; i < 31; ++i) t = fe_sqr(t);
    t = fe_mul(t, x31);

    for (int i = 0; i < 62; ++i) t = fe_sqr(t);
    t = fe_mul(t, x31);

    for (int i = 0; i < 32; ++i) t = fe_sqr(t);
    t = fe_mul(t, x31);

    for (int i = 0; i < 5; ++i) t = fe_sqr(t);
    t = fe_mul(t, x30);

    t = fe_sqr(t);
    t = fe_sqr(t);
    t = fe_mul(t, a);

    t = fe_sqr(t);
    t = fe_mul(t, a);

    t = fe_sqr(t);
    return t;
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

#ifdef __CUDACC__
__constant__ AffinePoint dev_G_table[16];
__device__ int dev_found_flag = 0;
__device__ uint64_t dev_found_offset = 0;
__constant__ uint32_t dev_target_w[5];
__constant__ uint64_t dev_target_h64;

CUDA_DEV CUDA_INLINE uint32_t ror32_gpu(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

CUDA_DEV CUDA_INLINE uint32_t rol32_gpu(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

CUDA_DEV CUDA_INLINE uint32_t bswap32_dev(uint32_t x) {
    return __byte_perm(x, 0, 0x0123);
}

__constant__ uint32_t K_SHA256_GPU[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

CUDA_DEV CUDA_INLINE void fast_sha256_into_ripemd_X(uint8_t prefix, const Fe& x, uint32_t X[16]) {
    uint32_t w[64];
    w[0] = ((uint32_t)prefix << 24) | ((uint32_t)(x.d[3] >> 56) << 16) | (((uint32_t)(x.d[3] >> 48) & 0xFF) << 8) | (((uint32_t)(x.d[3] >> 40) & 0xFF));
    w[1] = (((uint32_t)(x.d[3] >> 32) & 0xFF) << 24) | (((uint32_t)(x.d[3] >> 24) & 0xFF) << 16) | (((uint32_t)(x.d[3] >> 16) & 0xFF) << 8) | (((uint32_t)(x.d[3] >> 8) & 0xFF));
    w[2] = (((uint32_t)x.d[3] & 0xFF) << 24) | (((uint32_t)(x.d[2] >> 56) & 0xFF) << 16) | (((uint32_t)(x.d[2] >> 48) & 0xFF) << 8) | (((uint32_t)(x.d[2] >> 40) & 0xFF));
    w[3] = (((uint32_t)(x.d[2] >> 32) & 0xFF) << 24) | (((uint32_t)(x.d[2] >> 24) & 0xFF) << 16) | (((uint32_t)(x.d[2] >> 16) & 0xFF) << 8) | (((uint32_t)(x.d[2] >> 8) & 0xFF));
    w[4] = (((uint32_t)x.d[2] & 0xFF) << 24) | (((uint32_t)(x.d[1] >> 56) & 0xFF) << 16) | (((uint32_t)(x.d[1] >> 48) & 0xFF) << 8) | (((uint32_t)(x.d[1] >> 40) & 0xFF));
    w[5] = (((uint32_t)(x.d[1] >> 32) & 0xFF) << 24) | (((uint32_t)(x.d[1] >> 24) & 0xFF) << 16) | (((uint32_t)(x.d[1] >> 16) & 0xFF) << 8) | (((uint32_t)(x.d[1] >> 8) & 0xFF));
    w[6] = (((uint32_t)x.d[1] & 0xFF) << 24) | (((uint32_t)(x.d[0] >> 56) & 0xFF) << 16) | (((uint32_t)(x.d[0] >> 48) & 0xFF) << 8) | (((uint32_t)(x.d[0] >> 40) & 0xFF));
    w[7] = (((uint32_t)(x.d[0] >> 32) & 0xFF) << 24) | (((uint32_t)(x.d[0] >> 24) & 0xFF) << 16) | (((uint32_t)(x.d[0] >> 16) & 0xFF) << 8) | (((uint32_t)(x.d[0] >> 8) & 0xFF));
    w[8] = (((uint32_t)x.d[0] & 0xFF) << 24) | 0x00800000;
    w[9] = 0; w[10] = 0; w[11] = 0; w[12] = 0; w[13] = 0; w[14] = 0;
    w[15] = 264;

    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = ror32_gpu(w[i - 15], 7) ^ ror32_gpu(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = ror32_gpu(w[i - 2], 17) ^ ror32_gpu(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = 0x6a09e667, b = 0xbb67ae85, c = 0x3c6ef372, d = 0xa54ff53a;
    uint32_t e = 0x510e527f, f = 0x9b05688c, g = 0x1f83d9ab, h = 0x5be0cd19;

    #pragma unroll 64
    for (int i = 0; i < 64; ++i) {
        uint32_t S1 = ror32_gpu(e, 6) ^ ror32_gpu(e, 11) ^ ror32_gpu(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = h + S1 + ch + K_SHA256_GPU[i] + w[i];
        uint32_t S0 = ror32_gpu(a, 2) ^ ror32_gpu(a, 13) ^ ror32_gpu(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    X[0] = bswap32_dev(a + 0x6a09e667);
    X[1] = bswap32_dev(b + 0xbb67ae85);
    X[2] = bswap32_dev(c + 0x3c6ef372);
    X[3] = bswap32_dev(d + 0xa54ff53a);
    X[4] = bswap32_dev(e + 0x510e527f);
    X[5] = bswap32_dev(f + 0x9b05688c);
    X[6] = bswap32_dev(g + 0x1f83d9ab);
    X[7] = bswap32_dev(h + 0x5be0cd19);
    X[8] = 0x00000080;
    X[9] = 0; X[10] = 0; X[11] = 0; X[12] = 0; X[13] = 0;
    X[14] = 256;
    X[15] = 0;
}

CUDA_DEV CUDA_INLINE void fast_ripemd160_32(const uint32_t X[16], uint32_t out[5]) {
    uint32_t a = 0x67452301, b = 0xefcdab89, c = 0x98badcfe, d = 0x10325476, e = 0xc3d2e1f0;
    uint32_t aa = a, bb = b, cc = c, dd = d, ee = e;

    #define R1(A,B,C,D,E,x,s) { A += (B ^ C ^ D) + x; A = rol32_gpu(A, s) + E; C = rol32_gpu(C, 10); }
    R1(a, b, c, d, e, X[0], 11);  R1(e, a, b, c, d, X[1], 14);
    R1(d, e, a, b, c, X[2], 15);  R1(c, d, e, a, b, X[3], 12);
    R1(b, c, d, e, a, X[4], 5);   R1(a, b, c, d, e, X[5], 8);
    R1(e, a, b, c, d, X[6], 7);   R1(d, e, a, b, c, X[7], 9);
    R1(c, d, e, a, b, X[8], 11);  R1(b, c, d, e, a, X[9], 13);
    R1(a, b, c, d, e, X[10], 14); R1(e, a, b, c, d, X[11], 15);
    R1(d, e, a, b, c, X[12], 6);  R1(c, d, e, a, b, X[13], 7);
    R1(b, c, d, e, a, X[14], 9);  R1(a, b, c, d, e, X[15], 8);

    #define R2(A,B,C,D,E,x,s) { A += ((B & C) | (~B & D)) + x + 0x5a827999; A = rol32_gpu(A, s) + E; C = rol32_gpu(C, 10); }
    R2(e, a, b, c, d, X[7], 7);   R2(d, e, a, b, c, X[4], 6);
    R2(c, d, e, a, b, X[13], 8);  R2(b, c, d, e, a, X[1], 13);
    R2(a, b, c, d, e, X[10], 11); R2(e, a, b, c, d, X[6], 9);
    R2(d, e, a, b, c, X[15], 7);  R2(c, d, e, a, b, X[3], 15);
    R2(b, c, d, e, a, X[12], 7);  R2(a, b, c, d, e, X[0], 12);
    R2(e, a, b, c, d, X[9], 15);  R2(d, e, a, b, c, X[5], 9);
    R2(c, d, e, a, b, X[2], 11);  R2(b, c, d, e, a, X[14], 7);
    R2(a, b, c, d, e, X[11], 13); R2(e, a, b, c, d, X[8], 12);

    #define R3(A,B,C,D,E,x,s) { A += ((B | ~C) ^ D) + x + 0x6ed9eba1; A = rol32_gpu(A, s) + E; C = rol32_gpu(C, 10); }
    R3(d, e, a, b, c, X[3], 11);  R3(c, d, e, a, b, X[10], 13);
    R3(b, c, d, e, a, X[14], 6);  R3(a, b, c, d, e, X[4], 7);
    R3(e, a, b, c, d, X[9], 14);  R3(d, e, a, b, c, X[15], 9);
    R3(c, d, e, a, b, X[8], 13);  R3(b, c, d, e, a, X[1], 15);
    R3(a, b, c, d, e, X[2], 14);  R3(e, a, b, c, d, X[7], 8);
    R3(d, e, a, b, c, X[0], 13);  R3(c, d, e, a, b, X[6], 6);
    R3(b, c, d, e, a, X[13], 5);  R3(a, b, c, d, e, X[11], 12);
    R3(e, a, b, c, d, X[5], 7);   R3(d, e, a, b, c, X[12], 5);

    #define R4(A,B,C,D,E,x,s) { A += ((B & D) | (C & ~D)) + x + 0x8f1bbcdc; A = rol32_gpu(A, s) + E; C = rol32_gpu(C, 10); }
    R4(c, d, e, a, b, X[1], 11);  R4(b, c, d, e, a, X[9], 12);
    R4(a, b, c, d, e, X[11], 14); R4(e, a, b, c, d, X[10], 15);
    R4(d, e, a, b, c, X[0], 14);  R4(c, d, e, a, b, X[8], 15);
    R4(b, c, d, e, a, X[12], 9);  R4(a, b, c, d, e, X[4], 8);
    R4(e, a, b, c, d, X[13], 9);  R4(d, e, a, b, c, X[3], 14);
    R4(c, d, e, a, b, X[7], 5);   R4(b, c, d, e, a, X[15], 6);
    R4(a, b, c, d, e, X[14], 8);  R4(e, a, b, c, d, X[5], 6);
    R4(d, e, a, b, c, X[6], 5);   R4(c, d, e, a, b, X[2], 12);

    #define R5(A,B,C,D,E,x,s) { A += (B ^ (C | ~D)) + x + 0xa953fd4e; A = rol32_gpu(A, s) + E; C = rol32_gpu(C, 10); }
    R5(b, c, d, e, a, X[4], 9);   R5(a, b, c, d, e, X[0], 15);
    R5(e, a, b, c, d, X[5], 5);   R5(d, e, a, b, c, X[9], 11);
    R5(c, d, e, a, b, X[7], 6);   R5(b, c, d, e, a, X[12], 8);
    R5(a, b, c, d, e, X[2], 13);  R5(e, a, b, c, d, X[10], 12);
    R5(d, e, a, b, c, X[14], 5);  R5(c, d, e, a, b, X[1], 12);
    R5(b, c, d, e, a, X[3], 13);  R5(a, b, c, d, e, X[8], 14);
    R5(e, a, b, c, d, X[11], 11); R5(d, e, a, b, c, X[6], 8);
    R5(c, d, e, a, b, X[15], 5);  R5(b, c, d, e, a, X[13], 6);

    #define RR1(A,B,C,D,E,x,s) { A += (B ^ (C | ~D)) + x + 0x50a28be6; A = rol32_gpu(A, s) + E; C = rol32_gpu(C, 10); }
    RR1(aa, bb, cc, dd, ee, X[5], 8);   RR1(ee, aa, bb, cc, dd, X[14], 9);
    RR1(dd, ee, aa, bb, cc, X[7], 9);   RR1(cc, dd, ee, aa, bb, X[0], 11);
    RR1(bb, cc, dd, ee, aa, X[9], 13);  RR1(aa, bb, cc, dd, ee, X[2], 15);
    RR1(ee, aa, bb, cc, dd, X[11], 15); RR1(dd, ee, aa, bb, cc, X[4], 5);
    RR1(cc, dd, ee, aa, bb, X[13], 7);  RR1(bb, cc, dd, ee, aa, X[6], 7);
    RR1(aa, bb, cc, dd, ee, X[15], 8);  RR1(ee, aa, bb, cc, dd, X[8], 11);
    RR1(dd, ee, aa, bb, cc, X[1], 14);  RR1(cc, dd, ee, aa, bb, X[10], 14);
    RR1(bb, cc, dd, ee, aa, X[3], 12);  RR1(aa, bb, cc, dd, ee, X[12], 6);

    #define RR2(A,B,C,D,E,x,s) { A += ((B & D) | (C & ~D)) + x + 0x5c4dd124; A = rol32_gpu(A, s) + E; C = rol32_gpu(C, 10); }
    RR2(ee, aa, bb, cc, dd, X[6], 9);   RR2(dd, ee, aa, bb, cc, X[11], 13);
    RR2(cc, dd, ee, aa, bb, X[3], 15);  RR2(bb, cc, dd, ee, aa, X[7], 7);
    RR2(aa, bb, cc, dd, ee, X[0], 12);  RR2(ee, aa, bb, cc, dd, X[13], 8);
    RR2(dd, ee, aa, bb, cc, X[5], 9);   RR2(cc, dd, ee, aa, bb, X[10], 11);
    RR2(bb, cc, dd, ee, aa, X[14], 7);  RR2(aa, bb, cc, dd, ee, X[15], 7);
    RR2(ee, aa, bb, cc, dd, X[8], 12);  RR2(dd, ee, aa, bb, cc, X[12], 7);
    RR2(cc, dd, ee, aa, bb, X[4], 6);   RR2(bb, cc, dd, ee, aa, X[9], 15);
    RR2(aa, bb, cc, dd, ee, X[1], 13);  RR2(ee, aa, bb, cc, dd, X[2], 11);

    #define RR3(A,B,C,D,E,x,s) { A += ((B | ~C) ^ D) + x + 0x6d703ef3; A = rol32_gpu(A, s) + E; C = rol32_gpu(C, 10); }
    RR3(dd, ee, aa, bb, cc, X[15], 9);  RR3(cc, dd, ee, aa, bb, X[5], 7);
    RR3(bb, cc, dd, ee, aa, X[1], 15);  RR3(aa, bb, cc, dd, ee, X[3], 11);
    RR3(ee, aa, bb, cc, dd, X[7], 8);   RR3(dd, ee, aa, bb, cc, X[14], 6);
    RR3(cc, dd, ee, aa, bb, X[6], 6);   RR3(bb, cc, dd, ee, aa, X[9], 14);
    RR3(aa, bb, cc, dd, ee, X[11], 12); RR3(ee, aa, bb, cc, dd, X[8], 13);
    RR3(dd, ee, aa, bb, cc, X[12], 5);  RR3(cc, dd, ee, aa, bb, X[2], 14);
    RR3(bb, cc, dd, ee, aa, X[10], 13); RR3(aa, bb, cc, dd, ee, X[0], 13);
    RR3(ee, aa, bb, cc, dd, X[4], 7);   RR3(dd, ee, aa, bb, cc, X[13], 5);

    #define RR4(A,B,C,D,E,x,s) { A += ((B & C) | (~B & D)) + x + 0x7a6d76e9; A = rol32_gpu(A, s) + E; C = rol32_gpu(C, 10); }
    RR4(cc, dd, ee, aa, bb, X[8], 15);  RR4(bb, cc, dd, ee, aa, X[6], 5);
    RR4(aa, bb, cc, dd, ee, X[4], 8);   RR4(ee, aa, bb, cc, dd, X[1], 11);
    RR4(dd, ee, aa, bb, cc, X[3], 14);  RR4(cc, dd, ee, aa, bb, X[11], 14);
    RR4(bb, cc, dd, ee, aa, X[15], 6);  RR4(aa, bb, cc, dd, ee, X[0], 14);
    RR4(ee, aa, bb, cc, dd, X[5], 6);   RR4(dd, ee, aa, bb, cc, X[12], 9);
    RR4(cc, dd, ee, aa, bb, X[2], 12);  RR4(bb, cc, dd, ee, aa, X[13], 9);
    RR4(aa, bb, cc, dd, ee, X[9], 12);  RR4(ee, aa, bb, cc, dd, X[7], 5);
    RR4(dd, ee, aa, bb, cc, X[10], 15); RR4(cc, dd, ee, aa, bb, X[14], 8);

    #define RR5(A,B,C,D,E,x,s) { A += (B ^ C ^ D) + x; A = rol32_gpu(A, s) + E; C = rol32_gpu(C, 10); }
    RR5(bb, cc, dd, ee, aa, X[12], 8);  RR5(aa, bb, cc, dd, ee, X[15], 5);
    RR5(ee, aa, bb, cc, dd, X[10], 12); RR5(dd, ee, aa, bb, cc, X[4], 9);
    RR5(cc, dd, ee, aa, bb, X[1], 12);  RR5(bb, cc, dd, ee, aa, X[5], 5);
    RR5(aa, bb, cc, dd, ee, X[8], 14);  RR5(ee, aa, bb, cc, dd, X[7], 6);
    RR5(dd, ee, aa, bb, cc, X[6], 8);   RR5(cc, dd, ee, aa, bb, X[2], 13);
    RR5(bb, cc, dd, ee, aa, X[13], 6);  RR5(aa, bb, cc, dd, ee, X[14], 5);
    RR5(ee, aa, bb, cc, dd, X[0], 15);  RR5(dd, ee, aa, bb, cc, X[3], 13);
    RR5(cc, dd, ee, aa, bb, X[9], 11);  RR5(bb, cc, dd, ee, aa, X[11], 11);

    uint32_t t = d + cc + X[0];
    d = c + dd + X[1];
    c = b + ee + X[2];
    b = a + aa + X[3];
    a = e + bb + t;

    out[0] = b;
    out[1] = c;
    out[2] = d;
    out[3] = e;
    out[4] = a;
}

CUDA_DEV CUDA_INLINE uint64_t shfl_up64(uint64_t val, int delta) {
    uint32_t lo = (uint32_t)val;
    uint32_t hi = (uint32_t)(val >> 32);
    lo = __shfl_up_sync(0xFFFFFFFF, lo, delta);
    hi = __shfl_up_sync(0xFFFFFFFF, hi, delta);
    return ((uint64_t)hi << 32) | lo;
}

CUDA_DEV CUDA_INLINE uint64_t shfl_down64(uint64_t val, int delta) {
    uint32_t lo = (uint32_t)val;
    uint32_t hi = (uint32_t)(val >> 32);
    lo = __shfl_down_sync(0xFFFFFFFF, lo, delta);
    hi = __shfl_down_sync(0xFFFFFFFF, hi, delta);
    return ((uint64_t)hi << 32) | lo;
}

CUDA_DEV CUDA_INLINE uint64_t shfl64(uint64_t val, int srcLane) {
    uint32_t lo = (uint32_t)val;
    uint32_t hi = (uint32_t)(val >> 32);
    lo = __shfl_sync(0xFFFFFFFF, lo, srcLane);
    hi = __shfl_sync(0xFFFFFFFF, hi, srcLane);
    return ((uint64_t)hi << 32) | lo;
}

CUDA_DEV CUDA_INLINE Fe shfl_up_fe(const Fe& f, int delta) {
    Fe r;
    r.d[0] = shfl_up64(f.d[0], delta);
    r.d[1] = shfl_up64(f.d[1], delta);
    r.d[2] = shfl_up64(f.d[2], delta);
    r.d[3] = shfl_up64(f.d[3], delta);
    return r;
}

CUDA_DEV CUDA_INLINE Fe shfl_down_fe(const Fe& f, int delta) {
    Fe r;
    r.d[0] = shfl_down64(f.d[0], delta);
    r.d[1] = shfl_down64(f.d[1], delta);
    r.d[2] = shfl_down64(f.d[2], delta);
    r.d[3] = shfl_down64(f.d[3], delta);
    return r;
}

CUDA_DEV CUDA_INLINE Fe shfl_fe(const Fe& f, int srcLane) {
    Fe r;
    r.d[0] = shfl64(f.d[0], srcLane);
    r.d[1] = shfl64(f.d[1], srcLane);
    r.d[2] = shfl64(f.d[2], srcLane);
    r.d[3] = shfl64(f.d[3], srcLane);
    return r;
}

CUDA_DEV CUDA_INLINE Fe warp_montgomery_inv(const Fe& v, int lane) {
    Fe c = v;
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        Fe up = shfl_up_fe(c, offset);
        if (lane >= offset) {
            c = fe_mul(c, up);
        }
    }
    Fe inv_total;
    if (lane == 31) {
        inv_total = fe_inv(c);
    }
    Fe inv = shfl_fe(inv_total, 31);
    #pragma unroll
    for (int offset = 16; offset >= 1; offset /= 2) {
        Fe down = shfl_down_fe(inv, offset);
        if (lane + offset < 32) {
            inv = down;
        }
    }
    Fe prev_c;
    if (lane > 0) {
        prev_c = shfl_up_fe(c, 1);
        inv = fe_mul(inv, prev_c);
    }
    return inv;
}

CUDA_DEV CUDA_INLINE AffinePoint warp_montgomery_add_affine(const AffinePoint& P, const AffinePoint& Q, int lane) {
    Fe dx = fe_sub(Q.x, P.x);
    Fe dy = fe_sub(Q.y, P.y);
    Fe inv_dx = warp_montgomery_inv(dx, lane);
    Fe lambda = fe_mul(dy, inv_dx);
    Fe lambda2 = fe_sqr(lambda);
    Fe x3 = fe_sub(fe_sub(lambda2, P.x), Q.x);
    Fe y3 = fe_sub(fe_mul(lambda, fe_sub(P.x, x3)), P.y);
    AffinePoint res;
    res.x = x3;
    res.y = y3;
    return res;
}

CUDA_DEV CUDA_INLINE AffinePoint scalar_mul_G_windowed(const uint64_t scalar[4]) {
    JacobianPoint res;
    res.infinity = true;
    for (int limb = 3; limb >= 0; --limb) {
        uint64_t w = scalar[limb];
        for (int b = 60; b >= 0; b -= 4) {
            uint32_t window = (w >> b) & 0xF;
            for (int i = 0; i < 4; ++i) {
                if (!res.infinity) {
                    res = jacobian_double(res);
                }
            }
            if (window != 0) {
                AffinePoint pt = dev_G_table[window];
                res = jacobian_add_affine(res, pt);
            }
        }
    }
    return jacobian_to_affine(res);
}

CUDA_DEV CUDA_INLINE bool check_point_hash160(const AffinePoint& P, const uint32_t target_w[5], uint64_t target_h64) {
    uint8_t prefix = (P.y.d[0] & 1) ? 0x03 : 0x02;
    uint32_t X[16];
    fast_sha256_into_ripemd_X(prefix, P.x, X);
    uint32_t out[5];
    fast_ripemd160_32(X, out);
    uint64_t hash64 = ((uint64_t)out[0] << 32) | out[1];
    if (hash64 != target_h64) return false;
    return (out[2] == target_w[2] && out[3] == target_w[3] && out[4] == target_w[4]);
}

CUDA_GLOBAL void cuda_scan_kernel(
    u256 base_start,
    uint64_t total_keys,
    AffinePoint delta_G,
    uint32_t grid_threads,
    uint32_t steps
) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;

    u256 start_k = base_start + tid;
    uint64_t limbs[4] = {
        (uint64_t)start_k.low,
        (uint64_t)(start_k.low >> 64),
        (uint64_t)start_k.high,
        (uint64_t)(start_k.high >> 64)
    };
    AffinePoint P = scalar_mul_G_windowed(limbs);

    for (uint32_t s = 0; s < steps; ++s) {
        uint64_t offset = tid + (uint64_t)s * grid_threads;

        if (__any_sync(0xFFFFFFFF, dev_found_flag != 0)) break;

        if (offset < total_keys) {
            if (check_point_hash160(P, dev_target_w, dev_target_h64)) {
                if (atomicExch(&dev_found_flag, 1) == 0) {
                    dev_found_offset = offset;
                }
            }
        }

        if (s + 1 < steps) {
            P = warp_montgomery_add_affine(P, delta_G, lane);
        }
    }
}
#endif

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__)
inline void avx2_sha256_8way(const uint8_t in_chunks[8][33], uint8_t out_hashes[8][32]) {
    for (int i = 0; i < 8; ++i) {
        SHA256(in_chunks[i], 33, out_hashes[i]);
    }
}
inline void avx2_ripemd160_8way(const uint8_t in_chunks[8][32], uint8_t out_hashes[8][20]) {
    for (int i = 0; i < 8; ++i) {
        RIPEMD160(in_chunks[i], 32, out_hashes[i]);
    }
}
#endif
#endif

static AffinePoint G_TABLE[1024];
static bool g_table_initialized = false;
static std::mutex g_table_mtx;

void init_generator_table() {
    std::lock_guard<std::mutex> lock(g_table_mtx);
    if (g_table_initialized) return;
    AffinePoint G = get_generator_G();
    JacobianPoint cur;
    cur.x = G.x; cur.y = G.y;
    cur.z.d[0] = 1; cur.z.d[1] = 0; cur.z.d[2] = 0; cur.z.d[3] = 0;
    cur.infinity = false;

    G_TABLE[0] = G;
    for (int i = 1; i < 1024; ++i) {
        cur = jacobian_add_affine(cur, G);
        G_TABLE[i] = jacobian_to_affine(cur);
    }
    g_table_initialized = true;
}

void scan_worker_montgomery(
    u256 base_start,
    std::atomic<uint64_t>& work_offset,
    uint64_t total_keys,
    uint64_t slice_size,
    const uint8_t target_h160[20],
    uint64_t target_h64,
    std::atomic<bool>& found_flag,
    u256& found_key,
    std::mutex& found_mtx,
    std::atomic<uint64_t>& checked_counter
) {
    const uint32_t BATCH_SIZE = 1024;
    AffinePoint cur_points[1024];
    Fe dx_arr[1024];
    Fe prefix_prod[1024];

    while (g_running.load() && !found_flag.load()) {
        uint64_t offset = work_offset.fetch_add(slice_size);
        if (offset >= total_keys) break;
        uint64_t cur_slice = host_min(slice_size, total_keys - offset);

        for (uint64_t idx = 0; idx < cur_slice; idx += BATCH_SIZE) {
            if (!g_running.load() || found_flag.load()) break;
            uint32_t cur_batch = (uint32_t)host_min((uint64_t)BATCH_SIZE, cur_slice - idx);

            u256 k0 = base_start + (offset + idx);
            uint64_t limbs[4] = {
                (uint64_t)k0.low,
                (uint64_t)(k0.low >> 64),
                (uint64_t)k0.high,
                (uint64_t)(k0.high >> 64)
            };
            AffinePoint P0 = scalar_mul_G(limbs);
            cur_points[0] = P0;

            for (uint32_t i = 1; i < cur_batch; ++i) {
                dx_arr[i] = fe_sub(G_TABLE[i - 1].x, P0.x);
            }

            prefix_prod[1] = dx_arr[1];
            for (uint32_t i = 2; i < cur_batch; ++i) {
                prefix_prod[i] = fe_mul(prefix_prod[i - 1], dx_arr[i]);
            }

            Fe all_inv = fe_inv(prefix_prod[cur_batch - 1]);

            for (int i = cur_batch - 1; i >= 1; --i) {
                Fe inv_dxi;
                if (i == 1) {
                    inv_dxi = all_inv;
                } else {
                    inv_dxi = fe_mul(all_inv, prefix_prod[i - 1]);
                    all_inv = fe_mul(all_inv, dx_arr[i]);
                }
                Fe dy = fe_sub(G_TABLE[i - 1].y, P0.y);
                Fe lambda = fe_mul(dy, inv_dxi);
                Fe lambda2 = fe_sqr(lambda);
                Fe x3 = fe_sub(fe_sub(lambda2, P0.x), G_TABLE[i - 1].x);
                Fe y3 = fe_sub(fe_mul(lambda, fe_sub(P0.x, x3)), P0.y);
                cur_points[i].x = x3;
                cur_points[i].y = y3;
            }

            for (uint32_t i = 0; i < cur_batch; ++i) {
                uint8_t pub[33];
                pub[0] = (cur_points[i].y.d[0] & 1) ? 0x03 : 0x02;
                for (int b = 0; b < 4; ++b) {
                    uint64_t w = cur_points[i].x.d[3 - b];
                    for (int byte = 0; byte < 8; ++byte) {
                        pub[1 + b * 8 + byte] = (uint8_t)(w >> ((7 - byte) * 8));
                    }
                }

                uint8_t sha[32];
                SHA256(pub, 33, sha);
                uint8_t r160[20];
                RIPEMD160(sha, 32, r160);

                uint64_t h64 = 0;
                for (int b = 0; b < 8; ++b) {
                    h64 = (h64 << 8) | r160[b];
                }

                if (h64 == target_h64 && std::memcmp(r160, target_h160, 20) == 0) {
                    std::lock_guard<std::mutex> lock(found_mtx);
                    found_flag.store(true);
                    found_key = k0 + i;
                    checked_counter.fetch_add(i + 1);
                    return;
                }
            }
            checked_counter.fetch_add(cur_batch);
        }
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

    std::string api_base = "https://btcpuzzle.info/api.php";
    int current_puzzle = 70;
    std::string current_user = "guest";
    int requested_multiple = 1;

    unsigned int hw = std::thread::hardware_concurrency();
    int threads = (hw > 0) ? (int)hw : 4;
    bool force_cpu = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-s" || arg == "--server") && i + 1 < argc) {
            api_base = argv[++i];
        } else if ((arg == "-p" || arg == "--puzzle") && i + 1 < argc) {
            current_puzzle = std::atoi(argv[++i]);
        } else if ((arg == "-u" || arg == "--user") && i + 1 < argc) {
            current_user = argv[++i];
        } else if ((arg == "-m" || arg == "--multiple" || arg == "-b" || arg == "--batch") && i + 1 < argc) {
            requested_multiple = std::max(1, std::atoi(argv[++i]));
        } else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            threads = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "-cpu" || arg == "--cpu" || arg == "--cpu-only") {
            force_cpu = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                threads = std::max(1, std::atoi(argv[++i]));
            }
        }
    }

    std::cout << "==========================================================" << std::endl;
    std::cout << "  Bitcoin Puzzle Distributed Solver - Unified Engine      " << std::endl;
    std::cout << "==========================================================" << std::endl;
    std::cout << "[*] Target Server : " << api_base << std::endl;
    std::cout << "[*] Puzzle ID     : " << current_puzzle << std::endl;
    std::cout << "[*] Worker User   : " << current_user << std::endl;
    std::cout << "[*] Batch Multiple: " << requested_multiple << std::endl;

#ifdef __CUDACC__
    bool use_cuda = !force_cpu;
    if (use_cuda) {
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess || deviceCount == 0) {
            std::cout << "[!] No CUDA GPU detected. Falling back to CPU mode..." << std::endl;
            use_cuda = false;
        } else {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            std::cout << "[+] CUDA Device   : " << prop.name << " (" << prop.multiProcessorCount << " SMs)" << std::endl;
        }
    } else {
        std::cout << "[*] Mode          : Forced CPU mode (-cpu flag detected)" << std::endl;
        std::cout << "[*] CPU Threads   : " << threads << std::endl;
    }
#else
    std::cout << "[*] Mode          : Native CPU mode" << std::endl;
    std::cout << "[*] CPU Threads   : " << threads << std::endl;
#endif

    init_generator_table();

#ifdef __CUDACC__
    if (use_cuda) {
        AffinePoint h_table[16];
        AffinePoint G = get_generator_G();
        JacobianPoint cur;
        cur.x = G.x; cur.y = G.y;
        cur.z.d[0] = 1; cur.z.d[1] = 0; cur.z.d[2] = 0; cur.z.d[3] = 0;
        cur.infinity = false;
        h_table[0] = G;
        h_table[1] = G;
        for (int i = 2; i < 16; ++i) {
            cur = jacobian_add_affine(cur, G);
            h_table[i] = jacobian_to_affine(cur);
        }
        cudaMemcpyToSymbol(dev_G_table, h_table, sizeof(h_table));
    }
#endif

    int completed_ranges = 0;

    while (g_running.load()) {
        std::stringstream req_url;
        req_url << api_base << "?action=get_work&puzzle=" << current_puzzle
                << "&user=" << current_user << "&batch=" << requested_multiple;

        std::string resp;
        if (!http_get(req_url.str(), &resp)) {
            std::cerr << "[!] Network error fetching work. Retrying in 3s..." << std::endl;
            portable_sleep_ms(3000);
            continue;
        }

        std::string status = json_get_string(resp, "status");
        if (status == "solved") {
            std::cout << "[+] Puzzle #" << current_puzzle << " is already solved! Exiting..." << std::endl;
            break;
        }
        if (status != "ok" && status != "success") {
            std::cout << "[*] Server message: " << resp << ". Retrying in 3s..." << std::endl;
            portable_sleep_ms(3000);
            continue;
        }

        std::string str_block = json_get_string(resp, "block");
        std::string str_start = json_get_string(resp, "start");
        std::string str_end = json_get_string(resp, "end");
        std::string str_target = json_get_string(resp, "target");
        std::string str_range_count = json_get_string(resp, "range_count");
        if (str_range_count.empty()) str_range_count = json_get_string(resp, "multiple");

        if (str_start.empty() || str_end.empty() || str_target.empty()) {
            std::cerr << "[!] Invalid work payload: " << resp << std::endl;
            portable_sleep_ms(3000);
            continue;
        }

        uint64_t block_idx = (uint64_t)std::strtoull(str_block.c_str(), NULL, 10);
        int range_count = str_range_count.empty() ? requested_multiple : std::atoi(str_range_count.c_str());
        if (range_count <= 0) range_count = 1;

        u256 start_k = parse_u256(str_start);
        u256 end_k = parse_u256(str_end);
        u256 total_keys = end_k - start_k;
        uint64_t total_keys_count = (uint64_t)total_keys.low;

        uint8_t target_h160[20];
        if (!b58check_decode_hash160(str_target, target_h160)) {
            std::cerr << "[!] Error decoding target address base58check: " << str_target << std::endl;
            portable_sleep_ms(3000);
            continue;
        }

        uint64_t target_h64 = 0;
        for (int b = 0; b < 8; ++b) {
            target_h64 = (target_h64 << 8) | target_h160[b];
        }

        std::cout << "[*] Block " << block_idx << " | Ranges: " << range_count
                  << " | Keys: " << u256_to_dec(total_keys) << std::endl;

        bool hit = false;
        u256 found_key = 0;
        uint64_t checked = 0;
        auto t_start = std::chrono::high_resolution_clock::now();

#ifdef __CUDACC__
        if (use_cuda) {
            uint32_t h_target_w[5];
            for (int i = 0; i < 5; ++i) {
                h_target_w[i] = ((uint32_t)target_h160[i * 4] << 24) |
                                ((uint32_t)target_h160[i * 4 + 1] << 16) |
                                ((uint32_t)target_h160[i * 4 + 2] << 8) |
                                ((uint32_t)target_h160[i * 4 + 3]);
            }
            cudaMemcpyToSymbol(dev_target_w, h_target_w, sizeof(h_target_w));
            cudaMemcpyToSymbol(dev_target_h64, &target_h64, sizeof(uint64_t));

            int zero = 0;
            cudaMemcpyToSymbol(dev_found_flag, &zero, sizeof(int));

            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            uint32_t num_sms = prop.multiProcessorCount > 0 ? prop.multiProcessorCount : 40;
            uint32_t threadsPerBlock = 256;
            uint32_t numBlocks = num_sms * 16;
            uint32_t grid_threads = numBlocks * threadsPerBlock;
            uint32_t steps_per_launch = 1024;
            uint64_t chunk_size = (uint64_t)grid_threads * steps_per_launch;

            uint64_t delta_scalar[4] = { grid_threads, 0, 0, 0 };
            AffinePoint delta_G = scalar_mul_G(delta_scalar);

            uint64_t actual_checked = 0;
            while (actual_checked < total_keys_count && g_running.load() && !hit) {
                uint64_t cur_chunk = host_min(chunk_size, total_keys_count - actual_checked);
                uint32_t cur_steps = (uint32_t)((cur_chunk + grid_threads - 1) / grid_threads);
                u256 cur_start = start_k + actual_checked;

                cuda_scan_kernel<<<numBlocks, threadsPerBlock>>>(
                    cur_start, cur_chunk, delta_G, grid_threads, cur_steps
                );
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
            checked = actual_checked;
        } else
#endif
        {
            std::atomic<uint64_t> work_offset(0);
            uint64_t slice_size = 524288;
            std::atomic<bool> found_flag(false);
            std::mutex found_mtx;
            std::atomic<uint64_t> checked_counter(0);

            std::vector<std::thread> pool;
            for (int i = 0; i < threads; ++i) {
                pool.emplace_back(scan_worker_montgomery, start_k, std::ref(work_offset),
                                  total_keys_count, slice_size, target_h160, target_h64,
                                  std::ref(found_flag), std::ref(found_key),
                                  std::ref(found_mtx), std::ref(checked_counter));
            }
            for (auto& th : pool) if (th.joinable()) th.join();

            hit = found_flag.load();
            checked = checked_counter.load();
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t_end - t_start).count();
        if (elapsed <= 0.0) elapsed = 0.001;
        double speed = (double)checked / elapsed;

        if (hit) {
            std::cout << "\n========================================================" << std::endl;
            std::cout << "[!!!] PRIVATE KEY FOUND: 0x" << u256_to_hex64(found_key) << std::endl;
            std::cout << "========================================================\n" << std::endl;
        } else {
            std::cout << "[-] Range scan completed. Speed: " << std::fixed << std::setprecision(2)
                      << (speed / 1e6) << " MKeys/s | Elapsed: " << elapsed << "s" << std::endl;
        }

        if (!hit && !g_running.load()) break;

        std::stringstream json;
        json << "{\"action\":\"result\",\"puzzle\":" << current_puzzle
             << ",\"block\":" << block_idx
             << ",\"range_idx\":0"
             << ",\"range_count\":" << range_count
             << ",\"multiple\":" << range_count
             << ",\"status\":\"" << (hit ? "found" : "done") << "\""
             << ",\"private_key\":\"" << (hit ? u256_to_hex64(found_key) : "") << "\""
             << ",\"user\":\"" << current_user << "\""
             << ",\"speed\":" << std::fixed << std::setprecision(1) << speed
             << ",\"keys\":\"" << u256_to_dec(total_keys) << "\""
             << ",\"range_size\":\"" << u256_to_dec(total_keys) << "\""
             << ",\"count\":" << checked
             << ",\"elapsed\":" << std::fixed << std::setprecision(2) << elapsed << "}";

        std::string post_url = api_base + "?action=result&user=" + current_user;
        std::string ack;
        http_post(post_url, json.str(), &ack);

        completed_ranges++;
        if (hit) break;
    }

    curl_global_cleanup();
    return 0;
}
