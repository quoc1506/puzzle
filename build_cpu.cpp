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

#ifndef CURL_STATICLIB
#define CURL_STATICLIB
#endif

#include <curl/curl.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#if defined(__GNUC__) || defined(__clang__)
#include <x86intrin.h>
#endif
#endif

#if defined(__linux__) && !defined(__ANDROID__)
#include <pthread.h>
#include <sched.h>
#endif

// ============================================================================
// BITCOIN PUZZLE SOLVER - ULTRA-OPTIMIZED CPU WORKER (C++17 / AVX2 / BMI2)
// Designed for multi-core parallelism, AVX2 8-way SIMD hashing,
// and 1024-element Montgomery Batch Elliptic Curve Addition.
// ============================================================================

#define CUDA_HOSTDEV
#define CUDA_DEV
#define CUDA_GLOBAL
#define CUDA_INLINE inline
#define CUDA_CONSTANT static constexpr

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

CUDA_HOSTDEV CUDA_INLINE u256 operator<<(const u256& a, int shift) {
    if (shift == 0) return a;
    if (shift >= 256) return u256(0);
    u256 r;
    if (shift >= 128) {
        r.high = a.low << (shift - 128);
        r.low = 0;
    } else {
        r.high = (a.high << shift) | (a.low >> (128 - shift));
        r.low = a.low << shift;
    }
    return r;
}

CUDA_HOSTDEV CUDA_INLINE u256 operator*(const u256& a, uint64_t b) {
    u256 r = 0;
    for (int i = 0; i < 64; ++i) {
        if ((b >> i) & 1) {
            r = r + (a << i);
        }
    }
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
    uint64_t a0 = a.d[0], a1 = a.d[1], a2 = a.d[2], a3 = a.d[3];
    uint64_t b0 = b.d[0], b1 = b.d[1], b2 = b.d[2], b3 = b.d[3];

    // High-performance Comba Column Multiplication (Pure CPU Register Accumulation)
    u128 acc = (u128)a0 * b0;
    uint64_t t0 = (uint64_t)acc;
    acc >>= 64;

    acc += (u128)a0 * b1 + (u128)a1 * b0;
    uint64_t t1 = (uint64_t)acc;
    acc >>= 64;

    acc += (u128)a0 * b2 + (u128)a1 * b1 + (u128)a2 * b0;
    uint64_t t2 = (uint64_t)acc;
    acc >>= 64;

    acc += (u128)a0 * b3 + (u128)a1 * b2 + (u128)a2 * b1 + (u128)a3 * b0;
    uint64_t t3 = (uint64_t)acc;
    acc >>= 64;

    acc += (u128)a1 * b3 + (u128)a2 * b2 + (u128)a3 * b1;
    uint64_t t4 = (uint64_t)acc;
    acc >>= 64;

    acc += (u128)a2 * b3 + (u128)a3 * b2;
    uint64_t t5 = (uint64_t)acc;
    acc >>= 64;

    acc += (u128)a3 * b3;
    uint64_t t6 = (uint64_t)acc;
    uint64_t t7 = (uint64_t)(acc >> 64);

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

#if (defined(__x86_64__) || defined(_M_X64))
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

#if (defined(__x86_64__) || defined(_M_X64))
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

CUDA_HOSTDEV AffinePoint scalar_mul_G(const uint64_t limbs[4]) {
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

static AffinePoint G_TABLE[1024];
static bool g_table_initialized = false;
static std::mutex g_table_mtx;

void init_generator_table() {
    std::lock_guard<std::mutex> lock(g_table_mtx);
    if (g_table_initialized) return;
    JacobianPoint acc{G_POINT.x, G_POINT.y, Fe{{1, 0, 0, 0}}, false};
    G_TABLE[0] = G_POINT;
    for (int i = 1; i < 1024; ++i) {
        acc = jacobian_add_affine(acc, G_POINT);
        G_TABLE[i] = jacobian_to_affine(acc);
    }
    g_table_initialized = true;
}

// 4-bit windowed scalar multiplication using G_TABLE
AffinePoint scalar_mul_G_windowed(const uint64_t limbs[4]) {
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
                res = jacobian_add_affine(res, G_TABLE[nibble - 1]);
            }
        }
    }
    return jacobian_to_affine(res);
}

// ============================================================================
// ULTRA-FAST BITCOIN HASHING: SHA-256 + RIPEMD-160
// ============================================================================
static inline uint32_t rotr32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

static inline uint32_t bswap32(uint32_t x) {
    return __builtin_bswap32(x);
}

CUDA_HOSTDEV CUDA_INLINE void fast_sha256_into_ripemd_X(uint8_t prefix, const Fe& x, uint32_t X[8]) {
    uint8_t msg[64];
    msg[0] = prefix;
    for (int i = 0; i < 4; ++i) {
        uint64_t limb = x.d[3 - i];
        for (int b = 7; b >= 0; --b) {
            msg[1 + i * 8 + (7 - b)] = (uint8_t)(limb >> (b * 8));
        }
    }
    msg[33] = 0x80;
    std::memset(&msg[34], 0, 28);
    msg[62] = 0x01;
    msg[63] = 0x08; // 264 bits length (33 * 8)

    uint32_t W[64];
    for (int i = 0; i < 16; ++i) {
        W[i] = ((uint32_t)msg[i * 4] << 24) |
               ((uint32_t)msg[i * 4 + 1] << 16) |
               ((uint32_t)msg[i * 4 + 2] << 8) |
               ((uint32_t)msg[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = rotr32(W[i - 15], 7) ^ rotr32(W[i - 15], 18) ^ (W[i - 15] >> 3);
        uint32_t s1 = rotr32(W[i - 2], 17) ^ rotr32(W[i - 2], 19) ^ (W[i - 2] >> 10);
        W[i] = W[i - 16] + s0 + W[i - 7] + s1;
    }

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

    #pragma GCC unroll 16
    for (int i = 0; i < 64; ++i) {
        uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t temp1 = h + S1 + ch + K256[i] + W[i];
        uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;

        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }

    // Little-endian words for RIPEMD-160
    X[0] = bswap32(0x6a09e667 + a);
    X[1] = bswap32(0xbb67ae85 + b);
    X[2] = bswap32(0x3c6ef372 + c);
    X[3] = bswap32(0xa54ff53a + d);
    X[4] = bswap32(0x510e527f + e);
    X[5] = bswap32(0x9b05688c + f);
    X[6] = bswap32(0x1f83d9ab + g);
    X[7] = bswap32(0x5be0cd19 + h);
}

// RIPEMD-160 permutation constants
static const uint8_t host_rl_tab[80] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
    3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
    1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
    4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13
};

static const uint8_t host_sl_tab[80] = {
    11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
    7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
    11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
    11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
    9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6
};

static const uint8_t host_rr_tab[80] = {
    5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
    6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
    15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
    8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
    12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11
};

static const uint8_t host_sr_tab[80] = {
    8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
    9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
    9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
    15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
    8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11
};

CUDA_HOSTDEV CUDA_INLINE bool fast_ripemd160_32_check(const uint32_t X[8], const uint32_t target_w[5]) {
    auto rol32 = [](uint32_t val, int shift) -> uint32_t {
        return (val << shift) | (val >> (32 - shift));
    };

    uint32_t A = 0x67452301, B = 0xEFCDAB89, C = 0x98BADCFE, D = 0x10325476, E = 0xC3D2E1F0;
    for (int j = 0; j < 80; ++j) {
        uint32_t f = 0, K = 0;
        if (j < 16) { f = B ^ C ^ D; K = 0; }
        else if (j < 32) { f = (B & C) | (~B & D); K = 0x5A827999U; }
        else if (j < 48) { f = (B | ~C) ^ D; K = 0x6ED9EBA1U; }
        else if (j < 64) { f = (B & D) | (C & ~D); K = 0x8F1BBCD1U; }
        else { f = B ^ (C | ~D); K = 0xA953FD4EU; }

        uint8_t rl = host_rl_tab[j];
        uint32_t x_val = (rl < 8) ? X[rl] : (rl == 8 ? 0x00000080U : (rl == 14 ? 256U : 0U));
        uint32_t T = rol32(A + f + x_val + K, host_sl_tab[j]) + E;
        A = E; E = D; D = rol32(C, 10); C = B; B = T;
    }

    uint32_t Ap = 0x67452301, Bp = 0xEFCDAB89, Cp = 0x98BADCFE, Dp = 0x10325476, Ep = 0xC3D2E1F0;
    for (int j = 0; j < 80; ++j) {
        uint32_t fp = 0, Kp = 0;
        if (j < 16) { fp = Bp ^ (Cp | ~Dp); Kp = 0x50A28BE6U; }
        else if (j < 32) { fp = (Bp & Dp) | (Cp & ~Dp); Kp = 0x5C4DD124U; }
        else if (j < 48) { fp = (Bp | ~Cp) ^ Dp; Kp = 0x6D703EF3U; }
        else if (j < 64) { fp = (Bp & Cp) | (~Bp & Dp); Kp = 0x7A6D76E9U; }
        else { fp = Bp ^ Cp ^ Dp; Kp = 0; }

        uint8_t rr = host_rr_tab[j];
        uint32_t x_val = (rr < 8) ? X[rr] : (rr == 8 ? 0x00000080U : (rr == 14 ? 256U : 0U));
        uint32_t Tp = rol32(Ap + fp + x_val + Kp, host_sr_tab[j]) + Ep;
        Ap = Ep; Ep = Dp; Dp = rol32(Cp, 10); Cp = Bp; Bp = Tp;
    }

    if ((0xEFCDAB89U + C + Dp) != target_w[0]) return false;
    if ((0x98BADCFEU + D + Ep) != target_w[1]) return false;
    if ((0x10325476U + E + Ap) != target_w[2]) return false;
    if ((0xC3D2E1F0U + A + Bp) != target_w[3]) return false;
    return ((0x67452301U + B + Cp) == target_w[4]);
}

#if defined(__AVX2__)
void init_avx2_consts() {}
int fast_sha256_ripemd160_8x_avx2(const uint8_t prefixes[8], const Fe x[8], const uint32_t target_w[5]) {
    for (int lane = 0; lane < 8; ++lane) {
        uint32_t X[8];
        fast_sha256_into_ripemd_X(prefixes[lane], x[lane], X);
        if (fast_ripemd160_32_check(X, target_w)) return lane;
    }
    return -1;
}
#endif

// ============================================================================
// MONTGOMERY BATCH WORKER
// ============================================================================


void scan_worker_montgomery(
    u256 base_start,
    std::atomic<uint64_t>& work_offset,
    uint64_t total_keys,
    uint64_t slice_size,
    const uint8_t target_h160[20],
    uint64_t target_h64,
    const uint32_t target_w[5],
    std::atomic<bool>& found_flag,
    u256& found_key,
    std::mutex& found_mtx,
    std::atomic<uint64_t>& checked_counter
) {
#if defined(__AVX2__)
    init_avx2_consts();
#endif
    const uint32_t BATCH_SIZE = 512;
    alignas(64) Fe dx[512];
    alignas(64) Fe cum[513];
    alignas(64) Fe cur_x[512];
    alignas(64) uint8_t cur_prefix[512];

    uint64_t local_counter = 0;

    while (g_running.load(std::memory_order_relaxed) && !found_flag.load(std::memory_order_relaxed)) {
        uint64_t offset = work_offset.fetch_add(slice_size, std::memory_order_relaxed);
        if (offset >= total_keys) break;

        uint64_t cur_slice = std::min(slice_size, total_keys - offset);
        u256 slice_start = base_start + offset;
        u256 cur_k = slice_start;
        uint64_t remaining_in_slice = cur_slice;

        AffinePoint cur_base;
        bool cur_base_valid = false;

        if (cur_k > 1024) {
            u256 base_k = cur_k - 1;
            uint64_t limbs[4] = {
                (uint64_t)base_k.low,
                (uint64_t)(base_k.low >> 64),
                (uint64_t)base_k.high,
                (uint64_t)(base_k.high >> 64)
            };
            cur_base = scalar_mul_G_windowed(limbs);
            cur_base_valid = true;
        }

        while (remaining_in_slice > 0 && g_running.load(std::memory_order_relaxed) && !found_flag.load(std::memory_order_relaxed)) {
            uint32_t cur_batch = (uint32_t)std::min((uint64_t)BATCH_SIZE, remaining_in_slice);

            if (cur_k <= 1024) {
                for (uint32_t i = 0; i < cur_batch; ++i) {
                    uint64_t idx = (uint64_t)(cur_k.low + i);
                    if (idx >= 1 && idx <= 1024) {
                        cur_x[i] = G_TABLE[idx - 1].x;
                        cur_prefix[i] = (G_TABLE[idx - 1].y.d[0] & 1) ? 0x03 : 0x02;
                    } else {
                        uint64_t limbs[4] = { idx, 0, 0, 0 };
                        AffinePoint pt = scalar_mul_G_windowed(limbs);
                        cur_x[i] = pt.x;
                        cur_prefix[i] = (pt.y.d[0] & 1) ? 0x03 : 0x02;
                    }
                }
                cur_base_valid = false;
            } else {
                if (!cur_base_valid) {
                    u256 base_k = cur_k - 1;
                    uint64_t limbs[4] = {
                        (uint64_t)base_k.low,
                        (uint64_t)(base_k.low >> 64),
                        (uint64_t)base_k.high,
                        (uint64_t)(base_k.high >> 64)
                    };
                    cur_base = scalar_mul_G_windowed(limbs);
                    cur_base_valid = true;
                }

                // Forward pass of Montgomery Batch Inversion
                cum[0] = Fe{{1, 0, 0, 0}};
                for (uint32_t i = 0; i < cur_batch; ++i) {
                    dx[i] = fe_sub(G_TABLE[i].x, cur_base.x);
                    cum[i + 1] = fe_mul(cum[i], dx[i]);
                }

                Fe u = fe_inv(cum[cur_batch]);
                AffinePoint next_base;

                // Backward pass
                for (int i = (int)cur_batch - 1; i >= 1; --i) {
                    Fe inv_dx_i = fe_mul(u, cum[i]);
                    u = fe_mul(u, dx[i]);
                    Fe dy_i = fe_sub(G_TABLE[i].y, cur_base.y);
                    Fe lambda = fe_mul(dy_i, inv_dx_i);
                    Fe lambda2 = fe_sqr(lambda);
                    Fe xi = fe_sub(fe_sub(lambda2, cur_base.x), G_TABLE[i].x);
                    Fe yi = fe_sub(fe_mul(lambda, fe_sub(cur_base.x, xi)), cur_base.y);
                    uint8_t prefix = (yi.d[0] & 1) ? 0x03 : 0x02;
                    if (__builtin_expect(i == (int)cur_batch - 1, 0)) {
                        next_base.x = xi;
                        next_base.y = yi;
                    }
#if defined(__AVX2__)
                    cur_x[i] = xi;
                    cur_prefix[i] = prefix;
#else
                    uint32_t X[8];
                    fast_sha256_into_ripemd_X(prefix, xi, X);
                    if (fast_ripemd160_32_check(X, target_w)) {
                        std::lock_guard<std::mutex> lock(found_mtx);
                        found_flag.store(true, std::memory_order_release);
                        found_key = cur_k + (uint64_t)i;
                        checked_counter.fetch_add(local_counter + (uint64_t)(i + 1), std::memory_order_relaxed);
                        return;
                    }
#endif
                }
                // Handle i = 0 without unnecessary fe_mul
                {
                    Fe inv_dx_0 = u;
                    Fe dy_0 = fe_sub(G_TABLE[0].y, cur_base.y);
                    Fe lambda = fe_mul(dy_0, inv_dx_0);
                    Fe lambda2 = fe_sqr(lambda);
                    Fe x0 = fe_sub(fe_sub(lambda2, cur_base.x), G_TABLE[0].x);
                    Fe y0 = fe_sub(fe_mul(lambda, fe_sub(cur_base.x, x0)), cur_base.y);
                    uint8_t prefix = (y0.d[0] & 1) ? 0x03 : 0x02;
#if defined(__AVX2__)
                    cur_x[0] = x0;
                    cur_prefix[0] = prefix;
#else
                    uint32_t X[8];
                    fast_sha256_into_ripemd_X(prefix, x0, X);
                    if (fast_ripemd160_32_check(X, target_w)) {
                        std::lock_guard<std::mutex> lock(found_mtx);
                        found_flag.store(true, std::memory_order_release);
                        found_key = cur_k;
                        checked_counter.fetch_add(local_counter + 1, std::memory_order_relaxed);
                        return;
                    }
#endif
                }
                cur_base = next_base;
            }

#if defined(__AVX2__)
            uint32_t i = 0;
            for (; i + 8 <= cur_batch; i += 8) {
                int match_idx = fast_sha256_ripemd160_8x_avx2(&cur_prefix[i], &cur_x[i], target_w);
                if (__builtin_expect(match_idx >= 0, 0)) {
                    std::lock_guard<std::mutex> lock(found_mtx);
                    found_flag.store(true, std::memory_order_release);
                    found_key = cur_k + (uint64_t)(i + match_idx);
                    checked_counter.fetch_add(local_counter + (uint64_t)(i + match_idx + 1), std::memory_order_relaxed);
                    return;
                }
            }
            for (; i < cur_batch; ++i) {
                uint32_t X[8];
                fast_sha256_into_ripemd_X(cur_prefix[i], cur_x[i], X);
                if (fast_ripemd160_32_check(X, target_w)) {
                    std::lock_guard<std::mutex> lock(found_mtx);
                    found_flag.store(true, std::memory_order_release);
                    found_key = cur_k + (uint64_t)i;
                    checked_counter.fetch_add(local_counter + (uint64_t)(i + 1), std::memory_order_relaxed);
                    return;
                }
            }
#else
            if (!cur_base_valid) {
                for (uint32_t i = 0; i < cur_batch; ++i) {
                    uint32_t X[8];
                    fast_sha256_into_ripemd_X(cur_prefix[i], cur_x[i], X);
                    if (fast_ripemd160_32_check(X, target_w)) {
                        std::lock_guard<std::mutex> lock(found_mtx);
                        found_flag.store(true, std::memory_order_release);
                        found_key = cur_k + (uint64_t)i;
                        checked_counter.fetch_add(local_counter + (uint64_t)(i + 1), std::memory_order_relaxed);
                        return;
                    }
                }
            }
#endif
            cur_k = cur_k + (uint64_t)cur_batch;
            remaining_in_slice -= cur_batch;
            local_counter += cur_batch;
            if (local_counter >= 65536) {
                checked_counter.fetch_add(local_counter, std::memory_order_relaxed);
                local_counter = 0;
            }
        }
    }

    if (local_counter > 0) {
        checked_counter.fetch_add(local_counter, std::memory_order_relaxed);
    }
}

// ============================================================================
// HTTP CLIENT HELPERS
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





static const char* B58_CHARS = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";



// ============================================================================
// VERIFICATION RUNNER
// ============================================================================
int run_cpu_verify(const std::string& api_base, const std::string& current_user = "verify-node", int target_id = 0, int threads = 0) {
    if (threads <= 0) {
        threads = 1;
    }
    init_generator_table();

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
    std::cout << "[VERIFY] Running Montgomery 512-batch test scan (" << threads << " thread" << (threads > 1 ? "s" : "") << ")...\n";

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

        uint32_t target_w[5];
        for (int j = 0; j < 5; ++j) {
            target_w[j] = (uint32_t)target_h160[j * 4] |
                          ((uint32_t)target_h160[j * 4 + 1] << 8) |
                          ((uint32_t)target_h160[j * 4 + 2] << 16) |
                          ((uint32_t)target_h160[j * 4 + 3] << 24);
        }
        uint64_t target_h64 = (uint64_t)target_w[0] | ((uint64_t)target_w[1] << 32);

        alignas(64) std::atomic<uint64_t> work_offset(0);
        uint64_t slice_size = std::max((uint64_t)2048, total_keys_count / (uint64_t)(threads * 2));
        alignas(64) std::atomic<bool> found_flag(false);
        alignas(64) std::mutex found_mtx;
        alignas(64) u256 found_key = 0;
        alignas(64) std::atomic<uint64_t> checked_counter(0);

        auto t_scan_start = std::chrono::high_resolution_clock::now();
        std::vector<std::thread> pool;
        pool.reserve(threads);
        for (int t = 0; t < threads; ++t) {
            pool.emplace_back([=, &work_offset, &found_flag, &found_key, &found_mtx, &checked_counter]() {
                scan_worker_montgomery(start_k, std::ref(work_offset),
                                      total_keys_count, slice_size, target_h160, target_h64,
                                      target_w, std::ref(found_flag), std::ref(found_key),
                                      std::ref(found_mtx), std::ref(checked_counter));
            });
        }
        for (auto& th : pool) {
            if (th.joinable()) th.join();
        }
        auto t_scan_end = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(t_scan_end - t_scan_start).count();
        if (elapsed_sec <= 0.0) elapsed_sec = 0.0001;

        uint64_t actual_checked = checked_counter.load();
        if (actual_checked == 0) actual_checked = total_keys_count;
        total_keys_verified += actual_checked;
        double target_speed = (double)actual_checked / elapsed_sec;

        if (found_flag.load()) {
            passed++;
            std::cout << "[PASS] Target #" << pid
                      << " | Speed: " << format_speed(target_speed)
                      << " | Key: 0x" << u256_to_hex64(found_key)
                      << " -> Matched Server Target\n";
        } else {
            failed++;
            std::cerr << "[FAIL] Target #" << pid << " -> Key not found in range\n";
        }
    }

    auto t_global_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_global_end - t_global_start).count();
    if (total_sec <= 0.0) total_sec = 0.0001;
    double avg_verify_speed = (double)total_keys_verified / total_sec;

    std::cout << "\n";
    if (failed == 0 && passed > 0) {
        std::cout << "[OK] " << passed << "/" << tested << " targets verified successfully via server range scan.\n";
        std::cout << "[SPEED] Average Verification Throughput: " << format_speed(avg_verify_speed)
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

// ============================================================================
// MAIN ENTRYPOINT WITH FULL CORE FLOW CLI OPTIONS
// ============================================================================
int main(int argc, char* argv[]) {
    // 1. Default server URL is http://65.20.91.208/puzzle_server.php
    std::string api_base = "http://65.20.91.208/puzzle_server.php";
    std::string user = "worker-1";
    int threads = 1; // Default 1 thread as per user specification
    int puzzle_id = 71;
    bool verify_mode = false;
    bool explicit_puzzle = false;
    bool no_limit = false; // Default: limit to 50 ranges unless -nl / --no-limit is provided
    int max_ranges = 50;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        // Server URL: --server, -s, --api (supports both space and = delimiter)
        if ((arg == "--server" || arg == "-s" || arg == "--api") && i + 1 < argc) {
            api_base = argv[++i];
        } else if (arg.rfind("--server=", 0) == 0) {
            api_base = arg.substr(9);
        } else if (arg.rfind("-s=", 0) == 0) {
            api_base = arg.substr(3);
        } else if (arg.rfind("--api=", 0) == 0) {
            api_base = arg.substr(6);
        }
        // User: --user, -u
        else if ((arg == "--user" || arg == "-u") && i + 1 < argc) {
            user = argv[++i];
        } else if (arg.rfind("--user=", 0) == 0) {
            user = arg.substr(7);
        } else if (arg.rfind("-u=", 0) == 0) {
            user = arg.substr(3);
        }
        // Fast mode: --fast -> full CPU threads
        else if (arg == "--fast") {
            unsigned int hw = std::thread::hardware_concurrency();
            threads = (hw > 0) ? (int)hw : 4;
        }
        // Dual thread mode: -d -> 2 threads
        else if (arg == "-d") {
            threads = 2;
        }
        // Custom thread count: -t, --threads
        else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            threads = std::max(1, std::atoi(argv[++i]));
        } else if (arg.rfind("-t=", 0) == 0) {
            threads = std::max(1, std::atoi(arg.substr(3).c_str()));
        } else if (arg.rfind("--threads=", 0) == 0) {
            threads = std::max(1, std::atoi(arg.substr(10).c_str()));
        }
        // No limit mode: -nl, --no-limit
        else if (arg == "-nl" || arg == "--no-limit") {
            no_limit = true;
        }
        // Verify mode: --verify, -v
        else if (arg == "--verify" || arg == "-v") {
            verify_mode = true;
        }
        // Target puzzle ID: -p, --puzzle
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

    // In verify mode, if user did NOT explicitly specify --puzzle, test all sample puzzles {65..70} (target_id = 0)
    if (verify_mode) {
        int verify_target = explicit_puzzle ? puzzle_id : 0;
        return run_cpu_verify(api_base, user, verify_target, threads);
    }

    init_generator_table();

    std::cout << "[HARDWARE] Engine: Montgomery Batch 512x (Comba Column Registers) | Threads: " << threads << "\n";
    std::cout << "[WORKER] Connecting to coordinator: " << api_base << "\n";
    std::cout << "[CONFIG] User: " << user << " | Target Puzzle: #" << puzzle_id;
    if (no_limit) {
        std::cout << " | Mode: Unlimited Ranges (-nl)\n";
    } else {
        std::cout << " | Mode: Fixed 50 Ranges (use -nl for unlimited)\n";
    }

    int ranges_completed = 0;

    // Main solver scan loop
    while (g_running.load()) {
        if (!no_limit && ranges_completed >= max_ranges) {
            std::cout << "\n[STOP] Reached limit of " << max_ranges << " ranges completed without -nl. Exiting cleanly.\n";
            break;
        }

        std::string req_url = api_base + "?action=range&puzzle=" + std::to_string(puzzle_id) + "&user=" + user;
        std::string resp;
        if (!http_get(req_url, &resp)) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            continue;
        }

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
        u256 end_k = str_end.empty() ? (start_k + 268435456) : parse_u256(str_end);
        u256 diff = end_k - start_k;
        uint64_t total_keys_count = (diff.high > 0) ? 268435456 : (uint64_t)diff.low;
        if (total_keys_count == 0) total_keys_count = 268435456;

        uint8_t target_h160[20];
        if (!b58check_decode_hash160(str_target, target_h160)) continue;

        uint32_t target_w[5];
        for (int j = 0; j < 5; ++j) {
            target_w[j] = (uint32_t)target_h160[j * 4] |
                          ((uint32_t)target_h160[j * 4 + 1] << 8) |
                          ((uint32_t)target_h160[j * 4 + 2] << 16) |
                          ((uint32_t)target_h160[j * 4 + 3] << 24);
        }
        uint64_t target_h64 = (uint64_t)target_w[0] | ((uint64_t)target_w[1] << 32);

        alignas(64) std::atomic<uint64_t> work_offset(0);
        uint64_t slice_size = std::max((uint64_t)4194304, total_keys_count / (uint64_t)(threads * 2));
        alignas(64) std::atomic<bool> found_flag(false);
        alignas(64) std::mutex found_mtx;
        alignas(64) u256 found_key = 0;
        alignas(64) std::atomic<uint64_t> checked_counter(0);

        auto t_start = std::chrono::high_resolution_clock::now();
        std::vector<std::thread> pool;
        pool.reserve(threads);
        for (int t = 0; t < threads; ++t) {
            pool.emplace_back([=, &work_offset, &found_flag, &found_key, &found_mtx, &checked_counter]() {
                scan_worker_montgomery(start_k, std::ref(work_offset),
                                      total_keys_count, slice_size, target_h160, target_h64,
                                      target_w, std::ref(found_flag), std::ref(found_key),
                                      std::ref(found_mtx), std::ref(checked_counter));
            });
        }

        // Progress reporter
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            uint64_t cur = checked_counter.load();
            auto t_now = std::chrono::high_resolution_clock::now();
            double el = std::chrono::duration<double>(t_now - t_start).count();
            if (el > 0) {
                double spd = (double)cur / el;
                std::cout << "\r[SCAN] Block: " << str_block << " | Range: " << str_range
                          << " | Progress: " << std::fixed << std::setprecision(1) << ((double)cur / total_keys_count * 100.0) << "%"
                          << " | Speed: " << format_speed(spd) << std::flush;
            }
            if (cur >= total_keys_count || found_flag.load() || !g_running.load()) break;
        }

        for (auto& th : pool) {
            if (th.joinable()) th.join();
        }
        std::cout << "\n";

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t_end - t_start).count();
        double final_spd = (elapsed > 0) ? ((double)checked_counter.load() / elapsed) : 0;

        ranges_completed++;

        if (found_flag.load()) {
            std::cout << "\n[WINNER] FOUND KEY: 0x" << u256_to_hex64(found_key) << "\n";
            std::stringstream res_json;
            res_json << "{\"action\":\"result\",\"puzzle\":" << puzzle_id
                     << ",\"block\":" << str_block
                     << ",\"range_idx\":" << str_range
                     << ",\"status\":\"found\",\"user\":\"" << user
                     << "\",\"private_key\":\"" << u256_to_hex64(found_key)
                     << "\",\"speed\":" << (uint64_t)final_spd << "}";
            std::string ack;
            http_post(api_base, res_json.str(), &ack);
            break;
        } else {
            std::stringstream res_json;
            res_json << "{\"action\":\"result\",\"puzzle\":" << puzzle_id
                     << ",\"block\":" << str_block
                     << ",\"range_idx\":" << str_range
                     << ",\"status\":\"done\",\"user\":\"" << user
                     << "\",\"speed\":" << (uint64_t)final_spd << "}";
            std::string ack;
            http_post(api_base, res_json.str(), &ack);
        }
    }

    return 0;
}
