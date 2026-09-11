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
#ifndef SECP256K1_STATIC
#define SECP256K1_STATIC
#endif

#include <curl/curl.h>
#include <secp256k1.h>
#include <openssl/sha.h>
#include <openssl/ripemd.h>
#include <openssl/bn.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#if defined(__GNUC__) || defined(__clang__)
#include <x86intrin.h>
#endif
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

typedef __uint128_t u128;

struct u256 {
    __uint128_t low;
    __uint128_t high;

    u256() : low(0), high(0) {}
    u256(int v) : low((uint64_t)v), high(0) {}
    u256(uint32_t v) : low(v), high(0) {}
    u256(uint64_t v) : low(v), high(0) {}
    u256(__uint128_t l) : low(l), high(0) {}
    u256(__uint128_t l, __uint128_t h) : low(l), high(h) {}

    inline bool is_zero() const {
        return (low | high) == 0;
    }

    inline void to_bytes_be(uint8_t out[32]) const {
        for (int i = 0; i < 16; ++i) {
            out[31 - i] = (uint8_t)(low >> (i * 8));
            out[15 - i] = (uint8_t)(high >> (i * 8));
        }
    }
};

inline bool operator==(const u256& a, const u256& b) {
    return a.low == b.low && a.high == b.high;
}
inline bool operator!=(const u256& a, const u256& b) {
    return !(a == b);
}
inline bool operator<(const u256& a, const u256& b) {
    if (a.high != b.high) return a.high < b.high;
    return a.low < b.low;
}
inline bool operator<=(const u256& a, const u256& b) {
    return (a < b) || (a == b);
}
inline bool operator>(const u256& a, const u256& b) {
    return b < a;
}
inline bool operator>=(const u256& a, const u256& b) {
    return !(a < b);
}

inline u256 operator+(const u256& a, const u256& b) {
    u256 r;
    r.low = a.low + b.low;
    r.high = a.high + b.high + (r.low < a.low ? 1 : 0);
    return r;
}
inline u256 operator+(const u256& a, uint64_t b) {
    u256 r;
    r.low = a.low + b;
    r.high = a.high + (r.low < a.low ? 1 : 0);
    return r;
}
inline u256& operator+=(u256& a, const u256& b) {
    __uint128_t old_low = a.low;
    a.low += b.low;
    a.high += b.high + (a.low < old_low ? 1 : 0);
    return a;
}
inline u256& operator+=(u256& a, uint64_t b) {
    __uint128_t old_low = a.low;
    a.low += b;
    a.high += (a.low < old_low ? 1 : 0);
    return a;
}

inline u256 operator-(const u256& a, const u256& b) {
    u256 r;
    r.low = a.low - b.low;
    r.high = a.high - b.high - (a.low < b.low ? 1 : 0);
    return r;
}
inline u256 operator-(const u256& a, uint64_t b) {
    u256 r;
    r.low = a.low - b;
    r.high = a.high - (a.low < b ? 1 : 0);
    return r;
}
inline u256& operator-=(u256& a, const u256& b) {
    bool borrow = a.low < b.low;
    a.low -= b.low;
    a.high -= b.high + (borrow ? 1 : 0);
    return a;
}
inline u256& operator-=(u256& a, uint64_t b) {
    bool borrow = a.low < b;
    a.low -= b;
    a.high -= (borrow ? 1 : 0);
    return a;
}

u256 parse_u256(const std::string& str) {
    u256 res;
    if (str.empty()) return res;
    if (str.rfind("0x", 0) == 0 || str.rfind("0X", 0) == 0) {
        for (size_t i = 2; i < str.size(); ++i) {
            char c = str[i];
            int val = 0;
            if (c >= '0' && c <= '9') val = c - '0';
            else if (c >= 'a' && c <= 'f') val = c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') val = c - 'A' + 10;
            else continue;
            uint64_t carry = (uint64_t)(res.low >> 124);
            res.low = (res.low << 4) | val;
            res.high = (res.high << 4) | carry;
        }
        return res;
    }
    for (char c : str) {
        if (c >= '0' && c <= '9') {
            int val = c - '0';
            uint64_t l0 = (uint64_t)res.low;
            uint64_t l1 = (uint64_t)(res.low >> 64);
            uint64_t h0 = (uint64_t)res.high;
            uint64_t h1 = (uint64_t)(res.high >> 64);

            __uint128_t c0 = (__uint128_t)l0 * 10 + val;
            l0 = (uint64_t)c0;
            __uint128_t c1 = (__uint128_t)l1 * 10 + (c0 >> 64);
            l1 = (uint64_t)c1;
            __uint128_t c2 = (__uint128_t)h0 * 10 + (c1 >> 64);
            h0 = (uint64_t)c2;
            __uint128_t c3 = (__uint128_t)h1 * 10 + (c2 >> 64);
            h1 = (uint64_t)c3;

            res.low = ((__uint128_t)l1 << 64) | l0;
            res.high = ((__uint128_t)h1 << 64) | h0;
        }
    }
    return res;
}

std::string u256_to_dec(u256 v) {
    if (v.is_zero()) return "0";
    std::string s;
    while (!v.is_zero()) {
        uint64_t l0 = (uint64_t)v.low;
        uint64_t l1 = (uint64_t)(v.low >> 64);
        uint64_t h0 = (uint64_t)v.high;
        uint64_t h1 = (uint64_t)(v.high >> 64);

        uint64_t r = 0;
        __uint128_t cur = ((__uint128_t)r << 64) | h1;
        h1 = (uint64_t)(cur / 10);
        r = (uint64_t)(cur % 10);

        cur = ((__uint128_t)r << 64) | h0;
        h0 = (uint64_t)(cur / 10);
        r = (uint64_t)(cur % 10);

        cur = ((__uint128_t)r << 64) | l1;
        l1 = (uint64_t)(cur / 10);
        r = (uint64_t)(cur % 10);

        cur = ((__uint128_t)r << 64) | l0;
        l0 = (uint64_t)(cur / 10);
        r = (uint64_t)(cur % 10);

        s.push_back('0' + (int)r);

        v.low = ((__uint128_t)l1 << 64) | l0;
        v.high = ((__uint128_t)h1 << 64) | h0;
    }
    std::reverse(s.begin(), s.end());
    return s;
}

inline std::ostream& operator<<(std::ostream& os, const u256& v) {
    return os << u256_to_dec(v);
}

std::string u256_to_hex64(const u256& v) {
    char buf[65];
    snprintf(buf, sizeof(buf), "%016llx%016llx%016llx%016llx",
             (unsigned long long)(v.high >> 64),
             (unsigned long long)(v.high & 0xFFFFFFFFFFFFFFFFULL),
             (unsigned long long)(v.low >> 64),
             (unsigned long long)(v.low & 0xFFFFFFFFFFFFFFFFULL));
    return std::string(buf);
}

struct Fe {
    uint64_t d[4];
};

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC optimize("O3,unroll-loops,omit-frame-pointer")
#endif

static const uint64_t SECP_K = 0x1000003D1ULL;

static inline bool fe_is_zero(const Fe& a) {
    return (a.d[0] | a.d[1] | a.d[2] | a.d[3]) == 0;
}

static inline bool fe_eq(const Fe& a, const Fe& b) {
    return a.d[0] == b.d[0] && a.d[1] == b.d[1] && a.d[2] == b.d[2] && a.d[3] == b.d[3];
}

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
static inline __attribute__((always_inline)) Fe fe_add(const Fe& a, const Fe& b) {
    Fe r;
    unsigned char carry = 0;
    carry = _addcarry_u64(carry, a.d[0], b.d[0], (unsigned long long*)&r.d[0]);
    carry = _addcarry_u64(carry, a.d[1], b.d[1], (unsigned long long*)&r.d[1]);
    carry = _addcarry_u64(carry, a.d[2], b.d[2], (unsigned long long*)&r.d[2]);
    carry = _addcarry_u64(carry, a.d[3], b.d[3], (unsigned long long*)&r.d[3]);

    if (carry) {
        carry = _addcarry_u64(0, r.d[0], SECP_K, (unsigned long long*)&r.d[0]);
        carry = _addcarry_u64(carry, r.d[1], 0, (unsigned long long*)&r.d[1]);
        carry = _addcarry_u64(carry, r.d[2], 0, (unsigned long long*)&r.d[2]);
        _addcarry_u64(carry, r.d[3], 0, (unsigned long long*)&r.d[3]);
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
}

static inline __attribute__((always_inline)) Fe fe_sub(const Fe& a, const Fe& b) {
    Fe r;
    unsigned char borrow = 0;
    borrow = _subborrow_u64(borrow, a.d[0], b.d[0], (unsigned long long*)&r.d[0]);
    borrow = _subborrow_u64(borrow, a.d[1], b.d[1], (unsigned long long*)&r.d[1]);
    borrow = _subborrow_u64(borrow, a.d[2], b.d[2], (unsigned long long*)&r.d[2]);
    borrow = _subborrow_u64(borrow, a.d[3], b.d[3], (unsigned long long*)&r.d[3]);

    if (borrow) {
        borrow = _subborrow_u64(0, r.d[0], SECP_K, (unsigned long long*)&r.d[0]);
        borrow = _subborrow_u64(borrow, r.d[1], 0, (unsigned long long*)&r.d[1]);
        borrow = _subborrow_u64(borrow, r.d[2], 0, (unsigned long long*)&r.d[2]);
        _subborrow_u64(borrow, r.d[3], 0, (unsigned long long*)&r.d[3]);
    }
    return r;
}
#else
static inline __attribute__((always_inline)) Fe fe_add(const Fe& a, const Fe& b) {
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
        u128 c2 = (u128)r.d[0] + SECP_K;
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
}

static inline __attribute__((always_inline)) Fe fe_sub(const Fe& a, const Fe& b) {
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
        c = (u128)r.d[0] - SECP_K;
        r.d[0] = (uint64_t)c;
        c = (u128)r.d[1] - ((c >> 64) & 1);
        r.d[1] = (uint64_t)c;
        c = (u128)r.d[2] - ((c >> 64) & 1);
        r.d[2] = (uint64_t)c;
        r.d[3] -= (uint64_t)((c >> 64) & 1);
    }
    return r;
}
#endif

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__)) && defined(__BMI2__) && defined(__ADX__)
static inline __attribute__((always_inline)) Fe fe_mul(const Fe& a, const Fe& b) {
    uint64_t r0, r1, r2, r3;
    uint64_t t4, t5, t6, t7;

    __asm__ __volatile__ (
        // --- Row 0: a * b[0] ---
        "movq 0(%[b]), %%rdx\n\t"
        "mulx 0(%[a]), %[r0], %[r1]\n\t"
        "mulx 8(%[a]), %%rax, %[r2]\n\t"
        "addq %%rax, %[r1]\n\t"
        "mulx 16(%[a]), %%rax, %[r3]\n\t"
        "adcq %%rax, %[r2]\n\t"
        "mulx 24(%[a]), %%rax, %[t4]\n\t"
        "adcq %%rax, %[r3]\n\t"
        "adcq $0, %[t4]\n\t"
        "xorq %[t5], %[t5]\n\t"

        // --- Row 1: a * b[1] ---
        "movq 8(%[b]), %%rdx\n\t"
        "mulx 0(%[a]), %%rax, %%rcx\n\t"
        "adcx %%rax, %[r1]\n\t"
        "adox %%rcx, %[r2]\n\t"
        "mulx 8(%[a]), %%rax, %%rcx\n\t"
        "adcx %%rax, %[r2]\n\t"
        "adox %%rcx, %[r3]\n\t"
        "mulx 16(%[a]), %%rax, %%rcx\n\t"
        "adcx %%rax, %[r3]\n\t"
        "adox %%rcx, %[t4]\n\t"
        "mulx 24(%[a]), %%rax, %%rcx\n\t"
        "adcx %%rax, %[t4]\n\t"
        "adox %%rcx, %[t5]\n\t"
        "movq $0, %%rax\n\t"
        "adcx %%rax, %[t5]\n\t"
        "adox %%rax, %[t5]\n\t"
        "xorq %[t6], %[t6]\n\t"

        // --- Row 2: a * b[2] ---
        "movq 16(%[b]), %%rdx\n\t"
        "mulx 0(%[a]), %%rax, %%rcx\n\t"
        "adcx %%rax, %[r2]\n\t"
        "adox %%rcx, %[r3]\n\t"
        "mulx 8(%[a]), %%rax, %%rcx\n\t"
        "adcx %%rax, %[r3]\n\t"
        "adox %%rcx, %[t4]\n\t"
        "mulx 16(%[a]), %%rax, %%rcx\n\t"
        "adcx %%rax, %[t4]\n\t"
        "adox %%rcx, %[t5]\n\t"
        "mulx 24(%[a]), %%rax, %%rcx\n\t"
        "adcx %%rax, %[t5]\n\t"
        "adox %%rcx, %[t6]\n\t"
        "movq $0, %%rax\n\t"
        "adcx %%rax, %[t6]\n\t"
        "adox %%rax, %[t6]\n\t"
        "xorq %[t7], %[t7]\n\t"

        // --- Row 3: a * b[3] ---
        "movq 24(%[b]), %%rdx\n\t"
        "mulx 0(%[a]), %%rax, %%rcx\n\t"
        "adcx %%rax, %[r3]\n\t"
        "adox %%rcx, %[t4]\n\t"
        "mulx 8(%[a]), %%rax, %%rcx\n\t"
        "adcx %%rax, %[t4]\n\t"
        "adox %%rcx, %[t5]\n\t"
        "mulx 16(%[a]), %%rax, %%rcx\n\t"
        "adcx %%rax, %[t5]\n\t"
        "adox %%rcx, %[t6]\n\t"
        "mulx 24(%[a]), %%rax, %%rcx\n\t"
        "adcx %%rax, %[t6]\n\t"
        "adox %%rcx, %[t7]\n\t"
        "movq $0, %%rax\n\t"
        "adcx %%rax, %[t7]\n\t"
        "adox %%rax, %[t7]\n\t"

        // --- Reduction Pass 1: Add (t4, t5, t6, t7) * SECP_K to (r0, r1, r2, r3) ---
        "movabsq $0x1000003D1, %%rdx\n\t"
        "xorq %%rax, %%rax\n\t"

        "mulx %[t4], %%rax, %%rcx\n\t"
        "adcx %%rax, %[r0]\n\t"
        "adox %%rcx, %[r1]\n\t"

        "mulx %[t5], %%rax, %%rcx\n\t"
        "adcx %%rax, %[r1]\n\t"
        "adox %%rcx, %[r2]\n\t"

        "mulx %[t6], %%rax, %%rcx\n\t"
        "adcx %%rax, %[r2]\n\t"
        "adox %%rcx, %[r3]\n\t"

        "movq $0, %[t4]\n\t"
        "mulx %[t7], %%rax, %%rcx\n\t"
        "adcx %%rax, %[r3]\n\t"
        "adox %%rcx, %[t4]\n\t"
        "movq $0, %%rax\n\t"
        "adcx %%rax, %[t4]\n\t"
        "adox %%rax, %[t4]\n\t"

        // --- Reduction Pass 2: Add t4 * SECP_K to (r0, r1, r2, r3) ---
        "mulx %[t4], %%rax, %%rcx\n\t"
        "addq %%rax, %[r0]\n\t"
        "adcq %%rcx, %[r1]\n\t"
        "adcq $0, %[r2]\n\t"
        "adcq $0, %[r3]\n\t"
        "movq $0, %[t4]\n\t"
        "adcq $0, %[t4]\n\t"

        // Extra carry handling (t4 is 0 or 1)
        "imulq %%rdx, %[t4]\n\t"
        "addq %[t4], %[r0]\n\t"
        "adcq $0, %[r1]\n\t"
        "adcq $0, %[r2]\n\t"
        "adcq $0, %[r3]\n\t"

        // --- Final boundary check if r >= p: add SECP_K and cmovc ---
        "movq %[r0], %%rax\n\t"
        "addq %%rdx, %%rax\n\t"
        "movq %[r1], %%rcx\n\t"
        "adcq $0, %%rcx\n\t"
        "movq %[r2], %[t4]\n\t"
        "adcq $0, %[t4]\n\t"
        "movq %[r3], %[t5]\n\t"
        "adcq $0, %[t5]\n\t"
        "cmovcq %%rax, %[r0]\n\t"
        "cmovcq %%rcx, %[r1]\n\t"
        "cmovcq %[t4], %[r2]\n\t"
        "cmovcq %[t5], %[r3]\n\t"
        : [r0] "=&r"(r0), [r1] "=&r"(r1), [r2] "=&r"(r2), [r3] "=&r"(r3),
          [t4] "=&r"(t4), [t5] "=&r"(t5), [t6] "=&r"(t6), [t7] "=&r"(t7)
        : [a] "r"(a.d), [b] "r"(b.d)
        : "rax", "rcx", "rdx", "cc", "memory"
    );

    Fe r;
    r.d[0] = r0; r.d[1] = r1; r.d[2] = r2; r.d[3] = r3;
    return r;
}

static inline __attribute__((always_inline)) Fe fe_sqr(const Fe& a) {
    return fe_mul(a, a);
}
#else
static inline __attribute__((always_inline)) Fe fe_mul(const Fe& a, const Fe& b) {
    uint64_t t[8] = {0};
    u128 a0 = a.d[0], a1 = a.d[1], a2 = a.d[2], a3 = a.d[3];
    u128 b0 = b.d[0], b1 = b.d[1], b2 = b.d[2], b3 = b.d[3];

    u128 c;
    c = a0 * b0; t[0] = (uint64_t)c; c >>= 64;
    c += a0 * b1; t[1] = (uint64_t)c; c >>= 64;
    c += a0 * b2; t[2] = (uint64_t)c; c >>= 64;
    c += a0 * b3; t[3] = (uint64_t)c; t[4] = (uint64_t)(c >> 64);

    c = (u128)t[1] + a1 * b0; t[1] = (uint64_t)c; c >>= 64;
    c += (u128)t[2] + a1 * b1; t[2] = (uint64_t)c; c >>= 64;
    c += (u128)t[3] + a1 * b2; t[3] = (uint64_t)c; c >>= 64;
    c += (u128)t[4] + a1 * b3; t[4] = (uint64_t)c; t[5] = (uint64_t)(c >> 64);

    c = (u128)t[2] + a2 * b0; t[2] = (uint64_t)c; c >>= 64;
    c += (u128)t[3] + a2 * b1; t[3] = (uint64_t)c; c >>= 64;
    c += (u128)t[4] + a2 * b2; t[4] = (uint64_t)c; c >>= 64;
    c += (u128)t[5] + a2 * b3; t[5] = (uint64_t)c; t[6] = (uint64_t)(c >> 64);

    c = (u128)t[3] + a3 * b0; t[3] = (uint64_t)c; c >>= 64;
    c += (u128)t[4] + a3 * b1; t[4] = (uint64_t)c; c >>= 64;
    c += (u128)t[5] + a3 * b2; t[5] = (uint64_t)c; c >>= 64;
    c += (u128)t[6] + a3 * b3; t[6] = (uint64_t)c; t[7] = (uint64_t)(c >> 64);

    u128 carry = 0;
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
    if (extra) {
        u128 c3 = (u128)t[0] + (u128)extra * SECP_K;
        t[0] = (uint64_t)c3; c3 >>= 64;
        c3 += t[1]; t[1] = (uint64_t)c3; c3 >>= 64;
        c3 += t[2]; t[2] = (uint64_t)c3; c3 >>= 64;
        t[3] += (uint64_t)c3;
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

static inline __attribute__((always_inline)) Fe fe_sqr(const Fe& a) {
    uint64_t t[8] = {0};
    u128 a0 = a.d[0], a1 = a.d[1], a2 = a.d[2], a3 = a.d[3];

    u128 c = a0 * a1; t[1] = (uint64_t)c; c >>= 64;
    c += a0 * a2; t[2] = (uint64_t)c; c >>= 64;
    c += a0 * a3; t[3] = (uint64_t)c; t[4] = (uint64_t)(c >> 64);

    c = (u128)t[3] + a1 * a2; t[3] = (uint64_t)c; c >>= 64;
    c += (u128)t[4] + a1 * a3; t[4] = (uint64_t)c; t[5] = (uint64_t)(c >> 64);

    c = (u128)t[5] + a2 * a3; t[5] = (uint64_t)c; t[6] = (uint64_t)(c >> 64);

    uint64_t carry = 0;
    for (int i = 1; i < 7; ++i) {
        uint64_t v = (t[i] << 1) | carry;
        carry = t[i] >> 63;
        t[i] = v;
    }
    t[7] = carry;

    c = (u128)t[0] + a0 * a0; t[0] = (uint64_t)c; c >>= 64;
    c += (u128)t[1]; t[1] = (uint64_t)c; c >>= 64;
    c += (u128)t[2] + a1 * a1; t[2] = (uint64_t)c; c >>= 64;
    c += (u128)t[3]; t[3] = (uint64_t)c; c >>= 64;
    c += (u128)t[4] + a2 * a2; t[4] = (uint64_t)c; c >>= 64;
    c += (u128)t[5]; t[5] = (uint64_t)c; c >>= 64;
    c += (u128)t[6] + a3 * a3; t[6] = (uint64_t)c; c >>= 64;
    t[7] += (uint64_t)c;

    u128 red_carry = 0;
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
    if (extra) {
        u128 c3 = (u128)t[0] + (u128)extra * SECP_K;
        t[0] = (uint64_t)c3; c3 >>= 64;
        c3 += t[1]; t[1] = (uint64_t)c3; c3 >>= 64;
        c3 += t[2]; t[2] = (uint64_t)c3; c3 >>= 64;
        t[3] += (uint64_t)c3;
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
#endif

static inline Fe fe_inv(const Fe& a) {
    const uint64_t exp[4] = {
        0xFFFFFFFEFFFFFC2DULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL
    };
    Fe res = {{1, 0, 0, 0}};
    Fe base = a;
    for (int i = 0; i < 4; ++i) {
        uint64_t w = exp[i];
        for (int b = 0; b < 64; ++b) {
            if (i == 3 && w == 0) break;
            if (w & 1) {
                res = fe_mul(res, base);
            }
            base = fe_mul(base, base);
            w >>= 1;
        }
    }
    return res;
}

static inline Fe fe_from_bytes(const uint8_t b[32]) {
    Fe r;
    for (int i = 0; i < 4; ++i) {
        uint64_t val = 0;
        for (int j = 0; j < 8; ++j) {
            val = (val << 8) | b[(3 - i) * 8 + j];
        }
        r.d[i] = val;
    }
    return r;
}

static inline void fe_to_bytes(const Fe& a, uint8_t b[32]) {
    for (int i = 0; i < 4; ++i) {
        uint64_t val = a.d[i];
        for (int j = 7; j >= 0; --j) {
            b[(3 - i) * 8 + j] = (uint8_t)(val & 0xFF);
            val >>= 8;
        }
    }
}

struct AffinePoint {
    Fe x;
    Fe y;
};

static const int BATCH_SIZE = 1024;

static AffinePoint G_TABLE[BATCH_SIZE];

void init_generator_table() {
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
    for (int i = 1; i <= BATCH_SIZE; ++i) {
        uint8_t priv[32] = {0};
        priv[31] = (uint8_t)(i & 0xFF);
        priv[30] = (uint8_t)((i >> 8) & 0xFF);

        secp256k1_pubkey pub;
        if (!secp256k1_ec_pubkey_create(ctx, &pub, priv)) {
            continue;
        }

        uint8_t out65[65];
        size_t len65 = 65;
        secp256k1_ec_pubkey_serialize(ctx, out65, &len65, &pub, SECP256K1_EC_UNCOMPRESSED);

        G_TABLE[i - 1].x = fe_from_bytes(out65 + 1);
        G_TABLE[i - 1].y = fe_from_bytes(out65 + 33);
    }
    secp256k1_context_destroy(ctx);
}

static inline uint32_t ror32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

static inline uint32_t rol32(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

static const uint32_t K_SHA256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

static inline __attribute__((always_inline)) void fast_sha256_into_ripemd_X(uint8_t prefix, const Fe& x, uint32_t X[16]) {
    uint32_t w[64];
    uint64_t d3 = x.d[3], d2 = x.d[2], d1 = x.d[1], d0 = x.d[0];
    w[0] = ((uint32_t)prefix << 24) | (uint32_t)(d3 >> 40);
    w[1] = (uint32_t)(d3 >> 8);
    w[2] = ((uint32_t)(d3 & 0xFF) << 24) | (uint32_t)(d2 >> 40);
    w[3] = (uint32_t)(d2 >> 8);
    w[4] = ((uint32_t)(d2 & 0xFF) << 24) | (uint32_t)(d1 >> 40);
    w[5] = (uint32_t)(d1 >> 8);
    w[6] = ((uint32_t)(d1 & 0xFF) << 24) | (uint32_t)(d0 >> 40);
    w[7] = (uint32_t)(d0 >> 8);
    w[8] = ((uint32_t)(d0 & 0xFF) << 24) | 0x00800000U;
    w[9] = 0; w[10] = 0; w[11] = 0; w[12] = 0; w[13] = 0; w[14] = 0;
    w[15] = 264;

    w[16] = w[0] + (ror32(w[1], 7) ^ ror32(w[1], 18) ^ (w[1] >> 3));
    w[17] = w[1] + (ror32(w[2], 7) ^ ror32(w[2], 18) ^ (w[2] >> 3)) + 0x00A50000U;
    w[18] = w[2] + (ror32(w[3], 7) ^ ror32(w[3], 18) ^ (w[3] >> 3)) + (ror32(w[16], 17) ^ ror32(w[16], 19) ^ (w[16] >> 10));
    w[19] = w[3] + (ror32(w[4], 7) ^ ror32(w[4], 18) ^ (w[4] >> 3)) + (ror32(w[17], 17) ^ ror32(w[17], 19) ^ (w[17] >> 10));
    w[20] = w[4] + (ror32(w[5], 7) ^ ror32(w[5], 18) ^ (w[5] >> 3)) + (ror32(w[18], 17) ^ ror32(w[18], 19) ^ (w[18] >> 10));
    w[21] = w[5] + (ror32(w[6], 7) ^ ror32(w[6], 18) ^ (w[6] >> 3)) + (ror32(w[19], 17) ^ ror32(w[19], 19) ^ (w[19] >> 10));
    w[22] = w[6] + (ror32(w[7], 7) ^ ror32(w[7], 18) ^ (w[7] >> 3)) + 264 + (ror32(w[20], 17) ^ ror32(w[20], 19) ^ (w[20] >> 10));
    w[23] = w[7] + (ror32(w[8], 7) ^ ror32(w[8], 18) ^ (w[8] >> 3)) + w[16] + (ror32(w[21], 17) ^ ror32(w[21], 19) ^ (w[21] >> 10));

    #pragma GCC unroll 40
    for (int i = 24; i < 64; ++i) {
        uint32_t s0 = ror32(w[i-15], 7) ^ ror32(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = ror32(w[i-2], 17) ^ ror32(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    uint32_t a = 0x6a09e667, b = 0xbb67ae85, c = 0x3c6ef372, d = 0xa54ff53a;
    uint32_t e = 0x510e527f, f = 0x9b05688c, g = 0x1f83d9ab, h = 0x5be0cd19;

#define SHA256_STEP(a, b, c, d, e, f, g, h, kw) do {     uint32_t S1 = ror32(e, 6) ^ ror32(e, 11) ^ ror32(e, 25);     uint32_t ch = g ^ (e & (f ^ g));     uint32_t temp1 = h + S1 + ch + (kw);     uint32_t S0 = ror32(a, 2) ^ ror32(a, 13) ^ ror32(a, 22);     uint32_t maj = (a & b) | (c & (a ^ b));     uint32_t temp2 = S0 + maj;     d += temp1;     h = temp1 + temp2; } while (0)

    #pragma GCC unroll 8
    for (int i = 0; i < 64; i += 8) {
        SHA256_STEP(a, b, c, d, e, f, g, h, K_SHA256[i] + w[i]);
        SHA256_STEP(h, a, b, c, d, e, f, g, K_SHA256[i+1] + w[i+1]);
        SHA256_STEP(g, h, a, b, c, d, e, f, K_SHA256[i+2] + w[i+2]);
        SHA256_STEP(f, g, h, a, b, c, d, e, K_SHA256[i+3] + w[i+3]);
        SHA256_STEP(e, f, g, h, a, b, c, d, K_SHA256[i+4] + w[i+4]);
        SHA256_STEP(d, e, f, g, h, a, b, c, K_SHA256[i+5] + w[i+5]);
        SHA256_STEP(c, d, e, f, g, h, a, b, K_SHA256[i+6] + w[i+6]);
        SHA256_STEP(b, c, d, e, f, g, h, a, K_SHA256[i+7] + w[i+7]);
    }
#undef SHA256_STEP

    X[0] = __builtin_bswap32(0x6a09e667 + a);
    X[1] = __builtin_bswap32(0xbb67ae85 + b);
    X[2] = __builtin_bswap32(0x3c6ef372 + c);
    X[3] = __builtin_bswap32(0xa54ff53a + d);
    X[4] = __builtin_bswap32(0x510e527f + e);
    X[5] = __builtin_bswap32(0x9b05688c + f);
    X[6] = __builtin_bswap32(0x1f83d9ab + g);
    X[7] = __builtin_bswap32(0x5be0cd19 + h);
    X[8] = 0x00000080U;
    X[9] = 0; X[10] = 0; X[11] = 0; X[12] = 0; X[13] = 0;
    X[14] = 256;
    X[15] = 0;
}

static const uint8_t r_left[80] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
    3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
    1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
    4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13
};
static const uint8_t s_left[80] = {
    11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
    7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
    11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
    11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
    9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6
};
static const uint8_t r_right[80] = {
    5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
    6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
    15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
    8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
    12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11
};
static const uint8_t s_right[80] = {
    8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
    9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
    9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
    15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
    8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11
};

static inline __attribute__((always_inline)) void fast_ripemd160_32(const uint32_t X[16], uint32_t out_h[5]) {
    uint32_t A = 0x67452301, B = 0xEFCDAB89, C = 0x98BADCFE, D = 0x10325476, E = 0xC3D2E1F0;
    uint32_t Ap = A, Bp = B, Cp = C, Dp = D, Ep = E;

    #pragma GCC unroll 16
    for (int j = 0; j < 16; ++j) {
        uint32_t f = B ^ C ^ D;
        uint32_t fp = Bp ^ (Cp | ~Dp);
        uint32_t T = rol32(A + f + X[r_left[j]], s_left[j]) + E;
        A = E; E = D; D = rol32(C, 10); C = B; B = T;
        uint32_t Tp = rol32(Ap + fp + X[r_right[j]] + 0x50A28BE6U, s_right[j]) + Ep;
        Ap = Ep; Ep = Dp; Dp = rol32(Cp, 10); Cp = Bp; Bp = Tp;
    }

    #pragma GCC unroll 16
    for (int j = 16; j < 32; ++j) {
        uint32_t f = D ^ (B & (C ^ D));
        uint32_t fp = Cp ^ (Dp & (Bp ^ Cp));
        uint32_t T = rol32(A + f + X[r_left[j]] + 0x5A827999U, s_left[j]) + E;
        A = E; E = D; D = rol32(C, 10); C = B; B = T;
        uint32_t Tp = rol32(Ap + fp + X[r_right[j]] + 0x5C4DD124U, s_right[j]) + Ep;
        Ap = Ep; Ep = Dp; Dp = rol32(Cp, 10); Cp = Bp; Bp = Tp;
    }

    #pragma GCC unroll 16
    for (int j = 32; j < 48; ++j) {
        uint32_t f = (B | ~C) ^ D;
        uint32_t fp = (Bp | ~Cp) ^ Dp;
        uint32_t T = rol32(A + f + X[r_left[j]] + 0x6ED9EBA1U, s_left[j]) + E;
        A = E; E = D; D = rol32(C, 10); C = B; B = T;
        uint32_t Tp = rol32(Ap + fp + X[r_right[j]] + 0x6D703EF3U, s_right[j]) + Ep;
        Ap = Ep; Ep = Dp; Dp = rol32(Cp, 10); Cp = Bp; Bp = Tp;
    }

    #pragma GCC unroll 16
    for (int j = 48; j < 64; ++j) {
        uint32_t f = C ^ (D & (B ^ C));
        uint32_t fp = Dp ^ (Bp & (Cp ^ Dp));
        uint32_t T = rol32(A + f + X[r_left[j]] + 0x8F1BBCDCU, s_left[j]) + E;
        A = E; E = D; D = rol32(C, 10); C = B; B = T;
        uint32_t Tp = rol32(Ap + fp + X[r_right[j]] + 0x7A6D76E9U, s_right[j]) + Ep;
        Ap = Ep; Ep = Dp; Dp = rol32(Cp, 10); Cp = Bp; Bp = Tp;
    }

    #pragma GCC unroll 16
    for (int j = 64; j < 80; ++j) {
        uint32_t f = B ^ (C | ~D);
        uint32_t fp = Bp ^ Cp ^ Dp;
        uint32_t T = rol32(A + f + X[r_left[j]] + 0xA953FD4EU, s_left[j]) + E;
        A = E; E = D; D = rol32(C, 10); C = B; B = T;
        uint32_t Tp = rol32(Ap + fp + X[r_right[j]], s_right[j]) + Ep;
        Ap = Ep; Ep = Dp; Dp = rol32(Cp, 10); Cp = Bp; Bp = Tp;
    }

    out_h[0] = 0xEFCDAB89 + C + Dp;
    out_h[1] = 0x98BADCFE + D + Ep;
    out_h[2] = 0x10325476 + E + Ap;
    out_h[3] = 0xC3D2E1F0 + A + Bp;
    out_h[4] = 0x67452301 + B + Cp;
}

#if defined(__AVX2__)
#define AVX2_ROR(x, n) _mm256_or_si256(_mm256_srli_epi32(x, n), _mm256_slli_epi32(x, 32 - (n)))
#define AVX2_ROL(x, n) _mm256_or_si256(_mm256_slli_epi32(x, n), _mm256_srli_epi32(x, 32 - (n)))

static inline __attribute__((always_inline)) void avx2_sha256_8way(
    const __m256i W_in[16],
    __m256i X_out[16]
) {
    __m256i W[64];
    for (int i = 0; i < 16; ++i) W[i] = W_in[i];

    #pragma GCC unroll 48
    for (int i = 16; i < 64; ++i) {
        __m256i s0 = _mm256_xor_si256(AVX2_ROR(W[i-15], 7), _mm256_xor_si256(AVX2_ROR(W[i-15], 18), _mm256_srli_epi32(W[i-15], 3)));
        __m256i s1 = _mm256_xor_si256(AVX2_ROR(W[i-2], 17), _mm256_xor_si256(AVX2_ROR(W[i-2], 19), _mm256_srli_epi32(W[i-2], 10)));
        W[i] = _mm256_add_epi32(_mm256_add_epi32(W[i-16], s0), _mm256_add_epi32(W[i-7], s1));
    }

    __m256i a = _mm256_set1_epi32(0x6a09e667);
    __m256i b = _mm256_set1_epi32(0xbb67ae85);
    __m256i c = _mm256_set1_epi32(0x3c6ef372);
    __m256i d = _mm256_set1_epi32(0xa54ff53a);
    __m256i e = _mm256_set1_epi32(0x510e527f);
    __m256i f = _mm256_set1_epi32(0x9b05688c);
    __m256i g = _mm256_set1_epi32(0x1f83d9ab);
    __m256i h = _mm256_set1_epi32(0x5be0cd19);

    __m256i KW[64];
    #pragma GCC unroll 64
    for (int i = 0; i < 64; ++i) {
        KW[i] = _mm256_add_epi32(_mm256_set1_epi32(K_SHA256[i]), W[i]);
    }

#define AVX2_SHA256_STEP(a, b, c, d, e, f, g, h, kw) do { \
    __m256i S1 = _mm256_xor_si256(AVX2_ROR(e, 6), _mm256_xor_si256(AVX2_ROR(e, 11), AVX2_ROR(e, 25))); \
    __m256i ch = _mm256_xor_si256(g, _mm256_and_si256(e, _mm256_xor_si256(f, g))); \
    __m256i temp1 = _mm256_add_epi32(_mm256_add_epi32(h, S1), _mm256_add_epi32(ch, kw)); \
    __m256i S0 = _mm256_xor_si256(AVX2_ROR(a, 2), _mm256_xor_si256(AVX2_ROR(a, 13), AVX2_ROR(a, 22))); \
    __m256i maj = _mm256_or_si256(_mm256_and_si256(a, b), _mm256_and_si256(c, _mm256_xor_si256(a, b))); \
    __m256i temp2 = _mm256_add_epi32(S0, maj); \
    d = _mm256_add_epi32(d, temp1); \
    h = _mm256_add_epi32(temp1, temp2); \
} while (0)

    #pragma GCC unroll 8
    for (int i = 0; i < 64; i += 8) {
        AVX2_SHA256_STEP(a, b, c, d, e, f, g, h, KW[i]);
        AVX2_SHA256_STEP(h, a, b, c, d, e, f, g, KW[i+1]);
        AVX2_SHA256_STEP(g, h, a, b, c, d, e, f, KW[i+2]);
        AVX2_SHA256_STEP(f, g, h, a, b, c, d, e, KW[i+3]);
        AVX2_SHA256_STEP(e, f, g, h, a, b, c, d, KW[i+4]);
        AVX2_SHA256_STEP(d, e, f, g, h, a, b, c, KW[i+5]);
        AVX2_SHA256_STEP(c, d, e, f, g, h, a, b, KW[i+6]);
        AVX2_SHA256_STEP(b, c, d, e, f, g, h, a, KW[i+7]);
    }
#undef AVX2_SHA256_STEP

    const __m256i bswap_mask = _mm256_set_epi8(
        12, 13, 14, 15,  8,  9, 10, 11,  4,  5,  6,  7,  0,  1,  2,  3,
        12, 13, 14, 15,  8,  9, 10, 11,  4,  5,  6,  7,  0,  1,  2,  3
    );

    X_out[0] = _mm256_shuffle_epi8(_mm256_add_epi32(a, _mm256_set1_epi32(0x6a09e667)), bswap_mask);
    X_out[1] = _mm256_shuffle_epi8(_mm256_add_epi32(b, _mm256_set1_epi32(0xbb67ae85)), bswap_mask);
    X_out[2] = _mm256_shuffle_epi8(_mm256_add_epi32(c, _mm256_set1_epi32(0x3c6ef372)), bswap_mask);
    X_out[3] = _mm256_shuffle_epi8(_mm256_add_epi32(d, _mm256_set1_epi32(0xa54ff53a)), bswap_mask);
    X_out[4] = _mm256_shuffle_epi8(_mm256_add_epi32(e, _mm256_set1_epi32(0x510e527f)), bswap_mask);
    X_out[5] = _mm256_shuffle_epi8(_mm256_add_epi32(f, _mm256_set1_epi32(0x9b05688c)), bswap_mask);
    X_out[6] = _mm256_shuffle_epi8(_mm256_add_epi32(g, _mm256_set1_epi32(0x1f83d9ab)), bswap_mask);
    X_out[7] = _mm256_shuffle_epi8(_mm256_add_epi32(h, _mm256_set1_epi32(0x5be0cd19)), bswap_mask);
    X_out[8] = _mm256_set1_epi32(0x00000080U);
    X_out[9] = _mm256_setzero_si256();
    X_out[10] = _mm256_setzero_si256();
    X_out[11] = _mm256_setzero_si256();
    X_out[12] = _mm256_setzero_si256();
    X_out[13] = _mm256_setzero_si256();
    X_out[14] = _mm256_set1_epi32(256);
    X_out[15] = _mm256_setzero_si256();
}

static inline __attribute__((always_inline)) void avx2_ripemd160_8way(
    const __m256i X[16],
    __m256i out_h[5]
) {
    __m256i A = _mm256_set1_epi32(0x67452301);
    __m256i B = _mm256_set1_epi32(0xEFCDAB89);
    __m256i C = _mm256_set1_epi32(0x98BADCFE);
    __m256i D = _mm256_set1_epi32(0x10325476);
    __m256i E = _mm256_set1_epi32(0xC3D2E1F0);
    __m256i Ap = A, Bp = B, Cp = C, Dp = D, Ep = E;

    #pragma GCC unroll 16
    for (int j = 0; j < 16; ++j) {
        __m256i f = _mm256_xor_si256(B, _mm256_xor_si256(C, D));
        __m256i fp = _mm256_xor_si256(Bp, _mm256_or_si256(Cp, _mm256_xor_si256(Dp, _mm256_set1_epi32(-1))));
        __m256i T = _mm256_add_epi32(AVX2_ROL(_mm256_add_epi32(_mm256_add_epi32(A, f), X[r_left[j]]), s_left[j]), E);
        A = E; E = D; D = AVX2_ROL(C, 10); C = B; B = T;
        __m256i Tp = _mm256_add_epi32(AVX2_ROL(_mm256_add_epi32(_mm256_add_epi32(Ap, fp), _mm256_add_epi32(X[r_right[j]], _mm256_set1_epi32(0x50A28BE6U))), s_right[j]), Ep);
        Ap = Ep; Ep = Dp; Dp = AVX2_ROL(Cp, 10); Cp = Bp; Bp = Tp;
    }

    #pragma GCC unroll 16
    for (int j = 16; j < 32; ++j) {
        __m256i f = _mm256_xor_si256(D, _mm256_and_si256(B, _mm256_xor_si256(C, D)));
        __m256i fp = _mm256_xor_si256(Cp, _mm256_and_si256(Dp, _mm256_xor_si256(Bp, Cp)));
        __m256i T = _mm256_add_epi32(AVX2_ROL(_mm256_add_epi32(_mm256_add_epi32(A, f), _mm256_add_epi32(X[r_left[j]], _mm256_set1_epi32(0x5A827999U))), s_left[j]), E);
        A = E; E = D; D = AVX2_ROL(C, 10); C = B; B = T;
        __m256i Tp = _mm256_add_epi32(AVX2_ROL(_mm256_add_epi32(_mm256_add_epi32(Ap, fp), _mm256_add_epi32(X[r_right[j]], _mm256_set1_epi32(0x5C4DD124U))), s_right[j]), Ep);
        Ap = Ep; Ep = Dp; Dp = AVX2_ROL(Cp, 10); Cp = Bp; Bp = Tp;
    }

    #pragma GCC unroll 16
    for (int j = 32; j < 48; ++j) {
        __m256i f = _mm256_xor_si256(_mm256_or_si256(B, _mm256_xor_si256(C, _mm256_set1_epi32(-1))), D);
        __m256i fp = _mm256_xor_si256(_mm256_or_si256(Bp, _mm256_xor_si256(Cp, _mm256_set1_epi32(-1))), Dp);
        __m256i T = _mm256_add_epi32(AVX2_ROL(_mm256_add_epi32(_mm256_add_epi32(A, f), _mm256_add_epi32(X[r_left[j]], _mm256_set1_epi32(0x6ED9EBA1U))), s_left[j]), E);
        A = E; E = D; D = AVX2_ROL(C, 10); C = B; B = T;
        __m256i Tp = _mm256_add_epi32(AVX2_ROL(_mm256_add_epi32(_mm256_add_epi32(Ap, fp), _mm256_add_epi32(X[r_right[j]], _mm256_set1_epi32(0x6D703EF3U))), s_right[j]), Ep);
        Ap = Ep; Ep = Dp; Dp = AVX2_ROL(Cp, 10); Cp = Bp; Bp = Tp;
    }

    #pragma GCC unroll 16
    for (int j = 48; j < 64; ++j) {
        __m256i f = _mm256_xor_si256(C, _mm256_and_si256(D, _mm256_xor_si256(B, C)));
        __m256i fp = _mm256_xor_si256(Dp, _mm256_and_si256(Bp, _mm256_xor_si256(Cp, Dp)));
        __m256i T = _mm256_add_epi32(AVX2_ROL(_mm256_add_epi32(_mm256_add_epi32(A, f), _mm256_add_epi32(X[r_left[j]], _mm256_set1_epi32(0x8F1BBCDCU))), s_left[j]), E);
        A = E; E = D; D = AVX2_ROL(C, 10); C = B; B = T;
        __m256i Tp = _mm256_add_epi32(AVX2_ROL(_mm256_add_epi32(_mm256_add_epi32(Ap, fp), _mm256_add_epi32(X[r_right[j]], _mm256_set1_epi32(0x7A6D76E9U))), s_right[j]), Ep);
        Ap = Ep; Ep = Dp; Dp = AVX2_ROL(Cp, 10); Cp = Bp; Bp = Tp;
    }

    #pragma GCC unroll 16
    for (int j = 64; j < 80; ++j) {
        __m256i f = _mm256_xor_si256(B, _mm256_or_si256(C, _mm256_xor_si256(D, _mm256_set1_epi32(-1))));
        __m256i fp = _mm256_xor_si256(Bp, _mm256_xor_si256(Cp, Dp));
        __m256i T = _mm256_add_epi32(AVX2_ROL(_mm256_add_epi32(_mm256_add_epi32(A, f), _mm256_add_epi32(X[r_left[j]], _mm256_set1_epi32(0xA953FD4EU))), s_left[j]), E);
        A = E; E = D; D = AVX2_ROL(C, 10); C = B; B = T;
        __m256i Tp = _mm256_add_epi32(AVX2_ROL(_mm256_add_epi32(_mm256_add_epi32(Ap, fp), X[r_right[j]]), s_right[j]), Ep);
        Ap = Ep; Ep = Dp; Dp = AVX2_ROL(Cp, 10); Cp = Bp; Bp = Tp;
    }

    out_h[0] = _mm256_add_epi32(_mm256_add_epi32(_mm256_set1_epi32(0xEFCDAB89), C), Dp);
    out_h[1] = _mm256_add_epi32(_mm256_add_epi32(_mm256_set1_epi32(0x98BADCFE), D), Ep);
    out_h[2] = _mm256_add_epi32(_mm256_add_epi32(_mm256_set1_epi32(0x10325476), E), Ap);
    out_h[3] = _mm256_add_epi32(_mm256_add_epi32(_mm256_set1_epi32(0xC3D2E1F0), A), Bp);
    out_h[4] = _mm256_add_epi32(_mm256_add_epi32(_mm256_set1_epi32(0x67452301), B), Cp);
}
#endif

static const char* B58_CHARS = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

bool b58check_decode_hash160(const std::string& addr, uint8_t hash160_out[20]) {
    std::vector<uint8_t> bytes;
    for (char c : addr) {
        const char* p = strchr(B58_CHARS, c);
        if (!p) return false;
        int val = p - B58_CHARS;
        int carry = val;
        for (size_t i = 0; i < bytes.size(); ++i) {
            int cur = bytes[i] * 58 + carry;
            bytes[i] = cur & 0xFF;
            carry = cur >> 8;
        }
        while (carry > 0) {
            bytes.push_back(carry & 0xFF);
            carry >>= 8;
        }
    }
    for (char c : addr) {
        if (c == '1') bytes.push_back(0);
        else break;
    }
    std::reverse(bytes.begin(), bytes.end());
    if (bytes.size() != 25) return false;

    uint8_t sha1[32], sha2[32];
    SHA256(bytes.data(), 21, sha1);
    SHA256(sha1, 32, sha2);
    if (memcmp(sha2, bytes.data() + 21, 4) != 0) return false;

    memcpy(hash160_out, bytes.data() + 1, 20);
    return true;
}

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

std::string http_get(const std::string& url) {
    CURL* curl = curl_easy_init();
    if (!curl) return "";
    std::string readBuffer;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 20L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "worker/1.0");
    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    return (res == CURLE_OK) ? readBuffer : "";
}

bool http_post(const std::string& url, const std::string& json_data, std::string* response_out = nullptr) {
    CURL* curl = curl_easy_init();
    if (!curl) return false;
    std::string responseBuffer;
    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_data.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseBuffer);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 20L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "worker/1.0");

    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    if (res == CURLE_OK && response_out) *response_out = responseBuffer;
    return (res == CURLE_OK);
}

std::string json_get_string(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return "";
    pos += pattern.length();
    while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    if (pos >= json.length()) return "";

    if (json[pos] == '"') {
        pos++;
        size_t end = json.find('"', pos);
        if (end == std::string::npos) return "";
        return json.substr(pos, end - pos);
    } else {
        size_t end = pos;
        while (end < json.length() && json[end] != ',' && json[end] != '}' && json[end] != '\n') end++;
        return json.substr(pos, end - pos);
    }
}

struct RangeInfo {
    int puzzle;
    std::string user;
    int64_t block;
    int range_idx;
    u256 start;
    u256 end;
    uint64_t range_size;
    std::string target_address;
    u256 lower;
    u256 total;
};

bool parse_range_json(const std::string& json, RangeInfo& rng) {
    if (json.empty() || json.find("\"start\"") == std::string::npos) return false;
    std::string s_p = json_get_string(json, "puzzle");
    rng.puzzle = s_p.empty() ? 71 : std::stoi(s_p);
    rng.user = json_get_string(json, "user");
    std::string s_b = json_get_string(json, "block");
    rng.block = s_b.empty() ? 0 : std::stoll(s_b);
    std::string s_r = json_get_string(json, "range_idx");
    rng.range_idx = s_r.empty() ? 0 : std::stoi(s_r);
    rng.start = parse_u256(json_get_string(json, "start"));
    rng.end = parse_u256(json_get_string(json, "end"));
    std::string s_sz = json_get_string(json, "range_size");
    rng.range_size = s_sz.empty() ? 0 : std::stoull(s_sz);
    rng.target_address = json_get_string(json, "target_address");
    rng.lower = parse_u256(json_get_string(json, "lower"));
    rng.total = parse_u256(json_get_string(json, "total"));
    if (rng.end <= rng.start && rng.range_size > 0) {
        rng.end = rng.start + rng.range_size;
    }
    return (!rng.target_address.empty() && rng.end > rng.start);
}

void scan_worker_montgomery(
    u256 range_start,
    std::atomic<uint64_t>& work_offset,
    uint64_t total_keys,
    uint64_t slice_size,
    const uint8_t target_hash160[20],
    const uint64_t target_h64_first,
    std::atomic<bool>& found_flag,
    u256& found_key,
    std::mutex& found_mtx,
    std::atomic<uint64_t>& checked_counter
) {
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
    if (!ctx) return;

    Fe dx[BATCH_SIZE];
    Fe cum[BATCH_SIZE + 1];
    Fe inv_dx[BATCH_SIZE];

    uint32_t target_w[5];
    for (int i = 0; i < 5; ++i) {
        target_w[i] = (uint32_t)target_hash160[4*i] |
                      ((uint32_t)target_hash160[4*i + 1] << 8) |
                      ((uint32_t)target_hash160[4*i + 2] << 16) |
                      ((uint32_t)target_hash160[4*i + 3] << 24);
    }

    uint64_t local_counter = 0;
    bool has_cur_base = false;
    AffinePoint cur_base;
    u256 cur_k = 0;

    while (!found_flag.load(std::memory_order_relaxed) && g_running.load(std::memory_order_relaxed)) {
        uint64_t off = work_offset.fetch_add(slice_size, std::memory_order_relaxed);
        if (off >= total_keys) break;
        uint64_t count_in_slice = std::min(slice_size, total_keys - off);
        u256 slice_start = range_start + off;
        u256 slice_end = slice_start + count_in_slice;

        u256 base_k = (!slice_start.is_zero()) ? (slice_start - 1) : u256(0);
        if (!has_cur_base || cur_k != base_k) {
            uint8_t priv_start[32] = {0};
            base_k.to_bytes_be(priv_start);
            secp256k1_pubkey start_pub;
            if (!secp256k1_ec_pubkey_create(ctx, &start_pub, priv_start)) {
                continue;
            }
            uint8_t out65[65];
            size_t len65 = 65;
            secp256k1_ec_pubkey_serialize(ctx, out65, &len65, &start_pub, SECP256K1_EC_UNCOMPRESSED);
            cur_base.x = fe_from_bytes(out65 + 1);
            cur_base.y = fe_from_bytes(out65 + 33);
            has_cur_base = true;
            cur_k = base_k;
        }

        while (cur_k < slice_end - 1 && !found_flag.load(std::memory_order_relaxed) && g_running.load(std::memory_order_relaxed)) {
            u256 rem = (slice_end - 1 - cur_k);
            uint64_t remaining = (rem.high == 0) ? (uint64_t)rem.low : 0xFFFFFFFFFFFFFFFFULL;
            int current_batch = (int)std::min((uint64_t)BATCH_SIZE, remaining);

            cum[0] = {{1, 0, 0, 0}};
            for (int i = 0; i < current_batch; ++i) {
                dx[i] = fe_sub(G_TABLE[i].x, cur_base.x);
                cum[i + 1] = fe_mul(cum[i], dx[i]);
            }

            Fe u = fe_inv(cum[current_batch]);

            for (int i = current_batch - 1; i >= 0; --i) {
                inv_dx[i] = fe_mul(u, cum[i]);
                u = fe_mul(u, dx[i]);
            }

            AffinePoint next_base;
#if defined(__AVX2__)
            int i = 0;
            for (; i + 8 <= current_batch; i += 8) {
                Fe xi[8], yi[8];
                for (int k = 0; k < 8; ++k) {
                    Fe dy_k = fe_sub(G_TABLE[i + k].y, cur_base.y);
                    Fe slope = fe_mul(dy_k, inv_dx[i + k]);
                    Fe slope_sqr = fe_sqr(slope);
                    xi[k] = fe_sub(fe_sub(slope_sqr, cur_base.x), G_TABLE[i + k].x);
                    yi[k] = fe_sub(fe_mul(slope, fe_sub(cur_base.x, xi[k])), cur_base.y);
                }

                if (i + 8 == current_batch) {
                    next_base.x = xi[7];
                    next_base.y = yi[7];
                }

                alignas(32) uint32_t w0[8], w1[8], w2[8], w3[8], w4[8], w5[8], w6[8], w7[8], w8[8];
                for (int k = 0; k < 8; ++k) {
                    uint8_t prefix = (yi[k].d[0] & 1) ? 0x03 : 0x02;
                    uint64_t d3 = xi[k].d[3], d2 = xi[k].d[2], d1 = xi[k].d[1], d0 = xi[k].d[0];
                    w0[k] = ((uint32_t)prefix << 24) | (uint32_t)(d3 >> 40);
                    w1[k] = (uint32_t)(d3 >> 8);
                    w2[k] = ((uint32_t)(d3 & 0xFF) << 24) | (uint32_t)(d2 >> 40);
                    w3[k] = (uint32_t)(d2 >> 8);
                    w4[k] = ((uint32_t)(d2 & 0xFF) << 24) | (uint32_t)(d1 >> 40);
                    w5[k] = (uint32_t)(d1 >> 8);
                    w6[k] = ((uint32_t)(d1 & 0xFF) << 24) | (uint32_t)(d0 >> 40);
                    w7[k] = (uint32_t)(d0 >> 8);
                    w8[k] = ((uint32_t)(d0 & 0xFF) << 24) | 0x00800000U;
                }

                __m256i W[16];
                W[0] = _mm256_load_si256((const __m256i*)w0);
                W[1] = _mm256_load_si256((const __m256i*)w1);
                W[2] = _mm256_load_si256((const __m256i*)w2);
                W[3] = _mm256_load_si256((const __m256i*)w3);
                W[4] = _mm256_load_si256((const __m256i*)w4);
                W[5] = _mm256_load_si256((const __m256i*)w5);
                W[6] = _mm256_load_si256((const __m256i*)w6);
                W[7] = _mm256_load_si256((const __m256i*)w7);
                W[8] = _mm256_load_si256((const __m256i*)w8);
                W[9] = _mm256_setzero_si256();
                W[10] = _mm256_setzero_si256();
                W[11] = _mm256_setzero_si256();
                W[12] = _mm256_setzero_si256();
                W[13] = _mm256_setzero_si256();
                W[14] = _mm256_setzero_si256();
                W[15] = _mm256_set1_epi32(264);

                __m256i X[16];
                avx2_sha256_8way(W, X);

                __m256i H[5];
                avx2_ripemd160_8way(X, H);

                __m256i match0 = _mm256_cmpeq_epi32(H[0], _mm256_set1_epi32(target_w[0]));
                __m256i match1 = _mm256_cmpeq_epi32(H[1], _mm256_set1_epi32(target_w[1]));
                int mask = _mm256_movemask_epi8(_mm256_and_si256(match0, match1));
                if (__builtin_expect(mask != 0, 0)) {
                    alignas(32) uint32_t h_lanes[5][8];
                    for (int m = 0; m < 5; ++m) {
                        _mm256_store_si256((__m256i*)h_lanes[m], H[m]);
                    }
                    for (int k = 0; k < 8; ++k) {
                        if (h_lanes[0][k] == target_w[0] && h_lanes[1][k] == target_w[1] &&
                            h_lanes[2][k] == target_w[2] && h_lanes[3][k] == target_w[3] &&
                            h_lanes[4][k] == target_w[4]) {
                            std::lock_guard<std::mutex> lk(found_mtx);
                            found_key = cur_k + (uint64_t)(i + k + 1);
                            found_flag.store(true, std::memory_order_release);
                            break;
                        }
                    }
                    if (found_flag.load(std::memory_order_relaxed)) break;
                }
            }
            for (; i < current_batch; ++i) {
                Fe dy_i = fe_sub(G_TABLE[i].y, cur_base.y);
                Fe slope = fe_mul(dy_i, inv_dx[i]);
                Fe slope_sqr = fe_sqr(slope);
                Fe xi = fe_sub(fe_sub(slope_sqr, cur_base.x), G_TABLE[i].x);
                Fe yi = fe_sub(fe_mul(slope, fe_sub(cur_base.x, xi)), cur_base.y);

                if (i == current_batch - 1) {
                    next_base.x = xi;
                    next_base.y = yi;
                }

                uint8_t prefix = (yi.d[0] & 1) ? 0x03 : 0x02;
                uint32_t X[16];
                fast_sha256_into_ripemd_X(prefix, xi, X);

                uint32_t h[5];
                fast_ripemd160_32(X, h);

                uint64_t cur_h64 = (uint64_t)h[0] | ((uint64_t)h[1] << 32);
                if (__builtin_expect(cur_h64 == target_h64_first, 0)) {
                    if (h[2] == target_w[2] && h[3] == target_w[3] && h[4] == target_w[4]) {
                        std::lock_guard<std::mutex> lk(found_mtx);
                        found_key = cur_k + (uint64_t)(i + 1);
                        found_flag.store(true, std::memory_order_release);
                        break;
                    }
                }
            }
#else
            for (int i = 0; i < current_batch; ++i) {
                Fe dy_i = fe_sub(G_TABLE[i].y, cur_base.y);
                Fe slope = fe_mul(dy_i, inv_dx[i]);
                Fe slope_sqr = fe_sqr(slope);
                Fe xi = fe_sub(fe_sub(slope_sqr, cur_base.x), G_TABLE[i].x);
                Fe yi = fe_sub(fe_mul(slope, fe_sub(cur_base.x, xi)), cur_base.y);

                if (i == current_batch - 1) {
                    next_base.x = xi;
                    next_base.y = yi;
                }

                uint8_t prefix = (yi.d[0] & 1) ? 0x03 : 0x02;
                uint32_t X[16];
                fast_sha256_into_ripemd_X(prefix, xi, X);

                uint32_t h[5];
                fast_ripemd160_32(X, h);

                uint64_t cur_h64 = (uint64_t)h[0] | ((uint64_t)h[1] << 32);
                if (__builtin_expect(cur_h64 == target_h64_first, 0)) {
                    if (h[2] == target_w[2] && h[3] == target_w[3] && h[4] == target_w[4]) {
                        std::lock_guard<std::mutex> lk(found_mtx);
                        found_key = cur_k + (uint64_t)(i + 1);
                        found_flag.store(true, std::memory_order_release);
                        break;
                    }
                }
            }
#endif

            if (found_flag.load(std::memory_order_relaxed)) break;

            cur_base = next_base;
            cur_k += (uint64_t)current_batch;
            local_counter += current_batch;
        }
    }

    checked_counter.fetch_add(local_counter, std::memory_order_relaxed);

    secp256k1_context_destroy(ctx);
}

int main(int argc, char* argv[]) {
    signal(SIGINT, sigint_handler);
    signal(SIGTERM, sigint_handler);
    curl_global_init(CURL_GLOBAL_ALL);

    unsigned int hw = std::thread::hardware_concurrency();
    int threads = 1;
    bool no_limit = false;
    std::string api_base = "http://65.20.91.208/puzzle_server.php";
    std::string custom_user = "";
    int req_puzzle = 71;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            threads = std::max(1, std::atoi(argv[++i]));
        } else if ((arg == "-s" || arg == "--server") && i + 1 < argc) {
            api_base = argv[++i];
        } else if ((arg == "-u" || arg == "--user") && i + 1 < argc) {
            custom_user = argv[++i];
        } else if ((arg == "-p" || arg == "--puzzle") && i + 1 < argc) {
            req_puzzle = std::atoi(argv[++i]);
        } else if (arg == "-d" || arg == "--double") {
            threads = 2;
        } else if (arg == "--fast") {
            threads = (hw > 0) ? (int)hw : 4;
        } else if (arg == "--no-limit" || arg == "-nl" || arg == "--infinite") {
            no_limit = true;
        }
    }

    const int MAX_RANGES = 5;
    int completed_ranges = 0;

    init_generator_table();

    while (g_running.load() && (no_limit || completed_ranges < MAX_RANGES)) {
        std::string url = api_base + "?action=range&puzzle=" + std::to_string(req_puzzle);
        if (!custom_user.empty()) {
            url += "&user=" + custom_user;
        }

        std::string resp = http_get(url);

        RangeInfo rng;
        if (!parse_range_json(resp, rng)) {
            std::this_thread::sleep_for(std::chrono::seconds(3));
            continue;
        }

        int current_puzzle = (rng.puzzle > 0) ? rng.puzzle : req_puzzle;
        std::string current_user = !custom_user.empty() 
            ? custom_user 
            : (rng.user.empty() ? ("user-" + std::to_string(rng.block) + "-" + std::to_string(rng.range_idx)) : rng.user);

        uint8_t target_h160[20];
        if (!b58check_decode_hash160(rng.target_address, target_h160)) {
            std::this_thread::sleep_for(std::chrono::seconds(3));
            continue;
        }
        uint64_t target_h64 = *(const uint64_t*)target_h160;

        u256 total_keys = (rng.range_size > 0) ? u256(rng.range_size) : (rng.end - rng.start);
        uint64_t total_keys_count = (total_keys.high == 0) ? (uint64_t)total_keys.low : (uint64_t)rng.range_size;
        std::atomic<uint64_t> work_offset(0);
        uint64_t slice_size = 524288;

        std::atomic<bool> found_flag(false);
        u256 found_key = 0;
        std::mutex found_mtx;
        std::atomic<uint64_t> checked_counter(0);

        std::vector<std::thread> pool;
        auto t_start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < threads; ++i) {
            pool.emplace_back(scan_worker_montgomery, rng.start, std::ref(work_offset), total_keys_count, slice_size, target_h160, target_h64,
                              std::ref(found_flag), std::ref(found_key),
                              std::ref(found_mtx), std::ref(checked_counter));
        }

        for (auto& th : pool) if (th.joinable()) th.join();

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t_end - t_start).count();
        if (elapsed <= 0.0) elapsed = 0.001;

        bool hit = found_flag.load();
        uint64_t checked = checked_counter.load();
        if (!hit && g_running.load() && checked < total_keys_count) {
            checked = total_keys_count;
        }
        double speed = (double)checked / elapsed;

        if (!hit && !g_running.load()) {
            break;
        }

        std::stringstream json;
        json << "{\"action\":\"result\",\"puzzle\":" << current_puzzle
             << ",\"block\":" << rng.block
             << ",\"range_idx\":" << rng.range_idx
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

        if (hit) {
            break;
        }
    }

    curl_global_cleanup();
    return 0;
}
