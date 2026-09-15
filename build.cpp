#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <atomic>
#include <cstring>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <thread>
#include <mutex>

#if defined(__CUDACC__)
  #include <cuda_runtime.h>
  #define CUDA_HOSTDEV __host__ __device__
  #define CUDA_DEV __device__
  #define CUDA_GLOBAL __global__
  #define CUDA_INLINE __forceinline__
  #define CUDA_CONST __device__ __constant__ const
#else
  #define CUDA_HOSTDEV
  #define CUDA_DEV static
  #define CUDA_GLOBAL
  #define CUDA_INLINE inline __attribute__((always_inline))
  #define CUDA_CONST static const
#endif

#include <curl/curl.h>
#include <openssl/sha.h>
#include <openssl/ripemd.h>

#if (defined(__x86_64__) || defined(_M_X64)) && !defined(__CUDACC__)
#include <immintrin.h>
#if defined(__GNUC__) || defined(__clang__)
#include <x86intrin.h>
#endif
#endif

static std::atomic<bool> g_running(true);

void sigint_handler(int signum) {
    (void)signum;
    g_running = false;
}

typedef __uint128_t u128;

struct u256 {
    __uint128_t low;
    __uint128_t high;

    CUDA_HOSTDEV u256() : low(0), high(0) {}
    CUDA_HOSTDEV u256(uint64_t v) : low(v), high(0) {}
    CUDA_HOSTDEV u256(__uint128_t l, __uint128_t h) : low(l), high(h) {}

    CUDA_HOSTDEV inline bool is_zero() const { return (low | high) == 0; }
};

CUDA_HOSTDEV inline bool operator==(const u256& a, const u256& b) { return a.low == b.low && a.high == b.high; }
CUDA_HOSTDEV inline bool operator!=(const u256& a, const u256& b) { return !(a == b); }
CUDA_HOSTDEV inline bool operator<(const u256& a, const u256& b) {
    if (a.high != b.high) return a.high < b.high;
    return a.low < b.low;
}
CUDA_HOSTDEV inline bool operator<=(const u256& a, const u256& b) {
    return (a < b) || (a == b);
}
CUDA_HOSTDEV inline bool operator>(const u256& a, const u256& b) {
    return b < a;
}
CUDA_HOSTDEV inline bool operator>=(const u256& a, const u256& b) {
    return !(a < b);
}
CUDA_HOSTDEV inline u256 operator+(const u256& a, const u256& b) {
    u256 r;
    r.low = a.low + b.low;
    r.high = a.high + b.high + (r.low < a.low ? 1 : 0);
    return r;
}
CUDA_HOSTDEV inline u256 operator+(const u256& a, uint64_t b) {
    u256 r;
    r.low = a.low + b;
    r.high = a.high + (r.low < a.low ? 1 : 0);
    return r;
}
CUDA_HOSTDEV inline u256 operator-(const u256& a, const u256& b) {
    u256 r;
    r.low = a.low - b.low;
    r.high = a.high - b.high - (a.low < b.low ? 1 : 0);
    return r;
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

std::string u256_to_hex64(const u256& v) {
    char buf[65];
    snprintf(buf, sizeof(buf), "%016llx%016llx%016llx%016llx",
             (unsigned long long)(v.high >> 64),
             (unsigned long long)(v.high & 0xFFFFFFFFFFFFFFFFULL),
             (unsigned long long)(v.low >> 64),
             (unsigned long long)(v.low & 0xFFFFFFFFFFFFFFFFULL));
    return std::string(buf);
}

static const char* B58_DIGITS = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

bool b58check_decode_hash160(const std::string& addr, uint8_t hash160_out[20]) {
    std::vector<uint8_t> bytes(addr.length() + 1, 0);
    int length = 0;

    for (char c : addr) {
        const char* p = strchr(B58_DIGITS, c);
        if (!p) return false;
        int carry = (int)(p - B58_DIGITS);
        for (int i = 0; i < length; ++i) {
            int val = (int)bytes[i] * 58 + carry;
            bytes[i] = (uint8_t)(val & 0xFF);
            carry = val >> 8;
        }
        while (carry > 0) {
            bytes[length++] = (uint8_t)(carry & 0xFF);
            carry >>= 8;
        }
    }

    for (size_t i = 0; i < addr.length() && addr[i] == '1'; ++i) {
        bytes[length++] = 0;
    }

    std::reverse(bytes.begin(), bytes.begin() + length);
    bytes.resize(length);

    if (bytes.size() != 25) return false;

    uint8_t sha1[32], sha2[32];
    SHA256(bytes.data(), 21, sha1);
    SHA256(sha1, 32, sha2);
    if (memcmp(sha2, bytes.data() + 21, 4) != 0) return false;

    memcpy(hash160_out, bytes.data() + 1, 20);
    return true;
}

#define SECP_K 0x1000003D1ULL

struct Fe {
    uint64_t d[4];
};

CUDA_HOSTDEV CUDA_INLINE bool fe_is_zero(const Fe& a) {
    return (a.d[0] | a.d[1] | a.d[2] | a.d[3]) == 0;
}

CUDA_HOSTDEV CUDA_INLINE Fe fe_add(const Fe& a, const Fe& b) {
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

CUDA_HOSTDEV CUDA_INLINE Fe fe_sub(const Fe& a, const Fe& b) {
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

#if !defined(__CUDACC__) && (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__)) && defined(__BMI2__) && defined(__ADX__)
static inline __attribute__((always_inline)) Fe fe_mul(const Fe& a, const Fe& b) {
    uint64_t r0, r1, r2, r3;
    uint64_t t4, t5, t6, t7;
    __asm__ __volatile__ (
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

        "mulx %[t4], %%rax, %%rcx\n\t"
        "addq %%rax, %[r0]\n\t"
        "adcq %%rcx, %[r1]\n\t"
        "adcq $0, %[r2]\n\t"
        "adcq $0, %[r3]\n\t"
        "movq $0, %[t4]\n\t"
        "adcq $0, %[t4]\n\t"

        "imulq %%rdx, %[t4]\n\t"
        "addq %[t4], %[r0]\n\t"
        "adcq $0, %[r1]\n\t"
        "adcq $0, %[r2]\n\t"
        "adcq $0, %[r3]\n\t"

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
CUDA_HOSTDEV CUDA_INLINE Fe fe_mul(const Fe& a, const Fe& b) {
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

CUDA_HOSTDEV CUDA_INLINE Fe fe_sqr(const Fe& a) {
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

CUDA_HOSTDEV CUDA_INLINE Fe fe_inv(const Fe& a) {
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

struct AffinePoint {
    Fe x;
    Fe y;
};

struct JacobianPoint {
    Fe X;
    Fe Y;
    Fe Z;
};

CUDA_HOSTDEV CUDA_INLINE AffinePoint get_generator_G() {
    AffinePoint G;
    G.x.d[0] = 0x59F2815B16F81798ULL; G.x.d[1] = 0x029BFCDB2DCE28D9ULL;
    G.x.d[2] = 0x55A06295CE870B07ULL; G.x.d[3] = 0x79BE667EF9DCBBACULL;
    G.y.d[0] = 0x9C47D08FFB10D4B8ULL; G.y.d[1] = 0xFD17B448A6855419ULL;
    G.y.d[2] = 0x5DA4FBFC0E1108A8ULL; G.y.d[3] = 0x483ADA7726A3C465ULL;
    return G;
}

CUDA_HOSTDEV CUDA_INLINE void jacobian_double(JacobianPoint& R, const JacobianPoint& P) {
    if (fe_is_zero(P.Y)) {
        R.X = {{0, 0, 0, 0}}; R.Y = {{0, 0, 0, 0}}; R.Z = {{0, 0, 0, 0}};
        return;
    }
    Fe XX = fe_sqr(P.X);
    Fe YY = fe_sqr(P.Y);
    Fe YYYY = fe_sqr(YY);
    Fe ZZ = fe_sqr(P.Z);

    Fe S = fe_mul(P.X, YY);
    S = fe_add(S, S);
    S = fe_add(S, S);

    Fe M = fe_add(XX, XX);
    M = fe_add(M, XX);

    Fe M2 = fe_sqr(M);
    Fe out_X = fe_sub(fe_sub(M2, S), S);

    Fe Y8 = fe_add(YYYY, YYYY);
    Y8 = fe_add(Y8, Y8);
    Y8 = fe_add(Y8, Y8);
    Fe out_Y = fe_sub(fe_mul(M, fe_sub(S, out_X)), Y8);

    Fe out_Z = fe_add(fe_mul(P.Y, P.Z), fe_mul(P.Y, P.Z));

    R.X = out_X;
    R.Y = out_Y;
    R.Z = out_Z;
}

CUDA_HOSTDEV CUDA_INLINE void jacobian_add_affine(JacobianPoint& R, const JacobianPoint& P, const AffinePoint& Q) {
    if (fe_is_zero(P.Z)) {
        R.X = Q.x; R.Y = Q.y; R.Z = {{1, 0, 0, 0}};
        return;
    }
    Fe Z1Z1 = fe_sqr(P.Z);
    Fe Z1_3 = fe_mul(Z1Z1, P.Z);

    Fe U2 = fe_mul(Q.x, Z1Z1);
    Fe S2 = fe_mul(Q.y, Z1_3);

    Fe H = fe_sub(U2, P.X);
    Fe r = fe_sub(S2, P.Y);

    Fe HH = fe_sqr(H);
    Fe HHH = fe_mul(HH, H);
    Fe V = fe_mul(P.X, HH);

    Fe r2 = fe_sqr(r);
    Fe out_X = fe_sub(fe_sub(r2, HHH), fe_add(V, V));

    Fe out_Y = fe_sub(fe_mul(r, fe_sub(V, out_X)), fe_mul(P.Y, HHH));

    Fe out_Z = fe_mul(P.Z, H);

    R.X = out_X;
    R.Y = out_Y;
    R.Z = out_Z;
}

CUDA_HOSTDEV CUDA_INLINE AffinePoint jacobian_to_affine(const JacobianPoint& P) {
    AffinePoint out;
    Fe Zinv = fe_inv(P.Z);
    Fe Zinv2 = fe_sqr(Zinv);
    Fe Zinv3 = fe_mul(Zinv2, Zinv);
    out.x = fe_mul(P.X, Zinv2);
    out.y = fe_mul(P.Y, Zinv3);
    return out;
}

CUDA_DEV AffinePoint scalar_mul_G(const u256& k) {
    AffinePoint G = get_generator_G();
    JacobianPoint R;
    R.X = {{0, 0, 0, 0}}; R.Y = {{0, 0, 0, 0}}; R.Z = {{0, 0, 0, 0}};
    bool init = false;

    uint64_t limbs[4] = {
        (uint64_t)k.low,
        (uint64_t)(k.low >> 64),
        (uint64_t)k.high,
        (uint64_t)(k.high >> 64)
    };

    int top_limb = 3;
    while (top_limb > 0 && limbs[top_limb] == 0) {
        top_limb--;
    }
    if (limbs[top_limb] == 0) {
        return jacobian_to_affine(R);
    }

#if defined(__CUDA_ARCH__)
    int top_bit = 63 - __clzll((long long)limbs[top_limb]);
#else
    int top_bit = 63 - __builtin_clzll(limbs[top_limb]);
#endif

    for (int i = top_limb; i >= 0; --i) {
        uint64_t w = limbs[i];
        int start_b = (i == top_limb) ? top_bit : 63;
        for (int b = start_b; b >= 0; --b) {
            if (init) {
                jacobian_double(R, R);
            }
            if ((w >> b) & 1) {
                if (!init) {
                    R.X = G.x;
                    R.Y = G.y;
                    R.Z = {{1, 0, 0, 0}};
                    init = true;
                } else {
                    jacobian_add_affine(R, R, G);
                }
            }
        }
    }
    return jacobian_to_affine(R);
}

CUDA_HOSTDEV CUDA_INLINE uint32_t ror32_gpu(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

CUDA_HOSTDEV CUDA_INLINE uint32_t rol32_gpu(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

#if defined(__CUDA_ARCH__)
CUDA_DEV CUDA_INLINE uint32_t bswap32_dev(uint32_t x) {
    return __byte_perm(x, 0, 0x0123);
}
#define BSWAP32_GPU(x) bswap32_dev(x)
#else
#define BSWAP32_GPU(x) __builtin_bswap32(x)
#endif

CUDA_CONST uint32_t K_SHA256_GPU[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};
#ifndef K_SHA256
#define K_SHA256 K_SHA256_GPU
#endif

CUDA_DEV void fast_sha256_into_ripemd_X(uint8_t prefix, const Fe& x, uint32_t X[16]) {
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

    w[16] = w[0] + (ror32_gpu(w[1], 7) ^ ror32_gpu(w[1], 18) ^ (w[1] >> 3));
    w[17] = w[1] + (ror32_gpu(w[2], 7) ^ ror32_gpu(w[2], 18) ^ (w[2] >> 3)) + 0x00A50000U;
    w[18] = w[2] + (ror32_gpu(w[3], 7) ^ ror32_gpu(w[3], 18) ^ (w[3] >> 3)) + (ror32_gpu(w[16], 17) ^ ror32_gpu(w[16], 19) ^ (w[16] >> 10));
    w[19] = w[3] + (ror32_gpu(w[4], 7) ^ ror32_gpu(w[4], 18) ^ (w[4] >> 3)) + (ror32_gpu(w[17], 17) ^ ror32_gpu(w[17], 19) ^ (w[17] >> 10));
    w[20] = w[4] + (ror32_gpu(w[5], 7) ^ ror32_gpu(w[5], 18) ^ (w[5] >> 3)) + (ror32_gpu(w[18], 17) ^ ror32_gpu(w[18], 19) ^ (w[18] >> 10));
    w[21] = w[5] + (ror32_gpu(w[6], 7) ^ ror32_gpu(w[6], 18) ^ (w[6] >> 3)) + (ror32_gpu(w[19], 17) ^ ror32_gpu(w[19], 19) ^ (w[19] >> 10));
    w[22] = w[6] + (ror32_gpu(w[7], 7) ^ ror32_gpu(w[7], 18) ^ (w[7] >> 3)) + 264 + (ror32_gpu(w[20], 17) ^ ror32_gpu(w[20], 19) ^ (w[20] >> 10));
    w[23] = w[7] + (ror32_gpu(w[8], 7) ^ ror32_gpu(w[8], 18) ^ (w[8] >> 3)) + w[16] + (ror32_gpu(w[21], 17) ^ ror32_gpu(w[21], 19) ^ (w[21] >> 10));

    #pragma unroll
    for (int i = 24; i < 64; ++i) {
        uint32_t s0 = ror32_gpu(w[i-15], 7) ^ ror32_gpu(w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = ror32_gpu(w[i-2], 17) ^ ror32_gpu(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    uint32_t a = 0x6a09e667, b = 0xbb67ae85, c = 0x3c6ef372, d = 0xa54ff53a;
    uint32_t e = 0x510e527f, f = 0x9b05688c, g = 0x1f83d9ab, h = 0x5be0cd19;

#define SHA256_STEP_GPU(a, b, c, d, e, f, g, h, kw) do { \
    uint32_t S1 = ror32_gpu(e, 6) ^ ror32_gpu(e, 11) ^ ror32_gpu(e, 25); \
    uint32_t ch = g ^ (e & (f ^ g)); \
    uint32_t temp1 = h + S1 + ch + (kw); \
    uint32_t S0 = ror32_gpu(a, 2) ^ ror32_gpu(a, 13) ^ ror32_gpu(a, 22); \
    uint32_t maj = (a & b) | (c & (a ^ b)); \
    uint32_t temp2 = S0 + maj; \
    d += temp1; \
    h = temp1 + temp2; \
} while (0)

    #pragma unroll
    for (int i = 0; i < 64; i += 8) {
        SHA256_STEP_GPU(a, b, c, d, e, f, g, h, K_SHA256_GPU[i] + w[i]);
        SHA256_STEP_GPU(h, a, b, c, d, e, f, g, K_SHA256_GPU[i+1] + w[i+1]);
        SHA256_STEP_GPU(g, h, a, b, c, d, e, f, K_SHA256_GPU[i+2] + w[i+2]);
        SHA256_STEP_GPU(f, g, h, a, b, c, d, e, K_SHA256_GPU[i+3] + w[i+3]);
        SHA256_STEP_GPU(e, f, g, h, a, b, c, d, K_SHA256_GPU[i+4] + w[i+4]);
        SHA256_STEP_GPU(d, e, f, g, h, a, b, c, K_SHA256_GPU[i+5] + w[i+5]);
        SHA256_STEP_GPU(c, d, e, f, g, h, a, b, K_SHA256_GPU[i+6] + w[i+6]);
        SHA256_STEP_GPU(b, c, d, e, f, g, h, a, K_SHA256_GPU[i+7] + w[i+7]);
    }
#undef SHA256_STEP_GPU

    X[0] = BSWAP32_GPU(0x6a09e667 + a);
    X[1] = BSWAP32_GPU(0xbb67ae85 + b);
    X[2] = BSWAP32_GPU(0x3c6ef372 + c);
    X[3] = BSWAP32_GPU(0xa54ff53a + d);
    X[4] = BSWAP32_GPU(0x510e527f + e);
    X[5] = BSWAP32_GPU(0x9b05688c + f);
    X[6] = BSWAP32_GPU(0x1f83d9ab + g);
    X[7] = BSWAP32_GPU(0x5be0cd19 + h);
    X[8] = 0x00000080U;
    X[9] = 0; X[10] = 0; X[11] = 0; X[12] = 0; X[13] = 0;
    X[14] = 256;
    X[15] = 0;
}

CUDA_CONST uint8_t rl_gpu[80] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
    3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
    1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
    4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13
};
CUDA_CONST uint8_t sl_gpu[80] = {
    11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
    7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
    11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
    11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
    9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6
};
CUDA_CONST uint8_t rr_gpu[80] = {
    5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
    6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
    15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
    8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
    12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11
};
CUDA_CONST uint8_t sr_gpu[80] = {
    8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
    9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
    9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
    15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
    8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11
};

CUDA_DEV void fast_ripemd160_32(const uint32_t X[16], uint32_t out_h[5]) {
    uint32_t A = 0x67452301, B = 0xEFCDAB89, C = 0x98BADCFE, D = 0x10325476, E = 0xC3D2E1F0;
    uint32_t Ap = A, Bp = B, Cp = C, Dp = D, Ep = E;

    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        uint32_t f = B ^ C ^ D;
        uint32_t fp = Bp ^ (Cp | ~Dp);
        uint32_t T = rol32_gpu(A + f + X[rl_gpu[j]], sl_gpu[j]) + E;
        A = E; E = D; D = rol32_gpu(C, 10); C = B; B = T;
        uint32_t Tp = rol32_gpu(Ap + fp + X[rr_gpu[j]] + 0x50A28BE6U, sr_gpu[j]) + Ep;
        Ap = Ep; Ep = Dp; Dp = rol32_gpu(Cp, 10); Cp = Bp; Bp = Tp;
    }

    #pragma unroll
    for (int j = 16; j < 32; ++j) {
        uint32_t f = D ^ (B & (C ^ D));
        uint32_t fp = Cp ^ (Dp & (Bp ^ Cp));
        uint32_t T = rol32_gpu(A + f + X[rl_gpu[j]] + 0x5A827999U, sl_gpu[j]) + E;
        A = E; E = D; D = rol32_gpu(C, 10); C = B; B = T;
        uint32_t Tp = rol32_gpu(Ap + fp + X[rr_gpu[j]] + 0x5C4DD124U, sr_gpu[j]) + Ep;
        Ap = Ep; Ep = Dp; Dp = rol32_gpu(Cp, 10); Cp = Bp; Bp = Tp;
    }

    #pragma unroll
    for (int j = 32; j < 48; ++j) {
        uint32_t f = (B | ~C) ^ D;
        uint32_t fp = (Bp | ~Cp) ^ Dp;
        uint32_t T = rol32_gpu(A + f + X[rl_gpu[j]] + 0x6ED9EBA1U, sl_gpu[j]) + E;
        A = E; E = D; D = rol32_gpu(C, 10); C = B; B = T;
        uint32_t Tp = rol32_gpu(Ap + fp + X[rr_gpu[j]] + 0x6D703EF3U, sr_gpu[j]) + Ep;
        Ap = Ep; Ep = Dp; Dp = rol32_gpu(Cp, 10); Cp = Bp; Bp = Tp;
    }

    #pragma unroll
    for (int j = 48; j < 64; ++j) {
        uint32_t f = C ^ (D & (B ^ C));
        uint32_t fp = Dp ^ (Bp & (Cp ^ Dp));
        uint32_t T = rol32_gpu(A + f + X[rl_gpu[j]] + 0x8F1BBCDCU, sl_gpu[j]) + E;
        A = E; E = D; D = rol32_gpu(C, 10); C = B; B = T;
        uint32_t Tp = rol32_gpu(Ap + fp + X[rr_gpu[j]] + 0x7A6D76E9U, sr_gpu[j]) + Ep;
        Ap = Ep; Ep = Dp; Dp = rol32_gpu(Cp, 10); Cp = Bp; Bp = Tp;
    }

    #pragma unroll
    for (int j = 64; j < 80; ++j) {
        uint32_t f = B ^ (C | ~D);
        uint32_t fp = Bp ^ Cp ^ Dp;
        uint32_t T = rol32_gpu(A + f + X[rl_gpu[j]] + 0xA953FD4EU, sl_gpu[j]) + E;
        A = E; E = D; D = rol32_gpu(C, 10); C = B; B = T;
        uint32_t Tp = rol32_gpu(Ap + fp + X[rr_gpu[j]], sr_gpu[j]) + Ep;
        Ap = Ep; Ep = Dp; Dp = rol32_gpu(Cp, 10); Cp = Bp; Bp = Tp;
    }

    out_h[0] = 0xEFCDAB89 + C + Dp;
    out_h[1] = 0x98BADCFE + D + Ep;
    out_h[2] = 0x10325476 + E + Ap;
    out_h[3] = 0xC3D2E1F0 + A + Bp;
    out_h[4] = 0x67452301 + B + Cp;
}

CUDA_DEV bool check_key_hash160(const u256& k, const uint32_t target_w[5], uint64_t target_h64) {
    AffinePoint P = scalar_mul_G(k);
    uint8_t prefix = (P.y.d[0] & 1) ? 0x03 : 0x02;
    uint32_t X[16];
    fast_sha256_into_ripemd_X(prefix, P.x, X);
    uint32_t h[5];
    fast_ripemd160_32(X, h);

    uint64_t cur_h64 = (uint64_t)h[0] | ((uint64_t)h[1] << 32);
    if (cur_h64 == target_h64) {
        return (h[2] == target_w[2] && h[3] == target_w[3] && h[4] == target_w[4]);
    }
    return false;
}

#if defined(__CUDACC__)
__device__ int dev_found_flag = 0;
__device__ uint64_t dev_found_offset = 0;
__device__ uint32_t dev_target_w[5];
__device__ uint64_t dev_target_h64;

CUDA_GLOBAL void cuda_scan_kernel(u256 base_start, uint64_t total_keys) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_keys || dev_found_flag != 0) return;

    u256 cand_k = base_start + idx;
    if (check_key_hash160(cand_k, dev_target_w, dev_target_h64)) {
        if (atomicExch(&dev_found_flag, 1) == 0) {
            dev_found_offset = idx;
        }
    }
}
#else
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
        KW[i] = _mm256_add_epi32(_mm256_set1_epi32(K_SHA256_GPU[i]), W[i]);
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
} while(0)

    #pragma GCC unroll 64
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

    __m256i h0 = _mm256_add_epi32(_mm256_set1_epi32(0x6a09e667), a);
    __m256i h1 = _mm256_add_epi32(_mm256_set1_epi32(0xbb67ae85), b);
    __m256i h2 = _mm256_add_epi32(_mm256_set1_epi32(0x3c6ef372), c);
    __m256i h3 = _mm256_add_epi32(_mm256_set1_epi32(0xa54ff53a), d);
    __m256i h4 = _mm256_add_epi32(_mm256_set1_epi32(0x510e527f), e);
    __m256i h5 = _mm256_add_epi32(_mm256_set1_epi32(0x9b05688c), f);
    __m256i h6 = _mm256_add_epi32(_mm256_set1_epi32(0x1f83d9ab), g);
    __m256i h7 = _mm256_add_epi32(_mm256_set1_epi32(0x5be0cd19), h);

    const __m256i shuf_mask = _mm256_set_epi8(
        12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3,
        12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3
    );

    X_out[0] = _mm256_shuffle_epi8(h0, shuf_mask);
    X_out[1] = _mm256_shuffle_epi8(h1, shuf_mask);
    X_out[2] = _mm256_shuffle_epi8(h2, shuf_mask);
    X_out[3] = _mm256_shuffle_epi8(h3, shuf_mask);
    X_out[4] = _mm256_shuffle_epi8(h4, shuf_mask);
    X_out[5] = _mm256_shuffle_epi8(h5, shuf_mask);
    X_out[6] = _mm256_shuffle_epi8(h6, shuf_mask);
    X_out[7] = _mm256_shuffle_epi8(h7, shuf_mask);
    X_out[8] = _mm256_set1_epi32(0x00000080);
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

        __m256i T = _mm256_add_epi32(AVX2_ROL(_mm256_add_epi32(_mm256_add_epi32(A, f), X[rl_gpu[j]]), sl_gpu[j]), E);
        A = E; E = D; D = AVX2_ROL(C, 10); C = B; B = T;

        __m256i Tp = _mm256_add_epi32(AVX2_ROL(_mm256_add_epi32(_mm256_add_epi32(Ap, fp), _mm256_add_epi32(X[rr_gpu[j]], _mm256_set1_epi32(0x50A28BE6U))), sr_gpu[j]), Ep);
        Ap = Ep; Ep = Dp; Dp = AVX2_ROL(Cp, 10); Cp = Bp; Bp = Tp;
    }

    #pragma GCC unroll 16
    for (int j = 16; j < 32; ++j) {
        __m256i f = _mm256_xor_si256(D, _mm256_and_si256(B, _mm256_xor_si256(C, D)));
        __m256i fp = _mm256_xor_si256(Cp, _mm256_and_si256(Dp, _mm256_xor_si256(Bp, Cp)));

        __m256i T = _mm256_add_epi32(AVX2_ROL(_mm256_add_epi32(_mm256_add_epi32(A, f), _mm256_add_epi32(X[rl_gpu[j]], _mm256_set1_epi32(0x5A827999U))), sl_gpu[j]), E);
        A = E; E = D; D = AVX2_ROL(C, 10); C = B; B = T;

        __m256i Tp = _mm256_add_epi32(AVX2_ROL(_mm256_add_epi32(_mm256_add_epi32(Ap, fp), _mm256_add_epi32(X[rr_gpu[j]], _mm256_set1_epi32(0x5C4DD124U))), sr_gpu[j]), Ep);
        Ap = Ep; Ep = Dp; Dp = AVX2_ROL(Cp, 10); Cp = Bp; Bp = Tp;
    }

    #pragma GCC unroll 16
    for (int j = 32; j < 48; ++j) {
        __m256i f = _mm256_xor_si256(_mm256_or_si256(B, _mm256_xor_si256(C, _mm256_set1_epi32(-1))), D);
        __m256i fp = _mm256_xor_si256(_mm256_or_si256(Bp, _mm256_xor_si256(Cp, _mm256_set1_epi32(-1))), Dp);

        __m256i T = _mm256_add_epi32(AVX2_ROL(_mm256_add_epi32(_mm256_add_epi32(A, f), _mm256_add_epi32(X[rl_gpu[j]], _mm256_set1_epi32(0x6ED9EBA1U))), sl_gpu[j]), E);
        A = E; E = D; D = AVX2_ROL(C, 10); C = B; B = T;

        __m256i Tp = _mm256_add_epi32(AVX2_ROL(_mm256_add_epi32(_mm256_add_epi32(Ap, fp), _mm256_add_epi32(X[rr_gpu[j]], _mm256_set1_epi32(0x6D703EF3U))), sr_gpu[j]), Ep);
        Ap = Ep; Ep = Dp; Dp = AVX2_ROL(Cp, 10); Cp = Bp; Bp = Tp;
    }

    #pragma GCC unroll 16
    for (int j = 48; j < 64; ++j) {
        __m256i f = _mm256_xor_si256(C, _mm256_and_si256(D, _mm256_xor_si256(B, C)));
        __m256i fp = _mm256_xor_si256(Dp, _mm256_and_si256(Bp, _mm256_xor_si256(Cp, Dp)));

        __m256i T = _mm256_add_epi32(AVX2_ROL(_mm256_add_epi32(_mm256_add_epi32(A, f), _mm256_add_epi32(X[rl_gpu[j]], _mm256_set1_epi32(0x8F1BBCDCU))), sl_gpu[j]), E);
        A = E; E = D; D = AVX2_ROL(C, 10); C = B; B = T;

        __m256i Tp = _mm256_add_epi32(AVX2_ROL(_mm256_add_epi32(_mm256_add_epi32(Ap, fp), _mm256_add_epi32(X[rr_gpu[j]], _mm256_set1_epi32(0x7A6D76E9U))), sr_gpu[j]), Ep);
        Ap = Ep; Ep = Dp; Dp = AVX2_ROL(Cp, 10); Cp = Bp; Bp = Tp;
    }

    #pragma GCC unroll 16
    for (int j = 64; j < 80; ++j) {
        __m256i f = _mm256_xor_si256(B, _mm256_or_si256(C, _mm256_xor_si256(D, _mm256_set1_epi32(-1))));
        __m256i fp = _mm256_xor_si256(Bp, _mm256_xor_si256(Cp, Dp));

        __m256i T = _mm256_add_epi32(AVX2_ROL(_mm256_add_epi32(_mm256_add_epi32(A, f), _mm256_add_epi32(X[rl_gpu[j]], _mm256_set1_epi32(0xA953FD4EU))), sl_gpu[j]), E);
        A = E; E = D; D = AVX2_ROL(C, 10); C = B; B = T;

        __m256i Tp = _mm256_add_epi32(AVX2_ROL(_mm256_add_epi32(_mm256_add_epi32(Ap, fp), X[rr_gpu[j]]), sr_gpu[j]), Ep);
        Ap = Ep; Ep = Dp; Dp = AVX2_ROL(Cp, 10); Cp = Bp; Bp = Tp;
    }

    out_h[0] = _mm256_add_epi32(_mm256_add_epi32(_mm256_set1_epi32(0xEFCDAB89), C), Dp);
    out_h[1] = _mm256_add_epi32(_mm256_add_epi32(_mm256_set1_epi32(0x98BADCFE), D), Ep);
    out_h[2] = _mm256_add_epi32(_mm256_add_epi32(_mm256_set1_epi32(0x10325476), E), Ap);
    out_h[3] = _mm256_add_epi32(_mm256_add_epi32(_mm256_set1_epi32(0xC3D2E1F0), A), Bp);
    out_h[4] = _mm256_add_epi32(_mm256_add_epi32(_mm256_set1_epi32(0x67452301), B), Cp);
}
#endif

static const int BATCH_SIZE = 1024;
static AffinePoint G_TABLE[BATCH_SIZE];

void init_generator_table() {
    for (int i = 1; i <= BATCH_SIZE; ++i) {
        G_TABLE[i - 1] = scalar_mul_G(u256(i));
    }
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
            cur_base = scalar_mul_G(base_k);
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
            cur_k = cur_k + (uint64_t)current_batch;
            local_counter += current_batch;
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
#endif

static size_t curl_cb(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

bool http_get(const std::string& url, std::string* out) {
    CURL* curl = curl_easy_init();
    if (!curl) return false;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, out);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 15L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    return (res == CURLE_OK);
}

bool http_post(const std::string& url, const std::string& data, std::string* out) {
    CURL* curl = curl_easy_init();
    if (!curl) return false;
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, out);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 15L);
    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return (res == CURLE_OK);
}

std::string json_get_string(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return "";
    pos += pattern.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    if (pos >= json.size()) return "";
    if (json[pos] == '"') {
        pos++;
        size_t end = json.find('"', pos);
        if (end == std::string::npos) return "";
        return json.substr(pos, end - pos);
    } else {
        size_t end = pos;
        while (end < json.size() && json[end] != ',' && json[end] != '}' && json[end] != ' ' && json[end] != '\n' && json[end] != '\r') {
            end++;
        }
        return json.substr(pos, end - pos);
    }
}

int main(int argc, char* argv[]) {
    signal(SIGINT, sigint_handler);
    signal(SIGTERM, sigint_handler);

    std::string api_base = "http://65.20.91.208/puzzle_server.php";
    int req_puzzle = 71;
    int multiple = 1;
    std::string custom_user = "";
    bool no_limit = false;
    int threads = 1;
    unsigned int hw = std::thread::hardware_concurrency();

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-s" || arg == "--server") && i + 1 < argc) api_base = argv[++i];
        else if ((arg == "-p" || arg == "--puzzle") && i + 1 < argc) req_puzzle = std::atoi(argv[++i]);
        else if ((arg == "-m" || arg == "--multiple" || arg == "--batch") && i + 1 < argc) multiple = std::atoi(argv[++i]);
        else if ((arg == "-u" || arg == "--user") && i + 1 < argc) custom_user = argv[++i];
        else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (arg == "-d" || arg == "--double") threads = 2;
        else if (arg == "--fast") threads = (hw > 0) ? (int)hw : 4;
        else if (arg == "--no-limit" || arg == "-nl" || arg == "--infinite") no_limit = true;
    }

#if defined(__CUDACC__)
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        return 1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (custom_user.empty()) custom_user = "gpu-" + std::string(prop.name);
#else
    init_generator_table();
#endif

    curl_global_init(CURL_GLOBAL_DEFAULT);
    const int MAX_RANGES = 5;
    int completed_ranges = 0;

    while (g_running.load() && (no_limit || completed_ranges < MAX_RANGES)) {
        std::stringstream ss;
        ss << api_base << "?action=range&puzzle=" << req_puzzle;
        if (multiple > 1) ss << "&multiple=" << multiple;
        if (!custom_user.empty()) ss << "&user=" << custom_user;

        std::string resp;
        if (!http_get(ss.str(), &resp)) {
            std::this_thread::sleep_for(std::chrono::seconds(3));
            continue;
        }

        if (resp.empty() || resp.find("\"start\"") == std::string::npos) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            continue;
        }

        std::string s_start = json_get_string(resp, "start");
        std::string s_end   = json_get_string(resp, "end");
        std::string target_addr = json_get_string(resp, "target_address");
        int block = std::atoi(json_get_string(resp, "block").c_str());
        int range_idx = std::atoi(json_get_string(resp, "range_idx").c_str());
        int range_count = std::atoi(json_get_string(resp, "range_count").c_str());
        if (range_count <= 0) range_count = 1;

        int current_puzzle = std::atoi(json_get_string(resp, "puzzle").c_str());
        if (current_puzzle <= 0) current_puzzle = req_puzzle;

        std::string current_user = !custom_user.empty() 
            ? custom_user 
            : json_get_string(resp, "user");
        if (current_user.empty()) current_user = "user-" + std::to_string(block) + "-" + std::to_string(range_idx);

        u256 start_k = parse_u256(s_start);
        u256 end_k   = parse_u256(s_end);
        std::string s_sz = json_get_string(resp, "range_size");
        uint64_t r_sz = s_sz.empty() ? 0 : std::stoull(s_sz);
        if (end_k <= start_k && r_sz > 0) {
            end_k = start_k + r_sz;
        }
        u256 total_k = end_k - start_k;
        uint64_t total_keys_count = (uint64_t)total_k.low;

        uint8_t h160[20];
        if (!b58check_decode_hash160(target_addr, h160)) {
            continue;
        }

        uint32_t target_w[5];
        for (int i = 0; i < 5; ++i) {
            target_w[i] = (uint32_t)h160[4*i] | ((uint32_t)h160[4*i+1]<<8) | ((uint32_t)h160[4*i+2]<<16) | ((uint32_t)h160[4*i+3]<<24);
        }
        uint64_t target_h64 = (uint64_t)target_w[0] | ((uint64_t)target_w[1] << 32);

        auto t_start = std::chrono::high_resolution_clock::now();
        bool found = false;
        u256 found_key = 0;
        uint64_t actual_checked = 0;

#if defined(__CUDACC__)
        cudaMemcpyToSymbol(dev_target_w, target_w, 5 * sizeof(uint32_t));
        cudaMemcpyToSymbol(dev_target_h64, &target_h64, sizeof(uint64_t));
        int zero = 0;
        cudaMemcpyToSymbol(dev_found_flag, &zero, sizeof(int));

        uint64_t chunk_size = 16777216;
        while (actual_checked < total_keys_count && g_running.load() && !found) {
            uint64_t cur_chunk = std::min(chunk_size, total_keys_count - actual_checked);
            u256 cur_start = start_k + actual_checked;

            int threadsPerBlock = 256;
            int blocks = (cur_chunk + threadsPerBlock - 1) / threadsPerBlock;
            cuda_scan_kernel<<<blocks, threadsPerBlock>>>(cur_start, cur_chunk);
            cudaDeviceSynchronize();

            int h_found = 0;
            cudaMemcpyFromSymbol(&h_found, dev_found_flag, sizeof(int));
            if (h_found != 0) {
                uint64_t h_offset = 0;
                cudaMemcpyFromSymbol(&h_offset, dev_found_offset, sizeof(uint64_t));
                found = true;
                found_key = cur_start + h_offset;
                actual_checked += h_offset + 1;
                break;
            }

            actual_checked += cur_chunk;
        }
#else
        std::atomic<uint64_t> work_offset(0);
        std::atomic<uint64_t> checked_counter(0);
        std::atomic<bool> found_flag(false);
        std::mutex found_mtx;
        uint64_t slice_size = 524288;

        std::vector<std::thread> workers;
        workers.reserve(threads);
        for (int t = 0; t < threads; ++t) {
            workers.emplace_back(
                scan_worker_montgomery,
                start_k,
                std::ref(work_offset),
                total_keys_count,
                slice_size,
                h160,
                target_h64,
                std::ref(found_flag),
                std::ref(found_key),
                std::ref(found_mtx),
                std::ref(checked_counter)
            );
        }

        while (g_running.load() && !found_flag.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            uint64_t done = checked_counter.load(std::memory_order_relaxed);
            if (done >= total_keys_count) {
                break;
            }
        }

        for (auto& w : workers) {
            if (w.joinable()) w.join();
        }

        found = found_flag.load();
        actual_checked = checked_counter.load();
#endif

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t_end - t_start).count();
        if (elapsed <= 0.0) elapsed = 0.001;
        double speed = (double)actual_checked / elapsed;

        if (!found && !g_running.load()) {
            break;
        }

        std::stringstream json;
        json << "{\"action\":\"result\",\"puzzle\":" << current_puzzle
             << ",\"block\":" << block
             << ",\"range_idx\":" << range_idx
             << ",\"range_count\":" << range_count
             << ",\"multiple\":" << range_count
             << ",\"status\":\"" << (found ? "found" : "done") << "\""
             << ",\"private_key\":\"" << (found ? u256_to_hex64(found_key) : "") << "\""
             << ",\"user\":\"" << current_user << "\""
             << ",\"speed\":" << std::fixed << std::setprecision(1) << speed
             << ",\"keys\":\"" << u256_to_dec(total_k) << "\""
             << ",\"range_size\":\"" << u256_to_dec(total_k) << "\""
             << ",\"count\":" << total_keys_count
             << ",\"elapsed\":" << std::fixed << std::setprecision(2) << elapsed << "}";

        std::string post_url = api_base + "?action=result&user=" + current_user;
        std::string ack;
        http_post(post_url, json.str(), &ack);

        completed_ranges++;
        if (found) break;
    }

    curl_global_cleanup();
    return 0;
}
