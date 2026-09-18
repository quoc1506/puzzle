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
    Fe r;
    uint64_t c = 0;
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
}

CUDA_HOSTDEV CUDA_INLINE Fe fe_sub(const Fe& a, const Fe& b) {
    Fe r;
    uint64_t borrow = 0;
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
        for (int i = 1; i < 4; ++i) {
            diff = r.d[i] - b2;
            b2 = (r.d[i] < b2);
            r.d[i] = diff;
        }
    }
    return r;
}

CUDA_HOSTDEV CUDA_INLINE Fe fe_reduce(uint64_t t[8]) {
    const uint64_t K = 0x1000003D1ULL;
    uint64_t c = 0;
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
    uint64_t t[8] = {0};
    for (int i = 0; i < 4; ++i) {
        uint64_t carry = 0;
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
}

CUDA_HOSTDEV CUDA_INLINE Fe fe_sqr(const Fe& a) {
    return fe_mul(a, a);
}

// 100% Mathematically Verified Inversion modulo 2^256 - 2^32 - 977
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
            if (w & 1) {
                res = fe_mul(res, base);
            }
            if (i == 3 && b == 63) break;
            base = fe_sqr(base);
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
    return (x >> n) | (x << (32 - n));
}

CUDA_HOSTDEV CUDA_INLINE uint32_t rol32_dev(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
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

#ifdef __CUDACC__
__constant__ uint32_t dev_K_SHA256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};
#endif

static const uint32_t host_K_SHA256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

CUDA_HOSTDEV CUDA_INLINE uint32_t get_sha256_k(int i) {
#if defined(__CUDA_ARCH__)
    return dev_K_SHA256[i];
#else
    return host_K_SHA256[i];
#endif
}

CUDA_HOSTDEV CUDA_INLINE void fast_sha256_into_ripemd_X(uint8_t prefix, const Fe& x, uint32_t X[16]) {
    uint32_t w[64];
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

    #pragma unroll
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = ror32_dev(w[i - 15], 7) ^ ror32_dev(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = ror32_dev(w[i - 2], 17) ^ ror32_dev(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = 0x6a09e667, b = 0xbb67ae85, c = 0x3c6ef372, d = 0xa54ff53a;
    uint32_t e = 0x510e527f, f = 0x9b05688c, g = 0x1f83d9ab, h = 0x5be0cd19;

    #pragma unroll
    for (int i = 0; i < 64; ++i) {
        uint32_t S1 = ror32_dev(e, 6) ^ ror32_dev(e, 11) ^ ror32_dev(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + S1 + ch + get_sha256_k(i) + w[i];
        uint32_t S0 = ror32_dev(a, 2) ^ ror32_dev(a, 13) ^ ror32_dev(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;

        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }

    X[0] = bswap32_dev(0x6a09e667 + a);
    X[1] = bswap32_dev(0xbb67ae85 + b);
    X[2] = bswap32_dev(0x3c6ef372 + c);
    X[3] = bswap32_dev(0xa54ff53a + d);
    X[4] = bswap32_dev(0x510e527f + e);
    X[5] = bswap32_dev(0x9b05688c + f);
    X[6] = bswap32_dev(0x1f83d9ab + g);
    X[7] = bswap32_dev(0x5be0cd19 + h);
    X[8] = 0x00000080U;
    X[9] = 0; X[10] = 0; X[11] = 0; X[12] = 0; X[13] = 0;
    X[14] = 256;
    X[15] = 0;
}

#ifdef __CUDACC__
__constant__ uint8_t dev_rl_tab[80] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
    3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
    1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
    4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13
};
__constant__ uint8_t dev_sl_tab[80] = {
    11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
    7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
    11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
    11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
    9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6
};
__constant__ uint8_t dev_rr_tab[80] = {
    5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
    6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
    15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
    8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
    12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11
};
__constant__ uint8_t dev_sr_tab[80] = {
    8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
    9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
    9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
    15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
    8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11
};
#endif

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

CUDA_HOSTDEV CUDA_INLINE uint8_t get_ripemd_rl(int j) {
#if defined(__CUDA_ARCH__)
    return dev_rl_tab[j];
#else
    return host_rl_tab[j];
#endif
}

CUDA_HOSTDEV CUDA_INLINE uint8_t get_ripemd_sl(int j) {
#if defined(__CUDA_ARCH__)
    return dev_sl_tab[j];
#else
    return host_sl_tab[j];
#endif
}

CUDA_HOSTDEV CUDA_INLINE uint8_t get_ripemd_rr(int j) {
#if defined(__CUDA_ARCH__)
    return dev_rr_tab[j];
#else
    return host_rr_tab[j];
#endif
}

CUDA_HOSTDEV CUDA_INLINE uint8_t get_ripemd_sr(int j) {
#if defined(__CUDA_ARCH__)
    return dev_sr_tab[j];
#else
    return host_sr_tab[j];
#endif
}

CUDA_HOSTDEV CUDA_INLINE void fast_ripemd160_32(const uint32_t X[16], uint32_t out_h[5]) {
    uint32_t A = 0x67452301, B = 0xEFCDAB89, C = 0x98BADCFE, D = 0x10325476, E = 0xC3D2E1F0;
    uint32_t Ap = A, Bp = B, Cp = C, Dp = D, Ep = E;

    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        uint32_t f = B ^ C ^ D;
        uint32_t fp = Bp ^ (Cp | ~Dp);
        uint32_t T = rol32_dev(A + f + X[get_ripemd_rl(j)], get_ripemd_sl(j)) + E;
        A = E; E = D; D = rol32_dev(C, 10); C = B; B = T;
        uint32_t Tp = rol32_dev(Ap + fp + X[get_ripemd_rr(j)] + 0x50A28BE6U, get_ripemd_sr(j)) + Ep;
        Ap = Ep; Ep = Dp; Dp = rol32_dev(Cp, 10); Cp = Bp; Bp = Tp;
    }

    #pragma unroll
    for (int j = 16; j < 32; ++j) {
        uint32_t f = (B & C) | (~B & D);
        uint32_t fp = (Bp & Dp) | (Cp & ~Dp);
        uint32_t T = rol32_dev(A + f + X[get_ripemd_rl(j)] + 0x5A827999U, get_ripemd_sl(j)) + E;
        A = E; E = D; D = rol32_dev(C, 10); C = B; B = T;
        uint32_t Tp = rol32_dev(Ap + fp + X[get_ripemd_rr(j)] + 0x5C4DD124U, get_ripemd_sr(j)) + Ep;
        Ap = Ep; Ep = Dp; Dp = rol32_dev(Cp, 10); Cp = Bp; Bp = Tp;
    }

    #pragma unroll
    for (int j = 32; j < 48; ++j) {
        uint32_t f = (B | ~C) ^ D;
        uint32_t fp = (Bp | ~Cp) ^ Dp;
        uint32_t T = rol32_dev(A + f + X[get_ripemd_rl(j)] + 0x6ED9EBA1U, get_ripemd_sl(j)) + E;
        A = E; E = D; D = rol32_dev(C, 10); C = B; B = T;
        uint32_t Tp = rol32_dev(Ap + fp + X[get_ripemd_rr(j)] + 0x6D703EF3U, get_ripemd_sr(j)) + Ep;
        Ap = Ep; Ep = Dp; Dp = rol32_dev(Cp, 10); Cp = Bp; Bp = Tp;
    }

    #pragma unroll
    for (int j = 48; j < 64; ++j) {
        uint32_t f = C ^ (D & (B ^ C));
        uint32_t fp = Dp ^ (Bp & (Cp ^ Dp));
        uint32_t T = rol32_dev(A + f + X[get_ripemd_rl(j)] + 0x8F1BBCDCU, get_ripemd_sl(j)) + E;
        A = E; E = D; D = rol32_dev(C, 10); C = B; B = T;
        uint32_t Tp = rol32_dev(Ap + fp + X[get_ripemd_rr(j)] + 0x7A6D76E9U, get_ripemd_sr(j)) + Ep;
        Ap = Ep; Ep = Dp; Dp = rol32_dev(Cp, 10); Cp = Bp; Bp = Tp;
    }

    #pragma unroll
    for (int j = 64; j < 80; ++j) {
        uint32_t f = B ^ (C | ~D);
        uint32_t fp = Bp ^ Cp ^ Dp;
        uint32_t T = rol32_dev(A + f + X[get_ripemd_rl(j)] + 0xA953FD4EU, get_ripemd_sl(j)) + E;
        A = E; E = D; D = rol32_dev(C, 10); C = B; B = T;
        uint32_t Tp = rol32_dev(Ap + fp + X[get_ripemd_rr(j)], get_ripemd_sr(j)) + Ep;
        Ap = Ep; Ep = Dp; Dp = rol32_dev(Cp, 10); Cp = Bp; Bp = Tp;
    }

    // Standard RIPEMD-160 Parallel Branch Accumulation
    out_h[0] = 0xEFCDAB89 + C + Dp;
    out_h[1] = 0x98BADCFE + D + Ep;
    out_h[2] = 0x10325476 + E + Ap;
    out_h[3] = 0xC3D2E1F0 + A + Bp;
    out_h[4] = 0x67452301 + B + Cp;
}

#ifdef __CUDACC__
__constant__ AffinePoint dev_G_table[16];
__device__ int dev_found_flag = 0;
__device__ uint64_t dev_found_offset = 0;
__constant__ uint32_t dev_target_w[5];
__constant__ uint64_t dev_target_h64;

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

// True Warp-Level Montgomery Batch Inversion across 32 lanes
CUDA_DEV CUDA_INLINE Fe warp_montgomery_inv(const Fe& v, int lane) {
    // 1. Prefix scan
    Fe prefix = v;
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        Fe up = shfl_up_fe(prefix, offset);
        if (lane >= offset) {
            prefix = fe_mul(prefix, up);
        }
    }

    // 2. Suffix scan
    Fe suffix = v;
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        Fe down = shfl_down_fe(suffix, offset);
        if (lane + offset < 32) {
            suffix = fe_mul(suffix, down);
        }
    }

    // 3. Lane 31 computes inverse of all 32 elements combined
    Fe total_inv;
    if (lane == 31) {
        total_inv = fe_inv(prefix);
    }
    total_inv = shfl_fe(total_inv, 31);

    // 4. Multiply total inverse by prefix[lane-1] and suffix[lane+1]
    Fe p_prev = shfl_up_fe(prefix, 1);
    Fe s_next = shfl_down_fe(suffix, 1);

    Fe other;
    if (lane == 0) {
        other = s_next;
    } else if (lane == 31) {
        other = p_prev;
    } else {
        other = fe_mul(p_prev, s_next);
    }

    return fe_mul(total_inv, other);
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
            uint32_t window = (uint32_t)((w >> b) & 0xF);
            for (int i = 0; i < 4; ++i) {
                if (!res.infinity) {
                    res = jacobian_double(res);
                }
            }
            if (window != 0) {
                res = jacobian_add_affine(res, dev_G_table[window]);
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
    uint64_t hash64 = (uint64_t)out[0] | ((uint64_t)out[1] << 32);
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
    const uint32_t target_w[5],
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
                uint8_t prefix = (cur_points[i].y.d[0] & 1) ? 0x03 : 0x02;
                uint32_t X[16];
                fast_sha256_into_ripemd_X(prefix, cur_points[i].x, X);
                uint32_t out[5];
                fast_ripemd160_32(X, out);

                uint64_t cur_h64 = (uint64_t)out[0] | ((uint64_t)out[1] << 32);
                if (cur_h64 == target_h64 && out[2] == target_w[2] && out[3] == target_w[3] && out[4] == target_w[4]) {
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

    std::string api_base = "http://65.20.91.208/puzzle_server.php";
    int current_puzzle = 70;
    std::string current_user = "guest";
    int requested_multiple = 1;

    unsigned int hw = std::thread::hardware_concurrency();
    int threads = (hw > 0) ? (int)hw : 4;
    bool force_cpu = false;

    // Dedicated Pre-Pass for CPU mode detection (-cpu, --cpu, -c, etc.)
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        std::string alow = a;
        std::transform(alow.begin(), alow.end(), alow.begin(), ::tolower);
        if (alow == "-cpu" || alow == "--cpu" || alow == "-cpu-only" || alow == "--cpu-only" ||
            alow == "-c" || alow == "cpu" || alow == "--device=cpu" || alow == "-device=cpu") {
            force_cpu = true;
        }
        if ((alow == "-mode" || alow == "--mode" || alow == "-device" || alow == "--device") && i + 1 < argc) {
            std::string nxt = argv[i + 1];
            std::transform(nxt.begin(), nxt.end(), nxt.begin(), ::tolower);
            if (nxt == "cpu") force_cpu = true;
        }
    }
    const char* env_cpu = std::getenv("FORCE_CPU");
    if (env_cpu && (std::string(env_cpu) == "1" || std::string(env_cpu) == "true" || std::string(env_cpu) == "cpu")) {
        force_cpu = true;
    }

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
        } else if (arg == "-cpu" || arg == "--cpu" || arg == "-c" || arg == "--cpu-only") {
            force_cpu = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                char* endp = NULL;
                long th = std::strtol(argv[i + 1], &endp, 10);
                if (endp != argv[i + 1] && th > 0) {
                    threads = (int)th;
                    ++i;
                }
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
    }
    if (!use_cuda) {
        std::cout << "[*] Mode          : CPU mode" << (force_cpu ? " (forced with -cpu flag)" : "") << std::endl;
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
        std::memset(&h_table[0], 0, sizeof(AffinePoint));
        for (int i = 1; i < 16; ++i) {
            uint64_t s[4] = { (uint64_t)i, 0, 0, 0 };
            h_table[i] = scalar_mul_G(s);
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

        // Exact Synchronized Word and 64-bit Representation
        uint32_t target_w[5];
        for (int i = 0; i < 5; ++i) {
            target_w[i] = (uint32_t)target_h160[i * 4] |
                          ((uint32_t)target_h160[i * 4 + 1] << 8) |
                          ((uint32_t)target_h160[i * 4 + 2] << 16) |
                          ((uint32_t)target_h160[i * 4 + 3] << 24);
        }
        uint64_t target_h64 = (uint64_t)target_w[0] | ((uint64_t)target_w[1] << 32);

        std::cout << "[*] Block " << block_idx << " | Ranges: " << range_count
                  << " | Keys: " << u256_to_dec(total_keys) << std::endl;

        bool hit = false;
        u256 found_key = 0;
        uint64_t checked = 0;
        auto t_start = std::chrono::high_resolution_clock::now();

#ifdef __CUDACC__
        if (use_cuda) {
            cudaMemcpyToSymbol(dev_target_w, target_w, sizeof(target_w));
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
                                  target_w, std::ref(found_flag), std::ref(found_key),
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
