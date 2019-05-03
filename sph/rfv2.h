// RainForest hash algorithm
// Author: Bill Schneider
// Date: Feb 13th, 2018
//
// RainForest uses native integer operations which are extremely fast on
// modern 64-bit processors, significantly slower on 32-bit processors such
// as GPUs, and extremely slow if at all implementable on FPGAs and ASICs.
// It makes an intensive use of the L1 cache to maintain a heavy intermediary
// state favoring modern CPUs compared to GPUs (small L1 cache shared by many
// shaders) or FPGAs (very hard to implement the required low-latency cache)
// when scanning ranges for nonces. The purpose is to create a fair balance
// between all mining equipments, from mobile phones to extreme performance
// GPUs and to rule out farming factories relying on ASICs and FPGAs. The
// CRC32 instruction is used a lot as it is extremely fast on low-power ARM
// chips and allows such devices to rival high-end PCs mining performance.
//
// Tests on various devices have shown the following performance :
// +--------------------------------------------------------------------------+
// | CPU/GPU       Clock Threads Full hash  Nonce scan  Watts   Cost          |
// |               (MHz)         (80 bytes) (4 bytes)   total                 |
// | Core i7-6700k  4000      8   390 kH/s  1642 kH/s     200  ~$350+PC       |
// | Radeon RX560   1300   1024  1100 kH/s  1650 kH/s     300  ~$180+PC       |
// | RK3368 (8*A53) 1416      8   534 kH/s  1582 kH/s       6   $60 (Geekbox) |
// +--------------------------------------------------------------------------+
//
// Build instructions on Ubuntu 16.04 :
//   - on x86:   use gcc -march=native or -maes to enable AES-NI
//   - on ARMv8: use gcc -march=native or -march=armv8-a+crypto+crc to enable
//               CRC32 and AES extensions.
//
// Note: always use the same options to build all files!
//

#ifndef RAINFOREST2
#define RAINFOREST2


#ifdef _MSC_VER
#define inline __inline
# define __func__ __FUNCTION__
# define __thread __declspec(thread)
# define _ALIGN(x) __declspec(align(x))
#else
# define _ALIGN(x) __attribute__ ((aligned(x)))
#include <libgen.h>
#endif

// this seems necessary only for gcc, otherwise hash is bogus
#ifdef _MSC_VER
typedef uint8_t  rf_u8;
typedef uint16_t rf_u16;
typedef uint32_t rf_u32;
typedef uint64_t rf_u64;
#else
typedef __attribute__((may_alias)) uint8_t  rf_u8;
typedef __attribute__((may_alias)) uint16_t rf_u16;
typedef __attribute__((may_alias)) uint32_t rf_u32;
typedef __attribute__((may_alias)) uint64_t rf_u64;
#endif


#ifndef RF_ALIGN
#ifdef _MSC_VER
#define RF_ALIGN(x) __declspec(align(x))
#else
#define RF_ALIGN(x) __attribute__((aligned(x)))
#endif
#endif


#include <stdint.h>
#include <stddef.h>


#define RFV2_RAMBOX_HIST 1536

// number of loops run over the initial message. At 19 loops
// most runs are under 256 changes
#define RFV2_LOOPS 320

typedef union {
	rf_u8  b[32];
	rf_u16 w[16];
	rf_u32 d[8];
	rf_u64 q[4];
} rf_hash256_t;

typedef struct RF_ALIGN(16) rfv2_ctx {
	uint32_t word;  // LE pending message
	uint32_t len;   // total message length
	uint32_t crc;
	uint16_t changes; // must remain lower than RFV2_RAMBOX_HIST
	uint16_t left_bits;
	uint64_t *rambox;
	uint32_t rb_o;    // rambox offset
	uint32_t rb_l;    // rambox length
	rf_hash256_t RF_ALIGN(32) hash;
	uint32_t hist[RFV2_RAMBOX_HIST];
	uint64_t prev[RFV2_RAMBOX_HIST];
	unsigned char *test;
} rfv2_ctx_t;


#define RFV2_RAMBOX_SIZE (96*1024*1024/8)

#if defined(__cplusplus)
extern "C" {
#endif

	void rfv2_final(void *out, rfv2_ctx_t *ctx);
	void rfv2_update(rfv2_ctx_t *ctx, const void *msg, size_t len);
	void rfv2_init(rfv2_ctx_t *ctx, uint32_t seed, void *rambox);
	void rfv2_init_test(rfv2_ctx_t *ctx, uint32_t seed, void *rambox, void *test);

	int rfv2_hash(void *out, const void *in, size_t len, void *rambox, const void *rambox_template);
	int rfv2_hash2(void *out, const void *in, size_t len, void *rambox, const void *rambox_template, uint32_t seed);
	void rfv2_raminit(void *area);
	uint32_t rf_crc32_mem(uint32_t crc, const void *msg, size_t len);
	uint8_t sin_scaled(unsigned int x);
	void rfv2_pad256(rfv2_ctx_t *ctx);

#if defined(__cplusplus)
}
#endif


#endif
