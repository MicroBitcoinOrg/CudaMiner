

/*
* Rainforest kernel implementation.
*
* ==========================(LICENSE BEGIN)============================
* Copyright (c) 2018 Bill Schneider
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*
* ===========================(LICENSE END)=============================
*/

// djm34 cuda porting
#include <stdio.h>
#include <memory.h>


#include "lyra2/cuda_lyra2_vectors.h"

typedef uint8_t uchar;
typedef uint16_t ushort;
typedef uint32_t uint;
typedef uint64_t ulong;

#define __constant __constant__


uint32_t *d_aMinNonces[16];
uint32_t *h_aMinNonces[16];

__constant__  uint32_t pTarget[8];
__constant__  uint32_t pData[20]; // truncated data
uint16_t *TheIndex[16];

//__constant__  uint64_t TheCarry[5];
uint64_t * TheRamBox[16];

#define RFV2_RAMBOX_SIZE 12*1024*1024
#define AGGR 16

// these archs are fine with unaligned reads
//#define RF_UNALIGNED_LE64

#define RFV2_INIT_CRC 20180213

#ifndef RF_ALIGN
#define RF_ALIGN(x) __align__(x)
#endif

#define RFV2_RAMBOX_HIST 1536

// number of loops run over the initial message. At 19 loops
// most runs are under 256 changes
#define RFV2_LOOPS 320

typedef union {
	uchar  b[32];
	ushort w[16];
	uint   d[8];
	ulong  q[4];
} hash256_t;


typedef struct RF_ALIGN(64) rfv2_ctx {
	uint word;  // LE pending message
	uint len;   // total message length
	uint crc;
	uint rb_o;    // rambox offset
	uint rb_l;    // rambox length
	uint16_t changes; // must remain lower than RFV2_RAMBOX_HIST	
	uint16_t left_bits;
//	ulong *rambox;
	uint16_t gchanges;
	uint16_t * __restrict__ LocalIndex;
	hash256_t RF_ALIGN(32) hash;
	uint  hist[RFV2_RAMBOX_HIST];
	ulong prev[RFV2_RAMBOX_HIST];
} rfv2_ctx_t;


// the table is used as an 8 bit-aligned array of ulong for the first word,
// and as a 16 bit-aligned array of ulong for the second word. It is filled
// with the sha256 of "RainForestProCpuAntiAsic", iterated over and over until
// the table is filled. The highest offset being ((ushort *)table)[255] we
// need to add 6 extra bytes at the end to read an ulong. Maybe calculated
// on a UNIX system with this loop :
//
//   ref="RainForestProCpuAntiAsic"
//   for ((i=0;i<18;i++)); do
//     set $(echo -n $ref|sha256sum)
//     echo $1|sed 's/\(..\)/0x\1,/g'
//     ref=$(printf $(echo $1|sed 's/\(..\)/\\x\1/g'))
//   done

__constant static const uchar rfv2_table[256 * 2 + 6] = {
	0x8e,0xc1,0xa8,0x04,0x38,0x78,0x7c,0x54,0x29,0x23,0x1b,0x78,0x9f,0xf9,0x27,0x54,
	0x11,0x78,0x95,0xb6,0xaf,0x78,0x45,0x16,0x2b,0x9e,0x91,0xe8,0x97,0x25,0xf8,0x63,
	0x82,0x56,0xcf,0x48,0x6f,0x82,0x14,0x0d,0x61,0xbe,0x47,0xd1,0x37,0xee,0x30,0xa9,
	0x28,0x1e,0x4b,0xbf,0x07,0xcd,0x41,0xdf,0x23,0x21,0x12,0xb8,0x81,0x99,0x1d,0xe6,
	0x68,0xcf,0xfa,0x2d,0x8e,0xb9,0x88,0xa7,0x15,0xce,0x9e,0x2f,0xeb,0x1b,0x0f,0x67,
	0x20,0x68,0x6c,0xa9,0x5d,0xc1,0x7c,0x76,0xdf,0xbd,0x98,0x61,0xb4,0x14,0x65,0x40,
	0x1e,0x72,0x51,0x74,0x93,0xd3,0xad,0xbe,0x46,0x0a,0x25,0xfb,0x6a,0x5e,0x1e,0x8a,
	0x5a,0x03,0x3c,0xab,0x12,0xc2,0xd4,0x07,0x91,0xab,0xc9,0xdf,0x92,0x2c,0x85,0x6a,
	0xa6,0x25,0x1e,0x66,0x50,0x26,0x4e,0xa8,0xbd,0xda,0x88,0x1b,0x95,0xd4,0x00,0xeb,
	0x0d,0x1c,0x9b,0x3c,0x86,0xc7,0xb2,0xdf,0xb4,0x5a,0x36,0x15,0x8e,0x04,0xd2,0x54,
	0x79,0xd2,0x3e,0x3d,0x99,0x50,0xa6,0x12,0x4c,0x32,0xc8,0x51,0x14,0x4d,0x4b,0x0e,
	0xbb,0x17,0x80,0x8f,0xa4,0xc4,0x99,0x72,0xd7,0x14,0x4b,0xef,0xed,0x14,0xe9,0x17,
	0xfa,0x9b,0x5d,0x37,0xd6,0x2f,0xef,0x02,0xd6,0x71,0x0a,0xbd,0xc5,0x40,0x11,0x90,
	0x90,0x4e,0xb4,0x4c,0x72,0x51,0x7a,0xd8,0xba,0x30,0x4d,0x8c,0xe2,0x11,0xbb,0x6d,
	0x4b,0xbc,0x6f,0x14,0x0c,0x9f,0xfa,0x5e,0x66,0x40,0x45,0xcb,0x7d,0x1b,0x3a,0xc5,
	0x5e,0x9c,0x1e,0xcc,0xbd,0x16,0x3b,0xcf,0xfb,0x2a,0xd2,0x08,0x2a,0xf8,0x3d,0x46,
	0x93,0x90,0xb3,0x66,0x81,0x34,0x7f,0x6d,0x9b,0x8c,0x99,0x03,0xc5,0x27,0xa3,0xd9,
	0xce,0x90,0x88,0x0f,0x55,0xc3,0xa1,0x60,0x53,0xc8,0x0d,0x25,0xae,0x61,0xd9,0x72,
	0x48,0x1d,0x6c,0x61,0xd2,0x87,0xdd,0x3d,0x23,0xf5,0xde,0x93,0x39,0x4c,0x43,0x9a,
	0xf9,0x37,0xf2,0x61,0xd7,0xf8,0xea,0x65,0xf0,0xf1,0xde,0x3f,0x05,0x57,0x83,0x81,
	0xde,0x02,0x62,0x49,0xd4,0x32,0x7e,0x4a,0xd4,0x9f,0x40,0x7e,0xb9,0x91,0xb1,0x35,
	0xf7,0x62,0x3f,0x65,0x9e,0x4d,0x2b,0x10,0xde,0xd4,0x77,0x64,0x0f,0x84,0xad,0x92,
	0xe7,0xa3,0x8a,0x10,0xc1,0x14,0xeb,0x57,0xc4,0xad,0x8e,0xc2,0xc7,0x32,0xa3,0x7e,
	0x50,0x1f,0x7c,0xbb,0x2e,0x5f,0xf5,0x18,0x22,0xea,0xec,0x9d,0xa4,0x77,0xcd,0x85,
	0x04,0x2f,0x20,0x61,0x72,0xa7,0x0c,0x92,0x06,0x4d,0x01,0x70,0x9b,0x35,0xa1,0x27,
	0x32,0x6e,0xb9,0x78,0xe0,0xaa,0x5f,0x91,0xa6,0x51,0xe3,0x63,0xf8,0x97,0x2f,0x60,
	0xd9,0xfb,0x15,0xe5,0x59,0xcf,0x31,0x3c,0x61,0xc7,0xb5,0x61,0x2a,0x6b,0xdd,0xd1,
	0x09,0x70,0xc0,0xcf,0x94,0x7a,0xcc,0x31,0x94,0xb1,0xa2,0xf6,0x95,0xc0,0x38,0x3d,
	0xef,0x19,0x30,0x70,0xdd,0x62,0x32,0x8f,0x7c,0x30,0xb9,0x18,0xf8,0xe7,0x8f,0x0a,
	0xaa,0xb6,0x00,0x86,0xf2,0xe0,0x30,0x5f,0xa2,0xe8,0x00,0x8e,0x05,0xa0,0x22,0x18,
	0x9f,0x83,0xd4,0x3a,0x85,0x10,0xb9,0x51,0x8d,0x07,0xf0,0xb3,0xcd,0x9b,0x55,0xa1,
	0x14,0xce,0x0f,0xb2,0xcf,0xb8,0xce,0x2d,0xe6,0xe8,0x35,0x32,0x1f,0x22,0xb5,0xec,
	0xd0,0xb9,0x72,0xa8,0xb4,0x97
	//,0x6e,0x0a,0x47,0xcd,0x5a,0xf0,0xdc,0xeb,0xfd,0x46,
	//0xe5,0x6e,0x83,0xe6,0x1a,0xcc,0x4a,0x8b,0xa5,0x28,0x9e,0x50,0x48,0xa9,0xa2,0x6b,
};


__constant static const uchar SBOX[256] = {
	0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
	0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
	0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
	0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
	0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
	0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
	0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
	0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
	0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
	0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
	0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
	0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
	0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
	0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
	0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
	0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

/* shifts to do for shift_rows step */
__constant static const uchar shifts[16] = {
	0,  5, 10, 15,
	4,  9, 14,  3,
	8, 13,  2,  7,
	12,  1,  6, 11
};


// crc32 lookup tables
__constant static const uint rf_crc32_table[256] = {
	/* 0x00 */ 0x00000000, 0x77073096, 0xee0e612c, 0x990951ba,
	/* 0x04 */ 0x076dc419, 0x706af48f, 0xe963a535, 0x9e6495a3,
	/* 0x08 */ 0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988,
	/* 0x0c */ 0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91,
	/* 0x10 */ 0x1db71064, 0x6ab020f2, 0xf3b97148, 0x84be41de,
	/* 0x14 */ 0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
	/* 0x18 */ 0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec,
	/* 0x1c */ 0x14015c4f, 0x63066cd9, 0xfa0f3d63, 0x8d080df5,
	/* 0x20 */ 0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172,
	/* 0x24 */ 0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b,
	/* 0x28 */ 0x35b5a8fa, 0x42b2986c, 0xdbbbc9d6, 0xacbcf940,
	/* 0x2c */ 0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
	/* 0x30 */ 0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116,
	/* 0x34 */ 0x21b4f4b5, 0x56b3c423, 0xcfba9599, 0xb8bda50f,
	/* 0x38 */ 0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924,
	/* 0x3c */ 0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d,
	/* 0x40 */ 0x76dc4190, 0x01db7106, 0x98d220bc, 0xefd5102a,
	/* 0x44 */ 0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
	/* 0x48 */ 0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818,
	/* 0x4c */ 0x7f6a0dbb, 0x086d3d2d, 0x91646c97, 0xe6635c01,
	/* 0x50 */ 0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e,
	/* 0x54 */ 0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457,
	/* 0x58 */ 0x65b0d9c6, 0x12b7e950, 0x8bbeb8ea, 0xfcb9887c,
	/* 0x5c */ 0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
	/* 0x60 */ 0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2,
	/* 0x64 */ 0x4adfa541, 0x3dd895d7, 0xa4d1c46d, 0xd3d6f4fb,
	/* 0x68 */ 0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0,
	/* 0x6c */ 0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9,
	/* 0x70 */ 0x5005713c, 0x270241aa, 0xbe0b1010, 0xc90c2086,
	/* 0x74 */ 0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
	/* 0x78 */ 0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4,
	/* 0x7c */ 0x59b33d17, 0x2eb40d81, 0xb7bd5c3b, 0xc0ba6cad,
	/* 0x80 */ 0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a,
	/* 0x84 */ 0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683,
	/* 0x88 */ 0xe3630b12, 0x94643b84, 0x0d6d6a3e, 0x7a6a5aa8,
	/* 0x8c */ 0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
	/* 0x90 */ 0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe,
	/* 0x94 */ 0xf762575d, 0x806567cb, 0x196c3671, 0x6e6b06e7,
	/* 0x98 */ 0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc,
	/* 0x9c */ 0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5,
	/* 0xa0 */ 0xd6d6a3e8, 0xa1d1937e, 0x38d8c2c4, 0x4fdff252,
	/* 0xa4 */ 0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
	/* 0xa8 */ 0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60,
	/* 0xac */ 0xdf60efc3, 0xa867df55, 0x316e8eef, 0x4669be79,
	/* 0xb0 */ 0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236,
	/* 0xb4 */ 0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f,
	/* 0xb8 */ 0xc5ba3bbe, 0xb2bd0b28, 0x2bb45a92, 0x5cb36a04,
	/* 0xbc */ 0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
	/* 0xc0 */ 0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a,
	/* 0xc4 */ 0x9c0906a9, 0xeb0e363f, 0x72076785, 0x05005713,
	/* 0xc8 */ 0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38,
	/* 0xcc */ 0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21,
	/* 0xd0 */ 0x86d3d2d4, 0xf1d4e242, 0x68ddb3f8, 0x1fda836e,
	/* 0xd4 */ 0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
	/* 0xd8 */ 0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c,
	/* 0xdc */ 0x8f659eff, 0xf862ae69, 0x616bffd3, 0x166ccf45,
	/* 0xe0 */ 0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2,
	/* 0xe4 */ 0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db,
	/* 0xe8 */ 0xaed16a4a, 0xd9d65adc, 0x40df0b66, 0x37d83bf0,
	/* 0xec */ 0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
	/* 0xf0 */ 0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6,
	/* 0xf4 */ 0xbad03605, 0xcdd70693, 0x54de5729, 0x23d967bf,
	/* 0xf8 */ 0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94,
	/* 0xfc */ 0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d
};





/* add the round key to the state with simple XOR operation */
__device__ static void add_round_key(uchar * state, uchar * rkey)
{
	uchar i;

	for (i = 0; i < 16; i++)
		state[i] ^= rkey[i];
}

/* substitute all bytes using Rijndael's substitution box */
__device__ static void sub_bytes(uchar * state)
{
	uchar i;

	for (i = 0; i < 16; i++)
		state[i] = SBOX[state[i]];
}

/* imagine the state not as 1-dimensional, but a 4x4 grid;
* this step shifts the rows of this grid around */
__device__ static void shift_rows(uchar * state)
{
	uchar temp[16];
	uchar i;

	for (i = 0; i < 16; i++)
		temp[i] = state[shifts[i]];

	for (i = 0; i < 16; i++)
		state[i] = temp[i];
}

/* mix columns */
__device__ static void mix_columns(uchar * state)
{
	uchar a[4];
	uchar b[4];
	uchar h, i, k;

	for (k = 0; k < 4; k++) {
		for (i = 0; i < 4; i++) {
			a[i] = state[i + 4 * k];
			h = state[i + 4 * k] & 0x80; /* hi bit */
			b[i] = state[i + 4 * k] << 1;

			if (h == 0x80)
				b[i] ^= 0x1b; /* Rijndael's Galois field */
		}

		state[4 * k] = b[0] ^ a[3] ^ a[2] ^ b[1] ^ a[1];
		state[1 + 4 * k] = b[1] ^ a[0] ^ a[3] ^ b[2] ^ a[2];
		state[2 + 4 * k] = b[2] ^ a[1] ^ a[0] ^ b[3] ^ a[3];
		state[3 + 4 * k] = b[3] ^ a[2] ^ a[1] ^ b[0] ^ a[0];
	}
}


/* key schedule stuff */

#define __ENDIAN_LITTLE__ 1
__device__ static inline uint rotate32(uint in) {
#if __ENDIAN_LITTLE__
	//	 return rotate(in, (uint)24);
	in = (in >> 8) | (in << 24);
#else
	//	 return rotate(in, (uint)8);
	in = (in << 8) | (in >> 24);
#endif
	return in;
}

/* key schedule core operation */
__device__ static inline uint sbox(uint in, uchar n)
{
	in = (SBOX[in & 255]) | (SBOX[(in >> 8) & 255] << 8) | (SBOX[(in >> 16) & 255] << 16) | (SBOX[(in >> 24) & 255] << 24);
#if __ENDIAN_LITTLE__
	in ^= n;
#else
	in ^= n << 24;
#endif
	return in;
}

// this version is optimized for exactly two rounds.
// _state_ must be 16-byte aligned.
__device__ static void aes2r_encrypt(uchar * state, uchar * key)
{
	uint __align__(16) key_schedule[12];
	uint t;

	/* initialize key schedule; its first 16 bytes are the key */
	*(uint4 *)key_schedule = *(uint4 *)key;
	t = key_schedule[3];

	t = rotate32(t);
	t = sbox(t, 1);
	t = key_schedule[4] = key_schedule[0] ^ t;
	t = key_schedule[5] = key_schedule[1] ^ t;
	t = key_schedule[6] = key_schedule[2] ^ t;
	t = key_schedule[7] = key_schedule[3] ^ t;

	t = rotate32(t);
	t = sbox(t, 2);
	t = key_schedule[8] = key_schedule[4] ^ t;
	t = key_schedule[9] = key_schedule[5] ^ t;
	t = key_schedule[10] = key_schedule[6] ^ t;
	t = key_schedule[11] = key_schedule[7] ^ t;

	/* first round of the algorithm */
	add_round_key(state, (uchar*)&key_schedule[0]);
	sub_bytes(state);
	shift_rows(state);
	mix_columns(state);
	add_round_key(state, (uchar*)&key_schedule[4]);

	/* final round of the algorithm */
	sub_bytes(state);
	shift_rows(state);
	add_round_key(state, (uchar*)&key_schedule[8]);
}


__device__ static inline uint rf_crc32_32(uint crc, uint msg)
{
	crc = crc ^ msg;
	crc = rf_crc32_table[crc & 0xff] ^ (crc >> 8);
	crc = rf_crc32_table[crc & 0xff] ^ (crc >> 8);
	crc = rf_crc32_table[crc & 0xff] ^ (crc >> 8);
	crc = rf_crc32_table[crc & 0xff] ^ (crc >> 8);

	return crc;
}

__device__  static inline uint rf_crc32_8(uint crc, uchar msg)
{
	crc = crc ^ msg;
	crc = rf_crc32_table[crc & 0xff] ^ (crc >> 8);
	return crc;
}

__device__  static inline ulong rf_crc32_64(uint crc2, ulong msg)
{
uint32_t crc = crc2;

	crc ^= (uint32_t)(msg & 0xffffffff);
	crc = rf_crc32_table[crc & 0xff] ^ (crc >> 8);
	crc = rf_crc32_table[crc & 0xff] ^ (crc >> 8);
	crc = rf_crc32_table[crc & 0xff] ^ (crc >> 8);
	crc = rf_crc32_table[crc & 0xff] ^ (crc >> 8);

	crc ^= (uint32_t)((msg >> 32) & 0xffffffff);
	crc = rf_crc32_table[crc & 0xff] ^ (crc >> 8);
	crc = rf_crc32_table[crc & 0xff] ^ (crc >> 8);
	crc = rf_crc32_table[crc & 0xff] ^ (crc >> 8);
	crc = rf_crc32_table[crc & 0xff] ^ (crc >> 8);
	return (uint64_t)crc;
}

__device__  static inline uint rf_crc32_mem(uint crc, const uint8_t *msg, size_t len)
{

	const uchar *msg8 = msg;

	while (len--) {
		crc = rf_crc32_8(crc, *msg8++);
	}
	return crc;
}

/////////////////////////////// same as rfv2_core.c ///////////////////////////

__device__  static inline uint rf_crc32x4(uint *state, uint crc)
{

	crc = state[0] = rf_crc32_32(crc, state[0]);
	crc = state[1] = rf_crc32_32(crc, state[1]);
	crc = state[2] = rf_crc32_32(crc, state[2]);
	crc = state[3] = rf_crc32_32(crc, state[3]);
	return crc;
}

__device__  static inline ulong rf_add64_crc32(ulong msg)
{
	return msg + rf_crc32_64(0, msg);
}

__device__ static inline ulong rf_memr64(const uchar * __restrict__ p)
{

	ulong ret;
	int byte;

	for (ret = byte = 0; byte < 8; byte++)
		ret += (ulong)p[byte] << (byte * 8);
	return ret;

}

__device__ static inline ulong rf_wltable(uchar index)
{
	return rf_memr64(&rfv2_table[index]);
}

__device__ static inline ulong rf_whtable(uchar index)
{
	return rf_memr64(&rfv2_table[index * 2]);
}

__device__ static inline ulong rf_rotl64(ulong v, uchar bits)
{
	bits &= 63;
	return ROTL64(v, bits);
}

__device__ static inline ulong rf_rotr64(ulong v, uchar bits)
{
	bits &= 63;
	return ROTR64(v, bits);
}

__device__ static inline ulong rf_bswap64(ulong v)
{
/*
	v = ((v & 0xff00ff00ff00ff00ULL) >> 8) | ((v & 0x00ff00ff00ff00ffULL) << 8);
	v = ((v & 0xffff0000ffff0000ULL) >> 16) | ((v & 0x0000ffff0000ffffULL) << 16);
	v = (v >> 32) | (v << 32);
*/
	return cuda_swab64(v);
//	return v;
}

__device__ static inline ulong rf_revbit64(ulong v)
{
	v = ((v & 0xaaaaaaaaaaaaaaaaULL) >> 1) | ((v & 0x5555555555555555ULL) << 1);
	v = ((v & 0xccccccccccccccccULL) >> 2) | ((v & 0x3333333333333333ULL) << 2);
	v = ((v & 0xf0f0f0f0f0f0f0f0ULL) >> 4) | ((v & 0x0f0f0f0f0f0f0f0fULL) << 4);
/*
	v = ((v & 0xff00ff00ff00ff00ULL) >> 8) | ((v & 0x00ff00ff00ff00ffULL) << 8);
	v = ((v & 0xffff0000ffff0000ULL) >> 16) | ((v & 0x0000ffff0000ffffULL) << 16);
	v = (v >> 32) | (v << 32);
*/
	return cuda_swab64(v);
}

__device__ static inline int __builtin_clzll(int64_t x)
{
	int64_t y;
	int n = 64;

	y = x >> 32; if (y) { x = y; n -= 32; }
	y = x >> 16; if (y) { x = y; n -= 16; }
	y = x >> 8; if (y) { x = y; n -= 8; }
	y = x >> 4; if (y) { x = y; n -= 4; }
	y = x >> 2; if (y) { x = y; n -= 2; }
	y = x >> 1; if (y) { x = y; n -= 1; }
	return n - x;
}


__device__ static inline int __builtin_clrsbll(int64_t  x)
{
	return __builtin_clzll((x<0) ? ~(x << 1) : (x << 1));
//	return __clzll((x<0) ? ~(x << 1) : (x << 1));
}

__device__ static uint64_t rfv2_rambox_mod(rfv2_ctx_t *ctx, ulong old,const uint64_t * __restrict__ RamBox)
{
	ulong p;
	ulong k,ktest;
	uint32_t idx = 0;
	uint event_thread = (blockDim.x * blockIdx.x + threadIdx.x);
 
	k = old;
	
	old = rf_add64_crc32(old);
	old ^= rf_revbit64(k);

	if (__builtin_clrsbll((int64_t)old) >= ctx->left_bits) {


		idx = (ctx->rb_o + (uint32_t)((old % ctx->rb_l) &0xffffffff));

	uint chg_idx;
	uint changed = 0;
  

	uint32_t div = idx/AGGR;
	uint32_t rest = idx%AGGR;
	uint16_t TheVal = __ldg(&ctx->LocalIndex[div]);
	uint32_t stored_rest = (TheVal >> 12) & 0xf;
	TheVal = TheVal & 0xfff;
/*
if (event_thread == 1) {
		printf("TheVal %08x ******************************\n",TheVal);
}
*/
if (TheVal<RFV2_RAMBOX_HIST && TheVal!=0)
	if (ctx->hist[TheVal] == idx) {
//		if (event_thread == 1)
//			printf("******************************already existing changes \n");
		p = ctx->prev[TheVal]; // = old;
		changed = 1;

	} 
/*
else 
		if (event_thread == 1)
			printf("not same full %08x reduced index %08x %08x how often %d TheVal %d\n",idx, div,rest, stored_rest,TheVal);
*/
		if (changed == 0 ) 
			 	p = __ldg(&RamBox[idx]);
		
		ktest = p;
		uint8_t bit = (uint8_t)((old / (uint64_t)ctx->rb_l) & 0xff);
		old += rf_rotr64(ktest, (uint8_t)((old / (uint64_t)ctx->rb_l) & 0xff));

		if (changed == 0) {
			if (ctx->gchanges < RFV2_RAMBOX_HIST) {
			int count = stored_rest + 1;
			ctx->LocalIndex[idx/ AGGR] = (count << 12) | ctx->changes;
			ctx->hist[ctx->changes] = idx;
			ctx->prev[ctx->changes] = old;
			ctx->changes++;
			}
		}
		else { 
			ctx->prev[TheVal /*& 0xfff*/] = old;

		}
			ctx->gchanges++;
	}
	return old;
}



__device__ static inline void rfv2_div_mod(ulong &p, ulong &q)
{
	ulong x = p;
	p = x / q;
	q = rf_revbit64(rf_revbit64(q) + x);
}

__device__ static inline void rfv2_divbox(ulong *v0, ulong *v1)
{
	ulong pl, ql, ph, qh;

	//---- low word ----    ---- high word ----
	pl = ~v0[0];              ph = ~v1[0];
	ql = rf_bswap64(v0[0]);   qh = rf_bswap64(v1[0]);


	if (!pl || !ql) { pl = ql = 0; }
	else if (pl > ql) rfv2_div_mod(pl, ql);
	else              rfv2_div_mod(ql, pl);

	if (!ph || !qh) { ph = qh = 0; }
	else if (ph > qh) rfv2_div_mod(ph, qh);
	else              rfv2_div_mod(qh, ph);

	pl += qh;               ph += ql;
	v0[0] -= pl;              v1[0] -= ph;
}

__device__ static inline void rfv2_rotbox(ulong *v0, ulong *v1, uchar b0, uchar b1)
{
	ulong l, h;

	//---- low word ----         ---- high word ----
	l = v0[0];                   h = v1[0];
	l = rf_rotr64(l, b0);        h = rf_rotl64(h, b1);
	l += rf_wltable(b0);         h += rf_whtable(b1);
	b0 = (uint8_t)l;                     b1 = (uint8_t)h;
	l = rf_rotl64(l, b1);        h = rf_rotr64(h, b0);
	b0 = (uint8_t)l;                     b1 = (uint8_t)h;
	l = rf_rotr64(l, b1);        h = rf_rotl64(h, b0);
	v0[0] = l;                     v1[0] = h;
}

__device__ static inline uint rfv2_scramble(rfv2_ctx_t *ctx)
{
	ctx->crc = rf_crc32x4(ctx->hash.d, ctx->crc);
	return ctx->crc;
}

__device__ static inline void rfv2_inject(rfv2_ctx_t *ctx)
{
	uint32_t truc = rfv2_scramble(ctx);
	ctx->crc = rf_crc32_32(truc, ctx->word);
	ctx->word = 0;
}

__device__ static inline void rfv2_rot32x256(hash256_t *hash)
{
/*
	uint8 h0, h1;

	h0 = *(uint8 *)hash;
	h1.s0 = h0.s7;
	h1.s1 = h0.s0;
	h1.s2 = h0.s1;
	h1.s3 = h0.s2;
	h1.s4 = h0.s3;
	h1.s5 = h0.s4;
	h1.s6 = h0.s5;
	h1.s7 = h0.s6;
	*(uint8 *)hash = h1;
*/

	uint32_t t0, t1, t2;

	t0 = hash->d[0];
	t1 = hash->d[1];
	t2 = hash->d[2];
	hash->d[1] = t0;
	hash->d[2] = t1;

	t0 = hash->d[3];
	t1 = hash->d[4];
	hash->d[3] = t2;
	hash->d[4] = t0;

	t2 = hash->d[5];
	t0 = hash->d[6];
	hash->d[5] = t1;
	hash->d[6] = t2;

	t1 = hash->d[7];
	hash->d[7] = t0;
	hash->d[0] = t1;


}

__device__ static inline void rfv2_aesenc(rfv2_ctx_t *ctx)
{
	aes2r_encrypt((uchar *)ctx->hash.b, (uchar *)ctx->hash.b + 16);
}

__device__ static inline void rfv2_one_round(rfv2_ctx_t *ctx,uint64_t * __restrict__ RamBox)
{
	ulong carry;
	uint event_thread = (blockDim.x * blockIdx.x + threadIdx.x);

	rfv2_rot32x256(&ctx->hash);
/*
	if (event_thread == 1)
	{
		printf("GPU hash  %08x %08x %08x %08x   %08x %08x %08x %08x   \n", ctx->hash.d[0], ctx->hash.d[1], ctx->hash.d[2], ctx->hash.d[3],
			ctx->hash.d[4], ctx->hash.d[5], ctx->hash.d[6], ctx->hash.d[7]);
	}
*/
	carry = ((uint64_t)ctx->len);
	carry = carry << 32;
	carry += (uint64_t)ctx->crc;

	rfv2_scramble(ctx);
	rfv2_divbox(ctx->hash.q, ctx->hash.q + 1);
	rfv2_scramble(ctx);
 
	carry = rfv2_rambox_mod(ctx, carry, RamBox);  

	rfv2_rotbox(ctx->hash.q, ctx->hash.q + 1, (uint8_t)((carry) & 0xff), (uint8_t)((carry >> 56) & 0xff));
	rfv2_scramble(ctx);
	rfv2_divbox(ctx->hash.q, ctx->hash.q + 1);
	rfv2_scramble(ctx);

	carry = rfv2_rambox_mod(ctx, carry, RamBox);

	rfv2_rotbox(ctx->hash.q, ctx->hash.q + 1, (uint8_t)((carry >> 8) & 0xff), (uint8_t)((carry >> 48) & 0xff));
	rfv2_scramble(ctx);
	rfv2_divbox(ctx->hash.q, ctx->hash.q + 1);
	rfv2_scramble(ctx);

	carry = rfv2_rambox_mod(ctx, carry, RamBox);
	rfv2_rotbox(ctx->hash.q, ctx->hash.q + 1, (uint8_t)((carry >> 16) & 0xff), (uint8_t)((carry >> 40) & 0xff));
	rfv2_scramble(ctx);
	rfv2_divbox(ctx->hash.q, ctx->hash.q + 1);
	rfv2_scramble(ctx);

	carry = rfv2_rambox_mod(ctx, carry, RamBox);
	rfv2_rotbox(ctx->hash.q, ctx->hash.q + 1, (uint8_t)((carry >> 24) & 0xff), (uint8_t)((carry >> 32) & 0xff));
	rfv2_scramble(ctx);
	rfv2_divbox(ctx->hash.q, ctx->hash.q + 1);
	rfv2_inject(ctx);
	rfv2_aesenc(ctx);
	rfv2_scramble(ctx);

//	ctx->crc = crc2;

}
