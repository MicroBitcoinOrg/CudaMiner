/**
 * Rainforest algorithm
 * djm34 2019
 */

#include <string.h>
#include <stdint.h>


//#include "../sph/rainforest.h"
#include "../sph/rfv2.h"

#include <cuda_helper.h>
#include <miner.h>
#define RFV2_RAMBOX_SIZE (96*1024*1024/8)
#define A 64
#define debug_cpu 0

/* ############################################################################################################################### */


extern void rainforest_init(int thr_id, uint32_t threads);
//extern void rainforest_setBlockTarget(int thr_id, const void* pDataIn, const void *pTargetIn, const void * zElement);
//extern void rainforest_setBlockTarget(int thr_id, const void* pDataIn, const void *pTargetIn,
//	const void * zElement, const void * carry);

extern void rainforest_setBlockTarget(int thr_id, int throughput, const void* pDataIn, const void *pTargetIn,
	const void * zElement, const void * carry, const void * box);

extern uint32_t rainforest_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce);

static bool init[MAX_GPUS] = { 0 };



extern "C" int scanhash_rf256(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t _ALIGN(64) hash[8];
	uint32_t _ALIGN(64) endiandata[20];
	uint32_t *pdata = work->data;
	uint32_t *ptarget = work->target;
	const uint32_t Htarg = ptarget[7];
	const uint32_t first_nonce = pdata[19];
	uint32_t nonce = first_nonce;

//	rf256_ctx_t ctx, ctx_common;

	const int dev_id = device_map[thr_id];
	int intensity = (device_sm[dev_id] > 500 && !is_windows()) ? 22 : 20;
	if (device_sm[dev_id] >= 600) intensity = 23;
	if (device_sm[dev_id] < 350) intensity = 18;

	uint32_t throughput = cuda_default_throughput(thr_id, 1U << intensity);
	if (init[thr_id]) throughput = min(throughput, max_nonce - first_nonce);

	throughput = 1U << 13;
	if (opt_benchmark) {
		ptarget[7] = 0x0cff;
	}


	if (!init[thr_id]) {
		cudaSetDevice(dev_id);
//		if (opt_cudaschedule == -1 && gpu_threads == 1) {
			cudaDeviceReset();
			// reduce cpu usage
			cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			CUDA_LOG_ERROR();
//		}
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), 
			throughput);

		rainforest_init(thr_id,throughput);
		CUDA_LOG_ERROR();
		init[thr_id] = true;
	}


	for (int k = 0; k < 19; k++) {
		be32enc(&endiandata[k], pdata[k]);
	}

	// pre-compute the hash state based on the constant part of the header

//	unsigned char ThePreData[128];
	unsigned char ThePreData[128];
	uint64_t TheCarry[5];
//	uint64_t   TheRambox[96 * 1024 * 1024 / 8];
	uint64_t *TheRambox = (uint64_t*)malloc(96*1024*1024);


	unsigned char *TheTest = (unsigned char*)malloc(96 * 1024 * 1024/8);
//	uint64_t *TheRambox2 = (uint64_t*)malloc(96 * 1024 * 1024);
	rfv2_raminit(TheRambox);
//	rainforest_precompute3(endiandata,ThePreData,TheCarry,TheRambox);

//	rainforest_precompute(pdata, ThePreData);
	rainforest_setBlockTarget(thr_id, throughput,endiandata,ptarget,ThePreData,TheCarry,TheRambox);
//	do {
		
		work->nonces[0] = rainforest_cpu_hash(thr_id, throughput, pdata[19]);
//		work->nonces[0] = pdata[19] + 1;
		if (work->nonces[0] != UINT32_MAX)
		{
		be32enc(&endiandata[19], work->nonces[0]);
//		uint64_t *rambox = (uint64_t*)malloc(96 * 1024 * 1024);
//		rfv2_hash(hash, endiandata, 80, rambox, TheRambox);
		{
			rfv2_ctx_t ctx;
			unsigned int loop, loops;
			
			uint32_t msgh;
//			rfv2_raminit(TheRambox);
//			rfv2_init(&ctx, 20180213, TheRambox);

			rfv2_init_test(&ctx, 20180213, TheRambox,TheTest);
			


			msgh = rf_crc32_mem(0, endiandata, 80);
			ctx.rb_o = msgh % (ctx.rb_l / 2);
			ctx.rb_l = (ctx.rb_l / 2 - ctx.rb_o) * 2;

			printf("CPU rb_o = %08x rb_l = %08x \n", ctx.rb_o, ctx.rb_l);

			loops = sin_scaled(msgh);

			printf("msgh = %08x loops = %d\n",  msgh, loops);

			if (loops >= 128)
				ctx.left_bits = 4;
			else if (loops >= 64)
				ctx.left_bits = 3;
			else if (loops >= 32)
				ctx.left_bits = 2;
			else if (loops >= 16)
				ctx.left_bits = 1;
			else
				ctx.left_bits = 0;

			for (loop = 0; loop < loops; loop++) {
				rfv2_update(&ctx, endiandata, 80);
				// pad to the next 256 bit boundary
				rfv2_pad256(&ctx);
			}

			rfv2_final(hash, &ctx);
printf("number of changes %d\n",ctx.changes);
		}




//		rf256_hash(hash, endiandata, 80);
	printf("CPU hash %08x %08x %08x %08x   %08x %08x %08x %08x \n",hash[0],hash[1],hash[2],hash[3],
		hash[4], hash[5], hash[6], hash[7]
);
		if (((uint64_t*)hash)[3] <= ((uint64_t*)ptarget)[3]) {
//			if (hash[7] <= Htarg && fulltest(hash, ptarget)) {
			int res = 1;
			work_set_target_ratio(work, hash);
			pdata[19] = work->nonces[0];
			*hashes_done = pdata[19] - first_nonce;
			return res;
		}
		else {
			gpulog(LOG_WARNING, thr_id, "result for %08x does not validate on CPU!", work->nonces[0]);
		}


		}
	

//////////////////////////////////////////////////////
		if ((uint64_t)throughput + pdata[19] >= max_nonce) {
			pdata[19] = max_nonce;
//			break;
		}
		pdata[19] += throughput;

//	} while (nonce < max_nonce && !work_restart[thr_id].restart);
	free(TheRambox);
	*hashes_done = pdata[19] - first_nonce;
	return 0;
}
