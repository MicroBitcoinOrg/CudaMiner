/**
 * Rainforest algorithm
 * djm34 2019
 */

#include <string.h>
#include <stdint.h>


#include "../sph/rainforest.h"


#include <cuda_helper.h>
#include <miner.h>

#define A 64
#define debug_cpu 0

/* ############################################################################################################################### */


extern void rainforest_init(int thr_id, uint32_t threads);
extern void rainforest_setBlockTarget(int thr_id, const void* pDataIn, const void *pTargetIn, const void * zElement);
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

	unsigned char ThePreData[4128*4];
	rainforest_precompute(endiandata,ThePreData);

//	rainforest_precompute(pdata, ThePreData);
	rainforest_setBlockTarget(thr_id, endiandata,ptarget,ThePreData);
	do {
		
		work->nonces[0] = rainforest_cpu_hash(thr_id, throughput, pdata[19]);

		if (work->nonces[0] != UINT32_MAX)
		{
		be32enc(&endiandata[19], work->nonces[0]);
		
		rf256_hash(hash, endiandata, 80);

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
			break;
		}
		pdata[19] += throughput;

	} while (nonce < max_nonce && !work_restart[thr_id].restart);
	*hashes_done = pdata[19] - first_nonce;
	return 0;
}
