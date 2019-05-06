


#include "rainforest_function.h"
#include "lyra2/cuda_lyra2_vectors.h" 

//#include "math_functions.h" 


__device__ static void rfv2_update(rfv2_ctx_t *ctx, const void *msg, size_t len, uint64_t * __restrict__ RamBox)
{
	const uchar *msg8 = (const uchar *)msg;

	while (len > 0) {

		if (!(ctx->len & 3) && len >= 4) {
			ctx->word = *(uint *)msg8;
			ctx->len += 4;
			rfv2_one_round(ctx,RamBox);
			msg8 += 4;
			len -= 4;
			continue;
		}
/*
		ctx->word |= ((uint)*msg8++) << (8 * (ctx->len++ & 3));
		len--;
		if (!(ctx->len & 3))
			rfv2_one_round(ctx,RamBox);
*/
	}  
} 

__device__ static inline void rfv2_pad256(rfv2_ctx_t *ctx, uint64_t * __restrict__ RamBox)
{
	const uchar pad256[32] = { 0, };
	uint pad; 

	pad = (32 - ctx->len) & 0xF;
	if (pad)
		rfv2_update(ctx, pad256, pad,  RamBox);   
}  

__device__ static void rfv2_final( rfv2_ctx_t *ctx, uint64_t * __restrict__ RamBox)
{ 

	rfv2_one_round(ctx, RamBox);
	rfv2_one_round(ctx, RamBox);     

/*
	rfv2_one_round(ctx, RamBox);
	rfv2_one_round(ctx, RamBox);
	rfv2_one_round(ctx, RamBox); 
*/
}

__device__ static uint32_t sin_scaled(uint x)
{
	int i;

	i = ((x * 42722829) >> 24) - 128;
	x = 15 * i * i * abs(i);  // 0 to 15<<21
	x = (x + (x >> 4)) >> 17;
	return 257 - x;
}
  
////////////////////////// equivalent of rfv2_cpuminer.c ////////////////////////
         
#define TPB 64 
__global__ __launch_bounds__(TPB, 8)
void rf256v2_hash_gpu(uint32_t thr_id, uint32_t threads, uint32_t startNounce, uint32_t *   output, uint64_t * __restrict__ DieRambox, uint16_t * __restrict__ DieIndex)
{ 
 
	const uint32_t rfv2_iv[8] = { 0xd390e978,  0x7b9bc8b3,  0x6e86c40a,  0x6bb3384e,  0xed7c6833,  0x0a4b3573,  0x774c2597,  0x1b61aa7a };
	uint event_thread = (blockDim.x * blockIdx.x + threadIdx.x);
 
		uint64_t * __restrict__ RamBox = &DieRambox[0];
		rfv2_ctx_t ctx; 
		uint32_t data[20];
		uint32_t NonceIterator = cuda_swab32(startNounce + event_thread);
		((uint16 *)data)[0] = ((const uint16 *)pData)[0];
		((uint4 *)data)[4] = ((const uint4 *)pData)[4];
		data[19] = NonceIterator;
 
	uint loop, loops; 
	uint msgh;  

	((uint8*)ctx.hash.d)[0] = ((const uint8 *)rfv2_iv)[0];
	
	ctx.crc = RFV2_INIT_CRC;  
	ctx.word = ctx.len = 0; 
	ctx.changes = 0;    
	ctx.gchanges = 0;
	ctx.rb_o = 0;       
	ctx.rb_l = RFV2_RAMBOX_SIZE/2; 
	ctx.LocalIndex = &DieIndex[RFV2_RAMBOX_SIZE*event_thread/ AGGR];

	msgh = rf_crc32_mem(0, (uint8_t*)data, 80);
	ctx.rb_o = msgh % ctx.rb_l; 
	ctx.rb_l = (ctx.rb_l - ctx.rb_o) * 2;

	loops = sin_scaled(msgh);  

	ctx.left_bits = (loops >= 128)? 4 : (loops >= 64) ? 3 : (loops >= 32) ? 2 : (loops >= 16) ? 1 : 0;


/*  
	if (event_thread == 1) 
	{
	printf("rb_o = %08x rb_l = %08x \n", ctx.rb_o, ctx.rb_l);
	printf("event_thread = %d msgh = %08x loops = %d\n",event_thread,msgh,loops);
	}
*/
	for (loop = 0; loop < loops; loop++) {      
		rfv2_update(&ctx, (uint8_t*)data, 80, RamBox);                         
		// pad to the next 256 bit boundary 
		rfv2_pad256(&ctx, RamBox);
	} 
 
	rfv2_final( &ctx, RamBox);  

	uint64_t Sol = MAKE_ULONGLONG(ctx.hash.d[3], ctx.hash.d[4]);


	if (Sol <= ((uint64_t*)pTarget)[3]) {
//	if (ctx.hash.q[3] <= ((uint64_t*)pTarget)[3]) {

/*
		printf("GPU hash  %08x %08x %08x %08x   %08x %08x %08x %08x   \n", ctx.hash.d[0], ctx.hash.d[1], ctx.hash.d[2], ctx.hash.d[3],
			ctx.hash.d[4], ctx.hash.d[5], ctx.hash.d[6], ctx.hash.d[7]);
	printf("GPU number of changes %d global changes %d\n", ctx.changes, ctx.gchanges);
*/
		atomicMin(&output[0], cuda_swab32(NonceIterator));
	}
 
	for (int i = 0; i<ctx.changes; i++)
		ctx.LocalIndex[ctx.hist[i]/AGGR] = 0;

}




__host__
void rainforest_init(int thr_id, uint32_t threads, const void *box)
{  
//	cudaSetDevice(device_map[thr_id]);
	// just assign the device pointer allocated in main loop

	//	cudaMemcpyToSymbol(GYLocal,&hash1[thr_id], 8 * sizeof(uint32_t) * threads);
	//	cudaMalloc((void**)&GYLocal[thr_id], 8 * sizeof(uint32_t) * threads);
uint32_t aggr_size =(uint32_t) (RFV2_RAMBOX_SIZE/AGGR);
CUDA_SAFE_CALL(cudaMalloc((void**)&TheRamBox[thr_id], 1 * RFV2_RAMBOX_SIZE * sizeof(uint64_t)));
CUDA_SAFE_CALL(cudaMalloc((void**)&TheIndex[thr_id],  threads * aggr_size * sizeof(uint16_t)));
CUDA_SAFE_CALL(cudaMalloc(&d_aMinNonces[thr_id], 2 * sizeof(uint32_t)));
CUDA_SAFE_CALL(cudaMallocHost(&h_aMinNonces[thr_id], 2 * sizeof(uint32_t)));

uint16_t *TheCarry = (uint16_t*)calloc(threads * aggr_size, sizeof(uint16_t));

uint64_t *Boxptr1 = &TheRamBox[thr_id][0];
CUDA_SAFE_CALL(cudaMemcpyAsync(Boxptr1, box, RFV2_RAMBOX_SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice));


uint16_t *Boxptr = &TheIndex[thr_id][0];
CUDA_SAFE_CALL(cudaMemcpyAsync(Boxptr, TheCarry, threads * aggr_size * sizeof(uint16_t), cudaMemcpyHostToDevice));
free(TheCarry);


//	cudaMalloc(&Header[thr_id], sizeof(uint32_t) * 8); 
//	cudaMalloc(&buffer_a[thr_id], 4194304 * 64);
}
  
 


__host__
void rainforest_setBlockTarget(int thr_id, int throughput, const void* pDataIn, const void *pTargetIn)
{
	//	cudaSetDevice(device_map[thr_id]);

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(pData, pDataIn, 80, 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(pTarget, pTargetIn, 8 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
}


__host__
uint32_t rainforest_cpu_hash(int thr_id, uint32_t threads, uint32_t startNounce)
{
//	cudaSetDevice(device_map[thr_id]);
	uint32_t result[1] ={ UINT32_MAX};
	CUDA_SAFE_CALL(cudaMemset(d_aMinNonces[thr_id], 0xff, 2* sizeof(uint32_t)));
	CUDA_SAFE_CALL(cudaMemset(h_aMinNonces[thr_id], 0xff, 2 *sizeof(uint32_t)));
//	CUDA_SAFE_CALL(cudaMemset(h_MinNonces[thr_id], 0xff, sizeof(uint32_t)));
//	int dev_id = device_map[thr_id % MAX_GPUS];

	uint32_t tpb = TPB;

	dim3 gridyloop(threads / tpb);
	dim3 blockyloop(tpb);

	rf256v2_hash_gpu << < gridyloop, blockyloop >> >(thr_id, threads, startNounce, d_aMinNonces[thr_id],TheRamBox[thr_id],TheIndex[thr_id]);

	CUDA_SAFE_CALL(cudaMemcpy(h_aMinNonces[thr_id], d_aMinNonces[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost));
//	CUDA_SAFE_CALL(cudaMemset(d_aMinNonces[thr_id], 0xff, sizeof(uint32_t)));
//	CUDA_SAFE_CALL(cudaMemcpy(result, d_aMinNonces[thr_id], sizeof(uint32_t), cudaMemcpyDeviceToHost));

	result[0] = h_aMinNonces[thr_id][0];
	return result[0];

}
