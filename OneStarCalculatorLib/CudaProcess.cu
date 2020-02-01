#include "cuda_runtime.h"
#include "CudaProcess.cuh"
#include "Data.h"

//ホストメモリのポインタ
CudaInputMaster* pHostMaster; // 固定データ
_u64* cu_HostResult;

//デバイスメモリのポインタ
CudaInputMaster* pDeviceMaster;
_u64* pDeviceResult;

__device__ inline _u64 GetSignature(_u64 value)
{
	unsigned int a = (unsigned int)(value ^ (value >> 32));
	a = a ^ (a >> 16);
	a = a ^ (a >> 8);
	a = a ^ (a >> 4);
	a = a ^ (a >> 2);
	return (a ^ (a >> 1)) & 1;
}

// 計算するカーネル
__global__ void kernel_calc(CudaInputMaster* pSrc, _u64 *pResult, _u64 ivs)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x; //自分のスレッドxのindex

	ivs |= idx;

	_u64 target = 0;

	// 下位30bit = 個体値
	target |= (ivs & 0x3E000000ul) << 30; // iv0_0
	target |= (ivs & 0x1F00000ul) << 25; // iv1_0
	target |= (ivs & 0xF8000ul) << 20; // iv2_0
	target |= (ivs & 0x7C00ul) << 15; // iv3_0
	target |= (ivs & 0x3E0ul) << 10; // iv4_0
	target |= (ivs & 0x1Ful) << 5; // iv5_0

	// 隠された値を推定
	target |= ((32ul + pSrc->ivs[0] - ((ivs & 0x3E000000ul) >> 25)) & 0x1F) << 50;
	target |= ((32ul + pSrc->ivs[1] - ((ivs & 0x1F00000ul) >> 20)) & 0x1F) << 40;
	target |= ((32ul + pSrc->ivs[2] - ((ivs & 0xF8000ul) >> 15)) & 0x1F) << 30;
	target |= ((32ul + pSrc->ivs[3] - ((ivs & 0x7C00ul) >> 10)) & 0x1F) << 20;
	target |= ((32ul + pSrc->ivs[4] - ((ivs & 0x3E0ul) >> 5)) & 0x1F) << 10;
	target |= ((32ul + pSrc->ivs[5] - (ivs & 0x1Ful)) & 0x1F);

	// targetベクトル入力完了

	target ^= pSrc->constantTermVector;
	// 60bit側の計算結果キャッシュ

	_u64 processedTarget = 0;
	_u64 v;
//	unsigned int a;
	for(int i = 0; i < 60; ++i)
	{
		processedTarget |= (GetSignature(pSrc->answerFlag[i] & target) << (63 - i));
		/*
		v = pSrc->answerFlag[i] & target;
		v = (v ^ (v >> 32));
		v = v ^ (v >> 16);
		v = v ^ (v >> 8);
		v = v ^ (v >> 4);
		v = v ^ (v >> 2);
		processedTarget |= ((v ^ (v >> 1)) & 1) << (63 - i);
		*/
	}

	pResult[idx] = processedTarget;
	return;
}

// 初期化
void CudaInitialize(int* pIvs)
{
	// ホストメモリの確保
	cudaMallocHost(&pHostMaster, sizeof(CudaInputMaster));
	cudaMallocHost(&cu_HostResult, sizeof(_u64) * 1024 * 1024 * 16);

	// デバイスメモリの確保
	cudaMalloc(&pDeviceMaster, sizeof(CudaInputMaster));
	cudaMalloc(&pDeviceResult, sizeof(_u64) * 1024 * 1024 * 16);

	// マスターデータのセット
	for(int i = 0; i < 6; ++i)
	{
		pHostMaster->ivs[i] = pIvs[i];
	}
	pHostMaster->constantTermVector = g_ConstantTermVector;
	for(int i = 0; i < 64; ++i)
	{
		pHostMaster->answerFlag[i] = g_AnswerFlag[i];
	}

	// データを転送
	cudaMemcpy(pDeviceMaster, pHostMaster, sizeof(CudaInputMaster), cudaMemcpyHostToDevice);
}

// 計算
void CudaProcess(_u64 ivs, int freeBit)
{
	//カーネル
	dim3 block(1024, 1, 1);
	dim3 grid(1024*16, 1, 1);
	kernel_calc << < grid, block >> > (pDeviceMaster, pDeviceResult, ivs);

	//デバイス->ホストへ結果を転送
	cudaMemcpy(cu_HostResult, pDeviceResult, sizeof(_u64) * 1024 * 1024 * 16, cudaMemcpyDeviceToHost);
}

void Finish()
{
	//デバイスメモリの開放
	cudaFree(pDeviceMaster);
	cudaFree(pDeviceResult);
	//ホストメモリの開放
	cudaFreeHost(pHostMaster);
	cudaFreeHost(cu_HostResult);
}
