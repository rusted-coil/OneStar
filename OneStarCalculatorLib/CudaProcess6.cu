#include "CudaProcess6.cuh"
#include "Data.h"

//デバイスメモリのポインタ
static CudaInputMaster* pDeviceInput;
static _u32* pDeviceCoefficientData;
static _u32* pDeviceSearchPattern;
static int* pDeviceResultCount;
static _u64* pDeviceResult;

// 並列実行定数
const int c_SizeBlockX = 1024;
const int c_SizeGridX = 1024 * 512;
const int c_SizeResult = 32;

// 計算するカーネル
__global__ static void kernel_calc(
	CudaConst* pConst,
	CudaInputMaster* pSrc,
	_u32* pCoefficient,
	_u32* pSearchPattern,
	int* pResultCount,
	_u64* pResult,
	_u32 param)
{
	int targetId = blockDim.x * blockIdx.x + threadIdx.x; // 最大10bit + 19bit = 29bit

	param |= targetId; // 30bit

	_u32 targetUpper = 0;
	_u32 targetLower = 0;

	// 下位30bit = 個体値
	targetUpper |= (param & 0x3E000000ul); // iv0_0
	targetLower |= ((param & 0x7C00ul) << 15); // iv3_0
	targetUpper |= ((param & 0x1F00000ul) >> 5); // iv1_0
	targetLower |= ((param & 0x3E0ul) << 10); // iv4_0
	targetUpper |= ((param & 0xF8000ul) >> 10); // iv2_0
	targetLower |= ((param & 0x1Ful) << 5); // iv5_0

	// 隠された値を推定
	targetUpper |= ((32ul + pSrc->ivs[0] - ((param & 0x3E000000ul) >> 25)) & 0x1F) << 20;
	targetLower |= ((32ul + pSrc->ivs[3] - ((param & 0x7C00ul) >> 10)) & 0x1F) << 20;
	targetUpper |= ((32ul + pSrc->ivs[1] - ((param & 0x1F00000ul) >> 20)) & 0x1F) << 10;
	targetLower |= ((32ul + pSrc->ivs[4] - ((param & 0x3E0ul) >> 5)) & 0x1F) << 10;
	targetUpper |= ((32ul + pSrc->ivs[2] - ((param & 0xF8000ul) >> 15)) & 0x1F);
	targetLower |= ((32ul + pSrc->ivs[5] - (param & 0x1Ful)) & 0x1F);

	// targetベクトル入力完了

	targetUpper ^= pSrc->constantTermVector[0];
	targetLower ^= pSrc->constantTermVector[1];

	// 検索条件キャッシュ

	__shared__ _u32 answerFlag[128];
	__shared__ PokemonData pokemon[4];
	__shared__ int ecBit;
	__shared__ bool ecMod[3][6];

	if(threadIdx.x % 8 == 0)
	{
		answerFlag[threadIdx.x / 8] = pSrc->answerFlag[threadIdx.x / 8];
	}
	else if(threadIdx.x % 8 == 1)
	{
		pokemon[0] = pSrc->pokemon[0];
	}
	else if(threadIdx.x % 8 == 2)
	{
		pokemon[1] = pSrc->pokemon[1];
	}
	else if(threadIdx.x % 8 == 3)
	{
		pokemon[2] = pSrc->pokemon[2];
	}
	else if(threadIdx.x % 8 == 4)
	{
		pokemon[3] = pSrc->pokemon[3];
	}
	else if(threadIdx.x % 8 == 5)
	{
		ecBit = pSrc->ecBit;
	}
	else if(threadIdx.x % 8 == 6)
	{
		ecMod[0][0] = pSrc->ecMod[0][0];
		ecMod[0][1] = pSrc->ecMod[0][1];
		ecMod[0][2] = pSrc->ecMod[0][2];
		ecMod[0][3] = pSrc->ecMod[0][3];
		ecMod[0][4] = pSrc->ecMod[0][4];
		ecMod[0][5] = pSrc->ecMod[0][5];
		ecMod[1][0] = pSrc->ecMod[1][0];
		ecMod[1][1] = pSrc->ecMod[1][1];
		ecMod[1][2] = pSrc->ecMod[1][2];
	}
	else if(threadIdx.x % 8 == 7)
	{
		ecMod[1][3] = pSrc->ecMod[1][3];
		ecMod[1][4] = pSrc->ecMod[1][4];
		ecMod[1][5] = pSrc->ecMod[1][5];
		ecMod[2][0] = pSrc->ecMod[2][0];
		ecMod[2][1] = pSrc->ecMod[2][1];
		ecMod[2][2] = pSrc->ecMod[2][2];
		ecMod[2][3] = pSrc->ecMod[2][3];
		ecMod[2][4] = pSrc->ecMod[2][4];
		ecMod[2][5] = pSrc->ecMod[2][5];
	}

	__syncthreads();

	_u32 processedTargetUpper = 0;
	_u32 processedTargetLower = 0;
	for(int i = 0; i < 32; ++i)
	{
		processedTargetUpper |= (CudaGetSignature(answerFlag[i * 2] & targetUpper) ^ CudaGetSignature(answerFlag[i * 2 + 1] & targetLower)) << (31 - i);
		processedTargetLower |= (CudaGetSignature(answerFlag[(i + 32) * 2] & targetUpper) ^ CudaGetSignature(answerFlag[(i + 32) * 2 + 1] & targetLower)) << (31 - i);
	}

	_u32 seeds[7]; // S0Upper、S0Lower、S1Upper、S1Lower
	_u32 next[7]; // S0Upper、S0Lower、S1Upper、S1Lower
	_u64 temp64;
	_u32 temp32;
	_u32 temp32_2;
	for(int i = 0; i < 16; ++i)
	{
		seeds[0] = processedTargetUpper ^ pCoefficient[i * 2];
		seeds[1] = processedTargetLower ^ pCoefficient[i * 2 + 1] | pSearchPattern[i];

		// 遺伝箇所

		if(ecBit >= 0 && (seeds[1] & 1) != ecBit)
		{
			continue;
		}

		temp64 = ((_u64)seeds[0] << 32 | seeds[1]) + 0x82a2b175229d6a5bull;

		seeds[2] = 0x82a2b175ul;
		seeds[3] = 0x229d6a5bul;

		next[0] = (_u32)(temp64 >> 32);
		next[1] = (_u32)temp64;
		next[2] = 0x82a2b175ul;
		next[3] = 0x229d6a5bul;

		temp64 = ((_u64)seeds[0] << 32 | seeds[1]);

		// ここから絞り込み

		// EC
		temp32 = CudaNext(seeds, 0xFFFFFFFFu);
		// 1匹目個性
		if(ecMod[0][temp32 % 6] == false)
		{
			continue;
		}
		// 2匹目個性
		if(ecMod[1][temp32 % 6] == false)
		{
			continue;
		}

		// EC
		temp32 = CudaNext(next, 0xFFFFFFFFu);
		// 3匹目個性
		if(ecMod[2][temp32 % 6] == false)
		{
			continue;
		}

		// 2匹目を先にチェック
		CudaNext(next); // OTID
		CudaNext(next); // PID

		{
			// 個体値
			int ivs[8] = { -1, -1, -1, -1, -1, -1, 31, 31 };

			temp32 = 0;
			do {
				do {
					temp32_2 = CudaNext(next, 7u);
				} while(ivs[temp32_2] == 31);

				if(pokemon[2].ivs[temp32_2] != 31)
				{
					temp32 = 10;
					break;
				}

				ivs[temp32_2] = 31;
				++temp32;
			} while(temp32 < pokemon[2].flawlessIvs);

			if(temp32 == 10)
			{
				continue;
			}

			for(int i = 0; i < 6; ++i)
			{
				if(ivs[i] != 31)
				{
					if(pokemon[2].ivs[i] != CudaNext(next, 0x1Fu))
					{
						temp32 = 10;
						break;
					}
				}
			}

			if(temp32 == 10)
			{
				continue;
			}

			// 特性
			temp32 = 0;
			if(pokemon[2].abilityFlag == 3)
			{
				temp32 = CudaNext(next, 1u);
			}
			else
			{
				do {
					temp32 = CudaNext(next, 3u);
				} while(temp32 >= 3);
			}
			if((pokemon[2].ability >= 0 && pokemon[2].ability != temp32) || (pokemon[2].ability == -1 && temp32 >= 2))
			{
				continue;
			}

			// 性別値
			if(!pokemon[2].isNoGender)
			{
				temp32 = 0;
				do {
					temp32 = CudaNext(next, 0xFFu);
				} while(temp32 >= 253);
			}

			// 性格
			temp32 = 0;
			do {
				temp32 = CudaNext(next, pConst->natureTable[pokemon[2].natureTableId].randMax);
			} while(temp32 >= pConst->natureTable[pokemon[2].natureTableId].patternCount);

			if(pConst->natureTable[pokemon[2].natureTableId].natureId[temp32] != pokemon[2].nature)
			{
				continue;
			}
		}

		// 1匹目
		CudaNext(seeds); // OTID
		CudaNext(seeds); // PIT

		{
			// 状態を保存
			next[0] = seeds[0];
			next[1] = seeds[1];
			next[2] = seeds[2];
			next[3] = seeds[3];

			{
				// 個体値
				int ivs[8] = { -1, -1, -1, -1, -1, -1, 31, 31 };

				temp32 = 0;
				do {
					do {
						temp32_2 = CudaNext(seeds, 7u);
					} while(ivs[temp32_2] == 31);

					if(pokemon[0].ivs[temp32_2] != 31)
					{
						temp32 = 10;
						break;
					}

					ivs[temp32_2] = 31;
					++temp32;
				} while(temp32 < pokemon[0].flawlessIvs);

				if(temp32 == 10)
				{
					continue;
				}

				for(int i = 0; i < 6; ++i)
				{
					if(ivs[i] != 31)
					{
						if(pokemon[0].ivs[i] != CudaNext(seeds, 0x1Fu))
						{
							temp32 = 10;
							break;
						}
					}
				}

				if(temp32 == 10)
				{
					continue;
				}
			}
			{
				// 個体値
				int ivs[8] = { -1, -1, -1, -1, -1, -1, 31, 31 };

				temp32 = 0;
				do {
					do {
						temp32_2 = CudaNext(next, 7u);
					} while(ivs[temp32_2] == 31);

					if(pokemon[1].ivs[temp32_2] != 31)
					{
						temp32 = 10;
						break;
					}

					ivs[temp32_2] = 31;
					++temp32;
				} while(temp32 < pokemon[1].flawlessIvs);

				if(temp32 == 10)
				{
					continue;
				}

				for(int i = 0; i < 6; ++i)
				{
					if(ivs[i] != 31)
					{
						if(pokemon[1].ivs[i] != CudaNext(next, 0x1Fu))
						{
							temp32 = 10;
							break;
						}
					}
				}

				if(temp32 == 10)
				{
					continue;
				}
			}

			// 特性
			temp32 = 0;
			if(pokemon[0].abilityFlag == 3)
			{
				temp32 = CudaNext(seeds, 1u);
			}
			else
			{
				do {
					temp32 = CudaNext(seeds, 3u);
				} while(temp32 >= 3);
			}
			if((pokemon[0].ability >= 0 && pokemon[0].ability != temp32) || (pokemon[0].ability == -1 && temp32 >= 2))
			{
				continue;
			}
			temp32 = 0;
			if(pokemon[1].abilityFlag == 3)
			{
				temp32 = CudaNext(next, 1u);
			}
			else
			{
				do {
					temp32 = CudaNext(next, 3u);
				} while(temp32 >= 3);
			}
			if((pokemon[1].ability >= 0 && pokemon[1].ability != temp32) || (pokemon[1].ability == -1 && temp32 >= 2))
			{
				continue;
			}

			// 性別値
			if(!pokemon[0].isNoGender)
			{
				temp32 = 0;
				do {
					temp32 = CudaNext(seeds, 0xFFu);
				} while(temp32 >= 253);
			}
			if(!pokemon[1].isNoGender)
			{
				temp32 = 0;
				do {
					temp32 = CudaNext(next, 0xFFu);
				} while(temp32 >= 253);
			}

			// 性格
			temp32 = 0;
			do {
				temp32 = CudaNext(seeds, pConst->natureTable[pokemon[0].natureTableId].randMax);
			} while(temp32 >= pConst->natureTable[pokemon[0].natureTableId].patternCount);

			if(pConst->natureTable[pokemon[0].natureTableId].natureId[temp32] != pokemon[0].nature)
			{
				continue;
			}

			temp32 = 0;
			do {
				temp32 = CudaNext(next, pConst->natureTable[pokemon[1].natureTableId].randMax);
			} while(temp32 >= pConst->natureTable[pokemon[1].natureTableId].patternCount);

			if(pConst->natureTable[pokemon[1].natureTableId].natureId[temp32] != pokemon[1].nature)
			{
				continue;
			}
		}
		// 結果を書き込み
		int old = atomicAdd(pResultCount, 1);
		pResult[old] = temp64;
	}
	return;
}

// メモリ初期化
void Cuda6Initialize()
{
	// ホストメモリの確保
	cudaMallocHost(&cu_HostInputCoefficientData, sizeof(_u32) * 32);
	cudaMallocHost(&cu_HostInputSearchPattern, sizeof(_u32) * 16);

	// デバイスメモリの確保
	cudaMalloc(&pDeviceInput, sizeof(CudaInputMaster));
	cudaMalloc(&pDeviceCoefficientData, sizeof(_u32) * 32);
	cudaMalloc(&pDeviceSearchPattern, sizeof(_u32) * 16);
	cudaMalloc(&pDeviceResultCount, sizeof(int));
	cudaMalloc(&pDeviceResult, sizeof(_u64) * c_SizeResult);
}

// データセット
void Cuda6SetMasterData()
{
	// ホストデータの設定
	cu_HostInputMaster->constantTermVector[0] = (_u32)(g_ConstantTermVector >> 30);
	cu_HostInputMaster->constantTermVector[1] = (_u32)(g_ConstantTermVector & 0x3FFFFFFFull);
	for(int i = 0; i < 64; ++i)
	{
		cu_HostInputMaster->answerFlag[i * 2] = (_u32)(g_AnswerFlag[i] >> 30);
		cu_HostInputMaster->answerFlag[i * 2 + 1] = (_u32)(g_AnswerFlag[i] & 0x3FFFFFFFull);
	}
	for(int i = 0; i < 16; ++i)
	{
		cu_HostInputCoefficientData[i * 2] = (_u32)(g_CoefficientData[i] >> 32);
		cu_HostInputCoefficientData[i * 2 + 1] = (_u32)(g_CoefficientData[i] & 0xFFFFFFFFull);
		cu_HostInputSearchPattern[i] = (_u32)g_SearchPattern[i];
	}

	// データを転送
	cudaMemcpy(pDeviceInput, cu_HostInputMaster, sizeof(CudaInputMaster), cudaMemcpyHostToDevice);
	cudaMemcpy(pDeviceCoefficientData, cu_HostInputCoefficientData, sizeof(_u32) * 32, cudaMemcpyHostToDevice);
	cudaMemcpy(pDeviceSearchPattern, cu_HostInputSearchPattern, sizeof(_u32) * 16, cudaMemcpyHostToDevice);
	cudaMemcpy(pDeviceResultCount, cu_HostResultCount, sizeof(int), cudaMemcpyHostToDevice);
}

// 計算
void Cuda6Process(_u32 param, int partition)
{
	// 結果をリセット
	*cu_HostResultCount = 0;
	cudaMemcpy(pDeviceResultCount, cu_HostResultCount, sizeof(int), cudaMemcpyHostToDevice);

	//カーネル
	dim3 block(c_SizeBlockX, 1, 1);
	dim3 grid(c_SizeGridX / partition, 1, 1);
	kernel_calc << < grid, block >> > (cu_DeviceConstData, pDeviceInput, pDeviceCoefficientData, pDeviceSearchPattern, pDeviceResultCount, pDeviceResult, param);

	//デバイス->ホストへ結果を転送
	cudaMemcpy(cu_HostResult, pDeviceResult, sizeof(_u64) * c_SizeResult, cudaMemcpyDeviceToHost);
	cudaMemcpy(cu_HostResultCount, pDeviceResultCount, sizeof(int), cudaMemcpyDeviceToHost);
}

void Cuda6Finalize()
{
	//デバイスメモリの開放
	cudaFree(pDeviceResult);
	cudaFree(pDeviceResultCount);
	cudaFree(pDeviceSearchPattern);
	cudaFree(pDeviceCoefficientData);
	cudaFree(pDeviceInput);
	//ホストメモリの開放
	cudaFreeHost(cu_HostInputSearchPattern);
	cudaFreeHost(cu_HostInputCoefficientData);
}
