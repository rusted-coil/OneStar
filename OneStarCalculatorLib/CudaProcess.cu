#include "cuda_runtime.h"
#include "CudaProcess.cuh"
#include "Data.h"

//ホストメモリのポインタ
CudaInputMaster* cu_HostMaster;
_u64* cu_HostResult;

//デバイスメモリのポインタ
static CudaInputMaster* pDeviceMaster;
static _u64* pDeviceResult;

// 並列実行定数
const int c_SizeBlockX = 64;
const int c_SizeBlockY = 16;
const int c_SizeGrid = 1024 * 16;
const int c_SizeResult = 16;

// GPUコード
__device__ inline _u32 GetSignature(_u32 value)
{
	value ^= (value >> 16);
	value ^= (value >>  8);
	value ^= (value >>  4);
	value ^= (value >>  2);
	return (value ^ (value >> 1)) & 1;
}
__device__ inline _u32 Next(_u32* seeds, _u32 mask)
{
	_u32 value = (seeds[1] + seeds[3]) & mask;

	// m_S1 = m_S0 ^ m_S1;
	seeds[4] = seeds[0] ^ seeds[2];
	seeds[5] = seeds[1] ^ seeds[3];

	// m_S1 = RotateLeft(m_S1, 37);
	seeds[2] = seeds[5] << 5 | seeds[4] >> 27;
	seeds[3] = seeds[4] << 5 | seeds[5] >> 27;

	// m_S0 = RotateLeft(m_S0, 24) ^ m_S1 ^ (m_S1 << 16)
	seeds[6] = (seeds[0] << 24 | seeds[1] >> 8) ^ seeds[4] ^ (seeds[4] << 16 | seeds[5] >> 16);
	seeds[1] = (seeds[1] << 24 | seeds[0] >> 8) ^ seeds[5] ^ (seeds[5] << 16);

	seeds[0] = seeds[6];

	return value;
}
__device__ inline void Next(_u32* seeds)
{
	// m_S1 = m_S0 ^ m_S1;
	seeds[4] = seeds[0] ^ seeds[2];
	seeds[5] = seeds[1] ^ seeds[3];

	// m_S1 = RotateLeft(m_S1, 37);
	seeds[2] = seeds[5] << 5 | seeds[4] >> 27;
	seeds[3] = seeds[4] << 5 | seeds[5] >> 27;

	// m_S0 = RotateLeft(m_S0, 24) ^ m_S1 ^ (m_S1 << 16)
	seeds[6] = (seeds[0] << 24 | seeds[1] >> 8) ^ seeds[4] ^ (seeds[4] << 16 | seeds[5] >> 16);
	seeds[1] = (seeds[1] << 24 | seeds[0] >> 8) ^ seeds[5] ^ (seeds[5] << 16);

	seeds[0] = seeds[6];
}

// 計算するカーネル
__global__ void kernel_calc(CudaInputMaster* pSrc, _u64 *pResult, _u32 ivs)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x; //自分のスレッドxのindex
	int idy = blockDim.y * blockIdx.y + threadIdx.y;

	ivs |= idx;

	_u32 targetUpper = 0;
	_u32 targetLower = 0;

	// 下位30bit = 個体値
	targetUpper |= (ivs & 0x3E000000ul); // iv0_0
	targetLower |= ((ivs &    0x7C00ul) << 15); // iv3_0
	targetUpper |= ((ivs & 0x1F00000ul) >> 5); // iv1_0
	targetLower |= ((ivs &     0x3E0ul) << 10); // iv4_0
	targetUpper |= ((ivs &   0xF8000ul) >> 10); // iv2_0
	targetLower |= ((ivs &      0x1Ful) << 5); // iv5_0

	// 隠された値を推定
	targetUpper |= ((32ul + pSrc->ivs[0] - ((ivs & 0x3E000000ul) >> 25)) & 0x1F) << 20;
	targetLower |= ((32ul + pSrc->ivs[3] - ((ivs &     0x7C00ul) >> 10)) & 0x1F) << 20;
	targetUpper |= ((32ul + pSrc->ivs[1] - ((ivs &  0x1F00000ul) >> 20)) & 0x1F) << 10;
	targetLower |= ((32ul + pSrc->ivs[4] - ((ivs &      0x3E0ul) >> 5)) & 0x1F) << 10;
	targetUpper |= ((32ul + pSrc->ivs[2] - ((ivs &    0xF8000ul) >> 15)) & 0x1F);
	targetLower |= ((32ul + pSrc->ivs[5] - (ivs &        0x1Ful)) & 0x1F);
//	targetLower |= ((32ul + idy - (ivs &        0x1Ful)) & 0x1F);

	// targetベクトル入力完了

	targetUpper ^= pSrc->constantTermVector[0];
	targetLower ^= pSrc->constantTermVector[1];

	// 60bit側の計算結果キャッシュ

	__shared__ _u32 processedTargetUpper[64];
	__shared__ _u32 processedTargetLower[64];

//	_u32 processedTargetUpper = 0;
//	_u32 processedTargetLower = 0;
	processedTargetUpper[threadIdx.x] = 0;
	processedTargetLower[threadIdx.x] = 0;
	for(int i = 0; i < 32; ++i)
	{
		processedTargetUpper[threadIdx.x] |= (GetSignature(pSrc->answerFlag[i * 2] & targetUpper) ^ GetSignature(pSrc->answerFlag[i * 2 + 1] & targetLower)) << (31 - i);
		processedTargetLower[threadIdx.x] |= (GetSignature(pSrc->answerFlag[(i + 32) * 2] & targetUpper) ^ GetSignature(pSrc->answerFlag[(i + 32) * 2 + 1] & targetLower)) << (31 - i);
	}

	// スレッドを同期
	__syncthreads();

	_u32 seeds[7]; // S0Upper、S0Lower、S1Upper、S1Lower
	_u32 next[7]; // S0Upper、S0Lower、S1Upper、S1Lower
	_u64 temp64;
	_u32 temp32;
//	for(int i = 0; i < 16; ++i)
	{
		seeds[0] = processedTargetUpper[threadIdx.x] ^ pSrc->coefficientData[idy * 2];
		seeds[1] = processedTargetLower[threadIdx.x] ^ pSrc->coefficientData[idy * 2 + 1] | pSrc->searchPattern[idy];

		// 遺伝箇所

		if(pSrc->ecBit >= 0 && (seeds[1] & 1) != pSrc->ecBit)
		{
			return;
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
		temp32 = Next(seeds, 0xFFFFFFFFu);
		// 1匹目個性
		if(pSrc->ecMod[0][temp32 % 6] == false)
		{
			return;
		}
		// 2匹目個性
		if(pSrc->ecMod[1][temp32 % 6] == false)
		{
			return;
		}

		// EC
		temp32 = Next(next, 0xFFFFFFFFu);
		// 3匹目個性
		if(pSrc->ecMod[2][temp32 % 6] == false)
		{
			return;
		}

		// 2匹目を先にチェック
		Next(next); // OTID
		Next(next); // PID

		{
			int ivs[6] = { -1, -1, -1, -1, -1, -1 };
			temp32 = 0;
			do {
				int fixedIndex = 0;
				do {
					fixedIndex = Next(next, 7); // V箇所
				} while(fixedIndex >= 6);

				if(ivs[fixedIndex] == -1)
				{
					ivs[fixedIndex] = 31;
					++temp32;
				}
			} while(temp32 < pSrc->pokemon[2].flawlessIvs);

			// 個体値
			temp32 = 1;
			for(int i = 0; i < 6; ++i)
			{
				if(ivs[i] == 31)
				{
					if(pSrc->pokemon[2].ivs[i] != 31)
					{
						temp32 = 0;
						break;
					}
				}
				else if(pSrc->pokemon[2].ivs[i] != Next(next, 0x1F))
				{
					temp32 = 0;
					break;
				}
			}
			if(temp32 == 0)
			{
				return;
			}
			
			// 特性
			temp32 = 0;
			if(pSrc->pokemon[2].abilityFlag == 3)
			{
				temp32 = Next(next, 1);
			}
			else
			{
				do {
					temp32 = Next(next, 3);
				} while(temp32 >= 3);
			}
			if((pSrc->pokemon[2].ability >= 0 && pSrc->pokemon[2].ability != temp32) || (pSrc->pokemon[2].ability == -1 && temp32 >= 2))
			{
				return;
			}

			// 性別値
			if(!pSrc->pokemon[2].isNoGender)
			{
				temp32 = 0;
				do {
					temp32 = Next(next, 0xFF);
				} while(temp32 >= 253);
			}

			// 性格
			temp32 = 0;
			do {
				temp32 = Next(next, 0x1F);
			} while(temp32 >= 25);

			if(temp32 != pSrc->pokemon[2].nature)
			{
				return;
			}
		}

		// 1匹目
		Next(seeds); // OTID
		Next(seeds); // PID

		{
			// 状態を保存
			next[0] = seeds[0];
			next[1] = seeds[1];
			next[2] = seeds[2];
			next[3] = seeds[3];

			{
				int ivs[6] = { -1, -1, -1, -1, -1, -1 };
				temp32 = 0;
				do {
					int fixedIndex = 0;
					do {
						fixedIndex = Next(seeds, 7); // V箇所
					} while(fixedIndex >= 6);

					if(ivs[fixedIndex] == -1)
					{
						ivs[fixedIndex] = 31;
						++temp32;
					}
				} while(temp32 < pSrc->pokemon[0].flawlessIvs);

				// 個体値
				temp32 = 1;
				for(int i = 0; i < 6; ++i)
				{
					if(ivs[i] == 31)
					{
						if(pSrc->pokemon[0].ivs[i] != 31)
						{
							temp32 = 0;
							break;
						}
					}
					else if(pSrc->pokemon[0].ivs[i] != Next(seeds, 0x1F))
					{
						temp32 = 0;
						break;
					}
				}
				if(temp32 == 0)
				{
					return;
				}
			}
			{
				int ivs[6] = { -1, -1, -1, -1, -1, -1 };
				temp32 = 0;
				do {
					int fixedIndex = 0;
					do {
						fixedIndex = Next(next, 7); // V箇所
					} while(fixedIndex >= 6);

					if(ivs[fixedIndex] == -1)
					{
						ivs[fixedIndex] = 31;
						++temp32;
					}
				} while(temp32 < pSrc->pokemon[1].flawlessIvs);

				// 個体値
				temp32 = 1;
				for(int i = 0; i < 6; ++i)
				{
					if(ivs[i] == 31)
					{
						if(pSrc->pokemon[1].ivs[i] != 31)
						{
							temp32 = 0;
							break;
						}
					}
					else if(pSrc->pokemon[1].ivs[i] != Next(next, 0x1F))
					{
						temp32 = 0;
						break;
					}
				}
				if(temp32 == 0)
				{
					return;
				}
			}

			// 特性
			temp32 = 0;
			if(pSrc->pokemon[0].abilityFlag == 3)
			{
				temp32 = Next(seeds, 1);
			}
			else
			{
				do {
					temp32 = Next(seeds, 3);
				} while(temp32 >= 3);
			}
			if((pSrc->pokemon[0].ability >= 0 && pSrc->pokemon[0].ability != temp32) || (pSrc->pokemon[0].ability == -1 && temp32 >= 2))
			{
				return;
			}
			temp32 = 0;
			if(pSrc->pokemon[1].abilityFlag == 3)
			{
				temp32 = Next(next, 1);
			}
			else
			{
				do {
					temp32 = Next(next, 3);
				} while(temp32 >= 3);
			}
			if((pSrc->pokemon[1].ability >= 0 && pSrc->pokemon[1].ability != temp32) || (pSrc->pokemon[1].ability == -1 && temp32 >= 2))
			{
				return;
			}

			// 性別値
			if(!pSrc->pokemon[0].isNoGender)
			{
				temp32 = 0;
				do {
					temp32 = Next(seeds, 0xFF);
				} while(temp32 >= 253);
			}
			if(!pSrc->pokemon[1].isNoGender)
			{
				temp32 = 0;
				do {
					temp32 = Next(next, 0xFF);
				} while(temp32 >= 253);
			}

			// 性格
			temp32 = 0;
			do {
				temp32 = Next(seeds, 0x1F);
			} while(temp32 >= 25);
			if(temp32 != pSrc->pokemon[0].nature)
			{
				return;
			}
			temp32 = 0;
			do {
				temp32 = Next(next, 0x1F);
			} while(temp32 >= 25);
			if(temp32 != pSrc->pokemon[1].nature)
			{
				return;
			}
		}

		pResult[0] = temp64;
	}
	return;
}

// 初期化
void CudaInitializeImpl()
{
	// ホストメモリの確保
	cudaMallocHost(&cu_HostMaster, sizeof(CudaInputMaster));

	{
		auto errorCode = cudaGetLastError();
		auto errorStr = cudaGetErrorName(errorCode);
		;
	}

	cudaMallocHost(&cu_HostResult, sizeof(_u64) * c_SizeResult);

	{
		auto errorCode = cudaGetLastError();
		auto errorStr = cudaGetErrorName(errorCode);
		;
	}

	// データの初期化
	cu_HostMaster->ecBit = -1;

	// デバイスメモリの確保
	cudaMalloc(&pDeviceMaster, sizeof(CudaInputMaster));

	{
		auto errorCode = cudaGetLastError();
		auto errorStr = cudaGetErrorName(errorCode);
		;
	}

	cudaMalloc(&pDeviceResult, sizeof(_u64) * c_SizeResult);

	{
		auto errorCode = cudaGetLastError();
		auto errorStr = cudaGetErrorName(errorCode);
		;
	}

}

// データセット
void CudaSetMasterData()
{
	cu_HostMaster->constantTermVector[0] = (_u32)(g_ConstantTermVector >> 30);
	cu_HostMaster->constantTermVector[1] = (_u32)(g_ConstantTermVector & 0x3FFFFFFFull);
	for(int i = 0; i < 64; ++i)
	{
		cu_HostMaster->answerFlag[i * 2] = (_u32)(g_AnswerFlag[i] >> 30);
		cu_HostMaster->answerFlag[i * 2 + 1] = (_u32)(g_AnswerFlag[i] & 0x3FFFFFFFull);
	}
	for(int i = 0; i < 16; ++i)
	{
		cu_HostMaster->coefficientData[i * 2] = (_u32)(g_CoefficientData[i] >> 32);
		cu_HostMaster->coefficientData[i * 2 + 1] = (_u32)(g_CoefficientData[i] & 0xFFFFFFFFull);
		cu_HostMaster->searchPattern[i] = (_u32)g_SearchPattern[i];
	}

	// データを転送
	cudaMemcpy(pDeviceMaster, cu_HostMaster, sizeof(CudaInputMaster), cudaMemcpyHostToDevice);
}

// 計算
void CudaProcess(_u32 ivs, int freeBit)
{
	//カーネル
	dim3 block(c_SizeBlockX, c_SizeBlockY, 1);
	dim3 grid(c_SizeGrid, 1, 1);
	kernel_calc << < grid, block >> > (pDeviceMaster, pDeviceResult, ivs);

	auto errorCode = cudaGetLastError();
	auto errorStr = cudaGetErrorName(errorCode);

	//デバイス->ホストへ結果を転送
	cudaMemcpy(cu_HostResult, pDeviceResult, sizeof(_u64) * c_SizeResult, cudaMemcpyDeviceToHost);
}

void Finish()
{
	//デバイスメモリの開放
	cudaFree(pDeviceResult);
	cudaFree(pDeviceMaster);
	//ホストメモリの開放
	cudaFreeHost(cu_HostResult);
	cudaFreeHost(cu_HostMaster);
}
