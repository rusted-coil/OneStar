#include <iostream>
#include "Util.h"
#include "CudaCalculator.h"
#include "Const.h"
#include "XoroshiroState.h"
#include "Data.h"
#include "CudaProcess.cuh"

// 検索条件設定
static int g_CudaFixedIvs;
static int g_CudaIvOffset;
//static int g_CudaIvs[6];


/*
inline bool IsEnableECBit()
{
	return g_CudaECbit >= 0;
}
*/

void CudaInitialize()
{
	CudaInitializeImpl();
}

void SetCudaCondition(int index, int iv0, int iv1, int iv2, int iv3, int iv4, int iv5, int ability, int nature, int characteristic, bool isNoGender, int abilityFlag, int flawlessIvs)
{
	if(index < 0 || index >= 4)
	{
		return;
	}

	PokemonData* pokemon = &cu_HostMaster->pokemon[index];

	pokemon->ivs[0] = iv0;
	pokemon->ivs[1] = iv1;
	pokemon->ivs[2] = iv2;
	pokemon->ivs[3] = iv3;
	pokemon->ivs[4] = iv4;
	pokemon->ivs[5] = iv5;
	pokemon->ability = ability;
	pokemon->nature = nature;
	pokemon->characteristic = characteristic;
	pokemon->isNoGender = isNoGender;
	pokemon->abilityFlag = abilityFlag;
	pokemon->flawlessIvs = flawlessIvs;

	// ECbitが利用できるか？
	if(cu_HostMaster->ecBit == -1)
	{
		int target = (characteristic == 0 ? 5 : characteristic - 1);
		if(pokemon->IsCharacterized(target))
		{
			// EC mod6 がcharacteristicで確定
			if(index != 2) // SeedのECbitなので反転させる
			{
				cu_HostMaster->ecBit = 1 - characteristic % 2;
			}
			else // Nextなのでさらに反転されてそのまま
			{
				cu_HostMaster->ecBit = characteristic % 2;
			}
		}
	}

	// EC mod6として考えられるもののフラグを立てる
	bool flag = true;
	cu_HostMaster->ecMod[index][characteristic] = true;
	for(int i = 1; i < 6; ++i)
	{
		int target = (characteristic + 6 - i) % 6;
		if(flag && pokemon->IsCharacterized(target) == false)
		{
			cu_HostMaster->ecMod[index][target] = true;
		}
		else
		{
			cu_HostMaster->ecMod[index][target] = false;
			flag = false;
		}
	}
}

void SetCudaTargetCondition6(int iv1, int iv2, int iv3, int iv4, int iv5, int iv6)
{
	g_CudaFixedIvs = 6;
	cu_HostMaster->ivs[0] = iv1;
	cu_HostMaster->ivs[1] = iv2;
	cu_HostMaster->ivs[2] = iv3;
	cu_HostMaster->ivs[3] = iv4;
	cu_HostMaster->ivs[4] = iv5;
	cu_HostMaster->ivs[5] = iv6;
}

void SetCudaTargetCondition5(int iv1, int iv2, int iv3, int iv4, int iv5)
{
	g_CudaFixedIvs = 5;
	cu_HostMaster->ivs[0] = iv1;
	cu_HostMaster->ivs[1] = iv2;
	cu_HostMaster->ivs[2] = iv3;
	cu_HostMaster->ivs[3] = iv4;
	cu_HostMaster->ivs[4] = iv5;
	cu_HostMaster->ivs[5] = 0;
}

void PrepareCuda(int ivOffset)
{
	const int length = g_CudaFixedIvs * 10;

	g_CudaIvOffset = ivOffset;

	// 使用する行列値をセット
	// 使用する定数ベクトルをセット

	g_ConstantTermVector = 0;

	// r[(11 - FixedIvs) + offset]からr[(11 - FixedIvs) + FixedIvs - 1 + offset]まで使う

	// 変換行列を計算
	InitializeTransformationMatrix(); // r[1]が得られる変換行列がセットされる
	for(int i = 0; i <= 9 - g_CudaFixedIvs + ivOffset - 1; ++i)
	{
		ProceedTransformationMatrix(); // r[2 + i]が得られる
	}

	for(int a = 0; a < g_CudaFixedIvs; ++a)
	{
		for(int i = 0; i < 10; ++i)
		{
			int index = 59 + (i / 5) * 64 + (i % 5);
			int bit = a * 10 + i;
			g_InputMatrix[bit] = GetMatrixMultiplier(index);
			if(GetMatrixConst(index) != 0)
			{
				g_ConstantTermVector |= (1ull << (length - 1 - bit));
			}
		}
		ProceedTransformationMatrix();
	}

	// 行基本変形で求める
	CalculateInverseMatrix(length);

	// 事前データを計算
	CalculateCoefficientData(length);

	// Cuda初期化
	CudaSetMasterData(length);
}

void PreCalc(_u32 ivs, int freeBit)
{
	CudaProcess(ivs << 20, 24);
}
_u64 SearchCuda(int threadId)
{
	return cu_HostResult[0];
}