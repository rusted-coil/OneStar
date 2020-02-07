#include <iostream>
#include "Util.h"
#include "CudaCalculator.h"
#include "Const.h"
#include "XoroshiroState.h"
#include "Data.h"
#include "CudaProcess5.cuh"
#include "CudaProcess6.cuh"

// 検索条件設定
static int g_CudaFixedIvs;

// 条件設定前の初期化
void CudaInitialize()
{
	CudaInitializeImpl();
}

void SetCudaCondition(
	int index,
	int iv0, int iv1, int iv2, int iv3, int iv4, int iv5,
	int ability, int nature, int natureTableId,
	int characteristic, bool isNoGender, int abilityFlag, int flawlessIvs)
{
	if(index < 0 || index >= 4)
	{
		return;
	}

	PokemonData* pokemon = &cu_HostInputMaster->pokemon[index];

	pokemon->ivs[0] = iv0;
	pokemon->ivs[1] = iv1;
	pokemon->ivs[2] = iv2;
	pokemon->ivs[3] = iv3;
	pokemon->ivs[4] = iv4;
	pokemon->ivs[5] = iv5;
	pokemon->ability = ability;
	pokemon->nature = nature;
	pokemon->natureTableId = natureTableId;
	pokemon->characteristic = characteristic;
	pokemon->isNoGender = isNoGender;
	pokemon->abilityFlag = abilityFlag;
	pokemon->flawlessIvs = flawlessIvs;

	// ECbitが利用できるか？
	if(cu_HostInputMaster->ecBit == -1)
	{
		int target = (characteristic == 0 ? 5 : characteristic - 1);
		if(pokemon->IsCharacterized(target))
		{
			// EC mod6 がcharacteristicで確定
			if(index != 2) // SeedのECbitなので反転させる
			{
				cu_HostInputMaster->ecBit = 1 - characteristic % 2;
			}
			else // Nextなのでさらに反転されてそのまま
			{
				cu_HostInputMaster->ecBit = characteristic % 2;
			}
		}
	}

	// EC mod6として考えられるもののフラグを立てる
	bool flag = true;
	cu_HostInputMaster->ecMod[index][characteristic] = true;
	for(int i = 1; i < 6; ++i)
	{
		int target = (characteristic + 6 - i) % 6;
		if(flag && pokemon->IsCharacterized(target) == false)
		{
			cu_HostInputMaster->ecMod[index][target] = true;
		}
		else
		{
			cu_HostInputMaster->ecMod[index][target] = false;
			flag = false;
		}
	}
}

void SetCudaTargetCondition6(int iv1, int iv2, int iv3, int iv4, int iv5, int iv6)
{
	g_CudaFixedIvs = 6;
	cu_HostInputMaster->ivs[0] = iv1;
	cu_HostInputMaster->ivs[1] = iv2;
	cu_HostInputMaster->ivs[2] = iv3;
	cu_HostInputMaster->ivs[3] = iv4;
	cu_HostInputMaster->ivs[4] = iv5;
	cu_HostInputMaster->ivs[5] = iv6;
}

void SetCudaTargetCondition5(int iv1, int iv2, int iv3, int iv4, int iv5)
{
	g_CudaFixedIvs = 5;
	cu_HostInputMaster->ivs[0] = iv1;
	cu_HostInputMaster->ivs[1] = iv2;
	cu_HostInputMaster->ivs[2] = iv3;
	cu_HostInputMaster->ivs[3] = iv4;
	cu_HostInputMaster->ivs[4] = iv5;
	cu_HostInputMaster->ivs[5] = 0;
}

// 計算ループ前の初期化
void CudaCalcInitialize()
{
	if(g_CudaFixedIvs == 5)
	{
		Cuda5Initialize();
	}
	else
	{
		Cuda6Initialize();
	}
}

// 計算前の事前計算
void PrepareCuda(int ivOffset)
{
	const int length = g_CudaFixedIvs * 10;

	// 使用する行列値をセット
	// 使用する定数ベクトルをセット

	g_ConstantTermVector = 0;

	// r[(11 - FixedIvs) + offset]からr[(11 - FixedIvs) + FixedIvs - 1 + offset]まで使う

	// 変換行列を計算
	InitializeTransformationMatrix(); // r[1]が得られる変換行列がセットされる
	for(int i = 0; i <= 9 - g_CudaFixedIvs + ivOffset; ++i)
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
	if(g_CudaFixedIvs == 5)
	{
		Cuda5SetMasterData();
	}
	else
	{
		Cuda6SetMasterData();
	}
}

void SearchCuda(_u32 ivs, int partitionBit)
{
	if(g_CudaFixedIvs == 5)
	{
		Cuda5Process(ivs << (25 - partitionBit), 1 << partitionBit);
	}
	else
	{
		Cuda6Process(ivs << (30 - partitionBit), 1 << (partitionBit - 1));
	}
}
int GetResultCount()
{
	return *cu_HostResultCount;
}
_u64 GetResult(int index)
{
	return cu_HostResult[index];
}

// 計算ループ後の処理
void CudaCalcFinalize()
{
	if(g_CudaFixedIvs == 5)
	{
		Cuda5Finalize();
	}
	else
	{
		Cuda6Finalize();
	}
	CudaFinalizeImpl();
}
