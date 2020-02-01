#include <iostream>
#include "Util.h"
#include "CudaCalculator.h"
#include "Const.h"
#include "XoroshiroState.h"
#include "Data.h"
#include "CudaProcess.cuh"

// 検索条件設定
static PokemonData l_CudaPokemon[3];

static int g_CudaFixedIvs;
static int g_CudaIvs[6];

static int g_CudaIvOffset;

static int g_CudaECbit; // -1は利用不可

static bool l_CudaEnableEcMod[3][6];

inline bool IsEnableECBit()
{
	return g_CudaECbit >= 0;
}

void SetCudaCondition(int index, int iv0, int iv1, int iv2, int iv3, int iv4, int iv5, int ability, int nature, int characteristic, bool isNoGender, int abilityFlag, int flawlessIvs)
{
	if(index < 0 || index >= 3)
	{
		return;
	}

	// 初期化
	if(index == 0)
	{
		g_CudaECbit = -1;
	}

	l_CudaPokemon[index].ivs[0] = iv0;
	l_CudaPokemon[index].ivs[1] = iv1;
	l_CudaPokemon[index].ivs[2] = iv2;
	l_CudaPokemon[index].ivs[3] = iv3;
	l_CudaPokemon[index].ivs[4] = iv4;
	l_CudaPokemon[index].ivs[5] = iv5;
	l_CudaPokemon[index].ability = ability;
	l_CudaPokemon[index].nature = nature;
	l_CudaPokemon[index].characteristic = characteristic;
	l_CudaPokemon[index].isNoGender = isNoGender;
	l_CudaPokemon[index].abilityFlag = abilityFlag;
	l_CudaPokemon[index].flawlessIvs = flawlessIvs;

	// ECbitが利用できるか？
	if(g_CudaECbit == -1)
	{
		int target = (characteristic == 0 ? 5 : characteristic - 1);
		if(l_CudaPokemon[index].IsCharacterized(target))
		{
			// EC mod6 がcharacteristicで確定
			if(index != 2) // SeedのECbitなので反転させる
			{
				g_CudaECbit = 1 - characteristic % 2;
			}
			else // Nextなのでさらに反転されてそのまま
			{
				g_CudaECbit = characteristic % 2;
			}
		}
	}

	// EC mod6として考えられるもののフラグを立てる
	bool flag = true;
	l_CudaEnableEcMod[index][characteristic] = true;
	for(int i = 1; i < 6; ++i)
	{
		int target = (characteristic + 6 - i) % 6;
		if(flag && l_CudaPokemon[index].IsCharacterized(target) == false)
		{
			l_CudaEnableEcMod[index][target] = true;
		}
		else
		{
			l_CudaEnableEcMod[index][target] = false;
			flag = false;
		}
	}
}

void SetCudaTargetCondition6(int iv1, int iv2, int iv3, int iv4, int iv5, int iv6)
{
	g_CudaFixedIvs = 6;
	g_CudaIvs[0] = iv1;
	g_CudaIvs[1] = iv2;
	g_CudaIvs[2] = iv3;
	g_CudaIvs[3] = iv4;
	g_CudaIvs[4] = iv5;
	g_CudaIvs[5] = iv6;
}

void SetCudaTargetCondition5(int iv1, int iv2, int iv3, int iv4, int iv5)
{
	g_CudaFixedIvs = 5;
	g_CudaIvs[0] = iv1;
	g_CudaIvs[1] = iv2;
	g_CudaIvs[2] = iv3;
	g_CudaIvs[3] = iv4;
	g_CudaIvs[4] = iv5;
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
	InitializeTransformationMatrix(IsEnableECBit()); // r[1]が得られる変換行列がセットされる
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
	CudaInitialize(g_CudaIvs);
}

void PreCalc(_u32 ivs, int freeBit)
{
	CudaProcess(ivs << 24, 24);
}
_u64 SearchCuda(int threadId)
{
	const int length = g_CudaFixedIvs * 10;

	XoroshiroState xoroshiro;
	XoroshiroState nextoshiro;
	XoroshiroState oshiroTemp;

	// 下位を決める
	_u64 max = ((1 << (64 - length)) - 1);
	for(_u64 search = 0; search <= max; ++search)
	{
		_u64 processedTarget = ((_u64)cu_HostResult[threadId * 2] << 32 | cu_HostResult[threadId * 2 + 1]);
		_u64 seed = (processedTarget ^ g_CoefficientData[search]) | g_SearchPattern[search];

		if(g_CudaECbit >= 0 && ((seed & 1) != g_CudaECbit))
		{
			continue;
		}

		_u64 nextSeed = seed + 0x82a2b175229d6a5bull;

		// ここから絞り込み

		// 個性チェック
		{
			xoroshiro.SetSeed(seed);
			nextoshiro.SetSeed(nextSeed);

			// EC
			unsigned int ec = xoroshiro.Next(0xFFFFFFFFu);
			// 1匹目個性
			if(l_CudaEnableEcMod[0][ec % 6] == false)
			{
				continue;
			}
			// 2匹目個性
			if(l_CudaEnableEcMod[1][ec % 6] == false)
			{
				continue;
			}

			// EC
			ec = nextoshiro.Next(0xFFFFFFFFu);
			// 3匹目個性
			if(l_CudaEnableEcMod[2][ec % 6] == false)
			{
				continue;
			}
		}

		// 2匹目を先にチェック
		nextoshiro.Next(); // OTID
		nextoshiro.Next(); // PID

		int vCount = l_CudaPokemon[2].flawlessIvs;

		int ivs[6] = { -1, -1, -1, -1, -1, -1 };
		int fixedCount = 0;
		do {
			int fixedIndex = 0;
			do {
				fixedIndex = nextoshiro.Next(7); // V箇所
			} while(fixedIndex >= 6);

			if(ivs[fixedIndex] == -1)
			{
				ivs[fixedIndex] = 31;
				++fixedCount;
			}
		} while(fixedCount < vCount);

		// 個体値
		bool isPassed = true;
		for(int i = 0; i < 6; ++i)
		{
			if(ivs[i] == 31)
			{
				if(l_CudaPokemon[2].ivs[i] != 31)
				{
					isPassed = false;
					break;
				}
			}
			else if(l_CudaPokemon[2].ivs[i] != nextoshiro.Next(0x1F))
			{
				isPassed = false;
				break;
			}
		}
		if(!isPassed)
		{
			continue;
		}

		// 特性
		int ability = 0;
		if(l_CudaPokemon[2].abilityFlag == 3)
		{
			ability = nextoshiro.Next(1);
		}
		else
		{
			do {
				ability = nextoshiro.Next(3);
			} while(ability >= 3);
		}
		if((l_CudaPokemon[2].ability >= 0 && l_CudaPokemon[2].ability != ability) || (l_CudaPokemon[2].ability == -1 && ability >= 2))
		{
			continue;
		}

		// 性別値
		if(!l_CudaPokemon[2].isNoGender)
		{
			int gender = 0;
			do {
				gender = nextoshiro.Next(0xFF);
			} while(gender >= 253);
		}

		// 性格
		int nature = 0;
		do {
			nature = nextoshiro.Next(0x1F);
		} while(nature >= 25);

		if(nature != l_CudaPokemon[2].nature)
		{
			continue;
		}

		// 1匹目
		xoroshiro.Next(); // OTID
		xoroshiro.Next(); // PID

		{
			// 状態を保存
			oshiroTemp.Copy(&xoroshiro);

			int ivs[6] = { -1, -1, -1, -1, -1, -1 };
			int fixedCount = 0;
			int offset = -(8 - g_CudaFixedIvs);
			do {
				int fixedIndex = 0;
				do {
					fixedIndex = xoroshiro.Next(7); // V箇所
					++offset;
				} while(fixedIndex >= 6);

				if(ivs[fixedIndex] == -1)
				{
					ivs[fixedIndex] = 31;
					++fixedCount;
				}
			} while(fixedCount < (8 - g_CudaFixedIvs));

			// reroll回数
			if(offset != g_CudaIvOffset)
			{
				continue;
			}

			// 個体値
			bool isPassed = true;
			for(int i = 0; i < 6; ++i)
			{
				if(ivs[i] == 31)
				{
					if(l_CudaPokemon[0].ivs[i] != 31)
					{
						isPassed = false;
						break;
					}
				}
				else if(l_CudaPokemon[0].ivs[i] != xoroshiro.Next(0x1F))
				{
					isPassed = false;
					break;
				}
			}
			if(!isPassed)
			{
				continue;
			}

			// 特性
			int ability = 0;
			if(l_CudaPokemon[0].abilityFlag == 3)
			{
				ability = xoroshiro.Next(1);
			}
			else
			{
				do {
					ability = xoroshiro.Next(3);
				} while(ability >= 3);
			}
			if((l_CudaPokemon[0].ability >= 0 && l_CudaPokemon[0].ability != ability) || (l_CudaPokemon[0].ability == -1 && ability >= 2))
			{
				continue;
			}

			// 性別値
			if(!l_CudaPokemon[0].isNoGender)
			{
				int gender = 0;
				do {
					gender = xoroshiro.Next(0xFF); // 性別値
				} while(gender >= 253);
			}

			int nature = 0;
			do {
				nature = xoroshiro.Next(0x1F); // 性格
			} while(nature >= 25);

			if(nature != l_CudaPokemon[0].nature)
			{
				continue;
			}
		}

		{
			xoroshiro.Copy(&oshiroTemp); // つづきから

			int vCount = l_CudaPokemon[1].flawlessIvs;

			int ivs[6] = { -1, -1, -1, -1, -1, -1 };
			int fixedCount = 0;
			do {
				int fixedIndex = 0;
				do {
					fixedIndex = xoroshiro.Next(7); // V箇所
				} while(fixedIndex >= 6);

				if(ivs[fixedIndex] == -1)
				{
					ivs[fixedIndex] = 31;
					++fixedCount;
				}
			} while(fixedCount < vCount);

			// 個体値
			bool isPassed = true;
			for(int i = 0; i < 6; ++i)
			{
				if(ivs[i] == 31)
				{
					if(l_CudaPokemon[1].ivs[i] != 31)
					{
						isPassed = false;
						break;
					}
				}
				else if(l_CudaPokemon[1].ivs[i] != xoroshiro.Next(0x1F))
				{
					isPassed = false;
					break;
				}
			}
			if(!isPassed)
			{
				continue;
			}

			// 特性
			int ability = 0;
			if(l_CudaPokemon[1].abilityFlag == 3)
			{
				ability = xoroshiro.Next(1);
			}
			else
			{
				do {
					ability = xoroshiro.Next(3);
				} while(ability >= 3);
			}
			if((l_CudaPokemon[1].ability >= 0 && l_CudaPokemon[1].ability != ability) || (l_CudaPokemon[1].ability == -1 && ability >= 2))
			{
				continue;
			}

			// 性別値
			if(!l_CudaPokemon[1].isNoGender)
			{
				int gender = 0;
				do {
					gender = xoroshiro.Next(0xFF); // 性別値
				} while(gender >= 253);
			}

			// 性格
			int nature = 0;
			do {
				nature = xoroshiro.Next(0x1F); // 性格
			} while(nature >= 25);

			if(nature != l_CudaPokemon[1].nature)
			{
				continue;
			}
		}

		return seed;
	}
	return 0;
}
