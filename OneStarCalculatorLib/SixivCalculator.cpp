#include <iostream>
#include "Util.h"
#include "SixivCalculator.h"
#include "Const.h"
#include "XoroshiroState.h"
#include "Data.h"

// 検索条件設定
static PokemonData l_Pokemon[3];

static int g_FixedIvs;
static int g_Ivs[6];

static int g_IvOffset;

static int g_ECbit; // -1は利用不可

static bool l_EnableEcMod[3][6];

//#define LENGTH (60)

inline bool IsEnableECBit()
{
	return g_ECbit >= 0;
}

void Set35Condition(
	int index,
	int iv0, int iv1, int iv2, int iv3, int iv4, int iv5,
	int ability, int nature, int natureTableId,
	int characteristic, bool isNoGender, int abilityFlag, int flawlessIvs)
{
	if(index < 0 || index >= 3)
	{
		return;
	}

	// 初期化
	if(index == 0)
	{
		g_ECbit = -1;
	}

	l_Pokemon[index].ivs[0] = iv0;
	l_Pokemon[index].ivs[1] = iv1;
	l_Pokemon[index].ivs[2] = iv2;
	l_Pokemon[index].ivs[3] = iv3;
	l_Pokemon[index].ivs[4] = iv4;
	l_Pokemon[index].ivs[5] = iv5;
	l_Pokemon[index].ability = ability;
	l_Pokemon[index].nature = nature;
	l_Pokemon[index].natureTableId = natureTableId;
	l_Pokemon[index].characteristic = characteristic;
	l_Pokemon[index].isNoGender = isNoGender;
	l_Pokemon[index].abilityFlag = abilityFlag;
	l_Pokemon[index].flawlessIvs = flawlessIvs;

	// ECbitが利用できるか？
	if(g_ECbit == -1)
	{
		int target = (characteristic == 0 ? 5 : characteristic - 1);
		if(l_Pokemon[index].IsCharacterized(target))
		{
			// EC mod6 がcharacteristicで確定
			if(index != 2) // SeedのECbitなので反転させる
			{
				g_ECbit = 1 - characteristic % 2;
			}
			else // Nextなのでさらに反転されてそのまま
			{
				g_ECbit = characteristic % 2;
			}
		}
	}

	// EC mod6として考えられるもののフラグを立てる
	bool flag = true;
	l_EnableEcMod[index][characteristic] = true;
	for(int i = 1; i < 6; ++i)
	{
		int target = (characteristic + 6 - i) % 6;
		if(flag && l_Pokemon[index].IsCharacterized(target) == false)
		{
			l_EnableEcMod[index][target] = true;
		}
		else
		{
			l_EnableEcMod[index][target] = false;
			flag = false;
		}
	}
}

void SetTargetCondition6(int iv1, int iv2, int iv3, int iv4, int iv5, int iv6)
{
	g_FixedIvs = 6;
	g_Ivs[0] = iv1;
	g_Ivs[1] = iv2;
	g_Ivs[2] = iv3;
	g_Ivs[3] = iv4;
	g_Ivs[4] = iv5;
	g_Ivs[5] = iv6;
}

void SetTargetCondition5(int iv1, int iv2, int iv3, int iv4, int iv5)
{
	g_FixedIvs = 5;
	g_Ivs[0] = iv1;
	g_Ivs[1] = iv2;
	g_Ivs[2] = iv3;
	g_Ivs[3] = iv4;
	g_Ivs[4] = iv5;
}

void PrepareSix(int ivOffset)
{
	const int length = g_FixedIvs * 10;

	g_IvOffset = ivOffset;

	// 使用する行列値をセット
	// 使用する定数ベクトルをセット

	g_ConstantTermVector = 0;

	// r[(11 - FixedIvs) + offset]からr[(11 - FixedIvs) + FixedIvs - 1 + offset]まで使う

	// 変換行列を計算
	InitializeTransformationMatrix(); // r[1]が得られる変換行列がセットされる
	for(int i = 0; i <= 9 - g_FixedIvs + ivOffset; ++i)
	{
		ProceedTransformationMatrix(); // r[2 + i]が得られる
	}

	for(int a = 0; a < g_FixedIvs; ++a)
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
}

_u64 SearchSix(_u64 ivs)
{
	const int length = g_FixedIvs * 10;

	XoroshiroState xoroshiro;
	XoroshiroState nextoshiro;
	XoroshiroState oshiroTemp;

	_u64 target = 0;

	if(g_FixedIvs == 6)
	{
		// 下位30bit = 個体値
		target |= (ivs & 0x3E000000ul) << 30; // iv0_0
		target |= (ivs & 0x1F00000ul) << 25; // iv1_0
		target |= (ivs & 0xF8000ul) << 20; // iv2_0
		target |= (ivs & 0x7C00ul) << 15; // iv3_0
		target |= (ivs & 0x3E0ul) << 10; // iv4_0
		target |= (ivs & 0x1Ful) << 5; // iv5_0

		// 隠された値を推定
		target |= ((32ul + g_Ivs[0] - ((ivs & 0x3E000000ul) >> 25)) & 0x1F) << 50;
		target |= ((32ul + g_Ivs[1] - ((ivs & 0x1F00000ul) >> 20)) & 0x1F) << 40;
		target |= ((32ul + g_Ivs[2] - ((ivs & 0xF8000ul) >> 15)) & 0x1F) << 30;
		target |= ((32ul + g_Ivs[3] - ((ivs & 0x7C00ul) >> 10)) & 0x1F) << 20;
		target |= ((32ul + g_Ivs[4] - ((ivs & 0x3E0ul) >> 5)) & 0x1F) << 10;
		target |= ((32ul + g_Ivs[5] - (ivs & 0x1Ful)) & 0x1F);
	}
	else if(g_FixedIvs == 5)
	{
		// 下位25bit = 個体値
		target |= (ivs & 0x1F00000ul) << 25; // iv0_0
		target |= (ivs & 0xF8000ul) << 20; // iv1_0
		target |= (ivs & 0x7C00ul) << 15; // iv2_0
		target |= (ivs & 0x3E0ul) << 10; // iv3_0
		target |= (ivs & 0x1Ful) << 5; // iv4_0

		// 隠された値を推定
		target |= ((32ul + g_Ivs[0] - ((ivs & 0x1F00000ul) >> 20)) & 0x1F) << 40;
		target |= ((32ul + g_Ivs[1] - ((ivs & 0xF8000ul) >> 15)) & 0x1F) << 30;
		target |= ((32ul + g_Ivs[2] - ((ivs & 0x7C00ul) >> 10)) & 0x1F) << 20;
		target |= ((32ul + g_Ivs[3] - ((ivs & 0x3E0ul) >> 5)) & 0x1F) << 10;
		target |= ((32ul + g_Ivs[4] - (ivs & 0x1Ful)) & 0x1F);
	}
	else
	{
		return 0;
	}

	// targetベクトル入力完了

	target ^= g_ConstantTermVector;

	// 60bit側の計算結果キャッシュ
	_u64 processedTarget = 0;
	for (int i = 0; i < 64; ++i)
	{
		if(g_AnswerFlag[i] != 0)
		{
			processedTarget |= (GetSignature(g_AnswerFlag[i] & target) << (63 - i));
		}
	}

	// 下位を決める
	_u64 max = ((1 << (64 - length)) - 1);
	for (_u64 search = 0; search <= max; ++search)
	{
		_u64 seed = (processedTarget ^ g_CoefficientData[search]) | g_SearchPattern[search];

		if(g_ECbit >= 0 && ((seed & 1) != g_ECbit))
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
			if(l_EnableEcMod[0][ec % 6] == false)
			{
				continue;
			}
			// 2匹目個性
			if(l_EnableEcMod[1][ec % 6] == false)
			{
				continue;
			}

			// EC
			ec = nextoshiro.Next(0xFFFFFFFFu);
			// 3匹目個性
			if(l_EnableEcMod[2][ec % 6] == false)
			{
				continue;
			}
		}

		// 2匹目を先にチェック
		nextoshiro.Next(); // OTID
		nextoshiro.Next(); // PID

		int vCount = l_Pokemon[2].flawlessIvs;

		{
			// 個体値
			int ivs[8] = { -1, -1, -1, -1, -1, -1, 31, 31 };
			bool isPassed = true;
			int fixedCount = 0;
			do {
				_u32 fixedIndex = 0;
				do {
					fixedIndex = nextoshiro.Next(7u); // V箇所
				} while(ivs[fixedIndex] == 31);

				if(l_Pokemon[2].ivs[fixedIndex] != 31)
				{
					isPassed = false;
					break;
				}

				ivs[fixedIndex] = 31;
				++fixedCount;
			} while(fixedCount < vCount);
			if(!isPassed)
			{
				continue;
			}

			for(int i = 0; i < 6; ++i)
			{
				if(ivs[i] != 31)
				{
					if(l_Pokemon[2].ivs[i] != nextoshiro.Next(0x1Fu))
					{
						isPassed = false;
						break;
					}
				}
			}
			if(!isPassed)
			{
				continue;
			}
		}

		// 特性
		_u32 ability = 0;
		if(l_Pokemon[2].abilityFlag == 3)
		{
			ability = nextoshiro.Next(1u);
		}
		else
		{
			do {
				ability = nextoshiro.Next(3u);
			} while(ability >= 3);
		}
		if((l_Pokemon[2].ability >= 0 && l_Pokemon[2].ability != ability) || (l_Pokemon[2].ability == -1 && ability >= 2))
		{
			continue;
		}

		// 性別値
		if(!l_Pokemon[2].isNoGender)
		{
			_u32 gender = 0;
			do {
				gender = nextoshiro.Next(0xFFu);
			} while(gender >= 253);
		}

		// 性格
		_u32 nature = 0;
		do {
			nature = nextoshiro.Next(c_NatureTable[l_Pokemon[2].natureTableId].randMax);
		} while(nature >= c_NatureTable[l_Pokemon[2].natureTableId].patternCount);

		if(c_NatureTable[l_Pokemon[2].natureTableId].natureId[nature] != l_Pokemon[2].nature)
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
			int offset = -(8 - g_FixedIvs);
			do {
				int fixedIndex = 0;
				do {
					fixedIndex = xoroshiro.Next(7u); // V箇所
					++offset;
				} while (fixedIndex >= 6);

				if (ivs[fixedIndex] == -1)
				{
					ivs[fixedIndex] = 31;
					++fixedCount;
				}
			} while (fixedCount < (8 - g_FixedIvs));

			// reroll回数
			if (offset != g_IvOffset)
			{
				continue;
			}

			// 個体値
			bool isPassed = true;
			for (int i = 0; i < 6; ++i)
			{
				if (ivs[i] == 31)
				{
					if (l_Pokemon[0].ivs[i] != 31)
					{
						isPassed = false;
						break;
					}
				}
				else if (l_Pokemon[0].ivs[i] != xoroshiro.Next(0x1Fu))
				{
					isPassed = false;
					break;
				}
			}
			if (!isPassed)
			{
				continue;
			}

			// 特性
			int ability = 0;
			if (l_Pokemon[0].abilityFlag == 3)
			{
				ability = xoroshiro.Next(1u);
			}
			else
			{			
				do {
					ability = xoroshiro.Next(3u);
				} while(ability >= 3);
			}
			if ((l_Pokemon[0].ability >= 0 && l_Pokemon[0].ability != ability) || (l_Pokemon[0].ability == -1 && ability >= 2))
			{
				continue;
			}

			// 性別値
			if (!l_Pokemon[0].isNoGender)
			{
				int gender = 0;
				do {
					gender = xoroshiro.Next(0xFFu); // 性別値
				} while (gender >= 253);
			}

			// 性格
			_u32 nature = 0;
			do {
				nature = xoroshiro.Next(c_NatureTable[l_Pokemon[0].natureTableId].randMax);
			} while(nature >= c_NatureTable[l_Pokemon[0].natureTableId].patternCount);

			if(c_NatureTable[l_Pokemon[0].natureTableId].natureId[nature] != l_Pokemon[0].nature)
			{
				continue;
			}
		}

		{
			xoroshiro.Copy(&oshiroTemp); // つづきから

			int vCount = l_Pokemon[1].flawlessIvs;

			int ivs[6] = { -1, -1, -1, -1, -1, -1 };
			int fixedCount = 0;
			do {
				int fixedIndex = 0;
				do {
					fixedIndex = xoroshiro.Next(7u); // V箇所
				} while (fixedIndex >= 6);

				if (ivs[fixedIndex] == -1)
				{
					ivs[fixedIndex] = 31;
					++fixedCount;
				}
			} while (fixedCount < vCount);

			// 個体値
			bool isPassed = true;
			for (int i = 0; i < 6; ++i)
			{
				if (ivs[i] == 31)
				{
					if (l_Pokemon[1].ivs[i] != 31)
					{
						isPassed = false;
						break;
					}
				}
				else if (l_Pokemon[1].ivs[i] != xoroshiro.Next(0x1Fu))
				{
					isPassed = false;
					break;
				}
			}
			if (!isPassed)
			{
				continue;
			}

			// 特性
			int ability = 0;
			if (l_Pokemon[1].abilityFlag == 3)
			{
				ability = xoroshiro.Next(1u);
			}
			else
			{
				do {
					ability = xoroshiro.Next(3u);
				} while(ability >= 3);
			}
			if ((l_Pokemon[1].ability >= 0 && l_Pokemon[1].ability != ability) || (l_Pokemon[1].ability == -1 && ability >= 2))
			{
				continue;
			}

			// 性別値
			if (!l_Pokemon[1].isNoGender)
			{ 
				int gender = 0;
				do {
					gender = xoroshiro.Next(0xFFu); // 性別値
				} while (gender >= 253);
			}

			// 性格
			_u32 nature = 0;
			do {
				nature = xoroshiro.Next(c_NatureTable[l_Pokemon[1].natureTableId].randMax);
			} while(nature >= c_NatureTable[l_Pokemon[1].natureTableId].patternCount);

			if(c_NatureTable[l_Pokemon[1].natureTableId].natureId[nature] != l_Pokemon[1].nature)
			{
				continue;
			}
		}

		return seed;
	}
	return 0;
}
