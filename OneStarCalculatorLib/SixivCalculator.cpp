#include <iostream>
#include "Util.h"
#include "SixivCalculator.h"
#include "Const.h"
#include "XoroshiroState.h"
#include "Data.h"

// 検索条件設定
static PokemonData l_First;
static PokemonData l_Second;
static PokemonData l_Third;

static int g_Ivs[6];
static int g_Ability;
static int g_SecondIvCount;

static int g_IvOffset;

#define LENGTH (61)

void SetSixFirstCondition(int iv1, int iv2, int iv3, int iv4, int iv5, int iv6, int ability, int nature, bool isNoGender)
{
	l_First.ivs[0] = iv1;
	l_First.ivs[1] = iv2;
	l_First.ivs[2] = iv3;
	l_First.ivs[3] = iv4;
	l_First.ivs[4] = iv5;
	l_First.ivs[5] = iv6;
	l_First.ability = ability;
	l_First.nature = nature;
	l_First.isNoGender = isNoGender;
}

void SetSixSecondCondition(int iv1, int iv2, int iv3, int iv4, int iv5, int iv6, int ability, int nature, bool isNoGender)
{
	l_Second.ivs[0] = iv1;
	l_Second.ivs[1] = iv2;
	l_Second.ivs[2] = iv3;
	l_Second.ivs[3] = iv4;
	l_Second.ivs[4] = iv5;
	l_Second.ivs[5] = iv6;
	l_Second.ability = ability;
	l_Second.nature = nature;
	l_Second.isNoGender = isNoGender;
	g_SecondIvCount = 0;
	for (int i = 0; i < 6; ++i)
	{
		if (l_Second.ivs[i] == 31)
		{
			++g_SecondIvCount;
		}
	}
	/*
	// 個体値シーケンス確定
	int c = 0;
	g_Ability = -1;
	for (int i = 0; i < 6; ++i)
	{
		if (l_First.ivs[i] != 31)
		{
			g_Ivs[c++] = l_First.ivs[i];
		}
	}
	for (int i = 0; i < 6; ++i)
	{
		if (l_Second.ivs[i] != 31)
		{
			if (c == 6)
			{
				g_Ability = l_Second.ivs[i] % 2;
				break;
			}
			g_Ivs[c++] = l_Second.ivs[i];
		}
	}
	if (g_Ability < 0) // 個体値列を最後まで使っていれば2匹目の特性の位置
	{
		g_Ability = l_Second.ability;
	}*/
}

void SetSixThirdCondition(int iv1, int iv2, int iv3, int iv4, int iv5, int iv6, int ability, int nature, bool isNoGender)
{
	l_Third.ivs[0] = iv1;
	l_Third.ivs[1] = iv2;
	l_Third.ivs[2] = iv3;
	l_Third.ivs[3] = iv4;
	l_Third.ivs[4] = iv5;
	l_Third.ivs[5] = iv6;
	l_Third.ability = ability;
	l_Third.nature = nature;
	l_Third.isNoGender = isNoGender;
}

void SetTargetCondition(int iv1, int iv2, int iv3, int iv4, int iv5, int iv6, int ability)
{
	g_Ivs[0] = iv1;
	g_Ivs[1] = iv2;
	g_Ivs[2] = iv3;
	g_Ivs[3] = iv4;
	g_Ivs[4] = iv5;
	g_Ivs[5] = iv6;
	g_Ability = ability;
}

void PrepareSix(int ivOffset)
{
	g_IvOffset = ivOffset;

	// 使用する行列値をセット
	// 使用する定数ベクトルをセット
	g_ConstantTermVector = 0;
	for (int i = 0; i < LENGTH - 1; ++i)
	{
		int index = 10 * (2 + ivOffset) + i; // r[5+offset]からr[10+offset]まで使う
		g_InputMatrix[i] = Const::c_Matrix[index];
		if (Const::c_ConstList[index] > 0)
		{
			g_ConstantTermVector |= (1ull << (LENGTH - 1 - i));
		}
	}
	// Abilityは2つを圧縮 r[11+offset]
	int index = 10 * (8 + ivOffset) + 4;
	g_InputMatrix[LENGTH - 1] = Const::c_Matrix[index] ^ Const::c_Matrix[index + 5];
	if ((Const::c_ConstList[index] ^ Const::c_ConstList[index + 5]) != 0)
	{
		g_ConstantTermVector |= 1;
	}

	// 行基本変形で求める
	CalculateInverseMatrix(LENGTH);

	// 事前データを計算
	CalculateCoefficientData(LENGTH);
}

_u64 SearchSix(_u64 ivs)
{
	XoroshiroState xoroshiro;

	_u64 target = g_Ability;

	// 下位30bit = 個体値
	target |= (ivs & 0x3E000000ul) << 31; // iv0_0
	target |= (ivs &  0x1F00000ul) << 26; // iv1_0
	target |= (ivs &    0xF8000ul) << 21; // iv2_0
	target |= (ivs &     0x7C00ul) << 16; // iv3_0
	target |= (ivs &      0x3E0ul) << 11; // iv4_0
	target |= (ivs &       0x1Ful) <<  6; // iv5_0

	// 隠された値を推定
	target |= ((32ul + g_Ivs[0] - ((ivs & 0x3E000000ul) >> 25)) & 0x1F) << 51;
	target |= ((32ul + g_Ivs[1] - ((ivs &  0x1F00000ul) >> 20)) & 0x1F) << 41;
	target |= ((32ul + g_Ivs[2] - ((ivs &    0xF8000ul) >> 15)) & 0x1F) << 31;
	target |= ((32ul + g_Ivs[3] - ((ivs &     0x7C00ul) >> 10)) & 0x1F) << 21;
	target |= ((32ul + g_Ivs[4] - ((ivs &      0x3E0ul) >> 5)) & 0x1F) << 11;
	target |= ((32ul + g_Ivs[5] -  (ivs &        0x1Ful)) & 0x1F) << 1;

	// targetベクトル入力完了

	target ^= g_ConstantTermVector;

	// 60bit側の計算結果キャッシュ
	_u64 processedTarget = 0;
	int offset = 0;
	for (int i = 0; i < LENGTH; ++i)
	{
		while (g_FreeBit[i + offset] > 0)
		{
			++offset;
		}
		processedTarget |= (GetSignature(g_AnswerFlag[i] & target) << (63 - (i + offset)));
	}

	// 下位を決める
	_u64 max = ((1 << (64 - LENGTH)) - 1);
	for (_u64 search = 0; search <= max; ++search)
	{
		_u64 seed = (processedTarget ^ g_CoefficientData[search]) | g_SearchPattern[search];

		// ここから絞り込み
		// 1匹目
		{
			xoroshiro.SetSeed(seed);
			xoroshiro.Next(); // EC
			xoroshiro.Next(); // OTID
			xoroshiro.Next(); // PID

			int ivs[6] = { -1, -1, -1, -1, -1, -1 };
			int fixedCount = 0;
			int offset = -2;
			do {
				int fixedIndex = 0;
				do {
					fixedIndex = xoroshiro.Next(7); // V箇所
					++offset;
				} while (fixedIndex >= 6);

				if (ivs[fixedIndex] == -1)
				{
					ivs[fixedIndex] = 31;
					++fixedCount;
				}
			} while (fixedCount < 2);

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
					if (l_First.ivs[i] != 31)
					{
						isPassed = false;
						break;
					}
				}
				else if (l_First.ivs[i] != xoroshiro.Next(0x1F))
				{
					isPassed = false;
					break;
				}
			}
			if (!isPassed)
			{
				continue;
			}

			// 特性 -> チェック済み
			xoroshiro.Next();

			// 性別値
			if (!l_First.isNoGender)
			{
				int gender = 0;
				do {
					gender = xoroshiro.Next(0xFF); // 性別値
				} while (gender >= 253);
			}

			int nature = 0;
			do {
				nature = xoroshiro.Next(0x1F); // 性格
			} while (nature >= 25);

			if (nature != l_First.nature)
			{
				continue;
			}
		}

		{
			xoroshiro.SetSeed(seed);
			xoroshiro.Next(); // EC
			xoroshiro.Next(); // OTID
			xoroshiro.Next(); // PID

			int ivs[6] = { -1, -1, -1, -1, -1, -1 };
			int fixedCount = 0;
			do {
				int fixedIndex = 0;
				do {
					fixedIndex = xoroshiro.Next(7); // V箇所
				} while (fixedIndex >= 6);

				if (ivs[fixedIndex] == -1)
				{
					ivs[fixedIndex] = 31;
					++fixedCount;
				}
			} while (fixedCount < g_SecondIvCount);

			// 個体値
			bool isPassed = true;
			for (int i = 0; i < 6; ++i)
			{
				if (ivs[i] == 31)
				{
					if (l_Second.ivs[i] != 31)
					{
						isPassed = false;
						break;
					}
				}
				else if (l_Second.ivs[i] != xoroshiro.Next(0x1F))
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
			int ability = xoroshiro.Next(1);
			if (l_Second.ability >= 0 && l_Second.ability != ability)
			{
				continue;
			}

			// 性別値
			if (!l_Second.isNoGender)
			{ 
				int gender = 0;
				do {
					gender = xoroshiro.Next(0xFF); // 性別値
				} while (gender >= 253);
			}

			// 性格
			int nature = 0;
			do {
				nature = xoroshiro.Next(0x1F); // 性格
			} while (nature >= 25);

			if (nature != l_Second.nature)
			{
				continue;
			}
		}

		// 2匹目
		_u64 nextSeed = seed + 0x82a2b175229d6a5bull;
		{
			// V数2～4
			for (int vCount = 2; vCount <= 4; ++vCount)
			{
				xoroshiro.SetSeed(nextSeed);
				xoroshiro.Next(); // EC
				xoroshiro.Next(); // OTID
				xoroshiro.Next(); // PID

				int ivs[6] = { -1, -1, -1, -1, -1, -1 };
				int fixedCount = 0;
				do {
					int fixedIndex = 0;
					do {
						fixedIndex = xoroshiro.Next(7); // V箇所
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
						if (l_Third.ivs[i] != 31)
						{
							isPassed = false;
							break;
						}
					}
					else if (l_Third.ivs[i] != xoroshiro.Next(0x1F))
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
				int ability = xoroshiro.Next(1);
				if (l_Third.ability >= 0 && l_Third.ability != ability)
				{
					continue;
				}

				// 性別値
				if (!l_Third.isNoGender)
				{
					int gender = 0;
					do {
						gender = xoroshiro.Next(0xFF);
					} while (gender >= 253);
				}

				// 性格
				int nature = 0;
				do {
					nature = xoroshiro.Next(0x1F);
				} while (nature >= 25);

				if (nature != l_Third.nature)
				{
					continue;
				}

				return seed;
			}
		}
	}
	return 0;
}
