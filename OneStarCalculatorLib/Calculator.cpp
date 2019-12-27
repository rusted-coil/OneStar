#include <iostream>
#include "Util.h"
#include "Calculator.h"
#include "Const.h"
#include "XoroshiroState.h"
#include "Data.h"

// 検索条件設定
static int g_Rerolls;
static int g_Ivs[6];
static int g_Ability;
static int g_FixedIndex;

// 絞り込み条件設定
static int g_Nature;
static int g_NextIvs[6];
static int g_NextAbility;
static int g_NextNature;
static bool g_isNextNoGender;
static int g_VCount;

// V確定用参照
const int* g_IvsRef[30] = {
	&g_Ivs[1], &g_Ivs[2], &g_Ivs[3], &g_Ivs[4], &g_Ivs[5],
	&g_Ivs[0], &g_Ivs[2], &g_Ivs[3], &g_Ivs[4], &g_Ivs[5],
	&g_Ivs[0], &g_Ivs[1], &g_Ivs[3], &g_Ivs[4], &g_Ivs[5],
	&g_Ivs[0], &g_Ivs[1], &g_Ivs[2], &g_Ivs[4], &g_Ivs[5],
	&g_Ivs[0], &g_Ivs[1], &g_Ivs[2], &g_Ivs[3], &g_Ivs[5],
	&g_Ivs[0], &g_Ivs[1], &g_Ivs[2], &g_Ivs[3], &g_Ivs[4]
};

#define LENGTH (57)

void SetFirstCondition(int iv0, int iv1, int iv2, int iv3, int iv4, int iv5, int ability, int nature)
{
	g_Ivs[0] = iv0;
	g_Ivs[1] = iv1;
	g_Ivs[2] = iv2;
	g_Ivs[3] = iv3;
	g_Ivs[4] = iv4;
	g_Ivs[5] = iv5;
	g_FixedIndex = 0;
	for (int i = 0; i < 6; ++i)
	{
		if (g_Ivs[i] == 31)
		{
			g_FixedIndex = i;
		}
	}
	g_Ability = ability;
	g_Nature = nature;
}

void SetNextCondition(int iv0, int iv1, int iv2, int iv3, int iv4, int iv5, int ability, int nature, bool isNoGender)
{
	g_NextIvs[0] = iv0;
	g_NextIvs[1] = iv1;
	g_NextIvs[2] = iv2;
	g_NextIvs[3] = iv3;
	g_NextIvs[4] = iv4;
	g_NextIvs[5] = iv5;
	g_VCount = 0;
	for (int i = 0; i < 6; ++i)
	{
		if (g_NextIvs[i] == 31)
		{
			++g_VCount;
		}
	}
	g_NextAbility = ability;
	g_NextNature = nature;
	g_isNextNoGender = isNoGender;
}

void Prepare(int rerolls)
{
	g_Rerolls = rerolls;

	// 使用する行列値をセット
	// 使用する定数ベクトルをセット
	g_ConstantTermVector = 0;
	for (int i = 0; i < LENGTH - 1; ++i)
	{
		int index = (i < 6 ? rerolls * 10 + (i / 3) * 5 + 2 + i % 3 : i - 6 + (rerolls + 1) * 10); // r[3+rerolls]をV箇所、r[4+rerolls]からr[8+rerolls]を個体値として使う
		g_InputMatrix[i] = Const::c_Matrix[index];
		if (Const::c_ConstList[index] > 0)
		{
			g_ConstantTermVector |= (1ull << (LENGTH - 1 - i));
		}
	}
	// Abilityは2つを圧縮 r[9+rerolls]
	int index = (rerolls + 6) * 10 + 4;
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

_u64 Search(_u64 ivs)
{
	XoroshiroState xoroshiro;
	
	_u64 target = g_Ability;

	// 上位3bit = V箇所決定
	target |= (ivs & 0xE000000ul) << 29; // fixedIndex0

	// 下位25bit = 個体値
	target |= (ivs & 0x1F00000ul) << 26; // iv0_0
	target |= (ivs &   0xF8000ul) << 21; // iv1_0
	target |= (ivs &    0x7C00ul) << 16; // iv2_0
	target |= (ivs &     0x3E0ul) << 11; // iv3_0
	target |= (ivs &      0x1Ful) <<  6; // iv4_0

	// 隠された値を推定
	target |= ((8ul + g_FixedIndex - ((ivs & 0xE000000ul) >> 25)) & 7) << 51;

	target |= ((32ul + *g_IvsRef[g_FixedIndex * 5    ] - ((ivs & 0x1F00000ul) >> 20)) & 0x1F) << 41;
	target |= ((32ul + *g_IvsRef[g_FixedIndex * 5 + 1] - ((ivs &   0xF8000ul) >> 15)) & 0x1F) << 31;
	target |= ((32ul + *g_IvsRef[g_FixedIndex * 5 + 2] - ((ivs &    0x7C00ul) >> 10)) & 0x1F) << 21;
	target |= ((32ul + *g_IvsRef[g_FixedIndex * 5 + 3] - ((ivs &     0x3E0ul) >> 5)) & 0x1F) << 11;
	target |= ((32ul + *g_IvsRef[g_FixedIndex * 5 + 4] -  (ivs &      0x1Ful)) & 0x1F) << 1;

	// targetベクトル入力完了

	target ^= g_ConstantTermVector;

	// 57bit側の計算結果キャッシュ
	_u64 processedTarget = 0;
	int offset = 0;
	for (int i = 0; i < LENGTH; ++i)
	{
		if (g_FreeBit[i] > 0)
		{
			++offset;
		}
		processedTarget |= (GetSignature(g_AnswerFlag[i] & target) << (63 - (i + offset)));
	}

	// 下位7bitを決める
	_u64 max = ((1 << (64 - LENGTH)) - 1);
	for (_u64 search = 0; search <= max; ++search)
	{
		_u64 seed = (processedTarget ^ g_CoefficientData[search]) | g_SearchPattern[search];

		// ここから絞り込み
		{
			xoroshiro.SetSeed(seed);
			xoroshiro.Next(); // EC
			xoroshiro.Next(); // OTID
			xoroshiro.Next(); // PID

			// V箇所
			int offset = -1;
			int fixedIndex = 0;
			do {
				fixedIndex = xoroshiro.Next(7); // V箇所
				++offset;
			} while (fixedIndex >= 6);
			if (offset != g_Rerolls)
			{
				continue;
			}

			xoroshiro.Next(); // 個体値1
			xoroshiro.Next(); // 個体値2
			xoroshiro.Next(); // 個体値3
			xoroshiro.Next(); // 個体値4
			xoroshiro.Next(); // 個体値5
			xoroshiro.Next(); // 特性

			int gender = 0;
			do {
				gender = xoroshiro.Next(0xFF); // 性別値
			} while (gender >= 253);

			int nature = 0;
			do {
				nature = xoroshiro.Next(0x1F); // 性格
			} while (nature >= 25);

			if (nature != g_Nature)
			{
				continue;
			}
		}

		// 2匹目
		_u64 nextSeed = seed + 0x82a2b175229d6a5bull;
		for(int ivVCount = g_VCount; ivVCount >= 1; --ivVCount)
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
			} while (fixedCount < ivVCount);

			// 個体値
			bool isPassed = true;
			for (int i = 0; i < 6; ++i)
			{
				if (ivs[i] == 31)
				{
					if (g_NextIvs[i] != 31)
					{
						isPassed = false;
						break;
					}
				}
				else if(g_NextIvs[i] != xoroshiro.Next(0x1F))
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
			if (g_NextAbility >= 0 && g_NextAbility != ability)
			{
				continue;
			}

			// 性別値
			if (!g_isNextNoGender)
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

			if (nature != g_NextNature)
			{
				continue;
			}

			return seed;
		}
	}
	return 0;
}
