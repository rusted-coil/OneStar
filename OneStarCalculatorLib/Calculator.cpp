#include <iostream>
#include "Calculator.h"
#include "Const.h"
#include "XoroshiroState.h"

// 検索条件設定
int g_Ivs[6];
int g_Ability;
int g_FixedIndex;

// 絞り込み条件設定
int g_Nature;
int g_NextIvs[6];
int g_NextAbility;
int g_NextNature;
bool g_isNextNoGender;
int g_VCount;

// V確定用参照
const int* g_IvsRef[30] = {
	&g_Ivs[1], &g_Ivs[2], &g_Ivs[3], &g_Ivs[4], &g_Ivs[5],
	&g_Ivs[0], &g_Ivs[2], &g_Ivs[3], &g_Ivs[4], &g_Ivs[5],
	&g_Ivs[0], &g_Ivs[1], &g_Ivs[3], &g_Ivs[4], &g_Ivs[5],
	&g_Ivs[0], &g_Ivs[1], &g_Ivs[2], &g_Ivs[4], &g_Ivs[5],
	&g_Ivs[0], &g_Ivs[1], &g_Ivs[2], &g_Ivs[3], &g_Ivs[5],
	&g_Ivs[0], &g_Ivs[1], &g_Ivs[2], &g_Ivs[3], &g_Ivs[4]
};

// 事前計算データ
_u64 g_CoefficientData[0x80];

inline _u64 GetSignature(_u64 value)
{
	unsigned int a = (unsigned int)(value ^ (value >> 32));
	a = a ^ (a >> 16);
	a = a ^ (a >> 8);
	a = a ^ (a >> 4);
	a = a ^ (a >> 2);
	return (a ^ (a >> 1)) & 1;
}

inline _u64 GetSignature7(unsigned short value)
{
	value = value ^ (value >> 4);
	value = value ^ (value >> 2);
	return (value ^ (value >> 1)) & 1;
}

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

void Prepare()
{
	// データを作る
	for (unsigned short search = 0; search <= 0x7F; ++search)
	{
		g_CoefficientData[search] = 0;
		for (int i = 0; i < 57; ++i)
		{
			g_CoefficientData[search] |= (GetSignature7(Const::c_FormulaCoefficient[i] & search) << (56 - i));
		}
	}
}

int GetHiddenValue(int start, int end, _u64 seed)
{
	int value = 0;
	for (int i = start; i <= end; ++i)
	{
		if ((GetSignature(Const::c_Matrix[i] & seed) ^ Const::c_ConstList[i]) != 0)
		{
			value |= (1 << (end - i));
		}
	}
	return value;
}

_u64 Search(int ivs)
{
	XoroshiroState xoroshiro;
	
	_u64 ivs64 = (_u64)ivs;

	_u64 target = g_Ability;

	// 上位3bit = V箇所決定
	target |= (ivs64 & 0xE000000ul) << 29; // fixedIndex0

	// 下位25bit = 個体値
	target |= (ivs64 & 0x1F00000ul) << 26; // iv0_0
	target |= (ivs64 &   0xF8000ul) << 21; // iv1_0
	target |= (ivs64 &    0x7C00ul) << 16; // iv2_0
	target |= (ivs64 &     0x3E0ul) << 11; // iv3_0
	target |= (ivs64 &      0x1Ful) <<  6; // iv4_0

	// 隠された値を推定
	target |= ((8ul + g_FixedIndex - ((ivs64 & 0xE000000ul) >> 25)) & 7) << 51;

	target |= ((32ul + *g_IvsRef[g_FixedIndex * 5    ] - ((ivs64 & 0x1F00000ul) >> 20)) & 0x1F) << 41;
	target |= ((32ul + *g_IvsRef[g_FixedIndex * 5 + 1] - ((ivs64 &   0xF8000ul) >> 15)) & 0x1F) << 31;
	target |= ((32ul + *g_IvsRef[g_FixedIndex * 5 + 2] - ((ivs64 &    0x7C00ul) >> 10)) & 0x1F) << 21;
	target |= ((32ul + *g_IvsRef[g_FixedIndex * 5 + 3] - ((ivs64 &     0x3E0ul) >> 5)) & 0x1F) << 11;
	target |= ((32ul + *g_IvsRef[g_FixedIndex * 5 + 4] -  (ivs64 &      0x1Ful)) & 0x1F) << 1;

	// targetベクトル入力完了

	target ^= Const::c_Const;

	// 57bit側の計算結果キャッシュ
	_u64 processedTarget = 0;
	for (int i = 0; i < 57; ++i)
	{
		processedTarget |= (GetSignature(Const::c_FormulaAnswerFlag[i] & target) << (56 - i));
	}

	// 下位7bitを決める
	for (_u64 search = 0; search <= 0x7F; ++search)
	{
		_u64 seed = ((processedTarget ^ g_CoefficientData[search]) << 7) | search;

		// ここから絞り込み
		{
			xoroshiro.SetSeed(seed);
			xoroshiro.Next(); // EC
			xoroshiro.Next(); // OTID
			xoroshiro.Next(); // PID
			xoroshiro.Next(); // V箇所
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
			if (g_NextAbility >= 0 && g_NextAbility != xoroshiro.Next(1))
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
