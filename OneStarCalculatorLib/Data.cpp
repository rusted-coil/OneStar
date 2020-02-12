#include "Data.h"
#include "Util.h"
#include "Const.h"

// 定数
NatureTable c_NatureTable[3];

// 計算用データ
_u64 g_TempMatrix[256];
_u64 g_InputMatrix[64]; // CalculateInverseMatrixの前にセットする
_u64 g_ConstantTermVector;
_u64 g_Coefficient[64];
_u64 g_AnswerFlag[64];
int g_FreeBit[64];
int g_FreeId[64];

_u64 g_CoefficientData[0x4000];
_u64 g_SearchPattern[0x4000];

_u64 l_Temp[256];

// 定数データ初期化
void InitializeConstData()
{
	c_NatureTable[0].randMax = 0x1F;
	c_NatureTable[0].patternCount = 25;
	for(int i = 0; i < 25; ++i)
	{
		c_NatureTable[0].natureId[i] = i;
	}

	// ストリンダー専用
	// ハイ
	c_NatureTable[1].randMax = 0xF;
	c_NatureTable[1].patternCount = 13;
	c_NatureTable[1].natureId[0]  =  3;
	c_NatureTable[1].natureId[1]  =  4;
	c_NatureTable[1].natureId[2]  =  2;
	c_NatureTable[1].natureId[3]  =  8;
	c_NatureTable[1].natureId[4]  =  9;
	c_NatureTable[1].natureId[5]  = 19;
	c_NatureTable[1].natureId[6]  = 22;
	c_NatureTable[1].natureId[7]  = 11;
	c_NatureTable[1].natureId[8]  = 13;
	c_NatureTable[1].natureId[9]  = 14;
	c_NatureTable[1].natureId[10] =  0;
	c_NatureTable[1].natureId[11] =  6;
	c_NatureTable[1].natureId[12] = 24;
	// ロー
	c_NatureTable[2].randMax = 0xF;
	c_NatureTable[2].patternCount = 12;
	c_NatureTable[2].natureId[0] = 1;
	c_NatureTable[2].natureId[1] = 5;
	c_NatureTable[2].natureId[2] = 7;
	c_NatureTable[2].natureId[3] = 10;
	c_NatureTable[2].natureId[4] = 12;
	c_NatureTable[2].natureId[5] = 15;
	c_NatureTable[2].natureId[6] = 16;
	c_NatureTable[2].natureId[7] = 17;
	c_NatureTable[2].natureId[8] = 18;
	c_NatureTable[2].natureId[9] = 20;
	c_NatureTable[2].natureId[10] =21;
	c_NatureTable[2].natureId[11] =23;
}

// 変換行列計算
void InitializeTransformationMatrix()
{
	// r[0] は (C1, seed)
	// r[1] は c_N * (C1, seed)

	// c_N^1をセット
	for(int i = 0; i < 256; ++i)
	{
		g_TempMatrix[i] = Const::c_N[i];
	}
}
void ProceedTransformationMatrix()
{
	for(int i = 0; i < 256; ++i)
	{
		l_Temp[i] = g_TempMatrix[i];
	}

	// 変換行列にもう一つc_Nを左からかける
	for(int y = 0; y < 128; ++y)
	{
		g_TempMatrix[y * 2] = 0;
		g_TempMatrix[y * 2 + 1] = 0;
		for(int x = 0; x < 64; ++x)
		{
			_u64 t0 = 0;
			_u64 t1 = 0;
			for(int i = 0; i < 64; ++i)
			{
				if((Const::c_N[y * 2] & (1ull << (63 - i))) != 0
					&& (l_Temp[i * 2] & (1ull << (63 - x))) != 0)
				{
					t0 = 1 - t0;
				}
				if((Const::c_N[y * 2 + 1] & (1ull << (63 - i))) != 0
					&& (l_Temp[(i + 64) * 2] & (1ull << (63 - x))) != 0)
				{
					t0 = 1 - t0;
				}

				if((Const::c_N[y * 2] & (1ull << (63 - i))) != 0
					&& (l_Temp[i * 2 + 1] & (1ull << (63 - x))) != 0)
				{
					t1 = 1 - t1;
				}
				if((Const::c_N[y * 2 + 1] & (1ull << (63 - i))) != 0
					&& (l_Temp[(i + 64) * 2 + 1] & (1ull << (63 - x))) != 0)
				{
					t1 = 1 - t1;
				}
			}
			g_TempMatrix[y * 2]     |= (t0 << (63 - x));
			g_TempMatrix[y * 2 + 1] |= (t1 << (63 - x));
		}
	}
}
_u64 GetMatrixMultiplier(int index)
{
	// s0部分
	return g_TempMatrix[index * 2 + 1];
}

short GetMatrixConst(int index)
{
	// s1部分
	return (short)GetSignature(g_TempMatrix[index * 2] & Const::c_XoroshiroConst);
}

void CalculateInverseMatrix(int length)
{
	// 初期状態をセット
	for (int i = 0; i < 64; ++i)
	{
		if(i < length)
		{
			g_AnswerFlag[i] = (1ull << (length - 1 - i));
		}
		else
		{
			g_AnswerFlag[i] = 0;
		}
	}

	int skip = 0;
	for (int i = 0; i < 64; ++i)
	{
		g_FreeBit[i] = 0;
	}

	// 行基本変形で求める
	for (int rank = 0; rank < length; )
	{
		_u64 top = (1ull << (63 - (rank + skip)));
		bool rankUpFlag = false;
		for (int i = rank; i < length; ++i)
		{
			if ((g_InputMatrix[i] & top) != 0) // 一番左が1
			{
				for (int a = 0; a < length; ++a)
				{
					if (a == i) continue;

					// 同じ列の1を消す
					if ((g_InputMatrix[a] & top) != 0)
					{
						g_InputMatrix[a] ^= g_InputMatrix[i];
						g_AnswerFlag[a] ^= g_AnswerFlag[i];
					}
				}
				// 最後に一番上に持ってくる
				_u64 swapM = g_InputMatrix[rank];
				_u64 swapF = g_AnswerFlag[rank];
				g_InputMatrix[rank] = g_InputMatrix[i];
				g_AnswerFlag[rank] = g_AnswerFlag[i];
				g_InputMatrix[i] = swapM;
				g_AnswerFlag[i] = swapF;

				++rank;
				rankUpFlag = true;
			}
		}
		if (rankUpFlag == false)
		{
			// マーキングしてスキップ
			g_FreeBit[rank + skip] = 1;
			g_FreeId[skip] = rank + skip;
			++skip;
		}
	}

	// AnserFlagを使用する項のところにセットしておく
	for(int c = skip, i = length + skip - 1; c > 0; --i)
	{
		if(g_FreeBit[i] == 0)
		{
			g_AnswerFlag[i] = g_AnswerFlag[i - c];
		}
		else
		{
			g_AnswerFlag[i] = 0;
			--c;
		}
	}

	// 自由bit
	for (int i = length + skip; i < 64; ++i)
	{
		g_FreeBit[i] = 1;
		g_FreeId[i - length] = i;
		++skip;
	}

	// 係数部分だけ抜き出し
	for (int i = 0; i < length; ++i)
	{
		g_Coefficient[i] = 0;
		for (int a = 0; a < (64 - length); ++a)
		{
			g_Coefficient[i] |= (g_InputMatrix[i] & (1ull << (63 - g_FreeId[a]))) >> ((length + a) - g_FreeId[a]);
		}
	}
}

void CalculateCoefficientData(int length)
{
	// データを作る
	unsigned short max = ((1 << (64 - length)) - 1);
	for (unsigned short search = 0; search <= max; ++search)
	{
		g_CoefficientData[search] = 0;
		g_SearchPattern[search] = 0;
		int offset = 0;
		for (int i = 0; i < length; ++i)
		{
			while (g_FreeBit[i + offset] > 0)
			{
				++offset;
			}
			g_CoefficientData[search] |= (GetSignature(g_Coefficient[i] & search) << (63 - (i + offset)));
		}
		for (int a = 0; a < (64 - length); ++a)
		{
			g_SearchPattern[search] |= ((_u64)search & (1ull << (64 - length - 1 - a))) << ((length + a) - g_FreeId[a]);
		}
	}
}

