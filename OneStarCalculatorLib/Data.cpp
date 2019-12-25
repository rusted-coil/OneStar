#include "Data.h"
#include "Util.h"

// 計算用データ
_u64 g_InputMatrix[64]; // CalculateInverseMatrixの前にセットする
_u64 g_ConstantTermVector;
_u64 g_Coefficient[64];
_u64 g_AnswerFlag[64];

_u64 g_CoefficientData[0x80];

void CalculateInverseMatrix(int length)
{
	// 初期状態をセット
	for (int i = 0; i < length; ++i)
	{
		g_AnswerFlag[i] = (1ull << (length - 1 - i));
	}

	// 行基本変形で求める
	for (int rank = 0; rank < length; )
	{
		_u64 top = (1ull << (63 - rank));
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
			}
		}
	}

	// 係数部分だけ抜き出し
	_u64 mask = 0;
	for (int i = 0; i < 64 - length; ++i)
	{
		mask |= (1ull << i);
	}
	for (int i = 0; i < length; ++i)
	{
		g_Coefficient[i] = g_InputMatrix[i] & mask;
	}
}

void CalculateCoefficientData(int length)
{
	// データを作る
	unsigned short max = ((1 << (64 - length)) - 1);
	for (unsigned short search = 0; search <= max; ++search)
	{
		g_CoefficientData[search] = 0;
		for (int i = 0; i < length; ++i)
		{
			g_CoefficientData[search] |= (GetSignature7(g_Coefficient[i] & search) << (length - 1 - i));
		}
	}
}
