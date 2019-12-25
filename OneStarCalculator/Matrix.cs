using System;
using System.Collections.Generic;
using System.IO;

namespace OneStarCalculator
{
	public class Matrix
	{
		static UInt64 GetSignature(UInt64 value)
		{
			UInt32 a = (UInt32)(value ^ (value >> 32));
			a ^= (a >> 16);
			a ^= (a >> 8);
			a ^= (a >> 4);
			a ^= (a >> 2);
			return (a ^ (a >> 1)) & 1;
//			value ^= (value >> 32);
//			value ^= (value >> 16);
//			value ^= (value >> 8);
//			value ^= (value >> 4);
//			value ^= (value >> 2);
//			return (value ^ (value >> 1)) & 1;
			/*
			for (int i = 0; i < 64; ++i)
			{
				if ((value & (1ul << i)) != 0ul)
				{
					sign = 1 - sign;
				}
			}
			return sign;
			*/
		}

		static int GetHiddenValue(int start, int end, UInt64 seed)
		{
			int value = 0;
			for (int i = start; i <= end; ++i)
			{
				if ((GetSignature(XoroshiroConst.c_Matrix[i] & seed) ^ (UInt64)XoroshiroConst.c_ConstList[i]) != 0)
				{
					value |= (1 << (end - i));
				}
			}
			return value;
		}

		public void Test()
		{
			UInt64 seed = 0x4084401B20EF096Eul;

			int fixedIndex1 = GetHiddenValue(0, 2, seed);
			int fixedIndex0 = GetHiddenValue(3, 5, seed);
			int fixedIndex = (fixedIndex0 + fixedIndex1) % 8;
			if (fixedIndex < 6)
			{
				int iv01 = GetHiddenValue(6, 10, seed);
				int iv00 = GetHiddenValue(11, 15, seed);
				int iv0 = (iv01 + iv00) % 32;

				int iv11 = GetHiddenValue(16, 20, seed);
				int iv10 = GetHiddenValue(21, 25, seed);
				int iv1 = (iv11 + iv10) % 32;

				int iv21 = GetHiddenValue(26, 30, seed);
				int iv20 = GetHiddenValue(31, 35, seed);
				int iv2 = (iv21 + iv20) % 32;

				int iv31 = GetHiddenValue(36, 40, seed);
				int iv30 = GetHiddenValue(41, 45, seed);
				int iv3 = (iv31 + iv30) % 32;

				int iv41 = GetHiddenValue(46, 50, seed);
				int iv40 = GetHiddenValue(51, 55, seed);
				int iv4 = (iv41 + iv40) % 32;
				;
			}
			/*

			// 行基本変形で逆行列を求める
			UInt64[] rightFlag = new UInt64[57];
			for (int i = 0; i < 57; ++i)
			{
				rightFlag[i] = (1ul << (56 - i));
			}
			UInt64[] tempMatrix = new UInt64[XoroshiroConst.c_Matrix.Length];
			for (int i = 0; i < 57; ++i)
			{
				tempMatrix[i] = XoroshiroConst.c_Matrix[i];
			}
			for (int rank = 0; rank < 57; )
			{
				UInt64 top = (1ul << (63 - rank));
				for (int i = rank; i < 57; ++i)
				{
					if ((tempMatrix[i] & top) != 0) // 一番左が1
					{
						for (int a = 0; a < 57; ++a)
						{
							if (a == i) continue;

							// 同じ列の1を消す
							if ((tempMatrix[a] & top) != 0)
							{
								tempMatrix[a] ^= tempMatrix[i];
								rightFlag[a] ^= rightFlag[i];
							}
						}
						// 最後に一番上に持ってくる
						UInt64 swapM = tempMatrix[rank];
						UInt64 swapF = rightFlag[rank];
						tempMatrix[rank] = tempMatrix[i];
						rightFlag[rank] = rightFlag[i];
						tempMatrix[i] = swapM;
						rightFlag[i] = swapF;

						++rank;
					}
				}
			}
			;*/
		}

		public static List<UInt64> m_Result = new List<UInt64>();
		public static void Solve(UInt64 target)
		{
			target ^= XoroshiroConst.c_Const;
			
			// 下位7bitを決める
			for (UInt64 search = 0; search <= 0x7F; ++search)
			{
				UInt64 unitedTarget = (search << 57) | target;

				// 順番に決定
				UInt64 seed = search;
				for (int i = 0; i < 57; ++i)
				{
//					seed |= (1ul << i);
					seed |= GetSignature(((XoroshiroConst.c_FormulaCoefficient[i] << 57) | XoroshiroConst.c_FormulaAnswerFlag[i]) & unitedTarget) << (63 - i);
				}

				if (seed == 0xbde18ea5fe519f13ul)
				{
					m_Result.Add(seed);
				}				
			}
		}
	}
}
