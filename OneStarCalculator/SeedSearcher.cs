using System.Collections.Generic;
using System.Threading.Tasks;
using System.Runtime.InteropServices;

namespace OneStarCalculator
{
	public class SeedSearcher
	{
		// 結果
		public List<ulong> Result { get; } = new List<ulong>();

		[DllImport("OneStarCalculatorLib.dll")]
		static extern void Prepare(int rerolls);

		[DllImport("OneStarCalculatorLib.dll")]
		public static extern void SetFirstCondition(int iv0, int iv1, int iv2, int iv3, int iv4, int iv5, int ability, int nature);

		[DllImport("OneStarCalculatorLib.dll")]
		public static extern void SetNextCondition(int iv0, int iv1, int iv2, int iv3, int iv4, int iv5, int ability, int nature, bool noGender);

		[DllImport("OneStarCalculatorLib.dll")]
		static extern ulong Search(ulong ivs);

		// テスト
		[DllImport("OneStarCalculatorLib.dll")]
		public static extern void SetSixCondition(int fixed1, int fixed2, int iv1, int iv2, int iv3, int iv4, int iv5, int iv6, int nature);

		[DllImport("OneStarCalculatorLib.dll")]
		static extern void PrepareSix(int ivOffset);

		[DllImport("OneStarCalculatorLib.dll")]
		static extern ulong SearchSix(int ivs);

		public SeedSearcher()
		{
		}

		public void Calculate(bool isEnableStop)
		{
			// 探索範囲
			int searchLower = 0;
			int searchUpper = 0x7FFFFFFF;

			Result.Clear();

			// C++ライブラリ側の事前計算
			Prepare(2);

			// 並列探索
			if (isEnableStop)
			{
				// 中断あり
				Parallel.For(searchLower, searchUpper, (ivs, state) => {
					ulong ivs64 = (ulong)ivs;
					ulong result = Search(ivs64);
					if (result != 0)
					{
						Result.Add(result);
						state.Stop();
					}
					result = Search(ivs64 | (1ul << 31));
					if (result != 0)
					{
						Result.Add(result);
						state.Stop();
					}
				});
			}
			else
			{
				// 中断なし
				Parallel.For(searchLower, searchUpper, (ivs) => {
					ulong result = Search((ulong)ivs);
					if (result != 0)
					{
						Result.Add(result);
					}
				});
			}
		}

		public void CalculateSix()
		{
			// 探索範囲
			int searchLower = 0;
			int searchUpper = 0x3FFFFFFF;

			Result.Clear();

			PrepareSix(0);

			// 中断あり
			Parallel.For(searchLower, searchUpper, (ivs, state) => {
				ulong result = SearchSix(ivs);
				if (result != 0)
				{
					Result.Add(result);
//					state.Stop();
				}
			});
		}
	}
}
