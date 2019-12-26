using System.Collections.Generic;
using System.Threading.Tasks;
using System.Runtime.InteropServices;

namespace OneStarCalculator
{
	public class SeedSearcher
	{
		// モード
		public enum Mode {
			Star12,
			Star35
		};
		Mode m_Mode;

		// 結果
		public List<ulong> Result { get; } = new List<ulong>();

		// ★1～2検索
		[DllImport("OneStarCalculatorLib.dll")]
		static extern void Prepare(int rerolls);

		[DllImport("OneStarCalculatorLib.dll")]
		public static extern void SetFirstCondition(int iv0, int iv1, int iv2, int iv3, int iv4, int iv5, int ability, int nature);

		[DllImport("OneStarCalculatorLib.dll")]
		public static extern void SetNextCondition(int iv0, int iv1, int iv2, int iv3, int iv4, int iv5, int ability, int nature, bool noGender);

		[DllImport("OneStarCalculatorLib.dll")]
		static extern ulong Search(ulong ivs);

		// ★3～5検索
		[DllImport("OneStarCalculatorLib.dll")]
		static extern void PrepareSix(int ivOffset);

		[DllImport("OneStarCalculatorLib.dll")]
		public static extern void SetSixCondition(int fixed1, int fixed2, int iv1, int iv2, int iv3, int iv4, int iv5, int iv6, int nature);

		[DllImport("OneStarCalculatorLib.dll")]
		static extern ulong SearchSix(ulong ivs);

		public SeedSearcher(Mode mode)
		{
			m_Mode = mode;
		}

		public void Calculate(bool isEnableStop)
		{
			Result.Clear();

			if (m_Mode == Mode.Star12)
			{
				// 探索範囲
				int searchLower = 0;
				int searchUpper = 0xFFFFFFF;

				// C++ライブラリ側の事前計算
				Prepare(0);

				// 並列探索
				if (isEnableStop)
				{
					// 中断あり
					Parallel.For(searchLower, searchUpper, (ivs, state) =>
					{
						ulong result = Search((ulong)ivs);
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
					Parallel.For(searchLower, searchUpper, (ivs) =>
					{
						ulong result = Search((ulong)ivs);
						if (result != 0)
						{
							Result.Add(result);
						}
					});
				}
			}
			else if (m_Mode == Mode.Star35)
			{
				// 探索範囲
				int searchLower = 0;
				int searchUpper = 0x3FFFFFFF;

				Result.Clear();

				// C++ライブラリ側の事前計算
				PrepareSix(0);

				// 並列探索
				if (isEnableStop)
				{
					// 中断あり
					Parallel.For(searchLower, searchUpper, (ivs, state) => {
						ulong result = SearchSix((ulong)ivs);
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
						ulong result = SearchSix((ulong)ivs);
						if (result != 0)
						{
							Result.Add(result);
						}
					});
				}
			}
		}
	}
}
