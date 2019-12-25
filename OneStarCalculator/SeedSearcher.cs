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
		static extern void Prepare();

		[DllImport("OneStarCalculatorLib.dll")]
		public static extern void SetFirstCondition(int iv0, int iv1, int iv2, int iv3, int iv4, int iv5, int ability, int nature);

		[DllImport("OneStarCalculatorLib.dll")]
		public static extern void SetNextCondition(int iv0, int iv1, int iv2, int iv3, int iv4, int iv5, int ability, int nature);

		[DllImport("OneStarCalculatorLib.dll")]
		static extern ulong Search(int ivs);

		public SeedSearcher()
		{
			// C++ライブラリ側の事前計算
			Prepare();
		}

		public void Calculate()
		{
			// 探索範囲
			int searchLower = 0;
			int searchUpper = 0xFFFFFFF;

			Result.Clear();

			// 並列探索
			Parallel.For(searchLower, searchUpper, Method);
		}

		// 並列計算タスク
		void Method(int ivs)
		{
			ulong result = Search(ivs);
			if (result != 0)
			{
				Result.Add(result);
			}
		}
	}
}
