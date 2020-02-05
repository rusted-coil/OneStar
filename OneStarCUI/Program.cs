using System;
using OneStarCalculator;

namespace OneStarCUI
{
	class Program
	{
		static void Main(string[] args)
		{
			// 適当に計算したりするテスト用コンソールです

			SeedSearcher searcher = new SeedSearcher(SeedSearcher.Mode.Cuda35);
			SeedSearcher.CudaInitialize();
			SeedSearcher.SetCudaCondition(0, 31, 27, 3, 19, 18, 31, 1, 15, 3, false, 3, 2);
			SeedSearcher.SetCudaCondition(1, 31, 9, 31, 31, 15, 31, 1, 18, 2, false, 3, 4);
			SeedSearcher.SetCudaCondition(2, 31, 31, 31, 1, 31, 27, 1, 6, 2, true, 4, 4);
			SeedSearcher.SetCudaCondition(3, 31, 2, 23, 31, 31, 31, 1, 15, 2, true, 4, 4);
			SeedSearcher.SetCudaTargetCondition6(27, 3, 19, 18, 9, 15);
			searcher.CudaLoopPartition = 2;

			Console.WriteLine("計算中...");

			// 時間計測
			var sw = new System.Diagnostics.Stopwatch();
			sw.Start();

			// 計算
			searcher.Calculate(false, 0, 0, null);

			sw.Stop();

			Console.WriteLine($"{searcher.Result.Count} results");
			foreach (ulong result in searcher.Result)
			{
				Console.WriteLine($"0x{result:X}");
			}
			Console.WriteLine($"{sw.ElapsedMilliseconds}[ms]");
		}
	}
}
