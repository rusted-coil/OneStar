using System;
using OneStarCalculator;

namespace OneStarCUI
{
	class Program
	{
		static void Main(string[] args)
		{
			// 適当に計算したりするテスト用コンソールです
			// 0x123イベントレイド
			// マホミル★3
			// 4匹目3V 31 28 31  4 31 11 うっかりや 夢 ♀固定 個性1 うたれづよい
			// マホミル★4
			// 4匹目4V 31 10 31 31 31 27 ひかえめ 特性1 ♀固定 個性1 うたれづよい
			// マホミル★3
			// 5匹目3V 27 21 31 31 18 31 のうてんき 特性1 ♀固定 個性2 うたれづよい
			// マホミル★3
			// 6匹目3V 31 31  3 17 31  6 さみしがり 特性1 ♀固定 個性3 みえっぱり
			SeedSearcher searcher = new SeedSearcher(SeedSearcher.Mode.Cuda35);
			SeedSearcher.CudaInitialize();
			SeedSearcher.SetCudaCondition(0, 31, 28, 31,  4, 31, 11, 2, 19, 2, true, 4, 3);
			SeedSearcher.SetCudaCondition(1, 31, 10, 31, 31, 31, 27, 0, 15, 2, true, 4, 4);
			SeedSearcher.SetCudaCondition(2, 27, 21, 31, 31, 18, 31, 0,  9, 2, true, 4, 3);
			SeedSearcher.SetCudaCondition(3, 31, 31,  3, 17, 31,  6, 0,  1, 5, true, 4, 3);
			SeedSearcher.SetCudaTargetCondition5(28, 4, 11, 10, 27);

			// 0x123イベントレイド
			// ストリンダー★3
			// 4匹目3V 31 28 31  4 31 11 わんぱく　夢 ♂♀ 個性1 うたれづよい
			// マホミル★4
			// 4匹目4V 31 10 31 31 31 27 なまいき 特性1 ♂♀ 個性1 うたれづよい
			// マホミル★3
			// 5匹目3V 27 21 31 31 18 31 ゆうかん 特性1 ♂♀ 個性2 うたれづよい
			/*
			SeedSearcher searcher = new SeedSearcher(SeedSearcher.Mode.Cuda35);
			SeedSearcher.CudaInitialize();
			SeedSearcher.SetCudaCondition(0, 31, 28, 31, 4, 31, 11, 2, 8, 2, false, 4, 3);
			SeedSearcher.SetCudaCondition(1, 31, 10, 31, 31, 31, 27, 0, 22, 2, false, 4, 4);
			SeedSearcher.SetCudaCondition(2, 27, 21, 31, 31, 18, 31, 0, 2, 2, false, 4, 3);
			SeedSearcher.SetCudaTargetCondition5(28, 4, 11, 10, 27);
			SeedSearcher.SetCudaCondition(3, 31, 31, 3, 17, 31, 6, 0, 1, 5, true, 4, 3);
			*/
			searcher.CudaLoopPartition = 0;

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
