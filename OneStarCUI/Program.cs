using System;
using OneStarCalculator;

namespace OneStarCUI
{
	class Program
	{
		static void Main(string[] args)
		{
			SeedSearcher searcher = new SeedSearcher();

			// パラメータを設定
			/*
			Console.WriteLine("1匹目: ");
			string[] str = Console.ReadLine().Split(' ');
			SeedSearcher.SetFirstCondition(int.Parse(str[0]), int.Parse(str[1]), int.Parse(str[2]), int.Parse(str[3]), int.Parse(str[4]), int.Parse(str[5]), int.Parse(str[6]), int.Parse(str[7]));

			Console.WriteLine("2匹目: ");
			string[] str2 = Console.ReadLine().Split(' ');
			SeedSearcher.SetNextCondition(int.Parse(str2[0]), int.Parse(str2[1]), int.Parse(str2[2]), int.Parse(str2[3]), int.Parse(str2[4]), int.Parse(str2[5]), int.Parse(str2[6]), int.Parse(str2[7]), false);

			Console.WriteLine("計算中...");

	*/
			Console.WriteLine("1匹目: ");
			string[] str = Console.ReadLine().Split(' ');
			SeedSearcher.SetSixCondition(int.Parse(str[0]), int.Parse(str[1]), int.Parse(str[2]), int.Parse(str[3]), int.Parse(str[4]), int.Parse(str[5]), int.Parse(str[6]), int.Parse(str[7]), int.Parse(str[8]));

			Console.WriteLine("2匹目: ");
			string[] str2 = Console.ReadLine().Split(' ');

			Console.WriteLine("計算中...");

			// 時間計測
			var sw = new System.Diagnostics.Stopwatch();
			sw.Start();

			// 計算
			searcher.CalculateSix();

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
