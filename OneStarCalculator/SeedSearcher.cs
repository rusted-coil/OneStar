using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.Runtime.InteropServices;

namespace OneStarCalculator
{
	public class SeedSearcher
	{
		// モード
		public enum Mode {
			Star12,
			Star35_5,
			Star35_6,
			CudaTest,
		};
		Mode m_Mode;

		// 結果
		public List<ulong> Result { get; } = new List<ulong>();

		// ★1～2検索
		[DllImport("OneStarCalculatorLib.dll")]
		static extern void Prepare(int rerolls);

		[DllImport("OneStarCalculatorLib.dll")]
		public static extern void Set12Condition(int index, int iv0, int iv1, int iv2, int iv3, int iv4, int iv5, int ability, int nature, int characteristic, bool noGender, int abilityFlag, int flawlessIvs);

		[DllImport("OneStarCalculatorLib.dll")]
		static extern ulong Search(ulong ivs);

		// ★3～5検索
		[DllImport("OneStarCalculatorLib.dll")]
		static extern void PrepareSix(int ivOffset);

		[DllImport("OneStarCalculatorLib.dll")]
		public static extern void Set35Condition(int index, int iv0, int iv1, int iv2, int iv3, int iv4, int iv5, int ability, int nature, int characteristic, bool noGender, int abilityFlag, int flawlessIvs);

		[DllImport("OneStarCalculatorLib.dll")]
		public static extern void SetTargetCondition6(int iv1, int iv2, int iv3, int iv4, int iv5, int iv6);

		[DllImport("OneStarCalculatorLib.dll")]
		public static extern void SetTargetCondition5(int iv1, int iv2, int iv3, int iv4, int iv5);

		[DllImport("OneStarCalculatorLib.dll")]
		static extern ulong SearchSix(ulong ivs);

		// CUDAテスト
		[DllImport("OneStarCalculatorLib.dll")]
		static extern void PrepareCuda(int ivOffset);

		[DllImport("OneStarCalculatorLib.dll")]
		public static extern void SetCudaCondition(int index, int iv0, int iv1, int iv2, int iv3, int iv4, int iv5, int ability, int nature, int characteristic, bool noGender, int abilityFlag, int flawlessIvs);

		[DllImport("OneStarCalculatorLib.dll")]
		public static extern void SetCudaTargetCondition6(int iv1, int iv2, int iv3, int iv4, int iv5, int iv6);

		[DllImport("OneStarCalculatorLib.dll")]
		public static extern void SetCudaTargetCondition5(int iv1, int iv2, int iv3, int iv4, int iv5);

		[DllImport("OneStarCalculatorLib.dll")]
		static extern void PreCalc(ulong ivs, int freeBit);

		[DllImport("OneStarCalculatorLib.dll")]
		static extern ulong SearchCuda(int threadId);


		public SeedSearcher(Mode mode)
		{
			m_Mode = mode;
		}

		private readonly object lockObj = new object();
		public void Calculate(bool isEnableStop, int minRerolls, int maxRerolls, System.IProgress<int> p)
		{
			Result.Clear();

			if (m_Mode == Mode.CudaTest)
			{
				// 探索範囲
				int searchLower = 0;
				int searchUpper = 0x3F;

				// C++ライブラリ側の事前計算
				PrepareCuda(minRerolls);

				for (int i = searchLower; i <= searchUpper; ++i)
				{
					PreCalc((ulong)i, 24);
					Parallel.For(0, 1024 * 1024 * 16, (threadId) =>
					{
						ulong result = SearchCuda(threadId);
						if (result != 0)
						{
							Result.Add(result);
						}
					});
				}
			}

			else if (m_Mode == Mode.Star12)
			{
				// 探索範囲
				int searchLower = 0;
				int searchUpper = 0xFFFFFFF;

				// 途中経過
				int chunkPart = 16;

				int progress = 0;
				int chunkSize = searchUpper / chunkPart;
				int chunkMax = chunkPart * (maxRerolls - minRerolls + 1);
				int chunkCount = 1;

				p.Report(0);

				for (int i = minRerolls; i <= maxRerolls; ++i)
				{
					int chunkOffset = (i - minRerolls) * chunkPart;

					progress = 0;
					chunkCount = 0;

					// C++ライブラリ側の事前計算
					Prepare(i);

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
							Interlocked.Increment(ref progress);
							if (progress >= chunkCount * chunkSize)
							{
								lock (lockObj)
								{
									if (progress >= chunkCount * chunkSize)
									{
										p.Report((chunkCount + chunkOffset) * 1000 / chunkMax);
										++chunkCount;
									}
								}
							}
						});
						if (Result.Count > 0)
						{
							break;
						}
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
							Interlocked.Increment(ref progress);
							if (progress >= chunkCount * chunkSize)
							{
								lock (lockObj)
								{
									if (progress >= chunkCount * chunkSize)
									{
										p.Report((chunkCount + chunkOffset) * 1000 / chunkMax);
										++chunkCount;
									}
								}
							}
						});
					}
				}
			}
			else if (m_Mode == Mode.Star35_5 || m_Mode == Mode.Star35_6)
			{
				// 探索範囲
				int searchLower = 0;
				int searchUpper = (m_Mode == Mode.Star35_5 ? 0x1FFFFFF : 0x3FFFFFFF);

				// 途中経過
				int chunkPart = 32;

				int progress = 0;
				int chunkSize = searchUpper / chunkPart;
				int chunkMax = chunkPart * (maxRerolls - minRerolls + 1);
				int chunkCount = 1;

				p.Report(0);

				for (int i = minRerolls; i <= maxRerolls; ++i)
				{
					int chunkOffset = (i - minRerolls) * chunkPart;

					progress = 0;
					chunkCount = 0;

					// C++ライブラリ側の事前計算
					PrepareSix(i);

					// 並列探索
					if (isEnableStop)
					{
						// 中断あり
						Parallel.For(searchLower, searchUpper, (ivs, state) =>
						{
							ulong result = SearchSix((ulong)ivs);
							if (result != 0)
							{
								Result.Add(result);
								state.Stop();
							}
							Interlocked.Increment(ref progress);
							if (progress >= chunkCount * chunkSize)
							{
								lock (lockObj)
								{
									if (progress >= chunkCount * chunkSize)
									{
										p.Report((chunkCount + chunkOffset) * 1000 / chunkMax);
										++chunkCount;
									}
								}
							}
						});
						if (Result.Count > 0)
						{
							break;
						}
					}
					else
					{
						// 中断なし
						Parallel.For(searchLower, searchUpper, (ivs) =>
						{
							ulong result = SearchSix((ulong)ivs);
							if (result != 0)
							{
								Result.Add(result);
							}
							Interlocked.Increment(ref progress);
							if (progress >= chunkCount * chunkSize)
							{
								lock (lockObj)
								{
									if (progress >= chunkCount * chunkSize)
									{
										p.Report((chunkCount + chunkOffset) * 1000 / chunkMax);
										++chunkCount;
									}
								}
							}
						});
					}
				}

				// 結果を加工（[-3]にする）
				for (int i = 0; i < Result.Count; ++i)
				{
					for (int a = 0; a < 3; ++a)
					{
						Result[i] = Result[i] + 0x7d5d4e8add6295a5ul;
					}
				}
			}
		}
	}
}
