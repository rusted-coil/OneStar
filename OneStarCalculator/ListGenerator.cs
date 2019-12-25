using System;
using System.IO;
using System.Diagnostics;

namespace OneStarCalculator
{
	public class ListGenerator
	{
		UInt64 m_DenSeed;
		int m_MaxCount;
		bool m_isShinyCheck;

		public ListGenerator(UInt64 denSeed, int maxCount, bool isShinyCheck)
		{
			m_DenSeed = denSeed;
			m_MaxCount = maxCount;
			m_isShinyCheck = isShinyCheck;
		}

		public void Generate()
		{
			UInt64 seed = m_DenSeed; // 消費数0のDen Seed

			UInt32 ec, otid, pid;
			int[] ivs = new int[6];
			UInt32 fixedIndex;
			UInt32 ability;
			UInt32 gender;
			UInt32 nature;

			using (StreamWriter sw = new StreamWriter("result.txt"))
			{
				sw.WriteLine("消費数,H,A,B,C,D,S,特性,性格,色違い");

				for (int frame = 0; frame <= m_MaxCount; ++frame)
				{
					Xoroshiro xoroshiro = new Xoroshiro(seed);

					// seedを進める
					ec = xoroshiro.Next(0xFFFFFFFFu);
					otid = xoroshiro.Next(0xFFFFFFFFu);
					pid = xoroshiro.Next(0xFFFFFFFFu);

					bool isShiny = ((((otid ^ (otid >> 16)) >> 4) & 0xFFF) == (((pid ^ (pid >> 16)) >> 4) & 0xFFF));

					// V箇所決定
					for (int i = 0; i < 6; ++i)
					{
						ivs[i] = -1;
					}

					do
					{
						fixedIndex = xoroshiro.Next(7);
					} while (fixedIndex >= 6);

					ivs[fixedIndex] = 31;

					// 個体値を埋める
					for (int i = 0; i < 6; ++i)
					{
						if (ivs[i] == -1)
						{
							ivs[i] = (int)xoroshiro.Next(0x1F);
						}
					}

					// 特性
					ability = xoroshiro.Next(1);

					// 性別値
					do
					{
						gender = xoroshiro.Next(0xFF);
					} while (gender >= 253);

					// 性格
					do
					{
						nature = xoroshiro.Next(0x1F);
					} while (nature >= 25);

					// 出力
					if (isShiny)
					{
						sw.WriteLine($"{frame},{ivs[0]},{ivs[1]},{ivs[2]},{ivs[3]},{ivs[4]},{ivs[5]},{ability + 1},{Util.GetNatureString((int)nature)},★"); ;
					}
					else if(!m_isShinyCheck)
					{
						sw.WriteLine($"{frame},{ivs[0]},{ivs[1]},{ivs[2]},{ivs[3]},{ivs[4]},{ivs[5]},{ability + 1},{Util.GetNatureString((int)nature)},"); ;
					}

					seed = seed + 0x82a2b175229d6a5bul;
				}
			}

			Process ps = new Process();
			ps.StartInfo.FileName = "result.txt";
			ps.Start();
		}
	}
}
