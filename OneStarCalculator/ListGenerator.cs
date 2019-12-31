using System;
using System.IO;
using System.Diagnostics;

namespace OneStarCalculator
{
	public class ListGenerator
	{
		UInt64 m_DenSeed;
		int m_MaxCount;
		int m_VCount;
		bool m_isShinyCheck;
		bool m_isNoGender;
		bool m_isDream;
		bool m_isShowSeed;

		public ListGenerator(UInt64 denSeed, int maxCount, int vCount, bool isShinyCheck, bool isNoGender, bool isDream, bool isShowSeed)
		{
			m_DenSeed = denSeed;
			m_MaxCount = maxCount;
			m_VCount = vCount;
			m_isShinyCheck = isShinyCheck;
			m_isNoGender = isNoGender;
			m_isDream = isDream;
			m_isShowSeed = isShowSeed;
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

			using (StreamWriter sw = new StreamWriter("list.txt"))
			{
				if (m_isShowSeed)
				{
					sw.WriteLine("消費数,Seed,H,A,B,C,D,S,特性,性格,色違い");
				}
				else
				{
					sw.WriteLine("消費数,H,A,B,C,D,S,特性,性格,色違い");
				}

				for (int frame = 0; frame <= m_MaxCount; ++frame)
				{
					Xoroshiro xoroshiro = new Xoroshiro(seed);

					// seedを進める
					ec = xoroshiro.Next(0xFFFFFFFFu);
					otid = xoroshiro.Next(0xFFFFFFFFu);
					pid = xoroshiro.Next(0xFFFFFFFFu);

					bool isShiny = ((((otid ^ (otid >> 16)) >> 4) & 0xFFF) == (((pid ^ (pid >> 16)) >> 4) & 0xFFF));
					bool isSquare = (((otid ^ (otid >> 16)) & 0xFFFF) == ((pid ^ (pid >> 16)) & 0xFFFF));

					if (m_isShinyCheck && ! isShiny)
					{
						seed = seed + 0x82a2b175229d6a5bul;
						continue;
					}

					// V箇所決定
					for (int i = 0; i < 6; ++i)
					{
						ivs[i] = -1;
					}

					int fixedCount = 0;
					do
					{
						fixedIndex = 0;
						do
						{
							fixedIndex = xoroshiro.Next(7); // V箇所
						} while (fixedIndex >= 6);

						if (ivs[fixedIndex] == -1)
						{
							ivs[fixedIndex] = 31;
							++fixedCount;
						}
					} while (fixedCount < m_VCount);

					// 個体値を埋める
					for (int i = 0; i < 6; ++i)
					{
						if (ivs[i] == -1)
						{
							ivs[i] = (int)xoroshiro.Next(0x1F);
						}
					}

					// 特性
					if (m_isDream)
					{
						do
						{
							ability = xoroshiro.Next(3);
						} while (ability >= 3);
					}
					else
					{
						ability = xoroshiro.Next(1);
					}

					// 性別値
					if (!m_isNoGender)
					{
						do
						{
							gender = xoroshiro.Next(0xFF);
						} while (gender >= 253);
					}

					// 性格
					do
					{
						nature = xoroshiro.Next(0x1F);
					} while (nature >= 25);

					// 出力
					sw.Write($"{frame},");
					if (m_isShowSeed)
					{
						sw.Write($"{seed:X16},");
					}
					sw.Write($"{ivs[0]},{ivs[1]},{ivs[2]},{ivs[3]},{ivs[4]},{ivs[5]},");
					if (ability == 2)
					{
						sw.Write("夢,");
					}
					else
					{
						sw.Write($"{ ability + 1},");
					}
					sw.Write($"{Util.GetNatureString((int)nature)},");
					if (isShiny)
					{
						sw.WriteLine(isSquare ? "◆" : "★");
					}
					else if(!m_isShinyCheck)
					{
						sw.WriteLine($"");
					}

					seed = seed + 0x82a2b175229d6a5bul;
				}
			}

			Process ps = new Process();
			ps.StartInfo.FileName = "list.txt";
			ps.Start();
		}
	}
}
