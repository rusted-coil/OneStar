﻿using System;
using System.IO;
using System.Diagnostics;
using PKHeX.Core;

namespace OneStar
{
	public class ListGenerator
	{
		UInt64 m_DenSeed;
		int m_MaxCount;
		RaidData.Pokemon m_Pokemon;
		bool m_isShinyCheck;
		bool m_isShowSeed;
		bool m_isShowEc;
		bool m_isShowAbilityName;

		public ListGenerator(UInt64 denSeed, int maxCount, RaidData.Pokemon pokemon, bool isShinyCheck, bool isShowSeed, bool isShowEc, bool isShowAbilityName)
		{
			m_DenSeed = denSeed;
			m_MaxCount = maxCount;
			m_Pokemon = pokemon;
			m_isShinyCheck = isShinyCheck;
			m_isShowSeed = isShowSeed;
			m_isShowEc = isShowEc;
			m_isShowAbilityName = isShowAbilityName;
		}

		public void Generate()
		{
			PersonalInfo personalInfo = PersonalInfoProvider.GetPersonalInfo(m_Pokemon);
			var abilityList = PKHeX.Core.Util.GetAbilitiesList(Messages.Instance.LangCode);

			UInt64 seed = m_DenSeed; // 消費数0のDen Seed

			UInt32 ec, otid, pid;
			int[] ivs = new int[6];
			UInt32 fixedIndex;
			UInt32 ability;
			UInt32 gender = 0;
			UInt32 nature;

			using (StreamWriter sw = new StreamWriter("list.txt"))
			{
				sw.Write($"{Messages.Instance.ListLabel["Frame"]},");
				if (m_isShowSeed)
				{
					sw.Write("Seed,");
				}
				if (m_isShowEc)
				{
					sw.Write("EC,");
				}
				for (int i = 0; i < 6; ++i)
				{
					sw.Write($"{Messages.Instance.Status[i]},");
				}
				if (!m_Pokemon.IsFixedGender)
				{
					sw.Write($"{Messages.Instance.SystemLabel["Gender"]},");
				}
				sw.Write($"{Messages.Instance.ListLabel["Ability"]},");
				sw.Write($"{Messages.Instance.ListLabel["Nature"]},");
				sw.WriteLine($"{Messages.Instance.ListLabel["Shiny"]}");

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
					} while (fixedCount < m_Pokemon.FlawlessIvs);

					// 個体値を埋める
					for (int i = 0; i < 6; ++i)
					{
						if (ivs[i] == -1)
						{
							ivs[i] = (int)xoroshiro.Next(0x1F);
						}
					}

					// 特性
					if (m_Pokemon.Ability == 2)
					{
						ability = 2;
					}
					else if(m_Pokemon.Ability == 4)
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
					if (!m_Pokemon.IsFixedGender)
					{
						do
						{
							gender = xoroshiro.Next(0xFF);
						} while (gender >= 253);
					}

					// 性格
					do
					{
						nature = xoroshiro.Next(PokemonFormUtility.NatureTableList[m_Pokemon.NatureTableId].Max);
					} while (nature >= PokemonFormUtility.NatureTableList[m_Pokemon.NatureTableId].Pattern);

					// 出力
					sw.Write($"{frame},");
					if (m_isShowSeed)
					{
						sw.Write($"{seed:X16},");
					}
					if (m_isShowEc)
					{
						sw.Write($"{ec:X8},");
					}
					sw.Write($"{ivs[0]},{ivs[1]},{ivs[2]},{ivs[3]},{ivs[4]},{ivs[5]},");
					if (!m_Pokemon.IsFixedGender)
					{
						if (gender + 1 < personalInfo.Gender)
						{
							sw.Write($"{Messages.Instance.Gender[1]},");
						}
						else
						{
							sw.Write($"{Messages.Instance.Gender[0]},");
						}
					}
					if (m_isShowAbilityName)
					{
						sw.Write($"{abilityList[personalInfo.Abilities[(int)ability]]},");
					}
					else
					{
						sw.Write(ability == 2 ? $"{Messages.Instance.ListLabel["HiddenAbility"]}," : $"{ability + 1},");
					}
					sw.Write($"{PokemonFormUtility.GetNatureString(PokemonFormUtility.NatureTableList[m_Pokemon.NatureTableId].List[nature])},");
					if (isShiny)
					{
						sw.WriteLine(isSquare ? Messages.Instance.ListLabel["SquareShiny"] : Messages.Instance.ListLabel["StarShiny"]);
					}
					else if(!m_isShinyCheck)
					{
						sw.WriteLine("");
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
