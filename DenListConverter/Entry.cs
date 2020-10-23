using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DenListConverter
{
	public class Entry
	{
		public int Star { get; }
		public string Name { get; }
		public int AltForm { get; }
		public bool Gigantamax { get; }
		public int Gender { get; private set; }
		public int Ability { get; private set; }
		public int[] Probability { get; }

		public Entry(int star, string name, int altform, bool gigantamax)
		{
			Star = star;
			Name = name;
			AltForm = altform;
			Gigantamax = gigantamax;
			Probability = new int[5];
			for (int i = 0; i < 5; ++i)
			{
				Probability[i] = 0;
			}
		}

		public void Load(List<string> lines)
		{
			foreach (var line in lines)
			{
				// 性別
				if (line.IndexOf("\t\tGender: ") == 0)
				{
					string gender = line.Substring(10);
					if (gender == "Random")
					{
						Gender = 0;
					}
					else if (gender == "Female")
					{
						Gender = 1;
					}
					else if (gender == "Male")
					{
						Gender = 2;
					}
					else
					{
						;
					}
				}

				// 特性
				else if (line.IndexOf("\t\tAbility: ") == 0)
				{
					string ability = line.Substring(11);
					if (ability == "A2")
					{
						Ability = 2;
					}
					else if (ability == "A3")
					{
						Ability = 3;
					}
					else if (ability == "A4")
					{
						Ability = 4;
					}
					else
					{
						;
					}
				}

				// 確率
				if (line.IndexOf("\t\t\t1-Star Desired: ") == 0)
				{
					string[] elements = line.Split(' ');
					string percentage = elements[elements.Length - 1];
					string num = percentage.Substring(0, percentage.Length - 1);
					int probability = int.Parse(num);
					Probability[0] += probability;
				}
				else if (line.IndexOf("\t\t\t2-Star Desired: ") == 0)
				{
					string[] elements = line.Split(' ');
					string percentage = elements[elements.Length - 1];
					string num = percentage.Substring(0, percentage.Length - 1);
					int probability = int.Parse(num);
					Probability[1] += probability;
				}
				else if (line.IndexOf("\t\t\t3-Star Desired: ") == 0)
				{
					string[] elements = line.Split(' ');
					string percentage = elements[elements.Length - 1];
					string num = percentage.Substring(0, percentage.Length - 1);
					int probability = int.Parse(num);
					Probability[2] += probability;
				}
				else if (line.IndexOf("\t\t\t4-Star Desired: ") == 0)
				{
					string[] elements = line.Split(' ');
					string percentage = elements[elements.Length - 1];
					string num = percentage.Substring(0, percentage.Length - 1);
					int probability = int.Parse(num);
					Probability[3] += probability;
				}
				else if (line.IndexOf("\t\t\t5-Star Desired: ") == 0)
				{
					string[] elements = line.Split(' ');
					string percentage = elements[elements.Length - 1];
					string num = percentage.Substring(0, percentage.Length - 1);
					int probability = int.Parse(num);
					Probability[4] += probability;
				}
			}
		}

		public void Output(StreamWriter sw)
		{
			string output = "                {\"Species\":";

			int species = -1;

			// 日本語ポケモン名検索
			{
				var list = PKHeX.Core.Util.GetSpeciesList("ja");
				for (int i = 0; i < list.Length; ++i)
				{
					if (Name == list[i])
					{
						species = i;
						break;
					}
				}
			}
			// 英語ポケモン名検索
			if (species == -1)
			{
				var list = PKHeX.Core.Util.GetSpeciesList("en");
				for (int i = 0; i < list.Length; ++i)
				{
					if (Name == list[i])
					{
						species = i;
						break;
					}
				}
			}

			output += $"{species,3},";

			output += "\"Probabilities\":[";

			int min = -1;
			int max = -1;
			for (int i = 0; i < 5; ++i)
			{
				if (Probability[i] > 0)
				{
					if (min == -1)
					{
						min = i;
					}
					max = i;
				}

				if (i > 0)
				{
					output += ",";
				}
				output += $"{Probability[i],3}";
			}

			output += "],";

			int flawlessIVs = 0;

            if (Ability == 3) // レア判定
            {
                if (max == 4)
                {
                    flawlessIVs = 5;
                }
                else if (min == 0)
                {
                    flawlessIVs = 2;
                }
                else if (min == 1)
                {
                    flawlessIVs = 3;
                }
                else if (min == 2)
                {
                    flawlessIVs = 4;
                }
                else if (min == 3)
                {
                    // ?
                    flawlessIVs = 4;
                }
            }
            else
            {
                // 鎧の孤島、冠の雪原ノーマル
			    if (max == 4)
			    {
				    flawlessIVs = 4;
			    }
			    else if (min == 0)
			    {
				    flawlessIVs = 1;
			    }
			    else if (min == 1)
			    {
				    flawlessIVs = 2;
			    }
			    else if (min == 2)
			    {
				    flawlessIVs = 3;
			    }
			    else if (min == 3)
			    {
				    // ?
				    flawlessIVs = 3;
			    }
            }

            output += $"\"FlawlessIvs\":{flawlessIVs},";
			output += $"\"MinRank\":{min},\"MaxRank\":{max},";
			output += $"\"AltForm\":{AltForm},";
			output += $"\"Ability\":{Ability},";
			output += $"\"Gender\":{Gender},";
			if (Gigantamax)
			{
				output += $"\"Gigantamax\":true,";
			}
			else
			{
				output += $"\"Gigantamax\":false,";
			}
			output += $"\"ShinyType\":0";

			output += "},";

			sw.WriteLine(output);
		}
	}
}
