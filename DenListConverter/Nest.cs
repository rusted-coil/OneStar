using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DenListConverter
{
	public class Nest
	{
		public string Id { get; }
		public List<Entry> Entries { get; } = new List<Entry>();

		public Nest(string id)
		{
			Id = id;
		}

		public void Load(List<string> lines)
		{
			Entry entry = null;
			List<string> entryLines = new List<string>();
			foreach (var line in lines)
			{
				if (line.Length > 2)
				{
					// 最初の行
					if (line[0] == '\t' && line[1] != '\t')
					{
						if (entry != null)
						{
							entry.Load(entryLines);
							Entries.Add(entry);
							entryLines.Clear();
						}

						int index = line.IndexOf(' ');
						string first = line.Substring(1, index - 1);
						string second = line.Substring(index + 1);
						int star = 0;
						if (first == "1-Star")
						{
							star = 1;
						}
						else if (first == "2-Star")
						{
							star = 2;
						}
						else if (first == "3-Star")
						{
							star = 3;
						}
						else if (first == "4-Star")
						{
							star = 4;
						}
						else if (first == "5-Star")
						{
							star = 5;
						}
						else if (first == "0-Star")
						{
							star = 0;
						}
						else
						{
							;
						}

						bool gigantamax = false;
						if (second.IndexOf("Gigantamax") == 0)
						{
							second = second.Substring(11);
							gigantamax = true;
						}

						string name = second;
						int altform = 0;
						if (second[second.Length - 2] == '-' && second != "Jangmo-o" && second != "Hakamo-o" && second != "Kommo-o")
						{
							name = second.Substring(0, second.Length - 2);
							altform = int.Parse(second.Substring(second.Length - 1));
						}

						if (star > 0)
						{
							entry = new Entry(star, name, altform, gigantamax);
						}
						else
						{
							entry = null;
						}

						continue;
					}
					entryLines.Add(line);
				}
			}
			if (entry != null)
			{
				entry.Load(entryLines);
				Entries.Add(entry);
			}
		}

		public void Output(StreamWriter sw, string suffix)
		{
			sw.WriteLine($"        \"{Id + suffix}\":{{");
			sw.WriteLine("            \"Entries\":[");

			for (int i = 1; i <= 5; ++i)
			{
				foreach (var entry in Entries)
				{
					if (entry.Star == i)
					{
						entry.Output(sw);
					}
				}
			}

			sw.WriteLine("            ]},");
		}

		public void Intermediate(StreamWriter sw, string suffix)
		{
			sw.WriteLine($"====={Id + suffix}");

			for (int i = 0; i < 5; ++i)
			{
				foreach (var entry in Entries)
				{
					if (entry.Probability[i] > 0)
					{
						int species = -1;
						string name = entry.Name;
						{
							var list = PKHeX.Core.Util.GetSpeciesList("en");
							for (int a = 0; a < list.Length; ++a)
							{
								if (name == list[a])
								{
									species = a;
									break;
								}
							}
						}
						// 日本語ポケモン名検索
						if(species != -1)
						{
							name = PKHeX.Core.Util.GetSpeciesList("ja")[species];
						}

						string output = $"★{i + 1}: {name}";
						if (entry.Gigantamax)
						{
							output += "(キョダイ)";
						}
						if (entry.Ability == 2)
						{
							output += "[HA]";
						}

						sw.WriteLine(output);
					}
				}
			}
		}
	}
}
