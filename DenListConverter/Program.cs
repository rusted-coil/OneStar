using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using PKHeX.Core;
using System.IO;

namespace DenListConverter
{
	class Program
	{
		static void Main(string[] args)
		{
            // イベントレイド向け
			string inputPath = "input.csv";
			string outputPath = "output.json";

			using (StreamReader sr = new StreamReader(inputPath, Encoding.GetEncoding("shift-jis")))
			{
				using (StreamWriter sw = new StreamWriter(outputPath))
				{
					// ヘッダ出力
					sw.WriteLine("{");
					sw.WriteLine("    \"EventList\":{");

					int tableCount = 0;

					sr.ReadLine(); // 1行目は飛ばす
					for (int c = 2; ; ++c)
					{
						string line = sr.ReadLine();
						if (line == null || line == "")
						{
							break;
						}

						string[] elements = line.Split(',');

						if (elements.Length != 12)
						{
							Console.WriteLine($"{c}行目が不正です。スキップします。");
							continue;
						}

						// ID定義行
						if (elements[3].Length == 0)
						{
							if (tableCount > 0)
							{
								// 前の〆
								sw.WriteLine("            ]},");
							}

							string id = elements[0];
							string version = elements[1];
							string type = elements[2];

							string tableId = id;
							if (version == "ソード")
							{
								tableId += "_Sw";
							}
							else if(version == "シールド")
							{
								tableId += "_Sh";
							}
							if (type == "通常")
							{
								tableId += "_n";
							}
							else if(type == "レア")
							{
								tableId += "_r";
							}

							sw.WriteLine($"        \"{tableId}\":{{");
							sw.WriteLine("            \"Entries\":[");

							++tableCount;
							continue;
						}

						// ポケモン名
						string name = elements[0];

						// 確率
						int[] probability = new int[5];
						for (int i = 0; i < 5; ++i)
						{
							probability[i] = int.Parse(elements[i + 1]);
						}

						// V数
						int flawlessIVs = int.Parse(elements[6]);

						// その他のパラメータ
						int altForm = int.Parse(elements[7]);
						int ability = int.Parse(elements[8]);
						int gender = int.Parse(elements[9]);

						bool gigantamax = (elements[10].Length != 0);
						bool shiny = (elements[11].Length != 0);

						string output = "                {\"Species\":";

						int species = -1;

						// 日本語ポケモン名検索
						{
							var list = PKHeX.Core.Util.GetSpeciesList("ja");
							for (int i = 0; i < list.Length; ++i)
							{
								if (name == list[i])
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
								if (name == list[i])
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
							if (probability[i] > 0)
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
							output += $"{probability[i],3}";
						}

						output += "],";

						output += $"\"FlawlessIvs\":{flawlessIVs},";
						output += $"\"MinRank\":{min},\"MaxRank\":{max},";
						output += $"\"AltForm\":{altForm},";
						output += $"\"Ability\":{ability},";
						output += $"\"Gender\":{gender},";
						if (gigantamax)
						{
							output += $"\"Gigantamax\":true,";
						}
						else
						{
							output += $"\"Gigantamax\":false,";
						}
						if (shiny)
						{
							output += $"\"ShinyType\":1";
						}
						else
						{
							output += $"\"ShinyType\":0";
						}
						output += "},";

						sw.WriteLine(output);
					}

					// フッタ出力
					sw.WriteLine("            ]},");
					sw.WriteLine("    }");
					sw.WriteLine("}");
				}
			}
            /*
            DenSet shieldDen = new DenSet();
            DenSet swordDen = new DenSet();

            shieldDen.Load("ShieldDen.txt");
            swordDen.Load("SwordDen.txt");

            string outputPath = "output.json";

            using (StreamWriter sw = new StreamWriter(outputPath))
            {
                // ヘッダ出力
                sw.WriteLine("{");
                sw.WriteLine("    \"EventList\":{");

                foreach (var pair in swordDen.NestDictionary)
                {
                    Nest swordNest = pair.Value;
                    Nest shieldNest = shieldDen.NestDictionary[pair.Key];
                    swordNest.Output(sw, "_Sw");
                    shieldNest.Output(sw, "_Sh");
                }

                // フッタ出力
                sw.WriteLine("    }");
                sw.WriteLine("}");
            }

            using (StreamWriter sw = new StreamWriter("intermediate.txt", false, Encoding.GetEncoding("shift-jis")))
            {
                int rareCountSw = 0;
                int rareCountSh = 0;
                foreach (var pair in swordDen.NestDictionary)
                {
                    Nest swordNest = pair.Value;
                    Nest shieldNest = shieldDen.NestDictionary[pair.Key];
                    swordNest.Intermediate(sw, "_Sw");
                    shieldNest.Intermediate(sw, "_Sh");

                    foreach (var entry in swordNest.Entries)
                    {
                        if (entry.Ability == 2)
                        {
                            rareCountSw++;
                            break;
                        }
                    }
                }
                ;
            }
            */
        }
	}
}
