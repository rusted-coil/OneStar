using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OneStarCalculator
{
	public class Util
	{
		public static string GetNatureString(int index)
		{
			switch (index)
			{
				case 0: return "がんばりや";
				case 1: return "さみしがり";
				case 2: return "ゆうかん";
				case 3: return "いじっぱり";
				case 4: return "やんちゃ";
				case 5: return "ずぶとい";
				case 6: return "すなお";
				case 7: return "のんき";
				case 8: return "わんぱく";
				case 9: return "のうてんき";
				case 10: return "おくびょう";
				case 11: return "せっかち";
				case 12: return "まじめ";
				case 13: return "ようき";
				case 14: return "むじゃき";
				case 15: return "ひかえめ";
				case 16: return "おっとり";
				case 17: return "れいせい";
				case 18: return "てれや";
				case 19: return "うっかりや";
				case 20: return "おだやか";
				case 21: return "おとなしい";
				case 22: return "なまいき";
				case 23: return "しんちょう";
				case 24: return "きまぐれ";
			}
			return "";
		}

		public static string GetParameterString(int index)
		{
			switch (index)
			{
				case 0: return "H";
				case 1: return "A";
				case 2: return "B";
				case 3: return "C";
				case 4: return "D";
				case 5: return "S";
			}
			return "";
		}

	}
}
