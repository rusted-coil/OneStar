using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OneStar
{
	public class AbilityType
	{
		public enum Flag
		{
			HiddenPossible = 0, // 夢特性あり
			HiddenFixed = 1, // 夢特性固定あり
		}

		int m_Flag = 0;
	}
}
