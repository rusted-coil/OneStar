using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OneStar
{
	public static class FlagExtensions
	{
		public static bool TestBit(this int flag, int bit)
		{
			return (flag & (1 << bit)) > 0;
		}
	}
}
