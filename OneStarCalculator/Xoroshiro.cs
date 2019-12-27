using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OneStarCalculator
{
	public class Xoroshiro
	{
		UInt64 s0, s1;

		public int GetRaw(int index, UInt32 mask)
		{
			if (index == 0)
			{
				return (int)(s0 & mask);
			}
			return (int)(s1 & mask);
		}

		public Xoroshiro(UInt64 seed)
		{
			s0 = seed;
			s1 = 0x82a2b175229d6a5bul;
		}

		UInt64 RotateLeft(UInt64 value, int amount)
		{
			UInt64 left = (value << amount);
			UInt64 right = (value >> (64 - amount));
			return  left | right;
		}

		public UInt32 Next(UInt32 mask)
		{
			UInt32 value = (UInt32)(s0 + s1) & mask;

			s1 = s0 ^ s1;
			s0 = RotateLeft(s0, 24) ^ s1 ^ (s1 << 16);
			s1 = RotateLeft(s1, 37);

			return value;
		}
	}
}
