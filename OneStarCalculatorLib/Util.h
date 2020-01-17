#pragma once
#include "Type.h"

inline _u64 GetSignature(_u64 value)
{
	unsigned int a = (unsigned int)(value ^ (value >> 32));
	a = a ^ (a >> 16);
	a = a ^ (a >> 8);
	a = a ^ (a >> 4);
	a = a ^ (a >> 2);
	return (a ^ (a >> 1)) & 1;
}

struct PokemonData
{
	int ivs[6];
	int ability;
	int nature;
	int characteristic;
	bool isNoGender;
	int abilityFlag;
	int flawlessIvs;

	bool IsCharacterized(int index) // H A B "S" C D
	{
		switch (index)
		{
			case 0: return (ivs[0] == 31);
			case 1: return (ivs[1] == 31);
			case 2: return (ivs[2] == 31);
			case 3: return (ivs[5] == 31);
			case 4: return (ivs[3] == 31);
			case 5: return (ivs[4] == 31);
			default: return true;
		}
	}
};
