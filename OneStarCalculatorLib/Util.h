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

inline _u64 GetSignature7(unsigned short value)
{
	value = value ^ (value >> 4);
	value = value ^ (value >> 2);
	return (value ^ (value >> 1)) & 1;
}

struct PokemonData
{
	int ivs[6];
	int ability;
	int nature;
	bool isNoGender;
};
