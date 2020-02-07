#pragma once
#include "Type.h"

extern "C"
{
	__declspec(dllexport) void Prepare(int rerolls);
	__declspec(dllexport) void Set12Condition(int index, int iv0, int iv1, int iv2, int iv3, int iv4, int iv5, int ability, int nature, int natureTableId, int characteristic, bool isNoGender, int abilityFlag, int flawlessIvs);
	__declspec(dllexport) _u64 Search(_u64);
}
