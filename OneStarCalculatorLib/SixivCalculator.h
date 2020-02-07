#pragma once
#include "Type.h"

extern "C"
{
	__declspec(dllexport) void PrepareSix(int ivOffset);
	__declspec(dllexport) void Set35Condition(int index, int iv0, int iv1, int iv2, int iv3, int iv4, int iv5, int ability, int nature, int natureTableId, int characteristic, bool isNoGender, int abilityFlag, int flawlessIvs);
	__declspec(dllexport) void SetTargetCondition6(int iv1, int iv2, int iv3, int iv4, int iv5, int iv6);
	__declspec(dllexport) void SetTargetCondition5(int iv1, int iv2, int iv3, int iv4, int iv5);
	__declspec(dllexport) _u64 SearchSix(_u64);
}
