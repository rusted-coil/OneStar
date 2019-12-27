#pragma once
#include "Type.h"

extern "C"
{
	__declspec(dllexport) void PrepareSix(int ivOffset);
	__declspec(dllexport) void SetSixFirstCondition(int iv1, int iv2, int iv3, int iv4, int iv5, int iv6, int ability, int nature, bool isNoGender);
	__declspec(dllexport) void SetSixSecondCondition(int iv1, int iv2, int iv3, int iv4, int iv5, int iv6, int ability, int nature, bool isNoGender);
	__declspec(dllexport) void SetSixThirdCondition(int iv1, int iv2, int iv3, int iv4, int iv5, int iv6, int ability, int nature, bool isNoGender);
	__declspec(dllexport) void SetTargetCondition(int iv1, int iv2, int iv3, int iv4, int iv5, int iv6, int ability);
	__declspec(dllexport) _u64 SearchSix(_u64);
}
