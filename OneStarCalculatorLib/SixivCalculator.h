#pragma once
#include "Type.h"

extern "C"
{
	__declspec(dllexport) void PrepareSix(int ivOffset);
	__declspec(dllexport) void SetSixCondition(int fixed1, int fixed2, int iv1, int iv2, int iv3, int iv4, int iv5, int iv6, int nature);
	__declspec(dllexport) _u64 SearchSix(_u64);
}
