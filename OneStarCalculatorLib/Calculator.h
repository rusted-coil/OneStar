#pragma once

typedef unsigned long long _u64;

extern "C"
{
	__declspec(dllexport) void Prepare();
	__declspec(dllexport) void SetFirstCondition(int iv0, int iv1, int iv2, int iv3, int iv4, int iv5, int ability, int nature);
	__declspec(dllexport) void SetNextCondition(int iv0, int iv1, int iv2, int iv3, int iv4, int iv5, int ability, int nature);
	__declspec(dllexport) _u64 Search(int);
}
