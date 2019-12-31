#pragma once
#include "Type.h"

extern _u64 g_InputMatrix[64];
extern _u64 g_ConstantTermVector;
extern _u64 g_Coefficient[64];
extern _u64 g_AnswerFlag[64];
extern _u64 g_CoefficientData[0x4000];
extern _u64 g_SearchPattern[0x4000];
extern int g_FreeBit[64];
extern int g_FreeId[64];

void CalculateInverseMatrix(int length);
void CalculateCoefficientData(int length);
