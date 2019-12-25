#pragma once
#include "Type.h"

extern _u64 g_InputMatrix[64];
extern _u64 g_ConstantTermVector;
extern _u64 g_Coefficient[64];
extern _u64 g_AnswerFlag[64];
extern _u64 g_CoefficientData[0x80];

void CalculateInverseMatrix(int length);
void CalculateCoefficientData(int length);
