#pragma once
#include "CudaProcess.cuh"

// 5連個体値から計算用

void Cuda5Initialize();
void Cuda5SetMasterData();

void Cuda5Process(_u32 param, int partition); //処理関数
void Cuda5Finalize();
