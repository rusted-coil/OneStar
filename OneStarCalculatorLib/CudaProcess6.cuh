#pragma once
#include "CudaProcess.cuh"

// 6連個体値から計算用

void Cuda6Initialize();
void Cuda6SetMasterData();

void Cuda6Process(_u32 param, int partition); //処理関数
void Cuda6Finalize();
