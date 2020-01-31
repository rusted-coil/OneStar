#pragma once
#include "Type.h"

struct CudaInputMaster
{
	int ivs[6];
	_u64 constantTermVector;
	_u64 answerFlag[64];
};

//ホストメモリのポインタ
extern CudaInputMaster* pHostMaster; // 固定データ
extern _u64* cu_HostResult;

//デバイスメモリのポインタ
extern CudaInputMaster* pDeviceMaster;
extern _u64* pDeviceResult;

void CudaInitialize(int* pIvs);
void CudaProcess(_u64 ivs, int freeBit); //処理関数
void CudaFinalize();
