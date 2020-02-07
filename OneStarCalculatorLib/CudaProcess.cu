#include "CudaProcess.cuh"
#include "Data.h"

// 定数
static CudaConst* cu_HostConstData;
CudaConst* cu_DeviceConstData;

// 変数実体
CudaInputMaster* cu_HostInputMaster;
_u32* cu_HostInputCoefficientData;
_u32* cu_HostInputSearchPattern;

// 結果共通
int* cu_HostResultCount;
_u64* cu_HostResult;

// 定数
const int c_SizeResult = 32;

// 初期化
void CudaInitializeImpl()
{
	// ホストメモリの確保
	cudaMallocHost(&cu_HostConstData, sizeof(CudaConst));
	cudaMallocHost(&cu_HostInputMaster, sizeof(CudaInputMaster));
	cudaMallocHost(&cu_HostResultCount, sizeof(int));
	cudaMallocHost(&cu_HostResult, sizeof(_u64) * c_SizeResult);

	// デバイスメモリの確保
	cudaMalloc(&cu_DeviceConstData, sizeof(CudaConst));

	// データの初期化
	cu_HostInputMaster->ecBit = -1;

	// 定数データを転送
	cu_HostConstData->natureTable[0] = c_NatureTable[0];
	cu_HostConstData->natureTable[1] = c_NatureTable[1];
	cu_HostConstData->natureTable[2] = c_NatureTable[2];
	cudaMemcpy(cu_DeviceConstData, cu_HostConstData, sizeof(CudaConst), cudaMemcpyHostToDevice);
}

// 終了
void CudaFinalizeImpl()
{
	// デバイスメモリ解放
	cudaFree(cu_DeviceConstData);

	// ホストメモリ解放
	cudaFreeHost(cu_HostResult);
	cudaFreeHost(cu_HostResultCount);
	cudaFreeHost(cu_HostInputMaster);
	cudaFreeHost(cu_HostConstData);
}
