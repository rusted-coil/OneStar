#include "CudaProcess.cuh"

// 変数実態
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
	cudaMallocHost(&cu_HostInputMaster, sizeof(CudaInputMaster));
	cudaMallocHost(&cu_HostResultCount, sizeof(int));
	cudaMallocHost(&cu_HostResult, sizeof(_u64) * c_SizeResult);

	// データの初期化
	cu_HostInputMaster->ecBit = -1;
}

// 終了
void CudaFinalizeImpl()
{
	// ホストメモリ解放
	cudaFreeHost(cu_HostResult);
	cudaFreeHost(cu_HostResultCount);
	cudaFreeHost(cu_HostInputMaster);
}
