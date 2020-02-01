#pragma once
#include "Type.h"
#include "Util.h"

struct CudaInputMaster
{
	// seed計算定数
	_u32 constantTermVector[2];
	_u32 answerFlag[128];
	_u32 coefficientData[32];
	_u32 searchPattern[16];

	// 検索条件
	int ecBit;
	bool ecMod[3][6];
	int ivs[6];
};

// 入力
extern CudaInputMaster* cu_HostMaster;
extern PokemonData* cu_HostPokemon;

// 結果
extern _u64* cu_HostResult;

void CudaInitializeImpl();
void CudaSetMasterData();

void CudaProcess(_u32 ivs, int freeBit); //処理関数
void CudaFinalize();
