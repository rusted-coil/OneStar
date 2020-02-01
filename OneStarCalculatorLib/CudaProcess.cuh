#pragma once
#include "Type.h"
#include "Util.h"

struct CudaInputMaster
{
	// seedŒvZ’è”
	_u32 constantTermVector[2];
	_u32 answerFlag[128];
	_u32 coefficientData[32];
	_u32 searchPattern[16];

	// ŒŸõğŒ
	int ecBit;
	bool ecMod[3][6];
	int ivs[6];
};

// “ü—Í
extern CudaInputMaster* cu_HostMaster;
extern PokemonData* cu_HostPokemon;

// Œ‹‰Ê
extern _u64* cu_HostResult;

void CudaInitializeImpl();
void CudaSetMasterData();

void CudaProcess(_u32 ivs, int freeBit); //ˆ—ŠÖ”
void CudaFinalize();
