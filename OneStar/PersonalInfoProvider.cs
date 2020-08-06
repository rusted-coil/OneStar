using PKHeX.Core;

namespace OneStar
{
	public static class PersonalInfoProvider
	{
		public static PersonalInfo GetPersonalInfo(RaidData.Pokemon pokemon)
		{
			// ルガルガン（たそがれ）のみ別で処理
			if (pokemon.DisplaySpecies == 745.2m)
			{
				int dataSpecies = PersonalTable.USUM[745].FormeIndex(745, 2);
				return PersonalTable.USUM[dataSpecies];
			}

			// 剣盾にデータが無いものはUSUMから取ってくる
			if (pokemon.IsDataSWSH)
			{
				return PersonalTable.SWSH[pokemon.DataSpecies];
			}
			else
			{
				return PersonalTable.USUM[pokemon.DataSpecies];
			}
		}
	}
}
