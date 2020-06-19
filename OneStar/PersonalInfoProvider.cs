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

			PersonalInfo info = PersonalTable.SWSH[pokemon.DataSpecies];
			// 剣盾にデータが無いものはUSUMから取ってくる
			if (info.ATK == 0)
			{
				info = PersonalTable.USUM[pokemon.DataSpecies];
			}
			return info;
		}
	}
}
