using System;
using PKHeX_Raid_Plugin;

namespace OneStar
{
	public class RaidData
	{
		// 巣穴データ（Raid_Pluginより）
		readonly RaidTables c_RaidTables = new RaidTables();

		public RaidTemplateTable GetRaidTemplateTable(int raidIndex, int version, int rarity)
		{
			// イベントレイド
			if (raidIndex == -1)
			{
				RaidTemplateTable[] raidTables = (version == 0 ? c_RaidTables.SwordNestsEvent : c_RaidTables.ShieldNestsEvent);
				return Array.Find(raidTables, table => table.TableID == NestLocations.EventHash);
			}
			else
			{
				var detail = NestLocations.Nests[raidIndex];
				RaidTemplateTable[] raidTables = (version == 0 ? c_RaidTables.SwordNests : c_RaidTables.ShieldNests);
				if (rarity == 0)
				{
					return Array.Find(raidTables, table => table.TableID == detail.CommonHash);
				}
				else
				{
					return Array.Find(raidTables, table => table.TableID == detail.RareHash);
				}
			}
		}

		// 1ランク1ポケモンごとに対応するデータ
		public class Pokemon
		{
			public string Key { get; private set; }
			public int Rank { get; private set; }
			public RaidTemplate Entry { get; private set; }

			public decimal Species { get; private set; } // 個体値計算上の種族
			public bool IsFixedDream { get; private set; }

			public override string ToString() { return Key; }

			public Pokemon(string key, int rank, RaidTemplate entry)
			{
				Key = key;
				Rank = rank;
				Entry = entry;

				int rawSpecies = entry.Species;
				Species = rawSpecies;

				// TODO ガラル

				IsFixedDream = (entry.Ability == 2);
			}

			public void Merge(RaidTemplate entry)
			{
				// 夢特性固定があれば上書き
				IsFixedDream = IsFixedDream || (entry.Ability == 2);
			}
		}
	}
}
