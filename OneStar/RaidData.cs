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
	}
}
