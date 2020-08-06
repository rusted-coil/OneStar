﻿using System;
using System.Collections.Generic;
using PKHeX_Raid_Plugin;
using PKHeX.Core;
using System.Runtime.CompilerServices;

namespace OneStar
{
	public class RaidData
	{
		// 巣穴データ（Raid_Pluginより）
		readonly RaidTables c_RaidTables = new RaidTables();

		// イベントレイドデータ
		EventDenList m_EventDenList;

		// DLCレイドデータ
		EventDenList m_AdditionalDenList;

		// マップスケール
		readonly float c_MapScale = 250.0f / 458.0f; // ビュー上の幅/画像の幅

		// ガラルポケモン
		static readonly Dictionary<int, decimal> c_GalarForms = new Dictionary<int, decimal>{
			{ 562, 562.1m }, // デスマス
			{ 618, 618.1m }, // マッギョ
			{  77,  77.1m }, // ポニータ
			{  78,  78.1m }, // ギャロップ
			{ 122, 122.1m }, // バリヤード
			{ 222, 222.1m }, // サニーゴ
			{ 263, 263.1m }, // ジグザグマ
			{ 264, 264.1m }, // マッスグマ
			{  52,  52.2m }, // ニャース
			{  83,  83.1m }, // カモネギ
			{ 110, 110.1m }, // マタドガス
			{ 554, 993m }, // ダルマッカ
			{ 555, 555.2m }, // ヒヒダルマ
		};

		// ♂のみ：バルキー系、エルレイド、ナゲキ、ダゲキ、ウォーグル系、オーロンゲ系
		// ♀のみ：ビークイン、ユキメノコ、バルジーナ系、アマージョ系、ブリムオン系、マホイップ系
		// 性別不明：メタモン、ヌケニン、ネンドール系、ドータクン系、ロトム、ギギギアル系、ゴルーグ系、シルヴァディ系、ダダリン、ポットデス系、タイレーツ、化石、伝説
		static readonly Dictionary<int, int> c_FixedGender = new Dictionary<int, int> {
			{ 236, 1 }, // バルキー
			{ 106, 1 }, // サワムラー
			{ 107, 1 }, // エビワラー
			{ 237, 1 }, // カポエラー
			{ 475, 1 }, // エルレイド
			{ 538, 1 }, // ナゲキ
			{ 539, 1 }, // ダゲキ
			{ 627, 1 }, // ワシボン
			{ 628, 1 }, // ウォーグル
			{ 859, 1 }, // ベロバー
			{ 860, 1 }, // ギモー
			{ 861, 1 }, // オーロンゲ
			{ 416, 2 }, // ビークイン
			{ 478, 2 }, // ユキメノコ
			{ 629, 2 }, // バルチャイ
			{ 630, 2 }, // バルジーナ
			{ 761, 2 }, // アマカジ
			{ 762, 2 }, // アママイコ
			{ 763, 2 }, // アマージョ
			{ 856, 2 }, // ミブリム
			{ 857, 2 }, // テブリム
			{ 858, 2 }, // ブリムオン
			{ 868, 2 }, // マホミル
			{ 869, 2 }, // マホイップ
			{ 132, 3 }, // メタモン
			{ 292, 3 }, // ヌケニン
			{ 343, 3 }, // ヤジロン
			{ 344, 3 }, // ネンドール
			{ 436, 3 }, // ドーミラー
			{ 437, 3 }, // ドータクン
			{ 479, 3 }, // ロトム
			{ 599, 3 }, // ギアル
			{ 600, 3 }, // ギギアル
			{ 601, 3 }, // ギギギアル
			{ 622, 3 }, // ゴビット
			{ 623, 3 }, // ゴルーグ
			{ 772, 3 }, // タイプ：ヌル
			{ 773, 3 }, // シルヴァディ
			{ 781, 3 }, // ダダリン
			{ 854, 3 }, // ヤバチャ
			{ 855, 3 }, // ポットデス
			{ 870, 3 }, // タイレーツ
			{ 880, 3 }, // パッチラゴン
			{ 881, 3 }, // パッチルドン
			{ 882, 3 }, // ウオノラゴン
			{ 883, 3 }, // ウオチルドン
			{ 888, 3 }, // ザシアン
			{ 889, 3 }, // ザマゼンタ
			{ 890, 3 }, // ムゲンダイナ
		};

		public RaidData()
		{
			m_EventDenList = new EventDenList();
			m_EventDenList.Load("data/EventDen.json");

			// DLCレイドデータ読み込み
			m_AdditionalDenList = new EventDenList();
			m_AdditionalDenList.Load("data/AdditionalDen.json");
		}

		// 恒常レイド
		public RaidTemplate[] GetRaidEntries(int raidIndex, int version, int rarity)
		{
			// イベントレイド
			if (raidIndex == -1)
			{
				RaidTemplateTable[] raidTables = (version == 0 ? c_RaidTables.SwordNestsEvent : c_RaidTables.ShieldNestsEvent);
				return Array.Find(raidTables, table => table.TableID == NestLocations.EventHash).Entries;
			}
			// 本編レイド
			else if(raidIndex < NestLocations.Nests.Length)
			{
				var detail = NestLocations.Nests[raidIndex];
				RaidTemplateTable[] raidTables = (version == 0 ? c_RaidTables.SwordNests : c_RaidTables.ShieldNests);
				if (rarity == 0)
				{
					return Array.Find(raidTables, table => table.TableID == detail.CommonHash).Entries;
				}
				else
				{
					return Array.Find(raidTables, table => table.TableID == detail.RareHash).Entries;
				}
			}
			// DLCレイド
			else
			{
				return GetAdditionalRaidEntries(raidIndex, version, rarity);
			}
		}
		public int GetRaidMap(int raidIndex)
		{
			if (raidIndex >= NestLocations.Nests.Length)
			{
				return AdditionalNestIdTable.c_DLC1Table[raidIndex].MapIndex;
			}
			return 0;
		}
		public System.Drawing.Point GetRaidLocation(int raidIndex)
		{
			if (raidIndex >= 0 && raidIndex < NestLocations.Nests.Length)
			{
				var detail = NestLocations.Nests[raidIndex];
				return new System.Drawing.Point((int)(detail.MapX * c_MapScale), (int)(detail.MapY * c_MapScale));
			}
			else if (raidIndex >= NestLocations.Nests.Length)
			{
				return new System.Drawing.Point(AdditionalNestIdTable.c_DLC1Table[raidIndex].PosX, AdditionalNestIdTable.c_DLC1Table[raidIndex].PosY);
			}
			else
			{
				return System.Drawing.Point.Empty;
			}
		}
		RaidTemplate[] GetAdditionalRaidEntries(int raidIndex, int version, int rarity)
		{
			string id = (rarity == 0 ? AdditionalNestIdTable.c_DLC1Table[raidIndex].IdNormal : AdditionalNestIdTable.c_DLC1Table[raidIndex].IdRare);
			if (version == 0)
			{
				id += "_Sw";
			}
			else
			{
				id += "_Sh";
			}

			if (m_AdditionalDenList.EventList.ContainsKey(id))
			{
				return m_AdditionalDenList.EventList[id].RaidEntries;
			}
			else
			{
				return GetRaidEntries(-1, version, 0);
			}
		}

		// イベントレイド
		public RaidTemplate[] GetEventRaidEntries(string id, int version)
		{
			if (version == 0)
			{
				id += "_Sw";
			}
			else
			{
				id += "_Sh";
			}

			if (m_EventDenList.EventList.ContainsKey(id))
			{
				return m_EventDenList.EventList[id].RaidEntries;
			}
			else
			{
				return GetRaidEntries(-1, version, 0);
			}
		}

        public List<string> GetEventRaidIdList()
        {
            List<string> list = new List<string>();
			HashSet<string> existId = new HashSet<string>();
            foreach (string key in m_EventDenList.EventList.Keys)
            {
				// 末尾のバージョンを削除
				string id = key.Substring(0, key.Length - 3);
				if (!existId.Contains(id))
				{
					list.Add(id);
					existId.Add(id);
				}
			}
			list.Sort();
			list.Reverse();
            return list;
        }
		public void LoadEventRaidData()
		{
			m_EventDenList.Load("data/EventDen.json");
		}

        // 1ランク1ポケモンごとに対応するデータ
        public class Pokemon
		{
			public string Key { get; private set; }
			public int Rank { get; private set; }

			public decimal DisplaySpecies { get; private set; } // 名前表示上の種族
			public int DataSpecies { get; private set; } // PKHeXデータ上のインデックス
			public bool IsDataSWSH { get; private set; }

			public int FlawlessIvs { get; private set; }
			public bool IsGigantamax { get; private set; }
			public int Ability { get; private set; }
			public bool IsFixedGender { get; private set; }

			public override string ToString() { return Key; }

			// データ取得
			public int NatureTableId {
				get {
					// ストリンダー対応
					if (DisplaySpecies == 849)
					{
						return 1; // ハイ
					}
					if (DisplaySpecies == 1154)
					{
						return 2; // ロー
					}
					return 0;
				}
			}

			public Pokemon(RaidTemplate entry, int rank)
			{
				Rank = rank;

				int rawSpecies = entry.Species;
				decimal altForm = entry.AltForm;

				DisplaySpecies = rawSpecies;
				IsDataSWSH = true;

				// マホイップ、カラナクシ、トリトドンは無視
				if (rawSpecies == 869 || rawSpecies == 422 || rawSpecies == 423)
				{
					altForm = 0;
				}
				// FCロトムは全て1に
				else if (rawSpecies == 479)
				{
					if (altForm != 0)
					{
						altForm = 1;
					}
				}

				// ストリンダー（ロー）
				if (rawSpecies == 849 && altForm == 1)
				{
					DisplaySpecies = 1154;
				}
				else
				{
					DisplaySpecies = rawSpecies + altForm / 10m;
				}

				// PKHeX上のindexを取得
				if (altForm != 0)
				{
					DataSpecies = PersonalTable.SWSH[rawSpecies].FormeIndex(rawSpecies, entry.AltForm);
					// 存在しないものは以前のデータを使う
					if (DataSpecies == rawSpecies)
					{
						PersonalInfo tempInfo = PersonalTable.SWSH[DataSpecies];
						if (tempInfo.ATK == 0)
						{
							DataSpecies = PersonalTable.USUM[rawSpecies].FormeIndex(rawSpecies, entry.AltForm);
							IsDataSWSH = false;
						}
					}
				}
				else
				{
					DataSpecies = rawSpecies;
				}				

				FlawlessIvs = entry.FlawlessIVs;
				IsGigantamax = entry.IsGigantamax;

				// 特性
				Ability = entry.Ability;

				PersonalInfo info = PersonalTable.SWSH[DataSpecies];
				if (info.ATK == 0)
				{
					info = PersonalTable.USUM[DataSpecies];
				}

				// レイドデータで固定されているのはイエッサン、ニャオニクス、エンニュートのみ
				// ♂のみ：バルキー系、エルレイド、ナゲキ、ダゲキ、ウォーグル系、オーロンゲ系
				// ♀のみ：ビークイン、ユキメノコ、バルジーナ系、アマージョ系、ブリムオン系、マホイップ系
				if(info.Genderless || info.OnlyFemale || info.OnlyMale)
				{
					IsFixedGender = true;
				}
				else
				{
					IsFixedGender = (entry.Gender != 0);
				}

				RefreshKey();
			}

			public void RefreshKey()
			{
				// キーを作成
				string key = Messages.Instance.RankPrefix[Rank];
				bool isExist = false;
				foreach (var pokemon in Messages.Instance.Pokemon)
				{
					if (pokemon.Value == DisplaySpecies)
					{
						key += pokemon.Key;
						isExist = true;
						break;
					}
				}
				// 存在しない場合はPKHeXから取ってくる
				if (!isExist)
				{
					var list = PKHeX.Core.Util.GetSpeciesList(Messages.Instance.LangCode);
					int listId = (int)DisplaySpecies;
					if (listId < list.Length)
					{
						key += list[listId];
					}
				}

				if (IsGigantamax)
				{
					key += Messages.Instance.SystemLabel["Gigantamax"];
				}

				// V固定数のkeyを付与
				key += $" [{Messages.Instance.IvsInfo[FlawlessIvs]} / ";

				// 特性のkeyを付与
				if (Ability == 2)
				{
					key += $"{Messages.Instance.SystemLabel["HiddenFixed"]}]";
				}
				else if (Ability == 3)
				{
					key += $"{Messages.Instance.SystemLabel["NoHidden"]}]";
				}
				else if (Ability == 4)
				{
					key += $"{Messages.Instance.SystemLabel["HiddenPossible"]}]";
				}
				else
				{
					key += $"]";
				}

				Key = key;
			}
		}
	}
}
