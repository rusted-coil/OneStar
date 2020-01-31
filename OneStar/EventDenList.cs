using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using PKHeX_Raid_Plugin;

namespace OneStar
{
	public class EventDenList
	{
		public class EventDen
		{
			public class Data
			{
				public int Species;
				public int[] Probabilities;
				public int FlawlessIvs;
				public int MinRank;
				public int MaxRank;
				public int AltForm;
				public int Ability;
				public int Gender;
				public bool Gigantamax;
				public int ShinyType;

				public Data()
				{
					Probabilities = new int[5];
				}
			}

			public List<Data> Entries;

			public EventDen()
			{
				Entries = new List<Data>();
				Entries.Add(new Data());
				Entries.Add(new Data());
				Entries.Add(new Data());
			}

			// RaidPlugin互換のデータを取得
			[JsonIgnore]
			public RaidTemplate[] RaidEntries {
				get {
					RaidTemplate[] list = new RaidTemplate[Entries.Count];
					for (int i = 0; i < list.Length; ++i)
					{
						list[i] = new RaidTemplate( Entries[i].Species,
													Entries[i].Probabilities,
													Entries[i].FlawlessIvs,
													Entries[i].MinRank,
													Entries[i].MaxRank,
													Entries[i].AltForm,
													Entries[i].Ability,
													Entries[i].Gender,
													Entries[i].Gigantamax,
													Entries[i].ShinyType );
					}
					return list;
				}
			}
		}

		[JsonIgnore]
		public bool IsValid = false;
		public Dictionary<string, EventDen> EventList;

		public void Load()
		{
			string str;
			try
			{
				using (StreamReader sr = new StreamReader("EventDen.json"))
				{
					str = sr.ReadToEnd();
				}
			}
			catch (Exception)
			{
				// エラー
				return;
			}

			EventDenList tmp = null;
			try
			{
				tmp = JsonConvert.DeserializeObject<EventDenList>(str);
			}
			catch (Exception)
			{
				// エラー
				return;
			}

			if (tmp != null)
			{
				EventList = new Dictionary<string, EventDen>(tmp.EventList);

				IsValid = true;
			}
		}

		// でばっぐよう出力
		public void GenerateTemplate()
		{
			EventList = new Dictionary<string, EventDen>();
			EventList.Add("20200109", new EventDen());
			EventList.Add("20200131", new EventDen());

			using (StreamWriter sw = new StreamWriter("test.json"))
			{
				string str = JsonConvert.SerializeObject(this);
				sw.Write(str);
			}
		}
	}
}
