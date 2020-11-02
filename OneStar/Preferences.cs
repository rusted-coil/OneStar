using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;

namespace OneStar
{
	public class Preferences
	{
		// 言語設定
		[JsonProperty] Language m_Language = Language.Japanese;

		// バージョン
		[JsonProperty] int m_GameVersion = 0;

		// イベントレイド
		[JsonProperty] string m_EventId = "20201102";

		// GPU使用
		[JsonProperty] bool m_isUseGpu = false;

		// GPU設定
		[JsonProperty] int m_GpuLoop = 0;

		// 検索設定
		[JsonProperty] bool m_SearchShowDuration = false;
		[JsonProperty] bool m_SearchStop = true;

		// リスト設定
		[JsonProperty] int m_ListMaxFrame = 5000;
		[JsonProperty] bool m_ListOnlyShiny = false;
		[JsonProperty] bool m_ListShowSeed = false;
		[JsonProperty] bool m_ListShowEC = false;
		[JsonProperty] bool m_ListShowAbilityName = false;

        #region 取得
        [JsonIgnore]
		public Language Language {
			get { return m_Language; }
			set {
				m_Language = value;
				Serialize();
			}
		}
		[JsonIgnore]
		public int GameVersion {
			get { return m_GameVersion; }
			set {
				m_GameVersion = value;
				Serialize();
			}
		}
		[JsonIgnore]
		public string EventId {
			get { return m_EventId; }
			set {
				m_EventId = value;
				Serialize();
			}
		}
		[JsonIgnore]
		public bool IsUseGpu { 
			get { return m_isUseGpu; }
			set {
				m_isUseGpu = value;
				Serialize();
			}
		}
		[JsonIgnore]
		public int GpuLoop {
			get { return m_GpuLoop; }
			set {
				m_GpuLoop = value;
				Serialize();
			}
		}
		[JsonIgnore]
		public bool SearchShowDuration
		{
			get { return m_SearchShowDuration; }
			set
			{
				m_SearchShowDuration = value;
				Serialize();
			}
		}
		[JsonIgnore]
		public bool SearchStop
		{
			get { return m_SearchStop; }
			set
			{
				m_SearchStop = value;
				Serialize();
			}
		}
		[JsonIgnore]
		public int ListMaxFrame { 
			get { return m_ListMaxFrame; }
			set {
				m_ListMaxFrame = value;
				Serialize();
			}
		}
		[JsonIgnore]
		public bool ListOnlyShiny
		{
			get { return m_ListOnlyShiny; }
			set
			{
				m_ListOnlyShiny = value;
				Serialize();
			}
		}
		[JsonIgnore]
		public bool ListShowSeed
		{
			get { return m_ListShowSeed; }
			set
			{
				m_ListShowSeed = value;
				Serialize();
			}
		}
		[JsonIgnore]
		public bool ListShowEC
		{
			get { return m_ListShowEC; }
			set
			{
				m_ListShowEC = value;
				Serialize();
			}
		}
		[JsonIgnore]
		public bool ListShowAbilityName
		{
			get { return m_ListShowAbilityName; }
			set
			{
				m_ListShowAbilityName = value;
				Serialize();
			}
		}
        #endregion

        // 初期設定
        public void Initialize()
		{
			m_Language = Language.Japanese;
			m_GameVersion = 0;
			m_EventId = "20201102";
			m_isUseGpu = false;
			m_GpuLoop = 0;
			m_SearchShowDuration = false;
			m_SearchStop = true;
			m_ListMaxFrame = 5000;
			m_ListOnlyShiny = false;
			m_ListShowSeed = false;
			m_ListShowEC = false;
			m_ListShowAbilityName = false;
		}

		// コピー
		void Copy(Preferences src)
		{
			m_Language = src.m_Language;
			m_GameVersion = src.m_GameVersion;
			m_EventId = src.m_EventId;
			m_isUseGpu = src.m_isUseGpu;
			m_GpuLoop = src.m_GpuLoop;
			m_SearchShowDuration = src.m_SearchShowDuration;
			m_SearchStop = src.m_SearchStop;
			m_ListMaxFrame = src.m_ListMaxFrame;
			m_ListOnlyShiny = src.m_ListOnlyShiny;
			m_ListShowSeed = src.m_ListShowSeed;
			m_ListShowEC = src.m_ListShowEC;
			m_ListShowAbilityName = src.m_ListShowAbilityName;
		}

		// 設定をファイルに保存
		public void Serialize()
		{
			try
			{
				using (StreamWriter sw = new StreamWriter("data/config.json"))
				{
					string str = JsonConvert.SerializeObject(this);
					sw.Write(str);
				}
			}
			catch (Exception)
			{
			}
		}

		// ファイルから設定を読み込み
		public bool Deserialize()
		{
			string str;
			try
			{
				using (StreamReader sr = new StreamReader("data/config.json"))
				{
					str = sr.ReadToEnd();
				}
			}
			catch (Exception)
			{
				// エラー
				return false;
			}

			Preferences tmp = null;
			try
			{
				tmp = JsonConvert.DeserializeObject<Preferences>(str);
			}
			catch (Exception)
			{
				// エラー
				return false;
			}

			if (tmp != null)
			{
				Copy(tmp);

				return true;
			}
			return false;
		}
	}
}
