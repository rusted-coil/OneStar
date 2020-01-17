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

		// 初期設定
		public void Initialize()
		{
			m_Language = Language.Japanese;
			m_GameVersion = 0;
		}

		// コピー
		void Copy(Preferences src)
		{
			m_Language = src.m_Language;
			m_GameVersion = src.m_GameVersion;
		}

		// 設定をファイルに保存
		public void Serialize()
		{
			try
			{
				using (StreamWriter sw = new StreamWriter("config.json"))
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
				using (StreamReader sr = new StreamReader("config.json"))
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
