using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;

namespace OneStar
{
	public class Preferences
	{
		public Language Language { get; set; }

		// 初期設定
		public void Initialize()
		{
			Language = Language.Japanese;
		}

		// コピー
		void Copy(Preferences src)
		{
			Language = src.Language;
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
