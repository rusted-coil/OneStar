using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;

namespace OneStar
{
	public class Messages
	{
		// データ
		public Dictionary<string, int> Nature;
		public Dictionary<string, int> Characteristic;
		public List<string> Ability;
		public List<string> Status;
		public Dictionary<string, string> SystemLabel;
		public List<string> CheckIvsResult;
		public Dictionary<string, string> ErrorMessage;
		public Dictionary<string, string> SystemMessage;
		public Dictionary<string, string> ListLabel;

		[JsonIgnore] public static Messages Instance { get; private set; }
		[JsonIgnore] public static string ErrorText { get; private set; }

		// 言語を指定して初期化
		public static bool Initialize(Language language)
		{
			string fileName = "";
			string str = "";
			switch (language)
			{
				case Language.Japanese: fileName = "LanguageJp.json"; break;
				case Language.English: fileName = "LanguageEn.json"; break;
				case Language.ChineseZh: fileName = "LanguageZh.json"; break;
			}
			try
			{
				using (StreamReader sr = new StreamReader(fileName))
				{
					str = sr.ReadToEnd();
				}
			}
			catch (Exception e)
			{
				// エラー
				ErrorText = e.ToString();
				Instance = null;
				return false;
			}

			try
			{
				Instance = JsonConvert.DeserializeObject<Messages>(str);
			}
			catch (Exception e)
			{
				// エラー
				ErrorText = e.ToString();
				Instance = null;
				return false;
			}

			return Instance != null;
		}
	}
}
