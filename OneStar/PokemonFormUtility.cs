using System.Collections.Generic;
using System.Windows.Forms;
using OneStarCalculator;

namespace OneStar
{
	class PokemonFormUtility
	{
		// 逆引きリスト
		public static Dictionary<string, int> m_NatureDictionary;
		public static Dictionary<string, int> m_CharacteristicDictionary;

		static PokemonFormUtility()
		{
			m_NatureDictionary = new Dictionary<string, int>();
			for (int i = 0; i < 25; ++i)
			{
				m_NatureDictionary.Add(Util.GetNatureString(i), i);
			}
			m_CharacteristicDictionary = new Dictionary<string, int>();
			m_CharacteristicDictionary.Add("ひるねをよくする", 0);
			m_CharacteristicDictionary.Add("あばれることがすき", 1);
			m_CharacteristicDictionary.Add("うたれづよい", 2);
			m_CharacteristicDictionary.Add("ものおとにびんかん", 3);
			m_CharacteristicDictionary.Add("イタズラがすき", 4);
			m_CharacteristicDictionary.Add("ちょっぴりみえっぱり", 5);
		}

		// 性格リストをコンボボックスにセット
		public static void SetNatureComboBox(ComboBox comboBox)
		{
			foreach (var key in m_NatureDictionary.Keys)
			{
				comboBox.Items.Add(key);
			}
			comboBox.SelectedIndex = 0;
		}

		// 個性リストをコンボボックスにセット
		public static void SetCharacteristicComboBox(ComboBox comboBox)
		{
			foreach (var key in m_CharacteristicDictionary.Keys)
			{
				comboBox.Items.Add(key);
			}
			comboBox.SelectedIndex = 0;
		}

		// 特性リストをコンボボックスにセット
		public static void SetAbilityComboBox(ComboBox comboBox, int initialIndex = 0)
		{
			comboBox.Items.Add("特性1");
			comboBox.Items.Add("特性2");
			comboBox.SelectedIndex = initialIndex;
		}
}
}
