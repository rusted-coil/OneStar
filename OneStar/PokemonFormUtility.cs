using System.Collections.Generic;
using System.Windows.Forms;
using OneStarCalculator;

namespace OneStar
{
	class PokemonFormUtility
	{
		// 性格リストをコンボボックスにセット
		public static void SetNatureComboBox(ComboBox comboBox)
		{
			foreach (var key in Messages.Instance.Nature.Keys)
			{
				comboBox.Items.Add(key);
			}
			comboBox.SelectedIndex = 0;
		}

		// 性格IDのものを選択
		public static void SelectNatureComboBox(ComboBox comboBox, int nature)
		{
			string str = GetNatureString(nature);
			foreach (var item in comboBox.Items)
			{
				if (item.ToString() == str)
				{
					comboBox.SelectedItem = item;
				}
			}
		}

		// 個性リストをコンボボックスにセット
		public static void SetCharacteristicComboBox(ComboBox comboBox)
		{
			foreach (var key in Messages.Instance.Characteristic.Keys)
			{
				comboBox.Items.Add(key);
			}
			comboBox.SelectedIndex = 0;
		}

		// 個性IDのものを選択
		public static void SelectCharacteristicComboBox(ComboBox comboBox, int characteristic)
		{
			string str = GetCharacteristicString(characteristic);
			foreach (var item in comboBox.Items)
			{
				if (item.ToString() == str)
				{
					comboBox.SelectedItem = item;
				}
			}
		}

		// 特性リストをコンボボックスにセット
		public static void SetAbilityComboBox(ComboBox comboBox, int count, int initialIndex = 0)
		{
			for (int i = 0; i < count; ++i)
			{
				comboBox.Items.Add(Messages.Instance.Ability[i]);
			}
			comboBox.SelectedIndex = initialIndex;
		}

		public static string GetNatureString(int nature)
		{
			foreach (var pair in Messages.Instance.Nature)
			{
				if (pair.Value == nature)
				{
					return pair.Key;
				}
			}
			return "";
		}

		public static string GetCharacteristicString(int characteristic)
		{
			foreach (var pair in Messages.Instance.Characteristic)
			{
				if (pair.Value == characteristic)
				{
					return pair.Key;
				}
			}
			return "";
		}
	}
}
