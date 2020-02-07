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

		// 性格テーブル
		public class NatureTable
		{
			public uint Max { get; set; }
			public uint Pattern { get; set; }
			public int[] List { get; set; }
		}
		public static NatureTable[] NatureTableList = {
			new NatureTable{ Max = 0x1F, Pattern = 25, List = new int[]{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 ,17, 18, 19, 20, 21, 22, 23, 24 } },
			new NatureTable{ Max = 0xF, Pattern = 13, List = new int[]{ 3, 4, 2, 8, 9, 19, 22, 11, 13, 14, 0, 6, 24 } },
			new NatureTable{ Max = 0xF, Pattern = 12, List = new int[]{ 1, 5, 7, 10, 12, 15, 16, 17, 18, 20, 21, 23 } },
		};
	}
}
