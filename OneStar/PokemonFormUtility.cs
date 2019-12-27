using System.Windows.Forms;
using OneStarCalculator;

namespace OneStar
{
	class PokemonFormUtility
	{
		// 性格リストをコンボボックスにセット
		public static void SetNatureComboBox(ComboBox comboBox, int initialIndex = 0)
		{
			for (int i = 0; i < 25; ++i)
			{
				comboBox.Items.Add(Util.GetNatureString(i));
			}
			comboBox.SelectedIndex = initialIndex;
		}

		// 個性リストをコンボボックスにセット
		public static void SetCharacteristicComboBox(ComboBox comboBox, int initialIndex = 0)
		{
			comboBox.Items.Add("ひるねをよくする");
			comboBox.Items.Add("あばれることがすき");
			comboBox.Items.Add("うたれづよい");
			comboBox.Items.Add("ものおとにびんかん");
			comboBox.Items.Add("イタズラがすき");
			comboBox.Items.Add("ちょっぴりみえっぱり");
			comboBox.SelectedIndex = initialIndex;
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
