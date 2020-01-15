using System;
using OneStarCalculator;
using System.Windows.Forms;
using System.Threading.Tasks;
using System.IO;
using System.Collections.Generic;
using IVCalcNetFramework;
using System.Linq;
using PKHeX_Raid_Plugin;

namespace OneStar
{
	// 設定項目
	public enum Language
	{
		Japanese,
		English,
		ChineseZh,
		ChineseZh_TW
	}

	public enum AbilityType
	{
		First,
		Second,
		Num
	}

	public partial class MainForm : Form
	{
		public bool IsInitialized { get; private set; }

		// 環境設定
		Preferences m_Preferences;

		// 巣穴データ
		readonly RaidData c_RaidData = new RaidData();

		// ポケモン入力フォームごとのセット
		class PokemonInfoForm {
			public ComboBox ComboBoxName { get; set; } = null;
			public TextBox TextBoxLevel { get; set; } = null;
			public TextBox[] TextBoxIvs { get; set; } = new TextBox[6];
			public TextBox[] TextBoxStatus { get; set; } = new TextBox[6];
			public ComboBox ComboBoxNature { get; set; } = null;
			public ComboBox ComboBoxCharacteristic { get; set; } = null;
		};
		PokemonInfoForm[] m_PokemonInfo = new PokemonInfoForm[5];

//		TextBox[] m_TextBoxIvsList12 = new TextBox[12];
//		TextBox[] m_TextBoxIvsList35 = new TextBox[18];
//		TextBox[] m_TextBoxStatusList35 = new TextBox[18];

		// 言語設定可能コントロール
		Dictionary<string, Control[]> m_MultiLanguageControls;

		enum Star35PanelMode {
			From2V,
			From3V
		};

		Star35PanelMode Get35Mode() { return (Star35PanelMode)f_ComboBoxModeSelector_35.SelectedIndex; }

		public MainForm()
		{
			// 設定の読み込み
			m_Preferences = new Preferences();
			if (!m_Preferences.Deserialize())
			{
				m_Preferences.Initialize();
			}

			// 言語の初期化
			if (!Messages.Initialize(m_Preferences.Language))
			{
				MessageBox.Show("言語ファイルの読み込みに失敗しました。\n----------\n" + Messages.ErrorText, "エラー", MessageBoxButtons.OK, MessageBoxIcon.Error);
				IsInitialized = false;
				return;
			}

			InitializeComponent();

			// ビューの初期化
			InitializeView();

            //initialize Pokemon comboboxes, and IV Calculator
            populateComboBoxes();

            IsInitialized = true;

			//			RaidTemplateTable toUse = 
			// ポケモンデータチェック
			for (int i = -1; i <= 99; ++i)
			{
				if (i == 16)
				{
					continue;
				}

				for (int a = 0; a < 2; ++a)
				{
					for (int b = 0; b < 2; ++b)
					{
						RaidTemplateTable raidTable = c_RaidData.GetRaidTemplateTable(i, a, b);
						foreach (var entry in raidTable.Entries)
						{
							for (int c = 0; c < 5; ++c)
							{
								if (entry.Probabilities[c] > 0)
								{
									int species = entry.Species;
									bool isExist = false;
									foreach (var pokemon in Messages.Instance.Pokemon)
									{
										if (pokemon.Value == species)
										{
											isExist = true;
											break;
										}
									}
									if (isExist == false)
									{
										;
									}
								}
							}
						}
					}
				}
			}
		}

		void InitializeComboBox()
		{
			// 巣穴
			foreach (var key in Messages.Instance.Den.Keys)
			{
				f_ComboBoxDenName.Items.Add(key);
			}
			f_ComboBoxDenName.SelectedIndex = 0;
			foreach (var version in Messages.Instance.Version)
			{
				f_ComboBoxGameVersion.Items.Add(version);
			}
			f_ComboBoxGameVersion.SelectedIndex = 0;
			foreach (var rarity in Messages.Instance.DenRarity)
			{
				f_ComboBoxRarity.Items.Add(rarity);
			}
			f_ComboBoxRarity.SelectedIndex = 0;

			// モード
			f_ComboBoxModeSelector_35.Items.Clear();
			f_ComboBoxModeSelector_35.Items.Add(Messages.Instance.SystemLabel["Pokemon35_1_2V"]);
			f_ComboBoxModeSelector_35.Items.Add(Messages.Instance.SystemLabel["Pokemon35_1_3V"]);

			// ★3～5パネル
			PokemonFormUtility.SetNatureComboBox(f_ComboBoxNature_351);
			PokemonFormUtility.SetNatureComboBox(f_ComboBoxNature_352);
			PokemonFormUtility.SetNatureComboBox(f_ComboBoxNature_353);
			PokemonFormUtility.SetCharacteristicComboBox(f_ComboBoxCharacteristic_351);
			PokemonFormUtility.SetCharacteristicComboBox(f_ComboBoxCharacteristic_352);
			PokemonFormUtility.SetCharacteristicComboBox(f_ComboBoxCharacteristic_353);
			PokemonFormUtility.SetAbilityComboBox(f_ComboBoxAbility_351, 4);
			PokemonFormUtility.SetAbilityComboBox(f_ComboBoxAbility_352, 4);
			PokemonFormUtility.SetAbilityComboBox(f_ComboBoxAbility_353, 4);

			// ★1～2パネル
			PokemonFormUtility.SetNatureComboBox(f_ComboBoxNature_1);
			PokemonFormUtility.SetNatureComboBox(f_ComboBoxNature_2);
			PokemonFormUtility.SetCharacteristicComboBox(f_ComboBoxCharacteristic_1);
			PokemonFormUtility.SetCharacteristicComboBox(f_ComboBoxCharacteristic_2);
			PokemonFormUtility.SetAbilityComboBox(f_ComboBoxAbility_1, 4);
			PokemonFormUtility.SetAbilityComboBox(f_ComboBoxAbility_2, 4);
		}

		void InitializeView()
		{
			// ★3～5パネル
			SetCheckResult(-1);

			// 共通
			f_TextBoxMaxFrame.Text = "5000";

			f_TextBoxRerollsLower.Text = "0";
			f_TextBoxRerollsUpper.Text = "3";
			f_CheckBoxStop.Checked = true;
			f_TextBoxListVCount.Text = "4";

			// 扱いやすいようにキャッシュ
			m_PokemonInfo[0] = new PokemonInfoForm();
			m_PokemonInfo[0].ComboBoxName = f_ComboBoxPokemon_1;
			m_PokemonInfo[0].TextBoxLevel = f_TextBoxLevel_1;
			m_PokemonInfo[0].TextBoxIvs[0] = f_TextBoxIv0_1;
			m_PokemonInfo[0].TextBoxIvs[1] = f_TextBoxIv1_1;
			m_PokemonInfo[0].TextBoxIvs[2] = f_TextBoxIv2_1;
			m_PokemonInfo[0].TextBoxIvs[3] = f_TextBoxIv3_1;
			m_PokemonInfo[0].TextBoxIvs[4] = f_TextBoxIv4_1;
			m_PokemonInfo[0].TextBoxIvs[5] = f_TextBoxIv5_1;
			m_PokemonInfo[0].TextBoxStatus[0] = f_TextBoxStatus0_1;
			m_PokemonInfo[0].TextBoxStatus[1] = f_TextBoxStatus1_1;
			m_PokemonInfo[0].TextBoxStatus[2] = f_TextBoxStatus2_1;
			m_PokemonInfo[0].TextBoxStatus[3] = f_TextBoxStatus3_1;
			m_PokemonInfo[0].TextBoxStatus[4] = f_TextBoxStatus4_1;
			m_PokemonInfo[0].TextBoxStatus[5] = f_TextBoxStatus5_1;
			m_PokemonInfo[0].ComboBoxNature = f_ComboBoxNature_1;
			m_PokemonInfo[0].ComboBoxCharacteristic = f_ComboBoxCharacteristic_1;
			m_PokemonInfo[1] = new PokemonInfoForm();
			m_PokemonInfo[1].ComboBoxName = f_ComboBoxPokemon_2;
			m_PokemonInfo[1].TextBoxLevel = f_TextBoxLevel_2;
			m_PokemonInfo[1].TextBoxIvs[0] = f_TextBoxIv0_2;
			m_PokemonInfo[1].TextBoxIvs[1] = f_TextBoxIv1_2;
			m_PokemonInfo[1].TextBoxIvs[2] = f_TextBoxIv2_2;
			m_PokemonInfo[1].TextBoxIvs[3] = f_TextBoxIv3_2;
			m_PokemonInfo[1].TextBoxIvs[4] = f_TextBoxIv4_2;
			m_PokemonInfo[1].TextBoxIvs[5] = f_TextBoxIv5_2;
			m_PokemonInfo[1].TextBoxStatus[0] = f_TextBoxStatus0_2;
			m_PokemonInfo[1].TextBoxStatus[1] = f_TextBoxStatus1_2;
			m_PokemonInfo[1].TextBoxStatus[2] = f_TextBoxStatus2_2;
			m_PokemonInfo[1].TextBoxStatus[3] = f_TextBoxStatus3_2;
			m_PokemonInfo[1].TextBoxStatus[4] = f_TextBoxStatus4_2;
			m_PokemonInfo[1].TextBoxStatus[5] = f_TextBoxStatus5_2;
			m_PokemonInfo[1].ComboBoxNature = f_ComboBoxNature_2;
			m_PokemonInfo[1].ComboBoxCharacteristic = f_ComboBoxCharacteristic_2;
			m_PokemonInfo[2] = new PokemonInfoForm();
			m_PokemonInfo[2].ComboBoxName = f_ComboBoxPokemon_351;
			m_PokemonInfo[2].TextBoxLevel = f_TextBoxLevel_351;
			m_PokemonInfo[2].TextBoxIvs[0] = f_TextBoxIv0_351;
			m_PokemonInfo[2].TextBoxIvs[1] = f_TextBoxIv1_351;
			m_PokemonInfo[2].TextBoxIvs[2] = f_TextBoxIv2_351;
			m_PokemonInfo[2].TextBoxIvs[3] = f_TextBoxIv3_351;
			m_PokemonInfo[2].TextBoxIvs[4] = f_TextBoxIv4_351;
			m_PokemonInfo[2].TextBoxIvs[5] = f_TextBoxIv5_351;
			m_PokemonInfo[2].TextBoxStatus[0] = f_TextBoxStatus0_351;
			m_PokemonInfo[2].TextBoxStatus[1] = f_TextBoxStatus1_351;
			m_PokemonInfo[2].TextBoxStatus[2] = f_TextBoxStatus2_351;
			m_PokemonInfo[2].TextBoxStatus[3] = f_TextBoxStatus3_351;
			m_PokemonInfo[2].TextBoxStatus[4] = f_TextBoxStatus4_351;
			m_PokemonInfo[2].TextBoxStatus[5] = f_TextBoxStatus5_351;
			m_PokemonInfo[2].ComboBoxNature = f_ComboBoxNature_351;
			m_PokemonInfo[2].ComboBoxCharacteristic = f_ComboBoxCharacteristic_351;
			m_PokemonInfo[3] = new PokemonInfoForm();
			m_PokemonInfo[3].ComboBoxName = f_ComboBoxPokemon_352;
			m_PokemonInfo[3].TextBoxLevel = f_TextBoxLevel_352;
			m_PokemonInfo[3].TextBoxIvs[0] = f_TextBoxIv0_352;
			m_PokemonInfo[3].TextBoxIvs[1] = f_TextBoxIv1_352;
			m_PokemonInfo[3].TextBoxIvs[2] = f_TextBoxIv2_352;
			m_PokemonInfo[3].TextBoxIvs[3] = f_TextBoxIv3_352;
			m_PokemonInfo[3].TextBoxIvs[4] = f_TextBoxIv4_352;
			m_PokemonInfo[3].TextBoxIvs[5] = f_TextBoxIv5_352;
			m_PokemonInfo[3].TextBoxStatus[0] = f_TextBoxStatus0_352;
			m_PokemonInfo[3].TextBoxStatus[1] = f_TextBoxStatus1_352;
			m_PokemonInfo[3].TextBoxStatus[2] = f_TextBoxStatus2_352;
			m_PokemonInfo[3].TextBoxStatus[3] = f_TextBoxStatus3_352;
			m_PokemonInfo[3].TextBoxStatus[4] = f_TextBoxStatus4_352;
			m_PokemonInfo[3].TextBoxStatus[5] = f_TextBoxStatus5_352;
			m_PokemonInfo[3].ComboBoxNature = f_ComboBoxNature_352;
			m_PokemonInfo[3].ComboBoxCharacteristic = f_ComboBoxCharacteristic_352;
			m_PokemonInfo[4] = new PokemonInfoForm();
			m_PokemonInfo[4].ComboBoxName = f_ComboBoxPokemon_353;
			m_PokemonInfo[4].TextBoxLevel = f_TextBoxLevel_353;
			m_PokemonInfo[4].TextBoxIvs[0] = f_TextBoxIv0_353;
			m_PokemonInfo[4].TextBoxIvs[1] = f_TextBoxIv1_353;
			m_PokemonInfo[4].TextBoxIvs[2] = f_TextBoxIv2_353;
			m_PokemonInfo[4].TextBoxIvs[3] = f_TextBoxIv3_353;
			m_PokemonInfo[4].TextBoxIvs[4] = f_TextBoxIv4_353;
			m_PokemonInfo[4].TextBoxIvs[5] = f_TextBoxIv5_353;
			m_PokemonInfo[4].TextBoxStatus[0] = f_TextBoxStatus0_353;
			m_PokemonInfo[4].TextBoxStatus[1] = f_TextBoxStatus1_353;
			m_PokemonInfo[4].TextBoxStatus[2] = f_TextBoxStatus2_353;
			m_PokemonInfo[4].TextBoxStatus[3] = f_TextBoxStatus3_353;
			m_PokemonInfo[4].TextBoxStatus[4] = f_TextBoxStatus4_353;
			m_PokemonInfo[4].TextBoxStatus[5] = f_TextBoxStatus5_353;
			m_PokemonInfo[4].ComboBoxNature = f_ComboBoxNature_353;
			m_PokemonInfo[4].ComboBoxCharacteristic = f_ComboBoxCharacteristic_353;
			// コールバックをセット
			for (int a = 0; a < 5; ++a)
			{
				for (int b = 0; b < 6; ++b)
				{
					// ラムダ式へのキャプチャのためローカル変数にコピー
					var tb1 = m_PokemonInfo[a].TextBoxIvs[b];
					var tb2 = m_PokemonInfo[a].TextBoxStatus[b];
					m_PokemonInfo[a].TextBoxIvs[b].Enter += new System.EventHandler((object sender, EventArgs e) => { EnterTextBoxAllSelect(tb1); });
					m_PokemonInfo[a].TextBoxStatus[b].Enter += new System.EventHandler((object sender, EventArgs e) => { EnterTextBoxAllSelect(tb2); });
				}
				var tb3 = m_PokemonInfo[a].TextBoxLevel;
				m_PokemonInfo[a].TextBoxLevel.Enter += new System.EventHandler((object sender, EventArgs e) => { EnterTextBoxAllSelect(tb3); });
			}

			// 言語設定用コントロールをセット
			m_MultiLanguageControls = new Dictionary<string, Control[]>();
			m_MultiLanguageControls["Tab35"] = new Control[] { f_TabPage1 };
			m_MultiLanguageControls["Tab12"] = new Control[] { f_TabPage2 };
			m_MultiLanguageControls["TabList"] = new Control[] { f_TabPage3 };
			m_MultiLanguageControls["Ivs"] = new Control[]{
				f_LabelIvs_1,
				f_LabelIvs_2,
				f_LabelIvs_351,
				f_LabelIvs_352,
				f_LabelIvs_353,
			};
			m_MultiLanguageControls["Nature"] = new Control[]{
				f_LabelNature_1,
				f_LabelNature_2,
				f_LabelNature_351,
				f_LabelNature_352,
				f_LabelNature_353,
			};
			m_MultiLanguageControls["HiddenPossible"] = new Control[] {
				f_CheckBoxDream_1,
				f_CheckBoxDream_2,
				f_CheckBoxDream_351,
				f_CheckBoxDream_352,
				f_CheckBoxDream_353,
				f_CheckBoxDream_List,
			};
			m_MultiLanguageControls["GenderFixed"] = new Control[] {
				f_CheckBoxNoGender_1,
				f_CheckBoxNoGender_2,
				f_CheckBoxNoGender_351,
				f_CheckBoxNoGender_352,
				f_CheckBoxNoGender_353,
				f_CheckBoxNoGender_List,
			};
			m_MultiLanguageControls["Pokemon35_1_2V"] = new Control[] {  };
			m_MultiLanguageControls["Pokemon35_1_3V"] = new Control[] { };
			m_MultiLanguageControls["Pokemon35_2"] = new Control[] { f_GroupBoxPokemon_352 };
			m_MultiLanguageControls["Pokemon35_3"] = new Control[] { f_GroupBoxPokemon_353 };
			m_MultiLanguageControls["Pokemon12_1"] = new Control[] { f_GroupBoxPokemon_1 };
			m_MultiLanguageControls["Pokemon12_2"] = new Control[] { f_GroupBoxPokemon_2 };
			m_MultiLanguageControls["CheckIvsButton"] = new Control[] { f_ButtonIvsCheck };
			m_MultiLanguageControls["CheckIvsResultTitle"] = new Control[] { f_LabelCheckResultTitle };
			m_MultiLanguageControls["ListVCount"] = new Control[] { f_LabelListVCount };
			m_MultiLanguageControls["MaxFrame"] = new Control[] { f_LabelMaxFrame };
			m_MultiLanguageControls["OnlyShiny"] = new Control[] { f_CheckBoxListShiny };
			m_MultiLanguageControls["ShowSeed"] = new Control[] { f_CheckBoxShowSeed };
			m_MultiLanguageControls["ListButton"] = new Control[] { f_ButtonListGenerate };
			m_MultiLanguageControls["ShowDuration"] = new Control[] { f_CheckBoxShowResultTime };
			m_MultiLanguageControls["StartSearch"] = new Control[] { f_ButtonStartSearch };
			m_MultiLanguageControls["Rerolls"] = new Control[] { f_LabelRerolls };
			m_MultiLanguageControls["Range"] = new Control[] { f_LabelRerollsRange };
			m_MultiLanguageControls["SearchStop"] = new Control[] { f_CheckBoxStop };

			// 言語を適用
			ChangeLanguage(true, m_Preferences.Language);
		}

		void CreateErrorDialog(string text)
		{
			MessageBox.Show(text, Messages.Instance.ErrorMessage["DialogTitle"], MessageBoxButtons.OK, MessageBoxIcon.Error);
		}

		void SetCheckResult(int result)
		{
			switch (result)
			{
				case -1:
					f_LabelCheckResult.Text = "-";
					f_LabelCheckResult.ForeColor = System.Drawing.Color.Black;
					break;

				case 1:
					f_LabelCheckResult.Text = Messages.Instance.CheckIvsResult[0];
					f_LabelCheckResult.ForeColor = System.Drawing.Color.Red;
					break;

				case 3:
					f_LabelCheckResult.Text = Messages.Instance.CheckIvsResult[1];
					f_LabelCheckResult.ForeColor = System.Drawing.Color.Blue;
					break;

				case 4:
					f_LabelCheckResult.Text = Messages.Instance.CheckIvsResult[2];
					f_LabelCheckResult.ForeColor = System.Drawing.Color.Blue;
					break;

				case 7:
					f_LabelCheckResult.Text = Messages.Instance.CheckIvsResult[3];
					f_LabelCheckResult.ForeColor = System.Drawing.Color.Blue;
					break;
			}
		}

		// 個体値チェックボタン
		private void f_ButtonIvsCheck_Click(object sender, EventArgs e)
		{
			bool isCheckFailed = false;
			string errorText = "";

			// フォームから必要な情報を取得
			int[] ivs = new int[6];
			for (int i = 0; i < 6; ++i)
			{
				try
				{
					ivs[i] = int.Parse(m_PokemonInfo[2].TextBoxIvs[i].Text);
				}
				catch (Exception)
				{
					// エラー
					errorText = Messages.Instance.ErrorMessage["IvsFormat"];
					isCheckFailed = true;
				}
				if (ivs[i] < 0 || ivs[i] > 31)
				{
					// エラー
					errorText = Messages.Instance.ErrorMessage["IvsFormat"];
					isCheckFailed = true;
				}
			}
			if (isCheckFailed)
			{
				CreateErrorDialog(errorText);
				return;
			}

			// 1匹目はVが2or3箇所じゃないとエラー
			int strict = (Get35Mode() == Star35PanelMode.From3V ? 3 : 2);
			int c = 0;
			for (int i = 0; i < 6; ++i)
			{
				if (ivs[i] == 31)
				{
					++c;
				}
			}
			if (c != strict)
			{
				// エラー
				if (strict == 3)
				{
					CreateErrorDialog(Messages.Instance.ErrorMessage["IvsStrict351_3V"]);
					return;
				}
				else
				{
					CreateErrorDialog(Messages.Instance.ErrorMessage["IvsStrict351_2V"]);
					return;
				}
			}

			// 遺伝箇所チェック
			bool[] possible = { false, false, false, false, false };
			bool[] vFlag = { false, false, false, false, false, false };
			int fixedCount = 0;
			for (int i = 0; i < 6; ++i)
			{
				if (ivs[i] == 31)
				{
					vFlag[i] = true;
					++fixedCount;
				}
			}

			// 2V+3Vor4Vで6個確定
			// 3V+4Vで5個確定
			int needNumber = (Get35Mode() == Star35PanelMode.From2V ? 6 : 5);

			for (int vCount = fixedCount + 1; vCount <= 4; ++vCount)
			{
				c = fixedCount;
				int used = fixedCount;
				for (int i = 0; i < 6; ++i)
				{
					vFlag[i] = (ivs[i] == 31);
				}
				// 普通に個体値生成をやってみる
				for (int i = 0; i < 6; ++i)
				{
					if (ivs[i] != 31)
					{
						++used;
						int vPos = ivs[i] % 8;
						if (vPos < 6 && vFlag[vPos] == false)
						{
							vFlag[vPos] = true;
							++c;
							if (c == vCount) // 遺伝終わり
							{
								// 未知の部分が連続してneedNumber以上
								if ((6 - vCount) - (6 - used) + (6 - fixedCount) >= needNumber)
								{
									possible[vCount] = true;
								}
								break;
							}
						}
					}
				}
			}

			if (possible[3] && possible[4])
			{
				SetCheckResult(7);
			}
			else if (possible[3])
			{
				SetCheckResult(3);
			}
			else if (possible[4])
			{
				SetCheckResult(4);
			}
			else
			{
				SetCheckResult(1);
			}
		}

		// 検索開始ボタン
		private void ButtonStartSearch_Click(object sender, EventArgs e)
		{
			// ★3～5
			if (f_TabControlMain.SelectedIndex == 0)
			{
				SeedSearch35();
			}
			// ★1～2
			else if (f_TabControlMain.SelectedIndex == 1)
			{
				SeedSearch12();
			}
			else
			{
				CreateErrorDialog(Messages.Instance.ErrorMessage["NotSearchTab"]);
			}
		}

		// ★1～2検索
		void SeedSearch12()
		{
			bool isCheckFailed = false;
			string errorText = "";

			// フォームから必要な情報を取得
			int[] ivs1 = new int[6];
			int[] ivs2 = new int[6];
			for (int i = 0; i < 6; ++i)
			{
				try
				{
					ivs1[i] = int.Parse(m_PokemonInfo[0].TextBoxIvs[i].Text);
					ivs2[i] = int.Parse(m_PokemonInfo[1].TextBoxIvs[i].Text);
				}
				catch (Exception)
				{
					// エラー
					errorText = Messages.Instance.ErrorMessage["IvsFormat"];
					isCheckFailed = true;
					break;
				}
				if (ivs1[i] < 0 || ivs1[i] > 31 || ivs2[i] < 0 || ivs2[i] > 31)
				{
					// エラー
					errorText = Messages.Instance.ErrorMessage["IvsFormat"];
					isCheckFailed = true;
					break;
				}
			}

			if (isCheckFailed)
			{
				CreateErrorDialog(errorText);
				return;
			}

			// 1匹目はVが1箇所じゃないとエラー
			int c = 0;
			for (int i = 0; i < 6; ++i)
			{
				if (ivs1[i] == 31)
				{
					++c;
				}
			}
			if (c != 1)
			{
				// エラー
				CreateErrorDialog(Messages.Instance.ErrorMessage["IvsStrict1"]);
				return;
			}
			// 2匹目はVが1箇所以上じゃないとエラー
			c = 0;
			for (int i = 0; i < 6; ++i)
			{
				if (ivs2[i] == 31)
				{
					++c;
				}
			}
			if (c < 1)
			{
				// エラー
				CreateErrorDialog(Messages.Instance.ErrorMessage["IvsStrict2"]);
				return;
			}
			int ability1 = f_ComboBoxAbility_1.SelectedIndex;
			int ability2 = f_ComboBoxAbility_2.SelectedIndex;
			if (ability1 >= 2) { ability1 = ability1 * 3 - 7; }
			if (ability2 >= 2) { ability2 = ability2 * 3 - 7; }
			int nature1 = Messages.Instance.Nature[m_PokemonInfo[0].ComboBoxNature.Text];
			int nature2 = Messages.Instance.Nature[m_PokemonInfo[1].ComboBoxNature.Text];

			bool noGender1 = f_CheckBoxNoGender_1.Checked;
			bool noGender2 = f_CheckBoxNoGender_2.Checked;

			bool isDream1 = f_CheckBoxDream_1.Checked;
			bool isDream2 = f_CheckBoxDream_2.Checked;

			// 計算開始
			SeedSearcher searcher = new SeedSearcher(SeedSearcher.Mode.Star12);
			SeedSearcher.SetFirstCondition(ivs1[0], ivs1[1], ivs1[2], ivs1[3], ivs1[4], ivs1[5], ability1, nature1, noGender1, isDream1);
			SeedSearcher.SetNextCondition(ivs2[0], ivs2[1], ivs2[2], ivs2[3], ivs2[4], ivs2[5], ability2, nature2, noGender2, isDream2);

			SearchImpl(searcher);
		}

		// ★3～5検索
		void SeedSearch35()
		{
			bool isCheckFailed = false;
			string errorText = "";

			// フォームから必要な情報を取得
			int[] ivs = new int[18];
			for (int i = 0; i < 18; ++i)
			{
				try
				{
					ivs[i] = int.Parse(m_PokemonInfo[2 + i / 6].TextBoxIvs[i % 6].Text);
				}
				catch (Exception)
				{
					// エラー
					errorText = Messages.Instance.ErrorMessage["IvsFormat"];
					isCheckFailed = true;
				}
				if (ivs[i] < 0 || ivs[i] > 31)
				{
					// エラー
					errorText = Messages.Instance.ErrorMessage["IvsFormat"];
					isCheckFailed = true;
				}
			}

			if (isCheckFailed)
			{
				CreateErrorDialog(errorText);
				return;
			}

			// 1匹目はVが2or3箇所じゃないとエラー
			int strict = (Get35Mode() == Star35PanelMode.From3V ? 3 : 2);
			int c = 0;
			for (int i = 0; i < 6; ++i)
			{
				if (ivs[i] == 31)
				{
					++c;
				}
			}
			if (c != strict)
			{
				// エラー
				if (strict == 3)
				{
					CreateErrorDialog(Messages.Instance.ErrorMessage["IvsStrict351_3V"]);
					return;
				}
				else
				{
					CreateErrorDialog(Messages.Instance.ErrorMessage["IvsStrict351_2V"]);
					return;
				}
			}

			int ability1 = f_ComboBoxAbility_351.SelectedIndex;
			int ability2 = f_ComboBoxAbility_352.SelectedIndex;
			int ability3 = f_ComboBoxAbility_353.SelectedIndex;
			if (ability1 >= 2) { ability1 = ability1 * 3 - 7; }
			if (ability2 >= 2) { ability2 = ability2 * 3 - 7; }
			if (ability3 >= 2) { ability3 = ability3 * 3 - 7; }
			int nature1 = Messages.Instance.Nature[m_PokemonInfo[2].ComboBoxNature.Text];
			int nature2 = Messages.Instance.Nature[m_PokemonInfo[3].ComboBoxNature.Text];
			int nature3 = Messages.Instance.Nature[m_PokemonInfo[4].ComboBoxNature.Text];

			bool noGender1 = f_CheckBoxNoGender_351.Checked;
			bool noGender2 = f_CheckBoxNoGender_352.Checked;
			bool noGender3 = f_CheckBoxNoGender_353.Checked;

			bool isDream1 = f_CheckBoxDream_351.Checked;
			bool isDream2 = f_CheckBoxDream_352.Checked;
			bool isDream3 = f_CheckBoxDream_353.Checked;

			int characteristic1 = Messages.Instance.Characteristic[m_PokemonInfo[2].ComboBoxCharacteristic.Text];
			int characteristic2 = Messages.Instance.Characteristic[m_PokemonInfo[3].ComboBoxCharacteristic.Text];
			int characteristic3 = Messages.Instance.Characteristic[m_PokemonInfo[4].ComboBoxCharacteristic.Text];

			var mode = Get35Mode();

			// 計算開始
			SeedSearcher searcher = new SeedSearcher(mode == Star35PanelMode.From2V ? SeedSearcher.Mode.Star35_6 : SeedSearcher.Mode.Star35_5);
			SeedSearcher.SetSixFirstCondition(ivs[0], ivs[1], ivs[2], ivs[3], ivs[4], ivs[5], ability1, nature1, characteristic1, noGender1, isDream1);
			SeedSearcher.SetSixSecondCondition(ivs[6], ivs[7], ivs[8], ivs[9], ivs[10], ivs[11], ability2, nature2, characteristic2, noGender2, isDream2);
			SeedSearcher.SetSixThirdCondition(ivs[12], ivs[13], ivs[14], ivs[15], ivs[16], ivs[17], ability3, nature3, characteristic3, noGender3, isDream3);

			int vCount = 0;
			for (int i = 0; i < 6; ++i)
			{
				if (ivs[6 + i] == 31)
				{
					++vCount;
				}
			}
			// 3V+自然発生の4Vは考慮しない
			if (vCount > 4)
			{
				vCount = 4;
			}
			// 遺伝箇所チェック
			bool[] vFlag = { false, false, false, false, false, false };
			for (int i = 0; i < 6; ++i)
			{
				if (ivs[i] == 31)
				{
					vFlag[i] = true;
				}
			}
			c = (mode == Star35PanelMode.From3V ? 3 : 2);
			int[] conditionIv = new int[6];
			int cursor = 0;
			for (int i = 0; i < 6; ++i)
			{
				if (ivs[i] != 31)
				{
					conditionIv[cursor++] = ivs[i];
					int vPos = ivs[i] % 8;
					if (vPos < 6 && vFlag[vPos] == false)
					{
						vFlag[vPos] = true;
						++c;
						if (c == vCount) // 遺伝終わり
						{
							break;
						}
					}
				}
			}

			// 条件ベクトルを求める
			c = 0;
			for (int i = 0; i < 6; ++i)
			{
				if (vFlag[i] == false)
				{
					conditionIv[cursor++] = ivs[i + 6];
					if (cursor == 6)
					{
						break;
					}
				}
			}

			if (mode == Star35PanelMode.From2V)
			{
				SeedSearcher.SetTargetCondition6(conditionIv[0], conditionIv[1], conditionIv[2], conditionIv[3], conditionIv[4], conditionIv[5]);
			}
			else if (mode == Star35PanelMode.From3V)
			{
				SeedSearcher.SetTargetCondition5(conditionIv[0], conditionIv[1], conditionIv[2], conditionIv[3], conditionIv[4]);
			}

			SearchImpl(searcher);
		}

		public void ReportProgress(int permille)
		{
			f_ButtonStartSearch.Text = Messages.Instance.SystemLabel["Searching"] + $" {permille / 10.0f:0.0}%";
		}

		// 検索処理共通
		async void SearchImpl(SeedSearcher searcher)
		{
			int minRerolls = 0;
			int maxRerolls = 3;
			try
			{
				minRerolls = int.Parse(f_TextBoxRerollsLower.Text);
				maxRerolls = int.Parse(f_TextBoxRerollsUpper.Text);
			}
			catch (Exception)
			{ }
			bool isEnableStop = f_CheckBoxStop.Checked;

			// ボタンを無効化
			f_ButtonStartSearch.Enabled = false;
			f_ButtonStartSearch.Text = Messages.Instance.SystemLabel["Searching"];
			f_ButtonStartSearch.BackColor = System.Drawing.Color.WhiteSmoke;

			// 時間計測
			bool isShowResultTime = f_CheckBoxShowResultTime.Checked;
			System.Diagnostics.Stopwatch stopWatch = null;
			if (isShowResultTime)
			{
				stopWatch = new System.Diagnostics.Stopwatch();
				stopWatch.Start();
			}

			var p = new Progress<int>(ReportProgress);
			await Task.Run(() =>
			{
				searcher.Calculate(isEnableStop, minRerolls, maxRerolls, p);
			});

			if (isShowResultTime && stopWatch != null)
			{
				stopWatch.Stop();
				MessageBox.Show($"{stopWatch.ElapsedMilliseconds}[ms]");
			}

			f_ButtonStartSearch.Enabled = true;
			f_ButtonStartSearch.Text = Messages.Instance.SystemLabel["StartSearch"];
			f_ButtonStartSearch.BackColor = System.Drawing.Color.GreenYellow;

			// 結果が見つからなかったらエラー
			if (searcher.Result.Count == 0)
			{
				// エラー
				CreateErrorDialog(Messages.Instance.ErrorMessage["NotFound"]);
				return;
			}
			else
			{
				if (searcher.Result.Count > 1)
				{
					using (StreamWriter sw = new StreamWriter("seeds.txt"))
					{
						foreach (var t in searcher.Result)
						{
							sw.WriteLine($"{t:X}");
						}
					}

					MessageBox.Show(Messages.Instance.SystemMessage["FindManySeeds"], Messages.Instance.SystemMessage["ResultDialogTitle"], MessageBoxButtons.OK, MessageBoxIcon.Information);
				}

				f_TextBoxResultSeed.Text = $"{searcher.Result[0]:X}";
				f_TabControlMain.SelectedIndex = 2;

				// 見つかったらリスト出力
				ListGenerate();
			}
		}

		private void f_ButtonListGenerate_Click(object sender, EventArgs e)
		{
			ListGenerate();
		}

		void ListGenerate()
		{
			// パラメータを取得
			UInt64 denSeed = 0;
			try
			{
				denSeed = UInt64.Parse(f_TextBoxResultSeed.Text, System.Globalization.NumberStyles.AllowHexSpecifier);
			}
			catch (Exception)
			{
				// エラー
				CreateErrorDialog(Messages.Instance.ErrorMessage["DenSeedFormat"]);
				return;
			}

			int maxFrameCount = 0;
			try
			{
				maxFrameCount = int.Parse(f_TextBoxMaxFrame.Text);
			}
			catch (Exception)
			{
				// エラー
				CreateErrorDialog(Messages.Instance.ErrorMessage["MaxFrameFormat"]);
				return;
			}

			int vCount = 0;
			try
			{
				vCount = int.Parse(f_TextBoxListVCount.Text);
			}
			catch (Exception)
			{
				// エラー
				CreateErrorDialog(Messages.Instance.ErrorMessage["VCountFormat"]);
				return;
			}

			bool isShinyCheck = f_CheckBoxListShiny.Checked;

			bool isNoGender = f_CheckBoxNoGender_List.Checked;
			bool isDream = f_CheckBoxDream_List.Checked;
			bool isShowSeed = f_CheckBoxShowSeed.Checked;

			ListGenerator listGenerator = new ListGenerator(denSeed, maxFrameCount, vCount, isShinyCheck, isNoGender, isDream, isShowSeed);
			listGenerator.Generate();
		}

		private void f_MenuItemLanguageJp_Click(object sender, EventArgs e)
		{
			if (!f_MenuItemLanguageJp.Checked)
			{
				ChangeLanguage(false, Language.Japanese);
			}
		}

		private void f_MenuItemLanguageEn_Click(object sender, EventArgs e)
		{
			if (!f_MenuItemLanguageEn.Checked)
			{
				ChangeLanguage(false, Language.English);
			}
		}

		private void f_MenuItemLanguageZh_Click(object sender, EventArgs e)
		{
			if (!f_MenuItemLanguageZh.Checked)
			{
				ChangeLanguage(false, Language.ChineseZh);
			}
		}

		private void f_MenuItemLanguageZh_TW_Click(object sender, EventArgs e)
		{
			if (!f_MenuItemLanguageZh_TW.Checked)
			{
				ChangeLanguage(false, Language.ChineseZh_TW);
			}
		}
		void ChangeLanguage(bool isFirst, Language language)
		{
            
            int[] nature = new int[5];
            int[] characteristic = new int[5];
            decimal[] pokemon = new decimal[5];

            if (!isFirst)
            {
                nature = new int[]{
                    Messages.Instance.Nature[f_ComboBoxNature_1.Text],
                    Messages.Instance.Nature[f_ComboBoxNature_2.Text],
                    Messages.Instance.Nature[f_ComboBoxNature_351.Text],
                    Messages.Instance.Nature[f_ComboBoxNature_352.Text],
                    Messages.Instance.Nature[f_ComboBoxNature_353.Text],
                };

                characteristic = new int[]{
					Messages.Instance.Characteristic[f_ComboBoxCharacteristic_1.Text],
					Messages.Instance.Characteristic[f_ComboBoxCharacteristic_2.Text],
					Messages.Instance.Characteristic[f_ComboBoxCharacteristic_351.Text],
                    Messages.Instance.Characteristic[f_ComboBoxCharacteristic_352.Text],
                    Messages.Instance.Characteristic[f_ComboBoxCharacteristic_353.Text],
                };

                pokemon = new decimal[]
                {
					Messages.Instance.Pokemon.ContainsKey(f_ComboBoxPokemon_1.Text) ? Messages.Instance.Pokemon[f_ComboBoxPokemon_1.Text] : -1,
					Messages.Instance.Pokemon.ContainsKey(f_ComboBoxPokemon_2.Text) ? Messages.Instance.Pokemon[f_ComboBoxPokemon_2.Text] : -1,
					Messages.Instance.Pokemon.ContainsKey(f_ComboBoxPokemon_351.Text) ? Messages.Instance.Pokemon[f_ComboBoxPokemon_351.Text] : -1,
                    Messages.Instance.Pokemon.ContainsKey(f_ComboBoxPokemon_352.Text) ? Messages.Instance.Pokemon[f_ComboBoxPokemon_352.Text] : -1,
                    Messages.Instance.Pokemon.ContainsKey(f_ComboBoxPokemon_353.Text) ? Messages.Instance.Pokemon[f_ComboBoxPokemon_353.Text] : -1
                };
            }

			// 言語のロード
			if (!Messages.Initialize(language))
			{
				MessageBox.Show("言語ファイルの読み込みに失敗しました。\n----------\n" + Messages.ErrorText, "エラー", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			// コンボボックスは一旦値を退避してセット
			int modeIndex = 0;
			int[] abilityIndex = new int[5];
			int denIndex = 0;
			int versionIndex = 0;
			int rarityIndex = 0;

			if (!isFirst)
			{                
                modeIndex = f_ComboBoxModeSelector_35.SelectedIndex;
				f_ComboBoxModeSelector_35.Items.Clear();

				f_ComboBoxNature_1.Items.Clear();
				f_ComboBoxNature_2.Items.Clear();
				f_ComboBoxNature_351.Items.Clear();
				f_ComboBoxNature_352.Items.Clear();
				f_ComboBoxNature_353.Items.Clear();

				f_ComboBoxCharacteristic_1.Items.Clear();
				f_ComboBoxCharacteristic_2.Items.Clear();
				f_ComboBoxCharacteristic_351.Items.Clear();
				f_ComboBoxCharacteristic_352.Items.Clear();
				f_ComboBoxCharacteristic_353.Items.Clear();

				f_ComboBoxPokemon_1.Items.Clear();
				f_ComboBoxPokemon_2.Items.Clear();
				f_ComboBoxPokemon_351.Items.Clear();
				f_ComboBoxPokemon_352.Items.Clear();
				f_ComboBoxPokemon_353.Items.Clear();

				abilityIndex = new int[]{
					f_ComboBoxAbility_1.SelectedIndex,
					f_ComboBoxAbility_2.SelectedIndex,
					f_ComboBoxAbility_351.SelectedIndex,
					f_ComboBoxAbility_352.SelectedIndex,
					f_ComboBoxAbility_353.SelectedIndex,
				};
				f_ComboBoxAbility_1.Items.Clear();
				f_ComboBoxAbility_2.Items.Clear();
				f_ComboBoxAbility_351.Items.Clear();
				f_ComboBoxAbility_352.Items.Clear();
				f_ComboBoxAbility_353.Items.Clear();

				denIndex = f_ComboBoxDenName.SelectedIndex;
				f_ComboBoxDenName.Items.Clear();

				versionIndex = f_ComboBoxGameVersion.SelectedIndex;
				f_ComboBoxGameVersion.Items.Clear();

				rarityIndex = f_ComboBoxRarity.SelectedIndex;
				f_ComboBoxRarity.Items.Clear();
			}

			// 設定変更
			m_Preferences.Language = language;

			// メニューアイテムのチェック
			switch (language)
			{
				case Language.Japanese:
					f_MenuItemLanguageJp.Checked = true;
					f_MenuItemLanguageEn.Checked = false;
					f_MenuItemLanguageZh.Checked = false;
					f_MenuItemLanguageZh_TW.Checked = false;
					break;

				case Language.English:
					f_MenuItemLanguageJp.Checked = false;
					f_MenuItemLanguageEn.Checked = true;
					f_MenuItemLanguageZh.Checked = false;
					f_MenuItemLanguageZh_TW.Checked = false;
					break;

				case Language.ChineseZh:
					f_MenuItemLanguageJp.Checked = false;
					f_MenuItemLanguageEn.Checked = false;
					f_MenuItemLanguageZh.Checked = true;
					f_MenuItemLanguageZh_TW.Checked = false;
					break;

				case Language.ChineseZh_TW:
					f_MenuItemLanguageJp.Checked = false;
					f_MenuItemLanguageEn.Checked = false;
					f_MenuItemLanguageZh.Checked = false;
					f_MenuItemLanguageZh_TW.Checked = true;
					break;	
			}

			// コントロールにメッセージを適用
			foreach (var pair in m_MultiLanguageControls)
			{
				string str = Messages.Instance.SystemLabel[pair.Key];
				foreach (var control in pair.Value)
				{
					control.Text = str;
				}
			}

			// パラメータラベル
			f_LabelStatus0_1.Text = Messages.Instance.Status[0];
			f_LabelStatus1_1.Text = Messages.Instance.Status[1];
			f_LabelStatus2_1.Text = Messages.Instance.Status[2];
			f_LabelStatus3_1.Text = Messages.Instance.Status[3];
			f_LabelStatus4_1.Text = Messages.Instance.Status[4];
			f_LabelStatus5_1.Text = Messages.Instance.Status[5];
			f_LabelStatus0_2.Text = Messages.Instance.Status[0];
			f_LabelStatus1_2.Text = Messages.Instance.Status[1];
			f_LabelStatus2_2.Text = Messages.Instance.Status[2];
			f_LabelStatus3_2.Text = Messages.Instance.Status[3];
			f_LabelStatus4_2.Text = Messages.Instance.Status[4];
			f_LabelStatus5_2.Text = Messages.Instance.Status[5];
			f_LabelStatus0_351.Text = Messages.Instance.Status[0];
			f_LabelStatus1_351.Text = Messages.Instance.Status[1];
			f_LabelStatus2_351.Text = Messages.Instance.Status[2];
			f_LabelStatus3_351.Text = Messages.Instance.Status[3];
			f_LabelStatus4_351.Text = Messages.Instance.Status[4];
			f_LabelStatus5_351.Text = Messages.Instance.Status[5];
			f_LabelStatus0_352.Text = Messages.Instance.Status[0];
			f_LabelStatus1_352.Text = Messages.Instance.Status[1];
			f_LabelStatus2_352.Text = Messages.Instance.Status[2];
			f_LabelStatus3_352.Text = Messages.Instance.Status[3];
			f_LabelStatus4_352.Text = Messages.Instance.Status[4];
			f_LabelStatus5_352.Text = Messages.Instance.Status[5];
			f_LabelStatus0_353.Text = Messages.Instance.Status[0];
			f_LabelStatus1_353.Text = Messages.Instance.Status[1];
			f_LabelStatus2_353.Text = Messages.Instance.Status[2];
			f_LabelStatus3_353.Text = Messages.Instance.Status[3];
			f_LabelStatus4_353.Text = Messages.Instance.Status[4];
			f_LabelStatus5_353.Text = Messages.Instance.Status[5];
			f_LabelStatus20_1.Text = Messages.Instance.Status[0];
			f_LabelStatus21_1.Text = Messages.Instance.Status[1];
			f_LabelStatus22_1.Text = Messages.Instance.Status[2];
			f_LabelStatus23_1.Text = Messages.Instance.Status[3];
			f_LabelStatus24_1.Text = Messages.Instance.Status[4];
			f_LabelStatus25_1.Text = Messages.Instance.Status[5];
			f_LabelStatus20_2.Text = Messages.Instance.Status[0];
			f_LabelStatus21_2.Text = Messages.Instance.Status[1];
			f_LabelStatus22_2.Text = Messages.Instance.Status[2];
			f_LabelStatus23_2.Text = Messages.Instance.Status[3];
			f_LabelStatus24_2.Text = Messages.Instance.Status[4];
			f_LabelStatus25_2.Text = Messages.Instance.Status[5];
			f_LabelStatus20_351.Text = Messages.Instance.Status[0];
			f_LabelStatus21_351.Text = Messages.Instance.Status[1];
			f_LabelStatus22_351.Text = Messages.Instance.Status[2];
			f_LabelStatus23_351.Text = Messages.Instance.Status[3];
			f_LabelStatus24_351.Text = Messages.Instance.Status[4];
			f_LabelStatus25_351.Text = Messages.Instance.Status[5];
			f_LabelStatus20_352.Text = Messages.Instance.Status[0];
			f_LabelStatus21_352.Text = Messages.Instance.Status[1];
			f_LabelStatus22_352.Text = Messages.Instance.Status[2];
			f_LabelStatus23_352.Text = Messages.Instance.Status[3];
			f_LabelStatus24_352.Text = Messages.Instance.Status[4];
			f_LabelStatus25_352.Text = Messages.Instance.Status[5];
			f_LabelStatus20_353.Text = Messages.Instance.Status[0];
			f_LabelStatus21_353.Text = Messages.Instance.Status[1];
			f_LabelStatus22_353.Text = Messages.Instance.Status[2];
			f_LabelStatus23_353.Text = Messages.Instance.Status[3];
			f_LabelStatus24_353.Text = Messages.Instance.Status[4];
			f_LabelStatus25_353.Text = Messages.Instance.Status[5];
			f_LabelLevel_1.Text = Messages.Instance.Status[6];
			f_LabelLevel_2.Text = Messages.Instance.Status[6];
			f_LabelLevel_351.Text = Messages.Instance.Status[6];
            f_LabelLevel_352.Text = Messages.Instance.Status[6];
            f_LabelLevel_353.Text = Messages.Instance.Status[6];
			f_LabelPokemon_1.Text = Messages.Instance.SystemLabel["Pokemon"];
			f_LabelPokemon_2.Text = Messages.Instance.SystemLabel["Pokemon"];
			f_LabelPokemon_351.Text = Messages.Instance.SystemLabel["Pokemon"];
            f_LabelPokemon_352.Text = Messages.Instance.SystemLabel["Pokemon"];
            f_LabelPokemon_353.Text = Messages.Instance.SystemLabel["Pokemon"];
			f_LabelStatus_1.Text = Messages.Instance.SystemLabel["Status"];
			f_LabelStatus_2.Text = Messages.Instance.SystemLabel["Status"];
			f_LabelStatus_351.Text = Messages.Instance.SystemLabel["Status"];
			f_LabelStatus_352.Text = Messages.Instance.SystemLabel["Status"];
			f_LabelStatus_353.Text = Messages.Instance.SystemLabel["Status"];
			f_ButtonIvsCalc_1.Text = Messages.Instance.SystemLabel["CalculateIVs"];
			f_ButtonIvsCalc_2.Text = Messages.Instance.SystemLabel["CalculateIVs"];
			f_ButtonIvsCalc_351.Text = Messages.Instance.SystemLabel["CalculateIVs"];
			f_ButtonIvsCalc_352.Text = Messages.Instance.SystemLabel["CalculateIVs"];
			f_ButtonIvsCalc_353.Text = Messages.Instance.SystemLabel["CalculateIVs"];

            // コンボボックス再初期化
            InitializeComboBox();

			// 退避していた選択をセット
			f_ComboBoxModeSelector_35.SelectedIndex = modeIndex;
			if (!isFirst)
			{
				PokemonFormUtility.SelectNatureComboBox(f_ComboBoxNature_1, nature[0]);
				PokemonFormUtility.SelectNatureComboBox(f_ComboBoxNature_2, nature[1]);
				PokemonFormUtility.SelectNatureComboBox(f_ComboBoxNature_351, nature[2]);
				PokemonFormUtility.SelectNatureComboBox(f_ComboBoxNature_352, nature[3]);
				PokemonFormUtility.SelectNatureComboBox(f_ComboBoxNature_353, nature[4]);
				PokemonFormUtility.SelectCharacteristicComboBox(f_ComboBoxCharacteristic_1, characteristic[0]);
				PokemonFormUtility.SelectCharacteristicComboBox(f_ComboBoxCharacteristic_2, characteristic[1]);
				PokemonFormUtility.SelectCharacteristicComboBox(f_ComboBoxCharacteristic_351, characteristic[2]);
				PokemonFormUtility.SelectCharacteristicComboBox(f_ComboBoxCharacteristic_352, characteristic[3]);
				PokemonFormUtility.SelectCharacteristicComboBox(f_ComboBoxCharacteristic_353, characteristic[4]);
				f_ComboBoxAbility_1.SelectedIndex = abilityIndex[0];
				f_ComboBoxAbility_2.SelectedIndex = abilityIndex[1];
				f_ComboBoxAbility_351.SelectedIndex = abilityIndex[2];
				f_ComboBoxAbility_352.SelectedIndex = abilityIndex[3];
				f_ComboBoxAbility_353.SelectedIndex = abilityIndex[4];

                populateComboBoxes();
				for (int a = 0; a < 5; ++a)
				{
					if (pokemon[a] != -1)
					{
						var idxa = (from i in Messages.Instance.Pokemon.Keys
									where Messages.Instance.Pokemon[i] == pokemon[a]
									select i).First();
						m_PokemonInfo[a].ComboBoxName.SelectedIndex = m_PokemonInfo[a].ComboBoxName.Items.IndexOf(idxa);
					}
				}

				f_ComboBoxDenName.SelectedIndex = denIndex;
				f_ComboBoxGameVersion.SelectedIndex = versionIndex;
				f_ComboBoxRarity.SelectedIndex = rarityIndex;
            }
		}

        bool firstRun = true;
        private void populateComboBoxes()
        {
            if (firstRun)
                IVCalcNetFramework.IVCalculator.loadData();

            var Pokemon = Messages.Instance.Pokemon.Keys.ToList();
            Pokemon.Sort();

			f_ComboBoxPokemon_1.AutoCompleteMode = AutoCompleteMode.SuggestAppend;
			f_ComboBoxPokemon_1.AutoCompleteSource = AutoCompleteSource.ListItems;
			f_ComboBoxPokemon_1.Items.AddRange(Pokemon.ToArray());

			f_ComboBoxPokemon_2.AutoCompleteMode = AutoCompleteMode.SuggestAppend;
			f_ComboBoxPokemon_2.AutoCompleteSource = AutoCompleteSource.ListItems;
			f_ComboBoxPokemon_2.Items.AddRange(Pokemon.ToArray());

			f_ComboBoxPokemon_351.AutoCompleteMode = AutoCompleteMode.SuggestAppend;
            f_ComboBoxPokemon_351.AutoCompleteSource = AutoCompleteSource.ListItems;
            f_ComboBoxPokemon_351.Items.AddRange(Pokemon.ToArray());

			f_ComboBoxPokemon_352.AutoCompleteMode = AutoCompleteMode.SuggestAppend;
			f_ComboBoxPokemon_352.AutoCompleteSource = AutoCompleteSource.ListItems;
			f_ComboBoxPokemon_352.Items.AddRange(Pokemon.ToArray());

			f_ComboBoxPokemon_353.AutoCompleteMode = AutoCompleteMode.SuggestAppend;
			f_ComboBoxPokemon_353.AutoCompleteSource = AutoCompleteSource.ListItems;
			f_ComboBoxPokemon_353.Items.AddRange(Pokemon.ToArray());

            firstRun = false;
        }

		private void f_ButtonIvsCalc_1_Click(object sender, EventArgs e)
		{
			IvsCalculate(0);
		}

		private void f_ButtonIvsCalc_2_Click(object sender, EventArgs e)
		{
			IvsCalculate(1);
		}

		private void f_ButtonIvsCalc_351_Click(object sender, EventArgs e)
		{
			IvsCalculate(2);
		}

		private void f_ButtonIvsCalc_352_Click(object sender, EventArgs e)
		{
			IvsCalculate(3);
		}

		private void f_ButtonIvsCalc_353_Click(object sender, EventArgs e)
		{
			IvsCalculate(4);
		}

		void IvsCalculate(int index)
		{
			var pokemonInfo = m_PokemonInfo[index];
			string pokemon = pokemonInfo.ComboBoxName.Text;
			string levelText = pokemonInfo.TextBoxLevel.Text;
			if (pokemon == "" || !Messages.Instance.Pokemon.ContainsKey(pokemon))
			{
				CreateErrorDialog(Messages.Instance.ErrorMessage["InvalidPokemon"]);
			}
			else
			{
				decimal pokemonID = Messages.Instance.Pokemon[pokemon];
				try
				{
					int lv = int.Parse(levelText);
					if (lv > 100 || lv < 1)
					{
						CreateErrorDialog(Messages.Instance.ErrorMessage["LevelRange"]);
						return;
					}
					int HP = int.Parse(pokemonInfo.TextBoxStatus[0].Text);
					int Atk = int.Parse(pokemonInfo.TextBoxStatus[1].Text);
					int Def = int.Parse(pokemonInfo.TextBoxStatus[2].Text);
					int SpAtk = int.Parse(pokemonInfo.TextBoxStatus[3].Text);
					int SpDef = int.Parse(pokemonInfo.TextBoxStatus[4].Text);
					int Spd = int.Parse(pokemonInfo.TextBoxStatus[5].Text);
				
					int nature = Messages.Instance.Nature[pokemonInfo.ComboBoxNature.Text];
					int characteristic = Messages.Instance.Characteristic[pokemonInfo.ComboBoxCharacteristic.Text];

					var IVs = IVCalculator.getIVs(pokemonID, lv, nature, characteristic, new List<int>() { HP, Atk, Def, SpAtk, SpDef, Spd }, null);

					bool uncertain = false;

					for (int i = 0; i < 6; ++i)
					{
						if (IVs[i].Count == 1)
						{
							pokemonInfo.TextBoxIvs[i].Text = IVs[i].First().ToString();
						}
						else
						{
							uncertain = true;
							pokemonInfo.TextBoxIvs[i].Text = "";
						}
					}

					if (uncertain)
					{
						MessageBox.Show(Messages.Instance.SystemMessage["FindManyIvs"], Messages.Instance.SystemMessage["ResultDialogTitle"], MessageBoxButtons.OK, MessageBoxIcon.Information);
					}
				}
				catch (FormatException except)
				{
					CreateErrorDialog(Messages.Instance.ErrorMessage["VFormat"]);
				}
				catch (ArgumentOutOfRangeException except)
				{
					CreateErrorDialog(Messages.Instance.ErrorMessage["CouldNotCalculateIVs"]);
				}
				catch (IVCalculator.StatException s)
				{
					CreateErrorDialog(Messages.Instance.ErrorMessage["StatError"] + Messages.Instance.Status[s.Stat]);
				}
			}
		}

		void EnterTextBoxAllSelect(TextBoxBase textBox)
		{
			textBox.SelectAll();
		}

		/*
		bool testFlag = false;
		private void f_TextBoxIv0_351_Enter(object sender, EventArgs e)
		{
			f_TextBoxIv0_351.SelectAll();
			if (Control.MouseButtons != MouseButtons.None)
			{
				testFlag = true;
			}
		}

		private void f_TextBoxIv0_351_MouseDown(object sender, MouseEventArgs e)
		{
			if (testFlag)
			{
				f_TextBoxIv0_351.SelectAll();
				testFlag = false;
			}
		}
		*/
	}
}
