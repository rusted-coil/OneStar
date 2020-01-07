using System;
using OneStarCalculator;
using System.Windows.Forms;
using System.Threading.Tasks;
using System.IO;
using System.Collections.Generic;

namespace OneStar
{
	// 設定項目
	public enum Language
	{
		Japanese,
		English
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

		TextBox[] m_TextBoxIvsList12 = new TextBox[12];
		TextBox[] m_TextBoxIvsList35 = new TextBox[18];

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

			IsInitialized = true;
		}

		void InitializeComboBox()
		{
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
			PokemonFormUtility.SetAbilityComboBox(f_ComboBoxAbility_1, 4);
			PokemonFormUtility.SetAbilityComboBox(f_ComboBoxAbility_2, 4);
		}

		void InitializeView()
		{
			// ★3～5パネル
			SetCheckResult(-1);

			// 共通
			f_TextBoxMaxFrame.Text = "5000";

			f_TextBoxRerolls.Text = "3";
			f_CheckBoxStop.Checked = true;
			f_TextBoxListVCount.Text = "4";

			// 扱いやすいようにキャッシュ
			m_TextBoxIvsList12[0] = f_TextBoxIv0_1;
			m_TextBoxIvsList12[1] = f_TextBoxIv1_1;
			m_TextBoxIvsList12[2] = f_TextBoxIv2_1;
			m_TextBoxIvsList12[3] = f_TextBoxIv3_1;
			m_TextBoxIvsList12[4] = f_TextBoxIv4_1;
			m_TextBoxIvsList12[5] = f_TextBoxIv5_1;
			m_TextBoxIvsList12[6] = f_TextBoxIv0_2;
			m_TextBoxIvsList12[7] = f_TextBoxIv1_2;
			m_TextBoxIvsList12[8] = f_TextBoxIv2_2;
			m_TextBoxIvsList12[9] = f_TextBoxIv3_2;
			m_TextBoxIvsList12[10] = f_TextBoxIv4_2;
			m_TextBoxIvsList12[11] = f_TextBoxIv5_2;
			m_TextBoxIvsList35[0] = f_TextBoxIv0_351;
			m_TextBoxIvsList35[1] = f_TextBoxIv1_351;
			m_TextBoxIvsList35[2] = f_TextBoxIv2_351;
			m_TextBoxIvsList35[3] = f_TextBoxIv3_351;
			m_TextBoxIvsList35[4] = f_TextBoxIv4_351;
			m_TextBoxIvsList35[5] = f_TextBoxIv5_351;
			m_TextBoxIvsList35[6] = f_TextBoxIv0_352;
			m_TextBoxIvsList35[7] = f_TextBoxIv1_352;
			m_TextBoxIvsList35[8] = f_TextBoxIv2_352;
			m_TextBoxIvsList35[9] = f_TextBoxIv3_352;
			m_TextBoxIvsList35[10] = f_TextBoxIv4_352;
			m_TextBoxIvsList35[11] = f_TextBoxIv5_352;
			m_TextBoxIvsList35[12] = f_TextBoxIv0_353;
			m_TextBoxIvsList35[13] = f_TextBoxIv1_353;
			m_TextBoxIvsList35[14] = f_TextBoxIv2_353;
			m_TextBoxIvsList35[15] = f_TextBoxIv3_353;
			m_TextBoxIvsList35[16] = f_TextBoxIv4_353;
			m_TextBoxIvsList35[17] = f_TextBoxIv5_353;

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
			m_MultiLanguageControls["RerollsBefore"] = new Control[] { f_LabelRerollsBefore };
			m_MultiLanguageControls["RerollsAfter"] = new Control[] { f_LabelRerollsAfter };
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
					ivs[i] = int.Parse(m_TextBoxIvsList35[i].Text);
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
			int[] ivs = new int[12];
			for (int i = 0; i < 12; ++i)
			{
				try
				{
					ivs[i] = int.Parse(m_TextBoxIvsList12[i].Text);
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

			// 1匹目はVが1箇所じゃないとエラー
			int c = 0;
			for (int i = 0; i < 6; ++i)
			{
				if (ivs[i] == 31)
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
			for (int i = 6; i < 12; ++i)
			{
				if (ivs[i] == 31)
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
			int nature1 = Messages.Instance.Nature[f_ComboBoxNature_1.Text];
			int nature2 = Messages.Instance.Nature[f_ComboBoxNature_2.Text];

			bool noGender1 = f_CheckBoxNoGender_1.Checked;
			bool noGender2 = f_CheckBoxNoGender_2.Checked;

			bool isDream1 = f_CheckBoxDream_1.Checked;
			bool isDream2 = f_CheckBoxDream_2.Checked;

			// 計算開始
			SeedSearcher searcher = new SeedSearcher(SeedSearcher.Mode.Star12);
			SeedSearcher.SetFirstCondition(ivs[0], ivs[1], ivs[2], ivs[3], ivs[4], ivs[5], ability1, nature1, noGender1, isDream1);
			SeedSearcher.SetNextCondition(ivs[6], ivs[7], ivs[8], ivs[9], ivs[10], ivs[11], ability2, nature2, noGender2, isDream2);

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
					ivs[i] = int.Parse(m_TextBoxIvsList35[i].Text);
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
			int nature1 = Messages.Instance.Nature[f_ComboBoxNature_351.Text];
			int nature2 = Messages.Instance.Nature[f_ComboBoxNature_352.Text];
			int nature3 = Messages.Instance.Nature[f_ComboBoxNature_353.Text];

			bool noGender1 = f_CheckBoxNoGender_351.Checked;
			bool noGender2 = f_CheckBoxNoGender_352.Checked;
			bool noGender3 = f_CheckBoxNoGender_353.Checked;

			bool isDream1 = f_CheckBoxDream_351.Checked;
			bool isDream2 = f_CheckBoxDream_352.Checked;
			bool isDream3 = f_CheckBoxDream_353.Checked;

			int characteristic1 = Messages.Instance.Characteristic[f_ComboBoxCharacteristic_351.Text];
			int characteristic2 = Messages.Instance.Characteristic[f_ComboBoxCharacteristic_352.Text];
			int characteristic3 = Messages.Instance.Characteristic[f_ComboBoxCharacteristic_353.Text];

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

		// 検索処理共通
		async void SearchImpl(SeedSearcher searcher)
		{
			int maxRerolls = 0;
			try
			{
				maxRerolls = int.Parse(f_TextBoxRerolls.Text);
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

			await Task.Run(() =>
			{
				searcher.Calculate(isEnableStop, maxRerolls);
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
						for (int i = 0; i < searcher.Result.Count; ++i)
						{
							sw.WriteLine($"{searcher.Result[i]:X}");
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

		void ChangeLanguage(bool isFirst, Language language)
		{
            
            int[] nature = new int[5];
            int[] characteristic = new int[5];

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
                    Messages.Instance.Characteristic[f_ComboBoxCharacteristic_351.Text],
                    Messages.Instance.Characteristic[f_ComboBoxCharacteristic_352.Text],
                    Messages.Instance.Characteristic[f_ComboBoxCharacteristic_353.Text],
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

			if (!isFirst)
			{
                
                modeIndex = f_ComboBoxModeSelector_35.SelectedIndex;
				f_ComboBoxModeSelector_35.Items.Clear();

				f_ComboBoxNature_1.Items.Clear();
				f_ComboBoxNature_2.Items.Clear();
				f_ComboBoxNature_351.Items.Clear();
				f_ComboBoxNature_352.Items.Clear();
				f_ComboBoxNature_353.Items.Clear();
				
				f_ComboBoxCharacteristic_351.Items.Clear();
				f_ComboBoxCharacteristic_352.Items.Clear();
				f_ComboBoxCharacteristic_353.Items.Clear();

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
			}

			// 設定変更
			m_Preferences.Language = language;

			// メニューアイテムのチェック
			switch (language)
			{
				case Language.Japanese:
					f_MenuItemLanguageJp.Checked = true;
					f_MenuItemLanguageEn.Checked = false;
					break;

				case Language.English:
					f_MenuItemLanguageJp.Checked = false;
					f_MenuItemLanguageEn.Checked = true;
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
				PokemonFormUtility.SelectCharacteristicComboBox(f_ComboBoxCharacteristic_351, characteristic[0]);
				PokemonFormUtility.SelectCharacteristicComboBox(f_ComboBoxCharacteristic_352, characteristic[1]);
				PokemonFormUtility.SelectCharacteristicComboBox(f_ComboBoxCharacteristic_353, characteristic[2]);
				f_ComboBoxAbility_1.SelectedIndex = abilityIndex[0];
				f_ComboBoxAbility_2.SelectedIndex = abilityIndex[1];
				f_ComboBoxAbility_351.SelectedIndex = abilityIndex[2];
				f_ComboBoxAbility_352.SelectedIndex = abilityIndex[3];
				f_ComboBoxAbility_353.SelectedIndex = abilityIndex[4];
			}
		}
	}
}
