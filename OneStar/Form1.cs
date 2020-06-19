using System;
using OneStarCalculator;
using System.Windows.Forms;
using System.Threading.Tasks;
using System.IO;
using System.Collections.Generic;
using IVCalcNetFramework;
using System.Linq;
using PKHeX_Raid_Plugin;
using System.Net;
using PKHeX.Core;

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

	public partial class MainForm : Form
	{
		public bool IsInitialized { get; private set; }

		// 環境設定
		Preferences m_Preferences;

		// 巣穴データ
		readonly RaidData c_RaidData = new RaidData();

		// 現在の巣穴データ
		int m_CurrentDenIndex = -2;
		int m_CurrentGameVersion = -1;
		int m_CurrentRarity = -1;
		List<RaidData.Pokemon> m_EncounterList = new List<RaidData.Pokemon>();

		// イベントレイド関連
		ToolStripMenuItem m_MenuItemEventUpdate = null; // 更新ボタン
		List<ToolStripMenuItem> m_MenuItemEventIdList = new List<ToolStripMenuItem>(); // IDボタン

		// GPUスレッド数ボタン
		ToolStripMenuItem[] m_MenuItemGpuLoopList;

		// ポケモン入力フォームごとのセット
		class PokemonInfoForm {
			public ComboBox ComboBoxName { get; set; } = null;
			public TextBox TextBoxLevel { get; set; } = null;
			public TextBox[] TextBoxIvs { get; set; } = new TextBox[6];
			public TextBox[] TextBoxStatus { get; set; } = new TextBox[6];
			public ComboBox ComboBoxNature { get; set; } = null;
			public ComboBox ComboBoxCharacteristic { get; set; } = null;
			public ComboBox ComboBoxAbility { get; set; } = null;
		};
		PokemonInfoForm[] m_PokemonInfo = new PokemonInfoForm[6];

		// 言語設定可能コントロール
		Dictionary<string, Control[]> m_MultiLanguageControls;

		// DLCマップ用フォーム
		Form2 m_MapForm = null;

		class Star35PanelMode
		{
			public enum ModeType {
				From2V,
				From3V
			};

			public ModeType Mode { get; private set; }

			public override string ToString()
			{
				if (Mode == ModeType.From2V)
				{
					return Messages.Instance.SystemLabel["Pokemon35_1_2V"];
				}
				else if (Mode == ModeType.From3V)
				{
					return Messages.Instance.SystemLabel["Pokemon35_1_3V"];
				}
				return "";
			}

			public Star35PanelMode(ModeType mode)
			{
				Mode = mode;
			}
		};

		int m_is2VEnable = -1;
		int m_is3VEnable = -1;

		Star35PanelMode.ModeType Get35Mode()
		{
			return (f_ComboBoxModeSelector_35.SelectedItem as Star35PanelMode).Mode;
		}

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

			IVCalcNetFramework.IVCalculator.loadData();

			InitializeComponent();

			// ビューの初期化
			InitializeView();

            IsInitialized = true;
		}

		bool IsGpuAvailable()
		{
			// 32bit版は不可
#if _X86
			CreateErrorDialog(Messages.Instance.ErrorMessage["GpuError32bit"]);
			return false;
#endif
			// CUDAに使用できるデバイス
			if (SeedSearcher.GetCudaDeviceCount() <= 0)
			{
				CreateErrorDialog(Messages.Instance.ErrorMessage["NoCudaDevice"]);
				return false;
			}
			return true;
		}

		void InitializeComboBox()
		{
			// 巣穴
			foreach (var key in Messages.Instance.Den.Keys)
			{
				f_ComboBoxDenName.Items.Add(key);
			}

			foreach (var version in Messages.Instance.Version)
			{
				f_ComboBoxGameVersion.Items.Add(version);
			}
			foreach (var rarity in Messages.Instance.DenRarity)
			{
				f_ComboBoxRarity.Items.Add(rarity);
			}

			// ★3～5パネル
			PokemonFormUtility.SetNatureComboBox(f_ComboBoxNature_351);
			PokemonFormUtility.SetNatureComboBox(f_ComboBoxNature_352);
			PokemonFormUtility.SetNatureComboBox(f_ComboBoxNature_353);
			PokemonFormUtility.SetCharacteristicComboBox(f_ComboBoxCharacteristic_351);
			PokemonFormUtility.SetCharacteristicComboBox(f_ComboBoxCharacteristic_352);
			PokemonFormUtility.SetCharacteristicComboBox(f_ComboBoxCharacteristic_353);

			// ★1～2パネル
			PokemonFormUtility.SetNatureComboBox(f_ComboBoxNature_1);
			PokemonFormUtility.SetNatureComboBox(f_ComboBoxNature_2);
			PokemonFormUtility.SetNatureComboBox(f_ComboBoxNature_3);
			PokemonFormUtility.SetCharacteristicComboBox(f_ComboBoxCharacteristic_1);
			PokemonFormUtility.SetCharacteristicComboBox(f_ComboBoxCharacteristic_2);
			PokemonFormUtility.SetCharacteristicComboBox(f_ComboBoxCharacteristic_3);
		}

		void InitializeView()
		{
			f_PicturePoint.Parent = DenMap;
			f_PicturePoint.Visible = false;

			// ★3～5パネル
			SetCheckResult(-1);

			// 共通
			f_TextBoxRerollsLower.Text = "0";
			f_TextBoxRerollsUpper.Text = "3";

			// 設定をセット
			if (m_Preferences.IsUseGpu)
			{
				// 使用不可
				if (!IsGpuAvailable())
				{
					m_Preferences.IsUseGpu = false;
				}
			}
			f_MenuItemUseGpu.Checked = m_Preferences.IsUseGpu;
			m_MenuItemGpuLoopList = new ToolStripMenuItem[]{
				f_MenuItemGpuLoop29,
				f_MenuItemGpuLoop28,
				f_MenuItemGpuLoop27,
				f_MenuItemGpuLoop26,
				f_MenuItemGpuLoop25,
				f_MenuItemGpuLoop24,
				f_MenuItemGpuLoop23,
				f_MenuItemGpuLoop22,
				f_MenuItemGpuLoop21,
			};
			if (m_Preferences.GpuLoop < 0)
			{
				m_Preferences.GpuLoop = 0;
			}
			else if (m_Preferences.GpuLoop > 8)
			{
				m_Preferences.GpuLoop = 8;
			}
			m_MenuItemGpuLoopList[m_Preferences.GpuLoop].Checked = true;
			f_CheckBoxStop.Checked = m_Preferences.SearchStop;
			f_CheckBoxShowResultTime.Checked = m_Preferences.SearchShowDuration;
			f_TextBoxMaxFrame.Text = m_Preferences.ListMaxFrame.ToString();
			f_CheckBoxListShiny.Checked = m_Preferences.ListOnlyShiny;
			f_CheckBoxShowSeed.Checked = m_Preferences.ListShowSeed;
			f_CheckBoxShowEC.Checked = m_Preferences.ListShowEC;
			f_CheckBoxShowAbilityName.Checked = m_Preferences.ListShowAbilityName;

#region ポケモンフォーム情報キャッシュ
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
			m_PokemonInfo[0].ComboBoxAbility = f_ComboBoxAbility_1;
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
			m_PokemonInfo[1].ComboBoxAbility = f_ComboBoxAbility_2;
			m_PokemonInfo[2] = new PokemonInfoForm();
			m_PokemonInfo[2].ComboBoxName = f_ComboBoxPokemon_3;
			m_PokemonInfo[2].TextBoxLevel = f_TextBoxLevel_3;
			m_PokemonInfo[2].TextBoxIvs[0] = f_TextBoxIv0_3;
			m_PokemonInfo[2].TextBoxIvs[1] = f_TextBoxIv1_3;
			m_PokemonInfo[2].TextBoxIvs[2] = f_TextBoxIv2_3;
			m_PokemonInfo[2].TextBoxIvs[3] = f_TextBoxIv3_3;
			m_PokemonInfo[2].TextBoxIvs[4] = f_TextBoxIv4_3;
			m_PokemonInfo[2].TextBoxIvs[5] = f_TextBoxIv5_3;
			m_PokemonInfo[2].TextBoxStatus[0] = f_TextBoxStatus0_3;
			m_PokemonInfo[2].TextBoxStatus[1] = f_TextBoxStatus1_3;
			m_PokemonInfo[2].TextBoxStatus[2] = f_TextBoxStatus2_3;
			m_PokemonInfo[2].TextBoxStatus[3] = f_TextBoxStatus3_3;
			m_PokemonInfo[2].TextBoxStatus[4] = f_TextBoxStatus4_3;
			m_PokemonInfo[2].TextBoxStatus[5] = f_TextBoxStatus5_3;
			m_PokemonInfo[2].ComboBoxNature = f_ComboBoxNature_3;
			m_PokemonInfo[2].ComboBoxCharacteristic = f_ComboBoxCharacteristic_3;
			m_PokemonInfo[2].ComboBoxAbility = f_ComboBoxAbility_3;
			m_PokemonInfo[3] = new PokemonInfoForm();
			m_PokemonInfo[3].ComboBoxName = f_ComboBoxPokemon_351;
			m_PokemonInfo[3].TextBoxLevel = f_TextBoxLevel_351;
			m_PokemonInfo[3].TextBoxIvs[0] = f_TextBoxIv0_351;
			m_PokemonInfo[3].TextBoxIvs[1] = f_TextBoxIv1_351;
			m_PokemonInfo[3].TextBoxIvs[2] = f_TextBoxIv2_351;
			m_PokemonInfo[3].TextBoxIvs[3] = f_TextBoxIv3_351;
			m_PokemonInfo[3].TextBoxIvs[4] = f_TextBoxIv4_351;
			m_PokemonInfo[3].TextBoxIvs[5] = f_TextBoxIv5_351;
			m_PokemonInfo[3].TextBoxStatus[0] = f_TextBoxStatus0_351;
			m_PokemonInfo[3].TextBoxStatus[1] = f_TextBoxStatus1_351;
			m_PokemonInfo[3].TextBoxStatus[2] = f_TextBoxStatus2_351;
			m_PokemonInfo[3].TextBoxStatus[3] = f_TextBoxStatus3_351;
			m_PokemonInfo[3].TextBoxStatus[4] = f_TextBoxStatus4_351;
			m_PokemonInfo[3].TextBoxStatus[5] = f_TextBoxStatus5_351;
			m_PokemonInfo[3].ComboBoxNature = f_ComboBoxNature_351;
			m_PokemonInfo[3].ComboBoxCharacteristic = f_ComboBoxCharacteristic_351;
			m_PokemonInfo[3].ComboBoxAbility = f_ComboBoxAbility_351;
			m_PokemonInfo[4] = new PokemonInfoForm();
			m_PokemonInfo[4].ComboBoxName = f_ComboBoxPokemon_352;
			m_PokemonInfo[4].TextBoxLevel = f_TextBoxLevel_352;
			m_PokemonInfo[4].TextBoxIvs[0] = f_TextBoxIv0_352;
			m_PokemonInfo[4].TextBoxIvs[1] = f_TextBoxIv1_352;
			m_PokemonInfo[4].TextBoxIvs[2] = f_TextBoxIv2_352;
			m_PokemonInfo[4].TextBoxIvs[3] = f_TextBoxIv3_352;
			m_PokemonInfo[4].TextBoxIvs[4] = f_TextBoxIv4_352;
			m_PokemonInfo[4].TextBoxIvs[5] = f_TextBoxIv5_352;
			m_PokemonInfo[4].TextBoxStatus[0] = f_TextBoxStatus0_352;
			m_PokemonInfo[4].TextBoxStatus[1] = f_TextBoxStatus1_352;
			m_PokemonInfo[4].TextBoxStatus[2] = f_TextBoxStatus2_352;
			m_PokemonInfo[4].TextBoxStatus[3] = f_TextBoxStatus3_352;
			m_PokemonInfo[4].TextBoxStatus[4] = f_TextBoxStatus4_352;
			m_PokemonInfo[4].TextBoxStatus[5] = f_TextBoxStatus5_352;
			m_PokemonInfo[4].ComboBoxNature = f_ComboBoxNature_352;
			m_PokemonInfo[4].ComboBoxCharacteristic = f_ComboBoxCharacteristic_352;
			m_PokemonInfo[4].ComboBoxAbility = f_ComboBoxAbility_352;
			m_PokemonInfo[5] = new PokemonInfoForm();
			m_PokemonInfo[5].ComboBoxName = f_ComboBoxPokemon_353;
			m_PokemonInfo[5].TextBoxLevel = f_TextBoxLevel_353;
			m_PokemonInfo[5].TextBoxIvs[0] = f_TextBoxIv0_353;
			m_PokemonInfo[5].TextBoxIvs[1] = f_TextBoxIv1_353;
			m_PokemonInfo[5].TextBoxIvs[2] = f_TextBoxIv2_353;
			m_PokemonInfo[5].TextBoxIvs[3] = f_TextBoxIv3_353;
			m_PokemonInfo[5].TextBoxIvs[4] = f_TextBoxIv4_353;
			m_PokemonInfo[5].TextBoxIvs[5] = f_TextBoxIv5_353;
			m_PokemonInfo[5].TextBoxStatus[0] = f_TextBoxStatus0_353;
			m_PokemonInfo[5].TextBoxStatus[1] = f_TextBoxStatus1_353;
			m_PokemonInfo[5].TextBoxStatus[2] = f_TextBoxStatus2_353;
			m_PokemonInfo[5].TextBoxStatus[3] = f_TextBoxStatus3_353;
			m_PokemonInfo[5].TextBoxStatus[4] = f_TextBoxStatus4_353;
			m_PokemonInfo[5].TextBoxStatus[5] = f_TextBoxStatus5_353;
			m_PokemonInfo[5].ComboBoxNature = f_ComboBoxNature_353;
			m_PokemonInfo[5].ComboBoxCharacteristic = f_ComboBoxCharacteristic_353;
			m_PokemonInfo[5].ComboBoxAbility = f_ComboBoxAbility_353;
			// コールバックをセット
			for (int a = 0; a < 6; ++a)
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
#endregion

			// イベントレイドデータ
			RefreshEventRaidData();

			// 言語設定用コントロールをセット
			m_MultiLanguageControls = new Dictionary<string, Control[]>();
			m_MultiLanguageControls["Tab35"] = new Control[] { f_TabPage1 };
			m_MultiLanguageControls["Tab12"] = new Control[] { f_TabPage2 };
			m_MultiLanguageControls["TabList"] = new Control[] { f_TabPage3 };
			m_MultiLanguageControls["Ivs"] = new Control[]{
				f_LabelIvs_1,
				f_LabelIvs_2,
				f_LabelIvs_3,
				f_LabelIvs_351,
				f_LabelIvs_352,
				f_LabelIvs_353,
			};
			m_MultiLanguageControls["Pokemon35_2"] = new Control[] { f_GroupBoxPokemon_352 };
			m_MultiLanguageControls["Pokemon35_3"] = new Control[] { f_GroupBoxPokemon_353 };
			m_MultiLanguageControls["Pokemon12_1"] = new Control[] { f_GroupBoxPokemon_1 };
			m_MultiLanguageControls["Pokemon12_2"] = new Control[] { f_GroupBoxPokemon_2 };
			m_MultiLanguageControls["Pokemon12_3"] = new Control[] { f_CheckBoxThirdEnable };
			m_MultiLanguageControls["CheckIvsButton"] = new Control[] { f_ButtonIvsCheck };
			m_MultiLanguageControls["CheckIvsResultTitle"] = new Control[] { f_LabelCheckResultTitle };
			m_MultiLanguageControls["MaxFrame"] = new Control[] { f_LabelMaxFrame };
			m_MultiLanguageControls["OnlyShiny"] = new Control[] { f_CheckBoxListShiny };
			m_MultiLanguageControls["ShowSeed"] = new Control[] { f_CheckBoxShowSeed };
			m_MultiLanguageControls["ShowEC"] = new Control[] { f_CheckBoxShowEC };
			m_MultiLanguageControls["ListButton"] = new Control[] { f_ButtonListGenerate };
			m_MultiLanguageControls["ShowDuration"] = new Control[] { f_CheckBoxShowResultTime };
			m_MultiLanguageControls["StartSearch"] = new Control[] { f_ButtonStartSearch };
			m_MultiLanguageControls["Rerolls"] = new Control[] { f_LabelRerolls };
			m_MultiLanguageControls["Range"] = new Control[] { f_LabelRerollsRange };
			m_MultiLanguageControls["SearchStop"] = new Control[] { f_CheckBoxStop };
			m_MultiLanguageControls["Information"] = new Control[] {
				f_ButtonEncounterInfo_1,
				f_ButtonEncounterInfo_2,
				f_ButtonEncounterInfo_3,
				f_ButtonEncounterInfo_351,
				f_ButtonEncounterInfo_352,
				f_ButtonEncounterInfo_353,
			};
			m_MultiLanguageControls["Den"] = new Control[] { f_LabelDenName };
			m_MultiLanguageControls["GameVersion"] = new Control[] { f_LabelGameVersion };
			m_MultiLanguageControls["DenType"] = new Control[] { f_LabelRarity };
			m_MultiLanguageControls["Attribute"] = new Control[] {
				f_GroupBoxStatus1,
				f_GroupBoxStatus2,
				f_GroupBoxStatus3,
				f_GroupBoxStatus351,
				f_GroupBoxStatus352,
				f_GroupBoxStatus353,
			};

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

		// モード選択変更イベント
		private void f_ComboBoxModeSelector_35_SelectedIndexChanged(object sender, EventArgs e)
		{
			RefreshPokemonComboBox();
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
					ivs[i] = int.Parse(m_PokemonInfo[3].TextBoxIvs[i].Text);
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
			int strict = (Get35Mode() == Star35PanelMode.ModeType.From3V ? 3 : 2);
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
			int needNumber = (Get35Mode() == Star35PanelMode.ModeType.From2V ? 6 : 5);

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

			bool isEnableThird = f_CheckBoxThirdEnable.Checked;

			// フォームから必要な情報を取得
			RaidData.Pokemon[] pokemonData = new RaidData.Pokemon[3];
			for (int i = 0; i < 3; ++i)
			{
				pokemonData[i] = m_PokemonInfo[i].ComboBoxName.SelectedItem as RaidData.Pokemon;
			}

			int[] ivs = new int[18];
			for (int i = 0; i < (isEnableThird ? 18 : 12); ++i)
			{
				try
				{
					ivs[i] = int.Parse(m_PokemonInfo[i / 6].TextBoxIvs[i % 6].Text);
				}
				catch (Exception)
				{
					// エラー
					errorText = Messages.Instance.ErrorMessage["IvsFormat"];
					isCheckFailed = true;
					break;
				}
				if (ivs[i] < 0 || ivs[i] > 31)
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

			// V箇所が足りなかったらエラー
			for (int a = 0; a < (isEnableThird ? 3 : 2); ++a)
			{
				int c = 0;
				for (int b = 0; b < 6; ++b)
				{
					if (ivs[a * 6 + b] == 31)
					{
						++c;
					}
				}
				// 1匹目は1Vじゃないとエラー
				if (a == 0 && c != 1)
				{
					// エラー
					CreateErrorDialog(Messages.Instance.ErrorMessage["IvsStrict1"]);
					return;
				}
				if (c < pokemonData[a].FlawlessIvs)
				{
					// エラー
					CreateErrorDialog(Messages.Instance.ErrorMessage["IvsStrict"]);
					return;
				}
			}

			// 計算準備
			SeedSearcher searcher = new SeedSearcher(SeedSearcher.Mode.Star12);

			// 条件をセット
			for (int i = 0; i < (isEnableThird ? 3 : 2); ++i)
			{
				var pokemonInfo = m_PokemonInfo[i];
				var pokemon = pokemonData[i];

				int ability = (pokemonInfo.ComboBoxAbility.SelectedItem as PokemonFormUtility.AbilityItem).Type;
				int nature = Messages.Instance.Nature[pokemonInfo.ComboBoxNature.Text];
				int natureTableId = pokemon.NatureTableId;
				bool noGender = pokemon.IsFixedGender;
				int abilityFlag = pokemon.Ability;
				int characteristic = Messages.Instance.Characteristic[pokemonInfo.ComboBoxCharacteristic.Text];
				int flawlessIvs = pokemon.FlawlessIvs;
				
				SeedSearcher.Set12Condition(
					i,
					ivs[i * 6],
					ivs[i * 6 + 1],
					ivs[i * 6 + 2],
					ivs[i * 6 + 3],
					ivs[i * 6 + 4],
					ivs[i * 6 + 5],
					ability, nature, natureTableId, characteristic, noGender, abilityFlag, flawlessIvs);
			}

			// 計算開始
			SearchImpl(searcher);
		}

		// ★3～5検索
		void SeedSearch35()
		{
			bool isCheckFailed = false;
			string errorText = "";

			// フォームから必要な情報を取得
			RaidData.Pokemon[] pokemonData = new RaidData.Pokemon[3];
			for (int i = 0; i < 3; ++i)
			{
				pokemonData[i] = m_PokemonInfo[3 + i].ComboBoxName.SelectedItem as RaidData.Pokemon;
			}

			int[] ivs = new int[18];
			for (int i = 0; i < 18; ++i)
			{
				try
				{
					ivs[i] = int.Parse(m_PokemonInfo[3 + i / 6].TextBoxIvs[i % 6].Text);
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

			// V箇所が足りなかったらエラー
			int strict = (Get35Mode() == Star35PanelMode.ModeType.From3V ? 3 : 2);
			for (int a = 0; a < 3; ++a)
			{
				int c = 0;
				for (int b = 0; b < 6; ++b)
				{
					if (ivs[a * 6 + b] == 31)
					{
						++c;
					}
				}
				// 1匹目はVが2or3箇所じゃないとエラー
				if (a == 0 && c != strict)
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
				if (c < pokemonData[a].FlawlessIvs)
				{
					// エラー
					CreateErrorDialog(Messages.Instance.ErrorMessage["IvsStrict"]);
					return;
				}
			}

			// 計算準備
			var mode = Get35Mode();
			SeedSearcher.Mode searchMode;
			if (m_Preferences.IsUseGpu)
			{
				searchMode = SeedSearcher.Mode.Cuda35;
				SeedSearcher.CudaInitialize();
			}
			else
			{
				searchMode = (mode == Star35PanelMode.ModeType.From2V ? SeedSearcher.Mode.Star35_6 : SeedSearcher.Mode.Star35_5);
			}
			SeedSearcher searcher = new SeedSearcher(searchMode);

			// 条件をセット
			for (int i = 0; i < 3; ++i)
			{
				var pokemonInfo = m_PokemonInfo[3 + i];
				var pokemon = pokemonData[i];

				int ability = (pokemonInfo.ComboBoxAbility.SelectedItem as PokemonFormUtility.AbilityItem).Type;
				int nature = Messages.Instance.Nature[pokemonInfo.ComboBoxNature.Text];
				int natureTableId = pokemon.NatureTableId;
				bool noGender = pokemon.IsFixedGender;
				int abilityFlag = pokemon.Ability;
				int characteristic = Messages.Instance.Characteristic[pokemonInfo.ComboBoxCharacteristic.Text];
				int flawlessIvs = pokemon.FlawlessIvs;

				if (m_Preferences.IsUseGpu)
				{
					SeedSearcher.SetCudaCondition(
						i,
						ivs[i * 6],
						ivs[i * 6 + 1],
						ivs[i * 6 + 2],
						ivs[i * 6 + 3],
						ivs[i * 6 + 4],
						ivs[i * 6 + 5],
						ability, nature, natureTableId, characteristic, noGender, abilityFlag, flawlessIvs);
				}
				else
				{
					SeedSearcher.Set35Condition(
						i,
						ivs[i * 6],
						ivs[i * 6 + 1],
						ivs[i * 6 + 2],
						ivs[i * 6 + 3],
						ivs[i * 6 + 4],
						ivs[i * 6 + 5],
						ability, nature, natureTableId, characteristic, noGender, abilityFlag, flawlessIvs);
				}
			}

			// 遺伝箇所チェック
			{
				int vCount = pokemonData[1].FlawlessIvs;

				bool[] vFlag = { false, false, false, false, false, false };
				for (int i = 0; i < 6; ++i)
				{
					if (ivs[i] == 31)
					{
						vFlag[i] = true;
					}
				}
				int c = (mode == Star35PanelMode.ModeType.From3V ? 3 : 2);
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

				if (mode == Star35PanelMode.ModeType.From2V)
				{
					if (m_Preferences.IsUseGpu)
					{
						searcher.CudaLoopPartition = m_Preferences.GpuLoop + 1;
						SeedSearcher.SetCudaTargetCondition6(conditionIv[0], conditionIv[1], conditionIv[2], conditionIv[3], conditionIv[4], conditionIv[5]);
					}
					else
					{
						SeedSearcher.SetTargetCondition6(conditionIv[0], conditionIv[1], conditionIv[2], conditionIv[3], conditionIv[4], conditionIv[5]);
					}
				}
				else if (mode == Star35PanelMode.ModeType.From3V)
				{
					if (m_Preferences.IsUseGpu)
					{
						searcher.CudaLoopPartition = m_Preferences.GpuLoop;
						SeedSearcher.SetCudaTargetCondition5(conditionIv[0], conditionIv[1], conditionIv[2], conditionIv[3], conditionIv[4]);
					}
					else
					{
						SeedSearcher.SetTargetCondition5(conditionIv[0], conditionIv[1], conditionIv[2], conditionIv[3], conditionIv[4]);
					}
				}
			}

			// 計算開始
			SearchImpl(searcher);
		}

		public void ReportProgress(int permille)
		{
			f_ButtonStartSearch.Text = Messages.Instance.SystemLabel["Searching"] + $" {permille / 10.0f:0.0}%";
		}

		// 検索処理共通
		async void SearchImpl(SeedSearcher searcher, bool isTest = false, int testRerolls = 0, UInt64 testSeed = 0)
		{
			int minRerolls = 0;
			int maxRerolls = 3;
			if (isTest)
			{
				minRerolls = testRerolls;
				maxRerolls = testRerolls;
			}
			else
			{
				try
				{
					minRerolls = int.Parse(f_TextBoxRerollsLower.Text);
					maxRerolls = int.Parse(f_TextBoxRerollsUpper.Text);
				}
				catch (Exception)
				{ }
			}
			bool isEnableStop = (isTest ? false : f_CheckBoxStop.Checked);

			// ボタンを無効化
			f_ButtonStartSearch.Enabled = false;
			f_ButtonStartSearch.Text = Messages.Instance.SystemLabel["Searching"];
			f_ButtonStartSearch.BackColor = System.Drawing.Color.WhiteSmoke;

			// 時間計測
			bool isShowResultTime = f_CheckBoxShowResultTime.Checked;
			System.Diagnostics.Stopwatch stopWatch = null;
			if (isShowResultTime || isTest)
			{
				stopWatch = new System.Diagnostics.Stopwatch();
				stopWatch.Start();
			}

			var p = new Progress<int>(ReportProgress);
			await Task.Run(() =>
			{
				searcher.Calculate(isEnableStop, minRerolls, maxRerolls, p);
			});

			if (stopWatch != null)
			{
				stopWatch.Stop();
			}

			f_ButtonStartSearch.Enabled = true;
			f_ButtonStartSearch.Text = Messages.Instance.SystemLabel["StartSearch"];
			f_ButtonStartSearch.BackColor = System.Drawing.Color.GreenYellow;


			// テストモード
			if (isTest)
			{
				// 結果が一つかつ目的のseedだったら成功
				if (searcher.Result.Count == 1 && searcher.Result[0] == testSeed)
				{
					MessageBox.Show($"{Messages.Instance.SystemMessage["GpuCheckSuccessed"]} {stopWatch.ElapsedMilliseconds}[ms]");
				}
				else
				{
					// エラー
					CreateErrorDialog(Messages.Instance.ErrorMessage["GpuCheckFailed"]);
				}
				return;
			}

			if (isShowResultTime)
			{
				MessageBox.Show($"{stopWatch.ElapsedMilliseconds}[ms]");
			}

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

			// 設定を保存
			m_Preferences.ListMaxFrame = maxFrameCount;

			var pokemon = f_ComboBoxPokemon_List.SelectedItem as RaidData.Pokemon;

			bool isShinyCheck = f_CheckBoxListShiny.Checked;

			bool isShowSeed = f_CheckBoxShowSeed.Checked;
			bool isShowEc = f_CheckBoxShowEC.Checked;
			bool isShowAbilityName = f_CheckBoxShowAbilityName.Checked;

			ListGenerator listGenerator = new ListGenerator(denSeed, maxFrameCount, pokemon, isShinyCheck, isShowSeed, isShowEc, isShowAbilityName);
			listGenerator.Generate();
		}

#region 言語設定変更
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
			// コンボボックスは一旦値を退避してセット
			int modeIndex = 0;
			int[] nature = new int[6];
            int[] characteristic = new int[6];
			int[] pokemon = new int[7];
			int[] abilityIndex = new int[6];
			int denIndex = -1;
			int versionIndex = m_Preferences.GameVersion;
			int rarityIndex = 0;

			if (!isFirst)
            {
				modeIndex = f_ComboBoxModeSelector_35.SelectedIndex;
				f_ComboBoxModeSelector_35.Items.Clear();

				for (int i = 0; i < 6; ++i)
				{
					pokemon[i] = m_PokemonInfo[i].ComboBoxName.SelectedIndex;
					nature[i] = Messages.Instance.Nature[m_PokemonInfo[i].ComboBoxNature.Text];
					characteristic[i] = Messages.Instance.Characteristic[m_PokemonInfo[i].ComboBoxCharacteristic.Text];
					abilityIndex[i] = m_PokemonInfo[i].ComboBoxAbility.SelectedIndex;

					m_PokemonInfo[i].ComboBoxName.Items.Clear();
					m_PokemonInfo[i].ComboBoxNature.Items.Clear();
					m_PokemonInfo[i].ComboBoxCharacteristic.Items.Clear();
					m_PokemonInfo[i].ComboBoxAbility.Items.Clear();
				}
				pokemon[6] = f_ComboBoxPokemon_List.SelectedIndex;

				denIndex = Messages.Instance.Den[f_ComboBoxDenName.Text];
				f_ComboBoxDenName.Items.Clear();

				versionIndex = f_ComboBoxGameVersion.SelectedIndex;
				f_ComboBoxGameVersion.Items.Clear();

				rarityIndex = f_ComboBoxRarity.SelectedIndex;
				f_ComboBoxRarity.Items.Clear();
			}

			// 言語のロード
			if (!Messages.Initialize(language))
			{
				MessageBox.Show("言語ファイルの読み込みに失敗しました。\n----------\n" + Messages.ErrorText, "エラー", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
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
			f_StripMenuItemLanguage.Text = Messages.Instance.SystemLabel["Language"];
			f_StripMenuItemWindowSize.Text = Messages.Instance.SystemLabel["WindowSize"];
			f_MenuItemWindowSizeNormal.Text = Messages.Instance.SystemLabel["WindowSizeNormal"];
			f_MenuItemWindowSizeSmall.Text = Messages.Instance.SystemLabel["WindowSizeSmall"];
			m_MenuItemEventUpdate.Text = Messages.Instance.SystemLabel["UpdateEventDen"];
			f_StripMenuItemEventRaid.Text = Messages.Instance.SystemLabel["EventDen"];
			f_StripMenuItemGpuSetting.Text = Messages.Instance.SystemLabel["GpuSettings"];
			f_MenuItemUseGpu.Text = Messages.Instance.SystemLabel["UseGpu"];
			f_MenuItemThreadCount.Text = Messages.Instance.SystemLabel["ThreadSetting"];
			f_MenuItemGpuTest.Text = Messages.Instance.SystemLabel["GpuCheck"];

			foreach (var pair in m_MultiLanguageControls)
			{
				string str = Messages.Instance.SystemLabel[pair.Key];
				foreach (var control in pair.Value)
				{
					control.Text = str;
				}
			}

#region パラメータラベル
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
			f_LabelStatus0_3.Text = Messages.Instance.Status[0];
			f_LabelStatus1_3.Text = Messages.Instance.Status[1];
			f_LabelStatus2_3.Text = Messages.Instance.Status[2];
			f_LabelStatus3_3.Text = Messages.Instance.Status[3];
			f_LabelStatus4_3.Text = Messages.Instance.Status[4];
			f_LabelStatus5_3.Text = Messages.Instance.Status[5];
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
			f_LabelStatus20_3.Text = Messages.Instance.Status[0];
			f_LabelStatus21_3.Text = Messages.Instance.Status[1];
			f_LabelStatus22_3.Text = Messages.Instance.Status[2];
			f_LabelStatus23_3.Text = Messages.Instance.Status[3];
			f_LabelStatus24_3.Text = Messages.Instance.Status[4];
			f_LabelStatus25_3.Text = Messages.Instance.Status[5];
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
			f_LabelLevel_3.Text = Messages.Instance.Status[6];
			f_LabelLevel_351.Text = Messages.Instance.Status[6];
            f_LabelLevel_352.Text = Messages.Instance.Status[6];
            f_LabelLevel_353.Text = Messages.Instance.Status[6];
			f_ButtonIvsCalc_1.Text = Messages.Instance.SystemLabel["CalculateIVs"];
			f_ButtonIvsCalc_2.Text = Messages.Instance.SystemLabel["CalculateIVs"];
			f_ButtonIvsCalc_3.Text = Messages.Instance.SystemLabel["CalculateIVs"];
			f_ButtonIvsCalc_351.Text = Messages.Instance.SystemLabel["CalculateIVs"];
			f_ButtonIvsCalc_352.Text = Messages.Instance.SystemLabel["CalculateIVs"];
			f_ButtonIvsCalc_353.Text = Messages.Instance.SystemLabel["CalculateIVs"];
#endregion

			// コンボボックス再初期化
			InitializeComboBox();

			// 退避していた選択をセット
			{
				foreach (var item in f_ComboBoxDenName.Items)
				{
					if (Messages.Instance.Den[item.ToString()] == denIndex)
					{
						f_ComboBoxDenName.SelectedItem = item;
						break;
					}
				}
				f_ComboBoxGameVersion.SelectedIndex = versionIndex;
				f_ComboBoxRarity.SelectedIndex = rarityIndex;
			}

			if (isFirst)
			{
				f_ComboBoxModeSelector_35.SelectedIndex = modeIndex;

				for (int a = 0; a < 6; ++a)
				{
					m_PokemonInfo[a].ComboBoxName.SelectedIndex = 0;
				}
				f_ComboBoxPokemon_List.SelectedIndex = 0;
			}
			else
			{
				// 言語変更によりキーのみリフレッシュ
				foreach (var encounter in m_EncounterList)
				{
					encounter.RefreshKey();
				}

				RefreshModeComboBox();
				f_ComboBoxModeSelector_35.SelectedIndex = modeIndex;

				RefreshPokemonComboBox();

				for (int a = 0; a < 6; ++a)
				{
					m_PokemonInfo[a].ComboBoxName.SelectedIndex = pokemon[a];
				}
				f_ComboBoxPokemon_List.SelectedIndex = pokemon[6];

				for (int i = 0; i < 6; ++i)
				{
					PokemonFormUtility.SelectNatureComboBox(m_PokemonInfo[i].ComboBoxNature, nature[i]);
					PokemonFormUtility.SelectCharacteristicComboBox(m_PokemonInfo[i].ComboBoxCharacteristic, characteristic[i]);
					m_PokemonInfo[i].ComboBoxAbility.SelectedIndex = abilityIndex[i];
				}
            }
		}
#endregion

#region 個体値計算ボタン イベント定義
        private void f_ButtonIvsCalc_1_Click(object sender, EventArgs e)
		{
			IvsCalculate(0);
		}

		private void f_ButtonIvsCalc_2_Click(object sender, EventArgs e)
		{
			IvsCalculate(1);
		}

		private void f_ButtonIvsCalc_3_Click(object sender, EventArgs e)
		{
			IvsCalculate(2);
		}

		private void f_ButtonIvsCalc_351_Click(object sender, EventArgs e)
		{
			IvsCalculate(3);
		}

		private void f_ButtonIvsCalc_352_Click(object sender, EventArgs e)
		{
			IvsCalculate(4);
		}

		private void f_ButtonIvsCalc_353_Click(object sender, EventArgs e)
		{
			IvsCalculate(5);
		}
#endregion

		// 個体値計算
		void IvsCalculate(int index)
		{
			var pokemonInfo = m_PokemonInfo[index];
			// 計算に使うポケモン
			RaidData.Pokemon encounterData = pokemonInfo.ComboBoxName.SelectedItem as RaidData.Pokemon;

			if (encounterData == null)
			{
				CreateErrorDialog(Messages.Instance.ErrorMessage["InvalidPokemon"]);
			}
			else
			{
				string levelText = pokemonInfo.TextBoxLevel.Text;
				decimal pokemonID = encounterData.CalcSpecies;
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
				catch (FormatException)
				{
					CreateErrorDialog(Messages.Instance.ErrorMessage["VFormat"]);
				}
				catch (ArgumentOutOfRangeException)
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

		// ★1～2検索 3匹目有効化
		private void f_CheckBoxThirdEnable_CheckedChanged(object sender, EventArgs e)
		{
			f_GroupBoxPokemon_3.Enabled = f_CheckBoxThirdEnable.Checked;
		}

#region 巣穴選択変更イベント定義
        private void f_ComboBoxDenName_SelectedIndexChanged(object sender, EventArgs e)
		{
			// 選択情報を取得
			int denIndex = Messages.Instance.Den[f_ComboBoxDenName.Text];
			if (denIndex != m_CurrentDenIndex)
			{
				m_CurrentDenIndex = denIndex;
				RefreshDen();
			}
		}

		private void f_ComboBoxGameVersion_SelectedIndexChanged(object sender, EventArgs e)
		{
			// 選択情報を取得
			int gameVersion = f_ComboBoxGameVersion.SelectedIndex;
			if (gameVersion != m_CurrentGameVersion)
			{
				m_CurrentGameVersion = gameVersion;
				m_Preferences.GameVersion = gameVersion;
				RefreshDen();
			}
		}

		private void f_ComboBoxRarity_SelectedIndexChanged(object sender, EventArgs e)
		{
			// 選択情報を取得
			int rarity = f_ComboBoxRarity.SelectedIndex;
			if (rarity != m_CurrentRarity)
			{
				m_CurrentRarity = rarity;
				RefreshDen();
			}
		}

#endregion

		// 巣穴情報変更
		void RefreshDen()
		{
			if (m_CurrentDenIndex < -1 || m_CurrentGameVersion < 0 || m_CurrentRarity < 0)
			{
				return;
			}
			RaidTemplate[] raidEntries = null;

			// イベントレイド
			if (m_CurrentDenIndex < 0)
			{
				raidEntries = c_RaidData.GetEventRaidEntries(m_Preferences.EventId, m_CurrentGameVersion);
			}
			// 本編レイド
			else
			{
				raidEntries = c_RaidData.GetRaidEntries(m_CurrentDenIndex, m_CurrentGameVersion, m_CurrentRarity);
			}

			if (raidEntries == null)
			{
				return;
			}

			// 本編レイド
			if (m_CurrentDenIndex >= 0)
			{
				if (c_RaidData.GetRaidMap(m_CurrentDenIndex) == 0)
				{
					var location = c_RaidData.GetRaidLocation(m_CurrentDenIndex);
					f_PicturePoint.Location = new System.Drawing.Point(location.X - 5, location.Y - 5);
					f_PicturePoint.Visible = true;
				}
				else
				{
					// フォームが閉じていたら開く
					/*
					if (m_MapForm == null)
					{
						m_MapForm = new Form2();
						m_MapForm.FormClosed += new FormClosedEventHandler((x, y) => { m_MapForm = null; });
						m_MapForm.Show();
					}

					var location = c_RaidData.GetRaidLocation(m_CurrentDenIndex);
					m_MapForm.SetPoint(location);
					*/
				}
			}
			else
			{
				f_PicturePoint.Visible = false;
			}

			// データ構築
			m_EncounterList.Clear();
			Dictionary<string, int> encounterIndex = new Dictionary<string, int>();
			for (int c = 0; c < 5; ++c)
			{
				foreach (var entry in raidEntries)
				{
					if (entry.Probabilities[c] > 0)
					{
						var pokemon = new RaidData.Pokemon(entry, c);
						string key = pokemon.Key;

						// 全く同じ見た目のポケモンの場合マージする
						if (encounterIndex.ContainsKey(key))
						{
							m_EncounterList[encounterIndex[key]].Merge(entry);
						}
						else
						{
							encounterIndex.Add(key, m_EncounterList.Count);
							m_EncounterList.Add(pokemon);
						}
					}
				}
			}

			// ★3以上かつ2Vがいるかチェック
			bool isEnable2V = false;
			bool isEnable3V = false;
			foreach (var encounter in m_EncounterList)
			{
				if (encounter.FlawlessIvs <= 2 && encounter.Rank >= 2)
				{
					isEnable2V = true;
				}
				if (encounter.FlawlessIvs == 3 && encounter.Rank >= 2)
				{
					isEnable3V = true;
				}
			}
			if (m_is2VEnable == -1 || (m_is2VEnable == 1) != isEnable2V
				|| m_is3VEnable == -1 || (m_is3VEnable == 1) != isEnable3V)
			{
				m_is2VEnable = (isEnable2V ? 1 : 0);
				m_is3VEnable = (isEnable3V ? 1 : 0);
				RefreshModeComboBox();
				f_ComboBoxModeSelector_35.SelectedIndex = 0;
			}

			RefreshPokemonComboBox();
		}
		void RefreshModeComboBox()
		{
			// モード
			f_ComboBoxModeSelector_35.Items.Clear();
			if (m_is2VEnable == 1)
			{
				f_ComboBoxModeSelector_35.Items.Add(new Star35PanelMode(Star35PanelMode.ModeType.From2V));
			}
			if (m_is3VEnable == 1)
			{
				f_ComboBoxModeSelector_35.Items.Add(new Star35PanelMode(Star35PanelMode.ModeType.From3V));
			}
		}
		void RefreshPokemonComboBox()
		{
			var mode = Get35Mode();

			// コンボボックスをクリア
			for (int i = 0; i < 6; ++i)
			{
				m_PokemonInfo[i].ComboBoxName.Items.Clear();
			}
			f_ComboBoxPokemon_List.Items.Clear();

			// コンボボックスに反映
			foreach (var encounter in m_EncounterList)
			{
				// 1Vのみ
				if (encounter.FlawlessIvs == 1)
				{
					f_ComboBoxPokemon_1.Items.Add(encounter);
				}
				// ★3まで
				if (encounter.Rank < 3)
				{
					f_ComboBoxPokemon_2.Items.Add(encounter);
					f_ComboBoxPokemon_3.Items.Add(encounter);
				}
				// ★3以上
				if (encounter.Rank >= 2)
				{
					// 2Vモードor3Vモード
					if (mode == Star35PanelMode.ModeType.From2V && encounter.FlawlessIvs == 2)
					{
						f_ComboBoxPokemon_351.Items.Add(encounter);
					}
					else if (mode == Star35PanelMode.ModeType.From3V && encounter.FlawlessIvs == 3)
					{
						f_ComboBoxPokemon_351.Items.Add(encounter);
					}
					// 3V～4V
					if (encounter.FlawlessIvs >= 3 && encounter.FlawlessIvs <= 4)
					{
						f_ComboBoxPokemon_352.Items.Add(encounter);
					}
					f_ComboBoxPokemon_353.Items.Add(encounter);
				}
				f_ComboBoxPokemon_List.Items.Add(encounter);
			}

			for (int i = 0; i < 6; ++i)
			{
				m_PokemonInfo[i].ComboBoxName.SelectedIndex = 0;
			}
			f_ComboBoxPokemon_List.SelectedIndex = 0;
		}

#region レイド情報ボタン イベント定義
		private void f_ButtonEncounterInfo_1_Click(object sender, EventArgs e)
		{
			ShowEncounterInfo(0);
		}

		private void f_ButtonEncounterInfo_2_Click(object sender, EventArgs e)
		{
			ShowEncounterInfo(1);
		}

		private void f_ButtonEncounterInfo_3_Click(object sender, EventArgs e)
		{
			ShowEncounterInfo(2);
		}

		private void f_ButtonEncounterInfo_351_Click(object sender, EventArgs e)
		{
			ShowEncounterInfo(3);
		}

		private void f_ButtonEncounterInfo_352_Click(object sender, EventArgs e)
		{
			ShowEncounterInfo(4);
		}

		private void f_ButtonEncounterInfo_353_Click(object sender, EventArgs e)
		{
			ShowEncounterInfo(5);
		}
#endregion

		void ShowEncounterInfo(int index)
		{
			RaidData.Pokemon pokemon = m_PokemonInfo[index].ComboBoxName.SelectedItem as RaidData.Pokemon;
			PersonalInfo info = PersonalTable.SWSH[pokemon.DataSpecies];
			if (info.ATK == 0)
			{
				info = PersonalTable.USUM[pokemon.DataSpecies];
			}
			var abilityList = PKHeX.Core.Util.GetAbilitiesList(Messages.Instance.LangCode);

			// ポケモンの種類
			string str = pokemon.Key;
			str += "\n-----";

			// レイドの情報
			str += $"\n{Messages.Instance.SystemLabel["ListVCount"]} {pokemon.FlawlessIvs}";
			if (pokemon.Ability == 2)
			{
				str += $"\n{Messages.Instance.SystemLabel["HiddenFixed"]}";
			}
			else if (pokemon.Ability == 3)
			{
				str += $"\n{Messages.Instance.SystemLabel["NoHidden"]}";
			}
			else if (pokemon.Ability == 4)
			{
				str += $"\n{Messages.Instance.SystemLabel["HiddenPossible"]}";
			}
			str += "\n-----";

			// ポケモンの情報
			if (info.Genderless)
			{
				str += $"\n{Messages.Instance.SystemLabel["GenderNone"]}";
			}
			else if (info.OnlyFemale)
			{
				str += $"\n{Messages.Instance.SystemLabel["GenderFemale"]}";
			}
			else if (info.OnlyMale)
			{
				str += $"\n{Messages.Instance.SystemLabel["GenderMale"]}";
			}
			else if (info.Gender == 31)
			{
				str += $"\n{Messages.Instance.SystemLabel["GenderMF71"]}";
			}
			else if (info.Gender == 63)
			{
				str += $"\n{Messages.Instance.SystemLabel["GenderMF31"]}";
			}
			else if (info.Gender == 127)
			{
				str += $"\n{Messages.Instance.SystemLabel["GenderMF11"]}";
			}
			else if (info.Gender == 191)
			{
				str += $"\n{Messages.Instance.SystemLabel["GenderMF13"]}";
			}
			else if (info.Gender == 225)
			{
				str += $"\n{Messages.Instance.SystemLabel["GenderMF17"]}";
			}
			else
			{
				str += $"\n性別比率: {info.Gender}";
			}
			if (info.Abilities[0] == info.Abilities[1])
			{
				str += $"\n{Messages.Instance.Ability[2]}: {abilityList[info.Abilities[0]]}";

				// 夢特性
				if (pokemon.Ability != 3 && info.Abilities[2] != info.Abilities[0])
				{
					str += $"\n{Messages.Instance.Ability[3]}: {abilityList[info.Abilities[2]]}";
				}
			}
			else
			{
				str += $"\n{Messages.Instance.Ability[0]}: {abilityList[info.Abilities[0]]}";
				str += $"\n{Messages.Instance.Ability[1]}: {abilityList[info.Abilities[1]]}";

				// 夢特性
				if (pokemon.Ability != 3 && info.Abilities[2] != info.Abilities[0] && info.Abilities[2] != info.Abilities[1])
				{
					str += $"\n{Messages.Instance.Ability[3]}: {abilityList[info.Abilities[2]]}";
				}
			}

			MessageBox.Show(str, Messages.Instance.SystemMessage["EncounterInfoDialogTitle"], MessageBoxButtons.OK, MessageBoxIcon.Information);
		}

		void RefreshEventId()
		{
			foreach (var menuItem in m_MenuItemEventIdList)
			{
				if (m_Preferences.EventId == menuItem.Text)
				{
					menuItem.Checked = true;
				}
				else
				{
					menuItem.Checked = false;
				}
			}
		}

		// イベントレイドIDボタン
        private void MenuItemEventIdSelect(object sender, EventArgs e)
        {
            m_Preferences.EventId = ((ToolStripMenuItem)sender).Text;
            RefreshEventId();
            if (m_CurrentDenIndex == -1)
            {
                RefreshDen();
            }
        }

        private void UpdateEventToolStripMenuItem_Click(object sender, EventArgs e)
        {
            string url = "https://raw.githubusercontent.com/rusted-coil/OneStar/master/Data/EventDen.json";
            string path = Directory.GetCurrentDirectory() + "//data/EventDen.json";

            // 设置参数
            HttpWebRequest request = WebRequest.Create(url) as HttpWebRequest;
            //发送请求并获取相应回应数据
            HttpWebResponse response = request.GetResponse() as HttpWebResponse;
            //直到request.GetResponse()程序才开始向目标网页发送Post请求
            Stream responseStream = response.GetResponseStream();
            //创建本地文件写入流
            Stream stream = new FileStream(path, FileMode.Create);
            byte[] bArr = new byte[1024];
            int size = responseStream.Read(bArr, 0, (int)bArr.Length);
            while (size > 0)
            {
                stream.Write(bArr, 0, size);
                size = responseStream.Read(bArr, 0, (int)bArr.Length);
            }
            stream.Close();
            responseStream.Close();

			// comfirm close and reopen
			MessageBox.Show(Messages.Instance.SystemMessage["UpdateEventDenSuccessed"]);

			c_RaidData.LoadEventRaidData();
			RefreshEventRaidData();
		}
		void RefreshEventRaidData()
		{
			f_StripMenuItemEventRaid.DropDownItems.Clear();
			m_MenuItemEventIdList.Clear();
			List<string> eventIdList = c_RaidData.GetEventRaidIdList();
			foreach (var id in eventIdList)
			{
				ToolStripMenuItem newEvent = new ToolStripMenuItem();
				newEvent.Name = id;
				newEvent.Size = new System.Drawing.Size(177, 22);
				newEvent.Text = id;
				newEvent.Click += new System.EventHandler(this.MenuItemEventIdSelect);

				f_StripMenuItemEventRaid.DropDownItems.Add(newEvent);
				m_MenuItemEventIdList.Add(newEvent);
			}
			f_StripMenuItemEventRaid.DropDownItems.Add(new ToolStripSeparator());
			// 更新ボタン
			if (m_MenuItemEventUpdate == null)
			{
				m_MenuItemEventUpdate = new ToolStripMenuItem();
				m_MenuItemEventUpdate.Name = "f_StripMenuItemUpdateEventData";
				m_MenuItemEventUpdate.Size = new System.Drawing.Size(177, 22);
				m_MenuItemEventUpdate.Click += new System.EventHandler(this.UpdateEventToolStripMenuItem_Click);
			}
			f_StripMenuItemEventRaid.DropDownItems.Add(m_MenuItemEventUpdate);

			RefreshEventId();
		}

		private void f_MenuItemGpuTest_Click(object sender, EventArgs e)
		{
			if (!IsGpuAvailable())
			{
				return;
			}
			// テスト用データ
			SeedSearcher searcher = new SeedSearcher(SeedSearcher.Mode.Cuda35);
			SeedSearcher.CudaInitialize();
			SeedSearcher.SetCudaCondition(0, 31, 27, 3, 19, 18, 31, 1, 15, 0, 3, false, 3, 2);
			SeedSearcher.SetCudaCondition(1, 31, 9, 31, 31, 15, 31, 1, 18, 0, 2, false, 3, 4);
			SeedSearcher.SetCudaCondition(2, 31, 31, 31, 1, 31, 27, 1,  6, 0, 2, true, 4, 4);
			SeedSearcher.SetCudaCondition(3, 31, 2, 23, 31, 31, 31, 1, 15, 0, 2, true, 4, 4);
			SeedSearcher.SetCudaTargetCondition6(27, 3, 19, 18, 9, 15);
			searcher.CudaLoopPartition = m_Preferences.GpuLoop + 1;

			SearchImpl(searcher, true, 0, 0xFA13E146EABEE283ul);
		}

#region GPUスレッド分割数設定
		private void f_MenuItemGpuLoop21_Click(object sender, EventArgs e)
		{
			if (f_MenuItemGpuLoop21.Checked == false)
			{
				foreach (var menuItem in m_MenuItemGpuLoopList)
				{
					menuItem.Checked = false;
				}
				m_Preferences.GpuLoop = 8;
				f_MenuItemGpuLoop21.Checked = true;
			}
		}

		private void f_MenuItemGpuLoop22_Click(object sender, EventArgs e)
		{
			if (f_MenuItemGpuLoop22.Checked == false)
			{
				foreach (var menuItem in m_MenuItemGpuLoopList)
				{
					menuItem.Checked = false;
				}
				m_Preferences.GpuLoop = 7;
				f_MenuItemGpuLoop22.Checked = true;
			}
		}

		private void f_MenuItemGpuLoop23_Click(object sender, EventArgs e)
		{
			if (f_MenuItemGpuLoop23.Checked == false)
			{
				foreach (var menuItem in m_MenuItemGpuLoopList)
				{
					menuItem.Checked = false;
				}
				m_Preferences.GpuLoop = 6;
				f_MenuItemGpuLoop23.Checked = true;
			}
		}

		private void f_MenuItemGpuLoop24_Click(object sender, EventArgs e)
		{
			if (f_MenuItemGpuLoop24.Checked == false)
			{
				foreach (var menuItem in m_MenuItemGpuLoopList)
				{
					menuItem.Checked = false;
				}
				m_Preferences.GpuLoop = 5;
				f_MenuItemGpuLoop24.Checked = true;
			}
		}

		private void f_MenuItemGpuLoop25_Click(object sender, EventArgs e)
		{
			if (f_MenuItemGpuLoop25.Checked == false)
			{
				foreach (var menuItem in m_MenuItemGpuLoopList)
				{
					menuItem.Checked = false;
				}
				m_Preferences.GpuLoop = 4;
				f_MenuItemGpuLoop25.Checked = true;
			}
		}

		private void f_MenuItemGpuLoop26_Click(object sender, EventArgs e)
		{
			if (f_MenuItemGpuLoop26.Checked == false)
			{
				foreach (var menuItem in m_MenuItemGpuLoopList)
				{
					menuItem.Checked = false;
				}
				m_Preferences.GpuLoop = 3;
				f_MenuItemGpuLoop26.Checked = true;
			}
		}

		private void f_MenuItemGpuLoop27_Click(object sender, EventArgs e)
		{
			if (f_MenuItemGpuLoop27.Checked == false)
			{
				foreach (var menuItem in m_MenuItemGpuLoopList)
				{
					menuItem.Checked = false;
				}
				m_Preferences.GpuLoop = 2;
				f_MenuItemGpuLoop27.Checked = true;
			}
		}

		private void f_MenuItemGpuLoop28_Click(object sender, EventArgs e)
		{
			if (f_MenuItemGpuLoop28.Checked == false)
			{
				foreach (var menuItem in m_MenuItemGpuLoopList)
				{
					menuItem.Checked = false;
				}
				m_Preferences.GpuLoop = 1;
				f_MenuItemGpuLoop28.Checked = true;
			}
		}

		private void f_MenuItemGpuLoop29_Click(object sender, EventArgs e)
		{
			if (f_MenuItemGpuLoop29.Checked == false)
			{
				foreach (var menuItem in m_MenuItemGpuLoopList)
				{
					menuItem.Checked = false;
				}
				m_Preferences.GpuLoop = 0;
				f_MenuItemGpuLoop29.Checked = true;
			}
		}
#endregion

		// ポケモン変更
#region ポケモンComboBox イベント定義
		private void f_ComboBoxPokemon_1_SelectedIndexChanged(object sender, EventArgs e)
		{
			RefreshAbility(0);
		}

		private void f_ComboBoxPokemon_2_SelectedIndexChanged(object sender, EventArgs e)
		{
			RefreshAbility(1);
		}

		private void f_ComboBoxPokemon_3_SelectedIndexChanged(object sender, EventArgs e)
		{
			RefreshAbility(2);
		}

		private void f_ComboBoxPokemon_351_SelectedIndexChanged(object sender, EventArgs e)
		{
			RefreshAbility(3);
		}

		private void f_ComboBoxPokemon_352_SelectedIndexChanged(object sender, EventArgs e)
		{
			RefreshAbility(4);
		}

		private void f_ComboBoxPokemon_353_SelectedIndexChanged(object sender, EventArgs e)
		{
			RefreshAbility(5);
		}
#endregion

		void RefreshAbility(int index)
		{
			RaidData.Pokemon pokemon = m_PokemonInfo[index].ComboBoxName.SelectedItem as RaidData.Pokemon;
			PersonalInfo info = PersonalTable.SWSH[pokemon.DataSpecies];
			if (info.ATK == 0)
			{
				info = PersonalTable.USUM[pokemon.DataSpecies];
			}
			var list = PKHeX.Core.Util.GetAbilitiesList(Messages.Instance.LangCode);

			m_PokemonInfo[index].ComboBoxAbility.Items.Clear();
			// 特性1=2
			if (info.Abilities[0] == info.Abilities[1])
			{
				m_PokemonInfo[index].ComboBoxAbility.Items.Add(new PokemonFormUtility.AbilityItem(-1, info.Abilities[0], list[info.Abilities[0]]));

				// 夢特性
				if (pokemon.Ability != 3 && info.Abilities[2] != info.Abilities[0])
				{
					m_PokemonInfo[index].ComboBoxAbility.Items.Add(new PokemonFormUtility.AbilityItem(2, info.Abilities[2], list[info.Abilities[2]]));
				}
			}
			else
			{
				m_PokemonInfo[index].ComboBoxAbility.Items.Add(new PokemonFormUtility.AbilityItem(0, info.Abilities[0], list[info.Abilities[0]]));
				m_PokemonInfo[index].ComboBoxAbility.Items.Add(new PokemonFormUtility.AbilityItem(1, info.Abilities[1], list[info.Abilities[1]]));

				// 夢特性
				if (pokemon.Ability != 3 && info.Abilities[2] != info.Abilities[0] && info.Abilities[2] != info.Abilities[1])
				{
					m_PokemonInfo[index].ComboBoxAbility.Items.Add(new PokemonFormUtility.AbilityItem(2, info.Abilities[2], list[info.Abilities[2]]));
				}
			}
			m_PokemonInfo[index].ComboBoxAbility.SelectedIndex = 0;
		}

#region 設定変更イベント
		private void f_MenuItemWindowSizeNormal_Click(object sender, EventArgs e)
		{
			MainForm.ActiveForm.Size = new System.Drawing.Size(845, 796);
		}

		private void f_MenuItemWindowSizeSmall_Click(object sender, EventArgs e)
		{
			MainForm.ActiveForm.Size = new System.Drawing.Size(860, 640);
		}

		private void f_MenuItemUseGpu_Click(object sender, EventArgs e)
		{
			bool current = f_MenuItemUseGpu.Checked;
			if (current == false)
			{
				// オンにする場合は使用可能チェック
				if (!IsGpuAvailable())
				{
					return;
				}
			}
			m_Preferences.IsUseGpu = !current;
			f_MenuItemUseGpu.Checked = !current;
		}

		private void f_CheckBoxShowResultTime_CheckedChanged(object sender, EventArgs e)
		{
			bool current = f_CheckBoxShowResultTime.Checked;
			if (m_Preferences.SearchShowDuration != current)
			{
				m_Preferences.SearchShowDuration = current;
			}
		}
		private void f_CheckBoxStop_CheckedChanged(object sender, EventArgs e)
		{
			bool current = f_CheckBoxStop.Checked;
			if (m_Preferences.SearchStop != current)
			{
				m_Preferences.SearchStop = current;
			}
		}

		private void f_CheckBoxListShiny_CheckedChanged(object sender, EventArgs e)
		{
			bool current = f_CheckBoxListShiny.Checked;
			if (m_Preferences.ListOnlyShiny != current)
			{
				m_Preferences.ListOnlyShiny = current;
			}
		}

		private void f_CheckBoxShowSeed_CheckedChanged(object sender, EventArgs e)
		{
			bool current = f_CheckBoxShowSeed.Checked;
			if (m_Preferences.ListShowSeed != current)
			{
				m_Preferences.ListShowSeed = current;
			}
		}

		private void f_CheckBoxShowEC_CheckedChanged(object sender, EventArgs e)
		{
			bool current = f_CheckBoxShowEC.Checked;
			if (m_Preferences.ListShowEC != current)
			{
				m_Preferences.ListShowEC = current;
			}
		}

		private void f_CheckBoxShowAbilityName_CheckedChanged(object sender, EventArgs e)
		{
			bool current = f_CheckBoxShowAbilityName.Checked;
			if (m_Preferences.ListShowAbilityName != current)
			{
				m_Preferences.ListShowAbilityName = current;
			}
		}
#endregion

	}
}
