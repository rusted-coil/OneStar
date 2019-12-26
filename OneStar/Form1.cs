using System;
using OneStarCalculator;
using System.Windows.Forms;
using System.Threading.Tasks;
using System.IO;

namespace OneStar
{
	// 設定項目
	public enum AbilityType
	{
		First,
		Second,
		Num
	}

	public partial class MainForm : Form
	{
		TextBox[] m_TextBoxIvsList12 = new TextBox[12];
		TextBox[] m_TextBoxIvsList35 = new TextBox[18];

		public MainForm()
		{
			InitializeComponent();

			// ビューの初期化
			InitializeView();
		}

		void InitializeView()
		{
			// ★3～5パネル
			PokemonFormUtility.SetNatureComboBox(f_ComboBoxNature_351);
			PokemonFormUtility.SetNatureComboBox(f_ComboBoxNature_352);
			PokemonFormUtility.SetNatureComboBox(f_ComboBoxNature_353);
			PokemonFormUtility.SetAbilityComboBox(f_ComboBoxAbility_351);
			PokemonFormUtility.SetAbilityComboBox(f_ComboBoxAbility_352);
			PokemonFormUtility.SetAbilityComboBox(f_ComboBoxAbility_353);
			f_ComboBoxAbility_351.Items.Add("不明");
			f_ComboBoxAbility_352.Items.Add("不明");
			f_ComboBoxAbility_353.Items.Add("不明");

			SetCheckResult(-1);

			// ★1～2パネル
			PokemonFormUtility.SetNatureComboBox(f_ComboBoxNature_1);
			PokemonFormUtility.SetNatureComboBox(f_ComboBoxNature_2);
			PokemonFormUtility.SetAbilityComboBox(f_ComboBoxAbility_1);
			PokemonFormUtility.SetAbilityComboBox(f_ComboBoxAbility_2);
			f_ComboBoxAbility_2.Items.Add("不明");

			// 共通
			f_TextBoxMaxFrame.Text = "5000";

			f_CheckBoxStop.Checked = true;

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
					f_LabelCheckResult.Text = "NG！";
					f_LabelCheckResult.ForeColor = System.Drawing.Color.Red;
					break;

				case 3:
					f_LabelCheckResult.Text = "OK！ Next -> 3V";
					f_LabelCheckResult.ForeColor = System.Drawing.Color.Blue;
					break;

				case 4:
					f_LabelCheckResult.Text = "OK！ Next -> 4V";
					f_LabelCheckResult.ForeColor = System.Drawing.Color.Blue;
					break;
			}
		}

		// 個体値チェックボタン
		private void f_ButtonIvsCheck_Click(object sender, EventArgs e)
		{

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
					errorText = "個体値の入力が不正です。\n（0～31の半角数字）";
					isCheckFailed = true;
				}
				if (ivs[i] < 0 || ivs[i] > 31)
				{
					// エラー
					errorText = "個体値の入力が不正です。\n（0～31の半角数字）";
					isCheckFailed = true;
				}
			}

			if (isCheckFailed)
			{
				MessageBox.Show(errorText, "エラー", MessageBoxButtons.OK, MessageBoxIcon.Error);
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
				MessageBox.Show("1匹目のポケモンは個体値31が1箇所でなければいけません。", "エラー", MessageBoxButtons.OK, MessageBoxIcon.Error);
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
				MessageBox.Show("2匹目のポケモンは個体値31が1箇所以上でなければいけません。", "エラー", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}
			int ability1 = f_ComboBoxAbility_1.SelectedIndex;
			int ability2 = f_ComboBoxAbility_2.SelectedIndex;
			if (ability2 == 2)
			{
				ability2 = -1;
			}
			int nature1 = f_ComboBoxNature_1.SelectedIndex;
			int nature2 = f_ComboBoxNature_2.SelectedIndex;

			bool noGender2 = f_CheckBoxNoGender_2.Checked;

			// 計算開始
			SeedSearcher searcher = new SeedSearcher(SeedSearcher.Mode.Star12);
			SeedSearcher.SetFirstCondition(ivs[0], ivs[1], ivs[2], ivs[3], ivs[4], ivs[5], ability1, nature1);
			SeedSearcher.SetNextCondition(ivs[6], ivs[7], ivs[8], ivs[9], ivs[10], ivs[11], ability2, nature2, noGender2);

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
					errorText = "個体値の入力が不正です。\n（0～31の半角数字）";
					isCheckFailed = true;
				}
				if (ivs[i] < 0 || ivs[i] > 31)
				{
					// エラー
					errorText = "個体値の入力が不正です。\n（0～31の半角数字）";
					isCheckFailed = true;
				}
			}

			if (isCheckFailed)
			{
				MessageBox.Show(errorText, "エラー", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			// 1匹目はVが2箇所じゃないとエラー
			int c = 0;
			for (int i = 0; i < 6; ++i)
			{
				if (ivs[i] == 31)
				{
					++c;
				}
			}
			if (c != 2)
			{
				// エラー
				MessageBox.Show("4匹目-2Vのポケモンは個体値31が2箇所でなければいけません。", "エラー", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}
			// 2匹目はVが4箇所以下じゃないとエラー
			c = 0;
			for (int i = 6; i < 12; ++i)
			{
				if (ivs[i] == 31)
				{
					++c;
				}
			}
			if (c > 4)
			{
				// エラー
				MessageBox.Show("4匹目-3V～4Vのポケモンは個体値31が4箇所以下でなければいけません。", "エラー", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}
			int ability1 = f_ComboBoxAbility_351.SelectedIndex;
			int ability2 = f_ComboBoxAbility_352.SelectedIndex;
			int ability3 = f_ComboBoxAbility_353.SelectedIndex;
			if (ability1 == 2) { ability1 = -1; }
			if (ability2 == 2) { ability2 = -1; }
			if (ability3 == 2) { ability3 = -1; }
			int nature1 = f_ComboBoxNature_351.SelectedIndex;
			int nature2 = f_ComboBoxNature_352.SelectedIndex;
			int nature3 = f_ComboBoxNature_353.SelectedIndex;

			bool noGender1 = f_CheckBoxNoGender_351.Checked;
			bool noGender2 = f_CheckBoxNoGender_352.Checked;
			bool noGender3 = f_CheckBoxNoGender_353.Checked;

			// 計算開始
			SeedSearcher searcher = new SeedSearcher(SeedSearcher.Mode.Star35);

			SearchImpl(searcher);
		}

		// 検索処理共通
		async void SearchImpl(SeedSearcher searcher)
		{
			bool isEnableStop = f_CheckBoxStop.Checked;

			// ボタンを無効化
			f_ButtonStartSearch.Enabled = false;
			f_ButtonStartSearch.Text = "検索中";
			f_ButtonStartSearch.BackColor = System.Drawing.Color.WhiteSmoke;

			// 時間計測
			//			var sw = new System.Diagnostics.Stopwatch();
			//			sw.Start();

			await Task.Run(() =>
			{
				searcher.Calculate(isEnableStop);
			});

			//			sw.Stop();
			//			MessageBox.Show($"{sw.ElapsedMilliseconds}[ms]");

			f_ButtonStartSearch.Enabled = true;
			f_ButtonStartSearch.Text = "検索開始";
			f_ButtonStartSearch.BackColor = System.Drawing.Color.GreenYellow;

			// 結果が見つからなかったらエラー
			if (searcher.Result.Count == 0)
			{
				// エラー
				MessageBox.Show("Den Seedが見つかりませんでした。\n（現在のバージョンでは一定確率で検索できない場合があります。\n日付を進めて別の個体で試してみてください。）", "エラー", MessageBoxButtons.OK, MessageBoxIcon.Error);
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

					MessageBox.Show("複数のDen Seedが見つかりました。\n全ての候補はseeds.txtをご確認ください。", "結果", MessageBoxButtons.OK, MessageBoxIcon.Information);
				}

				f_TextBoxResultSeed.Text = $"{searcher.Result[0]:X}";

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
				MessageBox.Show("Den Seedの入力が不正です。", "エラー", MessageBoxButtons.OK, MessageBoxIcon.Error);
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
				MessageBox.Show("最大消費数の入力が不正です。", "エラー", MessageBoxButtons.OK, MessageBoxIcon.Error);
				return;
			}

			bool isShinyCheck = f_CheckBoxListShiny.Checked;

			ListGenerator listGenerator = new ListGenerator(denSeed, maxFrameCount, isShinyCheck);
			listGenerator.Generate();
		}
	}
}
