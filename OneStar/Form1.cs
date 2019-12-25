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
		TextBox[] m_TextBoxIvsList = new TextBox[12];

		public MainForm()
		{
			InitializeComponent();

			// ビューの初期化
			InitializeView();
		}

		void InitializeView()
		{
			PokemonFormUtility.SetNatureComboBox(f_ComboBoxNature_1);
			PokemonFormUtility.SetNatureComboBox(f_ComboBoxNature_2);
			PokemonFormUtility.SetAbilityComboBox(f_ComboBoxAbility_1);
			PokemonFormUtility.SetAbilityComboBox(f_ComboBoxAbility_2);
			f_ComboBoxAbility_2.Items.Add("不明");

			f_TextBoxMaxFrame.Text = "1000";

			f_CheckBoxStop.Checked = true;

			// 扱いやすいようにキャッシュ
			m_TextBoxIvsList[0] = f_TextBoxIv0_1;
			m_TextBoxIvsList[1] = f_TextBoxIv1_1;
			m_TextBoxIvsList[2] = f_TextBoxIv2_1;
			m_TextBoxIvsList[3] = f_TextBoxIv3_1;
			m_TextBoxIvsList[4] = f_TextBoxIv4_1;
			m_TextBoxIvsList[5] = f_TextBoxIv5_1;
			m_TextBoxIvsList[6] = f_TextBoxIv0_2;
			m_TextBoxIvsList[7] = f_TextBoxIv1_2;
			m_TextBoxIvsList[8] = f_TextBoxIv2_2;
			m_TextBoxIvsList[9] = f_TextBoxIv3_2;
			m_TextBoxIvsList[10] = f_TextBoxIv4_2;
			m_TextBoxIvsList[11] = f_TextBoxIv5_2;
		}

		private void ButtonStartSearch_Click(object sender, EventArgs e)
		{
			SeedSeach();
		}

		async void SeedSeach()
		{
			bool isCheckFailed = false;
			string errorText = "";

			// フォームから必要な情報を取得
			int[] ivs = new int[12];
			for (int i = 0; i < 12; ++i)
			{
				try
				{
					ivs[i] = int.Parse(m_TextBoxIvsList[i].Text);
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

			bool isEnableStop = f_CheckBoxStop.Checked;

			// 計算開始
			SeedSearcher searcher = new SeedSearcher();
			SeedSearcher.SetFirstCondition(ivs[0], ivs[1], ivs[2], ivs[3], ivs[4], ivs[5], ability1, nature1);
			SeedSearcher.SetNextCondition(ivs[6], ivs[7], ivs[8], ivs[9], ivs[10], ivs[11], ability2, nature2, noGender2);

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
