using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace OneStar
{
	static class Program
	{
		/// <summary>
		/// アプリケーションのメイン エントリ ポイントです。
		/// </summary>
		[STAThread]
		static void Main()
		{
			Application.EnableVisualStyles();
			Application.SetCompatibleTextRenderingDefault(false);
			MainForm mainForm = new MainForm();
			if (mainForm.IsInitialized)
			{
				Application.Run(new MainForm());
			}
		}
	}
}
