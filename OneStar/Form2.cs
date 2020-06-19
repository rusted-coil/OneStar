using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace OneStar
{
	public partial class Form2 : Form
	{
		public Form2()
		{
			InitializeComponent();

			f_PicturePoint.Parent = DenMap;
			f_PicturePoint.Visible = false;
		}

		public void SetPoint(System.Drawing.Point location)
		{
			f_PicturePoint.Location = new System.Drawing.Point(location.X - 5, location.Y - 5);
			f_PicturePoint.Visible = true;
		}
	}
}
