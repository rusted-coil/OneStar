namespace OneStar
{
	partial class Form2
	{
		/// <summary>
		/// Required designer variable.
		/// </summary>
		private System.ComponentModel.IContainer components = null;

		/// <summary>
		/// Clean up any resources being used.
		/// </summary>
		/// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
		protected override void Dispose(bool disposing)
		{
			if (disposing && (components != null))
			{
				components.Dispose();
			}
			base.Dispose(disposing);
		}

		#region Windows Form Designer generated code

		/// <summary>
		/// Required method for Designer support - do not modify
		/// the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			this.DenMap = new System.Windows.Forms.PictureBox();
			this.f_PicturePoint = new System.Windows.Forms.PictureBox();
			((System.ComponentModel.ISupportInitialize)(this.DenMap)).BeginInit();
			((System.ComponentModel.ISupportInitialize)(this.f_PicturePoint)).BeginInit();
			this.SuspendLayout();
			// 
			// DenMap
			// 
			this.DenMap.BackgroundImage = global::OneStar.Properties.Resources.map2;
			this.DenMap.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Zoom;
			this.DenMap.Location = new System.Drawing.Point(11, 11);
			this.DenMap.Margin = new System.Windows.Forms.Padding(2);
			this.DenMap.Name = "DenMap";
			this.DenMap.Size = new System.Drawing.Size(441, 428);
			this.DenMap.TabIndex = 1;
			this.DenMap.TabStop = false;
			// 
			// f_PicturePoint
			// 
			this.f_PicturePoint.BackColor = System.Drawing.Color.Transparent;
			this.f_PicturePoint.BackgroundImageLayout = System.Windows.Forms.ImageLayout.None;
			this.f_PicturePoint.Image = global::OneStar.Properties.Resources.point;
			this.f_PicturePoint.Location = new System.Drawing.Point(52, 37);
			this.f_PicturePoint.Name = "f_PicturePoint";
			this.f_PicturePoint.Size = new System.Drawing.Size(10, 10);
			this.f_PicturePoint.TabIndex = 28;
			this.f_PicturePoint.TabStop = false;
			// 
			// Form2
			// 
			this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 12F);
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.ClientSize = new System.Drawing.Size(459, 450);
			this.Controls.Add(this.f_PicturePoint);
			this.Controls.Add(this.DenMap);
			this.Name = "Form2";
			this.Text = "Form2";
			((System.ComponentModel.ISupportInitialize)(this.DenMap)).EndInit();
			((System.ComponentModel.ISupportInitialize)(this.f_PicturePoint)).EndInit();
			this.ResumeLayout(false);

		}

		#endregion

		private System.Windows.Forms.PictureBox DenMap;
		private System.Windows.Forms.PictureBox f_PicturePoint;
	}
}