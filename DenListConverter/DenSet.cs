using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DenListConverter
{
	public class DenSet
	{
		public Dictionary<string, Nest> NestDictionary { get; }  = new Dictionary<string, Nest>();

		public void Load(string path)
		{
			using (StreamReader sr = new StreamReader(path))
			{
				string line;
				Nest nest = null;
				List<string> nestLines = new List<string>();

				while ((line = sr.ReadLine()) != null)
				{
					// NestID定義
					if (line.IndexOf("Nest ID:") == 0)
					{
						if (nest != null)
						{
							nest.Load(nestLines);
							NestDictionary.Add(nest.Id, nest);
							nestLines.Clear();
						}

						string[] elements = line.Split(' ');
						string id = elements[2];
						nest = new Nest(id);

						continue;
					}
					nestLines.Add(line);
				}
				if (nest != null)
				{
					nest.Load(nestLines);
					NestDictionary.Add(nest.Id, nest);
				}
			}
		}
	}
}
