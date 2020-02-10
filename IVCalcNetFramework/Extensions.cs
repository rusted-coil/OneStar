namespace IVCalcNetFramework
{
    public static class Extensions
    {
        public static decimal ToDecimal(this string text)
        {
            return decimal.Parse(text, new System.Globalization.CultureInfo("en-US"));
        }
    }
}
