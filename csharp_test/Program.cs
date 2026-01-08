using static System.Net.Mime.MediaTypeNames;

class Program
{
    static void Main(string[] args)
    {
        IntentInference inf = new IntentInference(
            @"S:\dfs\saintaveline-intent-model\intent_model.onnx",
            @"S:\dfs\saintaveline-intent-model\vocab.json"
        );

        IntentResult r = inf.Run("hurry up and go to the roof vent now");

        Console.WriteLine("Hello, World!");
    }
}
