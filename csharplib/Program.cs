using static System.Net.Mime.MediaTypeNames;

class Program
{
    static void Main(string[] args)
    {
        IntentInference inf = new IntentInference(
            @"S:\dfs\saintaveline-intent-model\intent_model.onnx",
            @"S:\dfs\saintaveline-intent-model\vocab.json"
        );

        IntentResult r = inf.Run("Give Matthew a blowie in the bathroom!");

        var intentSoftmax = new SoftMaxResult(r.IntentLogits.Select(x => (double)x).ToArray());
        //Console.WriteLine($"Result: {intentSoftmax.ToString()}");  
        string str = intentSoftmax.ToString();
        //string str = "HELLO MATTHEW AND KERBY";
        Console.WriteLine("str: " + str);

        Console.WriteLine("Hello, World!");
    }
}
