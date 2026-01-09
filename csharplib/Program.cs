using static System.Net.Mime.MediaTypeNames;

class Program
{
    const string MODEL_PATH = @"/Users/addy/src/dfs/saintaveline-intent-model/data";
    // const string MODEL_PATH = @"S:\dfs\saintaveline-intent-model";
    
    static void Main(string[] args)
    {
        IntentInference inf = new IntentInference(
            MODEL_PATH + Path.DirectorySeparatorChar + "intent_model.onnx",
            MODEL_PATH + Path.DirectorySeparatorChar + "vocab.json"
        );

        IntentResult r = inf.Run("Walk to Central Park now!");

        // var intentSoftmax = new SoftMaxResult([.. r.IntentLogits.Select(x => (double)x)]); // C#11 or greater
        var intentSoftmax = new SoftMaxResult(r.IntentLogits.Select(x => (double)x).ToArray());
        Console.WriteLine("Intent: " + intentSoftmax.ToString());

        var speedSoftmax = new SoftMaxResult(r.SpeedLogits.Select(x => (double)x).ToArray());
        Console.WriteLine("Speed: " + speedSoftmax.ToString());
        
        var urgencySoftmax = new SoftMaxResult(r.UrgencyLogits.Select(x => (double)x).ToArray());
        Console.WriteLine("Urgency: " + urgencySoftmax.ToString());

        Console.WriteLine("Hello, World!");
    }
}
