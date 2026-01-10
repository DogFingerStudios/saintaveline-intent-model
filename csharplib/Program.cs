using Microsoft.ML.Tokenizers;
using static System.Net.Mime.MediaTypeNames;

class Program
{
    //const string DATA_PATH = @"/Users/addy/src/dfs/saintaveline-intent-model/data";
    const string DATA_PATH = @"S:\dfs\saintaveline-intent-model\data";

    //static void Main(string[] args)
    //{
    //    IntentInference inf = new IntentInference(
    //    IntentInference inf = new IntentInference(
    //        MODEL_PATH + Path.DirectorySeparatorChar + "intent_model.onnx",
    //        MODEL_PATH + Path.DirectorySeparatorChar + "vocab.json"
    //    );

    //    IntentResult r = inf.Run("Walk to Central Park now!");

    //    // var intentSoftmax = new SoftMaxResult([.. r.IntentLogits.Select(x => (double)x)]); // C#11 or greater
    //    var intentSoftmax = new SoftMaxResult(r.IntentLogits.Select(x => (double)x).ToArray());
    //    Console.WriteLine("Intent: " + intentSoftmax.ToString());

    //    var speedSoftmax = new SoftMaxResult(r.SpeedLogits.Select(x => (double)x).ToArray());
    //    Console.WriteLine("Speed: " + speedSoftmax.ToString());

    //    var urgencySoftmax = new SoftMaxResult(r.UrgencyLogits.Select(x => (double)x).ToArray());
    //    Console.WriteLine("Urgency: " + urgencySoftmax.ToString());

    //    Console.WriteLine("Hello, World!");
    //}

    static void Main()
    {
        string modelPath = Path.Combine(DATA_PATH, "MiniML", "model.onnx");
        string vocabPath = Path.Combine(DATA_PATH, "MiniML", "vocab.txt");

        using var embedder = new MiniLmEmbedder(modelPath, vocabPath);
        embedder.TestFunction();


        //float[] a = embedder.Embed("Go to the 5th floor bathroom");
        //float[] b = embedder.Embed("Head to the bathroom on the fifth floor");
        //float[] c = embedder.Embed("Follow Dad");

        //Console.WriteLine($"Dim: {a.Length}");

        //Console.WriteLine($"sim(a,b)={Cosine(a, b):F4}");
        //Console.WriteLine($"sim(a,c)={Cosine(a, c):F4}");

        Console.WriteLine("Hello, World!");
    }

    static float Cosine(float[] x, float[] y)
    {
        float dot = 0f;
        float nx = 0f;
        float ny = 0f;

        for (int i = 0; i < x.Length; i++)
        {
            dot += x[i] * y[i];
            nx += x[i] * x[i];
            ny += y[i] * y[i];
        }

        return dot / ((float)Math.Sqrt(nx) * (float)Math.Sqrt(ny));
    }
}
