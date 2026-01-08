using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.IO;
using System.Text.Json;

public class SoftMaxResult
{
    public IReadOnlyList<double> RawScores { get; }
    public IReadOnlyList<double> Probabilities { get; }
    public int PredictedClassIndex { get; }
    public double MaxProbability { get; }
    public double MaxRawScore { get; }
    public IReadOnlyList<(int Index, double Probability)> SortedProbabilities { get; }

    public SoftMaxResult(IReadOnlyList<double> rawScores)
    {
        this.RawScores = rawScores;
        this.MaxRawScore = RawScores.Max();

        // Subtract max for numerical stability. This prevents overflow in exp.
        var expScores = rawScores.Select(score => Math.Exp(score - MaxRawScore)).ToList();
        var sumExpScores = expScores.Sum();

        this.Probabilities = expScores.Select(expScore => expScore / sumExpScores).ToList();
        this.MaxProbability = this.Probabilities.Max();
        this.PredictedClassIndex = this.Probabilities.Select((value, index) => (value, index)).Max().index;

        this.SortedProbabilities = this.Probabilities
            .Select((value, index) => (Index: index, Probability: value))
            .OrderByDescending(item => item.Probability)
            .ToList();
    }

    public override string ToString()
    {
        // Number of top entries to show
        int topN = Math.Min(3, SortedProbabilities.Count);

        // Build top entries string
        var topEntries = SortedProbabilities
            .Take(topN)
            .Select((item, rank) => $"{rank + 1}. idx={item.Index}, prob={item.Probability:P2}")
            .ToArray();
        string topPart = topEntries.Length > 0 ? string.Join(", ", topEntries) : "none";

        // If there are more classes, indicate total count
        string moreInfo = SortedProbabilities.Count > topN
            ? $" (showing top {topN} of {SortedProbabilities.Count})"
            : string.Empty;

        return $"Predicted: {PredictedClassIndex} ({MaxProbability:P2}), RawMax: {MaxRawScore:G6}\nTop: {topPart}{moreInfo}";
    }
}

public class IntentResult
{
    public float[] IntentLogits;
    public float[] SpeedLogits;
    public float[] UrgencyLogits;

    public IntentResult(float[] intent, float[] speed, float[] urgency)
    {
        IntentLogits = intent;
        SpeedLogits = speed;
        UrgencyLogits = urgency;
    }

    private float _softSum = -1;

    List<float[]> SoftMaxResults()
    {
        List<float[]> result = new List<float[]>();

        _softSum = IntentLogits.Select(x => (float)Math.Pow(Math.E, x)).Sum();
        var intentSoftmax = IntentLogits.Select(x => (float)(Math.Pow(Math.E, x) / _softSum)).ToList();
        var score = intentSoftmax.Max();

        return result;
    }
}

public class IntentInference
{
    private InferenceSession _session;
    private Dictionary<string, int> _vocab;

    public IntentInference(string modelPath, string vocabPath)
    {
        _session = new InferenceSession(modelPath);
        _vocab = LoadVocab(vocabPath);
    }

    private Dictionary<string, int> LoadVocab(string path)
    {
        string json = File.ReadAllText(path);
        return JsonSerializer.Deserialize<Dictionary<string, int>>(json) ?? new Dictionary<string, int>();
    }

    // AI: Convert text → bag-of-words vector
    private float[] Vectorize(string text)
    {
        float[] vec = new float[_vocab.Count];
        string[] tokens = text.ToLower().Split(' ', StringSplitOptions.RemoveEmptyEntries);

        foreach (string token in tokens)
        {
            if (_vocab.TryGetValue(token, out int idx))
            {
                vec[idx] += 1.0f;
            }
        }

        return vec;
    }

    public IntentResult Run(string text)
    {
        float[] bow = Vectorize(text);

        DenseTensor<float> input = new DenseTensor<float>(
            bow,
            new int[] { 1, bow.Length }
        );

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("bow_input", input)
        };

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(inputs);

        float[] intent = results[0].AsEnumerable<float>().ToArray();
        float[] speed = results[1].AsEnumerable<float>().ToArray();
        float[] urgency = results[2].AsEnumerable<float>().ToArray();

        return new IntentResult(intent, speed, urgency);
    }
}
