using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

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
