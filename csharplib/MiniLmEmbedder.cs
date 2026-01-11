using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;

public sealed class MiniLmEmbedder : IDisposable
{
    private readonly InferenceSession _session;
    private readonly WordPieceTokenizer _tokenizer;
    private readonly int _maxLength;

    public MiniLmEmbedder(string modelPath, string vocabPath, int maxLength = 256)
    {
        _session = new InferenceSession(modelPath);
        _tokenizer = WordPieceTokenizer.Create(vocabPath);
        _maxLength = maxLength;
    }

    public float[] Embed(string text)
    {
        // AI: Tokenize
        IReadOnlyList<int> ids = _tokenizer.EncodeToIds(text);

        // AI: Build input_ids + attention_mask (BERT style)
        // Many MiniLM sentence-transformer ONNX models expect these names:
        // input_ids, attention_mask, token_type_ids (sometimes optional).
        var inputIds = new long[_maxLength];
        var attention = new long[_maxLength];
        var tokenTypeIds = new long[_maxLength];

        int length = Math.Min(ids.Count, _maxLength);

        for (int i = 0; i < length; i++)
        {
            inputIds[i] = ids[i];
            attention[i] = 1;
            tokenTypeIds[i] = 0;
        }

        var inputIdsTensor = new DenseTensor<long>(inputIds, new[] { 1, _maxLength });
        var attentionTensor = new DenseTensor<long>(attention, new[] { 1, _maxLength });
        var tokenTypeTensor = new DenseTensor<long>(tokenTypeIds, new[] { 1, _maxLength });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionTensor),

            // AI: Some models require token_type_ids; some ignore it.
            NamedOnnxValue.CreateFromTensor("token_type_ids", tokenTypeTensor)
        };

        using var results = _session.Run(inputs);

        // AI: If this ONNX graph includes pooling+normalize, output is typically [1, 384]
        var output = results.First().AsEnumerable<float>().ToArray();
        return output;
    }

    public void TestFunction()
    {
        var ids = _tokenizer.EncodeToIds("the cat is sleeping");
        Console.WriteLine(string.Join(", ", ids));

        ids = _tokenizer.EncodeToIds("the dog is sleeping");
        Console.WriteLine(string.Join(", ", ids));
    }

    public void Dispose()
    {
        _session.Dispose();
    }
}
