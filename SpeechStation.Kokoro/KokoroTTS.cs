using Microsoft.ML;
using Microsoft.ML.Data;

namespace SpeechStation.Kokoro;

public class Kokoro
{
    const string ONNX_MODEL_PATH = "weights/onnx/model.onnx";
    public record OnnxInput([property: ColumnName("input_ids")][property: VectorType(1, MAX_LENGTH)] long[] InputIds, [property: ColumnName("style")][property: VectorType(1, VOICE_DIM)] float[] Style, [property: ColumnName("speed")] float Speed)
    {
        public OnnxInput() : this(
            [],
            [],
            1.0f
        )
        { }

        public OnnxInput(long[] inputIds, Voice voice, float speed = 1f) : this([0, .. inputIds, 0], voice.GetRefS(inputIds.Length), speed) { }
    }
    const int MAX_LENGTH = 512;
    const int VOICE_DIM = 256;


    public record OnnxOutput
    {
        [ColumnName("waveform")]
        public float[] Waveform { get; set; } = [];
    }

    public record Voice(string Name, float[] Data)
    {
        internal float[] GetRefS(int length)
        {
            if (length > MAX_LENGTH)
            {
                throw new Exception($"Length should be less than {MAX_LENGTH}");
            }
            return Data.AsSpan(length * VOICE_DIM, VOICE_DIM).ToArray();
        }
    }
    public static Voice LoadVoice(string name)
    {
        var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "weights", "voices", name + ".bin");

        var bytes = File.ReadAllBytes(path);
        // using var file = File.OpenRead(path);

        // var voice = new float[512, 256];
        // var vs = voice.AsSpan2D();

        var data = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(bytes).ToArray();
        // should be 256 dimensional

        if (data.Length != VOICE_DIM * MAX_LENGTH)
        {
            throw new Exception($"Voice data should be {VOICE_DIM * MAX_LENGTH} dimensional, but got {data.Length}");
        }

        return new Voice(name, data);

    }
    public static ITransformer GetPredictionPipeline(MLContext mlContext)
    {
        var inputColumnNames = new string[]
        {
    "input_ids", "style", "speed"
        };

        var outputColumnNames = new string[] { "waveform" };
        var modelFile = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, ONNX_MODEL_PATH);
        var onnxPredictionPipeline = mlContext
        .Transforms
        .ApplyOnnxModel(
            outputColumnNames: outputColumnNames,
            inputColumnNames: inputColumnNames,
            modelFile: modelFile);

        var emptyDv = mlContext.Data.LoadFromEnumerable(Array.Empty<OnnxInput>());

        return onnxPredictionPipeline.Fit(emptyDv);
    }
}

// var voice = Kokoro.LoadVoice("af");

// MLContext mlContext = new MLContext();
// var onnxPredictionPipeline = Kokoro.GetPredictionPipeline(mlContext);
// var onnxPredictionEngine = mlContext.Model.CreatePredictionEngine<Kokoro.OnnxInput, Kokoro.OnnxOutput>(onnxPredictionPipeline);

// var input = new Kokoro.OnnxInput(tokens, voice, 1.0f);

// var prediction = onnxPredictionEngine.Predict(input);

// Console.WriteLine(prediction.Waveform.Length);