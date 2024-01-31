using MLModel1_ConsoleApp1;
using Microsoft.ML.Data;
using Microsoft.ML;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using System;
using System.Drawing;
using System.IO;
using System.Linq;

var vc = new VideoCapture(0, Emgu.CV.VideoCapture.API.DShow);
MLContext mlContext = new MLContext();
Mat frame = new();
Mat frameGray = new();

while (true)
{
    vc.Read(frame);
    var bitmap = Emgu.CV.BitmapExtension.ToBitmap(frame);

    using (MemoryStream stream = new MemoryStream())
    {
        bitmap.Save(stream, System.Drawing.Imaging.ImageFormat.Bmp);
        stream.Seek(0, SeekOrigin.Begin);

        var mlImage = MLImage.CreateFromStream(stream);
        MLModel1.ModelInput sampleData3 = new MLModel1.ModelInput()
        {
            Image = mlImage,
        };

        var predictionResult2 = MLModel1.Predict(sampleData3);

        if (predictionResult2.PredictedBoundingBoxes == null)
        {
            Console.WriteLine("No Predicted Bounding Boxes");           
        }
        else
        {
            var boxes3 =
            predictionResult2.PredictedBoundingBoxes.Chunk(4)
                .Select(x => new { XTop = x[0], YTop = x[1], XBottom = x[2], YBottom = x[3] })
                .Zip(predictionResult2.Score, (a, b) => new { Box = a, Score = b });

            foreach (var item in boxes3)
            {
                CvInvoke.Rectangle(frame, new Rectangle((int)item.Box.XTop, (int)item.Box.YTop, (int)(item.Box.XBottom - item.Box.XTop), (int)(item.Box.YBottom - item.Box.YTop)), new MCvScalar(0, 0, 255), 2);
                Console.WriteLine($"XTop: {item.Box.XTop},YTop: {item.Box.YTop},XBottom: {item.Box.XBottom},YBottom: {item.Box.YBottom}, Score: {item.Score}");
            }
        }        
    }

    CvInvoke.Imshow("face detection", frame);

    if (CvInvoke.WaitKey(1) == 27)
    {
        break;
    }
}
