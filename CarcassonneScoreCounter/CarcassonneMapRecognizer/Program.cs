using System;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
using System.Drawing;

namespace CarcassonneMapRecognizer
{
    class Program
    {
        static void Main(string[] args)
        {
            // ona mne nuzhna
            Console.WriteLine("Hello World!");

            string path = @"F:\Hackathon\Sveta\";
            Image<Rgb, Byte> img1 = new Image<Rgb, Byte>(path + "map2.jpg");

            Image<Gray, Byte>[] channels = img1.Split();

            for (int i = 0; i < channels.Length; i++)
            {
                channels[i].Save(path + i.ToString() + ".jpg");
            }

            Image<Gray, Byte> grayImage = img1.Convert<Gray, Byte>();
            grayImage.Save(path + "gray.jpg");
            int grayThresh = 179;
            Image<Gray, Byte> grayBinary = grayImage.ThresholdBinary(new Gray(grayThresh), new Gray(255));
            grayBinary.Save(path + "gray_binary.jpg");

            //255 - grayImage

            int redThresh = 149;
            Image<Gray, Byte> redBinary = channels[0].ThresholdBinary(new Gray(redThresh), new Gray(255));
            redBinary.Save(path + "red_binary.jpg");

            int greenThresh = 180;
            // 150 - good for city and roads
            Image<Gray, Byte> greenBinary = channels[1].ThresholdBinary(new Gray(greenThresh), new Gray(255));
            greenBinary.Save(path + "green_binary.jpg");

            int blueThresh = 133;
            Image<Gray, Byte> blueBinary = channels[2].ThresholdBinary(new Gray(blueThresh), new Gray(255));
            blueBinary.Save(path + "blue_binary.jpg");

            Mat kernel1 = CvInvoke.GetStructuringElement(ElementShape.Cross, new Size(3, 3), new Point(1, 1));
            Image<Gray, Byte> blueBinaryOpening = blueBinary.MorphologyEx(MorphOp.Open, kernel1, new Point(-1, -1), 1, BorderType.Default, new MCvScalar());
            blueBinaryOpening.Save(path + "blue_binary_opening.jpg");

            Image<Gray, Byte> blueBinaryClosing = blueBinary.MorphologyEx(MorphOp.Close, kernel1, new Point(-1, -1), 1, BorderType.Default, new MCvScalar());
            blueBinaryClosing.Save(path + "blue_binary_closing.jpg");

            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            Mat hierarchy = new Mat();
            CvInvoke.FindContours(
                blueBinary,
                contours,
                hierarchy,
                RetrType.Ccomp,
                ChainApproxMethod.ChainApproxSimple
                );

            Image<Rgb, Byte> blueContours = channels[2].Convert<Rgb, Byte>();

            //for (; contours != null; contours = contours.HNext)
            //{
            //    Contour<Point> currentContour = contours.ApproxPoly(contours.Perimeter * 0.05, storage);
            //    //Seq<Point> currentContour = contours.GetConvexHull(Emgu.CV.CvEnum.ORIENTATION.CV_CLOCKWISE);

            //    if (contours.Area > MinRectangleArea) //only consider contours with area greater than 20000
            //    {
            //        if (currentContour.Total == 4) //The contour has 4 vertices.
            //        {
            //            bool isRectangle = true;
            //            Point[] pts = currentContour.ToArray();
            //            LineSegment2D[] edges = PointCollection.PolyLine(pts, true);

            //            for (int i = 0; i < edges.Length; i++)
            //            {
            //                double angle = Math.Abs(edges[(i + 1) % edges.Length].GetExteriorAngleDegree(edges[i]));
            //                if (angle < 90 - RectangleAngleMargin || angle > RectangleAngleMargin + 90)
            //                {
            //                    isRectangle = false;
            //                    break;
            //                }
            //            }

            //            if (isRectangle)
            //            {
            //                boxList.Add(currentContour.GetMinAreaRect());
            //            }
            //        }
            //    }
            //}
            CvInvoke.DrawContours(blueContours, contours, -1, new MCvScalar(0, 0, 255));
            blueContours.Save(path + "blue_contours.jpg");

            Console.WriteLine("contours.Size: " + contours.Size);
            Console.WriteLine("hierarchy.Rows: " + hierarchy.Rows);
            Console.WriteLine("hierarchy.Cols: " + hierarchy.Cols);
            Console.WriteLine("hierarchy.Depth: " + hierarchy.Depth);
            Console.WriteLine("hierarchy.NumberOfChannels: " + hierarchy.NumberOfChannels);


            int blueThreshRoad = 160;
            Image<Gray, Byte> blueBinaryRoad = channels[2].ThresholdBinary(new Gray(blueThreshRoad), new Gray(255));
            blueBinaryRoad.Save(path + "blue_binary_road.jpg");

            int blueThreshRoad2 = 170;
            Image<Gray, Byte> blueBinaryRoad2 = channels[2].ThresholdBinary(new Gray(blueThreshRoad2), new Gray(255));
            blueBinaryRoad2.Save(path + "blue_binary_road" + blueThreshRoad2.ToString() + ".jpg");



            Image<Gray, Byte> cityBinary = redBinary - blueBinary;
            cityBinary.Save(path + "city_binary.jpg");
            //ImageBox imageBox = new ImageBox();

            //Mat matImage = capture.QueryFrame();
            //CamImageBox.Image = matImage; // Directly show Mat object in *ImageBox*
            //Image<Bgr, byte> iplImage = matImage.ToImage<Bgr, byte>();
            //imageBox.Image = img1; // Show Image<,> object in *ImageBox*


            Image<Gray, byte> roads = channels[2].Copy();

            //var temp = redBinary.Cmp(greenBinary, CmpType.LessThan);
            //var res = redBinary.Cmp(blueBinary, CmpType.LessThan);

            var res = blueBinary.Clone();
            res._Mul(redBinary);
            res._Mul(greenBinary);

            Image<Gray, Byte> resOpening = res.MorphologyEx(MorphOp.Close, kernel1, new Point(-1, -1), 1, BorderType.Default, new MCvScalar());
            resOpening.Save(path + "res_opening.jpg");


            //temp.Save(path + "temp.jpg");
            res.Save(path + "res.jpg");
        }
    }
}
