using System;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
using System.Drawing;
using System.Collections.Generic;

namespace CarcassonneMapRecognizer
{
    class Program
    {
        static void Main(string[] args)
        {
            // ona mne nuzhna
            Console.WriteLine("We can!");

            string path = @"F:\Hackathon\Sveta\";
            Image<Rgb, Byte> img1 = new Image<Rgb, Byte>(path + "map4.jpg");

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

            Image<Gray, Byte> grayGaussBinary = grayImage.ThresholdBinary(new Gray(grayThresh), new Gray(255));

            //CvInvoke.cvSmooth(grayImage, grayGaussBinary, SMOOTH_TYPE.CV_GAUSSIAN, 13, 13, 1.5, 0);
            CvInvoke.GaussianBlur(grayImage, grayGaussBinary, new Size(7, 7), 1.5, 0);
            grayGaussBinary.Save(path + "gray_gauss_binary.jpg");
            Image<Gray, Byte> cannyImg = grayImage.Clone();
            CvInvoke.Canny(channels[2], cannyImg, 71, 141);
            cannyImg.Save(path + "canny.jpg");
            //255 - grayImage

            int redThresh = 149;
            Image<Gray, Byte> redBinary = channels[0].ThresholdBinary(new Gray(redThresh), new Gray(255));
            redBinary.Save(path + "red_binary.jpg");

            int edgeThresh = 90;
            Image<Gray, Byte> redBinaryEdges  = channels[0].ThresholdBinary(new Gray(edgeThresh), new Gray(255));
            redBinaryEdges.Save(path + "edges_red_binary.jpg");
            edgeThresh = 90;
            Image<Gray, Byte> greenBinaryEdges = channels[1].ThresholdBinary(new Gray(edgeThresh), new Gray(255));
            greenBinaryEdges.Save(path + "edges_green_binary.jpg");
            edgeThresh = 80;
            Image<Gray, Byte> blueBinaryEdges = channels[2].ThresholdBinary(new Gray(edgeThresh), new Gray(255));
            blueBinaryEdges.Save(path + "edges_blue_binary.jpg");

            //nikita's govno-kod
            var redBinaryEdgesInv = 255 - redBinaryEdges;
            var greenBinaryEdgesInv = 255 - greenBinaryEdges;
            var blueBinaryEdgesInv = 255 - blueBinaryEdges;

            var resNikita = redBinaryEdgesInv.Clone();

            resNikita._Mul(greenBinaryEdgesInv);
            resNikita._Mul(blueBinaryEdgesInv);

            resNikita = 255 - resNikita;

            resNikita.Save(path + "edges_res.jpg");

            Mat kernel2 = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(31, 31), new Point(1, 1));
            Image<Gray, Byte> resGrad = resNikita.MorphologyEx(MorphOp.Gradient, kernel2, new Point(-1, -1), 1, BorderType.Default, new MCvScalar());
            resGrad.Save(path + "edges_res_grad.jpg");
  



            int greenThresh = 180;
            // 150 - good for city and roads
            Image<Gray, Byte> greenBinary = channels[1].ThresholdBinary(new Gray(greenThresh), new Gray(255));
            greenBinary.Save(path + "green_binary.jpg");

            int blueThresh = 110;
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

            var greenMinusRed = ThresholdByDelta(channels[1], channels[0], 25);
            var greenMinusBlue = ThresholdByDelta(channels[1], channels[2], 25);
            var blueMinusRed = ThresholdByDelta(channels[2], channels[0], 25);

            var strongGreen = channels[1].ThresholdBinary(new Gray(63), new Gray(255));
            var tryRes = strongGreen;// IntersectWhite(greenMinusRed, greenMinusBlue, blueMinusRed, strongGreen);
            tryRes.Save(path + "tryRes.jpg");

            var tryResRoadsCities = IntersectWhite(greenMinusRed, greenMinusBlue, blueMinusRed, strongGreen);
            tryResRoadsCities.Save(path + "tryResRoadsCities.jpg");

            var tryResCities = tryResRoadsCities - blueBinary;
            tryResCities.Save(path + "tryResCities.jpg");

            var greenMinusRed6 = ThresholdInvByDelta(channels[1], channels[0], 25);
            var greenMinusBlue6 = ThresholdInvByDelta(channels[1], channels[2], 25);
            var blueMinusRed6 = ThresholdInvByDelta(channels[2], channels[0], 25);
            var tryRes6 = IntersectWhite(greenMinusRed6, greenMinusBlue6, blueMinusRed6);
            tryRes6.Save(path + "tryRes6.jpg");

            //var greenMinusRed3 = ThresholdByDelta(channels[1], channels[0], 10);
            //var greenMinusBlue3 = ThresholdByDelta(channels[1], channels[2], 25);
            //var blueMinusRed3 = ThresholdByDelta(channels[2], channels[0], 25);
            //var tryRes4 = IntersectWhite(greenMinusRed3, greenMinusBlue3, blueMinusRed3);
            //tryRes4.Save(path + "tryRes123123123.jpg");

            var redMinusBlue2 = ChannelDominatingWithThreshold(channels[0], channels[2], 50);
            var greenMinusBlue2 = ChannelDominatingWithThreshold(channels[1], channels[2], 15);
            var redMinusGreen2 = ChannelDominatingWithThreshold(channels[0], channels[1], 1);

            var tryRes2 = IntersectWhite(redMinusBlue2, greenMinusBlue2, redMinusGreen2, strongGreen);
            tryRes2.Save(path + "tryRes2.jpg");

            var tryRes3 = tryRes2;

            tryRes3.Save(path + "tryRes3.jpg");

            var df_tmp = tryRes + tryRes3;

            df_tmp.Save(path + "df_tmp.jpg");

            //new logic))))
            

            //endNewLogic




            VectorOfVectorOfPoint contours1 = new VectorOfVectorOfPoint();
            Mat hierarchy1 = new Mat();
            CvInvoke.FindContours(
                df_tmp,
                contours1,
                hierarchy1,
                RetrType.External,
                ChainApproxMethod.ChainApproxSimple
                );

            List<Rectangle> contoursCleaned = new List<Rectangle>();
            Image<Rgb, Byte> df1 = df_tmp.Clone().Convert<Rgb, Byte>();

            for (int i = 0; i < contours1.Size; i++)
            {
                Rectangle r = CvInvoke.BoundingRectangle(contours1[i]);

                var area = r.Height * r.Width;
                if (area > 100000)
                {
                    contoursCleaned.Add(r);
                    CvInvoke.Rectangle(df1, r, new MCvScalar(255, 0, 255), 3);
                }
            }

            //, LineType.EightConnected, hierarchy1);
            df1.Save(path + "df1.jpg");


            //Image<Rgb, byte> resGradCopy = tryRes3.Clone().Convert<Rgb, Byte>();
            //LineSegment2D[] lines = CvInvoke.HoughLinesP(df_tmp,
            //    1, //Distance resolution in pixel-related units
            //    MathF.PI, //Angle resolution measured in radians.
            //    20, //threshold
            //    350, //min Line width
            //    0); //gap between lines

            //foreach (var line in lines)
            //{
            //    CvInvoke.Line(resGradCopy, line.P1, line.P2, new MCvScalar(0, 0, 255));
            //}


            //resGradCopy.Save(path + "houghnya.jpg");


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

        private static Image<Gray,byte> ThresholdByDelta(Image<Gray, byte> firstImage, Image<Gray,byte> secondImage, int threshold) 
        {
            Image<Gray, byte> result = (firstImage - secondImage) + (secondImage-firstImage);
            return result.ThresholdBinary(new Gray(threshold),new Gray(255));
        }
        private static Image<Gray,byte> ThresholdInvByDelta(Image<Gray, byte> firstImage, Image<Gray,byte> secondImage, int threshold) 
        {
            Image<Gray, byte> result = (firstImage - secondImage) + (secondImage-firstImage);
            return result.ThresholdBinaryInv(new Gray(threshold),new Gray(255));
        }
        private static Image<Gray, byte> ChannelDominatingWithThreshold(Image<Gray, byte> firstImage, Image<Gray, byte> secondImage, int threshold)
        {
            Image<Gray, byte> result = firstImage - secondImage;
            return result.ThresholdBinary(new Gray(threshold), new Gray(255));
        }

        private static Image<Gray, byte> IntersectWhite(params Image<Gray, byte>[] images) 
        {
            var result = images[0].Clone();
            foreach (var img in images)
            {
                result._Mul(img);
            }
            return result;
        }

        private static void AnalyzeRoI(Image<Rgb, byte> roi)
        {
            // detect roads: detect road direction
            // detect cities: detect cities direction

            Image<Gray, Byte>[] channels = roi.Split();

            //for (int i = 0; i < channels.Length; i++)
            //{
            //    channels[i].Save(path + i.ToString() + ".jpg");
            //}

            var greenMinusRed = ThresholdByDelta(channels[1], channels[0], 25);
            var greenMinusBlue = ThresholdByDelta(channels[1], channels[2], 25);
            var blueMinusRed = ThresholdByDelta(channels[2], channels[0], 25);

            var strongGreen = channels[1].ThresholdBinary(new Gray(63), new Gray(255));

            int blueThresh = 110;
            Image<Gray, Byte> blueBinary = channels[2].ThresholdBinary(new Gray(blueThresh), new Gray(255));

            var tryResRoadsCities = IntersectWhite(greenMinusRed, greenMinusBlue, blueMinusRed, strongGreen);
            //tryResRoadsCities.Save(path + "tryResRoadsCities.jpg");

            var tryResCities = tryResRoadsCities - blueBinary;
            //tryResCities.Save(path + "tryResCities.jpg");

            // road: check centers +- 0.1 of each: 1 - end; 2 - in and out; 3- cross roads; 4 - cross roads

            // northCenter
            Rectangle north = new Rectangle(blueBinary.Width / 2 - blueBinary.Width / 10, blueBinary.Height/40, 3, 3);
            bool roadNorth = blueBinary.GetSubRect(north).CountNonzero()[0] == 9;
            // eastCenter
            Rectangle east = new Rectangle(blueBinary.Width - blueBinary.Width / 40,blueBinary.Height / 2 - blueBinary.Height / 10, 3, 3);
            bool roadNorth = blueBinary.GetSubRect(north).CountNonzero()[0] == 9;
            // northCenter
            Rectangle south = new Rectangle(blueBinary.Width / 2 - blueBinary.Width / 10, blueBinary.Height/40, 3, 3);
            bool roadNorth = blueBinary.GetSubRect(north).CountNonzero()[0] == 9;
            // northCenter
            Rectangle north = new Rectangle(blueBinary.Width / 2 - blueBinary.Width / 10, blueBinary.Height/40, 3, 3);
            bool roadNorth = blueBinary.GetSubRect(north).CountNonzero()[0] == 9;



        }
    }
}
