using System;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.UI;

namespace CarcassonneMapRecognizer
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            string path = @"E:\HackathonMarchAI\";
            Image<Rgb, Byte> img1 = new Image<Rgb, Byte>(path + "map1.jpg");

            Image<Gray, Byte>[] channels = img1.Split();

            for (int i = 0; i < channels.Length; i++)
            {
                channels[i].Save(path + i.ToString() + ".jpg");
            }

            int redThresh = 205;
            Image<Gray, Byte> redBinary = channels[0].ThresholdBinary(new Gray(redThresh), new Gray(255));
            redBinary.Save(path + "red_binary.jpg");

            int greenThresh = 150;
            // 150 - good for city and roads
            Image<Gray, Byte> greenBinary = channels[1].ThresholdBinary(new Gray(greenThresh), new Gray(255));
            greenBinary.Save(path + "green_binary.jpg");

            int blueThresh = 133;
            Image<Gray, Byte> blueBinary = channels[2].ThresholdBinary(new Gray(blueThresh), new Gray(255));
            blueBinary.Save(path + "blue_binary.jpg");

            int blueThreshRoad = 160;
            Image<Gray, Byte> blueBinaryRoad = channels[2].ThresholdBinary(new Gray(blueThreshRoad), new Gray(255));
            blueBinaryRoad.Save(path + "blue_binary_road.jpg");



            Image<Gray, Byte> cityBinary = blueBinary - redBinary;
            cityBinary.Save(path + "city_binary.jpg");
            //ImageBox imageBox = new ImageBox();

            //Mat matImage = capture.QueryFrame();
            //CamImageBox.Image = matImage; // Directly show Mat object in *ImageBox*
            //Image<Bgr, byte> iplImage = matImage.ToImage<Bgr, byte>();
            //imageBox.Image = img1; // Show Image<,> object in *ImageBox*
        }
    }
}
