#include<opencv2\opencv.hpp>
#include<iostream>
using namespace std;
using namespace cv;

bool selectObj=false;
Rect select;
Mat image;

void OnMouse(int event, int x, int y, int, void *)
{
	Point origin;
	if (selectObj)
	{
		select.x = min(x, origin.x);
		select.y = min(y, origin.y);
		select.width = abs(x - origin.x);
		select.height = abs(y - origin.y);
	//	select &= Rect(0, 0, image.cols, image.rows);
	}
	switch (event)
	{
	case EVENT_LBUTTONDOWN:
		origin = Point(x, y);
		select = Rect(x, y, 0, 0);
		selectObj = true;
		break;
	case cv::EVENT_LBUTTONUP:
		selectObj = false;
		break;
	default:
		break;
	}
}

int main(int argc, char **argv)
{
	VideoCapture cap(0);
	if (!cap.isOpened())
	{
		cout << "cammera error" << endl;
		return false;
	}
	namedWindow("Get Object", 1);
	setMouseCallback("Get Object", OnMouse,0);
	while (true)
	{
		cap >> image;
		imshow("Get Object", image);
		Mat roi(image, select);
		if (!roi.empty())
		{

			break;
		}
		waitKey(1);
	}
	cout << "Seletion Over" << endl;
	return 0;
}
