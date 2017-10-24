#include "demo.h" 
#include<opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\video\video.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <iostream>
#include <stdio.h>
using namespace cv;  
using namespace std;

/*demo::demo(QWidget *parent)
: QGLWidget(parent), number(0)
{
	cam.open(0);//0
    //VideoCapture cam(0);
    cameraMatrix = Mat(3,3,CV_64FC1,1);//1
    distCoeffs = Mat(1,4,CV_64FC1,1);//1
    dobotPos = Mat(3, 1, CV_64FC1, Scalar(255, 255, 0));//0,0,0
	//cam.set(CV_CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH);
	//cam.set(CV_CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT);
	//video.open("F:\\MOVIE\\NBA\\2016年06月20日 NBA总决赛 骑士VS勇士 ILP 720P 30fps.mp4");
	initWidget();
	initializeGL();
	//resizeGL(640,480);
	clk.start(30);
	QObject::connect(&clk, SIGNAL(timeout()), this, SLOT(updateWindow()));
	

}
demo::~demo()
{

}

void demo::initializeGL()
{
	loadGLTextures();		//加载图片文件
	glEnable(GL_TEXTURE_2D);//启用纹理
	glShadeModel(GL_SMOOTH);
	glClearColor(0, 0, 0.0, 0.0);
	glClearDepth(1.0);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
}

void demo::initWidget()
{
	setGeometry(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT);//设置窗口位置及大小
	setWindowTitle(tr("opengl demo"));
}

void demo::paintGL()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();//重置坐标系至屏幕中央 上+下- 左-右+ 里-外+
	glTranslatef(3.2f, -0.6f, -100);//讲显示背景沿Z轴向后移动足够距离，防止遮挡渲染物体
	glScalef(8.35f,8.35f,1.0f);//平移 放大 实现窗口填充-
	//绑定纹理
	glBindTexture(GL_TEXTURE_2D, texturFrame);
	glBegin(GL_QUADS);//绘制图形接口，与glEnd()对应

	glTexCoord2f(0.0, 0.0); glVertex3f(-4, -3, 0);//0
	glTexCoord2f(1.0, 0.0); glVertex3f(4, -3, 0);
	glTexCoord2f(1.0, 1.0); glVertex3f(4, 3, 0);
	glTexCoord2f(0.0, 1.0); glVertex3f(-4, 3, 0);

	glEnd();
	//将变换矩阵统一进行渲染
	if (RotationMatrix.size())
	{
		for (int i = 0; i < RotationMatrix.size(); i++)
		{
			glLoadIdentity();//重置坐标系至屏幕中央 上+下- 左-右+ 里-外+
			for (int j = 0; j < 16; j++)
				rotMat[j] = RotationMatrix[i][j];
			glLoadMatrixd(rotMat);//加载变换矩阵
			glTranslatef(0, 0, -WINDOW_SIZE);//将物体向外移动WINDOW_SIZE个单位
			//glRotatef(m_x, 1.0, 0.0, 0.0);//旋转
			//glRotatef(m_y, 0.0, 1.0, 0.0);
			//glRotatef(m_z, 0.0, 0.0, 1.0);
			glBindTexture(GL_TEXTURE_2D, texturImage);
			//glLoadMatrixd();
			glBegin(GL_QUADS);//绘制图形接口，与glEnd()对应
			//glBegin(GL_LINE_STRIP);//绘制图形接口，与glEnd()对应
			
			glNormal3f(0.0, 0.0, 1.0);
			glTexCoord2f(0.0, 0.0); glVertex3f(-WINDOW_SIZE, WINDOW_SIZE, WINDOW_SIZE);//
			glTexCoord2f(1.0, 0.0); glVertex3f(WINDOW_SIZE, -WINDOW_SIZE, WINDOW_SIZE);
			glTexCoord2f(1.0, 1.0); glVertex3f(WINDOW_SIZE, WINDOW_SIZE, WINDOW_SIZE);
			glTexCoord2f(0.0, 1.0); glVertex3f(-WINDOW_SIZE, WINDOW_SIZE, WINDOW_SIZE);

			glNormal3f(0.0, 0.0, -1.0);
			glTexCoord2f(1.0, 0.0); glVertex3f(-WINDOW_SIZE, -WINDOW_SIZE, -WINDOW_SIZE);
			glTexCoord2f(1.0, 1.0); glVertex3f(-WINDOW_SIZE, WINDOW_SIZE, -WINDOW_SIZE);
			glTexCoord2f(0.0, 1.0); glVertex3f(WINDOW_SIZE, WINDOW_SIZE, -WINDOW_SIZE);
			glTexCoord2f(0.0, 0.0); glVertex3f(WINDOW_SIZE, -WINDOW_SIZE, -WINDOW_SIZE);

			glNormal3f(0.0, 1.0, 0.0);
			glTexCoord2f(0.0, 1.0); glVertex3f(-WINDOW_SIZE, WINDOW_SIZE, -WINDOW_SIZE);
			glTexCoord2f(0.0, 0.0); glVertex3f(-WINDOW_SIZE, WINDOW_SIZE, WINDOW_SIZE);
			glTexCoord2f(1.0, 0.0); glVertex3f(WINDOW_SIZE, WINDOW_SIZE, WINDOW_SIZE);
			glTexCoord2f(1.0, 1.0); glVertex3f(WINDOW_SIZE, WINDOW_SIZE, -WINDOW_SIZE);

			glNormal3f(0.0, -1.0, 0.0);
			glTexCoord2f(1.0, 1.0); glVertex3f(-WINDOW_SIZE, -WINDOW_SIZE, -WINDOW_SIZE);
			glTexCoord2f(0.0, 1.0); glVertex3f(WINDOW_SIZE, -WINDOW_SIZE, -WINDOW_SIZE);
			glTexCoord2f(0.0, 0.0); glVertex3f(WINDOW_SIZE, -WINDOW_SIZE, WINDOW_SIZE);
			glTexCoord2f(1.0, 0.0); glVertex3f(-WINDOW_SIZE, -WINDOW_SIZE, WINDOW_SIZE);

			glNormal3f(1.0, 0.0, 0.0);
			glTexCoord2f(1.0, 0.0); glVertex3f(WINDOW_SIZE, -WINDOW_SIZE, -WINDOW_SIZE);
			glTexCoord2f(1.0, 1.0); glVertex3f(WINDOW_SIZE, WINDOW_SIZE, -WINDOW_SIZE);
			glTexCoord2f(0.0, 1.0); glVertex3f(WINDOW_SIZE, WINDOW_SIZE, WINDOW_SIZE);
			glTexCoord2f(0.0, 0.0); glVertex3f(WINDOW_SIZE, -WINDOW_SIZE, WINDOW_SIZE);

			glNormal3f(-1.0, 0.0, 0.0);
			glTexCoord2f(0.0, 0.0); glVertex3f(-WINDOW_SIZE, -WINDOW_SIZE, -WINDOW_SIZE);
			glTexCoord2f(1.0, 0.0); glVertex3f(-WINDOW_SIZE, -WINDOW_SIZE, WINDOW_SIZE);
			glTexCoord2f(1.0, 1.0); glVertex3f(-WINDOW_SIZE, WINDOW_SIZE, WINDOW_SIZE);
			glTexCoord2f(0.0, 1.0); glVertex3f(-WINDOW_SIZE, WINDOW_SIZE, -WINDOW_SIZE);

			glEnd();
		}
		RotationMatrix.clear();
	}
	//纹理资源释放
	glDeleteTextures(1, &texturFrame);
	//glDeleteTextures(1, &texturImage);
	
	//glLoadIdentity();//重置坐标系至屏幕中央 上+下- 左-右+ 里-外+
}

void demo::resizeGL(int width, int height)
{
	if (0 == height) {
		height = 1;
	}

	glViewport(0, 0, (GLint)width, (GLint)height);//重置当前的物体显示位置
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);//选择矩阵模式
	double f_x = cameraMatrix.at<double>(0, 0);
	double f_y = cameraMatrix.at<double>(1, 1);

	double c_x = cameraMatrix.at<double>(0, 2);
	double c_y = cameraMatrix.at<double>(1, 2);

	projection_matrix[0] = 2 * f_x / IMAGE_WIDTH;
	projection_matrix[1] = 0.0f;
	projection_matrix[2] = 0.0f;
	projection_matrix[3] = 0.0f;

	projection_matrix[4] = 0.0f;
	projection_matrix[5] = 2 * f_y / IMAGE_HEIGHT;
	projection_matrix[6] = 0.0f;
	projection_matrix[7] = 0.0f;

	projection_matrix[8] = 1.0f - 2 * c_x / IMAGE_WIDTH;
	projection_matrix[9] = 2 * c_y / IMAGE_HEIGHT - 1.0f;
	projection_matrix[10] = -(0.01f + 100.0f) / (100.0f - 0.01f);
	projection_matrix[11] = -1.0f;

	projection_matrix[12] = 0.0f;
	projection_matrix[13] = 0.0f;
	projection_matrix[14] = -2.0f * 100 * 0.01 / (100.0f - 0.01f);
	projection_matrix[15] = 0.0f;
	//导入相机内部参数模型
	glMultMatrixf(projection_matrix);
	  
	//glLoadMatrixf(projection_matrix);
	//glEnableClientState(GL_VERTEX_ARRAY);  //启用客户端的某项功能
	//glEnableClientState(GL_NORMAL_ARRAY);
	
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}
void demo::updateWindow()
{
    dobotPos = Mat(3, 1, CV_64FC1, Scalar(0, 0, 0));
	//获取摄像头图像并进行格式转换
    cam.open(1);//0
	cam >> frame; // imshow("frame",frame);
	cvtColor(frame, frame, CV_BGR2RGB);
	for (char i = 0; i < 4; i++)
	{
		circle(frame, srcRect[i], 8, Scalar(255, 0, 0), -1, 8, 0);
	}
	circle(frame, Point(dobotPos.at<double>(0, 0), dobotPos.at<double>(1, 0)), 5, Scalar(0, 255, 255), -1, 8);
	imageProcess(frame);
	QImage buf, tex;
	//将Mat类型转换成QImage
	buf = QImage((const unsigned char*)frame.data, frame.cols, frame.rows, frame.cols * frame.channels(), QImage::Format_RGB888);
	tex = QGLWidget::convertToGLFormat(buf);
	glGenTextures(1, &texturFrame);//对应图片的纹理定义
	glBindTexture(GL_TEXTURE_2D, texturFrame);//进行纹理绑定
	//纹理创建
	glTexImage2D(GL_TEXTURE_2D, 0, 3, tex.width(), tex.height(), 0,
		GL_RGBA, GL_UNSIGNED_BYTE, tex.bits());
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	updateGL();
	
}
void demo::imageProcess(Mat image)
{
	Mat grayImage,tempImage;
	cvtColor(image,grayImage,CV_BGR2GRAY);
	adaptiveThreshold(grayImage, tempImage, 255, ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 7, 7);
	//imshow("二值化",tempImage);
	vector<vector<Point>> all_contours;
	vector<vector<Point>> contours;
	//Rect RectArea;
	findContours(tempImage, all_contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	for (int i = 0; i < all_contours.size(); ++i)
	{
		if (all_contours[i].size() > 100)
		{
			contours.push_back(all_contours[i]);
			//drawContours(frame,all_contours,i,Scalar(0,0,255),1,8);
		}
	}
	vector<Point> approxCurve;//返回结果为多边形，用点集表示//相似形状
	
	for (size_t i = 0; i < contours.size(); i++)
	{
		
		double eps = contours[i].size()*0.05;
		//输入图像的2维点集，输出结果，估计精度，是否闭合。输出多边形的顶点组成的点集//使多边形边缘平滑，得到近似的多边形 
		approxPolyDP(contours[i], approxCurve, eps, true);

		//我们感兴趣的多边形只有四个顶点
		if (approxCurve.size() != 4)
			continue;

		//检查轮廓是否是凸边形
		if (!isContourConvex(approxCurve))
			continue;

		//确保连续点之间的距离是足够大的。//确保相邻的两点间的距离“足够大”－大到是一条边而不是短线段就是了
		//float minDist = numeric_limits<float>::max();//代表float可以表示的最大值，numeric_limits就是模板类，这里表示max（float）;3.4e038
		float minDist = 1e10;//这个值就很大了

		//求当前四边形各顶点之间的最短距离
		for (int j = 0; j < 4; j++)
		{
			Point side = approxCurve[j] - approxCurve[(j + 1) % 4];//这里应该是2维的相减
			float squaredSideLength = side.dot(side);//求2维向量的点积，就是XxY
			minDist = min(minDist, squaredSideLength);//找出最小的距离
		}
		//检查距离是不是特别小，小的话就退出本次循环，开始下一次循环
		if (minDist < MIN_LENGTH*MIN_LENGTH)
			continue;
		//所有的测试通过了，保存标识候选，当四边形大小合适，则将该四边形maker放入possibleMarkers容器内 //保存相似的标记   
		drawContours(frame, contours, i, Scalar(255, 0, 255), 1, 8);
		for (int j = 0; j < 4; j++){
			Ximage.push_back(approxCurve[j]);}
		// Sort the points in anti - clockwise
		Point2f v1 = Ximage[1] - Ximage[0];
		Point2f v2 = Ximage[2] - Ximage[0];
		if (v1.cross(v2) > 0)	//由于图像坐标的Y轴向下，所以大于零才代表逆时针
		{
			swap(Ximage[1], Ximage[3]);
		}
		//possible_markers.push_back(marker);
		Mat rvec, tvec;
		solvePnP(Xworld, Ximage, cameraMatrix, distCoeffs, rvec,tvec);
		Mat rmat ;
		Rodrigues(rvec, rmat);

		//绕X轴旋转180度，从OpenCV坐标系变换为OpenGL坐标系
		static double d[] =
		{
			1, 0, 0,
			0,-1, 0,
			0, 0, -1
		};
		Mat_<double> rx(3, 3, d);

		rmat = rx*rmat;
		tvec = rx*tvec;

		rotMatrix.push_back(rmat.at<double>(0, 0));
		rotMatrix.push_back(rmat.at<double>(1, 0));
		rotMatrix.push_back(rmat.at<double>(2, 0));
		rotMatrix.push_back(0.0f);

		rotMatrix.push_back(rmat.at<double>(0, 1));
		rotMatrix.push_back(rmat.at<double>(1, 1));
		rotMatrix.push_back(rmat.at<double>(2, 1));
		rotMatrix.push_back(0.0f);

		rotMatrix.push_back(rmat.at<double>(0, 2));
		rotMatrix.push_back(rmat.at<double>(1, 2));
		rotMatrix.push_back(rmat.at<double>(2, 2));
		rotMatrix.push_back(0.0f);

		rotMatrix.push_back(tvec.at<double>(0, 0));
		rotMatrix.push_back(tvec.at<double>(1, 0));
		rotMatrix.push_back(tvec.at<double>(2, 0));
		rotMatrix.push_back(1.0f);
		
		RotationMatrix.push_back(rotMatrix);
		//RotMat.pop_back(rotMatrix);
		Ximage.clear();
		rotMatrix.clear();
	}
}
void demo::updateParams(int timerValue)
{
	clk.start(timerValue);
}
void demo::loadGLTextures()
{
	QImage tex;
	QImage buf;

	if (!buf.load("H:\\Mysql\\demo\\demo\\Resources\\butterfly.jpg"))
	{
		qWarning("load image failed!");
		QImage dummy(128, 128, QImage::Format_RGB32);
		dummy.fill(Qt::green);
		buf = dummy;

	}

	tex = QGLWidget::convertToGLFormat(buf);
	glGenTextures(1, &texturImage);//对应图片的纹理定义
	glBindTexture(GL_TEXTURE_2D, texturImage);//进行纹理绑定
	//纹理创建
	glTexImage2D(GL_TEXTURE_2D, 0, 3, tex.width(), tex.height(), 0,
		GL_RGBA, GL_UNSIGNED_BYTE, tex.bits());

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	
}
void demo::mousePressEvent(QMouseEvent *mouseEvent)
{
    //dobotPos = Mat(3, 1, CV_64FC1, Scalar(0, 0, 0));//////////////////////////////////////
	switch (mouseEvent->button())
	{
	case Qt::LeftButton:
	{
		if (number < 4)
		{
			srcRect[number].x = float(mouseEvent->windowPos().x());//记录起始点
			srcRect[number].y = float(mouseEvent->windowPos().y());//记录起始点
			number++;
			
			if (number == 4)
				warpMat = getPerspectiveTransform(srcRect, dstRect);	
		}
		else
		{
			dobotPos.at<double>(0, 0) = mouseEvent->windowPos().x();
			dobotPos.at<double>(1, 0) = mouseEvent->windowPos().y();
			dobotPos.at<double>(2, 0) = 1;
			dobotTargetPos = warpMat*dobotPos;//计算映射矩阵
			double s = dobotTargetPos.at<double>(2, 0);
			dobotFinalTargetPos.x = dobotTargetPos.at<double>(0, 0) / s;
			dobotFinalTargetPos.y = dobotTargetPos.at<double>(1, 0) / s;
		}
	}
		break;
	default:
		break;
	}
}*/

float distx = 0.0;
float disty = 0.0;
double minVal = -1;
double maxVal;
Point minLoc;
Point maxLoc;
Point matchLoc;
Point centered;
Mat templ, result;
Mat frame;  
//脸部和眼部的训练数据，就是以xml文件的形式，存储了标准的用来比对的模特数据，文件放在了当前目录  
string faceCascadeName = "haarcascade_frontalface_default.xml";//面部  
//string eyeCascadeName = "haarcascade_eye_tree_eyeglasses.xml";//眼部

demo::demo(QWidget *parent)  
: QGLWidget(parent)  
  
{  
    ui.setupUi(this);
    initWidget();  
    initializeGL();  
    clk.start(30);  
    QObject::connect(&clk, SIGNAL(timeout()), this, SLOT(updateWindow()));  
}  
  
demo::~demo()  
{  
  
}  
void demo::initializeGL()  
{  
    loadGLTextures();  
    glEnable(GL_TEXTURE_2D);  
  
    glShadeModel(GL_SMOOTH);  
    glClearColor(0.0, 0.0, 0.0, 0.0);  
    glClearDepth(1.0);  
    glEnable(GL_DEPTH_TEST);  
    glDepthFunc(GL_LEQUAL);  
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);  
}  
  
void demo::initWidget()  
{  
    setGeometry(0,160,800,640);//(0,200, 640, 480);  
    setWindowTitle(tr("opengl demo"));  
}  
  
void demo::paintGL()  
{  
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);  
    glLoadIdentity(); //初始层渲染，背景摄像头   

glTranslatef(3.2f, -0.6f, -100);//讲显示背景沿Z轴向后移动足够距离，防止遮挡渲染物体  
glScalef(15.5f,15.5f,5.0f);//(8.35f,8.35f,1.0f);//平移 放大 实现窗口填充-  
//绑定纹理  
glBindTexture(GL_TEXTURE_2D, texturFrame);  
glBegin(GL_QUADS);//绘制图形接口，与glEnd()对应  
  
glTexCoord2f(0.0, 0.0); glVertex3f(-4, -3, 0);//  
glTexCoord2f(1.0, 0.0); glVertex3f(4, -3, 0);  
glTexCoord2f(1.0, 1.0); glVertex3f(4, 3, 0);  
glTexCoord2f(0.0, 1.0); glVertex3f(-4, 3, 0);
//glDeleteTextures(1, &texturFrame);//及时释放不然会占用很多内存空间使电脑卡死  
glEnd();

glLoadIdentity();//新的一层渲染，即虚拟层
    //glTranslatef( (templ.cols + matchLoc.x - 400)/64, (320 - templ.rows - matchLoc.y)/64, -6.0);
    //glTranslatef( 0.0f, 0.0, -6.0);//沿坐标轴移动
    glTranslatef( (centered.x - 400)/64, (320 - centered.y)/64, -6.0);
    glRotatef(m_x, 1.0, 0.0, 0.0);//旋转  
    glRotatef(m_y, 0.0, 1.0, 0.0);  
    glRotatef(m_z, 0.0, 0.0, 1.0);  
           //绑定纹理特性  

    glBindTexture(GL_TEXTURE_2D, textur); //textur    
    glBegin(GL_QUADS);   


    glNormal3f(0.0, 0.0, 1.0);  
    glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0, 1.0);  
    glTexCoord2f(1.0, 0.0); glVertex3f(1.0, -1.0, 1.0);  
    glTexCoord2f(1.0, 1.0); glVertex3f(1.0, 1.0, 1.0);  
    glTexCoord2f(0.0, 1.0); glVertex3f(-1.0, 1.0, 1.0);  
  
    glNormal3f(0.0, 0.0, -1.0);  
    glTexCoord2f(1.0, 0.0); glVertex3f(-1.0, -1.0, -1.0);  
    glTexCoord2f(1.0, 1.0); glVertex3f(-1.0, 1.0, -1.0);  
    glTexCoord2f(0.0, 1.0); glVertex3f(1.0, 1.0, -1.0);  
    glTexCoord2f(0.0, 0.0); glVertex3f(1.0, -1.0, -1.0);  
  
    glNormal3f(0.0, 1.0, 0.0);  
    glTexCoord2f(0.0, 1.0); glVertex3f(-1.0, 1.0, -1.0);  
    glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, 1.0, 1.0);  
    glTexCoord2f(1.0, 0.0); glVertex3f(1.0, 1.0, 1.0);  
    glTexCoord2f(1.0, 1.0); glVertex3f(1.0, 1.0, -1.0);  
  
    glNormal3f(0.0, -1.0, 0.0);  
    glTexCoord2f(1.0, 1.0); glVertex3f(-1.0, -1.0, -1.0);  
    glTexCoord2f(0.0, 1.0); glVertex3f(1.0, -1.0, -1.0);  
    glTexCoord2f(0.0, 0.0); glVertex3f(1.0, -1.0, 1.0);  
    glTexCoord2f(1.0, 0.0); glVertex3f(-1.0, -1.0, 1.0);  
  
    glNormal3f(1.0, 0.0, 0.0);  
    glTexCoord2f(1.0, 0.0); glVertex3f(1.0, -1.0, -1.0);  
    glTexCoord2f(1.0, 1.0); glVertex3f(1.0, 1.0, -1.0);  
    glTexCoord2f(0.0, 1.0); glVertex3f(1.0, 1.0, 1.0);  
    glTexCoord2f(0.0, 0.0); glVertex3f(1.0, -1.0, 1.0);  
  
    glNormal3f(-1.0, 0.0, 0.0);  
    glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0, -1.0);  
    glTexCoord2f(1.0, 0.0); glVertex3f(-1.0, -1.0, 1.0);  
    glTexCoord2f(1.0, 1.0); glVertex3f(-1.0, 1.0, 1.0);  
    glTexCoord2f(0.0, 1.0); glVertex3f(-1.0, 1.0, -1.0);    

    glEnd(); 

    
	//纹理资源释放
	glDeleteTextures(1, &texturFrame);
}  
  
void demo::resizeGL(int width, int height)  
{  
    if (0 == height) {  
        height = 1;  
    }  
  
    glViewport(0, 0, (GLint)width, (GLint)height);  
  
    glMatrixMode(GL_PROJECTION);  
  
    glLoadIdentity();  
  
    //gluPerspective(45.0, (GLfloat)width/(GLfloat)height, 0.1, 100.0);  
  
    GLdouble aspectRatio = (GLfloat)width / (GLfloat)height;  
    GLdouble zNear = 0.1;  
    GLdouble zFar = 100.0;  
  
    GLdouble rFov = 50.0 * 3.14159265 / 180.0;  
    glFrustum(-zNear * tan(rFov / 2.0) * aspectRatio,  
        zNear * tan(rFov / 2.0) * aspectRatio,  
        -zNear * tan(rFov / 2.0),  
        zNear * tan(rFov / 2.0),  
        zNear, zFar);  
  
    glMatrixMode(GL_MODELVIEW);  
    glLoadIdentity();  
}  
void demo::updateWindow()  
{  
    m_x += 1;  
    m_y += 2;  
    m_z += 3; 


        //获取摄像头图像并进行格式转换
VideoCapture camera(1);
if(!camera.isOpened())
{
    return;
}
//Mat frame;
camera >> frame; 
cvtColor(frame, frame, CV_BGR2RGB);
/////////////////////////////////////////////////////r人脸检测
double scale = 1;//不缩小图片，这样可以提高准确率
CascadeClassifier faceCascade, eyeCascade;//定义级联分类器，由它们实现检测功能
if (!faceCascade.load(faceCascadeName))// || !eyeCascade.load(eyeCascadeName))//载入xml训练数据  
{  
    return;  
}
if (!frame.empty())  
{
    int i = 0;    
    vector<Rect> faces,eyes;//用来存储检测出来的面部和眼部数据，我们无法确定个数，因此定义成vector  
    Mat gray;  
  
    cvtColor(frame, gray, CV_BGR2GRAY);//图片颜色格式转化，CV_BGR2GRAY是从BGR到gray，彩色到灰色  
    //resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);//将灰色图像适应大小到smallImg中  
    equalizeHist(gray, gray);//加强对比度，提高检测准确率  
   
    //使用级联分类器进行识别，参数为灰度图片，面部数组，检测单元的增长率，是否融合检测出的矩形，最小检测单元大小  
    faceCascade.detectMultiScale(gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));  
    for( size_t i = 0; i < faces.size(); i++ )  
   {  
        Point centers( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
        centered = centers;
        rectangle( frame, centered, Point( faces[i].width*0.5, faces[i].height*0.5), Scalar( 255, 255, 0 ), 2, 8, 0 );  
        //cv::rectangle(frame, *r, Scalar(0,255,0), 1, 1, 0);//在img上绘制出检测到的面部矩形框，绿色框
    
/*templ = imread("13.png");
int result_cols = frame.cols - frame.cols + 1;
int result_rows = frame.rows - frame.rows + 1;
result.create(result_cols, result_rows, CV_32FC1);
matchTemplate(frame, templ, result, CV_TM_SQDIFF_NORMED);//这里我们使用的匹配算法是标准平方差匹配 method=CV_TM_SQDIFF_NORMED，数值越小匹配度越好
normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
matchLoc = minLoc;

rectangle(frame, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(0, 255, 0), 2, 8, 0);

distx = (matchLoc.x + templ.cols)/2;
disty = (matchLoc.y + templ.rows)/2;*/
/////////////////////////////////////////////////////
QImage bufs, texs;  
//将Mat类型转换成QImage  
bufs = QImage((const unsigned char*)frame.data, frame.cols, frame.rows, frame.cols * frame.channels(), QImage::Format_RGB888);  
texs = QGLWidget::convertToGLFormat(bufs);  
glGenTextures(1, &texturFrame);//对应图片的纹理定义  
glBindTexture(GL_TEXTURE_2D, texturFrame);//进行纹理绑定  
//纹理创建  
glTexImage2D(GL_TEXTURE_2D, 0, 3, texs.width(), texs.height(), 0,  
GL_RGBA, GL_UNSIGNED_BYTE, texs.bits());  
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);  
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); 

          updateGL();//刷新界面
     }///
   }///
}  
void demo::loadGLTextures()  
{  

    QImage tex;  
    QImage buf;  
  
    if (!buf.load("H:\\Mysql\\demo\\demo\\Resources\\butterfly.jpg"))  
    {  
        qWarning("load image failed!");  
        QImage dummy(128, 128, QImage::Format_RGB32);//128  
        dummy.fill(Qt::red);  
        buf = dummy;  
  
    }  
    
    tex = QGLWidget::convertToGLFormat(buf);  
    glGenTextures(1, &textur);  
    glBindTexture(GL_TEXTURE_2D, textur);  
  
    glTexImage2D(GL_TEXTURE_2D, 0, 3, tex.width(), tex.height(), 0,  
        GL_RGBA, GL_UNSIGNED_BYTE, tex.bits());  
  
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);  
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    
}