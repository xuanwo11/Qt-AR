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
	//video.open("F:\\MOVIE\\NBA\\2016��06��20�� NBA�ܾ��� ��ʿVS��ʿ ILP 720P 30fps.mp4");
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
	loadGLTextures();		//����ͼƬ�ļ�
	glEnable(GL_TEXTURE_2D);//��������
	glShadeModel(GL_SMOOTH);
	glClearColor(0, 0, 0.0, 0.0);
	glClearDepth(1.0);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
}

void demo::initWidget()
{
	setGeometry(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT);//���ô���λ�ü���С
	setWindowTitle(tr("opengl demo"));
}

void demo::paintGL()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();//��������ϵ����Ļ���� ��+��- ��-��+ ��-��+
	glTranslatef(3.2f, -0.6f, -100);//����ʾ������Z������ƶ��㹻���룬��ֹ�ڵ���Ⱦ����
	glScalef(8.35f,8.35f,1.0f);//ƽ�� �Ŵ� ʵ�ִ������-
	//������
	glBindTexture(GL_TEXTURE_2D, texturFrame);
	glBegin(GL_QUADS);//����ͼ�νӿڣ���glEnd()��Ӧ

	glTexCoord2f(0.0, 0.0); glVertex3f(-4, -3, 0);//0
	glTexCoord2f(1.0, 0.0); glVertex3f(4, -3, 0);
	glTexCoord2f(1.0, 1.0); glVertex3f(4, 3, 0);
	glTexCoord2f(0.0, 1.0); glVertex3f(-4, 3, 0);

	glEnd();
	//���任����ͳһ������Ⱦ
	if (RotationMatrix.size())
	{
		for (int i = 0; i < RotationMatrix.size(); i++)
		{
			glLoadIdentity();//��������ϵ����Ļ���� ��+��- ��-��+ ��-��+
			for (int j = 0; j < 16; j++)
				rotMat[j] = RotationMatrix[i][j];
			glLoadMatrixd(rotMat);//���ر任����
			glTranslatef(0, 0, -WINDOW_SIZE);//�����������ƶ�WINDOW_SIZE����λ
			//glRotatef(m_x, 1.0, 0.0, 0.0);//��ת
			//glRotatef(m_y, 0.0, 1.0, 0.0);
			//glRotatef(m_z, 0.0, 0.0, 1.0);
			glBindTexture(GL_TEXTURE_2D, texturImage);
			//glLoadMatrixd();
			glBegin(GL_QUADS);//����ͼ�νӿڣ���glEnd()��Ӧ
			//glBegin(GL_LINE_STRIP);//����ͼ�νӿڣ���glEnd()��Ӧ
			
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
	//������Դ�ͷ�
	glDeleteTextures(1, &texturFrame);
	//glDeleteTextures(1, &texturImage);
	
	//glLoadIdentity();//��������ϵ����Ļ���� ��+��- ��-��+ ��-��+
}

void demo::resizeGL(int width, int height)
{
	if (0 == height) {
		height = 1;
	}

	glViewport(0, 0, (GLint)width, (GLint)height);//���õ�ǰ��������ʾλ��
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);//ѡ�����ģʽ
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
	//��������ڲ�����ģ��
	glMultMatrixf(projection_matrix);
	  
	//glLoadMatrixf(projection_matrix);
	//glEnableClientState(GL_VERTEX_ARRAY);  //���ÿͻ��˵�ĳ���
	//glEnableClientState(GL_NORMAL_ARRAY);
	
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}
void demo::updateWindow()
{
    dobotPos = Mat(3, 1, CV_64FC1, Scalar(0, 0, 0));
	//��ȡ����ͷͼ�񲢽��и�ʽת��
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
	//��Mat����ת����QImage
	buf = QImage((const unsigned char*)frame.data, frame.cols, frame.rows, frame.cols * frame.channels(), QImage::Format_RGB888);
	tex = QGLWidget::convertToGLFormat(buf);
	glGenTextures(1, &texturFrame);//��ӦͼƬ��������
	glBindTexture(GL_TEXTURE_2D, texturFrame);//���������
	//������
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
	//imshow("��ֵ��",tempImage);
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
	vector<Point> approxCurve;//���ؽ��Ϊ����Σ��õ㼯��ʾ//������״
	
	for (size_t i = 0; i < contours.size(); i++)
	{
		
		double eps = contours[i].size()*0.05;
		//����ͼ���2ά�㼯�������������ƾ��ȣ��Ƿ�պϡ��������εĶ�����ɵĵ㼯//ʹ����α�Եƽ�����õ����ƵĶ���� 
		approxPolyDP(contours[i], approxCurve, eps, true);

		//���Ǹ���Ȥ�Ķ����ֻ���ĸ�����
		if (approxCurve.size() != 4)
			continue;

		//��������Ƿ���͹����
		if (!isContourConvex(approxCurve))
			continue;

		//ȷ��������֮��ľ������㹻��ġ�//ȷ�����ڵ������ľ��롰�㹻�󡱣�����һ���߶����Ƕ��߶ξ�����
		//float minDist = numeric_limits<float>::max();//����float���Ա�ʾ�����ֵ��numeric_limits����ģ���࣬�����ʾmax��float��;3.4e038
		float minDist = 1e10;//���ֵ�ͺܴ���

		//��ǰ�ı��θ�����֮�����̾���
		for (int j = 0; j < 4; j++)
		{
			Point side = approxCurve[j] - approxCurve[(j + 1) % 4];//����Ӧ����2ά�����
			float squaredSideLength = side.dot(side);//��2ά�����ĵ��������XxY
			minDist = min(minDist, squaredSideLength);//�ҳ���С�ľ���
		}
		//�������ǲ����ر�С��С�Ļ����˳�����ѭ������ʼ��һ��ѭ��
		if (minDist < MIN_LENGTH*MIN_LENGTH)
			continue;
		//���еĲ���ͨ���ˣ������ʶ��ѡ�����ı��δ�С���ʣ��򽫸��ı���maker����possibleMarkers������ //�������Ƶı��   
		drawContours(frame, contours, i, Scalar(255, 0, 255), 1, 8);
		for (int j = 0; j < 4; j++){
			Ximage.push_back(approxCurve[j]);}
		// Sort the points in anti - clockwise
		Point2f v1 = Ximage[1] - Ximage[0];
		Point2f v2 = Ximage[2] - Ximage[0];
		if (v1.cross(v2) > 0)	//����ͼ�������Y�����£����Դ�����Ŵ�����ʱ��
		{
			swap(Ximage[1], Ximage[3]);
		}
		//possible_markers.push_back(marker);
		Mat rvec, tvec;
		solvePnP(Xworld, Ximage, cameraMatrix, distCoeffs, rvec,tvec);
		Mat rmat ;
		Rodrigues(rvec, rmat);

		//��X����ת180�ȣ���OpenCV����ϵ�任ΪOpenGL����ϵ
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
	glGenTextures(1, &texturImage);//��ӦͼƬ��������
	glBindTexture(GL_TEXTURE_2D, texturImage);//���������
	//������
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
			srcRect[number].x = float(mouseEvent->windowPos().x());//��¼��ʼ��
			srcRect[number].y = float(mouseEvent->windowPos().y());//��¼��ʼ��
			number++;
			
			if (number == 4)
				warpMat = getPerspectiveTransform(srcRect, dstRect);	
		}
		else
		{
			dobotPos.at<double>(0, 0) = mouseEvent->windowPos().x();
			dobotPos.at<double>(1, 0) = mouseEvent->windowPos().y();
			dobotPos.at<double>(2, 0) = 1;
			dobotTargetPos = warpMat*dobotPos;//����ӳ�����
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
//�������۲���ѵ�����ݣ�������xml�ļ�����ʽ���洢�˱�׼�������ȶԵ�ģ�����ݣ��ļ������˵�ǰĿ¼  
string faceCascadeName = "haarcascade_frontalface_default.xml";//�沿  
//string eyeCascadeName = "haarcascade_eye_tree_eyeglasses.xml";//�۲�

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
    glLoadIdentity(); //��ʼ����Ⱦ����������ͷ   

glTranslatef(3.2f, -0.6f, -100);//����ʾ������Z������ƶ��㹻���룬��ֹ�ڵ���Ⱦ����  
glScalef(15.5f,15.5f,5.0f);//(8.35f,8.35f,1.0f);//ƽ�� �Ŵ� ʵ�ִ������-  
//������  
glBindTexture(GL_TEXTURE_2D, texturFrame);  
glBegin(GL_QUADS);//����ͼ�νӿڣ���glEnd()��Ӧ  
  
glTexCoord2f(0.0, 0.0); glVertex3f(-4, -3, 0);//  
glTexCoord2f(1.0, 0.0); glVertex3f(4, -3, 0);  
glTexCoord2f(1.0, 1.0); glVertex3f(4, 3, 0);  
glTexCoord2f(0.0, 1.0); glVertex3f(-4, 3, 0);
//glDeleteTextures(1, &texturFrame);//��ʱ�ͷŲ�Ȼ��ռ�úܶ��ڴ�ռ�ʹ���Կ���  
glEnd();

glLoadIdentity();//�µ�һ����Ⱦ���������
    //glTranslatef( (templ.cols + matchLoc.x - 400)/64, (320 - templ.rows - matchLoc.y)/64, -6.0);
    //glTranslatef( 0.0f, 0.0, -6.0);//���������ƶ�
    glTranslatef( (centered.x - 400)/64, (320 - centered.y)/64, -6.0);
    glRotatef(m_x, 1.0, 0.0, 0.0);//��ת  
    glRotatef(m_y, 0.0, 1.0, 0.0);  
    glRotatef(m_z, 0.0, 0.0, 1.0);  
           //����������  

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

    
	//������Դ�ͷ�
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


        //��ȡ����ͷͼ�񲢽��и�ʽת��
VideoCapture camera(1);
if(!camera.isOpened())
{
    return;
}
//Mat frame;
camera >> frame; 
cvtColor(frame, frame, CV_BGR2RGB);
/////////////////////////////////////////////////////r�������
double scale = 1;//����СͼƬ�������������׼ȷ��
CascadeClassifier faceCascade, eyeCascade;//���弶����������������ʵ�ּ�⹦��
if (!faceCascade.load(faceCascadeName))// || !eyeCascade.load(eyeCascadeName))//����xmlѵ������  
{  
    return;  
}
if (!frame.empty())  
{
    int i = 0;    
    vector<Rect> faces,eyes;//�����洢���������沿���۲����ݣ������޷�ȷ����������˶����vector  
    Mat gray;  
  
    cvtColor(frame, gray, CV_BGR2GRAY);//ͼƬ��ɫ��ʽת����CV_BGR2GRAY�Ǵ�BGR��gray����ɫ����ɫ  
    //resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);//����ɫͼ����Ӧ��С��smallImg��  
    equalizeHist(gray, gray);//��ǿ�Աȶȣ���߼��׼ȷ��  
   
    //ʹ�ü�������������ʶ�𣬲���Ϊ�Ҷ�ͼƬ���沿���飬��ⵥԪ�������ʣ��Ƿ��ںϼ����ľ��Σ���С��ⵥԪ��С  
    faceCascade.detectMultiScale(gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));  
    for( size_t i = 0; i < faces.size(); i++ )  
   {  
        Point centers( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
        centered = centers;
        rectangle( frame, centered, Point( faces[i].width*0.5, faces[i].height*0.5), Scalar( 255, 255, 0 ), 2, 8, 0 );  
        //cv::rectangle(frame, *r, Scalar(0,255,0), 1, 1, 0);//��img�ϻ��Ƴ���⵽���沿���ο���ɫ��
    
/*templ = imread("13.png");
int result_cols = frame.cols - frame.cols + 1;
int result_rows = frame.rows - frame.rows + 1;
result.create(result_cols, result_rows, CV_32FC1);
matchTemplate(frame, templ, result, CV_TM_SQDIFF_NORMED);//��������ʹ�õ�ƥ���㷨�Ǳ�׼ƽ����ƥ�� method=CV_TM_SQDIFF_NORMED����ֵԽСƥ���Խ��
normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
matchLoc = minLoc;

rectangle(frame, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(0, 255, 0), 2, 8, 0);

distx = (matchLoc.x + templ.cols)/2;
disty = (matchLoc.y + templ.rows)/2;*/
/////////////////////////////////////////////////////
QImage bufs, texs;  
//��Mat����ת����QImage  
bufs = QImage((const unsigned char*)frame.data, frame.cols, frame.rows, frame.cols * frame.channels(), QImage::Format_RGB888);  
texs = QGLWidget::convertToGLFormat(bufs);  
glGenTextures(1, &texturFrame);//��ӦͼƬ��������  
glBindTexture(GL_TEXTURE_2D, texturFrame);//���������  
//������  
glTexImage2D(GL_TEXTURE_2D, 0, 3, texs.width(), texs.height(), 0,  
GL_RGBA, GL_UNSIGNED_BYTE, texs.bits());  
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);  
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); 

          updateGL();//ˢ�½���
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