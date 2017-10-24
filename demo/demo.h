#ifndef DEMO_H  
#define DEMO_H  
  
#include <QtWidgets/QWidget>  
#include "ui_demo.h"  
#include <QtOpenGL/QtOpenGL>  
#include <QTimer> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#define MIN_LENGTH 35
#define IMAGE_WIDTH 640
#define IMAGE_HEIGHT 480

using namespace cv;
using namespace std;

/*class demo : public QGLWidget
{
	Q_OBJECT

public:
	demo(QWidget *parent = 0);
	~demo();
	//cam
	Mat frame;
	VideoCapture cam,video;
	vector<Point3f> Xworld;
	vector<Point2f> Ximage;
	GLfloat projection_matrix[16];//
	//GLdouble rotMatrix[16];
	vector<GLdouble> rotMatrix;//某一个标记的变换矩阵
	GLdouble rotMat[16];//为了对应opengl函数格式定义的数组
	//GLdouble *rotMatrix = new GLdouble(16);
	vector<vector<GLdouble> >  RotationMatrix;//将所有的变换矩阵保存同一渲染
	Mat  cameraMatrix;//= Mat(3,3,CV_64FC1,1);
	Mat  distCoeffs;//= Mat(1,4,CV_64FC1,1);
	//opengl
	QTimer clk;
	//GLfloat m_x, m_y, m_z;
	//GLfloat xx;
	float WINDOW_SIZE; 
	//float Z_JUST; 
	//int WINDOW_SIZE;
	GLuint texturImage;
	GLuint texturFrame;
	void imageProcess(Mat);
	//鼠标响应
	Mat warpMat;
	Point2f dstRect[4];
	Point2f srcRect[4];
	char number;
	Mat  dobotPos;// = Mat(3, 1, CV_64FC1, Scalar(0, 0, 0));
	Mat dobotTargetPos;
	Point2f dobotFinalTargetPos;//归一化后位置
protected:
	void initializeGL();
	void initWidget();
	void paintGL();
	void resizeGL(int width, int height);
	void loadGLTextures();
	void  mousePressEvent(QMouseEvent *);
private slots:
	
	void updateWindow();
	void updateParams(int);
};

#endif // OPENGL_H
*/
class demo : public QGLWidget  
{  
    Q_OBJECT  
  
public:  
    demo(QWidget *parent = 0);  
    ~demo(); 
    QTimer clk;  
    float m_x, m_y, m_z;  
    GLuint textur; 

    GLuint texturFrame; 
    GLdouble rotMat[16];//为了对应opengl函数格式定义的数组
    vector<vector<GLdouble> >  RotationMatrix;//将所有的变换矩阵保存同一渲染
    QTimer *timer;
protected:  
    void initializeGL();  
    void initWidget();  
    void paintGL();  
    void resizeGL(int width, int height);  
    void loadGLTextures();  
    private slots:  
    void updateWindow();  
private:  
    Ui::demoClass ui;  
};  
  
#endif // DEMO_H 