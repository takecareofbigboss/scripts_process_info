#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <list>
#include <string>
#include <math.h>
#include <sys/stat.h>
#include "face_inf.h"
#include "FaceApi.h"
#include "opencv2/opencv.hpp"
#include "caffe_interface.h"
#include <sys/time.h>
#include "FaUtil.h"
#include <iomanip>
using namespace std;
using namespace cv;
#define SQR(a) ((a)*(a)) 
struct Sample
{
    string ImgName;
    vector< vis_FaceInfo > faces;
};
void DetectInfo2RotatedBox(DetectedFaceInfo face, RotatedRect& rbox)
{
  float degree_rad = face.degree*CV_PI/180;
  float cos_degree = cos(degree_rad), sin_degree = sin(degree_rad);
  rbox.center.x = face.left + cos_degree* face.width/2 - sin_degree* face.height/2;
  rbox.center.y = face.top  + sin_degree* face.width/2 + cos_degree* face.height/2;
  rbox.size.width = face.width;
  rbox.size.height = face.height;
  rbox.angle = face.degree;
}
void DetectInfo2BoundingBox(DetectedFaceInfo face, Rect& box)
{
  RotatedRect rbox;
  DetectInfo2RotatedBox(face, rbox);
  box = rbox.boundingRect();
}

void loadTestSamples(string testImgList,bool has_rect, vector<Sample>& testSamples)
{
    fstream fs(testImgList.c_str(),ios::in);
    if(!fs)
        return;
    string line;
    while(getline(fs,line))
    {
        istringstream iss(line);
        string imgName;
        iss>>imgName;
        Sample s;
        s.ImgName = imgName;
        if(has_rect)
        {
            float x,y, w,h;
            while(iss>>x>>y>>w>>h)
            {
                vis_FaceInfo face;
                face.detectFaceInfo.left = x;
                face.detectFaceInfo.top  = y;
                face.detectFaceInfo.width = w;
                face.detectFaceInfo.height = h;
                face.detectFaceInfo.conf = 1;
                face.IsDetected = true;
                s.faces.push_back(face);
            }
            //cerr<<"number of face: "<<s.faces.size()<<endl;
        } 
        testSamples.push_back(s);
    }
    fs.close();
}

void predict(string face_conf, string testImgList,string imgsDir, string resultDir, string resultimgsDir,int has_rect,int is_saving_drawed_img,float threshold = 0.0f,float min_face_width=0.1f)
{
    //load file list
    vector<Sample> testSamples;
    loadTestSamples(testImgList, false, testSamples);
    cout << "Succeed to get images: " << testSamples.size() << endl;

    //init model
    FaceHandler mHandler;
    FacePara param;
    int op_type = DETECT;

    //load config file
    {
        if (LoadParam("./conf", face_conf.c_str(), param) != 0) 
        {
            cout<<"fail to init\n";
            return;
        }
        mHandler.Init(param,op_type);
    }
    //file to save the results
    mkdir(resultDir.c_str(),S_IRWXU);
    mkdir(resultimgsDir.c_str(),S_IRWXU);


    float cost_time = 0;
    int valid_num = 0;

    timer benchmark_timer;
    benchmark_timer.tic();
    //run predict
    int i = 0;
    for(i=0;i<testSamples.size();++i)
    {
        cerr<<"sample:"<<i+1<<"/"<<testSamples.size()<<endl;
        string imgName = imgsDir;
        if(imgsDir.size()!=0 && imgsDir[imgsDir.size()-1]!='/')
        {
            imgName += "/";
        }
        imgName += testSamples[i].ImgName + ".jpg";
        IplImage* img = cvLoadImage(imgName.c_str());

        if(img ==NULL)
        {
            cerr<<"loading image error! please check the path of image: "<<imgName<<endl;
            continue;//return;
        }

        timer timer1;
        timer1.tic();
        mHandler.FaceProcess(img, param, op_type, i%param.threadNum, testSamples[i].faces);
        cerr<<"process time = "<<timer1.toc()*1000<<" ms"<<endl;

        string imgSave_str;
        {
            imgSave_str = testSamples[i].ImgName;
            while(true){
                string::size_type pos(0);
                if( (pos = imgSave_str.find("/")) != string::npos)
                {
                    string curDir = testSamples[i].ImgName.substr(0,pos);
                    curDir = resultDir + curDir;
                    mkdir(curDir.c_str(),S_IRWXU);
                    imgSave_str.replace(pos,1,"_");
                }
                else
                    break;
            }
        }

        string resFile  = resultDir + testSamples[i].ImgName + ".txt";
        fstream outfile(resFile.c_str(),ios::out); 
        string::size_type pos(0);
        pos = testSamples[i].ImgName.rfind("/");

        outfile<<testSamples[i].ImgName.substr(pos+1, testSamples[i].ImgName.length())<<endl;
        outfile<<testSamples[i].faces.size()<<endl;


        IplImage* imgSave = cvCloneImage(img);
        for(int j=0;j<testSamples[i].faces.size();++j)
        {

          Rect box;
          DetectInfo2BoundingBox(testSamples[i].faces[j].detectFaceInfo, box);
          outfile<<box.x <<" "<<box.y <<" "<< box.width <<" "<<box.height<<" "<<testSamples[i].faces[j].detectFaceInfo.conf<<endl;
          //saving the aligned image
            if(is_saving_drawed_img)
            {
                Mat tempImg = cvarrToMat(imgSave);
                rectangle(tempImg, box, CV_RGB(0,0,255), 1, 8);
                ostringstream oss;
                oss<<setprecision(3)<<testSamples[i].faces[j].detectFaceInfo.conf<<"/"<<setprecision(3)<<testSamples[i].faces[j].landMarkInfo.score;
                putText(tempImg,oss.str(),cv::Point(box.x,box.y),cv::FONT_HERSHEY_COMPLEX,0.5,CV_RGB(255,0,0),1,8);

            }
        }
        
        if( is_saving_drawed_img)
        {
            imgSave_str = resultimgsDir + "/" + imgSave_str + ".jpg";
            
            string imgSave_makDir = imgSave_str.substr(0,imgSave_str.rfind('/'));
            imgSave_makDir = "mkdir -p "+ imgSave_makDir;
            system(imgSave_makDir.c_str());
           

            cvSaveImage(imgSave_str.c_str(), imgSave);
            cvReleaseImage(&imgSave);
        }
        outfile.close();

        cvReleaseImage(&img);
    }
    cerr<<"total time = "<<benchmark_timer.toc()<<"s"<<endl;
}

//input a picture name the program perform face detection alignment attribute recognition
int main(int argc,char* argv[])
{
    /*for(int i=0;i<10000;i++){
      float* a = new float[1000];
      a[0]=3;
      int* b = (int*) malloc(1000);
      b[0]=4;
      }
      printf("ssssssssss\n");
      return 0;
      */
    /////////////////////////////////////////////
    if(argc <8)
    {
        cout<<"Usage: *.exe ImgNameList is_has_rect imgsDir resultPtsListFile is_saving_drawed_img resultimgsDir face_conf [threshold=0.0] [min_face_width=0.0]"<<endl;
        return -1;
    }
    string src_img_list      = argv[1];
    int has_rect             = atoi(argv[2]);
    string Img_path          = argv[3];
    string resultDir         = argv[4];
    int is_saving_drawed_img = atoi(argv[5]);
    string resultimgsDir     = argv[6];
    string face_conf         = argv[7];
    float threshold          = 0.0f;
    float min_face_width     = 0.0f;
    if(argc==9)
        threshold            = atof(argv[8]);
    if(argc==10)
        min_face_width       = atof(argv[9]);

    predict(face_conf,src_img_list,Img_path,resultDir,resultimgsDir,has_rect,is_saving_drawed_img,threshold,min_face_width);
    return 0;
}
