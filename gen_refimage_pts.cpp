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
                //face.detectFaceInfo.pose = 0;
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

void reset_to_one_face_info(vector<vis_FaceInfo>& face_infos) {
  if (face_infos.size() < 1) {
    return;
  }

  int max_width = 0;
  int max_idx = 0;
  for(int i = 0; i < face_infos.size(); i++) {
    int width = face_infos[i].detectFaceInfo.width;
    if (width > max_width) {
      max_idx = i;
      max_width = width;
    }
  }
  // cerr << "max_width " << max_width << endl;
  vis_FaceInfo cur_face_info = face_infos[max_idx];
  for(int i = 0; i < face_infos.size(); i++) {
    if (i == max_idx)
      continue;
    releaseFaceInfo(face_infos[i]);
  }
  face_infos.clear();
  face_infos.push_back(cur_face_info);
}


void predict(string imgList,string srcDir, string dstDir, string failedList)
{
    //load file list
    vector<Sample> testSamples;
    loadTestSamples(imgList, false, testSamples);
    cout << "Succeed to get images: " << testSamples.size() << endl;

    //init model
    FaceHandler mHandler;
    FacePara param;
    int op_type = DETECT | ALIGN | GET_REF_IMAGE;

    if(testSamples.size()&&testSamples[0].faces.size())
    {
        op_type = ALIGN;//|PARSING;
    }
    //load config file
    {
        if (LoadParam("./conf", "faceconf.cfg", param) != 0) 
        {
            cout<<"fail to init\n";
            return;
        }
        mHandler.Init(param,op_type);
    }

    mkdir(dstDir.c_str(),S_IRWXU);
    fstream outfile_failed(failedList.c_str(),ios::out); 

    float cost_time = 0;
    int valid_num = 0;

    timer benchmark_timer;
    benchmark_timer.tic();
    //run predict
    int i = 0;
    for(i=0;i<testSamples.size();++i)
    {
        cerr<<"sample:"<<i+1<<"/"<<testSamples.size()<<endl;
        string imgName = srcDir;
        if(srcDir.size()!=0 && srcDir[srcDir.size()-1]!='/')
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
        cvReleaseImage(&img);
        cerr<<"process time = "<<timer1.toc()*1000<<" ms"<<endl;

        reset_to_one_face_info(testSamples[i].faces);
        if( testSamples[i].faces.size() != 1)
        {
            outfile_failed<<testSamples[i].ImgName<<endl;
            continue;
        }

        timer timer2;
        timer2.tic();

        string saveImgPath = dstDir + "/" + testSamples[i].ImgName + ".jpg";
        string imgSave_makDir = saveImgPath.substr(0,saveImgPath.rfind('/'));
        imgSave_makDir = "mkdir -p "+ imgSave_makDir;
        system(imgSave_makDir.c_str());

        string savePtsPath = dstDir + "/" + testSamples[i].ImgName + ".pts";

        IplImage* refImage = testSamples[i].faces[0].refimage;
        cvSaveImage(saveImgPath.c_str(), refImage);
        cerr<<"save_img time = "<<timer2.toc()*1000<<" ms"<<endl;


        timer timer3;
        timer3.tic();

        fstream outfile(savePtsPath .c_str(),ios::out); 
        for ( int k = 0 ; k < testSamples[i].faces[0].landMarkInfo_warp.landmarks.size(); ++k)
        {
            outfile<<(int)testSamples[i].faces[0].landMarkInfo_warp.landmarks[k].x<<" "
                   <<(int)testSamples[i].faces[0].landMarkInfo_warp.landmarks[k].y<<" ";
        }
        outfile.close();
        cerr<<"write_pts time = "<<timer3.toc()*1000<<" ms"<<endl;



    }
    outfile_failed.close();
    cerr<<"total time = "<<benchmark_timer.toc()<<"s"<<endl;
}

//input a picture name the program perform face detection alignment attribute recognition
int main(int argc,char* argv[])
{
    if(argc <5)
    {
        cout<<"Usage: *.exe imgList srcDir dstDir failedList" <<endl;
        return -1;
    }
    string imgList     = argv[1];
    string srcDir      = argv[2];
    string dstDir      = argv[3];
    string failedList  = argv[4];

    predict(imgList, srcDir, dstDir, failedList);
    return 0;
}
