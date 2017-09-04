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

void detectFaceInfo2RotatedRect(vis_FaceInfo face, RotatedRect& rbox)
{
  float degree_rad = face.detectFaceInfo.degree*CV_PI/180;
  float cos_degree = cos(degree_rad), sin_degree = sin(degree_rad);
  rbox.center.x = face.detectFaceInfo.left + cos_degree* face.detectFaceInfo.width/2 - sin_degree* face.detectFaceInfo.height/2;
  rbox.center.y = face.detectFaceInfo.top  + sin_degree* face.detectFaceInfo.width/2 + cos_degree* face.detectFaceInfo.height/2;
  rbox.size.width = face.detectFaceInfo.width;
  rbox.size.height = face.detectFaceInfo.height;
  rbox.angle = face.detectFaceInfo.degree;
}

void drawRotatedBox(Mat& img, const RotatedRect& box, const Scalar& color = Scalar(0,255,0))
{
  Point2f vertices[4];
  box.points(vertices);
  for (int j = 0; j < 4; j++)
    line(img, vertices[j], vertices[(j + 1) % 4], color,3);
}

void drawFaces(Mat& img, vector<float> points, const Scalar color = Scalar(0,255,0))
{
  for( int i = 0 ; i < 4 ; ++i )
    line(img, Point2f(points[2*i],points[2*i+1]),
              Point2f(points[(2*i+2)%8], points[(2*i+3)%8]), color, 3);
}

//draw face shape
void drawShape(IplImage* pImg,vector<CvPoint>& shape,vector<float>& scores)
{
  int* nPoints = NULL;
  int  nComponents = 0;
  {       
    nComponents = 9;
    int comp1[] ={0,1,2,3,4,5,6,7,8,9,10,11,12};
    int comp2[] ={13,14,15,16,17,18,19,20,13,21};
    int comp3[] ={22,23,24,25,26,27,28,29,22};
    int comp4[] ={30,31,32,33,34,35,36,37,30,38};
    int comp5[] ={39,40,41,42,43,44,45,46,39};
    int comp6[] ={47,48,49,50,51,52,53,54,55,56,47};
    int comp7[] ={51,57,52};
    int comp8[] ={58,59,60,61,62,63,64,65,58};
    int comp9[] ={58,66,67,68,62,69,70,71,58};

    int* Idx[]  ={comp1,comp2,comp3,comp4,comp5,comp6,comp7,comp8,comp9};
    int nPts[]  ={13,10,9,10,9,11,3,9,9};
    nPoints = nPts;
    for(int i=0;i<nComponents;++i)//nComponents
      for(int j=0;j<nPoints[i]-1;++j)
        cvLine(pImg,shape[Idx[i][j]],shape[Idx[i][j+1]],CV_RGB(0,0,255),2,8,0);
  }
  for(int i=0;i<shape.size();++i)
    cvCircle(pImg,shape[i],pImg->width >1000 ? 5:2,CV_RGB(255*(scores[i]<=0.5),255*(scores[i]>0.5),0),-1,8,0);
}

void loadTestSamples(string img_list,bool has_rect, vector<Sample>& testSamples)
{
  fstream fs(img_list.c_str(),ios::in);
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

void detectFaces(string img_list, string img_dir, string det_res_list, int is_saving = 0, string res_dir = "")
{ 
  //load file list
  vector<Sample> testSamples;
  loadTestSamples(img_list, false, testSamples);
  cout << "Succeed to get images: " << testSamples.size() << endl;

  //init model
  FaceHandler mHandler;
  FacePara param;
  int op_type = DETECT| ALIGN;
  //load config file
  if (LoadParam("./conf", "faceconf_evaluation.cfg", param) != 0) 
  {
    cout<<"fail to init\n";
    return;
  }
  mHandler.Init(param,op_type);

  //file to save the results
  fstream outfile(det_res_list.c_str(),ios::out); 
  if( is_saving)
      mkdir(res_dir.c_str(),S_IRWXU);

  float cost_time = 0;
  int valid_num = 0;

  double benchmark_timer= 0;
  //run predict
  int i = 0;
  for(i=0;i<testSamples.size();++i)
  {
    cerr<<"sample:"<<i+1<<"/"<<testSamples.size()<<endl;
    string imgName = img_dir;
    if(img_dir.size()!=0 && img_dir[img_dir.size()-1]!='/')
    {
      imgName += "/";
    }
    imgName += testSamples[i].ImgName + ".jpg";
    IplImage* img = cvLoadImage(imgName.c_str());

    if(img ==NULL)
    {
      outfile<<testSamples[i].ImgName<<endl;
      outfile<<0<<endl;
      cerr<<"loading image error! please check the path of image: "<<imgName<<endl;
      continue;//return;
    }

    timer timer1;
    timer1.tic();
    mHandler.FaceProcess(img, param, op_type, i%param.threadNum, testSamples[i].faces);
    benchmark_timer += timer1.toc()*1000;
    cerr<<"process time = "<<timer1.toc()*1000<<" ms"<<endl;

    string imgSave_str;
    if( is_saving)
    {
      imgSave_str = testSamples[i].ImgName;
      while(true){
        string::size_type pos(0);
        if( (pos = imgSave_str.find("/")) != string::npos)
          imgSave_str.replace(pos,1,"_");
        else
          break;
      }
    }


    outfile<<testSamples[i].ImgName<<endl;
    outfile<<testSamples[i].faces.size()<<endl;

    IplImage* imgSave = cvCloneImage(img);
    for(int j=0;j<testSamples[i].faces.size();++j)
    {
      RotatedRect rbox;
      detectFaceInfo2RotatedRect(testSamples[i].faces[j], rbox);

      Point2f vertices[4];
      rbox.points(vertices);
      for (int t = 1; t < 5; t++)
        outfile<<vertices[t%4].x<<" "<<vertices[t%4].y<<" ";
      outfile<<testSamples[i].faces[j].detectFaceInfo.conf<<endl;

      //saving the aligned image
      if(is_saving)
      {
        vector<CvPoint> shape;
        for ( int k = 0 ; k < testSamples[i].faces[j].landMarkInfo.landmarks.size() ; ++k)
          shape.push_back( cvPoint( (int)testSamples[i].faces[j].landMarkInfo.landmarks[k].x, 
                (int)testSamples[i].faces[j].landMarkInfo.landmarks[k].y ));
        drawShape(imgSave, shape, testSamples[i].faces[j].landMarkInfo.landmark_scores);

        Mat tempImg = cvarrToMat(imgSave);
        drawRotatedBox(tempImg, rbox);

        ostringstream oss;
        oss<<setprecision(3)<<testSamples[i].faces[j].detectFaceInfo.conf;
        putText(tempImg,oss.str(),cv::Point((int)rbox.center.x,(int)rbox.center.y),cv::FONT_HERSHEY_COMPLEX,0.5,CV_RGB(255,0,0),1,8);
      }
    }

    if( is_saving)
    {
      imgSave_str = res_dir + "/" + imgSave_str + ".jpg";

      cvSaveImage(imgSave_str.c_str(), imgSave);
      cvReleaseImage(&imgSave);
    }

    cvReleaseImage(&img);
  }
  cerr<<"total time = "<<benchmark_timer/1000<<"s"<<endl;
  outfile.close();
}

void getRect( const vector<float> rbox, Rect& box)
{
  float x_min = rbox[0],
        x_max = rbox[0],
        y_min = rbox[1],
        y_max = rbox[1];
 for ( int i = 1; i < 4 ; ++i ){
   x_min = MIN( x_min, rbox[2*i]);
   x_max = MAX( x_max, rbox[2*i]);
   y_min = MIN( y_min, rbox[2*i+1]);
   y_max = MAX( y_max, rbox[2*i+1]);
 }

 box = Rect((int)x_min, (int)y_min, int( x_max - x_min), int( y_max - y_min));
}
float GetOverlap( const vector<float> det_rbox, const vector<float> gt_rbox)
{
  Rect det_box, gt_box;
  getRect(det_rbox, det_box);
  getRect(gt_rbox, gt_box);

  Rect insert_box = det_box & gt_box;

  float overlap = 0.0f;
  if(  0 != gt_box.area()*det_box.area()  )
    overlap = (float)insert_box.area()/MAX(gt_box.area(),det_box.area());
  return overlap;
}
vector<bool> isGroundTruthFaces(
    const vector< std::pair<float, vector<float> > > &det_faces,
    const vector< std::pair<float, vector<float> > > &gt_faces, 
    float threshold){
  vector<bool> isFaces;
  vector<bool> gt_used ;
  gt_used.resize(gt_faces.size(),false);
  for(int det_id = 0 ; det_id < det_faces.size(); det_id ++){
     
    /*
    cerr<<"The "<<det_id<<"-th detected face: "<<endl;
    for( int i = 0 ; i < det_faces[det_id].second.size() ; ++i )
      cerr<<det_faces[det_id].second[i]<<" ";
    cerr<<endl;
    */
     
    float max_overlap = 0;
    float used_id = -1;
    for(int gt_id = 0; gt_id < gt_faces.size(); ++ gt_id)
    {         
      float overlap = GetOverlap(det_faces[det_id].second,gt_faces[gt_id].second);

      /*
      cerr<<" Overlaped with the "<<gt_id <<"-th gt face "<<endl;
      for( int i = 0 ; i < gt_faces[gt_id].second.size() ; ++i )
        cerr<<gt_faces[gt_id].second[i]<<" ";
      cerr<<" is "<<overlap<<endl;
      */
      if( overlap >  max_overlap)
      {       
        max_overlap = overlap;
        used_id = gt_id;
      }       
    }         
    bool trueFace = false;
    if(used_id != -1)
    {         
      trueFace = (max_overlap >= threshold && gt_used[used_id]==false);
      if( trueFace)
        gt_used[used_id] = true;
    }         

    /*
    if( trueFace )
      cerr<<" IT'S A FACE"<<endl<<endl<<endl;
    else
      cerr<<" IT'S NOTTTT FACE"<<endl<<endl<<endl;
    */

    isFaces.push_back(trueFace);
  }           
  return isFaces; 
} 

void compareDetectedWithGroundTruth(string det_list, string gt_list, 
     int& n_positive, vector< std::pair<float, vector<float> > >& detected_faces_with_gt, 
     float overlap_threshold,
     int is_saving, string img_dir, string res_dir, int gt_rect_rotated = 0)
{
  FILE* gt_file = NULL;
  FILE* det_file = NULL;

  det_file = fopen(det_list.c_str(), "r");
  if( !det_file) {
    cerr<<"can not find the detection res file: "<< det_list<<endl;
    return;
  }

  gt_file = fopen(gt_list.c_str(), "r");
  if( !gt_file) {
    cerr<<"can not find the gtection res file: "<< gt_list<<endl;
    return;
  }

  if( is_saving)
      mkdir(res_dir.c_str(),S_IRWXU);

  char img_name[PATH_MAX];
  char gt_img_name[PATH_MAX];
  detected_faces_with_gt.clear();

  Mat src_img;
  char scores_c[PATH_MAX];

  while( fscanf(gt_file, "%s", gt_img_name) == 1)
  {
    if( fscanf(det_file, "%s", img_name) != 1) {
      cerr<<"There are something wrong when read imgName in "<<det_list<<endl;
      return;
    }
    if( strcmp(gt_img_name, img_name) != 0 ) {
      cerr<<"The img name of gt ("<<gt_img_name<<") != det ("<<img_name<")\n";
      return;
    }

    string img_path = img_dir + img_name + ".jpg"; 
    Mat img_draw;
    string imgSave_str;
    if( is_saving)
    {
      img_draw = imread(img_path);
      imgSave_str = img_name;
      while(true){
        string::size_type pos(0);
        if( (pos = imgSave_str.find("/")) != string::npos)
          imgSave_str.replace(pos,1,"_");
        else
          break;
      }
    }

    int n_face = 0;

    vector< std::pair<float, vector<float> > > gt_faces;
    vector< std::pair<float, vector<float> > > det_faces;

    // Read faces from ground truth faces
    if( fscanf( gt_file, "%d", &n_face) != 1 ){
      cerr<<"There are something wrong when read numFace in"<<gt_list<<endl;
      return;
    }
    for( int i = 0 ; i < n_face ; ++i ) {
      float score;
      vector<float> temp;
      if( 0 == gt_rect_rotated )
      {
        float x, y, w, h;
        if( fscanf(gt_file, "%f %f %f %f %f", &x, &y, &w, &h, &score) != 5) {
          cerr<<"There are something wrong when read ground truth faces in"<<gt_list<<endl;
          return;
        }
        temp.push_back(x);
        temp.push_back(y);
        temp.push_back(x+w);
        temp.push_back(y);
        temp.push_back(x+w);
        temp.push_back(y+h);
        temp.push_back(x);
        temp.push_back(y+h);
      }
      else if ( 1 == gt_rect_rotated)
      {
        float x1, y1, x2, y2, x3, y3, x4, y4;
        if( fscanf(gt_file, "%f %f %f %f %f %f %f %f %f", &x1, &y1, &x2, &y2, &x3, &y3, &x4, &y4, &score) != 9) {
          cerr<<"There are something wrong when read detected faces in"<<det_list<<endl;
          return;
        }
        temp.push_back(x1);
        temp.push_back(y1);
        temp.push_back(x2);
        temp.push_back(y2);
        temp.push_back(x3);
        temp.push_back(y3);
        temp.push_back(x4);
        temp.push_back(y4);
      }
      gt_faces.push_back(std::make_pair(score,temp));
      n_positive += 1;
      if (is_saving )
        drawFaces(img_draw, temp, CV_RGB(255,0,0));
    }

    // Read faces from detected faces
    if( fscanf( det_file, "%d", &n_face) != 1 ){
      cerr<<"There are something wrong when read numFace in"<<det_list<<endl;
      return;
    }
    for( int i = 0 ; i < n_face ; ++i ) {
      float x1, y1, x2, y2, x3, y3, x4, y4, score;
      if( fscanf(det_file, "%f %f %f %f %f %f %f %f %f", &x1, &y1, &x2, &y2, &x3, &y3, &x4, &y4, &score) != 9) {
        cerr<<"There are something wrong when read detected faces in"<<det_list<<endl;
        return;
      }
      vector<float> temp;
      temp.push_back(x1);
      temp.push_back(y1);
      temp.push_back(x2);
      temp.push_back(y2);
      temp.push_back(x3);
      temp.push_back(y3);
      temp.push_back(x4);
      temp.push_back(y4);
      det_faces.push_back(std::make_pair(score,temp));
    }
    
    // Detected faces is one of ground truth faces or not
    vector<bool> isFace = isGroundTruthFaces(det_faces, gt_faces, overlap_threshold);
    for( int i = 0 ; i < isFace.size() ; ++i ){
      det_faces[i].second.push_back(isFace[i] == true ? 1 : 0 );
      detected_faces_with_gt.push_back(det_faces[i]);

      if (is_saving) {
        const Scalar now_color = isFace[i]? CV_RGB(0,255,0) : CV_RGB(0,0,255);
        drawFaces(img_draw, det_faces[i].second, now_color);

        ostringstream oss;
        oss<<setprecision(3)<<det_faces[i].first;
        cv::Point face_center( (int)((det_faces[i].second[0]+det_faces[i].second[4])/2),
            (int)((det_faces[i].second[1]+det_faces[i].second[5])/2));
        putText(img_draw, oss.str(), face_center, cv::FONT_HERSHEY_COMPLEX, 0.5, now_color, 1, 8);
      }

    }

    if( is_saving)
    {
      imgSave_str = res_dir + "/" + imgSave_str + ".jpg";
      imwrite(imgSave_str, img_draw);
    }
  }

  fclose(gt_file);
  fclose(det_file);
}

bool compareFaces( const pair<float, vector<float> >& f1, const pair<float, vector<float> >& f2) {
  return f1.first >= f2.first;
}

float getAP(vector< std::pair<float, vector<float> > >& detected_faces_with_gt,
    const int n_positive, vector<float> &recall, vector<int>& n_falseAlarm, vector<float>& score){

  std::stable_sort(detected_faces_with_gt.begin(), detected_faces_with_gt.end(), compareFaces);

  recall.clear();
  n_falseAlarm.clear();
  score.clear();

  vector<float> precision;
  int corrected_count = 0; 
  for(int i=0; i< detected_faces_with_gt.size(); i++) 
  {    
    int num = detected_faces_with_gt[i].second.size();
    corrected_count += int(detected_faces_with_gt[i].second[num-1]) == 1 ? 1:0; 

    precision.push_back(corrected_count/(i+0.0+1));
    recall.push_back(corrected_count/(0.0 + n_positive));
    n_falseAlarm.push_back(i-corrected_count+1);
    score.push_back(detected_faces_with_gt[i].first);
  }    

  float ap = precision[0]*recall[0];
  for(int i=1; i< detected_faces_with_gt.size(); i++) 
  {    
    ap += precision[i]*(recall[i]-recall[i-1]);
  }    
  return ap;
}

void evaluation(string img_list, string img_dir, string gt_list, string det_res_list, string res_list, int need_detect, float threshold, int is_saving, string res_dir, int gt_rect_rotated = 0) 
{
  if( need_detect)
      detectFaces(img_list, img_dir, det_res_list);

  // <score, <x1,y1,x2,y2,x3.y3,x4,y4,0/1> >
  int n_positive = 0;
  vector< std::pair<float, vector<float> > > detected_faces_with_gt;
  compareDetectedWithGroundTruth(det_res_list, gt_list, n_positive, detected_faces_with_gt, threshold, is_saving, img_dir, res_dir, gt_rect_rotated);

  vector<float> recall;
  vector<int> falsePositive;
  vector<float> score;
  float ap = getAP(detected_faces_with_gt, n_positive, recall, falsePositive, score);
  cerr<<"ap = "<<ap<<endl;


  fstream outfile(res_list.c_str(),ios::out); 
  for ( int i = 0 ; i < recall.size() ; ++i ) {
    outfile<<recall[i]<<" "<<falsePositive[i]<<" "<<score[i]<<" "<<n_positive<<endl;
  }
  outfile.close();

}

int main(int argc,char* argv[])
{
  if(argc < 10 || argc > 11)
  { 
    cout<<"Usage: *.exe imgageList imageDir groundTruthList detectionResFile evaluationResFile needDetectAgain overlapThreshold isSaveDetectionRes saveDir gt_rect_rotated[0]"<<endl;
    return -1;
  }
  string img_list      = argv[1];
  string img_dir       = argv[2];
  string gt_list       = argv[3];
  string det_res_list  = argv[4];
  string res_list      = argv[5];
  int need_detect      = atoi(argv[6]);
  float  threshold     = atof(argv[7]);
  int is_saving        = atoi(argv[8]);
  string res_dir       = argv[9];
  int gt_rect_rotated  = 0;
  if( argc == 11 )
    gt_rect_rotated    = atoi(argv[10]);
  

  evaluation(img_list, img_dir, gt_list, det_res_list, res_list, need_detect, threshold, is_saving, res_dir, gt_rect_rotated);
  return 0;
}
