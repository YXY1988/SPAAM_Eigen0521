/****************************************************
 Goal: Realize SPAAM method
 Author: Yin Xuyue 
 Date:  2016/05/08
 References: 
 Dependencis: Eigen 3
 Data: Dummy data
 ***************************************************/

#ifndef __SPAAM_SVD__
#define __SPAAM_SVD__

#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/StdVector>

using namespace std;
using Eigen::MatrixXd;
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;

class Correspondence_Pair
{
public:
	
	Vector3d worldPoint;//3d
	Vector2d screenPoint;//2d

public:
	//Default Constructor
	Correspondence_Pair()
	{
// 		worldPoint.setZero();
// 		screenPoint.setZero();
	}
	//Parameter Constructor
	Correspondence_Pair(double x1, double y1, double z1, double x2, double y2)
	{
		worldPoint(0)=x1;
 		worldPoint(1)=y1;
 		worldPoint(2)=z1;
 		screenPoint(0)=x2;
 		screenPoint(1)=y2;
	}
	//Default Destructor
	~Correspondence_Pair(){ }
};

class SPAAM_SVD
{
private:
	Vector3d fromShift;
	Vector3d fromScale;
	Vector2d toShift;
	Vector2d toScale;
	MatrixXd modMatrixWorld;
	MatrixXd modMatrixScreen;
	MatrixXd ProjMat_3x4;

public:
	double ProjGL3x4[16];//GL投影矩阵
	vector<Correspondence_Pair,aligned_allocator<Correspondence_Pair>> corr_points;

public:	
	SPAAM_SVD():modMatrixScreen(3,3),modMatrixWorld(4,4),ProjMat_3x4(3,4)
	{
		cout<<"SVD solver initialized"<<endl;
	}
	~SPAAM_SVD()
	{

	}

public:
	//1. Normalization
	void estimateNormalizationParameters()
	{
		//count point pairs
		const size_t n_pts = distance(corr_points.begin(),corr_points.end());

		fromShift.setZero();
		fromScale.setZero();
		toShift.setZero();
		toScale.setZero();

		for(vector<Correspondence_Pair,aligned_allocator<Correspondence_Pair>>::const_iterator it(corr_points.begin());it<corr_points.end();++it)
		{
			fromShift = fromShift + (*it).worldPoint;
			fromScale = fromScale + (*it).worldPoint.cwiseProduct((*it).worldPoint);//如果维数错了，编译无法通过
			toShift = toShift + (*it).screenPoint;
			toScale = toScale + (*it).screenPoint.cwiseProduct((*it).screenPoint);
		}

		fromShift *= static_cast<double>(1)/n_pts;
		fromScale *= static_cast<double>(1)/n_pts;
		toShift   *= static_cast<double>(1)/n_pts;
		toScale   *= static_cast<double>(1)/n_pts;

		for(size_t i=0;i<3;i++)
			fromScale(i) = sqrt(fromScale(i)-(fromShift(i)*fromShift(i)));
		for(size_t i=0;i<2;i++)
			toScale(i) = sqrt(static_cast<double>(toScale(i)-(toShift(i)*toShift(i))));
	}

	void generateNormalizationMatrix()
	{
		modMatrixWorld.setZero();
		modMatrixScreen.setZero();

		modMatrixWorld(3,3)=static_cast<double>(1);
		modMatrixScreen(2,2)=static_cast<double>(1);

		for(size_t i=0;i<2;i++)
		{
			modMatrixScreen(i,i) = toScale(i);
			modMatrixScreen(i,2) = toShift(i);
		}

		for(size_t i=0;i<3;i++)
		{
			modMatrixWorld(i,i)=static_cast<double>(1)/fromScale(i);
			modMatrixWorld(i,3)=-modMatrixWorld(i,i)*fromShift(i);
		}
	}

	//2 DLT(Direct Linear Transform) implementation by svd(Singular value decomposition) 
	MatrixXd projectionDLTImpl()
	{
		assert(corr_points.size()>=6);
		estimateNormalizationParameters();
		MatrixXd A(2*corr_points.size(),12);
		
		//why unsigned? YXY
		for(unsigned i=0;i<corr_points.size();i++)
		{
			Vector2d to = (corr_points[i].screenPoint-toShift).cwiseQuotient(toScale);
			Vector3d from = (corr_points[i].worldPoint-fromShift).cwiseQuotient(fromScale);

			// 2,4,6....mean rows
			A( i * 2,  0 ) = A( i * 2, 1 ) = A( i * 2, 2 ) = A( i * 2, 3 ) = 0;
			A( i * 2,  4 ) = -from( 0 );
			A( i * 2,  5 ) = -from( 1 );
			A( i * 2,  6 ) = -from( 2 );
			A( i * 2,  7 ) = -1;
			A( i * 2,  8 ) = to( 1 ) * from( 0 );
			A( i * 2,  9 ) = to( 1 ) * from( 1 );
			A( i * 2, 10 ) = to( 1 ) * from( 2 );
			A( i * 2, 11 ) = to( 1 );
			// 1,3,5...odd rows
			A( i * 2+1 ,  0 ) = from( 0 );
			A( i * 2 +1,  1 ) = from( 1 );
			A( i * 2 +1,  2 ) = from( 2 );
			A( i * 2 +1,  3 ) = 1;
			A( i * 2 +1,  4 ) = A( i * 2 + 1, 5 ) = A( i * 2 + 1, 6 ) = A( i * 2 + 1, 7 ) = 0;
			A( i * 2 +1,  8 ) = -to( 0 ) * from( 0 );
			A( i * 2 +1,  9 ) = -to( 0 ) * from( 1 );
			A( i * 2 +1, 10 ) = -to( 0 ) * from( 2 );
			A( i * 2 +1, 11 ) = -to( 0 );
		}

		VectorXd s(12);//A.PROJ=0
		s.setZero();
		MatrixXd Vt(12,12);//column_major 12x12
		MatrixXd u(2*corr_points.size(),2*corr_points.size());
		//VectorXd O(2*corr_points.size());
		//O.setZero();
		//O(2*corr_points.size()-1)=1;

		/********************************************************************
		lapack中的SVD调用方法：
		boost::numeric::bindings::lapack::gesvd('N','A',A,s,U,Vt);A-jacobi;s-eigenvals;U-eigenvectors;Vt-solution
		
		Eigen中SVD的调用方法：

		method1:
		
		MatrixXf m(3,2); // 3x2
		JacobiSVD<MatrixXf> svd(m, ComputeThinU|ComputeThinV);

		singular value:        svd.singularValue()
		left singular vector:  svd.matrixU()
		right singular vector: svd.matrixV()

		Vector3f rhs(1,0,0); //3x1
		svd.solve(rhs);      //return x, mx=rhs;x 2x1
		******************************************************/
		
		JacobiSVD<MatrixXd> svd(A,ComputeThinU|ComputeThinV);
		u  = svd.matrixU();
		Vt = svd.matrixV();
		s   = svd.singularValues();
		Vt*=(-1);
		VectorXd g(12);
		g=Vt.col(11);
		ProjMat_3x4.row(0)= g.segment(0,4).transpose();
		ProjMat_3x4.row(1)= g.segment(4,4).transpose();
		ProjMat_3x4.row(2)= g.segment(8,4).transpose();
		//cout<<"---------The eigenvalues are : -------" << endl <<svd.singularValues()<<endl;
	    //cout<<"---------The Vt is : --------"<< endl <<Vt.col(11)<<endl;

// 		ProjMat_3x4( 0, 0 ) = Vt( 0,11 ); ProjMat_3x4( 0, 1 ) = Vt( 1,11 ); ProjMat_3x4( 0, 2 ) = Vt( 2,  11 ); ProjMat_3x4( 0, 3 ) = Vt( 3,11 );
// 		ProjMat_3x4( 1, 0 ) = Vt( 4, 11 ); ProjMat_3x4( 1, 1 ) = Vt( 5, 11 ); ProjMat_3x4( 1, 2 ) = Vt( 6, 11 ); ProjMat_3x4( 1, 3 ) = Vt( 7,11 );
// 		ProjMat_3x4( 2, 0 ) = Vt( 8, 11 ); ProjMat_3x4( 2, 1 ) = Vt( 9, 11 ); ProjMat_3x4( 2, 2 ) = Vt( 10, 11 ); ProjMat_3x4( 2, 3 ) = Vt( 11, 11 );

		generateNormalizationMatrix( );
		//cout<<"----------The uncorrected ProjMat_3x4 is:-------"<<endl<<ProjMat_3x4<<endl;
		//const ublas::matrix< double, boost::numeric::ublas::column_major > toCorrect(( modMatrixScreen ));
		///这句可能有问题，不了解ublas矩阵的初始化方式,ublas中，一次括号是矩阵copy式的初始化，两次是什么呢？
		///增加括号，不影响结果 061517
		MatrixXd toCorrect((modMatrixScreen));
		MatrixXd Ptemp(3,4);
		Ptemp = toCorrect*ProjMat_3x4;
		MatrixXd fromCorrect((modMatrixWorld));
		ProjMat_3x4 = Ptemp*fromCorrect;///nolias可能只是内存优化的方法

		double fViewDirLen = sqrt(ProjMat_3x4(2,0)*ProjMat_3x4(2,0)+ProjMat_3x4(2,1)*ProjMat_3x4(2,1)+ProjMat_3x4(2,2)*ProjMat_3x4(2,2));
		const VectorXd p1st(corr_points[0].worldPoint);

		if(ProjMat_3x4(2,0)*p1st(0)+ProjMat_3x4(2,1)*p1st(1)+ProjMat_3x4(2,2)*p1st(2)+ProjMat_3x4(2,3)<0)
			fViewDirLen = -fViewDirLen;
		ProjMat_3x4 *= double(1)/fViewDirLen;
		
		cout<<"----------The corrected ProjMat_3x4 is:-------"<<endl<<ProjMat_3x4<<endl;
		return ProjMat_3x4;
	}


	//从标定参数创建反投影相机
	void BuildGLMatrix3x4(float ne,float fr,int right,int left,int top,int bottom)
	{
		ProjGL3x4[0] = ProjMat_3x4(0, 0); ProjGL3x4[1] = ProjMat_3x4(0, 1); ProjGL3x4[2] = ProjMat_3x4(0, 2); ProjGL3x4[3] = ProjMat_3x4(0, 3);
		ProjGL3x4[4] = ProjMat_3x4(1, 0); ProjGL3x4[5] = ProjMat_3x4(1, 1); ProjGL3x4[6] = ProjMat_3x4(1, 2); ProjGL3x4[7] = ProjMat_3x4(1, 3);
		ProjGL3x4[8] = ProjMat_3x4(2, 0); ProjGL3x4[9] = ProjMat_3x4(2, 1); ProjGL3x4[10] = ProjMat_3x4(2, 2); ProjGL3x4[11] = ProjMat_3x4(2, 3);

		double* aproj = ProjGL3x4;//aproj 4x4
		double* proj4x4 = new double[16];

		constructProjectionMatrix4x4_(aproj,aproj,ne,fr,right,left,top,bottom);
	}
private:
	void constructProjectionMatrix4x4_(double*& final/*result*/, double* m, float ne, float fr,int right, int left,int top, int bottom)
	{
		double* proj4x4 = new double[16];

		//Copy base 3x4 values//
		memcpy(proj4x4, m, sizeof(double)*12); 		
		//Duplicate third row into the fourth//
		memcpy(proj4x4+12, m + 8, sizeof(double)*4);

		//calculate extra parameters//
		double norm = sqrt(proj4x4[8] * proj4x4[8] + proj4x4[9] * proj4x4[9] + proj4x4[10] * proj4x4[10]);
		double add = fr*ne*norm;

		//Begin adjusting the 3x4 values for 4x4 use//
		proj4x4[8] *= (-fr - ne);
		proj4x4[9] *= (-fr - ne);
		proj4x4[10] *= (-fr - ne);
		proj4x4[11] *= (-fr - ne);
		proj4x4[11] += add;	

		//Create Orthographic projection matrix//
		double* ortho = new double[16];
		ortho[0] = 2.0f / (right - left);
		ortho[1] = 0.0f;
		ortho[2] = 0.0f;
		ortho[3] = (right + left) / (left - right);
		ortho[4] = 0.0f;
		ortho[5] = 2.0f / (top - bottom);
		ortho[6] = 0.0f;
		ortho[7] = (top + bottom) / (bottom - top);
		ortho[8] = 0.0f;
		ortho[9] = 0.0f;
		ortho[10] = 2.0f / (ne - fr);
		ortho[11] = (fr + ne) / (ne - fr);
		ortho[12] = 0.0f;
		ortho[13] = 0.0f;
		ortho[14] = 0.0f;
		ortho[15] = 1.0f;

		//Multiply the 4x4 projection by the orthographic projection//
		///可以用Eigen中的cwiseproduct代替
		final[0] = ortho[0]*proj4x4[0] + ortho[1]*proj4x4[4] + ortho[2]*proj4x4[8] + ortho[3]*proj4x4[12];
		final[1] = ortho[0]*proj4x4[1] + ortho[1]*proj4x4[5] + ortho[2]*proj4x4[9] + ortho[3]*proj4x4[13];
		final[2] = ortho[0]*proj4x4[2] + ortho[1]*proj4x4[6] + ortho[2]*proj4x4[10] + ortho[3]*proj4x4[14];
		final[3] = ortho[0]*proj4x4[3] + ortho[1]*proj4x4[7] + ortho[2]*proj4x4[11] + ortho[3]*proj4x4[15];

		final[4] = ortho[4]*proj4x4[0] + ortho[5]*proj4x4[4] + ortho[6]*proj4x4[8] + ortho[7]*proj4x4[12];
		final[5] = ortho[4]*proj4x4[1] + ortho[5]*proj4x4[5] + ortho[6]*proj4x4[9] + ortho[7]*proj4x4[13];
		final[6] = ortho[4]*proj4x4[2] + ortho[5]*proj4x4[6] + ortho[6]*proj4x4[10] + ortho[7]*proj4x4[14];
		final[7] = ortho[4]*proj4x4[3] + ortho[5]*proj4x4[7] + ortho[6]*proj4x4[11] + ortho[7]*proj4x4[15];

		final[8] = ortho[8]*proj4x4[0] + ortho[9]*proj4x4[4] + ortho[10]*proj4x4[8] + ortho[11]*proj4x4[12];
		final[9] = ortho[8]*proj4x4[1] + ortho[9]*proj4x4[5] + ortho[10]*proj4x4[9] + ortho[11]*proj4x4[13];
		final[10] = ortho[8]*proj4x4[2] + ortho[9]*proj4x4[6] + ortho[10]*proj4x4[10] + ortho[11]*proj4x4[14];
		final[11] = ortho[8]*proj4x4[3] + ortho[9]*proj4x4[7] + ortho[10]*proj4x4[11] + ortho[11]*proj4x4[15];

		final[12] = ortho[12]*proj4x4[0] + ortho[13]*proj4x4[4] + ortho[14]*proj4x4[8] + ortho[15]*proj4x4[12];
		final[13] = ortho[12]*proj4x4[1] + ortho[13]*proj4x4[5] + ortho[14]*proj4x4[9] + ortho[15]*proj4x4[13];
		final[14] = ortho[12]*proj4x4[2] + ortho[13]*proj4x4[6] + ortho[14]*proj4x4[10] + ortho[15]*proj4x4[14];
		final[15] = ortho[12]*proj4x4[3] + ortho[13]*proj4x4[7] + ortho[14]*proj4x4[11] + ortho[15]*proj4x4[15];

		proj4x4[0] = final[0]; proj4x4[1] = final[4]; proj4x4[2] = final[8]; proj4x4[3] = final[12];
		proj4x4[4] = final[1]; proj4x4[5] = final[5]; proj4x4[6] = final[9]; proj4x4[7] = final[13];
		proj4x4[8] = final[2]; proj4x4[9] = final[6]; proj4x4[10] = final[10]; proj4x4[11] = final[14];
		proj4x4[12] = final[3]; proj4x4[13] = final[7]; proj4x4[14] = final[11]; proj4x4[15] = final[15];

		//copy final matrix values//
		for (int i = 0; i < 16; i++)
		{
			final[i] = proj4x4[i];
			cout<<"the i-th item of proj4x4 is: "<<final[i]; 
		}

		//clean up//
		delete [] ortho;
		delete [] proj4x4;
	}
};
#endif