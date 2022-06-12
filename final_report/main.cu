#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <tuple>
#include <cmath>
#include <iterator>
#include <omp.h>
//#include <matplotlib-cpp/matplotlibcpp.h>
//namespace plt = matplotlibcpp;

const int M=1024;
const int nx=41;
const int ny=41;
const int nt=500;
const int nit=500;
const double dx=2./(double)(nx-1);
const double dy=2./(double)(ny-1);
const double dt=0.01;
const int rho=1;
const double nu=0.02;

__global__ void linspace(double *target,double min,double max,int n){
    double dx=(max-min)/(double)(n-1);
    int ii;
    ii=threadIdx.x;
    if(ii>=n)
        return;
    else if(ii==n-1)
        target[ii]=max;
    else
        target[ii]=(double)ii*dx+min;
}

__global__ void meshgrid(double *X, double *Y,double* vec1, double* vec2, int N){
    int jx=blockDim.x*blockIdx.x+threadIdx.x;
    int jy=blockDim.y*blockIdx.y+threadIdx.y;
    int i=nx*jy+jx;
    if(i>=N) return;
    X[i]=vec1[jx];
    Y[i]=vec2[jy];
}

__global__ void bBlock(double *b,double *u,double *v,int N){
    int jx=blockDim.x*blockIdx.x+threadIdx.x;
    int jy=blockDim.y*blockIdx.y+threadIdx.y;
    int i=nx*jy+jx;
    if(i>=N) return;
    if(jy>0&&jy<ny-1&&jx>0&&jx<nx-1){
        b[i]=rho*(1/dt*\
                ((u[i+1]-u[i-1])/(2*dx)+(v[i+nx]-v[i-nx])/(2*dy))-\
                std::pow(((u[i+1]-u[i-1])/(2*dx)),2)-2*((u[i+nx]-u[i-nx])/(2*dy)*\
                    (v[i+1]-v[i-1])/(2*dx)-std::pow(((v[i+nx]-v[i-nx])/(2*dy)),2)));
    }
}

__global__ void pBlock(double *p, double *pn, double *b,int N){
    int jx=blockDim.x*blockIdx.x+threadIdx.x;
    int jy=blockDim.y*blockIdx.y+threadIdx.y;
    int i=nx*jy+jx;
    if(i>=N) return;
    if(jy>0&&jy<ny-1&&jx>0&&jx<nx-1){
        p[i]=(std::pow(dy,2)*(pn[i+1]+pn[i-1])+\
                std::pow(dx,2)*(pn[i+nx]+pn[i-nx])-\
                b[i]*std::pow(dx,2)*std::pow(dy,2));
    }
}

__global__ void uvBlock(double *u,double *v,double *un,double *vn,double *p,int N){
    int jx=blockDim.x*blockIdx.x+threadIdx.x;
    int jy=blockDim.y*blockIdx.y+threadIdx.y;
    int i=nx*jy+jx;
    if(i>=N) return;
    if(jy>0&&jy<ny-1&&jx>0&&jx<nx-1){
        u[i]=un[i]-un[i]*dt/dx*(un[i]-un[i-1])\
             -un[i]*dt/dy*(un[i]-un[i-nx])\
             -dt/(2*rho*dx)*(p[i+1]-p[i-1])\
             +nu*dt/std::pow(dx,2)*(un[i+1]-2*un[i]+un[i-1])\
             +nu*dt/std::pow(dy,2)*(un[i+nx]-2*un[i]+un[i-nx]);
        v[i]=vn[i]-vn[i]*dt/dx*(vn[i]-vn[i-1])\
             -vn[i]*dt/dy*(vn[i]-vn[i-nx])\
             -dt/(2*rho*dx)*(p[i+nx]-p[i-nx])\
             +nu*dt/std::pow(dx,2)*(vn[i+1]-2*vn[i]+vn[i-1])\
             +nu*dt/std::pow(dy,2)*(vn[i+nx]-2*vn[i]+vn[i-nx]);
    }
}
__global__ void copy(double *dst, double *src, int N){
    int jx=blockDim.x*blockIdx.x+threadIdx.x;
    int jy=blockDim.y*blockIdx.y+threadIdx.y;
    int i=nx*jy+jx;
    if(i>=N) return;
    dst[i]=src[i];
}

int main(){
    double* x=NULL;
    double* y=NULL;
    double* u=NULL;
    double* v=NULL;
    double* p=NULL;
    double* b=NULL;
    double* X=NULL;
    double* Y=NULL;
    const int N=nx*ny;
    cudaMallocManaged(&x,nx*sizeof(double));
    cudaMallocManaged(&y,ny*sizeof(double));
    linspace<<<(N+M-1)/M,M>>>(x,0.,2.,nx);
    cudaDeviceSynchronize();
    linspace<<<(N+M-1)/M,M>>>(y,0.,2.,ny);
    cudaDeviceSynchronize();
    // zeros
    cudaMallocManaged(&u,ny*nx*sizeof(double));
    cudaMemset(u,0,sizeof(double)*ny*nx);
    cudaMallocManaged(&v,ny*nx*sizeof(double));
    cudaMemset(v,0,sizeof(double)*ny*nx);
    cudaMallocManaged(&p,ny*nx*sizeof(double));
    cudaMemset(p,0,sizeof(double)*ny*nx);
    cudaMallocManaged(&b,ny*nx*sizeof(double));
    cudaMemset(b,0,sizeof(double)*ny*nx);

    /*
    for(int i=0;i<ny;i++){
        for(int j=0;j<nx;j++)
            std::cout<<u[i*ny+j]<<" ";
        std::cout<<std::endl;
    }
    */

    cudaMallocManaged(&X,ny*nx*sizeof(double));
    cudaMallocManaged(&Y,ny*nx*sizeof(double));

    // meshgrid
    meshgrid<<<(N+M-1)/M,M>>>(X,Y,x,y,ny*nx);
    cudaDeviceSynchronize();

    for(int n=0;n<nt;n++){
        bBlock<<<(N+M-1)/M,M>>>(b,u,v,ny*nx);
        cudaDeviceSynchronize();
        
        for(int it=0;it<nit;it++){
            double* pn=NULL;
            cudaMallocManaged(&pn,ny*nx*sizeof(double));
            copy<<<(N+M-1)/M,M>>>(pn,p,ny*nx);
            cudaDeviceSynchronize();
            
            pBlock<<<(N+M-1)/M,M>>>(p, pn, b, ny*nx);
            cudaDeviceSynchronize();

#pragma omp parallel for
            for(int i=0;i<ny;i++) p[i*nx+nx-1]=p[i*nx+nx-2];
#pragma omp parallel for
            for(int j=0;j<nx;j++) p[0+j]=p[nx+j];
#pragma omp parallel for
            for(int i=0;i<ny;i++) p[i*nx]=p[i*nx+1];
#pragma omp parallel for
            for(int j=0;j<nx;j++) p[(ny-1)*nx+j]=0;
     
            cudaFree(pn);
        }

        double* un=NULL;
        cudaMallocManaged(&un,ny*nx*sizeof(double));
        copy<<<(N+M-1)/M,M>>>(un,u,ny*nx);
        cudaDeviceSynchronize();
        double* vn=NULL;;
        cudaMallocManaged(&vn,ny*nx*sizeof(double));
        copy<<<(N+M-1)/M,M>>>(vn,v,ny*nx);
        cudaDeviceSynchronize();

        uvBlock<<<(N+M-1)/M,M>>>(u,v,un,vn,p,ny*nx);
        cudaDeviceSynchronize();

#pragma omp parallel for
        for(int j=0;j<nx;j++) u[j]=0;
#pragma omp parallel for
        for(int i=0;i<ny;i++) u[i*nx]=0;
#pragma omp parallel for
        for(int i=0;i<ny;i++) u[i*nx+nx-1]=0;
#pragma omp parallel for
        for(int j=0;j<nx;j++) u[(ny-1)*nx+j]=1;
#pragma omp parallel for
        for(int j=0;j<nx;j++) v[j]=0;
#pragma omp parallel for
        for(int j=0;j<nx;j++) v[(ny-1)*nx+j]=0;
#pragma omp parallel for
        for(int i=0;i<ny;i++) v[i*nx]=0;
#pragma omp parallel for
        for(int i=0;i<ny;i++) v[i*nx+nx-1]=0;

        //plt::contourf(X,Y,p,0.5);
        //plt::quiver(X,Y,u,v);
        //plt::pause(0.01);
        //plt::clf();
        cudaFree(un);
        cudaFree(vn);
    }
    //plt::show();
    
    cudaFree(x);
    cudaFree(y);
    cudaFree(u);
    cudaFree(v);
    cudaFree(p);
    cudaFree(b);
    cudaFree(X);
    cudaFree(Y);
    return 0;
}
