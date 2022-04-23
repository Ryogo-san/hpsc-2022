#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range,0); 

  // バケツ作成
#pragma omp parallel for
  for (int i=0; i<n; i++)
#pragma omp atomic update
    bucket[key[i]]++;

  std::vector<int> offset(range,0);
  std::vector<int> tmp(range,0);

#pragma omp parallel for
  for (int i=1;i<range;i++){
#pragma omp parallel for
      for(int j=i-1;j>=0;j--) tmp[i]+=bucket[j];
  }

#pragma omp parallel for
  for (int i=1;i<range;i++)
   offset[i]=tmp[i];

#pragma omp parallel for
  for (int i=0; i<range; i++) {
    int j = offset[i];
#pragma omp parallel for shared(i,j)
    for (int k=bucket[i]; k>0; k--) {
      key[j] = i;
#pragma omp atomic update
      j+=1;
    }
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
