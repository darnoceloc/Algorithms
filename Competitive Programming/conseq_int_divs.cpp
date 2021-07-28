#include <iostream>
#include <math.h>

using namespace std;

int func(const int& A, const int& B){
    int res = 0;
    int j = A;
    for (int i = 2; j <= B && pow(i,2)+i <= B;){
       if(fmod(pow(i, 2) + i, j) == 0){
           res++;
           i++;
           j++;
        }
        else{
          j++;
        }
   }
  return res;
}

int gunc(const int& C, const int& D){
    int res = 0;
    int k = 2;
    for (int m = C; m <= D; m++){
       if(fmod(m, k) == 0 && fmod(m, k+1) == 0){
           res++;
           m++;
           k++;
        }
        else{
          continue;
        }
    }
   return res;
}





int main(){
    int x = func(2, 117);
    int y = gunc(2, 117);
    cout << "x " << x << endl;
    cout << "y " << y << endl;
    return 0;
}
