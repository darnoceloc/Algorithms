#include "stdio.h"

int main(){
    int n, reversed = 0, orig, remainder;
    scanf("%d", &n);
    orig = n;
     while (n != 0) {
        remainder = n % 10;
        reversed = (reversed * 10) + remainder;
        n /= 10;
    }
    if(orig == reversed){
        printf("true\n");
    }
    else{
        printf("false\n");
    }
    return 0;
}
