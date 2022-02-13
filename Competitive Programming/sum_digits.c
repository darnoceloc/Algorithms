#include <stdio.h>
#include <stdlib.h>

int sum_digits(int i){
    int sum = 0;
    i = abs(i);
    while(i > 1){
        sum += i % 10;
        i /= 10;
    }
    return sum;
}

int main(){
    int i = 0, n, sz;
    scanf("%d", &n);
    sz = n;
    int* arr = malloc(sizeof(int*)*n);
    while(n > 0){
        int val;
        scanf("%d", &val);
        arr[i] = sum_digits(val);
        i++;
        n--;
    }
    int max = arr[0];
    for (int c = 1; c < sz; c++) {
        if (arr[c] > max){
            max = arr[c];
        }
    }
    free(arr);
    printf("%d", max);
        
    return 0;
}
