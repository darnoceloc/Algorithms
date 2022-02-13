#include "stdio.h"
#include "math.h"
#include "stdlib.h"

int exist(int** values, int count, int res) {
    for (int i = 0; i < count; i++) {
        if (values[i][2] == res) return 1;
    }
    return 0;
}

int main() {

    int** values = malloc(sizeof(int*) * 20);
    int count = 0;

    for (int i = 1; i < 21; i++) {
        for (int j = 1; j < 21; j++) {
            for (int k = 1; k < 21; k++) {
                if (pow(i, 2) + pow(j, 2) == pow(k, 2) && !exist(values, count, k)) {
                    int* arr = malloc(sizeof(int) * 3);
                    arr[0] = i;
                    arr[1] = j;
                    arr[2] = k;
                    values[count++] = arr;
                }
            }
        }
    }

    int* temp = malloc(sizeof(int*)*3);
    for (int i = 0; i < count; i++) {
        for(int j = i+1; j < count; j++) {
            if(values[i][2] > values[j][2]){
                temp = values[j];
                values[j] = values[i];
                values[i] = temp;
            }
        }
        int* triple = values[i];
        printf("%d %d %d\n", triple[0], triple[1], triple[2]);
        free(values[i]);
    }
    free(temp);
    free(values);

}
