#include <stdio.h>


void* search(void *arr, int n, void* val, char c){
    if(c == 'c'){
        char* cp = (char*) arr;
        char* valc = (char*) val;
        while (cp < (char*)arr+n){
            if(*cp == *valc){
                return cp;
            }
            cp++;
        }
        return NULL;
    }
    if (c == 'i'){
        int* ip = (int*) arr;
        int* vali = (int*) val;
        while (ip < (int*)arr+n){
            if(*ip == *vali){
                return ip;
            }
            ip++;
        }
        return NULL;
    }
    if (c == 'f'){
        float* fp = (float*) arr;
        float* valf = (float*) val;
        while (fp < (float*)arr+n){
            if(*fp == *valf){
                return fp;
            }
            fp++;
        }
        return NULL;
    }
}
