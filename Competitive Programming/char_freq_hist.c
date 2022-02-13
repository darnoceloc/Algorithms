#include<stdio.h>
#include<string.h>
#define TOTAL_CHARS 128

int main() {
    int str[TOTAL_CHARS], c;
    char strput[80];
    
    for(int i = 0; i < TOTAL_CHARS; i++){
        str[i] = 0;
    }
    
    gets(strput);
    
    for(int j = 0; j < strlen(strput); j++){
        if(strput[j] >= 'A' && strput[j] <= 'Z'){
            strput[j] += 32;
            str[strput[j]] += 1;
        }
        else{
            str[strput[j]] += 1;
        }
    }
    
    for(int i = 31; i < TOTAL_CHARS; i++) {
        if(str[i] > 0){
            printf("%c", (char)i);
            printf(" %d\n", str[i]);
        }

    }
      
    return 0;
}
