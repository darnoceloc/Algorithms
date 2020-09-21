#include <iostream>
#include <vector>

void minimumBribes(std::vector<int> q) {
    int swap = 0;
    int bribes;
    int pos = 0;
    for(int i = q.size()-1; i >= 0; i--) {
        int j = 0;

        bribes = q[pos] - (pos+1);
        if (bribes > 2) {
            std::cout << "Too chaotic" << std::endl;
            return;
        }
        if (q[i] - 2 > 0){
            j = q[i] - 2;
        }
        
        while(j <= i) {
            if (q[j] > q[i]){
                swap++;
            }
            j++;
        }
        pos++;
    }
    std::cout << swap << std::endl;
}

void minimumBribes2(std::vector<int> q) {
    bool chaotic = false;
    int  bribes = 0;
    for (int i = 0; i < q.size(); i++) {
        if (q[i] - (i+1) > 2) { 
            chaotic = true;
        }
        for (int j = q[i] - 2; j < i; j++) {
            if (q[j] > q[i]) { 
                bribes++; 
            }
        }
   }
   if(chaotic == true) {
       std::cout << "Too chaotic" << std::endl;
   } 
   else {
        std::cout << bribes << std::endl;
   }
}