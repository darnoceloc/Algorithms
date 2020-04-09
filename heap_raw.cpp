#include <iostream>
#include <string>
#include <sstream>

int readheap(int* theheap) {
    int i = 0;
    int num;
    std::cout << "Enter the elements of heap" << '\n';
    while (num >= 0) {
         std::cin >> num;
         if (num >= 0) {
            theheap[i] = num;
            ++i;
         }
    }
    for (int k = 1; k < i; ++k) {
    
        if (theheap[k] < theheap[(k - 1) / 2])  { 
            int j = k; 
           
            while (theheap[j] < theheap[(j - 1) / 2])  { 
                std::swap(theheap[j], theheap[(j - 1) / 2]); 
                j = (j - 1) / 2; 
            }
        }
    }
    for (int k = i - 1; k > 0; --k) {
        std::swap(theheap[0], theheap[k]);           
       
        int j = 0, index; 
          
        do { 
            index = (2 * j + 1);  
           
            if (theheap[index] > theheap[index + 1] && index < (k - 1)) 
                ++index; 
          
            if (theheap[j] > theheap[index] && index < k) 
                std::swap(theheap[j], theheap[index]); 
          
            j = index; 
          
        } while (index < k); 
    } 
    std::cout << "Size of heap is " << i << '\n';
    return i;
}

void heapRemove(int* theheap, int& size) {
    for(int i=0; i<size-1; ++i){
        theheap[i] = theheap[i+1];
    }
    int* x = theheap+size-1;
    x = nullptr;
    delete x;
    size--; 
    for (int k = 1; k < size; ++k) {
        
        if (theheap[k] < theheap[(k - 1) / 2])  { 
            int j = k; 
             
            while (theheap[j] < theheap[(j - 1) / 2])  { 
                std::swap(theheap[j], theheap[(j - 1) / 2]); 
                j = (j - 1) / 2; 
            }
        }
    }
    for (int k = size - 1; k > 0; --k) {  
        std::swap(theheap[0], theheap[k]);      
   
        int j = 0, index; 
          
        do { 
            index = (2 * j + 1);  
             
            if (theheap[index] > theheap[index + 1] && index < (k - 1)) 
                ++index; 
             
            if (theheap[j] > theheap[index] && index < k) 
                std::swap(theheap[j], theheap[index]); 
          
            j = index; 
          
        } while (index < k); 
    }
}

void heapPrint(int* theheap, int size) {
    for (int i = size-1; i >= 0; --i){
        std::cout << theheap[i] << ' ';
    }
    std::cout << '\n';
}

int main() {
    int* theheap = new int[10];
    int size = readheap(theheap);
    heapRemove(theheap, size);
    heapPrint(theheap, size);
}