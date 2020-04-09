#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>
#include <cmath>

int readheap(int* theheap) {
    std::string num;
    int val = 0;
    int size = 0;
    std::cout << "Enter the elements of heap" << std::endl;
    std::getline(std::cin, num);
    std::istringstream iss(num);    
    while (iss >> val) {
        if (val != ' ' && val > 0){
            theheap[size] = val;
            ++size;
        }
    }
    if(size <= 1) {
        throw std::runtime_error("Invalid user input");
    }

for (int k = 1; k < size; ++k) {

        if (theheap[k] > theheap[(k - 1) / 2])  { 
            int j = k; 

            while (theheap[j] > theheap[(j - 1) / 2])  { 
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

            if (theheap[index] < theheap[index + 1] && index < (k - 1)) 
                ++index; 

            if (theheap[j] < theheap[index] && index < k) 
                std::swap(theheap[j], theheap[index]); 

            j = index; 

        } while (index < k); 
    } 
    std::cout << "Size of heap is " << size << '\n';
    return size;
}

void heapRemove(int* theheap, int& size) {
    for(int i=0; i<size-1; ++i){
        theheap[i] = theheap[i+1];
    }
    int* x = theheap;
    x = nullptr;
    delete x;
    size--; 
    for (int k = 1; k < size; ++k) {

        if (theheap[k] > theheap[(k - 1) / 2])  { 
            int j = k; 

            while (theheap[j] > theheap[(j - 1) / 2])  { 
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

            if (theheap[index] < theheap[index + 1] && index < (k - 1)) 
                ++index; 

            if (theheap[j] < theheap[index] && index < k) 
                std::swap(theheap[j], theheap[index]); 

            j = index; 

        } while (index < k); 
    }
}

void heapPrint(int* theheap, int size) {
    for (int i = 0; i < size; ++i){
        std::cout << theheap[i] << ' ';
    }
    std::cout << '\n';
}

int main() {
    int* theheap = new int[10];
    int size = readheap(theheap);
    heapPrint(theheap, size);
    heapRemove(theheap, size);
    heapPrint(theheap, size);
    std::cout << size << std::endl;
    return 0;
}
