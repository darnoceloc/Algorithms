#include <iostream>
#include <string>
#include <sstream>

int readHeap(int* theheap) {
    std::string num;
    int val = 0;
    int size = 0;
    std::getline(std::cin, num);
    std::istringstream iss(num);    
    while (iss >> val) {
        if (val != ' ' && val > 0){
            theheap[size] = val;
            ++size;
        }
    }
    if(size < 1) {
        return -1;
    }
    return size;
}

void heapRemove(int* theheap, int& size) {
    int index = 0;
    theheap[index] = theheap[--size];
    int l, r, smallest;
    while(index < size){
        l = index*2 + 1;
        r = index*2 + 2;
        smallest = index;
        if(l < size && theheap[l] < theheap[smallest])
            smallest = l;
        if(r < size && theheap[r] < theheap[smallest])
            smallest = r;
        if (smallest != index)
        {
            int temp = theheap[index];
            theheap[index] = theheap[smallest];
            theheap[smallest] = temp;
            index = smallest;
            continue;
        }
        break;
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
