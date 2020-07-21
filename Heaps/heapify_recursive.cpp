#include <iostream>
#include <sstream>

int readHeap(int * theheap){
  int input;
  int i = 0;
  while (true) {
    std::cin >> input;
    if (input == -1){
      break;
     }
    theheap[i] = input;
    i++;
  }
  return i;
}

void heapify(int * arr, int n, int i) {
  int smallest = i; // Initialize largest as root
  int l = 2 * i + 1; // left = 2*i + 1
  int r = 2 * i + 2; // right = 2*i + 2
  // If left child is smaller than root
  if (l < n && arr[l] < arr[smallest]){
    smallest = l;
   }
  // If right child is smaller than smaller so far
  if (r < n && arr[r] < arr[smallest]){
    smallest = r;
   }
  // If largest is not root
  if (smallest != i) {
    int temp = arr[i];
    arr[i] = arr[smallest];
    arr[smallest] = temp;
  // Recursively heapify the affected sub-tree
  heapify(arr, n, smallest);
  }
}

void heapRemove(int * theheap, int &size) {
  int index = 0;
  theheap[index] = theheap[--size];
  heapify(theheap, size, 0);
}

void heapPrint(int * theheap, int size){
  int i = 0;
  while(i < size){
    std::cout << theheap[i] << " ";
    i++;
  }
}
