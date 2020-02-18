// Overloading operators for Array class
#include<iostream>
#include<cstdlib>


// A class to represent an integer array
class Array{
private:
    int *ptr;
    std::size_t size;
public:
    Array(int *p = nullptr, std::size_t s = 0);
    Array(const Array&);
    ~Array();
    Array& operator= (Array);

    // Overloading [] operator to access elements in array style
    int &operator[] (std::size_t);

    int const& operator[](std::size_t) const;

    // Utility function to print contents
    void print() const;

    friend void swap(Array& first, Array& second);};
           

// Implementation of [] operator.  This function must return a
// reference as array element can be put on left side
int &Array::operator[](std::size_t index){   
    if (index >= size || index < 0){
       throw std::out_of_range("Index out of Range error");
    }
    return ptr[index];
}
    
// constructor for array class
Array::Array(int *p, std::size_t s){
    size = s;
    ptr = nullptr;
    if (s != 0){
        ptr = new int[s];
        for (int i = 0; i < s; i++)
            ptr[i] = p[i];}
}

// destructor for array class
Array::~Array(){
    delete[] ptr;
    ptr = nullptr;}

// copy constructor for array class
Array::Array(const Array& A) { 
    size = A.size;
    ptr  = new int[size];
    for (int i = 0; i < size; i++)
        ptr[i] = A.ptr[i];}

//swap friend function of assignment operator
void swap(Array& first, Array& second){
    using std::swap;
    swap(first.size, second.size);
    swap(first.ptr, second.ptr);}

//Assignment operator for array class
Array& Array::operator=(Array other){
    swap(*this, other); 
    return *this;}

//print function for array elements
void Array::print() const{
    std::cout << "{";
    for(int i = 0; i < size; i++)
        std::cout<<ptr[i]<<" ";
    std::cout<<"}"<<std::endl;}

// Driver program to test above methods
int main()
{
    int a[] = {1, 2, 3, 4, 5, 6};
    Array arr1(a, 6);
    arr1[0] = 7;
    arr1.print();
    Array arr2 = arr1;
    arr2.print();
    arr1[-1] = 4;
    return 0;
} 
