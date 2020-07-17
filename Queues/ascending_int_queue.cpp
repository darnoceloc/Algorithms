#include <iostream>
#include <queue>
#include <deque>
#include <iterator>

template<typename T, typename Container=std::deque<T> >
class iterable_queue : public std::queue<T,Container>
{
public:
    typedef typename Container::iterator iterator;
    typedef typename Container::const_iterator const_iterator;

    iterator begin() { return this->c.begin(); }
    iterator end() { return this->c.end(); }
    const_iterator begin() const { return this->c.begin(); }
    const_iterator end() const { return this->c.end(); }
};


bool checkValidity(iterable_queue<int>& q) {
    if (q.empty() || q.size() <= 1){
        std::cout << "invalid entry, insufficient elements" << '\n';
        return false;
    }
    while(q.size()){
        auto i = q.begin();
        auto j = ++q.begin();
        for(; i < q.end() && j < ++q.end();){
            std::cout << *i << " " << *j << '\n';
            if (*(i) > *(j)) {
                std::cout << "invalid entry, not properly sorted" << '\n';
                return false;
            }
            i++, j++;
        }   
        std::cout << "valid entry, properly sorted" << '\n';
        return true;
    }
    std::cout << "invalid entry, insufficient elements" << '\n';
    return false;
}

const char* bool_cast(const bool b) {
    return b ? "true" : "false";
}        
    
int main () {
    iterable_queue<int> numbers;
	int temp;
	
	std::cout << "Pushing..." << '\n';
	while(temp >= 0){
		std::cout << "Enter numbers: ";
		std::cin >> temp;
		if(temp >= 0){
			numbers.push(temp);
        }
	}

    bool ck = checkValidity(numbers);
    std::cout << bool_cast(ck) << '\n';

    std::cout << "{ ";
	while(numbers.size() > 0){
		std::cout << numbers.front();
		numbers.pop();
		std::cout << " ";
	}
	std::cout << "}" << '\n';

    return 0;
}
