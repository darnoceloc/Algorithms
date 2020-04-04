#include <iostream>
#include <stdexcept>

class Node {
    public:
        int value;
        Node* next = nullptr;
        ~Node(){
            std::cout << "destructor called" << std::endl;
            Node* current;
            Node* next;
            
            while (current != NULL) {
                next = current->next;
                delete current;
                current = next;
            }
        }
        
};

int find(Node* head, int n) {
    int index = -1;
    while (head) {
        ++index;
        if (head->value == n) {
            std::cout << "value found at index " << index << std::endl;
            return index;
        }
        head = head->next;
    }
    std::cout << "value not found " << std::endl;
    return -1;
}   



template<typename T>
class LinkedList {
    struct Node {
            T value;
            Node* next;
    };
    Node* root;

    public:
        LinkedList()
            : root(nullptr)
        {}
        ~LinkedList() {
            while(root) {
                Node* next = root->next;
                delete root;
                root = next;
            }
        }
        LinkedList(LinkedList const&)            = delete;
        LinkedList& operator=(LinkedList const&) = delete;

        void push(T const& new_data) 
        {
            root= new Node{new_data, root};
        }

        int find(T x) { 
            Node* result = findElement(root, x);
            if (result == nullptr) {
                std::cout << "value not found " << std::endl;
                return -1; 
            }
            std::cout << "value found " << std::endl;
            return result->value;
        }
    private:
        Node* findElement(Node* n, T val) {
            if (!n) {
                return nullptr;
            }
            if (n->value == val){
                 return n;
            }

            return findElement(n->next, val);
        } 
};

int main() { 
    LinkedList<int> llist; 
    llist.push(1); 
    llist.push(4); 
    llist.push(8); 
    llist.push(12); 
    llist.push(7);   
     
    int x = llist.find(1); 

    Node *list = new Node{1, new Node{2, new Node{3, new Node{7, new Node{9, new Node{11, nullptr}}}}}};     
    int idx = find(list, 9);  
    std::cout << idx << std::endl;    
  
    return 0;
} 