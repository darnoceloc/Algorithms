#include <iostream>

class Node {
    public:
        int value;
        Node* next = nullptr;
};

int size = 0; 

Node* getNode(const int& data) 
{ 
    // allocating space 
    Node* newNode = new Node(); 
    // inserting the required data 
    newNode->value = data; 
    return newNode; 
} 

void add(Node** head, int& index, const int& valueInput){
    if (index < 1 || index > size + 1){
        std::cout << "Invalid position!" << std::endl;
        return;
    }
    else if ( head == nullptr ){ // Check if pointer is invalid
        std::cout << "Invalid head pointer !" << std::endl;
    }
    else { 
        Node *pCurrentNode = *head;
        Node *pPreviousNode = *head;
    

    // Keep looping until the pos is zero 
        while (index--) { 

            if (index == 0) { 
                // adding Node at required position 
                Node* temp = getNode(valueInput); 

                // Insert new node BEFORE current node
                temp->next = pCurrentNode; 

                // Current != previous if we are not in a head insertion case
                if ( pCurrentNode != pPreviousNode)
                    pPreviousNode->next = temp; // insert new node AFTER previous node
                else 
                    *head = temp; // Update de content of HEAD and not juste it ADRESS!

                size++; // Increment size when we ADD a new node
            } 
            else
            {
                pPreviousNode = pCurrentNode; // Save previous node pointer
                pCurrentNode = pCurrentNode->next; // Get pointer of next node
            }
        }
    }
} 

void freeList(struct Node* head) { 
    Node* pCurrentNode = head;
    while ((pCurrentNode = head) != nullptr) 
    { 
        head = head->next;          
        delete pCurrentNode;               
    }
}  

void printList(struct Node* head) 
{ 
    while (head != nullptr) { 
        std::cout << " " << head->value; 
        head = head->next; 
    } 
    std::cout << std::endl; 
} 


int main() 
{ 
    // Creating the list 3->5->8->10 
    Node* head = nullptr; 
    head = getNode(3); 
    head->next = getNode(5); 
    head->next->next = getNode(8); 
    head->next->next->next = getNode(10); 
  
    size = 4; 
  
    std::cout << "Linked list before insertion: "; 
    printList(head); 
  
    int data = 12, pos = 3; 
    add(&head, pos, data); 
    std::cout << "Linked list after insertion of 12 at position 3: "; 
    printList(head); 
  
    // front of the linked list 
    data = 1, pos = 1; 
    add(&head, pos, data); 
    std::cout << "Linked list after insertion of 1 at position 1: "; 
    printList(head); 
  
    // insetion at end of the linked list 
    data = 15, pos = 9; 
    add(&head, pos, data); 
    std::cout << "Linked list after insertion of 15 at position 7: "; 
    printList(head); 

    freeList(head);
  
    return 0; 
} 
