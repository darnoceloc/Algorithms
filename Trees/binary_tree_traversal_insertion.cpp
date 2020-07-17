#include <iostream>
#include <string>

class Node {
    public:
        std::string name;
        Node* left = nullptr;
        Node* right = nullptr;
        std::string leftName;
        std::string rightName;

    void getData(){
        char word[5];
        std::cout << "Enter names of nodes" << '\n';
        std::cin >> word;
        std::cin >> this->name;
        std::cin >> word;
        std::cin >> this->leftName;
        std::cin >> word;
        std::cin >> this->rightName;
    }

    void attachLeft(Node* nodes, int size){
        for (int i=0; i<size; i++)
        {
            if (nodes[i].name == this->leftName)
            {
                left = &nodes[i];
            }
        }
    }

    void attachRight(Node* nodes, int size){
        for (int i=0; i<size; i++)
        {
            if (nodes[i].name == this->rightName)
            {
               right = &nodes[i];          
            }
        }
    }
};   


void preTraverse(Node *head) {
	if(head == nullptr) {
        return;
    }
    std::cout << head->name;
    preTraverse(head->left);     // Visit left subtree
	preTraverse(head->right);    // Visit right subtree
}

void inTraverse(Node* head){
	if(head == nullptr) {
        return;
    }
    inTraverse(head->left);       //Visit left subtree
    std::cout << head->name;
    inTraverse(head->right);      // Visit right subtree   
}


void posTraverse(Node* head){
	if(head == nullptr) {
        return;
    }
    posTraverse(head->left);       //Visit left subtree
    posTraverse(head->right);      // Visit right subtree 
    std::cout << head->name;
}

int main() {
    int number;
    std::cout << "Enter size of tree" << '\n';
    std::cin >> number;
    Node treeNodes[number];
    for (int i=0; i<number; i++) {
        treeNodes[i].getData();
    }
    for (int i=0; i<number; i++) {
        treeNodes[i].attachLeft(treeNodes, number);
        treeNodes[i].attachRight(treeNodes, number);
    }
    preTraverse(&treeNodes[number-1]);
    std::cout << '\n';
    inTraverse(&treeNodes[number-1]);
    std::cout << '\n';
    posTraverse(&treeNodes[number-1]);
    std::cout << std::endl;
    return 0;
}
