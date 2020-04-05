#include <iostream>

class Node {
    public:
        int name;
        Node* left = nullptr;
        Node* right = nullptr;
        
        static Node* newNode(const int& data){
            Node* n = new Node();
            n->left = nullptr;
            n->right = nullptr;
            n->name = data;
            return n;
        }

    Node* insert(Node* root, int key) {
        Node* tempRoot = root;
        Node* new_node = Node::newNode(key);

        if (root == nullptr){
            root = new_node;
            return root;
        }
        Node* prev = nullptr;
            while(root != nullptr){
                prev = root;
                if(root->name < key){
                    root = root->right;
                }else{
                    root = root->left;
                }
            }
            if(prev->name < key){
                prev->right = new_node;
            }else{
                prev->left = new_node;
            }
            return tempRoot;

    }

        void inTraverse(Node* head){
        if(head == nullptr) {
            return;
        }
        inTraverse(head->left);       //Visit left subtree
        std::cout << head->name << " ";
        inTraverse(head->right);      // Visit right subtree   
    }

};


int main(){
    Node* bst = new Node();
    Node* head = nullptr;
    head = bst->insert(head, 10);
    head = bst->insert(head, 15);
    head = bst->insert(head, 5);
    head = bst->insert(head, 7);
    head = bst->insert(head, 19);
    head = bst->insert(head, 20);
    head = bst->insert(head, -1);
    head = bst->insert(head, 21);
    bst->inTraverse(head);
    return 0;
}