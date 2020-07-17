
#include <iostream>
#include <list>

class Node
{
    public:
        int name;
        Node* left = nullptr;
        Node* right = nullptr;

        Node* insert(Node* root, int key) {
            if(root == nullptr)
            {
                Node* temp = new Node();
                temp->name = key;
                return temp;
            }
            if (key < root->name)
                root->left  = insert(root->left, key);
            else if (key > root->name)
                root->right = insert(root->right, key);   

            return root;
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

int height(Node* root) {
    if (root == nullptr) return 0;
    return 1 + std::max(height(root->left), height(root->right));
}


bool isAVL(Node* root)
{
     if(root == nullptr){
        return true;
    }
   
    int height_right = height(root->right);
    int height_left = height(root->left);
    
    if(abs(height_right - height_left) <=1 && isAVL(root->left) && isAVL(root->right)){
        return true;
    }
    
    else{
        return false;
    }
} 

  

int main() {
    Node* bst = new Node();
    Node* head = nullptr;
    head = bst->insert(head, 11);
    head = bst->insert(head, 5);
    head = bst->insert(head, 2);
    head = bst->insert(head, 15);
    head = bst->insert(head, 16);
    head = bst->insert(head, 9);
    head = bst->insert(head, 14);
    bst->inTraverse(head);
    std::cout << '\n';
    std::cout << std::boolalpha;   
    std::cout << isAVL(head) << std::endl;
    return 0;
}
