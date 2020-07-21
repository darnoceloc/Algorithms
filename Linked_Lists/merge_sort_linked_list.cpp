class Node {
public:
        int value;
        Node* next;
        Node() {next = nullptr;}
        Node(int val) {value = val; next = nullptr;}
};

//Merge subroutine to merge two sorted lists
Node* merge(Node* a, Node* b) {
      Node* result = nullptr; 
  
    /* Base cases */
    if (a == nullptr){ 
        return (b);
    } 
    else if (b == nullptr) {
        return (a); 
    }
    /* Pick either a or b, and recur */
    if (a->value <= b->value) { 
        result = a; 
        result->next = merge(a->next, b); 
    } 
    else { 
        result = b; 
        result->next = merge(a, b->next); 
    } 
    return (result); 
}

//Finding the middle element of the list for splitting
Node* getMiddle(Node* root) {
    if(root == nullptr){
        return root;
    }
    Node* fast; 
    Node* slow; 
    slow = root; 
    fast = root->next; 
  
    /* Advance 'fast' two nodes, and advance 'slow' one node */
    while (fast != nullptr) { 
        fast = fast->next; 
        if (fast != nullptr) { 
            slow = slow->next; 
            fast = fast->next; 
        } 
    } 
    /* 'slow' is before the midpoint in the list, so split it in two  
    at that point. */
    return slow;
}

Node* sortList(Node* root) {
    Node* head = root; 
  
    /* Base case -- length 0 or 1 */
    if ((head == nullptr) || (head->next == nullptr)) { 
        return head;
    } 
    /*split list down the middle*/
    Node* middle = getMiddle(head); 
    Node* split = middle->next;
    middle->next = nullptr;
  
    /* answer = merge the two sorted lists together */
    return merge(sortList(root), sortList(split));
}
