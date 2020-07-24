void levelOrder(Node* root) {
  queue<Node*> queue;
  int levelNum = 1;
  if (root == nullptr) return;
  queue.push(root);
  while (!queue.empty()) {
    int size = queue.size();
    vector<int> level;
    for (int i = 0; i < size; i++) {
      Node* node = queue.front();
      level.push_back(node->val);
      queue.pop();
      for (int j = 0; j < (node->children).size(); j++) {
        if ((node->children)[j] != nullptr) {
          queue.push((node->children)[j]);
        }
      }
    } 
    cout << "Level: " << levelNum << endl;
    for (int item : level) {
      cout << item << " ";
    }
    cout << endl;
    levelNum++;
  }
}
