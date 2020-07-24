vector<int> levelOrder(TreeNode* root) {
    queue<TreeNode*> q;
    q.push(root);
    vector<int> res;
    while (!q.empty()) {
        int size = q.size();
        int sum = 0;
        for (int i = 0; i < size; i++) {
            TreeNode* node = q.front();
            q.pop();
            sum += node->val;
            if (node->left != nullptr)
                q.push(node->left);
            if (node->right != nullptr)
                q.push(node->right);
        }
        res.push_back(sum);
    }
    return res;
}
