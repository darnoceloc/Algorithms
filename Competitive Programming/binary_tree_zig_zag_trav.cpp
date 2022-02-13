#define tn TreeNode
vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
vector<vector<int>> ans; if(!root) return ans;
stack<tn*> a, b;
stack<tn*> *stk1 = &a, *stk2 = &b; a.push(root);
while(!stk1->empty()){
    ans.push_back(vector<int> (0));
    while(!stk1->empty()){
        tn *cur = stk1->top();
        stk1->pop();
        ans.back().push_back(cur->val);
        if(stk1 != &b && cur->left) stk2->push(cur->left);
        if(cur->right)stk2->push(cur->right);
        if(stk1 == &b && cur->left) stk2->push(cur->left);
    }
    swap(stk1, stk2);
}
return ans;
	}

class Solution {
    vector<vector<int>> result;
    std::vector<int> res;
    int ctr = 1;
    queue<TreeNode*> temp;
    stack<TreeNode*> level;
public:
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        if(root == nullptr){
            return result;
        }
        temp.push(root);
        res.push_back(root->val);
        result.push_back(res);
        res.clear();
        while(!temp.empty()){
            while(fmod(ctr, 2) != 0){
                int x = temp.size();
                root = temp.front();
                int leaves = 0;
                while(x > 0){
                    temp.pop();
                    x--;
                    if(root->right){
                        res.push_back(root->right->val);
                        temp.push(root->right);
                        ++leaves;
                    }
                    if(root->left){
                        res.push_back(root->left->val);
                        temp.push(root->left);
                        ++leaves;
                    }
                    if(!root->left && !root->right && x >0){
                        root = temp.front();
                        continue;
                     }
                    if(res.size() >= 1 && x > 0){
                        root = temp.front();
                        continue;
                    }
                    if(res.size() < 1 && x <= 0){
                        root = temp.front();
                        break;
                    }
                    if(res.size() >= 1 && x <= 0){
                        root = temp.front();
                        result.push_back(res);
                        res.clear();
                        break;
                    }
                }
                ctr++;
                if(leaves == 0){
                    return result;
                }
                leaves = 0;
              }
            while(fmod(ctr, 2) == 0){
               int y = temp.size();
                root = temp.front();
                int leaves = 0;
                while(y > 0){
                    level.push(root);
                    temp.pop();
                    y--;
                    if(root->right){
                        temp.push(root->right);
                        ++leaves;
                    }
                    if(root->left){
                        temp.push(root->left);
                        ++leaves;
                    }
                    if(!root->left && !root->right && res.size() > 0 && y > 0){
                        root = temp.front();
                        result.push_back(res);
                        res.clear();
                        continue;
                    }
                    if(!root->left && !root->right && res.size() == 0 && y > 0){
                        root = temp.front();
                        continue;
                    }
                     if(res.size() > 0 && y <= 0){
                        result.push_back(res);
                        res.clear();
                        root = temp.front();
                        continue;
                    }
                    if(res.size() <= 0 && y <= 0){
                        root = temp.front();
                        continue;
                    }
                     if(!root->left && !root->right && res.size() > 0 && leaves == 0){
                         result.push_back(res);
                         res.clear();
                         return result;
                    }
                    root = temp.front();
                 }
                if(leaves == 0){
                    return result;
                }
                root = level.top();
                int lev = level.size();
                while(lev > 0){
                    level.pop();
                    --lev;
                    if(root->left){
                        res.push_back(root->left->val);
                    }
                    if(root->right){
                        res.push_back(root->right->val);
                    }
                    if(!root->left && !root->right && lev > 0){
                        root = level.top();
                        continue;
                    }
                    if(res.size() >= 1 && lev <= 0){
                        result.push_back(res);
                        res.clear();
                        break;
                    }
                    if(res.size() > 1 && lev > 0){
                        root = level.top();
                        continue;
                    }
                    if(res.size() <= 1 && lev > 0){
                        root = level.top();
                        continue;
                    }
                }
                ctr++;
                if(leaves == 0){
                    return result;
                }
                leaves = 0;
            }
        }
       return result;
    }
};
