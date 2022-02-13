#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

using namespace std;
vector<string> split(const string& dom){
  vector<string> res;
  string s;
  auto it = dom.end();
  for(it; ; --it){
    if(*it != '.' || it == dom.begin()){
      s = s+(*it);
    }
    else{
	 s = s+".";
        res.push_back(s);
    }
    if(it == dom.begin()){
       break;
    }
  }
  res.push_back(s);
  return res;
}

int main() {
  string domain1 = "mobile.sports.yahoo.com";
  string domain2 = "secure.mail.google.com";
  string domain3 = "karat.com";
    
  vector<string> sol = split(domain1);
  for(int i = 0; i < sol.size(); i++){
     if(i >= 1 && i < sol.size() -1){
	for(int j = sol[i].length() - 2; j >= 0; j--){
           cout << sol[i][j];
        }
      }
     else{
         for(int j = sol[i].length()-1; j >= 0; j--){
            cout << sol[i][j];
         }
     }
    cout << " ";
  }
  cout << sol.size() << endl;


  vector<string> sol3 = split(domain3);
  for(int i = 0; i < sol3.size(); i++){
     if(i >= 1 && i < sol3.size() -1){
        for(int j = sol3[i].length() -2; j >= 0; j--){
           cout << sol3[i][j];
        }
      }
     else{
         for(int j = sol3[i].length()-1; j >= 0; j--){
            cout << sol3[i][j];
         }
     }
    cout << " ";
  }
  cout << sol3.size() << endl;

  vector<string> sol2 = split(domain2);
  for(int i = 0; i < sol2.size(); i++){
     if(i >= 1 && i < sol2.size() -1){
        for(int j = sol2[i].length() -2; j >= 0; j--){
           cout << sol2[i][j];
        }
      }
     else{
         for(int j = sol2[i].length()-1; j >= 0; j--){
            cout << sol2[i][j];
         }
     }
    cout << " ";
  }
  cout << sol2.size() << endl;

  return 0;
}
