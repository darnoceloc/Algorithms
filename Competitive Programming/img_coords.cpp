#include <iostream>
#include <vector>


std::vector<int> findRectangle(const std::vector<std::vector<int>>& img){
std::vector<int> coord (4);
  int x = 0;
  int y = 0;
  int ctr = 0;
  if(img.size() == 1 && img[0].size() == 1){
     std::cout << "nil" << std::endl;
     coord[0] = 0;
     coord[1] = 0;
     coord[2] = 1;
     coord[3] = 1;
     return coord;
  }

  if(img[0][0] == 0 && img[1][0] != 0 && img[0][1] != 0){
     std::cout << "nil2" << std::endl;
     coord[0] = 0;
     coord[1] = 0;
     coord[2] = 1;
     coord[3] = 1;
     return coord;
  }

  for(int i = 0; i < img.size(); i++){
    for(int j = 0; j < img[i].size(); j++){
      if(img[i][j] == 0 && img[i][j-1] != 0 && j == img[i].size()-1 && ctr == 0){
            std::cout << "testing" << std::endl;
            ctr++;
	    coord[0] = i;
            coord[1] = j;
            x++;
            y++;
            while(img[i][j] == 0 && img[i][j-1] != 0 && j == img[i].size()-1){
              if(i == img.size() - 1 && j == img[i].size() - 1){
                coord[2] = x;
                coord[3] = y;
                return coord;
              }
              if(img[i+1][j] == 0 && j == img[i].size() -1 && i < img.size() -1){
                std::cout << "test2" << std::endl;
                y++;
                i++;
             }
             else{
                j++;
             }
          }
      }
      if(img[i][j] == 0 && img[i-1][j] != 0 && i == img.size()-1 && ctr == 0){
             std::cout << "testhw" << std::endl;
             ctr++;
             coord[0] = i;
             coord[1] = j;
             x++;
             y++;
             while(img[i][j] == 0 && img[i-1][j] != 0 && i == img.size()-1){
               if(i == img.size() - 1 && j == img[i].size() - 1){
                  coord[2] = x;
                  coord[3] = y;
                  return coord;
                }
               if(img[i][j+1] == 0 && i == img.size() -1 && j < img[i].size() -1){
                  std::cout << "test2hw" << std::endl;
                  x++;
                  j++;
	       }
               else{
		  j++;
	       }
            }
      }
      if(img[i][j] == 0 && j < img[i].size() -1 && i < img.size() - 1 && ctr == 0){
        ctr++;
        std::cout << "test" << std::endl;
        coord[0] = i;
        coord[1] = j;
        x++;
        y++;
        while(img[i][j] == 0 && j < img[i].size() - 1 && i < img.size() -1){
                if(img[i+1][j+1] == 0 && j < img[i].size() -1 && i < img.size() - 1){
           		std::cout << "dub" << std::endl;
           		x++;
           		y++;
           		i++;
           		j++;
        	}
        	else if(img[i][j+1] == 0 && j < img[i].size() - 1){
           		std::cout << "horiz" << std::endl;
           		x++;
           		j++;
        	}
		else if(img[i+1][j] == 0 && i < img.size()-1){
           		std::cout << "vertyn" << std::endl;
           		y++;
                        i++;
        	}
		else if(img[i+1][j] == 0 && img[i][j+1] != 0 && i < img.size()-1){
           		std::cout << "vertyw" << std::endl;
           		y++;
                        i++;
        	}
        	else{
                        std::cout << "increm." << std::endl;
	  		j++;
		}
	   }
	}
     }
  }
  coord[2] = x;
  coord[3] = y;
  return coord;
}

int main() {
std::vector<std::vector<int>> image1 = {
    {1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 0, 0, 0, 1},
    {1, 1, 1, 0, 0, 0, 1},
    {1, 1, 1, 1, 1, 1, 1},
  };
std::vector<int> sol = findRectangle(image1);
std::cout << sol[0] << sol[1] << sol[2] << sol[3] << std::endl;

std::vector<std::vector<int>> image2 = {
    {1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 0},
  };
std::vector<int> sol2 = findRectangle(image2);
std::cout << sol2[0] << sol2[1] << sol2[2] << sol2[3] << std::endl;

std::vector<std::vector<int>> image3 = {
    {1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 0, 0},
    {1, 1, 1, 1, 1, 0, 0},
  };
std::vector<int> sol3 = findRectangle(image3);
std::cout << sol3[0] << sol3[1] << sol3[2] << sol3[3] << std::endl;


std::vector<std::vector<int>> image5 = {
    {0},
};
std::vector<int> sol5 = findRectangle(image5);
std::cout << sol5[0] << sol5[1] << sol5[2] << sol5[3] << std::endl;

std::vector<std::vector<int>> image6 = {
    {1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 0},
    {1, 1, 1, 1, 1, 1, 0},
  };
std::vector<int> sol6 = findRectangle(image6);
std::cout << sol6[0] << sol6[1] << sol6[2] << sol6[3] << std::endl;

std::vector<std::vector<int>> image7 = {
    {1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 0, 1},
    {1, 1, 1, 1, 1, 0, 1},
    {1, 1, 1, 1, 1, 0, 1},
  };
std::vector<int> sol7 = findRectangle(image7);
std::cout << sol7[0] << sol7[1] << sol7[2] << sol7[3] << std::endl;

std::vector<std::vector<int>> image8 = {
    {1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 0, 0, 0, 0},
    {1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1},
  };
std::vector<int> sol8 = findRectangle(image8);
std::cout << sol8[0] << sol8[1] << sol8[2] << sol8[3] << std::endl;

std::vector<std::vector<int>> image9 = {
    {1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1},
    {0, 0, 0, 0, 1, 1, 1},
  };
std::vector<int> sol9 = findRectangle(image9);
std::cout << sol9[0] << sol9[1] << sol9[2] << sol9[3] << std::endl;

std::vector<std::vector<int>> image4 = {
    {0, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1},
};
std::vector<int> sol4 = findRectangle(image4);
std::cout << sol4[0] << sol4[1] << sol4[2] << sol4[3] << std::endl;


return 0;

}
