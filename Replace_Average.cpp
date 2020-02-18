#include <iostream>
#include <vector>

class replace_avg{

    private:
        int n;
        int tmpIn;
        double f;
        double l;
        double av;
        std::vector<double> intputData;

    public:
    std::vector<double> getData()
    {
        std::cout << "Enter array length\n";
        std::cin >> n;
        std::cout << "Enter the numbers\n";
        for(int i = 0; i < n; i++){
            std::cin >> tmpIn;
            intputData.push_back(tmpIn);
        }
        return intputData;
    }


    void modifyData(std::vector<double> &inputData)
    {
        f = inputData[0];
        l = inputData[inputData.size() - 1];
        av = (f + l)/2;
        for (auto &j: inputData)
        {
            if(j < av){
                j = av;
            }
        }
    }

    void printData(const std::vector<double> &inputData)
    {
        for (auto &i: inputData)
        {
            std::cout << i << " ";
        }
        std::cout << '\n';
    }
};

int main(){

    replace_avg vect;
    std::vector<double> v1 = vect.getData();
    vect.modifyData(v1);
    vect.printData(v1);

    return 0;
}
