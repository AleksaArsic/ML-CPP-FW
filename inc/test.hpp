#include <iostream>

class Test 
{
    public:
        Test() { this->count++; }

        friend std::ostream& operator<<(std::ostream& os, const Test& t)
        {
            os << t.count;
            return os;
        }


    private:
        inline static int count = 0;
};