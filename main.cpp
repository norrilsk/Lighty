#include <iostream>


#include "tests/base.hpp"
#include <chrono>
#include"linal/thensor.hpp"
#include"linal/thensor_data.hpp"

int main()
{
	test::thensor_test1();
	test::thensor_test2();
	test::thensor_test3();
	test::thensor_test4();
	//test::sequention_net(true);
	//test::dense_net(true);
	test::dense_net_sin(true);
    return 0;
}