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
	test::linal_conv2d();
	test::linal_conv2d_2();
	test::linal_conv2d_3();
	test::linal_conv_unroll();
	test::linal_conv2d_experimental(true);
	//test::sequention_net(true);
	//test::dense_net(true);
	//test::dense_net_sin(true);
	int a;
	std::cin >> a;
	system("pause");
    return 0;
}