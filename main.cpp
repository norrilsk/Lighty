#include <iostream>


#include "tests/base.hpp"
#include <chrono>
#include"linal/thensor.hpp"
#include"linal/thensor_data.hpp"
#include "Utils/BMP.hpp"

int32_t main()
{
	test::thensor_test1();
	test::thensor_test2();
	test::thensor_test3();
	test::thensor_test4();
	test::linal_conv2d();
	test::linal_conv2d_2();
	test::linal_conv2d_3();
	test::linal_conv_unroll();
	//test::linal_unroll_and_back(true);
	//test::linal_unroll_test1();
	//test::linal_conv2d_experimental(true);
    //test::linal_conv2d_2_experimental(true);
	//test::sequention_net(true);
	//test::dense_net(true);
	//test::dense_net_sin(true);
	//test::linal_conv2d_net(true);
	test::dump_up();
	//Utils::BMP pic("D:\\Lighty_data\\data\\3.bmp");
	//pic.write("D:\\Lighty_data\\data\\out.bmp");
	system("pause");
    return 0;
}