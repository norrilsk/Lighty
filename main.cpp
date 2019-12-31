#include <iostream>


#include "tests/base.hpp"
#include <chrono>
#include"linal/thensor.hpp"
#include"linal/thensor_data.hpp"
//#define _CRTDBG_MAP_ALLOC
//#include <stdlib.h>
//#include <crtdbg.h>
int main()
{
    //test::dense_layer(true);
	//_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_DEBUG);
	//_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	//test::matmul3x3(true);
    //test::sequention_net(true);
	linal::thensor<float,2> mat({ 3,3 });
	linal::thensor<float, 1> vec({ 3 });
	linal::thensor<float, 1> vec1({ 3 });
	mat[0][0] = 0;
	mat[0][1] = 1;
	mat[0][2] = 2;
	mat[1][0] = 3;
	mat[1][1] = 4;
	mat[1][2] = 5;
	mat[2][0] = 6;
	mat[2][1] = 7;
	mat[2][2] = 8;
	vec[0] = 11;
	vec[1] = 12;
	vec[2] = 17;
	std::cout << mat << " \n\n" << vec << '\n' <<std::endl;
	mat[1] = vec;
	vec1.copy(vec);
	std::cout << mat << " \n\n" << vec << std::endl;
	
	//_CrtDumpMemoryLeaks();
    return 0;
}