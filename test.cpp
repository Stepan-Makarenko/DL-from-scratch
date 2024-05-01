#include "layers.h"
// #include "catch_amalgamated.hpp"

// unsigned int Factorial( unsigned int number ) {
//     return number <= 1 ? number : Factorial(number-1)*number;
// }

// TEST_CASE( "Factorials are computed", "[factorial]" ) {
//     REQUIRE( Factorial(1) == 1 );
//     REQUIRE( Factorial(2) == 2 );
//     REQUIRE( Factorial(3) == 6 );
//     REQUIRE( Factorial(10) == 3628800 );
// }

// TEST_CASE( "Dot are correct", "[dot]" ) {
//     Matrix A({1, 2}, 1, 2);
//     Matrix B({5, -6}, 2, 1);
//     Matrix C({-7}, 1, 1);
//     REQUIRE(A.dot(B) == C);
// }

// TEST_CASE( "Dot are correct", "[dot]" ) {
//     SECTION("Init vectors") {
//         Matrix A({1, 2}, 1, 2);
//         Matrix B({5, -6}, 2, 1);
//         Matrix D = A.dot(B);
//         Matrix C({-7}, 1, 1);
//         REQUIRE(D == C);
//     }
// }

// TEST_CASE( "Dot are correct" ) {
//     // Matrix A({1, 2}, 1, 2);
//     // Matrix B({5, -6}, 2, 1);
//     // Matrix D = A.dot(B);
//     // Matrix C({-7}, 1, 1);
//     // REQUIRE(D == C);
//     REQUIRE(2 == 3);
// }

int main() {
    Matrix A({1, 2}, 1, 2);
    Matrix B({5, -6}, 2, 1);
    Matrix D = A.dot(B);
    Matrix C({-7}, 1, 1);
    std::cout << (D == C) << std::endl;
    D.printMatrix();

    Matrix A2({0.7880706737796804, -2.0061161704226222, 1.3250964007839183, -0.6085097453451549, -1.0469387661884957, 0.4323792789854483, -0.42331107574615234, 0.5224888641579111, 0.4315190218090298, -1.5400642989975295, 0.08594102398279234, -0.9781675562848229, 0.45755235965291624, 0.2100805769593201, -1.0664276074505197}, 3, 5);
    Matrix B2({0.5956233116919929, 0.15673231805410162, -0.759251268172996, 0.12602117684448344, 1.0587848853993982, -0.6159616981532127, 1.2678272839139064, -1.2999956674421425, 0.0590942498877253, -0.132819290554631, 0.8496655352045975, -0.9078862942123753, 0.9544661826675669, 1.249265662033314, 0.686691445853788, -1.2578791575126416, 0.13209131348490039, 0.9026740057039291, 0.9639916282636611, -0.3233141188603843}, 5, 4);
    Matrix D2 = A2.dot(B2);
    Matrix C2({-2.2954401277212546, -0.5220262370249665, -3.442962374227678, 2.6081408734035105, 0.048657094124945496, -0.5919794978800229, -1.6093177371340792, 0.08555677959852706, -0.8977928502512926, -0.14497832400354602, -1.800398627589243, 0.9475735810467485}, 3, 4);
    std::cout << (D2 == C2) << std::endl;
    D2.printMatrix();
    cout << "\n";
    D2.T().printMatrix();

    Matrix A3({1, 2}, 1, 2);
    Matrix B3({5, -6}, 1, 2);
    A3 += B3;
    Matrix C3({6, -4}, 1, 2);
    std::cout << (A3 == C3) << std::endl;


    cout << "start FF layer" << "\n";
    LinearLayer L1({1, 2, 3, 4, 5, 6}, {7, 8, 9}, 2, 3);
    Matrix A4({1, 2}, 1, 2);
    Matrix D4 = L1.forward(A4);

    Matrix C4({16, 20, 24}, 1, 3);

    std::cout << (D4 == C4) << std::endl;
    D4.printMatrix();

    Matrix A5({1, 2 ,3, 4, 5, 6}, 2, 3);
    Matrix B5 = A5.T();
    B5.printMatrix();
    Matrix D5 = A5.dot(A5.T());

    Matrix C5({14, 32, 32, 77}, 2, 2);
    std::cout << (D5 == C5) << std::endl;
    D5.printMatrix();


}