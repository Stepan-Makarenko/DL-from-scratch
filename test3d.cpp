#include "layers.h"
#include <gtest/gtest.h>


TEST(Matrix3dTest, VectorDot) {
    Matrix3d<2> A({1, 2}, {1, 2});
    Matrix3d<2> B({5, -6}, {2, 1});
    Matrix3d<2> D = A.dot(B);
    Matrix3d<2> C({-7}, {1, 1});
    ASSERT_TRUE(D == C);
}

TEST(Matrix3dTest, Dot2dx2d) {
    Matrix3d<2> A({0.7880706737796804, -2.0061161704226222, 1.3250964007839183, -0.6085097453451549, -1.0469387661884957, 0.4323792789854483, -0.42331107574615234, 0.5224888641579111, 0.4315190218090298, -1.5400642989975295, 0.08594102398279234, -0.9781675562848229, 0.45755235965291624, 0.2100805769593201, -1.0664276074505197}, {3, 5});
    Matrix3d<2> B({0.5956233116919929, 0.15673231805410162, -0.759251268172996, 0.12602117684448344, 1.0587848853993982, -0.6159616981532127, 1.2678272839139064, -1.2999956674421425, 0.0590942498877253, -0.132819290554631, 0.8496655352045975, -0.9078862942123753, 0.9544661826675669, 1.249265662033314, 0.686691445853788, -1.2578791575126416, 0.13209131348490039, 0.9026740057039291, 0.9639916282636611, -0.3233141188603843}, {5, 4});
    Matrix3d<2> D = A.dot(B);
    Matrix3d<2> C({-2.2954401277212546, -0.5220262370249665, -3.442962374227678, 2.6081408734035105, 0.048657094124945496, -0.5919794978800229, -1.6093177371340792, 0.08555677959852706, -0.8977928502512926, -0.14497832400354602, -1.800398627589243, 0.9475735810467485}, {3, 4});
    ASSERT_TRUE(D == C);
}

TEST(Matrix3dTest, Sum3dx1d) {
    Matrix3d<2> A({10, 20}, {1, 2});
    Matrix3d<2> B({7, -6}, {1, 2});
    A += B;
    Matrix3d<2> C({17, 14}, {1, 2});
    ASSERT_TRUE(A == C);
}

TEST(Matrix3dTest, TransposedDot2dx2d) {
    Matrix3d<2> A({1, 2 ,3, 4, 5, 6}, {2, 3});
    Matrix3d<2> B = A.T();
    Matrix3d<2> D = A.dot(A.T());
    Matrix3d<2> E = A.T().dot(A);
    Matrix3d<2> C({14, 32, 32, 77}, {2, 2});
    Matrix3d<2> F({17, 22, 27, 22, 29, 36, 27, 36, 45}, {3, 3});
    ASSERT_TRUE(D == C);
    ASSERT_TRUE(E == F);
}

TEST(Matrix3dTest, TransposedDot3dx2d) {
    Matrix3d<2> A({1, 2 ,3, 4, 5, 6}, {2, 3});
    Matrix3d<2> B = A.T();
    Matrix3d<2> D = A.dot(A.T());
    Matrix3d<2> E = A.T().dot(A);
    Matrix3d<2> C({14, 32, 32, 77}, {2, 2});
    Matrix3d<2> F({17, 22, 27, 22, 29, 36, 27, 36, 45}, {3, 3});
    ASSERT_TRUE(D == C);
    ASSERT_TRUE(E == F);
}

TEST(Matrix3dTest, Dot3dx3d) {
    Matrix3d<3> A({-0.43738618, -0.93387646, -0.31915838, -1.15497695,  0.03831201,
        0.43267736,  1.08120957, -0.53228033,  0.02487091,  0.14405412,
        0.84271182, -1.19927417, -0.90049443, -0.25542305, -0.39138483,
       -2.55341949, -1.50944834,  0.43441173, -0.09664254, -1.31001894,
       -1.55455388,  0.14019322,  2.08324254, -1.77474233}, {4, 2, 3});
    Matrix3d<3> B({1.52333746,  2.31745651, -0.23563892, -0.46266199, -0.9885804 ,
       -0.04544251, -1.27610713,  1.70941882, -0.17582235, -0.80099584,
        0.25824985,  0.68449115, -1.22603227,  0.5621324 ,  2.18923538,
       -0.27521001, -2.85668834, -0.38113164, -0.9489528 , -1.16788793,
       -0.33520121, -0.18242894,  0.4075656 ,  1.68551393}, {4, 3, 2});
    Matrix3d<3> C({-0.1307154 , -0.56705096, -2.19618382, -2.71399631, -1.27972956,
        2.29161824, -0.64170844, -1.24965237,  1.66291853, -0.28673297,
       -1.41494197, -1.18551258, -0.10275354, -2.26836921, -1.55466609,
       -3.53512663}, {4, 2, 2});
    Matrix3d<3> E({-1.69042102, -0.68398758,  0.57380016, -0.21250822, -1.54730232,
       -3.6284561 , -2.96121536,  0.44869342}, {4, 2, 1});


    Matrix3d<3> D = A.dot(B);
    Matrix3d<3> F = A.sum(2);
    ASSERT_TRUE(D == C);
    ASSERT_TRUE(E == F);
}

TEST(Matrix3dTest, Dot3dx2d) {
    Matrix3d<3> A({ 0.0703,  1.1969,  2.8960,  1.2038,  0.7007,  2.2674,  0.4320, -0.0696}, {2, 2, 2});
    Matrix3d<2> B({-2.3963,  0.5964, -0.7407, -0.5997,  0.2267, -0.7874}, {2, 3});
    Matrix3d<3> C({-0.8862,  0.3133, -0.9946, -7.6617,  2.0003, -3.0929, -3.0388,  0.9321,
        -2.3045, -0.9934,  0.2419, -0.2651}, {2, 2, 3});
    Matrix3d<3> D = A.dot(B);
    ASSERT_TRUE(D == C);
}


TEST(Matrix3dTest, Softmax) {
    Matrix3d<3> A({1.6766, -1.3411,  0.1589,  1.4561, -1.0347,  0.4669, -0.0720, -0.0955,
        -1.1835,  0.3533, -0.3107, -1.1106, -0.5482, -0.4215, -0.9140,  0.2324,
        -0.4815, -0.9962, -1.1422, -0.9634,  0.0801,  0.4324,  0.7568, -0.1168,
         1.1740, -1.8007,  0.2685,  2.3505,  1.6799, -0.3152,  0.1379, -0.2019,
        -0.3637, -0.1740,  1.2519,  2.2814, -1.6953,  0.3802,  0.7339,  0.0865,
        -0.5124, -1.4129, -0.9513, -1.4817,  2.6131,  0.5795, -0.0178, -1.3011}, {4, 3, 4});
    Matrix3d<3> C({0.4830, 0.0236, 0.1059, 0.3875, 0.0938, 0.4209, 0.2455, 0.2398, 0.1097,
        0.5099, 0.2625, 0.1180, 0.1995, 0.2265, 0.1384, 0.4356, 0.3661, 0.2188,
        0.1891, 0.2261, 0.1919, 0.2729, 0.3775, 0.1576, 0.2128, 0.0109, 0.0861,
        0.6902, 0.6657, 0.0905, 0.1424, 0.1014, 0.0469, 0.0567, 0.2359, 0.6605,
        0.0381, 0.3035, 0.4322, 0.2262, 0.4115, 0.1672, 0.2653, 0.1561, 0.8178,
        0.1070, 0.0589, 0.0163}, {4, 3, 4});
    Matrix3d<3> D = A.softmax(2);
    ASSERT_TRUE(D == C);
}

TEST(LinearLayerTest, Forward) {
    LinearLayer<2> L({1, 2, 3, 4, 5, 6}, {7, 8, 9}, 2, 3);
    Matrix3d<2> A({1, 2, 3, 4}, {2, 2});
    Matrix3d<2> D = L.forward(A);
    Matrix3d<2> C({16, 20, 24, 26, 34, 42}, {2, 3});
    ASSERT_TRUE(D == C);
}

TEST(AttentionLayer, Forward) {
    AttentionLayer<3> L({-2.3963,  0.5964, -0.7407, -0.5997,  0.2267, -0.7874},
                         {1.1988, -0.9085,  2.0541, -1.3181,  0.0613,  0.0090},
                         {1.2587, -0.4042,  0.4455, -0.8844,  0.3305,  1.1941}, 2, 3);
    Matrix3d<3> X({0.0703,  1.1969,  2.8960,  1.2038,  0.7007,  2.2674,  0.4320, -0.0696}, {2, 2, 2});
    Matrix3d<3> D = L.forward(X);
    Matrix3d<3> C({2.5684, -0.7688,  2.7233, -0.9700,  0.3672,  1.4606, -0.5188,  0.2341,
         2.0021,  0.3847, -0.1129,  0.4808}, {2, 2, 3});
    // D.printMatrix();
    ASSERT_TRUE(D == C);
};

TEST(AttentionLayer, Bacward) {
    AttentionLayer<3> L({-0.1409,  0.2812, -0.7163, -0.2501,  0.4649, -0.9400},
                         {-0.6800,  0.0778, -0.4292, -0.3748,  0.5013,  1.9680},
                         {-0.4150, -1.1287, -0.3930, -0.2455, -1.4234, -0.9383}, 2, 3);
    Matrix3d<3> X({0.4841,  0.2335,  1.6216,  0.8432, -0.8458,  1.4275,  0.9611, -0.4014,
         0.7831, -1.4496, -0.4583,  1.8772,  0.5043,  1.2233, -0.5215,  0.1485,
         0.3289,  0.8983,  0.7647, -0.0587,  2.7779, -1.5825, -1.3271, -0.9417}, {3, 4, 2});
    Matrix3d<3> D = L.forward(X);
    // D.printMatrix();
    L.backward(Matrix3d<3>(1, {3, 4, 3}));
    // dKW =  [[1.1755, -0.4526, -0.7880],
        // [ 1.6724, -1.1970, -3.7812]]
    ASSERT_TRUE(1 == 1);
}