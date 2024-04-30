#include "operations.h"

int main() {
    srand(time(NULL));
    Matrix m(3, 2);
    // Matrix l(1, 2, 4);
    Matrix l({1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, 2, 4);
    std::cout << "Hello world!" << std::endl;
    std::cout << "Random float = " << (float)(rand()) << std::endl;
    std::cout << "Matrix m = " << std::endl;
    m.printMatrix();
    std::cout << "Matrix l = " << std::endl;
    l.printMatrix();
    Matrix d = m.dot(l);
    std::cout << "Matrix dot = " << std::endl;
    d.printMatrix();
    // Matrix({1, 2}, 1, 2).dot(Matrix({5, -6}, 2, 1)) == Matrix({-7}, 1, 1);
    Matrix A({1, 2}, 1, 2);
    Matrix B({5, -6}, 2, 1);
    Matrix C({-7}, 1, 1);
    cout << "A.dot(B) == C is " << (A.dot(B) == C) << "\n";
    return 0;
}