#include <iostream>
#include <fstream>
#include <time.h>
#include <omp.h>
#include <complex>

using namespace std;

const int matrixSize[] = { 50, 125, 250 };

bool exists(const char* name) {
    ifstream f(name);
    return f.is_open();
}

void GenerateMatrix() {
    srand(time(0));
    char filenameA[20], filenameB[20];

    for (int N : matrixSize) {

        sprintf_s(filenameA, "matrixA%d.txt", N);
        sprintf_s(filenameB, "matrixB%d.txt", N);

        ofstream matrixA(filenameA);
        ofstream matrixB(filenameB);

        if (!matrixA.is_open() || !matrixB.is_open()) {
            cout << "Ошибка чтения файла" << endl;
            exit(1);
        }

        complex<double>* matrA = new complex<double>[N];
        complex<double>* matrB = new complex<double>[N];

        for (int i = 0; i < N; i++) {
            complex<double> sum = 0.0;
            for (int j = 0; j < N; j++) {
                matrA[j] = complex<double>(rand() % 100, rand() % 100);
                sum += matrA[j];
            }

            matrA[i] = sum + complex<double>(1.0, 1.0);
            for (int j = 0; j < N; j++)
                matrixA << matrA[j] << "\t";
            matrixA << endl;
            matrB[i] = complex<double>(rand() % 100, rand() % 100);
            matrixB << matrB[i] << "\t";
        }

        matrixA.close();
        matrixB.close();

        delete[] matrA;
        delete[] matrB;
    }
}

void JacobiMethod() {

    for (int N : matrixSize) {

        complex<double>** matrixA = new complex<double>* [N];
        complex<double>* matrixB = new complex<double>[N];
        complex<double>* matrixX = new complex<double>[N];
        complex<double>* matrixXX = new complex<double>[N];

        for (int i = 0; i < N; i++)
            matrixA[i] = new complex<double>[N];

        char filenameA[50], filenameB[50], filenameX[50];

        sprintf_s(filenameA, "matrixA%d.txt", N);
        sprintf_s(filenameB, "matrixB%d.txt", N);
        sprintf_s(filenameX, "result_Jacobi%d.txt", N);

        ifstream matrA(filenameA);
        ifstream matrB(filenameB);
        ofstream matrX(filenameX);

        if (!matrA.is_open() || !matrB.is_open() || !matrX.is_open()) {
            cout << "Ошибка работы с файлом!" << endl;
            exit(1);
        }

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++)
                matrA >> matrixA[i][j];
            matrB >> matrixB[i];
            matrixX[i] = matrixXX[i] = 1.0;
        }

        double xmax = -1.0;
        double eps = 0.0001;
        double tn = omp_get_wtime();

        do {
            xmax = -1.0;
            for (int i = 0; i < N; i++) {

                complex<double> sum = 0.0;

                for (int j = 0; j < N; j++) {
                    if (i != j)
                        sum += matrixA[i][j] * matrixX[j];
                }

                matrixXX[i] = (1.0 / matrixA[i][i]) * (matrixB[i] - sum);

                complex<double> diff = matrixX[i] - matrixXX[i];

                if (xmax < abs(diff))
                    xmax = abs(diff);
            }

            for (int i = 0; i < N; i++)
                matrixX[i] = matrixXX[i];

        } while (xmax > eps);

        double tk = omp_get_wtime();

        for (int i = 0; i < N; i++)
            matrX << "X[" << i + 1 << "] = " << matrixX[i] << endl;

        matrA.close();
        matrB.close();
        matrX.close();

        for (int i = 0; i < N; i++)
            delete[] matrixA[i];

        delete[] matrixA;
        delete[] matrixB;
        delete[] matrixX;
        delete[] matrixXX;

        printf("Время решения СЛАУ методом простой итерации (методом Якоби) при n = %d %f\n", N, tk - tn);
    }
    cout << endl;
}

void GaussSeidelMethod() {

    for (int N : matrixSize) {

        complex<double>** matrixA = new complex<double>* [N];
        complex<double>* matrixB = new complex<double>[N];
        complex<double>* matrixX = new complex<double>[N];

        for (int i = 0; i < N; i++)
            matrixA[i] = new complex<double>[N];

        char filenameA[50], filenameB[50], filenameX[50];

        sprintf_s(filenameA, "matrixA%d.txt", N);
        sprintf_s(filenameB, "matrixB%d.txt", N);
        sprintf_s(filenameX, "result_GaussSeidel%d.txt", N);

        ifstream matrA(filenameA);
        ifstream matrB(filenameB);
        ofstream matrX(filenameX);

        if (!matrA.is_open() || !matrB.is_open() || !matrX.is_open()) {
            cout << "Ошибка работы с файлом!" << endl;
            exit(1);
        }

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++)
                matrA >> matrixA[i][j];
            matrB >> matrixB[i];
            matrixX[i] = 1.0;
        }

        double xmax = -1.0;
        double eps = 0.0001;
        double tn = omp_get_wtime();

        do {
            xmax = -1.0;
            for (int i = 0; i < N; i++) {

                complex<double> tmp = matrixX[i];
                complex<double> sum(0.0, 0.0);

                for (int j = 0; j < N; j++) {
                    if (i != j)
                        sum += matrixA[i][j] * matrixX[j];
                }

                matrixX[i] = (1.0 / matrixA[i][i]) * (matrixB[i] - sum);

                complex<double> diff = matrixX[i] - tmp;

                if (xmax < abs(diff))
                    xmax = abs(diff);
            }

        } while (xmax > eps);

        double tk = omp_get_wtime();

        for (int i = 0; i < N; i++)
            matrX << "X[" << i + 1 << "] = " << matrixX[i] << endl;

        matrA.close();
        matrB.close();
        matrX.close();

        for (int i = 0; i < N; i++)
            delete[] matrixA[i];

        delete[] matrixA;
        delete[] matrixB;
        delete[] matrixX;

        printf("Время решения СЛАУ методом Гаусса-Зейделя при n = %d %f\n", N, tk - tn);
    }
    cout << endl;
}

void GaussMethod() {

    for (int N : matrixSize) {

        complex<double>** matrixA = new complex<double>* [N];
        complex<double>* matrixB = new complex<double>[N];
        complex<double>* matrixX = new complex<double>[N];

        for (int i = 0; i < N; i++)
            matrixA[i] = new complex<double>[N];

        char filenameA[50], filenameB[50], filenameX[50];

        sprintf_s(filenameA, "matrixA%d.txt", N);
        sprintf_s(filenameB, "matrixB%d.txt", N);
        sprintf_s(filenameX, "result_Gauss%d.txt", N);

        ifstream matrA(filenameA);
        ifstream matrB(filenameB);
        ofstream matrX(filenameX);

        if (!matrA.is_open() || !matrB.is_open() || !matrX.is_open()) {
            cout << "Ошибка работы с файлом!" << endl;
            exit(1);
        }

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++)
                matrA >> matrixA[i][j];
            matrB >> matrixB[i];
            matrixX[i] = 1.0;
        }

        double tn = omp_get_wtime();

        // Прямой ход
        for (int i = 0; i < N; i++) {
            for (int j = i + 1; j < N; j++) {
                complex<double> w = -matrixA[j][i] / matrixA[i][i];
                for (int k = i; k < N; k++)
                    matrixA[j][k] += matrixA[i][k] * w;
                matrixB[j] += matrixB[i] * w;
            }
        }

        // Обратный ход
        for (int i = N - 1; i >= 0; i--) {
            complex<double> sum = 0.0;
            for (int j = i + 1; j < N; j++)
                sum += matrixA[i][j] * matrixX[j];
            matrixX[i] = (matrixB[i] - sum) / matrixA[i][i];
        }

        double tk = omp_get_wtime();

        for (int i = 0; i < N; i++)
            matrX << "X[" << i + 1 << "] = " << matrixX[i] << endl;

        matrA.close();
        matrB.close();
        matrX.close();

        for (int i = 0; i < N; i++)
            delete[] matrixA[i];

        delete[] matrixA;
        delete[] matrixB;
        delete[] matrixX;

        printf("Время решения СЛАУ методом Гаусса при n = %d %f\n", N, tk - tn);
    }
    cout << endl;
}

int main() {
    setlocale(LC_ALL, "ru");

    if (!exists("matrixA50.txt")) {
        cout << "Файлы не найдены. Происходит генерация" << endl;
        GenerateMatrix();
    }

    JacobiMethod();
    GaussSeidelMethod();
    GaussMethod();

    return 0;
}