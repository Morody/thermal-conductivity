// Finite differences.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//
#include <iostream>
#include <vector>
#include <numeric>
#include <iomanip>
#include <limits>
#include <numbers>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <fstream>
const int N = 30;
const int x_min = 0;
const int x_max = 1;
const int y_min = 0;
const int y_max = 0.5;
const float dx = 0.03; //(x_max - x_min) / N
const float dy = 0.016; //(y_max - y_min) / N
const float dt = 0.2;
const int Tx_start = 600; // border value temperature by axis X
const int Tx_end = 1200; // border value temperature by axis X
using namespace std;
float getLambda(double x, double y) {
    return 0.25 <= x && x <= 0.65 &&
        0.1 <= y && y <= 0.25 ? 0.01 : 0.0001;
}



void tridiagonal_matrix(double* temperature, double* F, double* lambda, double stepByAxis, double T_max, double T_min) {

    double A[N];
    double B[N];
    double C[N];
    double lambdaPlusHalf[N];
    double lambdaMinusHalf[N];
    for (int i = 0; i < N; i++)
    {
        lambdaPlusHalf[i] = (lambda[i + 1] + lambda[i]) / 2;
        lambdaMinusHalf[i] = (lambda[i] + lambda[i - 1]) / 2;
    }

    for (int i = 0; i < N; i++)
    {
        A[i] = -lambdaMinusHalf[i] / (2 * stepByAxis * stepByAxis);
        B[i] = -lambdaPlusHalf[i] / (2 * stepByAxis * stepByAxis);
        C[i] = 1 / dt - A[i] - B[i];
    }


    double alpha[N];
    double beta[N];

    alpha[0] = alpha[N - 1] = 0;
    beta[0] = T_min;
    beta[N - 1] = T_max;

    // fill F
    for (int i = 1; i < N - 1; i++)
    {
        F[i] = temperature[i] / dt + (lambdaPlusHalf[i] * (temperature[i + 1] - temperature[i]) - lambdaMinusHalf[i] * (temperature[i] - temperature[i - 1]))
            / (2 * stepByAxis * stepByAxis);
    }


    for (int i = 0; i < N - 2; i++) {
        alpha[i + 1] = - B[i + 1] / (C[i + 1] + A[i + 1] * alpha[i]);
        beta[i + 1] = (F[i + 1] - A[i + 1] * beta[i]) / (C[i + 1] + A[i + 1] * alpha[i]);
    }

    temperature[N - 1] = T_max;
    for (int i = N - 2; i >= 0; i--) {
        temperature[i] = alpha[i] * temperature[i + 1] + beta[i];
    }


}

int main(int argc, char** argv)
{
    // Initialization primary values of matrix By axis X
    double lambdaByX[N][N];
    double lambdaByY[N][N];
    //double F_x[N][N];
    //double tempByX[N][N];
    for (int i = 0; i < N; i++) {
        auto stepX = i * dx;
        for (int j = 0; j < N; j++)
        {
            auto stepY = j * dy;
            double lambda = getLambda(stepX, stepY);
            //tempByX[i][j] = 300;
            //F_x[i][j] = 0;
            lambdaByX[i][j] = lambda;
            lambdaByY[j][i] = lambda;
        }
    }


    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Datatype columnType, not_resized_columnType;

    MPI_Type_vector(size, 1, size, MPI_DOUBLE, &not_resized_columnType);
    MPI_Type_commit(&not_resized_columnType);

    MPI_Type_create_resized(not_resized_columnType, 0, sizeof(double), &columnType);
    MPI_Type_commit(&columnType);

    double* lambdaY = lambdaByY[rank];
    double* lambdaX = lambdaByX[rank];

    double temperatureReceive[N]; // N
    double fReceive[N]; // N

    if (rank == 0)
    {
        double temperature[N * N]{};
        double F[N * N]{};

        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                temperature[i * size + j] = 300;
                F[i * size + j] = 0;
            }
        }
        //fill F and temperature
        // 
        for (int i = 0; i < size; ++i) {
            tridiagonal_matrix((temperature + i * size), (F + i * size), lambdaByX[i], dx, Tx_end, Tx_start);
        }
        // ----------------------------- i = 0

        for (int i = 0; i < 3000; i++)
        {

            // n + 1 by Y
            MPI_Scatter(temperature, 1, columnType,
                temperatureReceive, size, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

            MPI_Scatter(F, 1, columnType,
                fReceive, size, MPI_DOUBLE,
                0, MPI_COMM_WORLD);


            double x = x_min + rank * dx;
            tridiagonal_matrix(temperatureReceive, fReceive, lambdaY, dy, 600 * (1 + x * x * x), 600 * (1 + x));

            MPI_Gather(temperatureReceive, size, MPI_DOUBLE,
                temperature, 1, columnType,
                0, MPI_COMM_WORLD);

            MPI_Gather(fReceive, size, MPI_DOUBLE,
                F, 1, columnType,
                0, MPI_COMM_WORLD);


            // n + 1/2 by X
            MPI_Scatter(F, size, MPI_DOUBLE,
                fReceive, size, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

            MPI_Scatter(temperature, size, MPI_DOUBLE,
                temperatureReceive, size, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

            tridiagonal_matrix(temperatureReceive, fReceive, lambdaX, dx, Tx_end, Tx_start);


            MPI_Gather(temperatureReceive, size, MPI_DOUBLE,
                temperature, size, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
            MPI_Gather(fReceive, size, MPI_DOUBLE,
                F, size, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

        }

        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                std::cout << std::setprecision(0) << std::fixed << temperature[i * size + j] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << std::endl;

        setlocale(LC_ALL, "rus");
        fstream fs;
        fs.open("file1.txt", fstream::app | fstream::in);

        for (size_t i = 0; i < size; i++)
        {
            for (size_t j = 0; j < size; j++)
            {
                fs << std::setprecision(0) << std::fixed << temperature[i * size + j] << " ";
            }
            fs << endl;
        }
        fs.close();
        // --------------------------- i = 1

       // // n + 1/2 by X
       //   MPI_Scatter(F, size, MPI_DOUBLE,
       //       fReceive, size, MPI_DOUBLE,
       //       0, MPI_COMM_WORLD);
       //
       //   MPI_Scatter(temperature, size, MPI_DOUBLE,
       //       temperatureReceive, size, MPI_DOUBLE,
       //       0, MPI_COMM_WORLD);
       //
       //   //for (int i = 0; i < size; i++)
       //   //{
       //   //    std::cout << temperatureReceive[i] << " ";
       //   //}
       //   //std::cout << std::endl;
       //
       //   tridiagonal_matrix(temperatureReceive, fReceive, lambdaX, dx, Tx_end, Tx_start);
       //
       //   MPI_Gather(temperatureReceive, size, MPI_DOUBLE,
       //       temperature, 1, columnType,
       //       0, MPI_COMM_WORLD);
       //   MPI_Gather(fReceive, size, MPI_DOUBLE,
       //       F, 1, columnType,
       //       0, MPI_COMM_WORLD);
       //
       //   // n + 1 by Y
       //   MPI_Scatter(temperature, 1, columnType,
       //       temperatureReceive, size, MPI_DOUBLE,
       //       0, MPI_COMM_WORLD);
       //
       //   MPI_Scatter(F, 1, columnType,
       //       fReceive, size, MPI_DOUBLE,
       //       0, MPI_COMM_WORLD);
       //
       //   x = x_min + 1 * dx;
       //   tridiagonal_matrix(temperatureReceive, fReceive, lambdaY, dy, 600 * (1 + x * x * x), 600 * (1 + x));
       //
       //   MPI_Gather(temperatureReceive, size, MPI_DOUBLE,
       //       temperature, 1, columnType,
       //       0, MPI_COMM_WORLD);
       //
       //   MPI_Gather(fReceive, size, MPI_DOUBLE,
       //       F, 1, columnType,
       //       0, MPI_COMM_WORLD);

           //for (int i = 0; i < size; ++i) {
           //    for (int j = 0; j < size; ++j) {
           //        std::cout << std::setprecision(2) << std::fixed << temperature[i * size + j] << "     ";
           //    }
           //    std::cout << std::endl;;
           //}
    }
    else {

        // ----------------------------- i = 0;
        for (int t = 0; t < 3000; t++)
        {
            // n + 1 by Y
            MPI_Scatter(nullptr, 0, MPI_DOUBLE,
                temperatureReceive, size, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

            MPI_Scatter(nullptr, 0, MPI_DOUBLE,
                fReceive, size, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

            double x = x_min + rank * dx;
            tridiagonal_matrix(temperatureReceive, fReceive, lambdaY, dy, 600 * (1 + x * x * x), 600 * (1 + x));

            MPI_Gather(temperatureReceive, size, MPI_DOUBLE,
                nullptr, 0, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

            MPI_Gather(fReceive, size, MPI_DOUBLE,
                nullptr, 0, MPI_DOUBLE,
                0, MPI_COMM_WORLD);


            // n + 1/2 by X

            MPI_Scatter(NULL, size, MPI_DOUBLE,
                fReceive, size, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

            MPI_Scatter(NULL, size, MPI_DOUBLE,
                temperatureReceive, size, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

            tridiagonal_matrix(temperatureReceive, fReceive, lambdaX, dx, Tx_end, Tx_start);

            MPI_Gather(temperatureReceive, size, MPI_DOUBLE,
                nullptr, 0, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

            MPI_Gather(fReceive, size, MPI_DOUBLE,
                nullptr, 0, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
        }
        // ------------------------------------------------- i = 1

        // n + 1/2 by X
   //  MPI_Scatter(NULL, size, MPI_DOUBLE,
   //      fReceive, size, MPI_DOUBLE,
   //      0, MPI_COMM_WORLD);
   //
   //  MPI_Scatter(NULL, size, MPI_DOUBLE,
   //      temperatureReceive, size, MPI_DOUBLE,
   //      0, MPI_COMM_WORLD);
   //
   //  tridiagonal_matrix(temperatureReceive, fReceive, lambdaX, dx, Tx_end, Tx_start);
   //
   //  MPI_Gather(temperatureReceive, size, MPI_DOUBLE,
   //      nullptr, 0, MPI_DOUBLE,
   //      0, MPI_COMM_WORLD);
   //
   //  MPI_Gather(fReceive, size, MPI_DOUBLE,
   //      nullptr, 0, MPI_DOUBLE,
   //      0, MPI_COMM_WORLD);
   //
   //   // n + 1 by Y
   //
   //   MPI_Scatter(nullptr, 0, MPI_DOUBLE,
   //               temperatureReceive, size, MPI_DOUBLE,
   //               0, MPI_COMM_WORLD);
   //
   //   MPI_Scatter(nullptr, 0, MPI_DOUBLE,
   //               fReceive, size, MPI_DOUBLE,
   //               0, MPI_COMM_WORLD);
   //
   //   x = x_min + 1 * dx;
   //   tridiagonal_matrix(temperatureReceive, fReceive, lambdaY, dy, 600 * (1 + x * x * x), 600 * (1 + x));
   //
   //   MPI_Gather(temperatureReceive, size, MPI_DOUBLE,
   //               nullptr, 0, MPI_DOUBLE,
   //               0, MPI_COMM_WORLD);
   //
   //   MPI_Gather(fReceive, size, MPI_DOUBLE,
   //               nullptr, 0, MPI_DOUBLE,
   //               0, MPI_COMM_WORLD);
    }

    MPI_Type_free(&columnType);
    MPI_Type_free(&not_resized_columnType);

    MPI_Finalize();

    /////--------------- Шаги по слоям n + 1 и n + 1/2
    //// Method ADI's and Algorithm Thomas
    //for (int i = 0; i < N; i++)
    //{
    //    tridiagonal_matrix(tempByX[i], F_x[i], lambdaByX[i], dx, Tx_end, Tx_start);
    //}


    // Initialization primary values of matrix By axis Y
    //double F_y[N][N];
    //double tempByY[N][N];

    //Swap columns and rows
    //for (int x = 0; x < N; x++) {
    //    for (int y = 0; y < N; y++)
    //    {
    //        F_y[x][y] = F_x[y][x];
    //        lambdaByY[x][y] = lambdaByX[y][x];
    //        tempByY[x][y] = tempByX[y][x];
    //    }
    //}
    //
    //for (int i = 0; i < N; i++)
    //{
    //    double x = x_min + i * dx;
    //    tridiagonal_matrix(tempByY[i], F_y[i], lambdaByY[i], dy, 600 * (1 + x * x * x), 600 * (1 + x));
    //}
    //
    //for (int x = 0; x < N; x++) {
    //    for (int y = 0; y < N; y++)
    //    {
    //        tempByX[x][y] = tempByY[y][x];
    //    }
    //}

    //for (int n = 0; n < 3000; n++)
    //{
    //    //n+1/2
    //    for (int i = 0; i < N; i++)
    //    {
    //        tridiagonal_matrix(tempByX[i], F_x[i], lambdaByX[i], dx, Tx_end, Tx_start);
    //    }
    //
    //    for (int x = 0; x < N; x++) {
    //        for (int y = 0; y < N; y++)
    //        {
    //            tempByY[x][y] = tempByX[y][x];
    //        }
    //    }
    //
    //
    //
    //    //n+1
    //    for (int i = 0; i < N; i++)
    //    {
    //        double x = i * dy;
    //        tridiagonal_matrix(tempByY[i], F_y[i], lambdaByY[i], dy, 600 * (1 + x * x * x), 600 * (1 + x));
    //    }
    //
    //    for (int x = 0; x < N; x++) {
    //        for (int y = 0; y < N; y++)
    //        {
    //            tempByX[x][y] = tempByY[y][x];
    //        }
    //    }
    //
    //}
    // конечная матрица на слое n + 1
    //for (int i = 0; i < N; i++)
    //{
    //    tridiagonal_matrix(tempByX[i], F_x[i], lambdaByX[i], dx, Tx_end, Tx_start);
    //}

    //for (int i = 0; i < N; i++)
    //{
    //    for (int j = 0; j < N; j++)
    //    {
    //        //std::cout << tempByX[i][j] << "   ";
    //        std::cout << std::setprecision(2) << std::fixed << tempByX[i][j] << "     ";
    //    }
    //    std::cout << std::endl;
    //}

}


