/*
    Autores: Nicolas Alejandro Avila Perez
             Andres Felipe Duque Bran
             Andres Felipe Villacob Hernandez

    Ultima Actualizacion: Agosto 2 de 2021.
*/

// Librerias Incluidas.
#define _USE_MATH_DEFINES
#include <omp.h>
#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>
#include <fstream>

// Constantes Teoricas y Experimentales empleadas
const double kB = 1.380649e-23;                 // nits: N*m/K
const double vs = 1;                            // units: pN*nm
const double ai = 10;                           // units: nm
const double x0 = 1.96;                         // units: nm
const double k = 0.72;                          // units: pN/nm
const double T = 293;                           // units: K
const double m = 110e3 * 1.66054e-27;           // units: Kg
const double gamma_number = 6 * M_PI * 0.001 * 2.5e-9; // units: N*s/m
const double km = 24633.0;                      // units: uM
const double k1 = 4.0;                          // units: 1/(uM*s)
const double k3 = 600.0;                        // units: 1/s
const double k4 = 140.0;                        // units: 1/s
const double k5 = 600.0;                        // units: 1/s
const double F0 = -6.0;                         // units: pN
const double c0 = 4.0;                          // units: pN
const double L = 8.0;                           // units: nm

// Parametro Aleatorio de la libreria Random
const int seed = 1;
std::mt19937 gen(seed);

// Parametros secundarios
const double Beta = 1 / (kB * T * 1e21);                       // units: (1/pN*nm)
const double dt = m * 1 / gamma_number;                               // units: s
const double sigma = std::sqrt(2 * kB * T * dt / gamma_number) * 1e9; // units: nm
// const double alpha = std::sqrt(6.0 * kB * T * dt / gamma) * 1e9; // units: nm

// Numero de Iteraciones
const int N = 1000;
const int NATP = 100000;
const double dATP = 10.0;

// Declaracion de Funciones
double Potential(double x, double F);
std::vector<double> Histogram(int Nbins,
                              std::vector<double> waux);
std::vector<double> FirstPassageTime(double F,
                                     int N);
double T_integrate(std::vector<double> h,
                   double ATP, double Fext);
void Data_Velocity(double Fext, int Nbins);

// Cabeza de la Kinesina
class head
{
private:
    double E, x;

public:
    void InitialStep(double F);
    void OneStep(double F);
    double GetE(void) { return E; };
    double Getx(void) { return x; };
};

// Posicion y Energia Iniciales
void head::InitialStep(double F)
{
    x = 0;
    E = Potential(x, F);
}

// Un paso de la Kinesina
void head::OneStep(double F)
{
    std::normal_distribution<double> displ(0, sigma);
    double dx = displ(gen); //* alpha;
    std::uniform_real_distribution<double> unif(0.0,
                                                1.0);
    double P = unif(gen);
    double nE = Potential(x + dx, F);
    double dE = nE - E;
    if (dE <= 0)
    {
        x += dx;
        E = nE;
    }
    else if (P < std::exp(-Beta * dE))
    {
        x += dx;
        E = nE;
    }
}

int main(void)
{
    // Declaracion de la Fuerza
    const double dFext = 1.5;
    const int Nbins = 30;
    double Fext = 0.0;

    // Inicio del programa
    for (int i = 0; i < 21; i++)
    {
        Fext =-15.0 + i*dFext;
        Data_Velocity(Fext, Nbins);
    }
    
    //std::cout << "Program finished" << '\n';
    return 0;
}

// Declaracion del Potencial empleado
double Potential(double x, double F)
{
    double V = 0;
    V = 0.5 * k * (x - 8 - x0) * (x - 8 - x0) - vs * (std::exp(-(x / ai) * (x / ai)) + std::exp(-((x - 16.0) / ai) * ((x - 16.0) / ai))) + F * x;
    return V;
}

// Tiempo de Difusion en todas las Iteraciones
std::vector<double> FirstPassageTime(double F,
                                     int N)
{
    /*std::cout << "Calculating first passage time"
              << " with an external force F="
              << F
              << " pN"
              << '\n';*/
    std::vector<double> w(N, 0);
    std::string name1 = "first_passage_time_Fext=" + std::to_string(F) + ".txt";
    std::ofstream mout(name1);
// Paralelizacion de un paso de la Kinesina
#pragma omp parallel for
    for (int j = 0; j < N; j++)
    {
        int t = 0;
        head kinesin;
        kinesin.InitialStep(F);
        while (kinesin.Getx() <= 2.0 * L)
        {
            kinesin.OneStep(F);
            t += 1;
        }
        w[j] = t * dt;
        /*std::cout << "Diffusion process "
                  << j << " executed" << '\n';*/
    }
    /*std::cout << "Diffusion simulation"
              << " finished for F="
              << F << " pN" << '\n';*/
    for (int n = 0; n < N; n++)
    {
        mout << w[n] << "\n";
    }
    return w;
}

// Creacion de datos para el Histograma
std::vector<double> Histogram(int Nbins,
                              std::vector<double> waux)
{
    /*std::cout << "Calculating distribution obtained"
              << '\n';*/
    int N = waux.size();
    std::vector<double> h(Nbins + 2, 0.0);
    std::sort(waux.begin(), waux.end());
    double wmax = waux.back();
    double wmin = waux[0];
    const double dw = (wmax - wmin) / Nbins;
    for (int n = 0; n < Nbins; n++)
    {
        int count = 0;
        for (int i = 0; i < N; i++)
        {
            if (
                (n + 1.0) * dw >= waux[i] && waux[i] > n * dw)
            {
                count += 1;
            }
        }
        h[n] = (double)count / N;
    }
    h[Nbins] = wmin + 0.5 * dw;
    h[Nbins + 1] = dw;
    return h;
}

// Intergacion del Tiempo de Paso Completo
double T_integrate(std::vector<double> w,
                   double ATP, double Fext)
{
    double t1 = 1 / (k1 * ATP);
    double sum;
    for (int i = 0; i < N; i++)
    {
        sum += w[i];
    }
    sum /= N;
    double t2 = ((km + ATP) / ATP) * sum;
    double t3 = (1 / k3) * std::exp(std::abs(Fext - F0) / c0);
    double t4 = 1 / k4;
    double t5 = 1 / k5;
    return t1 + t2 + t3 + t4 + t5;
}

// Variacion del [ATP]
void Data_Velocity(double Fext, int Nbins)
{
    std::string name = "datosVel_Fext=" + std::to_string(Fext) + ".txt";
    std::ofstream fout(name);
    std::vector<double> w(N, 0);
    std::vector<double> h(Nbins + 2, 0);
    w = FirstPassageTime(Fext, N);
    // h = Histogram(Nbins, w);
    double ATP = 0.0;
    double T = 0.0;
    /*std::cout << "Calculating ATP dependency with"
              << "an external force F="
              << Fext << " pN" << '\n';*/
    for (int i = 0; i < NATP; i++)
    {
        T = T_integrate(w, ATP, Fext);
        fout << ATP << '\t' << L / T << '\n';
        ATP += dATP;
    }
}
