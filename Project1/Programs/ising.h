#include <iostream>
#include <cstdlib>
#include <random>
#include <math.h>
#include <fstream>
#include <stdlib.h>
#include <chrono> 
#include <string.h>
using namespace std;
using namespace std::chrono; 

void allocate_system(int L, int n_br, int ** &nbr, int * &spins);
void init_periodic_system(int L, int n_br, int ** &nbr, int * &spins);
void init_free_system(int L, int n_br, int ** &nbr, int * &spins);
void energy_moments(int L, int n_br, int n_steps, double T, int ** &nbr,
                        int * &init_spins, double * &Em);
void energy_temperature(int L, int n_br, int n_steps, int ** &nbr,
                        int * &init_spins, int option);
double magnetization_moments(int L, int n_br, int n_steps, double T, 
                             int ** &nbr, int * &spins, int option); 
void magnetization_temperature(int L, int n_br, int n_steps, int ** &nbr,
    int * &spins, int option);

void moments(int L, int n_br, int n_steps, double beta, int ** &nbr,
             int * &init_spins, string name, int seed);

