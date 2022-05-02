#include "ising.h"
/*
This function will alloacte the memory for a system of sping that are in a grid
  LxL, as well as for nbr array saves the neighbors of the i particle in nbr[i],
  note that n_br is the numbde  of neighborhoods of each particle.
*/
void allocate_system(int L, int n_br, int ** &nbr, int * &spins)
{
  // General counter
  int i;
  // Total number of particle; grid with same spaces in any direction
  int N = L * L;

  // Creating space for the neighbor of every particle; nbr[i] saves the
  //  neighbors of the i particle.
  nbr = new int*[N];
  for(i = 0; i < N; ++i)
    nbr[i] = new int [n_br];

  /* In the following the neighbors are supposed as the following picture show
       for L = 6 for example; but they are in an array.  
    0  ...  5
    6  ...  11
    .       .
    .       .
    24 ...  35 */
 
  // Space for the spins of each particle and init; last space is 0 so that in
  //   free fronteir this neigborhood is a spin of zero; that is not particle.
  spins = new int[N + 1];

}


/*
This routine will simply give initial conditions to the spins and define
  the correct neighbors for the perdiodic representation of the grid of 
  neighbors.
*/
void init_periodic_system(int L, int n_br, int ** &nbr, int * &spins){  
  // General counter
  int i;
  // Total number of particle; grid with same spaces in any direction
  int N = L * L;

  // Fill neighbors according to above convention in allocate_system and the
  //   spins.
  for(i = 0; i < N; i++)
  {
    // each spin defined
    spins[i] = -1 + 2 * (rand() % 2);
    // Right neighbor: L rows passed + (pos + 1toright) % L; the % is because
    //   of periodic condition.
    nbr[i][0] = (i / L) * L + (i + 1) % L;
    // Down neigbor: simply advance a complete row
    nbr[i][1] = (i + L) % N;
    // Left neighbor.
    nbr[i][2] = (i / L) * L + (i - 1 + L) % L;
    // Upp neighbor.
    nbr[i][3] = ((i - L) + N) % N;
    
    //  Verify correct initialization
    /*  
    cout << "particle" << i << "\n";
    cout << "neighbors " << nbr[i][0] << " " << nbr[i][1] << " " << nbr[i][2]
        << " " << nbr[i][3] << "\n";*/
     
  }
  // The gohst particle
  spins[N] = 0; 
}


/*
This routine will simply give initial consitions to the spins and define
  the correct neighbors for the free representation of the grid of neighbors.
*/
void init_free_system(int L, int n_br, int ** &nbr, int * &spins){  
  // General counter
  int i;

  // Fill neighbors according to above convention in allocate_system and the
  //   spins; first constructed
  init_periodic_system(L, n_br, nbr, spins);

  int right = 0, down = 1, left = 2, up = 3;

  /* Erase incorrect neighbors */
  // Number of particles; index of gohst particle. 
  int N = L * L;

  // Corners
  // left up corner
  nbr[0][left] = N; nbr[0][up] = N;  
  // right up corner
  nbr[L - 1][right] = N; nbr[L - 1][up] = N; 
  // left down corner
  nbr[L * L - L][left] = N; nbr[L * L - L][down] = N;
  // right down corner
  nbr[L * L - 1][right] = N; nbr[L * L - 1][down] = N; 

  // sides
  for(i = 1; i < (L - 1); i++)
  { 
    // left side
    nbr[i * L][left] = N;
    // right side
    nbr[i * L + L - 1][right] = N;
    // upp side
    nbr[i][up] = N;
    // down side
    nbr[L * L - L + i][down] = N;
  }

/*  // Verify neighbors
  for(i = 0; i < L * L; i++){
    cout << "particle" << i << "\n";
    cout << "neighbors " << nbr[i][0] << " " << nbr[i][1] << " " << nbr[i][2]
        << " " << nbr[i][3] << "\n";
  }  */
}


/*
This routine calculate ths first two energy moments for a given temperature 
  given an initial configurarion of spins and n_steps intervention in the 
  system.*/
void energy_moments(int L, int n_br, int n_steps, double T, int ** &nbr, 
                        int * &spins, double * &Em){
  // General variables for loops
  int i, j;
  // Number of particles
  int N = L * L;
  // Random particle chosen
  int k;
  // Energy in a step and spin of k particle
  double sk = 0, E_step = 0;
  // Sum of spins different to the actual in the loop
  int h = 0;
  // beta parameter
  double beta = 1. / T;

  double M = 0, M_step = 0;

  mt19937 gen(1); //Standard mersenne_twister_engine seeded with rd()
  uniform_real_distribution<> dis(0, 1);

  // Seed for k's
  srand(10); 

  // Calculate initial energy; note index are repeated
  for(i = 0; i < N; i++)
  {
    for(j = 0; j < n_br; j++)
          E_step -= spins[i] * spins[nbr[i][j]];
  }
  E_step *= 0.5;

  // Calculate convergent moments
  for(i = 0; i < n_steps; i++)
  {
    // random chosen particle and spin  
    k = rand() % N;
    sk = spins[k];

    // Evaluate probability of being up.
    h = 0;
    for(j = 0; j < n_br; j++) h += spins[nbr[k][j]];

    spins[k] = -1;
    if (dis(gen) < 1. / (1 + exp(- 2. * beta * h))){
      spins[k] = 1;
    }
    
    
    // Actualizing energy in case of change in the sping by the local 
    //  thermalization process
    if (spins[k] != sk){
      E_step -= 2 * h  * spins[k];
    }
    
    Em[0] += E_step;
    Em[1] += E_step * E_step;
  }

  // Unbised estimators.
  Em[0] = Em[0] / n_steps;
  Em[1] = Em[1] / n_steps;
}


/* This routine calculate the two first moments of the energy for a given
  temperature; the range is assumed; for generalization it would be 
  usefult to pass al in a class
*/
void energy_temperature(int L, int n_br, int n_steps, int ** &nbr,
    int * &spins, int option){

  // Values to map the main interval in detail
  double T_min = 0.8, T_max = 4.8;
  double n_points = 50;
  double T_step = (T_max - T_min) / n_points;
  double T_pos = T_min;

  // Values to map with more detail
  double T_min2 = option == 1 ? 2.1 : 1.6;
  double T_max2 = option == 1 ? 2.6 : 2.5;;
  double n_points2 = 50;
  double T_step2 = (T_max2 - T_min2) / n_points2;

  // Opening correct file name
  ofstream EmvsT;
  char name[200];
  char dummy[200];

  option == 1 ? sprintf(name, "Periodic_data/") : sprintf(name, "Free_data/");
  sprintf(dummy, "EmvsTn_steps%dL%dn_points%.0f.txt", n_steps, L, n_points);
  strcat(name, dummy);
  cout << name << endl;
  EmvsT.open(name);

  // Variable to save energy moments and calculate them
  double * Em = new double[2];

  Em[0] = 0; Em[1] = 0;
  // cout << T_pos << " " << Em[0] << " " << Em[1] << '\n'; 
  while(T_pos  <= T_max)
  {
    energy_moments(L, n_br, n_steps, T_pos, nbr, spins, Em);
    EmvsT << T_pos << " " << Em[0] << " " << Em[1] << '\n';
    T_pos += T_step;
    while((T_pos >= T_min2) && (T_pos <= T_max2))
    {
        energy_moments(L, n_br, n_steps, T_pos, nbr, spins, Em);
        EmvsT << T_pos << " " << Em[0] << " " << Em[1] << '\n';
        T_pos += T_step2;
        cout << T_pos << " " << Em[0] << " " << Em[1] << '\n';
    }
  }
 
  EmvsT.close();
}


/* This routine calculate the magnetiztion moment after n_steps.
*/
double magnetization_moments(int L, int n_br, int n_steps, double T,
                             int ** &nbr, int * &spins, int option){

  // General variables for loops
  int i, j;
  // Number of particles
  int N = L * L;
  // Random particle chosen
  int k;
  // Energy in a step and spin of k particle
  double sk = 0, M_step = 0;
  // Sum of spins different to the actual in the loop
  int h = 0;
  // beta parameter
  double beta = 1. / T;
  // Result for the magnetization
  double M = 0;


  mt19937 gen(1); //Standard mersenne_twister_engine seeded with rd()
  uniform_real_distribution<> dis(0, 1);

  // Seed for k's
  srand(1); 

  // Calculate initial magnetization.
  for(i = 0; i < N; i++)
    M_step += spins[i];

  // Calculate convergent moments
  for(i = 0; i < n_steps; i++)
  {
    // random chosen particle and spin  
    k = rand() % N;
    sk = spins[k];

    // Evaluate probability of being up.
    h = 0;
    for(j = 0; j < n_br; j++) h += spins[nbr[k][j]];

    spins[k] = -1;
    if (dis(gen) < 1. / (1 + exp(- 2. * beta * h))){
      spins[k] = 1;
    }
    
    // Actualizing emagnetization in case of change in the sping by the local 
    //  thermalization process
    if (spins[k] != sk)
      M_step += 2 * spins[k];
    
    M += abs(M_step);
  }

  // Unbised estimators.
  return  M / n_steps;
}


/* This routine calculate the magnetiztion moment after n_steps for each 
temperature.
*/
void magnetization_temperature(int L, int n_br, int n_steps, int ** &nbr,
    int * &spins, int option){

  // Values to map the main interval in detail
  double T_min = 0.1, T_max = 3.8;
  double n_points = 70;
  double T_step = (T_max - T_min) / n_points;
  double T_pos = T_min;

  // Values to map with more detail
  double T_min2 = option == 1 ? 1 : 1;
  double T_max2 = option == 1 ? 1 : 1;
  double n_points2 = 50;
  double T_step2 = (T_max2 - T_min2) / n_points2;

  // Opening correct file name
  ofstream MmvsT;
  char name[200];
  char dummy[200];

  option == 1 ? sprintf(name, "Periodic_data/") : sprintf(name, "Free_data/");
  sprintf(dummy, "MmvsTn_steps%dL%dn_points%.0f.txt", n_steps, L, n_points);
  strcat(name, dummy);
  cout << name << endl;
  MmvsT.open(name);

  // Variable to save the magnetization and calculation
  double M = 0;

  // cout << T_pos << " " << Em[0] << " " << Em[1] << '\n'; 
  while(T_pos  <= T_max)
  {
    M = magnetization_moments(L, n_br, n_steps, T_pos, nbr, spins, option);
    MmvsT << T_pos << " " << M << '\n';
    T_pos += T_step;
    while((T_pos >= T_min2) && (T_pos <= T_max2))
    {
        magnetization_moments(L, n_br, n_steps, T_pos, nbr, spins, option);
        MmvsT << T_pos << " " << M  << '\n';
        T_pos += T_step2;
    }
  }
 
  MmvsT.close();
}

/*
This routine will save all the calculated data in txt files.
*/
void moments(int L, int n_br, int n_steps, double beta, int ** &nbr, 
             int * &spins, string name, int seed){
  
  // General variables for loops
  int i, j;
  // Number of particles
  int N = L * L;
  // Random particle chosen
  int k = 0;
  // Energy in a step and spin of k particle
  int sk = 0;
  int E_step = 0;
  // Sum of spins different to the actual in the loop
  int h = 0;
  double M = 0, M_step = 0;

  ofstream Spins5;
  ofstream Energies5;
  ofstream Magnetizations5;
  
  Spins5.open(name.append("Spins.txt"));
  Energies5.open(name.append("Energies.txt"));
  Magnetizations5.open(name.append("Magnetizations.txt"));

/*  random_device rd;  //Will be used to obtain a seed for the random number engine*/
  mt19937 gen(seed); //Standard mersenne_twister_engine seeded with rd()
  uniform_real_distribution<> dis(0, 1);

  // Seed for k's
  srand(seed); 
/*  cout << "seed of " << seed << "\n";
*/
  // Calculate initial energy; note index are repeated 
  for(i = 0; i < N; i++)
  {
    for(j = 0; j < n_br; j++)
          E_step -= spins[i] * spins[nbr[i][j]];
  }
  E_step *= 0.5;
/*  cout << "init energy" << E_step << endl;
*/
  // Calculate initial magnetization.
  for(i = 0; i < N; i++)
    M_step += spins[i];

  // Calculate convergent moments
  for(i = 0; i < n_steps; i++)
  {
    // random chosen particle and spin  
    k = rand() % N;
    sk = spins[k];

    // Evaluate probability of being up.
    h = 0;
    for(j = 0; j < n_br; j++){
        h += spins[nbr[k][j]];
    }
    
    
    if (dis(gen) < (1. / (1 + exp(- 2. * beta * h)))){
      spins[k] = 1;
    }

    if(beta == 0){
        spins[k] = 1;
    }
  
    // Actualizing energy in case of change in the sping by the local 
    //  thermalization process
    if (spins[k] != sk){
      E_step -= 2 * h  * spins[k];     
        
/*        cout << endl;
        cout << h << " " << spins[k] << " " << E_step << endl;
        for(int j = 0; j < N; j++){
            cout << spins[j] << " ";
        }
        cout << "\n";

        cout << "particle " << k << endl; 
        for(int j = 0; j < n_br; j++){
            cout << spins[nbr[k][j]] << " ";
        }
        cout << "\n";*/


      M_step += 2 * spins[k];
    }

    // Saving variables that define the system
    for(int j = 0; j < N; j++){
        Spins5 << spins[j] << " ";
    }
    Spins5 << "\n"; 

/*    cout  << M_step << ", ";
*/    Energies5 << E_step / N << endl;
    Magnetizations5 << M_step / N << endl; 
  }

/*  if (E_step != -50)
*/  /*cout  << M_step << " " << E_step << endl;*/


  for(int j = 0; j < N; j++){
        cout << spins[j] << ", ";
    }
    cout << endl << endl;


  Spins5.close();
  Energies5.close();
  Magnetizations5.close();
}


