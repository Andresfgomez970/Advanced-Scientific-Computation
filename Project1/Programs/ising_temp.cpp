#include "ising.h"

// Start counting running time
auto start = high_resolution_clock::now(); 

int main(int argc, char *argv[])
{
  if(argc != 5)
  {
    cout << "please supply the arguments L , n_steps, system_type" << endl;
    exit(0);
  }

  /* Initiating parameters for type of system to be modelated*/

  // Number of particle aling a row
  int L = (int) strtol(argv[1], NULL, 10); 
  // number of thermalization to be done
  int n_steps = strtol(argv[2], NULL, 10);
  // bool of run option
  int system_type = strtol(argv[3], NULL, 10);
  // specific_heat, magnetization
  int variable_type = strtol(argv[4], NULL, 10);


  /* Initiating general variables to modelate each system*/

  // Number of neigbohrs for each particle
  int n_br = 4;
  // Neigbohrs of each particle
  int ** nbr;
  // Spins of each particle
  int * spins;
  // allocating memory
  allocate_system(L, n_br, nbr, spins);
  

  srand(3);  // choosing seed
  /* Init and running modelation*/
  system_type ? init_periodic_system(L, n_br, nbr, spins) : 
               init_free_system(L, n_br, nbr, spins);

  // Running type of system
  cout << variable_type << endl;
  variable_type ? 
        energy_temperature(L, n_br, n_steps, nbr, spins, system_type) :
        magnetization_temperature(L, n_br, n_steps, nbr, spins, system_type);

  /* Stop counting running time */
  auto stop = high_resolution_clock::now(); 
  auto duration = duration_cast<minutes>(stop - start); 
  cout << "It took "<< duration.count() << " minutes" << endl; 

  return 0;
}