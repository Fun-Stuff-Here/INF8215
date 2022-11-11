include "globals.mzn";

enum ATTRACTION;

% Il faut sélectionner entre l et u attractions
int: l;
int: u;

% Attrait touristique total minimum
int: m;

% Attrait touristique par paire d'attraction
array[ATTRACTION, ATTRACTION] of int: pair_attraction;

% Coût de construction par attraction
array[ATTRACTION] of int: cost;

% Ensemble d'attractions à construire
var set of ATTRACTION: x;

% Coût total
var int: tot_cost = sum(i in x)(cost[i]);

% Attrait de la ville 

constraint m < sum(i in x, j in x) (if i > j then pair_attraction[i, j] else 0 endif);
constraint card(x) <= u;
constraint card(x) >= l;

% Minimiser le coût
solve minimize tot_cost;

output [
  "Attractions : ", show(x), "\n",
  "Cost : ", show(tot_cost)
];