% Copyright (C) 2022, Elizabeth Michaud 2073093, Nicolas Dépelteau 2083544
% Polytechnique Montréal
%
% Problème 2 - normal level (3 pts)

include "globals.mzn";

% Nombre de segments horizontaux et verticaux
int: h; % nb de rows
int: v; % nb de cols

% La demande par segment horizontal et vertical
array[1..h] of int: demand_per_h;
array[1..v] of int: demand_per_v;

% Le coût du permis et de construction de maison associé à chaque carré
array[1..h, 1..v] of int: cost_per_house;
array[1..h, 1..v] of int: cost_per_permit;

% Le nombre max de maison qui peut être construit par carré
int: m;

% Le nombre de maisons à construire par carré
array[1..h, 1..v] of var int: x;

% Le coût total
var int: total_cost = 
sum(i in 1..h, j in 1..v)(
if x[i, j] !=0 then
  % cout fixe
  cost_per_permit[i, j] +
  % cout variable
  x[i, j] * cost_per_house[i, j]
else 0 endif);

% contrainte de construction minimale
constraint forall(i in 1..h)( demand_per_h[i] <= sum(j in 1..v)(x[i, j]));
constraint forall(j in 1..v)( demand_per_v[j] <= sum(i in 1..h)(x[i, j]));

% contrainte du nombre maximal de maison par pater
constraint forall(i in 1..h, j in 1..v)(x[i, j] <= m);

solve minimize total_cost;

output
  
  ["Configuration : \n"]  
  
  ++
  
  [
    show(x[i,j]) ++ if j = v then "\n" else "," endif 
    | i in 1..h, j in 1..v
  ]
  
  ++
  
  ["----------------\n"]
  
  ++
  
  [
  "Cost : " ++ show(total_cost)
  ]
  