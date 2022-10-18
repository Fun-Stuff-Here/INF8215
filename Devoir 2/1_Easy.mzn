% Copyright (C) 2022, Elizabeth Michaud 2073093, Nicolas Dépelteau 2083544
% Polytechnique Montréal
%
% Problème 1 - easy level (3 pts)

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
var int: total_cost = sum(i in x)(cost[i]);

constraint l <= card(x) /\ card(x) <= u; % Vous devez sélectionner entre l et u attractions parmis un ensemble de A attractions qui vous est donné

constraint m < sum(i in x, j in x) (pair_attraction[i, j]); % Vous devez sélectionner les attraits touristiques à construire afin d’avoir un attrait touristique total supérieur à m

solve minimize total_cost;

output [
  "Attractions : ", show(x), "\n",
  "Cost : ", show(total_cost)
];