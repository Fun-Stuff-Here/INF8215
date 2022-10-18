include "globals.mzn";

int: n = 5;

array[1..n] of int: t_b = [2,2,3,2,1]; % top to bottom
array[1..n] of int: b_t = [3,2,1,3,4]; % bottom to top
array[1..n] of int: l_r = [4,1,3,2,3]; % left to right
array[1..n] of int: r_l = [1,4,2,2,2]; % right to left

array[1..n, 1..n] of var 1..n: x;      % hauteur des tours à chaque position

% Pour chaque segment horizontal et vertical du quartier, on ne peut pas avoir deux tours de la même taille
constraint forall(i in 1..n)(all_different([x[i, j] | j in 1..n]));
constraint forall(i in 1..n)(all_different([x[j, i] | j in 1..n]));

% fonction qui calcul le nombre de tour visible
function var int: number_of_tower_in_vision(array[1..n] of var 1..n: vision) = 
if n > 1 then
  1 + sum(i in 2..n)(
    if max([vision[j]| j in 1..i]) > max([vision[j]| j in 1..i-1]) then
    1
    else 0 endif)
else n endif;

% constraint des vues
constraint forall(i in 1..n)( l_r[i] == number_of_tower_in_vision([x[i, j] | j in 1..n])); %left to right
constraint forall(i in 1..n)( r_l[i] == number_of_tower_in_vision([x[i, n+1-j] | j in 1..n])); %right to left
constraint forall(i in 1..n)( t_b[i] == number_of_tower_in_vision([x[j, i] | j in 1..n])); %top to bottom
constraint forall(i in 1..n)( b_t[i] == number_of_tower_in_vision([x[n+1-j, i] | j in 1..n])); %bottom to top

solve satisfy;

output 

["Configuration : \n"]

++

[ show(x[i,j]) ++ if j = n then "\n" else "," endif | i in 1..n, j in 1..n]
