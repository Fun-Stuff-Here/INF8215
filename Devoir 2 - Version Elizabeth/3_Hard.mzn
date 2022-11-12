include "globals.mzn";

int: n = 5;

array[1..n] of int: t_b = [2,2,3,2,1]; % top to bottom
array[1..n] of int: b_t = [3,2,1,3,4]; % bottom to top
array[1..n] of int: l_r = [4,1,3,2,3]; % left to right
array[1..n] of int: r_l = [1,4,2,2,2]; % right to left

array[1..n, 1..n] of var 1..n: x;      % hauteur des tours à chaque position

function var int: number_of_visible_towers(array[1..n] of var 1..n: y) = 
  sum(i in 1..n)(if i == 1 then 1 
                 else if forall(j in 1..i-1)(y[j] < y[i])then 1 
                      else 0 
                      endif 
                 endif);

constraint forall(i in 1..n)(all_different(x[i, 1..n]));
constraint forall(i in 1..n)(all_different(x[1..n, i]));

constraint forall(i in 1..n)(t_b[i] == number_of_visible_towers([x[j, i] | j in 1..n]));
constraint forall(i in 1..n)(b_t[i] == number_of_visible_towers([x[n+1-j, i] | j in 1..n]));
constraint forall(i in 1..n)(l_r[i] == number_of_visible_towers([x[i,j] | j in 1..n]));
constraint forall(i in 1..n)(r_l[i] == number_of_visible_towers([x[i,n+1-j] | j in 1..n]));

solve satisfy;

output 

["Configuration : \n"]

++

[ show(x[i,j]) ++ if j = n then "\n" else "," endif | i in 1..n, j in 1..n]
