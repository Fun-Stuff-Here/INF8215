include "globals.mzn";

var int: nombre;
array[1..4] of var 1..9: chiffres;


constraint chiffres[4] mod 2 == 0;
constraint all_different(chiffres);
constraint chiffres[1] > chiffres[2];
constraint chiffres[3] < chiffres[4];
constraint chiffres[2] > chiffres[4];
constraint (sum(i in 1..4)(chiffres[i]) > 15);

solve minimize chiffres[2] * chiffres[3] * chiffres[4];
nombre = chiffres[1] * 10^3 + chiffres[2] * 10^2 + chiffres[3] * 10 + chiffres[4];

output [
  show(nombre)
];


