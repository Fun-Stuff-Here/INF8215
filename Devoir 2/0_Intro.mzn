% Copyright (C) 2022, Elizabeth Michaud 2073093, Nicolas Dépelteau 2083544
% Polytechnique Montréal
%
% Problème 0 - introductory level (1 pt)

include "globals.mzn";

var int: nombre;
array[1..4] of var 1..9: chiffres; % Le chiffre 0 n’est pas présent dans le nombre
var int: base = 10;
nombre = chiffres[1]*base^0 + chiffres[2]*base^1 + chiffres[3]*base^2 + chiffres[4]*base^3;

predicate even(var int:x) =
          x mod 2 == 0;

%--------------------------------constraint------------------------------------------------
constraint even(nombre); %C’est un nombre pair
constraint all_different(chiffres); % Les 4 chiffres sont différents
constraint chiffres[4] > chiffres[3]; % Le chiffre à la position des milliers est supérieur à celui à la position des centaines
constraint chiffres[2] < chiffres[1]; % Le chiffre à la position des dizaines est inférieur à celui à la position des unités.
constraint chiffres[3] > chiffres[1]; % Le chiffre à la position des centaines est supérieur à celui à la position des unités.
constraint sum(chiffres) > 15; % La somme des 4 chiffres est supérieure à 15

%-------------------------------- end constraint--------------------------------------------

solve minimize (chiffres[3] * chiffres[2] * chiffres[1]); %Le produit des 3 derniers chiffres (chiffre à la position des centaines × chiffre à la position des dizaines × chiffre à la position des unités) doit être minimisé

output [
  show(nombre)
];
