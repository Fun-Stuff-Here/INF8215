% Copyright (C) 2022, Elizabeth Michaud 2073093, Nicolas Dépelteau 2083544
% Polytechnique Montréal
%
% Problème 4 - extreme level (4 pts)

include "globals.mzn";

enum PROJECT;
enum WORKER = { ARCHI, ING, ELEC };

int: P; % nombre de jours (et de projets à choisir)

array[PROJECT, PROJECT] of int: travel_costs;     % Couts de deplacements entre projets

set of PROJECT: project_with_precedences;         % Ensemble des projets ayant des contraintes de précédences
array[PROJECT, 1..3] of WORKER: precedences;      % Précédences (pour les projets n'ayant pas de contraintes de précédences, ignorez la rangée)

set of PROJECT: project_must_be_done;             % Ensemble des projets qui DOIVENT êtres faits

var set of PROJECT: selected_project;             % Ensemble des projets choisis

array[PROJECT] of int: revenu;                    % Revenu des projets

array[WORKER, 1..P] of var PROJECT: schedule;     % Pour chaque employé, pour chaque jour, on assigne un projet
array[WORKER, PROJECT] of var 0..P: day;          % Pour chaque employé, pour chaque projet, on détermine le jour 

predicate completed_project(PROJECT:p, array[WORKER, 1..P] of var PROJECT: schedule) =
  forall(worker in WORKER)(
    get_index_of_project(p, [schedule[worker, i]| i in 1..P]) != 0
  );

% Profit (gain - dépenses) a maximiser
var int: profit =
% revenue
sum(p in selected_project)(if completed_project(p, schedule) then revenu[p] else 0 endif)
-
%cost
sum(worker in WORKER, i in 1..P-1) (
    travel_costs[schedule[worker, i], schedule[worker, i+1]]
);

constraint card(selected_project) == P; % P projet choisis
constraint project_must_be_done subset selected_project; % s'assure que les projets obligatoires sont pr/sent

constraint forall(i in 1..P)(all_different([schedule[j,i] |j in WORKER])); % Il ne peut pas y avoir plus d’un employé assigné à un projet par jour

function var int: get_index_of_project(PROJECT:p, array[int] of var PROJECT: projects) =
  sum(  [ if p = projects[i] 
              then i
           else 0 endif  
          | i in index_set(projects) ]
);

constraint forall(p in project_with_precedences)(
  if p in selected_project then
      get_index_of_project(p, [schedule[precedences[p, 1], i]| i in 1..P]) != 0
      /\
      get_index_of_project(p, [schedule[precedences[p, 1], i]| i in 1..P]) <
      get_index_of_project(p, [schedule[precedences[p, 2], i]| i in 1..P])
      /\ 
      get_index_of_project(p, [schedule[precedences[p, 2], i]| i in 1..P]) <
      get_index_of_project(p, [schedule[precedences[p, 3], i]| i in 1..P])

  else true endif
); %certains projets ont des contraintes sur l’ordre de passage des employés

solve maximize profit;

output

["Horaire : \n"] 
++
[if p = 1 then show(w) ++ ":" else "" endif ++ show(schedule[w,p]) ++ if p = P then "\n" else "," endif | w in WORKER, p in 1..P]
++
["-----------\n"]
++
["Profit : " ++ show(profit)]
