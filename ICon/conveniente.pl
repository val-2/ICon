
% Prende in input una lista di features e salva man mano il prezzo piu alto
macchina_piu_costosa([Features|AltreMacchine], Macchina, Prezzo) :-
    predirre_prezzo(Features, PrezzoCorrente),
    macchina_piu_costosa(AltreMacchine, MaxMacchina, MaxPrezzo),
    (PrezzoCorrente > MaxPrezzo ->
     Macchina = Features, Prezzo = PrezzoCorrente
     ;
     Macchina = MaxMacchina, Prezzo = MaxPrezzo).

macchina_piu_costosa([], [], 0).

prezzo_conveniente(Features, PrezzoProposto) :-
    predirre_prezzo(Features, PrezzoPredetto),
    PrezzoProposto < PrezzoPredetto.

predirre_prezzo(Features, Prezzo) :-
    reverse(Features, FeaturesInvertite),  % Si invertono le feature per poter fare in modo che quando la lista totale è lunga ad es. 6, la prima feature sia la 5 e l'ultima la 0
    calcolare_incremento(FeaturesInvertite, 0, Prezzo).


% Predicato per calcolare gli incrementi basati sulle caratteristiche
calcolare_incremento([], PrezzoParziale, PrezzoParziale).

calcolare_incremento([Feature|AltreFeature], PrezzoParziale, Prezzo) :-
    length(AltreFeature, Indice),
    incremento_feature(Indice, Feature, Incremento),
    NuovoPrezzoParziale is PrezzoParziale + Incremento,
    calcolare_incremento(AltreFeature, NuovoPrezzoParziale, Prezzo).


% vehicleType,yearOfRegistration,gearboxAutomatic,powerHP,kilometer,fuelTypeDiesel,brand,notRepairedDamage

incremento_feature(0, bus, 5000).
incremento_feature(0, cabrio, 4000).
incremento_feature(0, citycar, 3500).
incremento_feature(0, coupe, 4500).
incremento_feature(0, limousine, 7000).
incremento_feature(0, stationwagon, 6000).

% yearOfRegistration
incremento_feature(1, Anno, Incremento) :-
    Incremento is 3000 + (2019 - Anno) * -200.

% gearboxAutomatic
incremento_feature(2, false, 0).
incremento_feature(2, true, 2000).

% powerHP
incremento_feature(3, HP, Incremento) :-
    Incremento is HP * 20.

% kilometer
incremento_feature(4, KM, Incremento) :-
    Incremento is 4000 - (KM * 0.05).

% fuelTypeDiesel
incremento_feature(5, false, 0).
incremento_feature(5, true, 1500).

% brand
incremento_feature(6, audi, 10000).
incremento_feature(6, bmw, 9000).
incremento_feature(6, ford, 5000).
incremento_feature(6, mercedes_benz, 10000).
incremento_feature(6, fiat, 6000).
incremento_feature(6, opel, 5000).
incremento_feature(6, peugeot, 7000).
incremento_feature(6, renault, 7500).
incremento_feature(6, seat, 6000).
incremento_feature(6, skoda, 4000).
incremento_feature(6, volkswagen, 6000).
incremento_feature(6, mazda, 8000).

% notRepairedDamage
incremento_feature(7, false, 1000).
incremento_feature(7, true, -2500).

% Esempio di uso
% Caratteristiche: vehicleType = suv, yearOfRegistration = 2020, gearboxAutomatic = yes, powerHP = 150, kilometer = 30000, fuelTypeDiesel = no, brand = bmw, notRepairedDamage = no
?- predirre_prezzo([vehicleType-suv, yearOfRegistration-2020, gearboxAutomatic-yes, powerHP-150, kilometer-30000, fuelTypeDiesel-no, brand-bmw, notRepairedDamage-no], Prezzo).
