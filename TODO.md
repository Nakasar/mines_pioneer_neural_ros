25/04 : tests concluants pour la position du robot en apprentissage et hors apprentissage, (après modification du return de theta_s dans online_trainer) cependant le robot n'arrive pas à avoir le bon theta (hors apprentissage).
Pistes : modifications du alpha[2] dans online trainer (hors apprentissage)

Le robot modifie sa position initiale au fur et à mesure des expériences, il faudrait donc redémarrer le robot à chaque démo si on veut l'avoir à (0, 0, 0).

02/05 : remarque : le robot semble garder sa position initiale après 4 tests, à vérifier sur plus de tests
En fait le robot conserve sa position initiale si on ne le bouge pas à la main, sinon il y a des erreurs.

12/12 :
Revoir le calcul de Theta_S (cf 25/04) (contradiction entre la méthode de calcul du robot et la méthode de calcul de l'algorithme, possible décalage dans le modulo)
