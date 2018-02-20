# Moniteur d'apprentissage en ligne pour Pioneer

Dernière édition : 7 Septembre 2016

(NB : la racine du projet est notée "racine" ; elle correspond actuellement à ~/Documents/sources_python3-5)

## Installation :
Le moniteur d'apprentissage nécessite l'environnement suivant :
* Système d'exploitation : Ubuntu Wily (15.10) (nécessaire pour installer Ros) : http://releases.ubuntu.com/15.10/
* Python 3.5 : https://docs.python.org/3/using/unix.html
* Ros hydro ou kinetic : http://wiki.ros.org/hydro/Installation/Ubuntu ou http://wiki.ros.org/kinetic/Installation/Ubuntu
* V-rep : http://www.coppeliarobotics.com/downloads.html (si nécessaire, copier vrep.py, vrepConst.py et remoteApi.so depuis V-REP/programming/remoteApiBindings/python/python et V-REP/programming/remoteApiBindings/lib/XXBit vers la racine du projet)
* Rospy : http://wiki.ros.org/rospy
* Rospkg : http://wiki.ros.org/rospkg
* Matplotlib (facultatif, premet de générer les graphes) : $ sudo pip3.5 install matplotlib

## Utilisation pour une simulation :
* Lancer V-rep : dans le dossier V-rep (actuellement ~/Documents/V-REP) : $ sh vrep.sh
* Si nécessaire, éditer les fichiers config.cfg (expérience unique) ou script.py (série de simulation)
* Pour une seule simulation, exécuter run.py avec python 3.5 dans le dossier source du moniteur d'apprentissage : $ python run.py ( 11/16 : Attention : pas $ python3 run.py lancera la  version python 3.4)
* Pour exécuter une série de simulations prédéfinies, exécuter script.py à la place : $ python3 script.py
* Suivant les paramètres choisis, la simulation finira automatiquement, ou nécessitera d'appuyer sur la touche entrée

## Utilisation pour une expérience avec le robot réel :
* Allumer la borne wifi ASUS
* Brancher un cable USB-Série entre l'ordinateur et le robot
* Démarrer l'ordinateur et se connecter au wifi ASUS (mdp : depinfonancy)
* Démarrer le robot et attendre que les capteurs ultrasons se mettent à "claquer"
* Sur l'ordinateur, lancer le logiciel RosAria : $ rosrun rosaria RosAria (les capteurs vont arrêter de claquer puis recommencer)
* Exécuter run.py avec python 3.5 dans le dossier source du moniteur d'apprentissage : $ python3 run.py
* Suivant les paramètres choisis, l'expérience finira automatiquement, ou nécessitera d'appuyer sur la touche entrée

## Fichiers de configuration importants :
* racine/config.cfg : Le fichier de configuration des expériences/simulations.
* ~/.bashrc : Le fichier de configuration du shell. Il contient notemment l'adresse IP du robot (actuellement 192.168.2.183 ou 192.168.2.184 pour les 2 minoïdes) ainsi que le port utilisé (actuellement 11311).
* ~/catkin_ws/src/.commonBash.bash : Contient des alias d'utilisation de Ros

## Liste des scripts (situés dans la racine du projet) :
* run.py : Le script de lancement des expériences/simulation. Récupère les paramètres dans le fichier config.cfg.
* script.py : Le script de lancement de séries de simulations pré-établies. Les séries proposées sont codées dans ce même fichier (cf "Ajouter un script personnalisé).
* graph_displayer.py : Affiche les graphes des données d'un fichier de log (.json). Récupère les logs dans le dossier logs.
* display_network.py : Affiche les valeurs des coefficients du réseau de neurones (entrée et sortie). Récupère les logs de neurones dans le dossier networks.
* old_theta.py : Affiche les lignes de directions générées par l'ancien theta_strategique.
* new_theta.py : Affiche les lignes de directions générées par le nouveau theta_strategique.
* get_position.py : Affiche la position actuelle mesurée par le robot réel. Utile pour le débuggage.

## Liste des autres fichiers de la racine :
* simul.ttt : La scène V-REP utilisée en simulation.
* vrep_pioneer_simulation.py : Le modèle du Pioneer simulé (expose les méthodes get_position, set_motor_velocity et set_position2)
* rdn.py : Le modèle du Pioneer réel (expose les méthodes get_position et set_motor_velocity)
* BackProp_Python_v2.py : Le modèle du réseau de neurones
* online_trainer.py : L'entraineur en ligne utilisé par run.py ; il récupère les informations du robot fourni, calcule le gradient, actualise le réseau de neurones et actionne les moteurs du robot.
* ask.py : Utilitaire qui facilite l'interaction avec l'utilisateur

## Ajouter un script personnalisé :
* Ouvrir le fichier source/scripts.py
* Ajouter le nom du script dans la liste "choices" (ligne ~70)
* Ajouter un cas dans l'évaluation de la variable choice (elif choice=="nom_du_script": ...). Il est conseillé de s'inspiré des scripts existants.
* Pour modifier les paramètres : options['NomDuParametre'] = 'valeur'
* Pour lancer une simulation : simulation(options). Il est recommandé d'attendre une seconde (time.sleep(1)) entre chaque simulation pour éviter des conflits avec V-REP.

## Ajouter un paramètre d'expérience :
* Ajouter le paramètre et sa valeur dans le fichier conf.cfg : NomDuParametre = valeur
* Définir la valeur par défaut dans la fonction get_standard_options du fichier script.py : options['NomDuParametre'] = 'valeur'
* Extraire la valeur de l'option dans le fichier run.py (ligne ~66) : NomDuParametre = int(options["NomDuParametre"]). NB : options["NomDuParametre] renvoit toujours une string et nécessite d'être convertie en le type voulu (ici en int par exemple).
* Ajouter la valeur du paramètre ainsi extraite en argument de la fonction threading.Thread (ligne ~172 de run.py).
* Dans online_trainer.py, ajouter un arguement correspondant à la valeur du paramètre dans la définition de la méhode train (ligne ~100) avec éventuellement une valeur par défaut. NB : l'ordre des paramètres doit être le même que précédemment.
* Modifier le reste de la méthode train de online_trainer.py pour prendre en compte le nouveau paramètre

## Pour plus d'information : greinervictor@gmail.com

## Remarques (Projet Dep Info 2016) :
* 21/11/16 : Nombre de neurones dans la couche cachée : entre ni et no et jamais plus que 2*ni
Ici on a 3 neurones en entrée et 2 en sortie, il vaut donc mieux mettre le nombre de neurones cachés à 3.
(source : http://www.faqs.org/faqs/ai-faq/neural-nets/part3/section-10.html)

Vérifier continuité lorsque le robot est à x = 0 : selon si x<0 ou x > 0, le theta target à un sens opposé, peut être que le robot a un problème là dessus.
Mettre le robot de -3pi/2 à 3pi/2 pour lui donner une plus grande marge de manoeuvre et éviter un problème de discontinuité des theta autour de pi comme c'est le cas actuellement
Pour cela, soit on revient en arrière quand on atteint une borne, soit on arrête de regarder car il est peu probable que le robot fasse plus de 3/4 de tour.;

## Recommandations pour l'expérience :
* Le robot ne conserve correctement en mémoire sa position initiale que s'il n'est pas bougé à la main, autrement il modifie rapidement sa position initiale, et il faut donc soit prendre en compte ce changement dans l'expérience soit redémarrer le robot.
