Se trouve dans ce dossier :

- un fichier contenant l’ensemble des classes et méthodes utilisées lors de ce test: synthesio_classes.py
- Un notebook jupyter décrivant l’ensemble du process qui a été établie durant ce test.
- Un fichier requirements.txt contenant les divers package utilisés lors du test.


Le code est organisé en classe. Les classes sont instanciées dans le notebook directement,
les méthodes sont aussi appelées dans le notebook.
Il y a trois classes :
- La classe Data qui gère le dataset, et toute les fonctions liées au dataset.
- La classe Preprocesser qui preprocess les différents données du test
- La classe Model qui gère les différents modèles que l’on utilise au cours du
Test.


Il y a deux façon d'exécuter le rapport jupyter.
1)
À la racine du fichier, exécuter les commandes suivantes :
$ virtualenv <env_name>
$ source <env_name>/bin/activate
(<env_name>)$ pip install -r requirements.txt
