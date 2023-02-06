








class Fifo_Glissant(list):  # Mise en place d'un objet First In First Out (FIFO) pour réaliser la médiane des n dernières valeurs

    def __init__(self, n=0):  # n est la taille maximale de l'objet FIFO, ici n doit être impair
        self.n = n

    def ajouter(self, x):  # Ajouter élément mais retirer le plus ancien s'il y en a déjà n
        if self.size() == self.n:
            self.retirer()
        self.insert(0, x)

    def retirer(self):  # Retirer le plus ancien élément s'il y en a un
        if self == []:
            pass
        else:
            self.pop()

    def size(self):  # Retourne la taille actuelle de l'objet FIFO
        return len(self)

    def mediane(self):  # Retourne l'élément médian des éléments de l'objet FIFO
        n = self.size()
        s = sorted(self)
        if self == []:
            return None
        else:
            return (s[n // 2])  # n étant impair, on retourne l'élément central de la liste triée

    def moyenne(self):  # Retourne la moyenne des éléments de l'objet FIFO
        return (sum(self) / self.size())


class MedianFilter():  # On définit un filtre médian pour supprimer les spikes

    def __init__(self, M=1):  # On utilise l'objet FIFO_Glissant pour stocker les 2*M+1 dernières valeurs
        self.M = M
        self.data = Fifo_Glissant(2 * M + 1)

    def ajouter(self, x):  # On ajoute une mesure
        self.data.ajouter(x)

    def median_radar(self, distance):  # On applique le filtre médian pour supprimer les spikes
        self.ajouter(distance)
        return self.data.mediane()

    def median_reset(self): #  Réinitialise le filtre
        for k in range(2 * self.M + 1):
            self.data.retirer()


class MovingAverageFilter():  # On définit un filtre à moyenne glissante pour supprimer le bruit

    def __init__(self, N=3):  # On utilise l'objet FIFO_Glissant pour stocker les N dernières valeurs
        self.N = N
        self.data = Fifo_Glissant(N)

    def ajouter(self, x):  # On ajoute une mesure
        self.data.ajouter(x)

    def fir_radar(self, distance):
        self.ajouter(distance)
        return self.data.moyenne()  # On retourne la moyenne des n derniers éléments

    def fir_reset(self): #  Réinitialise le filtre
        for k in range(self.N):
            self.data.retirer()