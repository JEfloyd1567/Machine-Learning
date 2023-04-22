import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

class RegresionLogistica:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
            if(self.verbose == True and i % 10000 == 0):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print(f'loss: {self.__loss(h, y)} \t')
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold

##############################################################################################
#como usarla
##############################################################################################

# Cargamos el dataset de iris
iris = datasets.load_iris()
X = iris.data[:, :2]
y = (iris.target != 0) * 1

# Separamos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creamos una instancia de Regresión Logística
reg_log = RegresionLogistica(lr=0.1, num_iter=300000)

# Entrenamos el modelo con los datos de entrenamiento
reg_log.fit(X_train, y_train)

# Realizamos una predicción con los datos de prueba
y_pred = reg_log.predict(X_test, 0.5)

# Calculamos la precisión del modelo
accuracy = np.sum(y_pred == y_test) / y_test.size
print(f'Precisión del modelo: {accuracy}')