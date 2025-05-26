import re
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

class TF_IDF:
    def __init__(self, documents):
        self.documents = documents
        self.raw_tokens = [re.findall("[a-zA-Z0-9\-]+", document, re.IGNORECASE) for document in documents]
        self.tokens = [[raw_token.lower() for raw_token in raw_tokens] for raw_tokens in self.raw_tokens]
        # self.vocab = list(set([token for tokens in self.tokens for token in tokens]))
        # element-wise multiplication
        TFs = self.TF()
        IDF = self.IDF()
        TF_ID = {}
        for TF in TFs:
            for k, v in zip(TF.keys(), TF.values()):
                if k in IDF:
                    TF_ID[k] = IDF[k]*v
        self.vocab = TF_ID


    def TF(self):
        total_counts  = []
        counts = {}
        for tokens in self.tokens:
            for token in tokens:
                if token in counts:
                    counts[token] += 1
                else:
                    counts[token] = 1
            total_counts.append(counts)
            counts = {}

        Total_TFs = []
        TFs = {}
        for counts in total_counts:
            TFs = {k: v / sum(counts.values()) for k, v in counts.items()}
            Total_TFs.append(TFs)
            TFs = {}
        return Total_TFs


    def IDF(self):
        counts = {}
        unique_tokens = []
        for token in self.tokens:
            unique_tokens.append(list(set(token))) # unique tokens across all documents
        for i in range(len(unique_tokens)):
            for token in unique_tokens[i]:
                if token in counts:
                    counts[token] += 1
                else:
                    counts[token] = 1
        IDFs = {k: np.log(len(self.documents) / v) for k, v in counts.items()}
        return IDFs

    def TFIDF(self):
        TF_IDFs = []
        # element-wise multiplication
        TFs = self.TF()
        IDF = self.IDF()
        for TF in TFs:
            TF_ID = {}
            for k, v in zip(TF.keys(), TF.values()):
                if k in IDF:
                    TF_ID[k] = IDF[k]*v
            TF_IDFs.append(TF_ID)

        # n documents, m unique words -> n * m TF-IDF Matrix
        # here it is 3 * 28
        TF_IDF_df = pd.DataFrame(TF_IDFs).fillna(0)
        TF_IDF_matrix = TF_IDF_df.to_numpy()
        X = TF_IDF_matrix
        return X
        # return TF_IDFs

    def transform(self, document): # to transform single document into fixed-length vector for matmul
        raw_tokens = re.findall("[a-zA-Z0-9\-]+", document, re.IGNORECASE)
        tokens = [raw_token.lower() for raw_token in raw_tokens]

        token_vector = [self.vocab[token] if token in self.vocab else 0 for token in tokens]
        token_vec = [self.vocab.get(key) if value in token_vector else 0 for key, value in zip(self.vocab.keys(), self.vocab.values())]
        token_vec = np.array(token_vec)
        return token_vec


documents = pd.read_csv("training_data.csv")
size = len(documents["message"])
label = documents["label"][:int(size*0.8)]
# train set
train = documents["message"][:int(size*0.8)]
vectorizer = TF_IDF(train)
X  = vectorizer.TFIDF()
# test set
test = documents["message"][int(size*0.2):]
vectorizer_test = TF_IDF(test)
X_test = vectorizer_test.TFIDF()

f = lambda x: 1 if x == "spam" else 0
y = list(map(f, label))
y = np.array(y)
y_test = np.array(list(map(f, documents["label"][int(size*0.2):])))

class LogisticRegression:
    def __init__(self, X, W, y):
        self.y = y
        self.W = W
        self.X = X
        self.loss_history = []

    def sigmoid(self, z):
        return 1 / (1+np.exp(-z))
        
    def nll(self):
        self.y_pred = self.sigmoid(self.X @ self.W)
        return -np.sum(self.y*np.log(self.y_pred) + (1-self.y)*np.log(1-self.y_pred))
    
    def predict(self, X):
        y_pred = self.sigmoid(X[:self.W.shape[0]] @ self.W)
        preds = "spam" if y_pred > 0.5 else "ham"
        return preds
    
    def fit(self, epochs, lr):
        # x = x_0 - lr*gradient
        for i in range(epochs):
            self.y_pred = self.sigmoid(self.X @ self.W)
            gradient = np.dot(self.X.T, (self.y_pred - self.y)) / self.y.size
            self.W -= lr*gradient

            self.loss_history.append(self.nll())

            if i % 100 == 0:
                print(f"Epoch: {i}, NLL: {self.nll()}")
    
    def plot_loss_curve(self):
        plt.plot(self.loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Negative Log Likelihood")
        plt.title("Loss Curve")
        plt.show()

# try:
print("Loading. . .")
with open("logistic_regression_model_2.pkl", "rb") as f:
        model = pickle.load(f)
        data_given = input("Enter Email: ")
        X_t = vectorizer.transform(data_given)
        y_t = model.predict(X_t)
        print(y_t)
        # count = 0
        # for i in range(y_t):
        #     if y_t[i]==y_test[i]:
        #         count+=1
        # print(f"Accuracy: {count / len(y_t)}.4f")
        

# except:
#     print("Training. . .")
#     W  = np.random.randn(X.shape[1])
#     model = LogisticRegression(X, W, y)
#     model.fit(100000, 0.01)
#     model.plot_loss_curve()

#     with open("logistic_regression_model_2.pkl", "wb") as f:
#         pickle.dump(model, f)

