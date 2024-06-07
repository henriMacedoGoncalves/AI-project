# %%
import pandas as pd
import math
import numpy as np

# %%
def calcEntropy(df, attribut = 'Class'):
    size = df.shape[0]
    series = df[attribut].value_counts()
    entropy = 0
    for v in series:
        tmp = v/size
        entropy += tmp*math.log(tmp,2)
        
    return -entropy

# %%
def calcGain(df,attribut = 'Attr_A',attribut_cible = 'Class'):
    entropy = calcEntropy(df)
    gain = 0
    tmp_gain = 0
    split_value = 0
    partitions = [None,None]
    tmp_partitions = [None,None]
    tmp = 0
    
    quartiles = df[attribut].quantile([0.25,0.5,0.75])
    
    for quartile_value in quartiles:
        tmp_partitions[0] = df[df[attribut] <= quartile_value]
        tmp_partitions[1] = df[df[attribut] > quartile_value]
        
        for i in range(len(tmp_partitions)):
            tmp += (len(tmp_partitions[i]) / len(df)) * calcEntropy(tmp_partitions[i])
        tmp_gain = entropy - tmp
        
        if tmp_gain > gain:
            gain = tmp_gain
            split_value = quartile_value
            partitions = tmp_partitions
            
    return (attribut, gain, split_value, partitions)

# %%
def meilleur_attribut(df,columns):
    result = (0,0,0,[])
    for i in columns:
        tmp = calcGain(df,i)
        if result[1] < tmp[1]:
                result = tmp
    return result

# %%
class Noeud:
    def __init__(self,attribut=None,split_value=None, prediction=None, feuille=False, gauche=None, droite=None):
        self.attribut = attribut
        self.split_value = split_value
        self.prediction = prediction
        self.feuille = feuille
        self.gauche = gauche
        self.droite = droite
        
    def __str__(self):
        return "<"+str(self.split_value)+" "+str(self.gauche)+" "+str(self.droite)+">"
    
    def __repr__(self):
        return str(self.split_value)
    
    def node_result(self, spacing=' '):
        s = ''
        for v in range(len(self.prediction.values)):
            s += ' Class ' + str(self.prediction.index[v]) + ' Count: ' + str(self.prediction.values[v]) + '\n' + spacing
        return s

# %%
def construction_arbre(df=None,cible = 'Class',seuil = 3,attributs_restants = [],profondeur = 0):
    attribut,gain,split,partitions = meilleur_attribut(df,attributs_restants)
    prediction = df[cible].value_counts()
    
    if (profondeur > seuil) or (len(attributs_restants)==0) or (len(partitions)==0):
        return Noeud(prediction=prediction,feuille=True)
    attributs_restants = attributs_restants.drop(attribut)
    gauche = construction_arbre(df=partitions[0],seuil=seuil,attributs_restants=attributs_restants,profondeur=profondeur+1)
    droite = construction_arbre(df=partitions[1],seuil=seuil,attributs_restants=attributs_restants,profondeur=profondeur+1)
    return Noeud(split_value=split,attribut=attribut,gauche=gauche,droite=droite,prediction=prediction)

# %%

def print_tree(node, spacing = ' '):
    if node is None:
        return
    if node.feuille:
        print(spacing + node.node_result(spacing))
        return
    print('{}[Attribute: {} Split value: {}]'.format(spacing, node.attribut, node.split_value))
        
    print(spacing + '> Gauche')
    print_tree(node.gauche, spacing + '-')
        
    print(spacing + '> Droite')
    print_tree(node.droite,spacing + '-')
    return

# %%
def predictPercentage(prediction):
    tmp = prediction.sum()
        
    result = prediction/tmp
    return result         

# %%
def inference(instance,noeud,attribut=None):
    if noeud.feuille:
        result = predictPercentage(noeud.prediction)
        return result.idxmax()
    else:
        valeur_attribut = instance[noeud.attribut]
        if valeur_attribut < noeud.split_value:
            return inference(instance,noeud.gauche,noeud.attribut)
        else:
            return inference(instance,noeud.droite,noeud.attribut)

# %%
# Fonction pour evaluer le resultat du modèle
# La matrix contient pour chaque classe ses valeurs TP,FP,FN et TN
# On calcule les metriques et renvoie f1score
def evaluateResult(matrix):
    accuracies = []
    precisions = []
    recalls = []
    f1scores = []
        
    for i in range(len(matrix)):
        TP = matrix[i,0]
        FP = matrix[i,1]
        FN = matrix[i,2]
        TN = matrix[i,3]
        
        if TP + FP + FN + TN > 0:
            accuracy = round((TP + TN) / (TP + FP + FN + TN),4)
        else:
            accuracy = 0.0
        
        if TP + FP > 0:
            precision = round(TP / (TP + FP),4)
        else:
            precision = 0.0
        
        if TP + FN > 0:
            recall = round(TP / (TP + FN),4)
        else:
            recall = 0.0
        
        if precision + recall > 0:
            f1score = round(2 * ( (precision * recall) / (precision + recall)),4)
        else:
            f1score = 0.0
        
        accuracies.append(accuracy)    
        precisions.append(precision)
        recalls.append(recall)
        f1scores.append(f1score)
        
    avg_accuracy = round(sum(accuracies) / len(matrix),4)
    avg_precision = round(sum(precisions) / len(matrix),4)
    avg_recall = round(sum(recalls) / len(matrix),4)
    avg_f1score = round(sum(f1scores) / len(matrix),4)
    
    print("Metric\t\tAverage\t\tFor each class (0,1,2,3)")
    print(f"Accuracy:\t{avg_accuracy}\t\t{accuracies}")
    print(f"Precision:\t{avg_precision}\t\t{precisions}")
    print(f"Recall:\t\t{avg_recall}\t\t{recalls}")
    print(f"F1Score:\t{avg_f1score}\t\t{f1scores}\n") 
    
    return avg_f1score

# %%
def print_matrix(matrix_confusion,matrix_results):
    
    print("Matrix de confusion")
    
    print("Class\t0\t1\t2\t3")
    
    for i in range(len(matrix_confusion)):
    
        if i == 1:        
            print(f"{i}\t{matrix_confusion[i,0]}\t{matrix_confusion[i,1]}\t{matrix_confusion[i,2]}\t{matrix_confusion[i,3]}\tTrue label")
        else:
            print(f"{i}\t{matrix_confusion[i,0]}\t{matrix_confusion[i,1]}\t{matrix_confusion[i,2]}\t{matrix_confusion[i,3]}")
    
    print("\tPredicted label\n")
    
    print("Resultats")
    print("Class\tTP\tFP\tFN\tTN")
    
    for i in range(len(matrix_results)):
        
        print(f"{i}\t{matrix_results[i,0]}\t{matrix_results[i,1]}\t{matrix_results[i,2]}\t{matrix_results[i,3]}")
    
    print()
    

# %%
# Fonction créé pour évaluer le modèle
# On teste le modèle avec les données de test
# Pour chaque prediction, on vérifie si c'est la bonne prédiction
# On remplit une matrice de taille 4 classes * 4 valeurs (tp,fp,fn,tn)
# Avec la matrice, on calcule le résultat à retourner
def evaluateModel(df,tree):
    matrix_results = np.zeros((4,4), dtype=np.int32)
    matrix_confusion = np.zeros((4,4), dtype=np.int32)
    tp = 0
    
    for index,instance in df.iterrows():
        
        true_value = int(instance.iloc[-1])
        predicted_value = int(inference(noeud=tree,instance=instance))
        
        matrix_confusion[true_value,predicted_value] += 1
        
        if true_value == predicted_value:
            matrix_results[true_value,0] += 1
        else:
            matrix_results[true_value,2] += 1
            matrix_results[predicted_value,1] += 1
        
        for j in range(4):
            if j != true_value and j != predicted_value:
                matrix_results[j,3] += 1
                
    print_matrix(matrix_confusion,matrix_results)
    
    results = evaluateResult(matrix_results)
    return results

# %%
# Fonction créé pour déterminer les 2 meilleurs seuils
# On crée un arbre pour chaque seuil et on l'évalue
# Le résultat retourné est comparé avec les 2 meuilleures résultats obtenus
# On retourne les 2 meuilleures modèles avec leur résultats
def meilleur_seuil(train_df,test_df):
    tmp = 0
    tmp_tree = None
    max1 = 0
    tree1 = None
    max2 = 0
    tree2 = None
    seuil1 = 0
    seuil2 = 0
    for i in range(3,9):
        print(f"Seuil: {i}")
        tmp_tree = construction_arbre(df=train_df,seuil=i,attributs_restants=train_df.columns[:-1])
        tmp = evaluateModel(test_df,tmp_tree)
        if (max1 < tmp):
            max2 = max1
            tree2 = tree1
            seuil2 = seuil1
            max1 = tmp
            tree1 = tmp_tree
            seuil1 = i
        elif (max2 < tmp):
            max2 = tmp
            tree2 = tmp_tree
            seuil2 = i
    
    return tree1,seuil1,tree2,seuil2

# %%
df = pd.read_csv('synthetic.csv')

train_df = df.sample(frac=0.80, random_state=42)
train_index = train_df.index
test_df = df.drop(index=train_index)

arbre1,seuil1,arbre2,seuil2 = meilleur_seuil(train_df,test_df)

# %%
print(f"Seuil: {seuil1}")
evaluateModel(df=test_df,tree=arbre1)
print("\nArbre de décision")
print_tree(arbre1)

# %%
print(f"Seuil: {seuil2}")
evaluateModel(df=test_df,tree=arbre2)
print_tree(arbre2)

# %%



