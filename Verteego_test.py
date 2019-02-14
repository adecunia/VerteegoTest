from elmoformanylangs import Embedder
import numpy as np

e = Embedder('C:/Users/a2c18/Desktop/150/')

words = ['souris','mulot','escargot', 'chaise']

#Envoi des données aux modèles ELMO

embeddings = e.sents2elmo(words)

print("Souris : ", embeddings[0],"\n")
print("Mulot : ", embeddings[1],"\n")
print("Escargot : ", embeddings[2],"\n")
print("Chaise : ", embeddings[3],"\n")

#On affecte chaque mots à son vecteur correspondant
souris = embeddings[0][0]
mulot = embeddings[1][0]
escargot = embeddings[2][0]
chaise = embeddings[3][0]


#Calcul distance
dist_souris_mulot = np.linalg.norm(souris-mulot)
dist_souris_escargot = np.linalg.norm(souris-escargot)
dist_souris_chaise = np.linalg.norm(souris-chaise)

dist_mulot_escargot = np.linalg.norm(mulot-escargot)
dist_mulot_chaise = np.linalg.norm(mulot-chaise)

dist_escargot_chaise = np.linalg.norm(escargot-chaise)

print("Distance euclidienne\n")
print("Souris mulot :" , dist_souris_mulot,"\n")
print("Souris escargot :" , dist_souris_escargot,"\n")
print("Souris chaise :" , dist_souris_chaise,"\n")
print("mulot escargot :" , dist_mulot_escargot,"\n")
print("mulot_chaise :" , dist_mulot_chaise,"\n")
print("escargot_chaise :" , dist_escargot_chaise,"\n")



