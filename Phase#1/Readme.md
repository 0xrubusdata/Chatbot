# 🚀 Phase #1 – Fondations du NLP Moderne
## 🎯 Objectif de cette phase :

Comprendre comment une machine représente et apprend le langage,
et pourquoi les architectures modernes (Transformers) ont supplanté les LSTM/RNN.

### 🧩 Étape 1 — Représentation du texte : du mot au vecteur
#### 🗣️ 1. Texte brut → Tokens

Un modèle ne comprend pas les mots, mais des entiers.

Exemple :
```
"Le chat dort."
→ ["Le", "chat", "dort", "."]
→ [101, 3056, 980, 13]
```
Ces entiers sont ensuite transformés en vecteurs continus (embeddings) :
```
embedding("chat") = [0.12, -0.45, 0.87, ...]
```

Chaque mot devient un point dans un espace vectoriel →
proche s’il a un sens similaire.  
👉 C’est la base de la compréhension sémantique.

### ⚙️ Étape 2 — Modèles séquentiels : le LSTM

Voyons le cœur du mécanisme :

#### 🔁 Problème :

Le texte est séquentiel → chaque mot dépend du précédent.

Exemple :

“Le chat dort sur le canapé.”

Le mot “dort” dépend du contexte “Le chat”.

Les RNN puis LSTM ont été conçus pour ça : ils gardent une “mémoire” à chaque pas de temps :

```
h_t = f(W * x_t + U * h_(t-1))
```

Mais… les LSTM ont une mémoire limitée →
ils oublient vite le contexte quand les phrases deviennent longues (👋 adieu Bitcoin trends à long terme).

### ⚡ Étape 3 — Les Transformers

Les Transformers ont changé la donne car :

- ils traitent toute la séquence en parallèle, 
- chaque mot “regarde” tous les autres via le mécanisme d’attention.

“Attention is all you need.”
(Vaswani et al., 2017 — le papier fondateur)

Grâce à ça :

- plus de perte de contexte,
- meilleure scalabilité,
- plus d’efficacité sur du texte long et complexe.

#### 🤖 Étape 4 — Embeddings & Modèles préentraînés

Aujourd’hui, on n’entraîne presque plus de modèles “from scratch”.
On utilise des modèles préentraînés sur d’énormes corpus (BERT, GPT, Mistral…)
puis on les fine-tune sur notre tâche.

Ces modèles utilisent des embeddings contextuels :

Le mot “banc” dans “le banc du parc” ≠ “le banc de poissons”.

### 📘 Étape 5 — Ce qu’on va faire concrètement
#### 1. Créer un mini-LSTM en PyTorch (comme ton modèle BTC)

👉 pour comprendre la logique séquentielle.

#### 2. Le comparer à un petit modèle HF local (ex: distilgpt2)

👉 pour voir la différence de performance et de complexité.

#### 3. Visualiser les embeddings

👉 pour “voir” comment les mots sont représentés.
