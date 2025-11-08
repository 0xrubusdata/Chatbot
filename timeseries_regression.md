# Code : LSTM régressif + Transformer

## Explications & instructions rapides

1 - Format CSV attendu : un fichier CSV avec une colonne numérique (ex. close) triée chronologiquement. Si tu as une colonne date, le script trie dessus.  
Exemple minimal :
```
date,open,high,low,close,volume
2020-01-01,...
```

2 - Prétraitement : on normalise via MinMaxScaler. C’est simple et stable pour les réseaux. Tu peux remplacer par StandardScaler si tu préfères.

3 - Séquences : window=30 → on utilise 30 pas de temps pour prédire le pas suivant. Ajuste selon la granularité (daily, hourly…).

4 - Comparaison modèles :
- LSTMRegressor est le classique LSTM.
- TransformerRegressor est une version transformer-encoder (PyTorch natif) adaptée aux séries temporelles ; cela montre la différence d’architecture.

5 -Evaluation : on calcule MSE sur la validation (on sauvegarde le meilleur modèle). Pour un usage réel, calcule aussi MAE et regarde des prédictions inversées (scaler.inverse_transform) pour lire en unités $$.

6 - Où brancher tes données : lance le script avec ```--csv /chemin/vers/btc.csv --target_col close```.  
Exemple :
```
python timeseries_regression.py --csv data/btc_daily.csv --target_col close --window 60 --epochs 50 --batch_size 128
```

## Remarques importantes & prochaines étapes

Prédire le prix du BTC est intrinsèquement bruyant 
— un LSTM peut apprendre des patterns locaux, mais ce n’est pas une garantie de prédiction fiable. 

C’est parfait pour apprendre cependant.

Si tu veux comparer avec un modèle Hugging Face (p.ex. pour utiliser transformers), on peut :
- soit tokenizer/quantifier la série (moins intuitif),
- soit utiliser HF pour entraîner un sequence model en adaptant Trainer (je peux te fournir un script qui wrappe ces modèles HF).

Pour fine-tuning sérieux sur ton serveur (Phase 4) : 
- LoRA/QLoRA s’appliquent surtout aux LLMs textuels. 

Ici, pour séries tu peux utiliser techniques classiques (checkpointing, mixed precision, scheduler, etc.).
