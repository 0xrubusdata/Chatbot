## Explications rapides & conseils pratiques

### Tokeni``ation
- On utilise ```AutoTokenizer``` (par défaut ```gpt2```) pour convertir ton texte en ```input_ids```. Cela évite d’implémenter ton propre vocabulaire.
- Pour LSTM, l’```Embedding``` utilise la même taille de vocab que le tokenizer pour comparabilité.

### Formulation de la tâche
- Modèle prédit le next token pour chaque position (teacher forcing). Les targets ```y``` sont ```input_ids``` décalés d’un pas.

### Comparaison LSTM vs Transformer
- LSTM : plus simple, pédagogique, mais limité pour dépendances longues et texte riche.
- GPT-2 (Transformer) : architecture moderne, meilleurs résultats même avec fine-tuning court.

### Évaluation
- On calcule ```loss``` cross-entropy et ```perplexity = exp(loss)``` sur validation — métrique standard en LM.
- Observe que perplexity peut exploser si la loss est élevée.

### Génération
- LSTM : génération token-by-token en échantillonnant la distribution softmax. Attention : si seq_len utilisé en entrée est plus court que le prompt, il trunc.
- GPT-2 : on utilise ```model.generate``` (sampling si ```temperature>0```).

### Précautions mémoire & perf
- GPT2 base est ~500MB / 124M params — idéal pour GPU unique. Si tu es limité en VRAM, réduis ```batch_size``` ou utilise ```--max_tokens``` pour tests rapides.
- Pour entraînements sérieux sur corpus large, tu peux switcher au ```Trainer``` HF ou utiliser gradient accumulation / mixed precision (amp).

### Fine-tuning GPT-2 local / Ollama
- Ollama sert pour l’inférence. Pour fine-tuning local, ```transformers``` + PyTorch comme ci-dessus est la voie. Une fois fine-tuné, exporte (saved dir) et tu peux convertir / packager pour inference dans ta stack (ONNX / TorchScript) si besoin.
