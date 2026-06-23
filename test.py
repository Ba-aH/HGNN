for f in ~/checkpoints/*/best_model.pt; do
    echo "$f"
    python -c "import torch; ck=torch.load('$f', map_location='cpu'); print('  epoch:', ck.get('epoch'), '  val_mrr:', round(ck.get('val_mrr',0), 4))"
done