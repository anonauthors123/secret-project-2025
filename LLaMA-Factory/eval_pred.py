from ast import arg
import os
import json
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

def main(args):
    with open(os.path.join(args.pred_root, 'generated_predictions.jsonl'), 'r') as f:
        predictions = [json.loads(line) for line in f]

    for p in predictions:
        p['label'] = p['label'].strip()
        p['predict'] = p['predict'].strip()

    labels = [p['label'] == 'yes' for p in predictions]
    predictions = [p['predict'] == 'yes' for p in predictions]
    
    acc = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    mcc = matthews_corrcoef(labels, predictions)
    tp, fp, tn, fn = confusion_matrix(labels, predictions).ravel()
    fpr = fp / (fp + tn + 1e-8)
    
    results = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mcc': mcc,
            'fpr': fpr
        }

    print(results)
    
    with open(os.path.join(args.pred_root, 'metrics.json'), 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_root', type=str)
    args = parser.parse_args()
    main(args)
    