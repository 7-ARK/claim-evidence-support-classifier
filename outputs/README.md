# Outputs

outputs/full holds the results of the full GPU run with configs/train.yaml: 4 epochs on 840 training rows, scored on 180 test rows.

metrics.json has Macro F1, accuracy, per class scores, and the confusion matrix for LoRA and both baselines.
baseline_metrics.json has the majority class and frozen backbone baselines on their own.
report.txt is a plain text summary with the comparison table.
confusion_matrix.csv and confusion_matrix.png show the LoRA confusion matrix.
errors.json lists every missed test row. error_analysis.md quotes a sample of them with notes.

Full run test scores: LoRA Macro F1 0.8751, frozen backbone 0.3167, majority class 0.1667.

outputs/smoke holds the short CPU wiring check from configs/smoke.yaml. Those scores are not a quality claim. Each method lands on Macro F1 0.1667 because each predicts one class.

adapters/lora holds the adapter from the full run. Its train_meta.json reads run_name full. Running either config again overwrites that folder.
