# Brief 3 data splits

The split functions are implemented in `src/brief3/pipeline.py` and read `src/brief3/data/raw/support_messages.csv`. The raw dataset contains 44,600 rows. All strategies use stratification by `intent` and `random_state=54`.

The helper first separates the training set, then divides the remaining temporary set equally into test and validation. The returned order is always `x_train, x_test, x_val, y_train, y_test, y_val`.

| Name | Input rows | Train | Test | Validation | Use |
| --- | ---: | ---: | ---: | ---: | --- |
| `default` | 44,600 | 70% | 15% | 15% | Baseline comparison |
| `balanced` | 44,600 | 75% | 12.5% | 12.5% | More training data with stratified evaluation |
| `drop_duplicates` | 28,032 unique `message_text` rows | 70% | 15% | 15% | Removes repeated messages before splitting |
| `mixed_language` | 34,620 rows after removing `language_hint == "mixed"` | 70% | 15% | 15% | Evaluates without mixed-language messages |
| `imbalance_aware` | 44,600 | 80% | 10% | 10% | Gives minority intent classes a larger training share |

The `balanced` and `imbalance_aware` names describe their training proportions; all strategies still use stratification rather than synthetic resampling or class weights. The `mixed_language` implementation filters those rows from the input before creating all three subsets.

## Select a split

Pass a split name to an individual model:

```powershell
python -c "from src.brief3.baseline_models import train_logistic_regression; train_logistic_regression(split_name='balanced')"
```

Or pass it to the aggregate runner:

```powershell
python -c "from src.brief3.train_all_models import train_all_models; train_all_models('drop_duplicates')"
```
