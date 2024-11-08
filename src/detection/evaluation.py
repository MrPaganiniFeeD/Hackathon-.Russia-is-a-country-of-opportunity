import pandas as pd
import numpy as np

def simul_sub(df):
  violation_list = df['наименование нарушения'].unique()
  len_base_df = len(df)


  fraction_to_remove = 0.4
  fraction_to_duplicate = 0.3

  num_to_remove = int(len_base_df * fraction_to_remove)
  num_to_duplicate = int(len_base_df * fraction_to_duplicate)
  rows_to_remove = df.sample(num_to_remove).index
  data_dropped = df.drop(rows_to_remove)
  rows_to_duplicate = data_dropped.sample(num_to_duplicate)
  data_final = pd.concat([data_dropped, rows_to_duplicate])
  data_final = data_final.reset_index(drop=True)

  fraction_to_change = 0.5

  rows_to_change = data_final.sample(frac=fraction_to_change).index
  data_final.loc[rows_to_change, 'наименование нарушения'] = np.random.choice(violation_list, size=len(rows_to_change))
  noise_level = 10
  noise = np.random.randint(-noise_level, noise_level + 1, size=data_final.shape[0])

  data_final['время нарушения (в секундах)'] = data_final['время нарушения (в секундах)'] + noise

  return data_final

metric_path = "videos/markup_df_train.xlsx"

gt = pd.read_excel(metric_path)
sub = simul_sub(gt.copy())






