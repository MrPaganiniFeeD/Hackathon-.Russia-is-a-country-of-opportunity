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

def pre_calc_score(gt, sub):
  #sub - submission файл
  #gt - grounded true файл
  pred_seconds = []
  correct_predictions = []
  AE_count_rules_FP = []
  AE_count_rules_FN = []

  #Проходимся по всем нарушениям из gt
  for i, r_gt in gt.iterrows():
    #Берем строки из sub, которые относятся к одному видео и имеют нарушение того же правила, что и анализируемое нарушение в строке gt
    video_sub = sub[(sub['номер видео'] == r_gt['номер видео']) & (sub['наименование нарушения'] == r_gt['наименование нарушения'])]
    #Берем строки из gt, которые относятся к одному видео и имеют нарушение того же правила, что и анализируемое нарушение в строке gt
    video_gt = gt[(gt['номер видео'] == r_gt['номер видео']) & (gt['наименование нарушения'] == r_gt['наименование нарушения'])]

    #Обработка случая, когда есть нарушения правила в grounded true, а в submission его нарушений нет
    if len(video_sub)==0:
      #Если в предсказании не было найдено нарушеного правила, хотя на самом деле оно было, то correct_prediction = False
      pred_seconds.append(np.NaN)
      correct_predictions.append(False)

      #В предсказании было больше нарушений, чем есть на самом деле
      FP = max(0 - len(video_gt), 0) # всегда будет 0
      #В предсказании было меньше нарушений, чем есть на самом деле
      FN = abs(min(0 - len(video_gt), 0))

      AE_count_rules_FP.append(FP)
      AE_count_rules_FN.append(FN)
      continue

    true_sec = r_gt['время нарушения (в секундах)']

    #Берем ближайшую pred сенунду к true секунде для взятого правила
    pred_sec = min(video_sub['время нарушения (в секундах)'].values, key=lambda x: abs(x - true_sec))
    pred_seconds.append(pred_sec)

    #При попадание в интервал 5 секунд от фактического нарушения зачисляется предсказание
    correct_prediction = np.abs(pred_sec-true_sec)<5
    correct_predictions.append(correct_prediction)


    #В предсказании было больше нарушений, чем есть на самом деле
    FP = max(len(video_sub) - len(video_gt), 0)
    #В предсказании было меньше нарушений, чем есть на самом деле
    FN = abs(min(len(video_sub) - len(video_gt), 0))


    AE_count_rules_FP.append(FP)
    AE_count_rules_FN.append(FN)


  gt['pred_seconds'] = pred_seconds
  gt['Корректность прдсказания'] = correct_predictions
  gt['В предсказании было больше нарушений на кол-во'] = AE_count_rules_FP
  gt['В предсказании было меньше нарушений на кол-во'] = AE_count_rules_FN
  return gt




