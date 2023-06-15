def test_load():
  return 'loaded'

def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]

def cond_prob(table, evidence, evidence_value, target, target_value):
  t_subset = up_table_subset(table, target, 'equals', target_value)
  e_list = up_get_column(t_subset, evidence)
  p_b_a = sum([1 if v==evidence_value else 0 for v in e_list])/len(e_list)
  return p_b_a + .01

def cond_probs_product(table, evidence_val, target_column, target_val):
  cond_prob_list = []
  evidence_columns = up_list_column_names(table)[:-1]
  evidence_row = up_zip_lists(evidence_columns, evidence_val)
  for evidence_column, evidence_val in evidence_row:
    cond_prob_list += [cond_prob(table, evidence_column, evidence_val, target_column, target_val)]
  partial_numerator = up_product(cond_prob_list)
  return partial_numerator

def prior_prob(table, target, target_value):
  t_list = up_get_column(table, target)
  p_a = sum([1 if v==target_value else 0 for v in t_list])/len(t_list)
  return p_a

def naive_bayes(table, evidence_row, target):
  #compute P(Flu=0|...) by collecting cond_probs in a list, take the produce of the list, finally multiply by P(Flu=0)
  partial_numerator_1 = cond_probs_product(table, evidence_row, target, 0) 
  full_numerator_1 = partial_numerator_1 * prior_prob(table, target, 0)
  #do same for P(Flu=1|...)
  partial_numerator_2 = cond_probs_product(table, evidence_row, target, 1)
  full_numerator_2 = partial_numerator_2  * prior_prob(table, target, 1)
  #Use compute_probs to get 2 probabilities
  neg, pos = compute_probs(full_numerator_1, full_numerator_2)
  
  #return your 2 results in a list
  return [neg, pos]

def metrics(zipped_list):
  assert type(zipped_list) == list, f"Parameter should be a list, but instead is {type(zipped_list)}"
  assert all(type(x) == list for x in zipped_list), f"Parameter is not a list of lists"
  assert all(len(x) == 2 for x in zipped_list), f"Parameter is not a zipped list of pairs"
  assert all(isinstance(item, (int, float)) for mini_list in zipped_list for item in mini_list), f"Items inside of nested lists are not integers"
  assert all(item >= 0 for mini_list in zipped_list for item in mini_list), f"Items inside of nested lists should be positive but are negative"

  accuracy = sum(pred==act for pred,act in zipped_list)/len(zipped_list)
  tn = sum([1 if pair==[0,0] else 0 for pair in zipped_list])
  tp = sum([1 if pair==[1,1] else 0 for pair in zipped_list])
  fp = sum([1 if pair==[1,0] else 0 for pair in zipped_list])
  fn = sum([1 if pair==[0,1] else 0 for pair in zipped_list])
  precision = 0 if (tp + fp == 0) else tp/(tp + fp)
  recall = 0 if (tp + fn == 0) else tp/(tp + fn)
  f1 = 0 if (precision + recall == 0) else (2*precision*recall)/(precision + recall)
  mets_dict = {'Precision': f'{precision}', 'Recall': f'{recall}', 'F1': f'{f1}', 'Accuracy': f'{accuracy}'}

  return mets_dict

def try_archs(full_table, target, architectures, thresholds):
  train_table, test_table = up_train_test_split(full_table, target, .4)

  #copy paste code here
  for arch in architectures:
    all_results = up_neural_net(train_table, test_table, arch, target)
    #loop through thresholds
    all_mets = []
    for t in thresholds:
      all_predictions = [1 if pos>=t else 0 for neg,pos in all_results]
      pred_act_list = up_zip_lists(all_predictions, up_get_column(test_table, target))
      mets = metrics(pred_act_list)
      mets['Threshold'] = t
      all_mets = all_mets + [mets]
  
    print(f'Architecture: {arch}')
    print(up_metrics_table(all_mets))

  return None  #main use is to print out threshold tables, not return anything useful.

def run_random_forest(train, test, target, n):
  X = up_drop_column(train, target)
  y = up_get_column(train, target)  
  k_feature_table = up_drop_column(test, target) 
  k_actuals = up_get_column(test, target)  
  clf = RandomForestClassifier(n, max_depth=2, random_state=0)  
  clf.fit(X, y)
  probs = clf.predict_proba(k_feature_table)
  pos_probs = [p for n,p in probs]
  pos_probs[:5]
  all_mets = []
  for t in thresholds:
    all_predictions = [1 if pos>t else 0 for pos in pos_probs]
    pred_act_list = up_zip_lists(all_predictions, k_actuals)
    mets = metrics(pred_act_list)
    mets['Threshold'] = t
    all_mets = all_mets + [mets]
  all_mets[:2]
  metrics_table = up_metrics_table(all_mets)
  metrics_table
  print(metrics_table)  #output we really want - to see the table
  return None


