import os
import json
import pandas as pd
import pickle
import numpy as np
import random
import argparse
import sys
import sklearn.ensemble as ensemble
import sklearn.metrics as met
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
random.seed(1023)
import joblib

RKG_ROOT_PATH = os.getcwd()
INDIR = os.path.join(RKG_ROOT_PATH,"Split")

# trainX from embeddings
def generate_Xy(emb, df, pair_emb="concatenate"):
    if pair_emb == "concatenate":
        X = np.vstack([np.hstack([emb[df.loc[index,'source']], emb[df.loc[index,'target']]]) for index in range(len(df))])
        y = np.array(df['y'].values)
    return [X,y]

def calculate_f1score(preds, labels, average='binary'):
    preds = np.array(preds)
    y_pred_tags = np.argmax(preds, axis=1)
    abels = np.array(labels)
    f1score = f1_score(labels, y_pred_tags, average=average)
    return f1score
 
def calculate_acc(preds, labels):
    preds = np.array(preds)
    y_pred_tags = np.argmax(preds, axis=1)
    labels = np.array(labels)
    acc = (y_pred_tags == labels).astype(float).mean()
    return acc

def calculate_mrr(drug_disease_pairs, random_pairs, N):
    '''
    This function is used to calculate Mean Reciprocal Rank (MRR)
    reference paper: Knowledge Graph Embedding for Link Prediction: A Comparative Analysis
    '''
    
    ## only use tp pairs
    drug_disease_pairs = drug_disease_pairs.loc[drug_disease_pairs['y']==1,:].reset_index(drop=True)
    
    Q_n = len(drug_disease_pairs)
    score = 0
    ranklist = []
    for index in range(Q_n):
        
        query_drug = drug_disease_pairs['source'][index]
        query_disease = drug_disease_pairs['target'][index]
        #print(f"index is {index}, {query_drug}, {query_disease}")
        this_query_score = drug_disease_pairs['prob'][index]
        all_random_probs_for_this_query = list(random_pairs.loc[random_pairs['source'].isin([query_drug]),'prob'])
        all_random_probs_for_this_query += list(random_pairs.loc[random_pairs['target'].isin([query_disease]),'prob'])
        all_random_probs_for_this_query = all_random_probs_for_this_query[:N]
        all_in_list = np.array(([this_query_score] + all_random_probs_for_this_query), dtype=np.float64)
        if len(all_in_list) > 2:
            order = all_in_list.argsort()
            rank = int(len(all_in_list) - np.where(order==0)[0][0])
            #print(all_in_list, rank)
            ranklist.append(rank)
            #rank = list(torch.tensor(all_in_list).sort(descending=True).indices.numpy()).index(0)+1
            score += 1/rank
        else:
            print(f"{query_drug} or {query_disease} not in random pairs.")
            
    final_score = score/Q_n
    
    return final_score, ranklist, Q_n

def hitk_postmrr(ranklist, Q_n, ks):
    hitks = {}
    
    for k in ks:
        count = 0
        for r in ranklist:
            if r <= k:
                count+=1
        hitks[f"hit_at_{k}"] = count/Q_n
    
    return hitks


def calculate_hitk(drug_disease_pairs, random_pairs, N, k=1):
    '''
    This function is used to calculate Hits@K (H@K)
    reference paper: Knowledge Graph Embedding for Link Prediction: A Comparative Analysis
    '''
    
    ## only use tp pairs
    drug_disease_pairs = drug_disease_pairs.loc[drug_disease_pairs['y']==1,:].reset_index(drop=True)
    
    Q_n = len(drug_disease_pairs)
    count = 0
    for index in range(Q_n):
        query_drug = drug_disease_pairs['source'][index]
        query_disease = drug_disease_pairs['target'][index]
        this_query_score = drug_disease_pairs['prob'][index]
        all_random_probs_for_this_query = list(random_pairs.loc[random_pairs['source'].isin([query_drug]),'prob'])
        all_random_probs_for_this_query += list(random_pairs.loc[random_pairs['target'].isin([query_disease]),'prob'])
        all_random_probs_for_this_query = all_random_probs_for_this_query[:N]
        all_in_list = [this_query_score] + all_random_probs_for_this_query
        rank = list(torch.tensor(all_in_list).sort(descending=True).indices.numpy()).index(0)+1
        if rank <= k:
            count += 1
        
    final_score = count/Q_n
    
    return final_score


def calculate_rank(data_pos_df, all_drug_ids, all_disease_ids, entity_embeddings_dict, all_tp_pairs_dict, fitModel, mode='both'):
    res_dict = dict()
    total = data_pos_df.shape[0]
    for index, (source, target) in enumerate(data_pos_df[['source','target']].to_numpy()):
        print(f"calculating rank {index+1}/{total}", flush=True)
        this_pair = source + '_' + target
        X_drug = np.vstack([np.hstack([entity_embeddings_dict[drug_id],entity_embeddings_dict[target]]) for drug_id in all_drug_ids])
        X_disease = np.vstack([np.hstack([entity_embeddings_dict[source],entity_embeddings_dict[disease_id]]) for disease_id in all_disease_ids])
        all_X = np.concatenate([X_drug,X_disease],axis=0)
        pred_probs = fitModel.predict_proba(all_X)
        temp_df = pd.concat([pd.DataFrame(zip(all_drug_ids,[target]*len(all_drug_ids))),pd.DataFrame(zip([source]*len(all_disease_ids),all_disease_ids))]).reset_index(drop=True)
        temp_df[2] = temp_df[0] + '_' + temp_df[1]
        temp_df[3] = pred_probs[:,1]
        this_row = temp_df.loc[temp_df[2]==this_pair,:].reset_index(drop=True).iloc[[0]]
        temp_df = temp_df.loc[temp_df[2]!=this_pair,:].reset_index(drop=True)
        if mode == 'both':
            ## without filter
            # (1) for drug
            temp_df_1 = pd.concat([temp_df.loc[temp_df[1] == target,:],this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            w_drug_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            # (2) for disease
            temp_df_1 = pd.concat([temp_df.loc[temp_df[0] == source,:],this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            w_disease_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            # (3) for both 
            temp_df_1 = pd.concat([temp_df,this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            w_both_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            ## filter 
            temp_df = temp_df.loc[~temp_df[2].isin(list(all_tp_pairs_dict.keys())),:].reset_index(drop=True)
             # (1) for drug
            temp_df_1 = pd.concat([temp_df.loc[temp_df[1] == target,:],this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            drug_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            # (2) for disease
            temp_df_1 = pd.concat([temp_df.loc[temp_df[0] == source,:],this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            disease_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            # (3) for both 
            temp_df_1 = pd.concat([temp_df,this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            both_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            res_dict[(source, target)] = [(w_drug_rank, w_disease_rank, w_both_rank), (drug_rank, disease_rank, both_rank)]

        elif mode == 'filter':
            ## filter 
            temp_df = temp_df.loc[~temp_df[2].isin(list(all_tp_pairs_dict.keys())),:].reset_index(drop=True)
             # (1) for drug
            temp_df_1 = pd.concat([temp_df.loc[temp_df[1] == target,:],this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            drug_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            # (2) for disease
            temp_df_1 = pd.concat([temp_df.loc[temp_df[0] == source,:],this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            disease_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            # (3) for both 
            temp_df_1 = pd.concat([temp_df,this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            both_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            res_dict[(source, target)] = [(None, None, None), (drug_rank, disease_rank, both_rank)] 
        else:
            ## without filter
            # (1) for drug
            temp_df_1 = pd.concat([temp_df.loc[temp_df[1] == target,:],this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            w_drug_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            # (2) for disease
            temp_df_1 = pd.concat([temp_df.loc[temp_df[0] == source,:],this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            w_disease_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            # (3) for both 
            temp_df_1 = pd.concat([temp_df,this_row]).reset_index(drop=True)
            temp_df_1 = temp_df_1.sort_values(by=3,ascending=False).reset_index(drop=True)
            w_both_rank = (temp_df_1.loc[temp_df_1[2] == this_pair,:].index[0]+1,temp_df_1.shape[0])
            res_dict[(source, target)] = [(w_drug_rank, w_disease_rank, w_both_rank), (None, None, None)]
    return res_dict

def evaluate(model, X, y_true, calculate_metric=True): 

    probas = model.predict_proba(X)
    
    if calculate_metric is True:

        ## calculate accuracy
        acc = calculate_acc(probas,y_true)
        ## calculate macro F1 score
        macro_f1score = calculate_f1score(probas,y_true,'macro')
        ## calculate micro F1 score
        micro_f1score = calculate_f1score(probas,y_true,'micro')
        
        return [acc, macro_f1score, micro_f1score, y_true, probas]
    else:
        return [None, None, None, y_true, probas]

def run_RF(emb_name, style, batch, tpstyle="stringent", tnstyle="stringent"):
    # read labeled pairs
    print("Read dd pair file")
    dfori = pd.read_csv(f"{os.path.join(INDIR, 'all_drug_disease_pairs_edges.tsv')}", sep='\t', header=0)
    print("Read embedding file")
    
    if emb_name == "biobert":
    # biobert embeddings
        print("Read Biobert embedding layer.")
        with open(f"{os.path.join(RKG_ROOT_PATH, style, 'embedding_biobert_namecat.pkl')}", "rb") as infile:
            bioemd_dict = pickle.load(infile)
    elif emb_name == "graphsage":
    # Graphsage output embeddings
        print("Read graphsage embedding layer.")
        with open(f"{os.path.join(RKG_ROOT_PATH, style, f'{style}_node_embed.pkl')}", "rb") as infile:
            bioemd_dict = pickle.load(infile)
    elif emb_name == "batch_mode":
        print("Use early batch embedding results.")
        with open(f"{os.path.join(RKG_ROOT_PATH, style, f'{batch}_node_embed.pkl')}", "rb") as infile:
            bioemd_dict = pickle.load(infile)

    dftp = dfori[dfori['y']==1].drop_duplicates(subset=['subject', 'object']).reset_index(drop=True)[['subject', 'object', 'y']].rename(columns={'subject':'source', 'object':'target', 'y':'y'})  
    print(f"TP has {dftp.shape[0]} DD pairs.")
    dftn = dfori[dfori['y']==0].drop_duplicates(subset=['subject', 'object']).reset_index(drop=True)[['subject', 'object', 'y']].rename(columns={'subject':'source', 'object':'target', 'y':'y'})
    print(f"TN has {dftn.shape[0]} DD pairs.")
    dftall = pd.concat([dftp, dftn], axis=0).reset_index(drop=True)
    emb_keys = []
    for k, v in bioemd_dict.items():
        emb_keys.append(k)
    emb_keys = set(emb_keys)
    dftall = dftall[(dftall['source'].isin(emb_keys)) & (dftall['target'].isin(emb_keys))].reset_index(drop=True)
    #breakpoint()
    allX, ally = generate_Xy(bioemd_dict, dftall)
    print("Read random pair files")
    dfrand = pd.read_csv(f"{os.path.join(INDIR, 'random_pairs.txt')}", sep='\t', header=0)
    dfrand = dfrand[dfrand['source'].isin(emb_keys)].reset_index(drop=True)
    dfrand = dfrand[dfrand['target'].isin(emb_keys)].reset_index(drop=True)
    print(f"{dfrand.shape[0]} random dd pairs.")
    randX, randy = generate_Xy(bioemd_dict, dfrand)
    
    ###### SPACE to generate relaxed TP samples for evaluating model later #####
    
    
    # train test split
    frac = 0.9
    print(f"train/test split ratio is {frac}/{(1-frac)}")
    test_idx = []
    for i in range(len(ally)):
        r = random.random()
        if r > frac:
            test_idx.append(i)
    mask = np.ones(ally.size, dtype=bool)
    mask[test_idx]=False
    train_X = allX[mask]
    train_y = ally[mask]
    test_X  = allX[test_idx]
    test_y  = ally[test_idx]

    dftrain = dftall[mask]
    dftest  = dftall[~mask]
    
    model_name = f'RF_model_{emb_name}_{tpstyle}_{tnstyle}.pt'
    model_file = os.path.join(RKG_ROOT_PATH, style, model_name)
    if not os.path.exists(model_file):
        # RF model and grid setup
        print("model does not exist. New training model")
        print("Random forest, grid search, crossvalidation=10")
        RF_model = ensemble.RandomForestClassifier(class_weight='balanced', random_state=1023, max_features="sqrt", oob_score=True, n_jobs=-1)
        param_grid = { 'max_depth' : [depth for depth in range(20,21,5)],
                        'n_estimators': [2000],
                        'class_weight': ["balanced", "balanced_subsample"]
            }

        gs_rf = GridSearchCV(estimator=RF_model, param_grid=param_grid, cv= 5, scoring='f1_macro', return_train_score=True)
        gs_rf.fit(train_X, train_y)
        print(f"The best hyper-parameter set of RF model based on gridseaerchCV is {gs_rf.best_estimator_}")
        best_rf = gs_rf.best_estimator_

        fitModel = best_rf.fit(train_X, train_y)

        # saves the model
        
        joblib.dump(fitModel, model_file)
    
    else:
        print("Model trained. Loading model")
        fitModel = joblib.load(model_file)
    
    print("Get accuracy and f1 scores.")
    train_acc, train_macro_f1score, train_micro_f1score, train_y_true, train_y_probs = evaluate(fitModel, train_X, train_y)
    test_acc, test_macro_f1score, test_micro_f1score, test_y_true, test_y_probs = evaluate(fitModel, test_X, test_y)
    
    dftrain.loc[:, "prob"] = train_y_probs[:,1]
    
    dftest.loc[:, "prob"] = test_y_probs[:,1]
    
    dfrand.loc[:, "prob"] = fitModel.predict_proba(randX)[:,1]
    
    dftrain.to_csv(os.path.join(RKG_ROOT_PATH, style, f"{emb_name}_dftrain.csv"))
    dftest.to_csv(os.path.join(RKG_ROOT_PATH, style, f"{emb_name}_dftest.csv"))
    ### Also add dfrelax to fitModel.predict ####
    
    print("Get MRR nd hit@k")
    test_mrr, test_ranklist, n_pairs = calculate_mrr(dftest, dfrand, 500)
    ks = [1,2,3,4,5,6,7,8,9,10,20,50]
    test_hitks = hitk_postmrr(test_ranklist, n_pairs, ks)

    metrics = {
        "train_acc": train_acc,
        "train_macro_f1score": train_macro_f1score,
        "train_micro_f1score": train_micro_f1score,
        "test_acc": test_acc,
        "test_macro_f1score": test_macro_f1score,
        "test_micro_f1score": test_micro_f1score,
        "test_mrr": test_mrr,
        "test_hitks": test_hitks
        
    }
    print(metrics)
    
    # dataframe dump
    dfallprob = pd.concat([dftrain, dftest], axis=0)
    dfallprob = dfallprob.reset_index(drop=True)
    dfallprob.to_csv(os.path.join(RKG_ROOT_PATH, style, f"{emb_name}_dfallprob.csv"))
    dfrand.to_csv(os.path.join(RKG_ROOT_PATH, style, f"{emb_name}_dfrandprob.csv"))
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_name", type=str, help="[graphsage|biobert|batchmode]", required=True)
    parser.add_argument("--style", type=str, default="keep_CCD")
    parser.add_argument("--batch", type=str)
    args = parser.parse_args()
    
    #logger = utils.get_logger(os.path.join(args.log_dir,args.log_name))
    #logger.info(args)
    
    run_RF(emb_name=args.emb_name, style=args.style, batch=args.batch)