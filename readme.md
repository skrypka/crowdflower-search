## Preparing

install R packages:
`RWeka stringr readr stringdist tm qdap SnowballC combinat e1071 xgboost h2o`

install Python packages
`pandas numpy keras Levenshtein BeautifulSoup nltk`

download Word2vec Google News( `https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing` ) unzip and put link into word2_vec_test.py in line 89

## Data cleaning
```
Rscript cleanData_02.R
```

## Generating features
```
python word2_vec_test.py
Rscript create_Okapi.R
Rscript alt_query.R
Rscript createFeatures07_Ngram_match.R
Rscript extractProductName_NEW.R
python auto_correct.py
python dict_for_clean.py
python AddProps.py
python AddProps2.py
python AddProps3.py
python RelevanceGroup.py
python word_features.py
python extract_TF_IDF.py
```

## Creating modeling datasets (needed for the R-based models)
```
Rscript create_modeling_set_08.R
Rscript create_modeling_set_08b.R
Rscript create_modeling_set_08c.R
Rscript create_modeling_set_09.R
Rscript create_modeling_set_10.R
```

## Generating 5-fold cross validation keys
```
Rscript generate_5fold_keys.R
```

## Creating single-layer models
```
python knn_bagging_5.py
python rf_bagging_5.py
python ann10b_ver2.py
python ann_250_tfidf.py
python ann_alt.py
python ann_wm_c1r2.py
python ann10b.py
python ann_1234_7_ver2.py
python ann_alt_ngram_wm.py
python ann10b_noamazon.py
python ann_tfidf.py
Rscript Xgboost_training_v10.R
Rscript Xgboost_training_v10b.R
Rscript Xgboost_training_v10c.R
Rscript Xgboost_training_v11.R
Rscript Xgboost_training_v12.R
Rscript svm_model_v20150703.R
Rscript h2o_training_v20150616.R
python svm_alejandro.py
```

## Creating second-layer models (stacking)
```
Rscript masterset_v04.R
python ensemblenn.py
```

## Creating final ensemble
```
Rscript create_ensemble.R
```
