# Preparing

install R packages:
`RWeka stringr readr stringdist tm qdap SnowballC combinat`

install Python packages
`pandas numpy keras Levenshtein BeautifulSoup nltk`

download Word2vec Google News( `https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing` ) unzip and put link into word2_vec_test.py in line 89

# Generate features
```
python word2_vec_test.py
Rscript cleanData_02.R
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
Rscript generate_5fold_keys.R
python word_features.py
python extract_TF_IDF.py
```

# Create modeling datasets (needed for the R-based models)
```
Rscript create_modeling_set_08.R
Rscript create_modeling_set_08b.R
Rscript create_modeling_set_08c.R
Rscript create_modeling_set_09.R
Rscript create_modeling_set_10.R
```

# Create different models
```
python knn_bagging_5.py
python rf_bagging_5.py
```
