# HR Job Change Prediction
```AI Supervised learning from data```
```Created in July 2021```

---

### Description
- Program using Machine Learning based on most significant data features try to predict whether person will change the job or not
- As part of learning some different classification algorithms was used and compared (including deep neural network)

### Train and test data
- Data used to train and test describe values describe people - their gender, city, city development, education level and other
- train data input shape (14368, 14)
- test data shape (4790, 14)
---

### Data details and dependencies
> As seen in the charts below, input data counts belong to each category is varies greatly and is not evenly distributed
> ![data_counts_details](img/data_counts_details.png)

> As seen in the charts below, input data does not show any clear dependencies between a specific category and target. However, there could be a relationship between the city development index and the target 
> ![data_dependencies_unsorted](img/data_dependencies_unsorted.png)
> ![data_dependencies_sorted](img/data_dependencies_sorted.png)

---

### Dimensionality reduction
To reduce data dimensionality from 14 to *n* (when n << 14) 3 algorithms was used:
* PCA
* Kernel PCA
* LDA

---
## TODO ...
1. Data dimensionality reduction description and results,
2. Using and comparing different classification algorithms,
3. Using team classification,   
4. Showing differences between ready estimators from sklearn package and
   tensorflow.keras deep neural network,
5. Summary
   * what has gone well/wrong,
   * what to change in the future programs, 
   * general thoughts

---

### Technology used
+ Python 3.9.5
    + scikit-learn
    + pandas
    + tensorflow
    + matplotlib
    + numpy

---

## License & copyright
© All rights reserved
