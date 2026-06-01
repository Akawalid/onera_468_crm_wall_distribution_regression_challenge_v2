<h1>ONERA 468 CRM - Wall distribution regression challenge: Evaluation</h1>
<p>The problem is a regression problem. 
    <br />You are given for training a data matrix X_train of dimension [np*n_train, 9] and an array Y_train of dimension [np*n_train, 4]. You must train a model which predicts the 4 considered aerodynamic coefficients [Cp, Cfx, Cfy, Cfz] corresponding to the geometric points and aerodynamic conditions of the test matrix  X_test of dimension [np*n_test, 9].<br />
   Your submission must be a numpy array called "Yhat.npy" of size [np*n_test, 4]. You must submit this matrix inside a zip file, for instance "my_submission_1.zip". The points in the "Yhat.npy" matrix must be in the same orders as in the X_test matrix.<br />
    <br />
    One phase is currently setup:
</p>
<ul>
    <li><strong>Phase 1:</strong>  Regression on the unstructured data.</li>
</ul>
<p>The submissions are evaluated using the R2 (coefficient of determination) and the worst relative mean absolute error (wrMAE) metrics. These metrics involve a weight which describes the confidence one can have in the CFD computation at each aerodynamic conditions. These weights can be found in the file "describe_train_test_repartition_with_weights.csv" using the column "confidence_weight_simple". A detailed description of these metrics is presented in the <a href="http://www.arxiv.org/abs/2505.06265">associated paper</a>. The R2 and wrMAE are computed for each components of the Y matrix (ie Cp, Cfx, Cfy, Cfz). A mean value across the 4 quantity of interest is then computed. Finally, a score aggregating the two metrics is computed with the formula: score = 5*R2_mean + 5*(1-wrMAE_mean). The code computing the metrics is available by downloading the "scoring_program" file, with the two scripts "score.py" and "eval_score_plot_fields.py".</p>