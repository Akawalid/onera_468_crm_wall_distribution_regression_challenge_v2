<h1>ONERA 468 CRM - Wall distribution regression challenge : data</h1>
<p>The pressure coefficient (Cp) and the friction coefficient along each axis (Cfx, Cfy, Cfz) at the skin of the aircraft have been selected as the outpout quantities of interest of the regression exercise. The scattered fields (size np), directly extracted from the CFD computations, have been retained. At each of the skin points, the coordinates in the Cartesian frame of reference (x, y, z) and the components of the normal vectors (nx, ny, nz) are provided. In addition to these 6 geometric parameters, the three flow conditions variables (M∞, AoA, 10^-5 pi) are given. Combining these 9 numbers, yields an input tensor X of dimensions [np*nf , 9]. The corresponding output is described by a tensor Y of dimensions [np*nf , 4], where the four columns respectively include the Cp, Cfx, Cfy and Cfz values. The described X vector provides the relevant inputs for pointwise regressors that predict the local outputs from local geometrical inputs and flow conditions. For regressors predicting complete wall distributions, relevant Xg of size [nf , 3] may be easily extracted from X (the geometric data being not needed in this case). Similarly, the corresponding wall distributions Yg of size [nf, np, 4] are obtained by a simple reordering of Y.
<br/>
The complete X and Y tensors have been split into two tensors: train and test. For this challenge, only the X_train, X_test and Y_train tensors are made available to the participants. The split between the train and test set has been done quasi-randomly: for each (M∞,pi), 4 angles of attack have been chosen randomly among the 12 to be in the test set, and the remaining 8 are part of the training set. A minor exception is made for M∞ = 0.3, M∞ = 0.82 and M∞ = 0.96 (for all pi values) where the two extreme angles of attack are forcibly included in the train set to limit extrapolation by the regressor.</p>

<p>From the "Files" tab, one can download the input data corresponding to the X_train tensor of size [np*n_train, 9], the corresponding Y_train tensor of size [np*n_train, 4] and the X_test vector of size [np*n_test, 9]. Each of them is a numpy array, with type float32 (single precision).</p>
<p>A starting kit is also available, demonstrating how to produce the array necessary to make a submission.</p>
<p><strong>All the sizes are given here:</strong>
	<ul>
		<li> np = 260,774 ; number of points on the aircraft skin</li>
		<li> nf = 468 ; total number of aerodynamic conditions present in the database </li>
		<li> n_train = 312 ; number of conditions selected in the train set </li>
		<li> n_test = 156 ; number of conditions selected in the test set </li>
	</ul>
</p>
<p><strong>Available files for download:</strong>
	<ul>
		<li> input_data
			<ul>
				<li> "describe_train_test_repartition_with_weights.csv", csv file with the following colums:
					<ul>
						<li>"Pi *1e-5": stagnation pressure</li>
						<li>"Mach": infinite Mach number</li>
						<li>"AoA": Angle of attack</li>
						<li>"std_drag": standard deviation of the drag coefficient on the last 20% iterations of the CFD computation</li>
						<li>"std_lift": standard deviation of the lift coefficient on the last 20% iterations of the CFD computation</li>
						<li>"std_mom": standard deviation of the pitching moment coefficient on the last 20% iterations of the CFD computation</li>
						<li>"confidence_weight_simple": confidence weight based on the AoA value (see the reference paper)</li>
					</ul>
				</li>
				<li> "X_train.npy" train input matrix of size [np*n_train, 9]. Numpy array, can be used with numpy.load('X_train.npy')</li>
				<li> "Ytrain.npy" train output matrix of size [np*n_train, 4]. </li>
				<li> "X_test.npy" test input matrix of size [np*n_test, 9]. </li>
			</ul>
		</li>
		<li> scoring_program
			<ul>
				<li> "eval_score_plot_fields.py": Python files containing the functions to evaluate a submission (score computation) and a matplotlib based function to plot and visualize the data</li>
				<li> "score.py": main script used by the plateform to score the submission (calling the function from  "eval_score_plot_fields.py")</li>
			</ul>
			<li> starting-kit
			<ul>
				<li> "dummy_model.py": Python script showing how to produce the "Yhat.npy" array to be submitted on the platform. Using a simple linear regression model for illustration.</li>
				<li>  "dummy_model.ipynb": same as "dummy_model.py" but as a Jupyter notebook</li>
				<li>  "eval_score_plot_fields.py": functions used to score and plot the output of a model (same as what is used in the scoring_program)</li>
			</ul>
		</li>
	</ul>
</p>