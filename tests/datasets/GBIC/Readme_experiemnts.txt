Description of 5 biclusters

Dataset Type: 		Heterogeneous
Symbol Type: 		Custom	
List of Symbols: 	{aa,bb,cc,dd,ee,ff,gg,hh,ii,jj}
Data Type:		Real Values
Numeric Range:		[-10,+10]
Categorical Background: Uniform
Numeric Background: 	 Uniform

------------------------------------------------------
Experiment 1: Heterogeneity Level (HL)
------------------------------------------------------
Background data matrix: 	1000 x 500
Number of planted biclusters: 	5
Size of planted biclusters: 	50 x 50
Noise values on data matrix: 	0.0%

Heterogeneity Level: {0,25,50,75,100} % of categorical attributes

Random seed: [101,...,120]

hdataset_exp1_HL0_101 = hdataset_exp1_HL {value} _ randomseed 
------------------------------------------------------
Experiment 2: Number of Biclusters (NB)
------------------------------------------------------
Background data matrix: 	1000 x 500
Size of planted biclusters: 	50 x 50
Noise values on data matrix: 	0.0%
Heterogeneity Level: 		50% (50% numeric and 50% categorical)

Number of planted biclusters: {3,5,8,10}

------------------------------------------------------
Experiment 3: Size of Biclusters (SB)
------------------------------------------------------
Background data matrix: 	1000 x 500
Number of planted biclusters: 	3
Noise values on data matrix: 	0.0%
Heterogeneity Level: 		50% (50% numeric and 50% categorical)

Sizes of planted biclusters: {(25,25),(50,50),(75,75),(100,100)}, (n.rows, n.cols)

------------------------------------------------------
Experiment 4: Size of Data Matrix (SM)
------------------------------------------------------
Number of planted biclusters: 	5
Size of planted biclusters: 	50 x 50
Noise values on data matrix: 	0.0%
Heterogeneity Level: 		50% (50% numeric and 50% categorical)

Sizes of planted biclusters: {(500,250),(1000,500),(1500,750),(2000,1000)}

------------------------------------------------------
Experiment 5: Noise Level (NL)
------------------------------------------------------
Background data matrix: 	1000 x 500
Number of planted biclusters: 	3
Size of planted biclusters: 	50 x 50
Heterogeneity Level: 		50% (50% numeric and 50% categorical)

Noise values on data matrix:: {5,10,15,20} % of noise in data matrix and planted biclusters

