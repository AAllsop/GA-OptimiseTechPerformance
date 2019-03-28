# GA-OptimiseTechPerformance
Runs the code as stand alone. i.e. not part of bench marking

If running against DW
1. Run: SQL\Generate OutstandingJobsForProcessing.sql
		Save results to Inputs\OutstandingJobsForProcessing.csv

2. Run: SQL\Generate LocationDistancesNorm.sql
		Save results to Inputs\LocationDistancesNorm.csv
		
Otherwise,
1. Run python script: Python\OptimiseTechPerformanceGA.py
		Check: working_path variable
		Best solution is from best_solution_compiled variable, where BestJobIDOrdering <> 0 and <> 1000
		
		
		

 
