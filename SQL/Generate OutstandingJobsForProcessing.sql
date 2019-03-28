drop table dbo.OutstandingJobsForProcessing
EXEC Reporting.DropTempTablesFromSession

--initial starting location co-ordinates
DECLARE @ResourceStartLongitude DECIMAL(18,10) = 145.284036
DECLARE @ResourceStartLatitude DECIMAL(18,10) = -37.791168

--get all outstanding jobs
--	with no response
SELECT
	'Response' JobType
   ,ffe.ResourceKey
   ,ffe.faultid
   ,ffe.CalloutDate CalloutDateTime
   ,ff.StoreKey
   ,s.Longitude
   ,s.Latitude
   ,ffe.Priority
   ,ffe.ResponseTargetDate KPITargetDate
   ,1 EstimatedJobDuration
INTO #OutstandingJobs
FROM Fault.FactFaultEvent ffe
INNER JOIN Fault.FactFaults ff
	ON ffe.faultid = ff.faultid
INNER JOIN Common.Stores s
	ON ff.StoreKey = s.StoreKey
WHERE ffe.Fixed = 0
AND ffe.Completed = 0
AND ff.Cancelled = 0
AND ffe.Cancelled = 0
AND ff.AlarmDateKey > 20180101
AND ff.KPI = 1
AND ff.FirstOnSiteDate IS NULL

--	with no repair
INSERT INTO #OutstandingJobs
	SELECT
		'Repair' JobType
	   ,ffe.ResourceKey
	   ,ffe.faultid
	   ,ffe.CalloutDate CalloutDateTime
	   ,ff.StoreKey
	   ,s.Longitude
	   ,s.Latitude
	   ,ffe.Priority
	   ,ffe.RepairTargetDate KPITargetDate
	   ,1 EstimatedJobDuration
	FROM Fault.FactFaultEvent ffe
	INNER JOIN Fault.FactFaults ff
		ON ffe.faultid = ff.faultid
	INNER JOIN Common.Stores s
		ON ff.StoreKey = s.StoreKey
	WHERE ffe.Fixed = 0
	AND ffe.Completed = 0
	AND ff.Cancelled = 0
	AND ffe.Cancelled = 0
	AND ff.AlarmDateKey > 20180101
	AND ff.KPI = 1
	AND NOT EXISTS (SELECT
			''
		FROM #OutstandingJobs oj
		WHERE ffe.faultid = oj.faultid
		AND ffe.CalloutDate = oj.CalloutDateTime)


SELECT
	ROW_NUMBER() OVER (PARTITION BY oj.ResourceKey ORDER BY oj.JobType) GeneID
   ,*
   ,DATEDIFF(HOUR,oj.CalloutDateTime,oj.KPITargetDate) AS HoursToTarget
INTO #OutstandingJobsWithCL
FROM #OutstandingJobs oj
UNION
SELECT
	0 GeneID
   ,'Current Location' JobType
   ,ResourceKey
   ,'0' FaultID
   ,'2099-12-31' CalloutDateTime
   ,'0' StoreKey
   ,AVG(Longitude) Longitude
   ,AVG(Latitude) Latitude
   ,'Low' Priority
   ,'2099-12-31' KPITargetDate
   ,0 EstimatedJobDuration
   ,0 HoursToTarget
FROM #OutstandingJobs oj
GROUP BY ResourceKey
UNION
SELECT
	-1 GeneID
   ,'Dummy Job' JobType
   ,ResourceKey
   ,'0' FaultID
   ,'2099-12-31' CalloutDateTime
   ,'0' StoreKey
   ,0 Longitude
   ,0 Latitude
   ,'Low' Priority
   ,'2099-12-31' KPITargetDate
   ,0 EstimatedJobDuration
   ,0 HoursToTarget
FROM #OutstandingJobs oj
GROUP BY ResourceKey


SELECT
	GeneID
   ,JobType
   ,o.ResourceKey
   ,faultid
   ,CalloutDateTime
   ,StoreKey
   ,Longitude
   ,Latitude
   ,Priority
   ,KPITargetDate
   ,EstimatedJobDuration
   ,HoursToTarget
INTO dbo.OutstandingJobsForProcessing
FROM #OutstandingJobsWithCL o
Inner Join Fault.DimResources dr
ON o.ResourceKey = dr.ResourceKey
WHERE dr.SubType IN ('RST','RHVACT')





