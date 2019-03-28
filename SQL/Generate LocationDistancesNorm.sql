EXEC Reporting.DropTempTablesFromSession
DROP TABLE dbo.LocationDistances
DROP TABLE dbo.LocationDistancesNorm

--get distances between locations
SELECT DISTINCT
	oj.StoreKey LocationKey
   ,oj.Longitude
   ,oj.Latitude
   ,oj.ResourceKey
INTO #Locations
FROM dbo.OutstandingJobsForProcessing oj
WHERE oj.GeneID > 0

DELETE FROM #Locations
WHERE Longitude IS NULL

--insert a starting location for each tech
INSERT INTO #Locations
(LocationKey,Longitude,Latitude,ResourceKey)
	SELECT
		0
	   ,AVG(Longitude)
	   ,AVG(Latitude)
	   ,ResourceKey
	FROM #Locations 
	GROUP BY ResourceKey

SELECT
	l.LocationKey LocationKeyStartPoint
   ,l1.LocationKey LocationKeyEndPoint
   ,GEOGRAPHY::Point(l.Latitude,l.Longitude,4326).STDistance(GEOGRAPHY::Point(l1.Latitude,l1.Longitude,4326)) AS MetersBetweenPoints
   ,CAST(l.LocationKey AS VARCHAR) + '|' + CAST(l1.LocationKey AS VARCHAR) LocationLookupKey
   ,l.ResourceKey
INTO dbo.LocationDistances
FROM #Locations l
CROSS JOIN #Locations l1
WHERE l.ResourceKey = l1.ResourceKey

---loop through and create for each resource
CREATE TABLE dbo.LocationDistancesNorm (
	LocationKeyStartPoint INT
   ,LocationKeyEndPoint INT
   ,MetersBetweenPoints DECIMAL(10,2)
   ,MetersBetweenPoints_norm DECIMAL(10,2)
   ,LocationLookupKey VARCHAR(20)
   ,ResourceKey INT
)

SELECT DISTINCT
	ResourceKey
   ,0 Processed
INTO #ResourcesToProcess
FROM #Locations

DECLARE @ResourceKey AS INT = 0
WHILE @ResourceKey IS NOT NULL
BEGIN

SET @ResourceKey = (SELECT
		MIN(ResourceKey)
	FROM #ResourcesToProcess rtp
	WHERE Processed = 0)

DECLARE @Min_Dist DECIMAL(18,4) = (SELECT
		MIN(MetersBetweenPoints)
	FROM dbo.LocationDistances
	WHERE MetersBetweenPoints > 0
	AND ResourceKey = @ResourceKey)
DECLARE @inv_contraction_rate DECIMAL(18,4) = .03

;
WITH a
AS
(SELECT
		LocationKeyStartPoint
	   ,LocationKeyEndPoint
	   ,MetersBetweenPoints
	   ,(((MetersBetweenPoints / @Min_Dist - 1) * @inv_contraction_rate) * @Min_Dist) + @Min_Dist AS NewDistance
	   ,LocationLookupKey
	   ,ResourceKey
	FROM dbo.LocationDistances
	WHERE ResourceKey = @ResourceKey)

INSERT INTO dbo.LocationDistancesNorm
	SELECT
		LocationKeyStartPoint
	   ,LocationKeyEndPoint
	   ,MetersBetweenPoints
	   ,NewDistance / n.MaxNewDistance MetersBetweenPoints_norm
	   ,LocationLookupKey
	   ,ResourceKey
	FROM a
	CROSS JOIN (SELECT
			MAX(NewDistance) MaxNewDistance
		FROM a) n
	WHERE a.ResourceKey = @ResourceKey

UPDATE dbo.LocationDistancesNorm
SET MetersBetweenPoints = 0
   ,MetersBetweenPoints_norm = 0
WHERE LocationKeyStartPoint = LocationKeyEndPoint
AND ResourceKey = @ResourceKey

UPDATE #ResourcesToProcess
SET Processed = 1
WHERE ResourceKey = @ResourceKey

END

