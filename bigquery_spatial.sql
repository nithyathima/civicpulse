-- CivicPulse: Spatial Deficit Clustering via BigQuery ST_CLUSTERDBSCAN
-- eps_meters = 350 meters, min_points = 15 complaints

WITH RawGrievances AS (
  SELECT
    ticket_id,
    ward_id,
    locality,
    dumping_time_flag,
    ST_GEOGPOINT(longitude, latitude) AS geo_point
  FROM
    `YOUR_PROJECT_ID.civicpulse_swm.grievance_logs`
  WHERE
    status IN ('OPEN', 'INVESTIGATING')
),

ClusteredPoints AS (
  SELECT
    ticket_id,
    ward_id,
    locality,
    dumping_time_flag,
    geo_point,
    -- Core BigQuery Spatial Clustering function
    ST_CLUSTERDBSCAN(geo_point, 350, 15) OVER () AS cluster_id
  FROM
    RawGrievances
)

SELECT
  cluster_id,
  COUNT(ticket_id) AS total_complaints,
  COUNTIF(dumping_time_flag = 'Night Dump Window') AS night_dump_count,
  SAFE_DIVIDE(COUNTIF(dumping_time_flag = 'Night Dump Window'), COUNT(ticket_id)) AS night_dump_ratio,
  ST_CENTROID(ST_UNION_AGG(geo_point)) AS cluster_centroid,
  ARRAY_AGG(DISTINCT locality IGNORE NULLS LIMIT 3) AS primary_localities
FROM
  ClusteredPoints
WHERE
  cluster_id IS NOT NULL -- Exclude noise points (-1)
GROUP BY
  cluster_id
ORDER BY
  total_complaints DESC;