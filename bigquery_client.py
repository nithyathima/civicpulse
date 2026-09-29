import os
import pandas as pd

def fetch_spatial_clusters():
    """
    Fetches DBSCAN clusters directly from Google BigQuery.
    Gracefully falls back to simulated data if offline or credentials aren't set.
    """
    project_id = os.getenv("GCP_PROJECT_ID")
    
    if project_id:
        try:
            from google.cloud import bigquery
            bq_client = bigquery.Client(project=project_id)
            query = """
            SELECT
              cluster_id,
              COUNT(ticket_id) AS total_complaints,
              COUNTIF(dumping_time_flag = 'Night Dump Window') AS night_dump_count,
              ARRAY_AGG(DISTINCT locality LIMIT 1)[OFFSET(0)] AS locality,
              ST_Y(ST_CENTROID(ST_UNION_AGG(ST_GEOGPOINT(longitude, latitude)))) AS lat,
              ST_X(ST_CENTROID(ST_UNION_AGG(ST_GEOGPOINT(longitude, latitude)))) AS lon
            FROM
              `civicpulse_swm.grievance_logs`
            GROUP BY cluster_id
            """
            return bq_client.query(query).to_dataframe()
        except Exception:
            pass  # Fall back to simulation if credentials aren't configured yet
            
    # Mock fallback matching the BigQuery schema
    return pd.DataFrame([
        {"cluster_id": 0, "locality": "Madiwala Market", "lat": 12.9226, "lon": 77.6174, "total_complaints": 45, "night_dump_count": 31},
        {"cluster_id": 1, "locality": "Bellandur Canal Buffer", "lat": 12.9260, "lon": 77.6762, "total_complaints": 48, "night_dump_count": 36}
    ])