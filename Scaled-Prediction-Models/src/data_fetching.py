import psycopg2
import pandas as pd
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DataFetcher:
    """Handles database connections and data retrieval"""
    
    def __init__(self, db_config: Dict):
        """
        Initialize database connection
        
        Args:
            db_config: Dictionary with database connection parameters
        """
        self.db_config = db_config
        self.connection = None
        
        # Automatically connect on initialization
        self.connect()
        
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(**self.db_config)
            db_name = self.db_config.get('database', 'unknown')
            logger.info(f"Connected to database: {db_name}")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    def close(self):
        """Alias for disconnect() - for compatibility with main.py"""
        self.disconnect()
    
    def get_focal_hotels(self) -> pd.DataFrame:
        """
        Get list of focal hotels from database
        EXACT query from DB_Connect notebook
        
        Returns:
            DataFrame with global_id and masked_id columns
        """
        query = """
        SELECT DISTINCT(global_id)
        FROM public.hotels
        WHERE status != 'inactive'
          AND global_id !~ '_[12]$'
          AND global_id !~ '^(SandBox_|Sandbox_)'
        ORDER BY global_id ASC;
        """
        
        try:
            df = pd.read_sql_query(query, self.connection)
            
            # Create hotel mapping exactly as in DB_Connect
            focal_hotels = df['global_id'].tolist()
            hotel_mapping = {hotel: f"Hotel_{i+1:02d}" for i, hotel in enumerate(focal_hotels)}
            
            # Create DataFrame with mapping
            mapping_df = pd.DataFrame([
                {'masked_id': v, 'actual_id': k} 
                for k, v in hotel_mapping.items()
            ])
            
            logger.info(f"Retrieved {len(mapping_df)} focal hotels")
            return mapping_df
            
        except Exception as e:
            logger.error(f"Failed to get focal hotels: {e}")
            raise
    
    def get_focal_data(self, hotel_id: str, 
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get focal hotel pricing data
        EXACT query from DB_Connect notebook
        
        Args:
            hotel_id: Hotel global_id
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with focal hotel prices
        """
        # Default date range: 2 years historical + 1 year future
        if not start_date:
            today = datetime.now().date()
            start_date = (today - timedelta(days=730)).strftime('%Y-%m-%d')
        if not end_date:
            today = datetime.now().date()
            end_date = (today + timedelta(days=365)).strftime('%Y-%m-%d')
        
        query = f"""
        SELECT hotel_id, stay_date, room_type, price 
        FROM public.rate_amounts
        WHERE hotel_id = '{hotel_id}' 
          AND is_active = True 
          AND stay_date >= '{start_date}'
          AND stay_date <= '{end_date}'
        ORDER BY stay_date, room_type;
        """
        
        try:
            df = pd.read_sql_query(query, self.connection)
            logger.debug(f"{hotel_id}: Retrieved {len(df)} focal records")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get focal data for {hotel_id}: {e}")
            return pd.DataFrame()
    
    def get_competitor_data(self, focal_hotel_id: str,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get competitor pricing data for a focal hotel
        EXACT query from DB_Connect notebook - uses JSONB competitor extraction
        
        Args:
            focal_hotel_id: Focal hotel global_id
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with competitor prices
        """
        # Default date range
        if not start_date:
            today = datetime.now().date()
            start_date = (today - timedelta(days=730)).strftime('%Y-%m-%d')
        if not end_date:
            today = datetime.now().date()
            end_date = (today + timedelta(days=365)).strftime('%Y-%m-%d')
        
        query = f"""
        SELECT hotel_id, stay_date, price, can_check_in, min_length_of_stay 
        FROM public.ota_rates 
        WHERE hotel_id IN (
            SELECT competitor ->> 'hotel_id' AS extracted_hotel_id
            FROM hotels,
            jsonb_array_elements(hotel_settings::jsonb -> 'competitors') AS competitor
            WHERE global_id = '{focal_hotel_id}'
        ) 
        AND is_active = True 
        AND stay_date >= '{start_date}'
        AND stay_date <= '{end_date}'
        ORDER BY hotel_id, stay_date;
        """
        
        try:
            df = pd.read_sql_query(query, self.connection)
            num_competitors = df['hotel_id'].nunique() if not df.empty else 0
            logger.debug(f"{focal_hotel_id}: Retrieved {len(df)} competitor records from {num_competitors} competitors")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get competitor data for {focal_hotel_id}: {e}")
            return pd.DataFrame()
    
    def fetch_all_data(self, output_dir: str, 
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      hotel_limit: Optional[int] = None) -> Dict:
        """
        Fetch data for all hotels and save to CSV files
        
        Args:
            output_dir: Base directory (will create 'raw' subdirectory inside)
            start_date: Start date for data fetch (YYYY-MM-DD)
            end_date: End date for data fetch (YYYY-MM-DD)
            hotel_limit: Optional limit on number of hotels to process
            
        Returns:
            Dictionary with summary statistics
        """
        import os
        
        # output_dir is the base_dir, create 'raw' subdirectory
        base_dir = output_dir
        raw_dir = os.path.join(base_dir, 'raw')
        os.makedirs(raw_dir, exist_ok=True)
        
        # Get list of focal hotels
        hotel_mapping = self.get_focal_hotels()
        
        if hotel_limit:
            hotel_mapping = hotel_mapping.head(hotel_limit)
            logger.info(f"Limited to {hotel_limit} hotels for processing")
        
        # Save hotel mapping in base_dir
        mapping_file = os.path.join(base_dir, 'hotel_mapping.csv')
        hotel_mapping.to_csv(mapping_file, index=False)
        logger.info(f"Saved hotel mapping to {mapping_file}")
        
        summary = []
        
        for idx, row in hotel_mapping.iterrows():
            hotel_id = row['actual_id']  # This is the global_id
            masked_id = row['masked_id']
            
            logger.info(f"[{idx+1}/{len(hotel_mapping)}] Processing {masked_id}")
            
            # Get focal data (pass start_date and end_date)
            focal_df = self.get_focal_data(hotel_id, start_date, end_date)
            
            # Get competitor data (pass start_date and end_date)
            comp_df = self.get_competitor_data(hotel_id, start_date, end_date)
            
            # Save to CSV in raw_dir
            if not focal_df.empty:
                focal_file = os.path.join(raw_dir, f'{masked_id}_raw.csv')
                focal_df.to_csv(focal_file, index=False)
            
            if not comp_df.empty:
                comp_file = os.path.join(raw_dir, f'{masked_id}_competitors.csv')
                comp_df.to_csv(comp_file, index=False)
            
            # Summary statistics
            summary.append({
                'masked_id': masked_id,
                'actual_id': hotel_id,
                'focal_records': len(focal_df),
                'focal_date_range': f"{focal_df['stay_date'].min()} to {focal_df['stay_date'].max()}" if not focal_df.empty else 'No data',
                'num_competitors': comp_df['hotel_id'].nunique() if not comp_df.empty else 0,
                'competitor_records': len(comp_df)
            })
        
        # Save summary
        summary_df = pd.DataFrame(summary)
        summary_file = os.path.join(base_dir, 'data_fetch_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Saved data fetch summary to {summary_file}")
        
        # Count successful and failed
        successful = len([s for s in summary if s['focal_records'] > 0])
        failed = len(summary) - successful
        
        logger.info(f"Data fetching complete: {successful}/{len(hotel_mapping)} hotels successful")
        
        return {
            'total_hotels': len(hotel_mapping),
            'successful': successful,
            'failed': failed,
            'summary': summary_df
        }
    
    def fetch_complete_dataset(self, output_dir: str,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None,
                              hotel_limit: Optional[int] = None) -> Dict:
        """
        Alias for fetch_all_data() - for compatibility with main.py
        
        Args:
            output_dir: Directory to save raw data files
            start_date: Start date for data fetch (YYYY-MM-DD)
            end_date: End date for data fetch (YYYY-MM-DD)
            hotel_limit: Optional limit on number of hotels to process
            
        Returns:
            Dictionary with summary statistics
        """
        return self.fetch_all_data(output_dir, start_date, end_date, hotel_limit)