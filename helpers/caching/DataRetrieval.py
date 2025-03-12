import sqlite3 as sql
import pandas as pd

class DataRetrieval:
    def __init__(self):
        self.database = "helpers/caching/ContinuousOutlineCache.db"
        self.conn = sql.connect(self.database)
        self.table_mappings = pd.read_sql_table('TableMappings', self.conn)

    def retrieve_and_wipe_data(self, level):
        """
        Retrieves dataframes for tables with levels <= the given level.
        Deletes all records from tables with levels > the given level.
        Pass in 0 to wipe all data, start over

        Args:
            level (int): The maximum level for data retrieval.

        Returns:
            dict: A dictionary where keys are table names and values are DataFrames.
        """
        retrieved, discarded = {}, {}
        for index, row in self.table_mappings.iterrows():
            table_name = row['table']
            table_level = row['level']

            if table_level <= level:
                df = pd.read_sql_table(table_name, self.conn)
                retrieved[table_name] = df
            else:
                with self.conn.cursor() as cursor:
                    cursor.execute(f"DELETE FROM {table_name}")
                    self.conn.commit()
                df = pd.read_sql_table(table_name, self.conn)
                discarded[table_name] = df

        return retrieved, discarded

    def wipe_data(self):
        for index, row in self.table_mappings.iterrows():
            table_name = row['table']
            with self.conn.cursor() as cursor:
                cursor.execute(f"DELETE FROM {table_name}")
                self.conn.commit()

    def set_data(self, set_data):
        """
        Bulk inserts data from a dictionary into SQLite3 tables using pandas.

        Args:
            set_data (dict): A dictionary where keys are table names and values are pandas DataFrames.
        """
        try:
            for table_name, df in set_data.items():
                df.to_sql(table_name, self.conn, if_exists='append', index=False)
        except Exception as e:
            print(f"An error occurred: {e}")


    def close_connection(self):
        self.conn.close()