import sqlite3 as sql
import pandas as pd
import time

class DataRetrieval:
    def __init__(self):
        self.database = "helpers/caching/ContinuousOutlineCache.db"
        self.conn = sql.connect(self.database)
        self.table_mappings = self.read_sql_table('TableMappings', self.conn)
        self.table_mappings = self.table_mappings[self.table_mappings['active'] ==1]

    def read_sql_table(self, table_name, conn):
        #NOTE: Pd.read_sql_query does not support sqllite
        return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

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
        cursor = self.conn.cursor()
        for index, row in self.table_mappings.iterrows():
            table_name = row['table']
            table_level = row['level']

            if table_level <= level:
                df = self.read_sql_table(table_name, self.conn)
                retrieved[table_name] = df
            else:
                cursor.execute(f"DELETE FROM {table_name}")
                self.conn.commit()
                df = self.read_sql_table(table_name, self.conn)
                discarded[table_name] = df

        cursor.close()
        return retrieved, discarded

    def wipe_data(self):
        for index, row in self.table_mappings.iterrows():
            table_name = row['table']
            with self.conn.cursor() as cursor:
                cursor.execute(f"DELETE FROM {table_name}")
                self.conn.commit()

    def set_data(self, set_data, only_level=-1):
        """
        Bulk inserts data from a dictionary into SQLite3 tables using pandas.

        Args:
            set_data (dict): A dictionary where keys are table names and values are pandas DataFrames.
        """
        try:
            for table_name, df in set_data.items():
                if (only_level != -1 and
                        self.table_mappings.loc[self.table_mappings['table'] == table_name, 'level'].values[0] != only_level):
                    continue
                start = time.time_ns()
                df.to_sql(table_name, self.conn, if_exists='append', index=False)
                end = time.time_ns()
                print(str((end - start) / 1e6) + " ms to do " + table_name)
        except Exception as e:
            print(f"An error occurred: {e}")


    def close_connection(self):
        self.conn.close()