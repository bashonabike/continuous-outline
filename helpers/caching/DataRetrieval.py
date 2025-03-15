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
        cursor = self.conn.cursor()
        for index, row in self.table_mappings.iterrows():
            table_name = row['table']
            cursor.execute(f"DELETE FROM {table_name}")

        #Also clear the params table
        cursor.execute("DELETE FROM ParamsVals")
        self.conn.commit()
        cursor.close()

    def clear_and_set_single_table(self, table_name, df):
        cursor = self.conn.cursor()
        cursor.execute(f"DELETE FROM {table_name}")
        self.conn.commit()
        df.to_sql(table_name, self.conn, if_exists='append', index=False)

    def set_data(self, set_data, only_level=-1, min_level=0):
        """
        Bulk inserts data from a dictionary into SQLite3 tables using pandas.

        Args:
            set_data (dict): A dictionary where keys are table names and values are pandas DataFrames.
        """
        try:
            for table_name, df in set_data.items():
                if (only_level != -1 and
                        self.table_mappings.loc[self.table_mappings['table'] == table_name,
                        'level'].values[0] != only_level):
                    continue
                elif self.table_mappings.loc[self.table_mappings['table'] == table_name,
                        'level'].values[0] < min_level:
                    continue
                start = time.time_ns()
                df.to_sql(table_name, self.conn, if_exists='append', index=False)
                end = time.time_ns()
                print(str((end - start) / 1e6) + " ms to do " + table_name)
        except Exception as e:
            print(f"An error occurred: {e}")

    def level_of_update(self, new_params_df):
        """
        Logs mismatched parameters and their lowest matching param_level.

        Args:
            new_params_df (pd.DataFrame): DataFrame with 'param_name' and 'param_val'..
        """

        old_params_df = self.read_sql_table('ParamsVals', self.conn)
        old_params_df.set_index('param_name', inplace=True)
        param_levels_df = self.read_sql_table('ParamsLevels', self.conn)
        param_levels_df.set_index('param_name', inplace=True)
        new_params_df.set_index('param_name', inplace=True)

        merged_df = pd.merge(new_params_df, old_params_df, suffixes=('_new', '_old'),
                             join='left')

        mismatched_params = merged_df[merged_df['param_val_old'] != merged_df['param_val_new']]
        if mismatched_params.empty:
            return 9999

        mismatched_levels = pd.merge(mismatched_params, param_levels_df)
        level = mismatched_levels['param_level'].min()
        return level

    def get_selection_match_level(self, select_info_df):
        """
        Checks if the selection matches the old selection in the database.

        Args:
            select_info_df (pd.DataFrame): DataFrame with 'param_name' and 'param_val'..
        """
        old_select_df = self.read_sql_table('SelectionInfo', self.conn)
        old_select_df.set_index('line', inplace=True)
        merged_df = pd.merge(select_info_df, old_select_df, suffixes=('_new', '_old'),
                             join='left')
        mismatched_params = merged_df[merged_df['selection_info_old'] != merged_df['selection_info_new']]

        if mismatched_params.empty: return 9999
        elif mismatched_params.at[0, 'selection_info_new'] != select_info_df.at[0, 'selection_info_new']:
            #If only focus regions changed, dont need to rerun img proc
            return 2

        return 0

    def close_connection(self):
        self.conn.close()