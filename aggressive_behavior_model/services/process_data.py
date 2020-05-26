import pandas as pd
from datetime import datetime, timedelta
import pickle
import pyproj
import open_cp

siedco_dict = {
                'date': 'FECHA_HECHO',
                'latitude': 'LATITUD_Y',
                'longitude': 'LONGITUD_X',
                'time': 'HORA_HECHO',
                'time_stamp':''
               }

nuse_dict = {
                'date': 'FECHA',
                'latitude': 'LATITUD',
                'longitude': 'LONGITUD',
                'time': 'HORA',
                'time_stamp':''
            }

rnmc_dict = {
                'date': 'FECHA',
                'latitude': 'LATITUD',
                'longitude': 'LONGITUD',
                'time': 'HORA',
                'time_stamp':''
            }

DATASET_DICT = {'SIEDCO':siedco_dict, 'NUSE':nuse_dict, 'RNMC':rnmc_dict}

class ProcessData:

    def __init__(self, dataset_name, dataset_path):
        """
            :dataset_name: string (SIEDCO, NUSE or RNMC)
            :dataset_path: string with dataset file path location (file must be .csv)
            :dataset_dict: dictionary of dataset fields names as defined in process_data.py
        """
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.dataset_dict = self.set_dictionary()

    def add_timestamp(self, df):
        dataset_dict = self.dataset_dict
        if dataset_dict['time_stamp'] == '':
            df = self.format_date_fields(df)
            df['TIME_STAMP']=pd.to_datetime(df[dataset_dict['date']]+ ' '+df[dataset_dict['time']])
            self.dataset_dict['time_stamp'] = 'TIME_STAMP'
        return df

    def filter_by_date(df, dataset_dict, initial_date, final_date):
        time_stamp_field = dataset_dict['time_stamp']
        if isinstance(initial_date,str) and isinstance(final_date,str):
            initial_date = datetime.strptime(initial_date,'%Y-%m-%d')
            real_final_date = datetime.strptime(final_date,'%Y-%m-%d')+timedelta(days=1)
        elif isinstance(initial_date,datetime) and isinstance(final_date,datetime):
            real_final_date = final_date+timedelta(days=1)
        else:
            raise TypeError("initial_date and final_date formats don't match.")

        df_filtered = df[(df[time_stamp_field] > initial_date) & (df[time_stamp_field] < real_final_date)]
        if len(df_filtered) == 0:
            raise ValueError("Empty filter result, check dates.")
        return df_filtered

    def filter_by_field(df, name_field, value):
        if name_field == '':
            return df
        try:
            df_filtered = df[df[name_field] == value]
        except KeyError:
            raise ValueError("Field doesn't exist.")
        if len(df_filtered) == 0:
            raise ValueError("Empty filter result, check filter value.")
        return df_filtered

    def format_date_fields(self, df):
        dataset_dict = self.dataset_dict
        #format date field to standard
        df[dataset_dict['date']] = pd.to_datetime(df[dataset_dict['date']]).dt.strftime('%Y-%m-%d')
        #format time field to standard
        time_series = df[dataset_dict['time']]
        match_format_flag = False
        regex_standard_hour = r'^\d{2}:\d{2}:\d{2}$'
        regex_short_hour = r'^\d{2}:\d{2}$'
        regex_int_hour = r'^\d{1,4}$'
        if time_series.astype(str).str.match(regex_standard_hour).all():
            match_format_flag = True
            df.loc[df[dataset_dict['time']] == '00:00:00',dataset_dict['time']] = '00:00:01'
        if time_series.astype(str).str.match(regex_short_hour).all():
            match_format_flag = True
            df.loc[df[dataset_dict['time']] == '00:00',dataset_dict['time']] = '00:01'
        if time_series.astype(str).str.match(regex_int_hour).all():
            match_format_flag = True
            df[dataset_dict['time']] = pd.to_datetime(df[dataset_dict['time']].astype(str).str.rjust(4,'0'),format= '%H%M').dt.strftime('%H:%M')
            df.loc[df[dataset_dict['time']] == '00:00',dataset_dict['time']] = '00:01'

        if match_format_flag == False:
            return "Date/time format error, check type and structure on date/time columns"
        else:
            return df

    def get_formated_df(self):
        df = self.load_dataset_from_csv()
        df = self.add_timestamp(df)
        return df

    def get_time_space_points(df_subset, dataset_dict):
        timestamps = df_subset[dataset_dict['time_stamp']]
        xcoords, ycoords = (df_subset[dataset_dict['longitude']].values,df_subset[dataset_dict['latitude']].values)
        proj = pyproj.Proj(init="EPSG:3116")
        xcoords, ycoords = proj(xcoords,ycoords)
        time_space_points = open_cp.TimedPoints.from_coords(timestamps, xcoords, ycoords)
        region = ProcessData.set_data_region(xcoords,ycoords)
        return (time_space_points, region)

    def load_dataset_from_csv(self):
        #db is already filtered with the type of event we want to model
        current_path = self.dataset_path
        dataset_dict = self.dataset_dict
        try:
            df = pd.read_csv(current_path)
        except FileNotFoundError as err:
            raise FileNotFoundError("Error loading csv, file not found.")
        return df

    def save_element(path, file_name, element):
        outfile = open(path+file_name+'.pkl','wb')
        pickle.dump(element, outfile)
        outfile.close()

    def select_data(df, dataset_dict, filter, dates):
        df_filtered = ProcessData.filter_by_field(df, filter['field'], filter['value'])
        df_train = ProcessData.filter_by_date(df_filtered, dataset_dict, dates['initial'], dates['final'])
        return df_train

    def set_data_region(xcoords,ycoords):
        # Extracts the bounding regions
        maxx = max(xcoords)
        minx = min(xcoords)
        maxy = max(ycoords)
        miny = min(ycoords)
        region = open_cp.RectangularRegion(xmin=minx, xmax=maxx, ymin=miny, ymax=maxy)
        return region

    def set_dictionary(self):
        return DATASET_DICT[self.dataset_name].copy()
