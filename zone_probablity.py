# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:25:59 2017

@author: KAMMO, TANNY, AJ, SUSHI, GOWTHAMI
"""

import pandas as pd
import numpy as np


def get_hit_rate_stats(infile):
    # pull the labels for a given patient
    df = pd.read_csv(infile)

    # Separate the zone and patient id into a df
    df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
    df = df[['Subject', 'Zone', 'Probability']]

    # make a df of the sums and counts by zone and calculate hit rate per zone, then sort high to low
    df_summary = df.groupby('Zone')['Probability'].agg(['sum','count'])
    df_summary['Zone'] = df_summary.index
    df_summary['pct'] = df_summary['sum'] / df_summary['count']
    df_summary.sort_values('pct', axis=0, ascending= False, inplace=True)
    
    return df_summary

def print_hit_rate_stats(df_summary):
    # print the table of values readbly
    print ('{:6s}   {:>4s}   {:6s}'.format('Zone', 'Hits', 'Pct %'))
    print ('------   ----- ----------')
    for zone in df_summary.iterrows():
        print ('{:6s}   {:>4d}   {:>6.3f}%'.format(zone[0], np.int16(zone[1]['sum']), zone[1]['pct']*100))
    print ('------   ----- ----------')
    print ('{:6s}   {:>4d}   {:6.3f}%'.format('Total ', np.int16(df_summary['sum'].sum(axis=0)), 
                                             ( df_summary['sum'].sum(axis=0) / df_summary['count'].sum(axis=0))*100))
    
def get_subject_labels(infile, subject_id):

    # read labels into a dataframe
    df = pd.read_csv(infile)

    # Separate the zone and subject id into a df
    df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
    df = df[['Subject', 'Zone', 'Probability']]
    threat_list = df.loc[df['Subject'] == subject_id]
    
    return threat_list

def get_subject_zone_label(zone_num, df):
    
    # Dict to convert a 0 based threat zone index to the text we need to look up the label
    zone_index = {0: 'Zone1', 1: 'Zone2', 2: 'Zone3', 3: 'Zone4', 4: 'Zone5', 5: 'Zone6', 
                  6: 'Zone7', 7: 'Zone8', 8: 'Zone9', 9: 'Zone10', 10: 'Zone11', 11: 'Zone12', 
                  12: 'Zone13', 13: 'Zone14', 14: 'Zone15', 15: 'Zone16',
                  16: 'Zone17'
                 }
    # get the text key from the dictionary
    key = zone_index.get(zone_num)
    
    # select the probability value and make the label
    print("kps ",df.loc[df['Zone'] == key]['Probability'].values)
    if df.loc[df['Zone'] == key]['Probability'].values[0] == 1:
        # threat present
        return [0,1]
    else:
        #no threat present
        return [1,0]