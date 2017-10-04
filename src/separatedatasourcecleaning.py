import pandas as pd
import re
from anonymizingdata import AnonymizingData


class SeparateDataSets(object):

    def __init__(self):
        # Importing hidden anonymous functions
        self.hidden = AnonymizingData()

    def google_plus_EDD(self):
        '''
        Creates one file for all combined transaction data from EDD and Google Analytics
        '''
        # Reading in all the separate csv files
        df1 = pd.read_csv('../data/google-data/google_dimensions_9-20-17_enddate_2014-10-31.csv',
                          low_memory=False, na_values='(not set)')
        df2 = pd.read_csv('../data/google-data/google_dimensions_9-20-17_enddate_2015-01-31.csv',
                          low_memory=False, na_values='(not set)')
        df3 = pd.read_csv('../data/google-data/google_dimensions_9-20-17_enddate_2015-04-31.csv',
                          low_memory=False, na_values='(not set)')
        df4 = pd.read_csv('../data/google-data/google_dimensions_9-20-17_enddate_2015-07-31.csv',
                          low_memory=False, na_values='(not set)')
        df5 = pd.read_csv('../data/google-data/google_dimensions_9-20-17_enddate_2015-10-31.csv',
                          low_memory=False, na_values='(not set)')
        df6 = pd.read_csv('../data/google-data/google_dimensions_9-20-17_enddate_2016-01-31.csv',
                          low_memory=False, na_values='(not set)')
        df7 = pd.read_csv('../data/google-data/google_dimensions_9-20-17_enddate_2016-04-31.csv',
                          low_memory=False, na_values='(not set)')
        df8 = pd.read_csv('../data/google-data/google_dimensions_9-20-17_enddate_2016-07-31.csv',
                          low_memory=False, na_values='(not set)')
        df9 = pd.read_csv('../data/google-data/google_dimensions_9-20-17_enddate_2016-10-31.csv',
                          low_memory=False, na_values='(not set)')
        df10 = pd.read_csv('../data/google-data/google_dimensions_9-20-17_enddate_2017-01-31.csv',
                           low_memory=False, na_values='(not set)')
        df11 = pd.read_csv('../data/google-data/google_dimensions_9-20-17_enddate_2017-04-31.csv',
                           low_memory=False, na_values='(not set)')
        df12 = pd.read_csv('../data/google-data/google_dimensions_9-20-17_enddate_2017-07-31.csv',
                           low_memory=False, na_values='(not set)')
        df13 = pd.read_csv('../data/google-data/google_dimensions_9-20-17_enddate_2017-08-31.csv',
                           low_memory=False, na_values='(not set)')

        # Concatenating files
        df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8,
                        df9, df10, df11, df12, df13], axis=0)

        # Reading in all the separate csv files for the rest of the dimensions
        df1 = pd.read_csv('../data/google-data/google_dimensions_second_run_9-20-17_enddate_2014-10-31.csv',
                          low_memory=False, na_values='(not set)')
        df2 = pd.read_csv('../data/google-data/google_dimensions_second_run_9-20-17_enddate_2015-01-31.csv',
                          low_memory=False, na_values='(not set)')
        df3 = pd.read_csv('../data/google-data/google_dimensions_second_run_9-20-17_enddate_2015-04-31.csv',
                          low_memory=False, na_values='(not set)')
        df4 = pd.read_csv('../data/google-data/google_dimensions_second_run_9-20-17_enddate_2015-07-31.csv',
                          low_memory=False, na_values='(not set)')
        df5 = pd.read_csv('../data/google-data/google_dimensions_second_run_9-20-17_enddate_2015-10-31.csv',
                          low_memory=False, na_values='(not set)')
        df6 = pd.read_csv('../data/google-data/google_dimensions_second_run_9-20-17_enddate_2016-01-31.csv',
                          low_memory=False, na_values='(not set)')
        df7 = pd.read_csv('../data/google-data/google_dimensions_second_run_9-20-17_enddate_2016-04-31.csv',
                          low_memory=False, na_values='(not set)')
        df8 = pd.read_csv('../data/google-data/google_dimensions_second_run_9-20-17_enddate_2016-07-31.csv',
                          low_memory=False, na_values='(not set)')
        df9 = pd.read_csv('../data/google-data/google_dimensions_second_run_9-20-17_enddate_2016-10-31.csv',
                          low_memory=False, na_values='(not set)')
        df10 = pd.read_csv('../data/google-data/google_dimensions_second_run_9-20-17_enddate_2017-01-31.csv',
                           low_memory=False, na_values='(not set)')
        df11 = pd.read_csv('../data/google-data/google_dimensions_second_run_9-20-17_enddate_2017-04-31.csv',
                           low_memory=False, na_values='(not set)')
        df12 = pd.read_csv('../data/google-data/google_dimensions_second_run_9-20-17_enddate_2017-07-31.csv',
                           low_memory=False, na_values='(not set)')
        df13 = pd.read_csv('../data/google-data/google_dimensions_second_run_9-20-17_enddate_2017-08-31.csv',
                           low_memory=False, na_values='(not set)')

        # Concatenating files
        dfb = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8,
                         df9, df10, df11, df12, df13], axis=0)

        # Loading metrics
        dfc = pd.read_csv('../data/google-data/google_metrics_9-20-17.csv')

        # Consolidating data to one row per transaction id using lists
        df = df.groupby('ga:transactionId').first().reset_index()
        dfb = dfb.groupby('ga:transactionId').first().reset_index()
        dfc = dfc.groupby('ga:transactionId').first().reset_index()

        # Merging all google datasets
        google_df = pd.merge(df, dfb, how='outer', on='ga:transactionId')
        google_df = pd.merge(google_df, dfc, how='outer', on='ga:transactionId')

        # Reading in EDD_df
        EDD_df = pd.read_csv('../data/EDD/payment-history-9-1-17.csv')

        # Renaming columns for datasource
        EDD_df.rename(columns=lambda x: 'EDD:' + x, inplace=True)

        # Merging transactions data with google data
        all_transactions_df = pd.merge(EDD_df, google_df, how='left',
                                       left_on='EDD:Payment ID', right_on='ga:transactionId')

        # Creating pickled dataset
        all_transactions_df.to_pickle('../data/EDD/all_transactions_google_EDD')

    def meta_data_cleaning(self):
        '''
        Cleans meta data file and creates renewal variable for use later
        '''
        # Reading CSV
        meta_df = pd.read_csv('../data/EDD/meta-data.csv', error_bad_lines=False)

        # Cutting rows that are random code
        meta_df = meta_df[[type(x) == int or x.isdigit() for x in meta_df['meta_id']]]

        # Creating list of payments that are really just renewals
        self.renewals = list(meta_df[meta_df['meta_key'] == '_edd_sl_is_renewal']['post_id'])

    def revenue_data_cleaning(self):
        '''
        Cleans revenue file and pickles the cleaned version
        '''
        # Reading CSV
        revenue_df = pd.read_csv('../data/EDD/customer-revenue.csv')

        # Renaming columns for datasource
        revenue_df.rename(columns=lambda x: (
            'revenue:' + x).strip().lower().replace(" ", "_"), inplace=True)

        # Parsing email to get domain
        lst = []
        for item in revenue_df['revenue:email']:
            lst.append(item.partition('@')[2])
        revenue_df['revenue:domain'] = lst

        # Pickling
        revenue_df.to_pickle('../data/all-datasets/revenue_df')

    def transaction_data_cleaning(self):
        '''
        Cleans transaction file and pickles the cleaned version
        '''
        # Reading pickle
        EDD_df = pd.read_pickle('../data/EDD/all_transactions_google_EDD')

        # Keeping completed statuses only
        EDD_df = EDD_df[EDD_df['EDD:Status'] == 'complete']

        # Keeping payments that are not renewals only
        EDD_df['edd:renewal_flag'] = [x in self.renewals for x in EDD_df['EDD:Payment ID']]
        EDD_df = EDD_df[EDD_df['edd:renewal_flag'] == False]
        EDD_df.drop('edd:renewal_flag', inplace=True, axis=1)

        # Dropping test payment rows
        EDD_df = EDD_df[EDD_df['EDD:Payment Method'] != 'Test Payment']

        # Create licenses variable
        licenses = []
        for item in EDD_df['EDD:Products (Verbose)']:
            try:
                # Searching for name of license within column value
                licenses.append(set(re.findall("(\S+)\s*License", item)))
            except:
                licenses.append(set())
                continue
        EDD_df['EDD:licenses'] = licenses

        # Hidden anonymizing function for different licenses
        EDD_df = self.hidden.licenses(EDD_df)

        # Change discount codes to boolean
        EDD_df['EDD:used_code'] = EDD_df['EDD:Discount Code'] != 'none'

        # Grouping by customer (using email) & summing amounts spent
        total_df = pd.DataFrame(EDD_df.groupby('EDD:Email').sum()['EDD:Amount ($)'])
        # Renaming indices to match
        total_df.rename(index=str, columns={'EDD:Amount ($)': 'EDD:total_spent'}, inplace=True)
        # Resetting email variable after transformation
        total_df['EDD:Email'] = total_df.index
        # Merging the temp df in with the original df
        EDD_df = pd.merge(EDD_df, total_df, how='left', on='EDD:Email')
        # Dropping the amount per transaction and only having the total spent per customer variable
        EDD_df.drop('EDD:Amount ($)', inplace=True, axis=1)

        # Making dummy variables
        temp_df = EDD_df[['EDD:Email', 'EDD:licenses', 'EDD:Payment Method']]
        # Making dummies for licenses and payment methods
        temp_df = pd.get_dummies(temp_df, dummy_na=True,
                                 columns=['EDD:licenses', 'EDD:Payment Method'])
        # Consolidating back to one row per customer email
        temp_df = temp_df.groupby('EDD:Email').max()
        # Resetting email variable
        temp_df['EDD:Email'] = temp_df.index
        # Merging data back in
        EDD_df = pd.merge(EDD_df, temp_df, how='left', on='EDD:Email')
        # Dropping original columns
        EDD_df.drop(['EDD:licenses', 'EDD:Payment Method'], inplace=True, axis=1)

        # Renaming columns for easier use
        EDD_df.rename(columns=lambda x: x.strip().lower().replace(" ", "_"), inplace=True)

        # Consolidating to one row per customer - taking the latest purchase they have made only
        EDD_df['edd:date'] = pd.to_datetime(EDD_df['edd:date'])
        EDD_df = EDD_df.sort_values('edd:date', ascending=False)
        EDD_df = EDD_df.groupby('edd:email').first().reset_index()

        # Pickling data
        EDD_df.to_pickle('../data/all-datasets/EDD_df')

    def intercom_data_cleaning(self):
        '''
        Cleans Intercom file (customer communication data) and pickles the cleaned version
        '''
        intercom_df = pd.read_csv('../data/intercom/intercom-9-1-17.csv')

        # Removing duplicate records
        intercom_df.sort_values('First Seen (MDT)')
        intercom_df = intercom_df.groupby('Email').first().reset_index()

        # Renaming columns for datasource
        intercom_df.rename(columns=lambda x: (
            'intercom:' + x).strip().lower().replace(" ", "_"), inplace=True)

        # Creating dummy variables from list data
        intercom_df['index'] = intercom_df.index
        lst = []
        for x in intercom_df['intercom:segment']:
            try:
                if ',' in x:
                    temp = x.split(',')
                    lst.append([x.strip('"') for x in temp])
                else:
                    lst.append([x.strip('"')])
            except:
                lst.append([])
        intercom_df['intercom:segment'] = lst
        temp_df = pd.DataFrame(intercom_df['intercom:segment'].apply(pd.Series).stack())
        temp_df.columns = ['intercom:segment']
        temp_df['values'] = 1
        temp_df['index'] = [x[0] for x in temp_df.index]
        temp_df = pd.DataFrame(temp_df.pivot(
            columns='intercom:segment', index='index', values='values')).reset_index()
        intercom_df = pd.merge(intercom_df, temp_df, left_on='index',
                               right_on='index', how='left')
        intercom_df.drop('index', inplace=True, axis=1)

        # Pickling data
        intercom_df.to_pickle('../data/all-datasets/intercom_df')

    def drip_data_cleaning(self):
        '''
        Cleans Drip file (customer emails) and pickles the cleaned version
        '''
        # Reading CSV
        drip_df = pd.read_csv('../data/drip/drip-subscribers.csv')

        # Renaming columns
        drip_df.rename(columns=lambda x: (
            'drip:' + x).strip().lower().replace(" ", "_"), inplace=True)

        # Creating dummy variables from list data
        drip_df['index'] = drip_df.index
        lst = []
        for x in drip_df['drip:campaign_names']:
            try:
                if ',' in x:
                    lst.append(x.split(','))
                else:
                    lst.append([x.strip('"')])
            except:
                lst.append([])
        drip_df['drip:campaign_names'] = lst
        # Unstacking campaign name variable to pull out each label from the list
        temp_df = pd.DataFrame(drip_df['drip:campaign_names'].apply(pd.Series).stack())
        temp_df.columns = ['drip:campaign_names']
        temp_df['values'] = 1
        temp_df['index'] = [x[0] for x in temp_df.index]
        # Pivoting a smaller dataframe to get a dummy variable for each campaign name
        temp_df = pd.DataFrame(temp_df.pivot(
            columns='drip:campaign_names', index='index', values='values')).reset_index()
        # Merging pivoted table of new dummy variables with original dataframe
        drip_df = pd.merge(drip_df, temp_df, left_on='index',
                           right_on='index', how='left')
        # Dropping index variable
        drip_df.drop('index', inplace=True, axis=1)

        # Pickling data
        drip_df.to_pickle('../data/all-datasets/drip_df')

    def hubspot_cust_data_cleaning(self):
        '''
        Cleans Hubspot customer file (marketing data) and pickles the cleaned version
        '''
        # Reading CSV
        hub_cust_df = pd.read_csv(
            '../data/hubspot/hubspot-crm-view-contacts-all-contacts-2017-09-02.csv')

        # Renaming columns
        hub_cust_df.rename(columns=lambda x: (
            'hubcust:' + x).strip().lower().replace(" ", "_"), inplace=True)

        # Pickling data
        hub_cust_df.to_pickle('../data/all-datasets/hub_cust_df')

    def hubspot_comp_data_cleaning(self):
        '''
        Cleans Hubspot company file (marketing data on customer's companies) and pickles the cleaned version
        '''
        # Reading CSV
        hub_comp_df = pd.read_csv(
            '../data/hubspot/hubspot-crm-view-companies-all-companies-2017-09-02.csv')

        # Renaming columns
        hub_comp_df.rename(columns=lambda x: (
            'hubcomp:' + x).strip().lower().replace(" ", "_"), inplace=True)

        # Pickling data
        hub_comp_df.to_pickle('../data/all-datasets/hub_comp_df')

    def turk_data_cleaning(self):
        '''
        Cleans MechanicalTurk file (experiment to have participants label category of customer website) and pickles the cleaned version
        '''
        # Reading CSV
        turk_df = pd.read_csv('../data/mechanical-turk/MechanicalTurkData.csv')

        # Filtering only for those respondents who actually got the page to work
        turk_df = turk_df[turk_df['Answer.can_access'] == 'yes']
        turk_df = turk_df[turk_df['Answer.does_it_load'] == 'yes']

        # Pulling out the only three variables that are useful
        turk_df = turk_df[['Input.website_url', 'Answer.industry', 'Answer.well-made']]

        # Pulling the website domain out of the website URL for use in joining later
        lst = []
        for item in turk_df['Input.website_url']:
            item = item.partition('//')[2]
            item = item.partition('/')[0]
            lst.append(item.replace('www.', ''))
        turk_df['turk_domain'] = lst

        # Making an average rating variable
        avg_df = pd.DataFrame(turk_df.groupby('Input.website_url').mean()['Answer.well-made'])
        avg_df['Input.website_url'] = avg_df.index
        # Merging average rating back with original dataset
        turk_df = pd.merge(turk_df, avg_df, how='left', on='Input.website_url')
        turk_df.drop('Answer.well-made_x', inplace=True, axis=1)

        # Making dummy variables for each industry
        turk_df = pd.get_dummies(turk_df, drop_first=True, dummy_na=True,
                                 columns=['Answer.industry'])

        # Renaming columns
        turk_df.rename(columns=lambda x: (
            'turk:' + x).strip().lower().replace(" ", "_"), inplace=True)

        # Pickling data
        turk_df.to_pickle('../data/all-datasets/turk_df')

    def helpscout_data_cleaning(self):
        '''
        Cleans Helpscout file (support ticket data) and creates a CSV of the cleaned version
        '''
        # Reading CSV
        help_scout_df = pd.read_pickle('../data/help-scout/helpscout_simplified')

        # Renaming columns
        help_scout_df.rename(columns=lambda x: (
            'helpscout:' + x).strip().lower().replace(" ", "_"), inplace=True)

        # Taking emails out of list form
        lst = []
        for x in help_scout_df['helpscout:emails']:
            try:
                lst.append(x[0])
            except:
                lst.append(x)
        help_scout_df['helpscout:emails'] = lst

        # Creating dummy variables from list data
        help_scout_df['index'] = help_scout_df.index
        lst = []
        for x in help_scout_df['helpscout:email_types']:
            try:
                lst.append(list(set(x)))
            except:
                lst.append([])
        help_scout_df['helpscout:email_types'] = lst
        for column in ['helpscout:phone_types', 'helpscout:email_types']:
            # Unstacking phone and email types from lists per customer
            temp_df = pd.DataFrame(help_scout_df[column].apply(pd.Series).stack())
            temp_df.columns = [column]
            # Creating a constant to become a dummy variable later
            temp_df['values'] = 1
            temp_df['index'] = [x[0] for x in temp_df.index]
            # Pivoting to get dummy variables for each phone & email type
            temp_df = pd.DataFrame(temp_df.pivot(
                columns=column, index='index', values='values')).reset_index()
            # Renaming columns of new dummy variables
            if column == 'helpscout:phone_types':
                temp_df.columns = ['index', 'home_phone', 'mobile_phone', 'work_phone']
            elif column == 'helpscout:email_types':
                temp_df.columns = ['index', 'other_type_email', 'work_email']
            help_scout_df = pd.merge(help_scout_df, temp_df, left_on='index',
                                     right_on='index', how='left')
        help_scout_df.drop('index', inplace=True, axis=1)

        # Creating CSV (to eliminate data type related pickling errors that occurred)
        help_scout_df.to_csv('../data/all-datasets/help_scout_df')

    def pickle_all(self):
        '''
        Function to easily run all above functions at once
        '''
        # self.google_plus_EDD()
        self.meta_data_cleaning()
        self.revenue_data_cleaning()
        self.transaction_data_cleaning()
        self.intercom_data_cleaning()
        self.drip_data_cleaning()
        self.hubspot_cust_data_cleaning()
        self.hubspot_comp_data_cleaning()
        self.turk_data_cleaning()
        self.helpscout_data_cleaning()


if __name__ == '__main__':
    pass
