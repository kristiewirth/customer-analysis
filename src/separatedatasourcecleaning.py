import pandas as pd
import re
from anonymizingdata import AnonymizingData


class SeparateDataSets(object):

    def __init__(self):
        # Importing hidden anonymous functions
        self.hidden = AnonymizingData()

    def google_plus_EDD(self):
        # Reading in all the separate csv files
        df1 = pd.read_csv('google-data/google_dimensions_9-20-17_enddate_2014-10-31',
                          low_memory=False, na_values='(not set)')
        df2 = pd.read_csv('google-data/google_dimensions_9-20-17_enddate_2015-01-31',
                          low_memory=False, na_values='(not set)')
        df3 = pd.read_csv('google-data/google_dimensions_9-20-17_enddate_2015-04-31',
                          low_memory=False, na_values='(not set)')
        df4 = pd.read_csv('google-data/google_dimensions_9-20-17_enddate_2015-07-31',
                          low_memory=False, na_values='(not set)')
        df5 = pd.read_csv('google-data/google_dimensions_9-20-17_enddate_2015-10-31',
                          low_memory=False, na_values='(not set)')
        df6 = pd.read_csv('google-data/google_dimensions_9-20-17_enddate_2016-01-31',
                          low_memory=False, na_values='(not set)')
        df7 = pd.read_csv('google-data/google_dimensions_9-20-17_enddate_2016-04-31',
                          low_memory=False, na_values='(not set)')
        df8 = pd.read_csv('google-data/google_dimensions_9-20-17_enddate_2016-07-31',
                          low_memory=False, na_values='(not set)')
        df9 = pd.read_csv('google-data/google_dimensions_9-20-17_enddate_2016-10-31',
                          low_memory=False, na_values='(not set)')
        df10 = pd.read_csv('google-data/google_dimensions_9-20-17_enddate_2017-01-31',
                           low_memory=False, na_values='(not set)')
        df11 = pd.read_csv('google-data/google_dimensions_9-20-17_enddate_2017-04-31',
                           low_memory=False, na_values='(not set)')
        df12 = pd.read_csv('google-data/google_dimensions_9-20-17_enddate_2017-07-31',
                           low_memory=False, na_values='(not set)')
        df13 = pd.read_csv('google-data/google_dimensions_9-20-17_enddate_2017-08-31',
                           low_memory=False, na_values='(not set)')

        # Concatenating files
        df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8,
                        df9, df10, df11, df12, df13], axis=0)

        # Reading in all the separate csv files for the rest of the dimensions
        df1 = pd.read_csv('google-data/google_dimensions_second_run_9-20-17_enddate_2014-10-31',
                          low_memory=False, na_values='(not set)')
        df2 = pd.read_csv('google-data/google_dimensions_second_run_9-20-17_enddate_2015-01-31',
                          low_memory=False, na_values='(not set)')
        df3 = pd.read_csv('google-data/google_dimensions_second_run_9-20-17_enddate_2015-04-31',
                          low_memory=False, na_values='(not set)')
        df4 = pd.read_csv('google-data/google_dimensions_second_run_9-20-17_enddate_2015-07-31',
                          low_memory=False, na_values='(not set)')
        df5 = pd.read_csv('google-data/google_dimensions_second_run_9-20-17_enddate_2015-10-31',
                          low_memory=False, na_values='(not set)')
        df6 = pd.read_csv('google-data/google_dimensions_second_run_9-20-17_enddate_2016-01-31',
                          low_memory=False, na_values='(not set)')
        df7 = pd.read_csv('google-data/google_dimensions_second_run_9-20-17_enddate_2016-04-31',
                          low_memory=False, na_values='(not set)')
        df8 = pd.read_csv('google-data/google_dimensions_second_run_9-20-17_enddate_2016-07-31',
                          low_memory=False, na_values='(not set)')
        df9 = pd.read_csv('google-data/google_dimensions_second_run_9-20-17_enddate_2016-10-31',
                          low_memory=False, na_values='(not set)')
        df10 = pd.read_csv('google-data/google_dimensions_second_run_9-20-17_enddate_2017-01-31',
                           low_memory=False, na_values='(not set)')
        df11 = pd.read_csv('google-data/google_dimensions_second_run_9-20-17_enddate_2017-04-31',
                           low_memory=False, na_values='(not set)')
        df12 = pd.read_csv('google-data/google_dimensions_second_run_9-20-17_enddate_2017-07-31',
                           low_memory=False, na_values='(not set)')
        df13 = pd.read_csv('google-data/google_dimensions_second_run_9-20-17_enddate_2017-08-31',
                           low_memory=False, na_values='(not set)')

        # Concatenating files
        dfb = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8,
                         df9, df10, df11, df12, df13], axis=0)

        # Consolidating data to one row per transaction id using lists
        df = df.groupby('ga:transactionId').first().reset_index()
        dfb = dfb.groupby('ga:transactionId').first().reset_index()

        # Merging all google datasets
        google_df = pd.merge(df, dfb, how='outer', on='ga:transactionId')

        # Reading in EDD_df
        EDD_df = pd.read_csv('../EDD/payment-history-9-1-17.csv')

        # Renaming columns for datasource
        EDD_df.rename(columns=lambda x: 'EDD:' + x, inplace=True)

        # Merging transactions data with google data
        all_transactions_df = pd.merge(EDD_df, google_df, how='left',
                                       left_on='EDD:Payment ID', right_on='ga:transactionId')

        # Creating pickled dataset
        all_transactions_df.to_pickle('all_transactions_google_EDD')

    def meta_data_cleaning(self):
        # Reading CSV
        meta_df = pd.read_csv('../data/EDD/meta-data.csv', error_bad_lines=False)

        # Cutting rows that are random code
        meta_df = meta_df[[type(x) == int or x.isdigit() for x in meta_df['meta_id']]]

        # Creating list of payments that are really just renewals
        self.renewals = list(meta_df[meta_df['meta_key'] == '_edd_sl_is_renewal']['post_id'])

    def revenue_data_cleaning(self):
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
        # Reading CSV
        EDD_df = pd.read_pickle('../data/google-analytics/all_transactions_google_EDD')

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

        # Consolidating to one row per customer
        EDD_df = EDD_df.groupby('edd:email').first().reset_index()

        # Pickling data
        EDD_df.to_pickle('../data/all-datasets/EDD_df')

    def intercom_data_cleaning(self):
        intercom_df = pd.read_csv('../data/intercom/intercom-9-1-17.csv')

        # Removing duplicate records
        intercom_df.sort_values('First Seen (MDT)')
        intercom_df = intercom_df.groupby('Email').first().reset_index()

        # Renaming columns for datasource
        intercom_df.rename(columns=lambda x: (
            'intercom:' + x).strip().lower().replace(" ", "_"), inplace=True)

        # Pickling data
        intercom_df.to_pickle('../data/all-datasets/intercom_df')

    def drip_data_cleaning(self):
        # Reading CSV
        drip_df = pd.read_csv('../data/drip/drip-subscribers.csv')

        # Renaming columns
        drip_df.rename(columns=lambda x: (
            'drip:' + x).strip().lower().replace(" ", "_"), inplace=True)

        # Pickling data
        drip_df.to_pickle('../data/all-datasets/drip_df')

    def hubspot_cust_data_cleaning(self):
        # Reading CSV
        hub_cust_df = pd.read_csv(
            '../data/hubspot/hubspot-crm-view-contacts-all-contacts-2017-09-02.csv')

        # Renaming columns
        hub_cust_df.rename(columns=lambda x: (
            'hubcust:' + x).strip().lower().replace(" ", "_"), inplace=True)

        # Pickling data
        hub_cust_df.to_pickle('../data/all-datasets/hub_cust_df')

    def hubspot_comp_data_cleaning(self):
        # Reading CSV
        hub_comp_df = pd.read_csv(
            '../data/hubspot/hubspot-crm-view-companies-all-companies-2017-09-02.csv')

        # Renaming columns
        hub_comp_df.rename(columns=lambda x: (
            'hubcomp:' + x).strip().lower().replace(" ", "_"), inplace=True)

        # Pickling data
        hub_comp_df.to_pickle('../data/all-datasets/hub_comp_df')

    def turk_data_cleaning(self):
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
        # Fixing unicode problem
        help_scout_df['helpscout:emails'] = help_scout_df['helpscout:emails'].astype(str)

        # Pickling data
        help_scout_df.to_pickle('../data/all-datasets/help_scout_df')

    def pickle_all(self):
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
