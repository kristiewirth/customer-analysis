import os
from client import Client
import pandas as pd
import time
import progressbar
from pandas.tseries.offsets import BusinessHour


class HelpScoutAPI(object):

    def __init__(self):
        # Setting up HelpScout api
        self.client = Client()
        self.client.api_key = os.environ['HELPSCOUT_KEY']

    def query_customer_data(self):
        '''
        Queries customer data using HelpScout API
        '''
        # Create empty dataframe to add customers to
        self.all_customer_df = pd.DataFrame()

        # Set index to 0 for rows in customer dataframe
        i = 1

        # Finding total pages for below iteration
        page_1 = self.client.customers(page=1)
        num_pages = page_1['pages']

        bar = progressbar.ProgressBar()
        # Iterate through all pages of customer data
        for n in bar(range(num_pages + 1)):
            all_customers = self.client.customers(page=n)
            # Pull the desired data from the list that is returned
            all_customers = all_customers['items']
            # Iterate through each separate customer dictionary on a page
            for customer in all_customers:
                # Set index for each customer's row
                all_temp_df = pd.DataFrame(customer, index=[i])
                # Join new data for a single customer to overall customer df
                self.all_customer_df = pd.concat([self.all_customer_df, all_temp_df], axis=0)
                i += 1

        # Remove duplicate rows
        self.all_customer_df = self.all_customer_df.drop_duplicates()

    def query_customer_data_detailed(self):
        '''
        Queries additional customer data using HelpScout API
        '''
        # Iterate through customer IDs and create another df from extra data from another location
        ind_cust_df = pd.DataFrame()
        errors = []
        row_errors = []
        bar = progressbar.ProgressBar()
        for cust_id in bar(self.all_customer_df.id):
            try:
                data = self.client.customer(cust_id)
            except:
                # Sleep if there are errors before retrying the API
                time.sleep(20)
                try:
                    data = self.client.customer(cust_id)
                except:
                    errors.append(cust_id)
                    continue
            # Pulling the useful data out of the json object
            data = data['item']
            # Lots of feature engineering from embedded dictionaries
            try:
                data['email_types'] = [item['location'] for item in data['emails']]
                data['emails'] = [item['value'] for item in data['emails']]
            except:
                data['email_types'] = None
                data['emails'] = None
            try:
                data['socialProfiles_links'] = [item['value'] for item in data['socialProfiles']]
                data['socialProfiles'] = [item['type'] for item in data['socialProfiles']]
            except:
                data['socialProfiles_links'] = None
                data['socialProfiles'] = None
            try:
                data['websites'] = [item['value'] for item in data['websites']]
            except:
                data['websites'] = None
            try:
                data['city'] = data['address']['city']
            except:
                data['city'] = None
            try:
                data['country'] = data['address']['country']
            except:
                data['country'] = None
            try:
                data['postalCode'] = data['address']['postalCode']
            except:
                data['postalCode'] = None
            try:
                data['state'] = data['address']['state']
            except:
                data['state'] = None
            try:
                data['chats'] = [item['type'] for item in data['chats']]
            except:
                data['chats'] = None
            try:
                data['phone_types'] = [item['location'] for item in data['phones']]
                data['phones'] = [item['value'] for item in data['phones']]
            except:
                data['phone_types'] = None
                data['phones'] = None
            del data['address']
            try:
                # Creating a temporary df and then adding to the overall dataframe
                ind_temp_df = pd.DataFrame.from_dict(dict(data), orient='index').transpose()
                ind_cust_df = pd.concat([ind_cust_df, ind_temp_df], axis=0)
            except:
                # Keeping track of any errors to append later
                row_errors.append(data)
                pass

        # Adding rows that originally caused errors
        for x in row_errors:
            ind_temp_df = pd.DataFrame.from_dict(dict(data), orient='index')
            ind_temp_df = ind_temp_df.transpose()
            ind_cust_df = pd.concat([ind_cust_df, ind_temp_df], axis=0)

        # Merging individual customer data queried with overall customer data queried
        self.df = pd.merge(self.all_customer_df, ind_cust_df, how='outer', on='id')

    def query_conversation_data(self):
        '''
        Queries customer conversation data using HelpScout API
        '''
        # Iterate through customer IDs and create another df from overall conversation data
        self.convo_df = pd.DataFrame()
        convo_errors = []
        convo_row_errors = []
        bar = progressbar.ProgressBar()
        for cust_id in bar(self.df['id']):
            try:
                # Querying for customer overall conversation data using the main mailbox number
                convo_data = self.client.conversations_for_customer_by_mailbox(37582, cust_id)
            except:
                # Sleep if there are errors before retrying the API
                time.sleep(20)
                try:
                    convo_data = self.client.conversations_for_customer_by_mailbox(37582, cust_id)
                except:
                    # Storing any errors to look at later
                    convo_errors.append(cust_id)
                    continue
            # Pulling out the useful data
            convo_data = convo_data['items']
            for item in convo_data:
                try:
                    # Merging the temporary information with the overall df
                    convo_temp_df = pd.DataFrame.from_dict(dict(item), orient='index').transpose()
                    self.convo_df = pd.concat([self.convo_df, convo_temp_df], axis=0)
                except:
                    # Storing any errors to look at later
                    convo_row_errors.append(item)
                    pass

    def query_conversation_data_detailed(self):
        '''
        Queries additional customer conversation data using HelpScout API
        '''
        # Iterate through customer IDs and create another df from detailed conversation text data
        ind_convo_df = pd.DataFrame()
        ind_convo_errors = []
        ind_convo_row_errors = []
        bar = progressbar.ProgressBar()
        for convo_id in bar(self.convo_df['id']):
            try:
                ind_convo_data = self.client.conversation(convo_id)
            except:
                # Sleeping if there are any errors before requerying the api
                time.sleep(20)
                try:
                    ind_convo_data = self.client.conversation(convo_id)
                except:
                    # Storing any erorrs to look at later
                    ind_convo_errors.append(convo_id)
                    continue
            # Pulling out the useful information
            ind_convo_data = ind_convo_data['item']
            try:
                # Merging the temporary information with the overall df
                ind_convo_temp_df = pd.DataFrame.from_dict(
                    dict(ind_convo_data), orient='index').transpose()
                ind_convo_df = pd.concat([ind_convo_df, ind_convo_temp_df], axis=0)
            except:
                # Storing any errors to look at later
                ind_convo_row_errors.append(ind_convo_data)
                pass

        # Merge both conversation dataframes
        self.convo_df_2 = pd.merge(self.convo_df, ind_convo_df, how='outer', on='id')

    def initial_data_cleaning(self):
        '''
        Cleans and combines all queried data from HelpScout API
        '''
        # Rename id column to indicate it is the conversation id (not the customer id)
        self.convo_df_2.rename(columns={'id': 'convo_id'}, inplace=True)

        # Setting up customer id column
        self.convo_df_2['id'] = [customer['id'] for customer in self.convo_df_2.customer_x]

        # Removing helpscout test rows from both datasets
        self.df = self.df[self.df['id'] != 45630411]
        self.convo_df_2 = self.convo_df_2[self.convo_df_2['id'] != 45630411]

        # Recasting id as int before grouping
        self.convo_df_2['id'] = self.convo_df_2['id'].astype(int)
        self.df['id'] = self.df['id'].astype(int)
        # Getting number of row per id (i.e., number of tickets)
        num_tickets = pd.DataFrame(self.convo_df_2.groupby('id').size())
        num_tickets.rename(index=str, columns={0: 'number_support_tickets'}, inplace=True)
        num_tickets['id'] = num_tickets.index.astype(int)
        # Creating number of support tickets variable
        self.df = self.df.merge(num_tickets, how='outer', on='id')

        # Creating hours per ticket - then total time per customer
        self.convo_df_2['closedAt'] = pd.to_datetime(self.convo_df_2['closedAt'])
        self.convo_df_2['createdAt'] = pd.to_datetime(self.convo_df_2['createdAt'])
        new_created_dates = []
        for i in range(self.convo_df_2.shape[0]):
            new_created_dates.append(BusinessHour().rollforward(self.convo_df_2['createdAt'][i]))
        self.convo_df_2['createdAt'] = new_created_dates
        self.convo_df_2['time_spent'] = self.convo_df_2['closedAt'] - self.convo_df_2['createdAt']
        self.convo_df_2['time_spent'] = self.convo_df_2['time_spent'].dt.total_seconds() / 3600.
        self.convo_df_2['time_spent'] = [x if x > 0 else 0 for x in self.convo_df_2['time_spent']]
        # Grouping by customer (using email) & summing amounts spent
        temp_df = pd.DataFrame(self.convo_df_2.groupby('id').sum()['time_spent'])
        # Resetting email variable after transformation
        temp_df['id'] = temp_df.index
        # Merging the temp df in with the original df
        self.df = pd.merge(self.df, temp_df, how='left', on='id')

        # Pickling df with all customer data (leaving support ticket data for later)
        self.df.to_pickle('helpscout_simplified')

    # def query_ticket_text(self):
        # # Working with all support tickets - Commented out due to lack of time to explore text analysis options here
        #
        # convo_df_cleaned = pd.DataFrame()
        # bar = progressbar.ProgressBar()
        # for column in bar(self.convo_df_2.columns):
        #     try:
        #         temp_df = self.convo_df_2.groupby('id', as_index=True)[column].agg(lambda x: list(x))
        #         convo_df_cleaned = pd.concat([convo_df_cleaned, temp_df], axis=1)
        #     except:
        #         continue
        #
        # # Reset column names
        # convo_df_cleaned.columns = self.convo_df_2.columns
        #
        # # Dropping duplicate id and resetting index
        # convo_df_cleaned.drop('id', inplace=True, axis=1)
        # convo_df_cleaned.reset_index()
        #
        # # Merging all customer data with all conversation data
        # df = pd.merge(df, self.convo_df_2, how='outer', on='id')
        #
        # # Exporting data to pickle
        # # df.to_pickle('final_helpscout_pickle')


if __name__ == '__main__':
    # Force pandas to display all data
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('max_info_columns', 100000)
    pd.set_option('max_seq_items', None)

    hs = HelpScoutAPI()
    hs.query_customer_data()
    hs.query_customer_data_detailed()
    hs.query_conversation_data()
    hs.query_conversation_data_detailed()
    hs.initial_data_cleaning()
