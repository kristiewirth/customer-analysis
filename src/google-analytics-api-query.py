from apiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import progressbar


def get_service(api_name, api_version, scope, key_file_location):
    '''
    Default Google API code.

    INPUT:
    api_name: The name of the api to connect to.
    api_version: The api version to connect to.
    scope: A list auth scopes to authorize for the application.
    key_file_location: The path to a valid service account JSON key file.

    OUTPUT:
    A service that is connected to the specified API.
    '''
    credentials = ServiceAccountCredentials.from_json_keyfile_name(key_file_location, scopes=scope)
    service = build(api_name, api_version, credentials=credentials)
    return service


if __name__ == '__main__':
    # Indicate relevant URL & location of key file
    scope = ['https://www.googleapis.com/auth/analytics.readonly']
    key_file_location = '../data/google-analytics/key_info.json'

    # Call default function to connect
    service = get_service('analytics', 'v3', scope, key_file_location)

    # All metrics
    metrics = ['ga:users', 'ga:newUsers', 'ga:percentNewSessions', 'ga:sessionsPerUser', 'ga:sessions', 'ga:bounces', 'ga:bounceRate', 'ga:sessionDuration',
               'ga:avgSessionDuration', 'ga:uniqueDimensionCombinations', 'ga:hits', 'ga:organicSearches', 'ga:goalXXStarts', 'ga:goalStartsAll', 'ga:goalXXCompletions',
               'ga:goalCompletionsAll', 'ga:goalXXValue', 'ga:goalValueAll', 'ga:goalValuePerSession', 'ga:goalXXConversionRate', 'ga:goalConversionRateAll',
               'ga:goalXXAbandons', 'ga:goalAbandonsAll', 'ga:goalXXAbandonRate', 'ga:goalAbandonRateAll', 'ga:pageValue', 'ga:entrances', 'ga:entranceRate',
               'ga:pageviews', 'ga:pageviewsPerSession', 'ga:uniquePageviews', 'ga:timeOnPage', 'ga:avgTimeOnPage', 'ga:exits', 'ga:exitRate', 'ga:searchResultViews',
               'ga:searchUniques', 'ga:avgSearchResultViews', 'ga:searchSessions', 'ga:percentSessionsWithSearch', 'ga:searchDepth', 'ga:avgSearchDepth',
               'ga:searchRefinements', 'ga:percentSearchRefinements', 'ga:searchDuration', 'ga:avgSearchDuration', 'ga:searchExits', 'ga:searchExitRate',
               'ga:searchGoalXXConversionRate', 'ga:searchGoalConversionRateAll', 'ga:goalValueAllPerSearch', 'ga:screenviews', 'ga:screenviewsPerSession',
               'ga:timeOnScreen', 'ga:avgScreenviewDuration', 'ga:totalEvents', 'ga:uniqueEvents', 'ga:eventValue', 'ga:avgEventValue', 'ga:sessionsWithEvent',
               'ga:eventsPerSessionWithEvent', 'ga:transactions', 'ga:transactionsPerSession', 'ga:transactionRevenue', 'ga:revenuePerTransaction',
               'ga:transactionRevenuePerSession', 'ga:transactionShipping', 'ga:transactionTax', 'ga:totalValue', 'ga:itemQuantity', 'ga:uniquePurchases',
               'ga:revenuePerItem', 'ga:itemRevenue', 'ga:itemsPerPurchase', 'ga:buyToDetailRate', 'ga:cartToDetailRate', 'ga:internalPromotionCTR',
               'ga:internalPromotionClicks', 'ga:internalPromotionViews', 'ga:productAddsToCart', 'ga:productCheckouts', 'ga:productDetailViews', 'ga:productListCTR',
               'ga:productListClicks', 'ga:productListViews', 'ga:productRefundAmount', 'ga:productRefunds', 'ga:productRemovesFromCart', 'ga:productRevenuePerPurchase',
               'ga:quantityAddedToCart', 'ga:quantityCheckedOut', 'ga:quantityRefunded', 'ga:quantityRemovedFromCart', 'ga:refundAmount', 'ga:revenuePerUser',
               'ga:totalRefunds', 'ga:transactionsPerUser', 'ga:socialInteractions', 'ga:uniqueSocialInteractions', 'ga:socialInteractionsPerSession', 'ga:metricXX',
               'ga:calcMetric_', 'ga:dcmFloodlightQuantity', 'ga:dcmFloodlightRevenue', 'ga:adsenseRevenue', 'ga:adsenseAdUnitsViewed', 'ga:adsenseAdsViewed',
               'ga:adsenseAdsClicks', 'ga:adsensePageImpressions', 'ga:adsenseCTR', 'ga:adsenseECPM', 'ga:adsenseExits', 'ga:adsenseViewableImpressionPercent',
               'ga:adsenseCoverage', 'ga:adxImpressions', 'ga:adxCoverage', 'ga:adxMonetizedPageviews', 'ga:adxImpressionsPerSession', 'ga:adxViewableImpressionsPercent',
               'ga:adxClicks', 'ga:adxCTR', 'ga:adxRevenue', 'ga:adxRevenuePer1000Sessions', 'ga:adxECPM', 'ga:dfpImpressions', 'ga:dfpCoverage', 'ga:dfpMonetizedPageviews',
               'ga:dfpImpressionsPerSession', 'ga:dfpViewableImpressionsPercent', 'ga:dfpClicks', 'ga:dfpCTR', 'ga:dfpRevenue', 'ga:dfpRevenuePer1000Sessions', 'ga:dfpECPM',
               'ga:backfillImpressions', 'ga:backfillCoverage', 'ga:backfillMonetizedPageviews', 'ga:backfillImpressionsPerSession', 'ga:backfillViewableImpressionsPercent',
               'ga:backfillClicks', 'ga:backfillCTR', 'ga:backfillRevenue', 'ga:backfillRevenuePer1000Sessions', 'ga:backfillECPM']

    # All dimensions
    dimensions = ['ga:userType', 'ga:sessionCount', 'ga:daysSinceLastSession', 'ga:userDefinedValue', 'ga:userBucket', 'ga:sessionDurationBucket', 'ga:referralPath',
                  'ga:fullReferrer', 'ga:campaign', 'ga:source', 'ga:medium', 'ga:sourceMedium', 'ga:keyword', 'ga:socialNetwork', 'ga:campaignCode', 'ga:adGroup',
                  'ga:adDistributionNetwork', 'ga:adMatchType', 'ga:adKeywordMatchType', 'ga:adMatchedQuery', 'ga:adPlacementDomain', 'ga:adPlacementUrl', 'ga:adFormat',
                  'ga:adTargetingType', 'ga:adTargetingOption', 'ga:adDisplayUrl', 'ga:adDestinationUrl', 'ga:adwordsCustomerID', 'ga:adwordsCampaignID', 'ga:adwordsAdGroupID',
                  'ga:adwordsCreativeID', 'ga:adwordsCriteriaID', 'ga:adQueryWordCount', 'ga:browser', 'ga:browserVersion', 'ga:operatingSystem', 'ga:operatingSystemVersion',
                  'ga:mobileDeviceBranding', 'ga:mobileDeviceModel', 'ga:mobileInputSelector', 'ga:mobileDeviceInfo', 'ga:mobileDeviceMarketingName', 'ga:deviceCategory',
                  'ga:browserSize', 'ga:dataSource', 'ga:continent', 'ga:subContinent', 'ga:country', 'ga:region', 'ga:metro', 'ga:city', 'ga:latitude', 'ga:longitude',
                  'ga:networkDomain', 'ga:networkLocation', 'ga:cityId', 'ga:continentId', 'ga:countryIsoCode', 'ga:metroId', 'ga:regionId', 'ga:regionIsoCode',
                  'ga:subContinentCode', 'ga:screenColors', 'ga:sourcePropertyDisplayName', 'ga:sourcePropertyTrackingId',
                  'ga:screenResolution', 'ga:hostname', 'ga:pagePath', 'ga:landingPagePath', 'ga:secondPagePath', 'ga:exitPagePath', 'ga:previousPagePath', 'ga:pageDepth',
                  'ga:searchUsed', 'ga:searchKeyword', 'ga:searchKeywordRefinement', 'ga:searchCategory', 'ga:searchStartPage', 'ga:searchDestinationPage',
                  'ga:searchAfterDestinationPage', 'ga:appInstallerId', 'ga:appVersion', 'ga:appName', 'ga:appId', 'ga:screenName', 'ga:screenDepth', 'ga:landingScreenName',
                  'ga:exitScreenName', 'ga:affiliation', 'ga:sessionsToTransaction', 'ga:daysToTransaction', 'ga:productSku', 'ga:productName', 'ga:productCategory',
                  'ga:checkoutOptions', 'ga:internalPromotionCreative', 'ga:internalPromotionId', 'ga:internalPromotionName', 'ga:internalPromotionPosition',
                  'ga:orderCouponCode', 'ga:shoppingStage', 'ga:socialInteractionNetwork', 'ga:socialInteractionAction', 'ga:socialInteractionNetworkAction',
                  'ga:socialInteractionTarget', 'ga:socialEngagementType', 'ga:dimensionXX', 'ga:customVarNameXX', 'ga:customVarValueXX', 'ga:date', 'ga:year',
                  'ga:month', 'ga:week', 'ga:day', 'ga:hour', 'ga:minute', 'ga:nthMonth', 'ga:nthWeek', 'ga:nthDay', 'ga:nthMinute', 'ga:dayOfWeek', 'ga:dayOfWeekName',
                  'ga:dateHour', 'ga:dateHourMinute', 'ga:yearMonth', 'ga:yearWeek', 'ga:isoWeek', 'ga:isoYear', 'ga:isoYearIsoWeek', 'ga:nthHour', 'ga:dcmClickAd',
                  'ga:dcmClickAdId', 'ga:dcmClickAdType', 'ga:dcmClickAdTypeId', 'ga:dcmClickAdvertiser', 'ga:dcmClickAdvertiserId', 'ga:dcmClickCampaign',
                  'ga:dcmClickCampaignId', 'ga:dcmClickCreativeId', 'ga:dcmClickCreative', 'ga:dcmClickRenderingId', 'ga:dcmClickCreativeType', 'ga:dcmClickCreativeTypeId',
                  'ga:dcmClickCreativeVersion', 'ga:dcmClickSite', 'ga:dcmClickSiteId', 'ga:dcmClickSitePlacement', 'ga:dcmClickSitePlacementId', 'ga:dcmClickSpotId',
                  'ga:dcmFloodlightActivity', 'ga:dcmFloodlightActivityAndGroup', 'ga:dcmFloodlightActivityGroup', 'ga:dcmFloodlightActivityGroupId',
                  'ga:dcmFloodlightActivityId', 'ga:dcmFloodlightAdvertiserId', 'ga:dcmFloodlightSpotId', 'ga:userAgeBracket', 'ga:userGender', 'ga:interestOtherCategory',
                  'ga:interestAffinityCategory', 'ga:interestInMarketCategory', 'ga:channelGrouping', 'ga:dbmClickAdvertiser', 'ga:dbmClickAdvertiserId',
                  'ga:dbmClickCreativeId', 'ga:dbmClickExchange', 'ga:dbmClickExchangeId', 'ga:dbmClickInsertionOrder', 'ga:dbmClickInsertionOrderId',
                  'ga:dbmClickLineItem', 'ga:dbmClickLineItemId', 'ga:dbmClickSite', 'ga:dbmClickSiteId', 'ga:dsAdGroup', 'ga:dsAdGroupId', 'ga:dsAdvertiser',
                  'ga:dsAdvertiserId', 'ga:dsAgency', 'ga:dsAgencyId', 'ga:dsCampaign', 'ga:dsCampaignId', 'ga:dsEngineAccount', 'ga:dsEngineAccountId', 'ga:dsKeyword',
                  'ga:dsKeywordId', 'ga:flashVersion', 'ga:javaEnabled', 'ga:language']

    # Setting up progress bar
    bar = progressbar.ProgressBar()

    # Dates in 3 month intervals to pull the data in pieces
    start_dates = ['2014-07-25', '2014-11-01', '2015-02-01', '2015-05-01', '2015-08-01', '2015-11-01', '2016-02-01', '2016-05-01', '2016-08-01',
                   '2016-11-01', '2017-02-01', '2017-05-01', '2017-08-01']
    end_dates = ['2014-10-31', '2015-01-31', '2015-04-31', '2015-07-31', '2015-10-31', '2016-01-31', '2016-04-31', '2016-07-31',
                 '2016-10-31', '2017-01-31', '2017-04-31', '2017-07-31', '2017-08-31']
    dates = zip(start_dates, end_dates)

    # Intializing overall df for all data and df for one date range
    overall_df = pd.DataFrame()
    one_date_range_df = pd.DataFrame()

    for drange in bar(dates):
        if one_date_range_df.empty:
            pass
        else:
            # Adding metrics from one date range to the bottom of the existing overall dataframe
            overall_df = pd.concat([overall_df, one_date_range_df], axis=0)
            one_date_range_df = pd.DataFrame()
        for metric in metrics:
            print(drange)
            print(metric)
            try:
                # Query filters
                results = service.data().ga().get(
                    # ID = my service account ID
                    ids='ga:89051129',
                    start_date=drange[0],
                    end_date=drange[1],
                    filters='ga:transactionId!=0',
                    dimensions='ga:transactionId',
                    metrics=metric).execute()
                # Select actual data from returned dictionary
                data = results['rows']
                # Create temporary dataframe for single metric in this query
                one_metric_df = pd.DataFrame(data)
                # Pull out header text
                headers = [info['name'] for info in results['columnHeaders']]
                one_metric_df.columns = headers
                if one_date_range_df.empty:
                    # Put into dataframe for this date range
                    one_date_range_df = one_metric_df
                else:
                    # Add to existing dataframe for this date range
                    one_date_range_df = pd.merge(
                        one_date_range_df, one_metric_df, how='left', on='ga:transactionId')
            except:
                continue

    # Saving metrics data to csv
    overall_df.to_csv('../data/google-analytics/google_metrics_9-20-17')

    # Resetting before cycling through dimensions
    one_date_range_df = pd.DataFrame()

    # Resetting progress bar
    bar = progressbar.ProgressBar()

    for drange in bar(dates):
        if one_date_range_df.empty:
            pass
        else:
            # Adding metrics from one date range to the bottom of the existing overall dataframe
            # overall_df = pd.concat([overall_df, one_date_range_df], axis=0)
            one_date_range_df.to_csv('google_dimensions_9-20-17_enddate_{}'.format(drange[1]))
            one_date_range_df = pd.DataFrame()
        for dimension in dimensions:
            print(drange)
            print(dimension)
            try:
                # Query filters
                results = service.data().ga().get(
                    # ID = my service account ID
                    ids='ga:89051129',
                    start_date=drange[0],
                    end_date=drange[1],
                    filters='ga:transactionId!=0',
                    dimensions='ga:transactionId' + ',' + dimension,
                    metrics='ga:totalValue').execute()
                # Select actual data from returned dictionary
                data = results['rows']
                # Create temporary dataframe for single metric in this query
                one_dimension_df = pd.DataFrame(data)
                # Pull out header text
                headers = [info['name'] for info in results['columnHeaders']]
                one_dimension_df.columns = headers
                if one_date_range_df.empty:
                    # Put into dataframe for this date range
                    one_date_range_df = one_dimension_df
                else:
                    # Add to existing dataframe for this date range
                    one_date_range_df = pd.merge(
                        one_date_range_df, one_dimension_df, how='left', on=['ga:transactionId', 'ga:totalValue'])
            except:
                continue

        # Saving final date frame to csv
        one_date_range_df.to_csv(
            '../data/google-analytics/google_dimensions_second_run_9-20-17_enddate_{}'.format(drange[1]))
