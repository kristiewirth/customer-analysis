## Customer Analysis

This repository contains code used for an analysis of customer data for a software company.

## Updates

#### Current Progress

* Obtained data from all 7 data sources
* Started data munging

#### Next Steps

* Join all 7 datasets together
* Engineer features

#### Technologies Used

* Python
* Pandas
* Numpy
* AWS (EC2 instance)

## Project Outline

#### Data Sources

Note: These data sources will be stored separately in a private repository.

* Mechanical Turk - Data from past experiment of identifying customers' website categories
* Intercom - Tracks data on customer communication such as Facebook messages and live webpage chat
* Drip - Tracks data on email related customer communication
* EDD - Tracks data on customer purchases
* Google Analytics - Tracks data on customer webpage browsing
* HubSpot - Tracks data on marketing and sales campaigns
* HelpScout - Tracks data on support tickets

File date range: 07/25/14 - 08/31/17

#### Project Goals

* Customer segmentation
  * Which groups generate the most profit?
  * What factors are most predictive of profits for each group?
  * Which packages are used by each group?
  * Use this information to create custom marketing/landing pages/pricing for each group

#### Planned Methods

  * Obtaining data - will use API's and/or web-scraping techniques to obtain the data from the above sources
  * Customer segmentation - KMeans clustering
  * Profit predictions per cluster
    * Regressions (Lasso, Ridge, ElasticNet)
    * KNeighbors
    * Decision tree
    * SVR (linear kernel)
    * SGD

#### Measuring Project Impact

Will use measurements of overall profit, before & after any relevant changes in marketing/landing pages/pricing based on recommendations

#### Deliverables

Slideshow presentation for broad project overview
