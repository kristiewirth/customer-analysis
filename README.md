## Not All Customers Are Created Equal

Have you ever wondered who your most valuable customers are? This project, created for a software company, aims to discover what makes their users unique and how to identify those who stand out above the rest.

<img src="images/people.jpg" width="600">

## Updates

#### Current Progress

* Developed mini model to fill null values in dataset and working to implement
* Added information on number of support tickets, number of business days of those tickets, and an adjusted revenue variable for loss due to extra work

#### Next Steps

* Engineer more features in order to reduce errors
* Use SMOTE to create synthetic samples of target

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

* Which group of customers generate the most revenue?
* Which will generate more revenue - many small purchases with little support work or a few large purchases with many hours of support work needed?
* What factors contribute to a customer being more likely to be more profitable?

#### Planned Methods

  * Obtaining data - used a combination of API's and direct downloads to obtain the data from the above sources
  * Predicting type of license bought
    * Regression analysis
    * KNeighbors
    * Decision trees
    * SVC (linear kernel)
    * SGD

#### Technologies Used

* Python
* Pandas
* Numpy
* AWS (EC2 instance)
* Sklearn
* Seaborn
* Matplotlib

#### Deliverables

Slideshow presentation for broad project overview
