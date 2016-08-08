## Zombie Debt Slayer

Zombie Debt Slayer - ZDS - is a classification application that predicts the
likelihood a "Zombie" debt will be forgiven. A "zombie" debt is a debt that
has long since been forgotten to get paid and comes back to "haunt" the debtor.

### Motivation

Imagine opening and email to find that you owe $10,000! The shock of
frustration and rage start to consume you. As you read on you realize that
this debt isn't even yours, but your long since deceased husband. It is from
his medical bills that should have been paid long ago! At the bottom of the
email, there was court date to dispute this debt that has already passed.

According the the Consumer Financial Protection Bureau (CFPB):
http://www.consumerfinance.gov/

There are 3000 complaints filed a week. After my research, I found the
majority of debts and fees to range from $50 to $10,000. That's $150,000 to
$30,000,000 a week!

As a debtor, it takes lots of time to dispute debts. The process also requires
the debtor to write a formal complaint which can be challenge. Debtors wonder
if it is worth their time to dispute these debts, or if they develop a
restructured payment plan to pay off the debt.

Not just debtors are concerned with the outcomes of their debts. Lenders are
always trying to predict which penalties and refunds they will be liable to pay.

With the Zombie Debt Slayer, both debtors and lenders are able to predict the
likelihood a debt will be forgiven. It provides consumers with more insight,
and enables consumers to make informed decisions about filing formal
complaints. ZDS provides lenders information about future penalties and
consumer refunds.

### Overview

Collecting the data was a standard process of downloading a 194 MB .csv file.
The data was then processed with Pandas and Numpy and modeled with Scikit Learn
Python libraries. The goal of this project was to showcase strategies for data
science, and thus, the CFPB API and AWS web servers were a last resort for data
mining and processing. In the future, and due to other challenges that became
apparent as the analysis progressed, I would integrate the CFPB API, store the
data in a Mongo database, export it to JSON, and move it to an AWS instance for
high power processing.

Currently, the front end and integration of machine learning text predictions
of the ZDS application are built. The complete application would require a
backend database integration with search, categorical features, and hosting
on AWS.

### Data

Upon inspection, the data had NaN, null, missing values, empty strings, and
mixed datatypes. It was dirty to say the least.

There were approximately 600,000 rows of released categorical variables, such as:
* Product - type of debt
* Issue - the problem the debtor is reporting
* Company - the lender
* State, Zip code - location in the United States
* Submitted via - the method of submission
* Consumer complaint narrative - description of what happened
* Company response to consumer - the outcome from the complaint
NOTE: there were approximately 80,000 rows with released narratives.

### Challenges

The discovery that seven different classes existed within the complaint
outcomes, made it apparent that simplification was needed. Thus, the classes
were combined down to three:
* 0 - the debt was not forgiven
* 1 - the debt was forgiven or had some sort of relief
* 2 - the debt was possibly forgiven because the outcome was still pending or
unclear if there had been relief
After this simplification, there were class imbalances.

There were a few initial features that had a majority of missing values,
and were dropped from the analysis. They were also related to another feature
that was included, such as "Sub-product."

After the initial dummification of features, the feature matrix had over 4000
features, and had the need for dimensionality reduction. This was due to the
fact that ZIP code and Company had a large number of unique values. This also
produced a very sparse matrix.

Discluding ZIP code and Company revealed no predictive signal with the
remaining features.

### Process & Solutions

Due to sparsity and curse of dimensionality, I removed KNN and SVMs from the
list of models to compare. Logistic Regression, Random Forest Classifier,
Gradient Boosting Classifier, and their combination were chosen because they
are better equipped to handle sparsity and high dimensions. As the analysis
progressed, Random Forest (on average) was about 10% more accurate.

To reduce dimensionality, state political affiliations from the 2012
presidential election were scrapped from Wikipedia:
https://en.wikipedia.org/wiki/United_States_presidential_election,_2012

Because the original hypothesis that the type of debt would be a major
predictor was incorrect, and AWS was not being used to deal with the multitude
of unique companies, a new hypothesis was rectified. Due to domain knowledge
of writing formal complaints, the revised hypothesis was that how a narrative
was written impacted the outcome of the complaint. Thus, all rows which didn't
include a text narrative were dropped. This left about 80,000 rows with text
narratives to analyze.

The reduction of ZIP code digits - to the first three digits - caused drastic
improvement in the models. IMPORTANT: This concludes that the location of where
the complaints are filed has drastic impact on the likelihood of forgiveness.

Class imbalance was solved with Random Oversampling found here:
https://github.com/fmfn/UnbalancedDataset

Unfortunately, this oversampling module does not handle strings within the
feature matrix input, and so the work-around was to merge an oversampled
feature matrix (containing only categorical and boolean features) with a
temporary dataframe that had a mapping to the associated complaint narratives.

Narrative analysis proved to have signal as well. The following features were
engineered to capture the signal from the text narratives:
NOTE: all features were normalized over the length of the narrative.
* Formatted - whether there were paragraphs - hard returns - in the data
* Narrative length - the length of the narrative
* First person - the use of how many times "I", "Me", "My", "Mine" occurred
* Punctuation - the occurrences of punctuation
* Polite - third party vocabulary lists of polite and rude vocabulary words
associated with forgiven and not forgiven respectively
* Formal - third party vocabulary lists of formal and informal vocabulary words
associated with forgiven and not forgiven respectively
* Polarity - sentiment analysis
NOTE: the models run with Polite, Formal, and Polarity features all
together, and with further analysis may reveal some multicollinearity. A third
party should always be used with text analysis because after an individual's
initial inspection of the text, bias is automatically generated.

**Accuracy**, rather the amount of correct predictions to all outcomes was only
one scoring metric used.

**Precision**, rather the proportion of debtors predicted as having their debt
forgiven and actually did was also used.

**Recall**, rather the proportion of debtors that were accurately predicted as
having their debts forgiven to all forgiven debts was also used.

### Conclusion

The final grid searched Random Forest Classifier had 79% accuracy,
20% precision, and 3% recall.

ZIP code improved the Random Forest Classifier from 52% to 73% accuracy. This
reveals that the location of where the debtor was filing the complaint impacts
the outcome of the complaint. How a complaint narrative was written impacts the
outcome as well. Using more polite and formal vocabulary and writing style can
only help the chances of debt forgiveness.

Currently, the model uses a low threshold for recall to explain to debtors it’s
worth their time to fight debts even if they have a small chance of being
forgiven. My assumption is that lenders would care more about precision, so
they can better predict which debts would actually cause penalties and refunds.

## Future

Short Term:
* Explore the best scoring metrics for debtors and lenders
* Gather more data with narratives and information about the lenders, such as:
 * Debt amounts
 * Different types of lenders (e.g. mortgages, credit cards, etc.)
* Engineer more narrative features to inspect the need for documentation
and written proof of resolutions.

Long Term:
* Develop a bot to help debtors write formal narratives and submit complaints
